//! Measured host-link bandwidth on the topology graph (P5 S-13).
//!
//! On a board without GPU peer access every GPU↔GPU byte crosses host memory, so what a
//! collective or a pipeline hand-off gets is set by each GPU's host link — and those differ when
//! the slots do (novanas: GPU0 PCIe Gen5 x8, GPU1 Gen4 x8). At startup a [`LinkProbe`] (the
//! server's, over the kernel library's pinned copy streams) times a bounded host↔device copy per
//! GPU; [`apply_link_probe`] writes the result onto the GPU's upstream PCIe edge and, for a
//! GPU↔GPU edge without peer access, the slower of the two links as `cost_gbps`. A failed probe
//! leaves the edge nominal and logs a WARN.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use turbine_core::types::DeviceId;

use super::{EdgeKind, P2pStatus, TopologyGraph, gpu_id};

/// Bytes copied per direction per GPU by the startup probe.
pub const PROBE_BYTES: u64 = 64 << 20;
/// The whole probe's time bound; GPUs not reached by then stay nominal.
pub const PROBE_BUDGET: Duration = Duration::from_secs(2);

/// One GPU's measured host-link bandwidth, GB/s (10^9 bytes per second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkBandwidth {
    pub h2d_gbps: f64,
    pub d2h_gbps: f64,
}

impl LinkBandwidth {
    /// The slower direction: what a round trip through host memory is limited by.
    pub fn slower(&self) -> f64 {
        self.h2d_gbps.min(self.d2h_gbps)
    }
}

/// Times host↔device copies of one device (implemented where the kernel library is loaded).
pub trait LinkProbe {
    /// Copies `bytes` each way between pinned host memory and `device` and returns the
    /// bandwidth; `Err` names what failed.
    fn host_link(&mut self, device: DeviceId, bytes: u64) -> Result<LinkBandwidth, String>;
}

/// Runs `probe` on every GPU vertex of `graph` in device order within [`PROBE_BUDGET`] and
/// applies the results; returns them (GPUs past the budget are reported as skipped).
pub fn probe_links(
    graph: &mut TopologyGraph,
    probe: &mut dyn LinkProbe,
) -> BTreeMap<DeviceId, Result<LinkBandwidth, String>> {
    let start = Instant::now();
    let mut results = BTreeMap::new();
    for d in graph.gpu_devices() {
        let r = if start.elapsed() > PROBE_BUDGET {
            Err(format!("skipped: the probe exceeded {PROBE_BUDGET:?}"))
        } else {
            probe.host_link(d, PROBE_BYTES)
        };
        results.insert(d, r);
    }
    apply_link_probe(graph, &results);
    results
}

/// Writes measured bandwidth onto each GPU's upstream PCIe edge, and `cost_gbps` (the slower
/// measured link of the two ends) onto GPU↔GPU edges without peer access. Failures are logged
/// (`event = "topology_link_probe_failed"`) and leave the edges nominal.
pub fn apply_link_probe(
    graph: &mut TopologyGraph,
    results: &BTreeMap<DeviceId, Result<LinkBandwidth, String>>,
) {
    let mut measured: BTreeMap<String, LinkBandwidth> = BTreeMap::new();
    for (&d, r) in results {
        let id = gpu_id(d);
        match r {
            Ok(bw) => {
                let upstream = graph
                    .edges
                    .iter_mut()
                    .find(|e| e.kind == EdgeKind::Pcie && e.b == id && e.a.starts_with("pcie:"));
                match upstream {
                    Some(e) => {
                        e.measured_h2d_gbps = Some(bw.h2d_gbps);
                        e.measured_d2h_gbps = Some(bw.d2h_gbps);
                        measured.insert(id.clone(), *bw);
                        tracing::info!(
                            event = "topology_link_measured",
                            device = d.0,
                            h2d_gbps = bw.h2d_gbps,
                            d2h_gbps = bw.d2h_gbps,
                            "host link measured"
                        );
                    }
                    None => tracing::warn!(
                        event = "topology_link_probe_failed",
                        device = d.0,
                        error = "no upstream PCIe edge in the graph",
                        "host link measured but not placed; the edge stays nominal"
                    ),
                }
            }
            Err(error) => tracing::warn!(
                event = "topology_link_probe_failed",
                device = d.0,
                error = error.as_str(),
                "host link probe failed; the edge stays nominal"
            ),
        }
    }
    for e in &mut graph.edges {
        if e.p2p == P2pStatus::Enabled {
            continue;
        }
        if let (Some(a), Some(b)) = (measured.get(&e.a), measured.get(&e.b)) {
            e.cost_gbps = Some(a.slower().min(b.slower()));
        }
    }
}

impl TopologyGraph {
    /// The device indices of the graph's GPU vertices, ascending.
    pub fn gpu_devices(&self) -> Vec<DeviceId> {
        let mut v: Vec<DeviceId> = self
            .vertices
            .iter()
            .filter_map(|v| match &v.attrs {
                super::VertexAttrs::Gpu(g) => Some(DeviceId(g.device_index)),
                _ => None,
            })
            .collect();
        v.sort();
        v
    }

    /// `device`'s measured host-link bandwidth (the slower direction), `None` when unmeasured.
    pub fn host_link_gbps(&self, device: DeviceId) -> Option<f64> {
        let id = gpu_id(device);
        self.edges.iter().find_map(|e| {
            (e.kind == EdgeKind::Pcie && e.b == id && e.a.starts_with("pcie:"))
                .then(|| e.measured_h2d_gbps.zip(e.measured_d2h_gbps))
                .flatten()
                .map(|(h, d)| h.min(d))
        })
    }
}
