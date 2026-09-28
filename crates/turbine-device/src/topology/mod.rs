//! Node-local topology graph (contract §5.3, P5 S-1): NUMA nodes, the PCIe tree, GPUs, NICs
//! and NVMe controllers as typed vertices, joined by edges whose every value is tagged with its
//! source. Built once at startup from a sysfs root plus a [`TopologyVendor`]; discovery never
//! fails — a source that cannot be read leaves its values `unknown`/`null` and logs one WARN.

mod sysfs;
mod vendor;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use turbine_core::types::{DeviceId, Vendor};

use crate::discovery::DiscoveryOptions;
use crate::inventory::{DeviceInfo, DeviceInventory};
use sysfs::{PciDevice, PciFunction, Sysfs};
pub use vendor::{AmdSmiTopology, NoVendorTopology, NvmlTopology};

/// `GET /turbine/v1/topology` document (P5 §Data).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TopologyGraph {
    pub node: TopologyNode,
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TopologyNode {
    pub hostname: String,
    /// RFC 3339 UTC time of discovery.
    pub captured_at: String,
}

/// Ids: `numa<N>`, `gpu<index>`, `pcie:<bdf>`, `nic:<netdev>`, `nvme:<controller>`; stable
/// for the process lifetime.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Vertex {
    pub id: String,
    pub kind: VertexKind,
    #[serde(flatten)]
    pub attrs: VertexAttrs,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum VertexKind {
    Numa,
    PcieRoot,
    PcieSwitch,
    Gpu,
    Nic,
    Nvme,
}

/// Per-kind attributes, flattened into the vertex object. Each variant has one required field
/// that tells them apart when read back.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum VertexAttrs {
    Gpu(GpuAttrs),
    Nic(NicAttrs),
    Nvme(NvmeAttrs),
    Numa(NumaAttrs),
    Pcie(PcieAttrs),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct NumaAttrs {
    pub node: u32,
    pub cpus: Option<String>,
    pub memory_bytes: Option<u64>,
    /// Row of the kernel's NUMA distance matrix (`distance`), indexed by node.
    pub distances: Option<Vec<u32>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GpuAttrs {
    pub device_index: u32,
    pub vendor: Vendor,
    pub arch: Option<String>,
    pub pci_bus_id: Option<String>,
    pub numa: Option<u32>,
    pub numa_source: Option<AttrSource>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct NicAttrs {
    pub netdev: String,
    pub rdma_device: Option<String>,
    /// `ethernet` | `infiniband`.
    pub link_layer: Option<String>,
    pub rate_gbps: Option<f64>,
    pub ipv4: Vec<String>,
    pub mac: Option<String>,
    pub pci_bus_id: Option<String>,
    pub numa: Option<u32>,
    pub numa_source: Option<AttrSource>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct NvmeAttrs {
    pub controller: String,
    pub model: Option<String>,
    pub pci_bus_id: Option<String>,
    pub numa: Option<u32>,
    pub numa_source: Option<AttrSource>,
}

/// A root port (`pcie_root`) or a bridge below it (`pcie_switch`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PcieAttrs {
    pub pci_bus_id: String,
    pub link_gts: Option<f64>,
    pub width: Option<u32>,
    pub numa: Option<u32>,
    pub numa_source: Option<AttrSource>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Edge {
    pub a: String,
    pub b: String,
    pub kind: EdgeKind,
    pub path: Option<PathClass>,
    pub hops: Option<u32>,
    pub link_gts: Option<f64>,
    pub width: Option<u32>,
    pub p2p: P2pStatus,
    pub vendor_interconnect: Option<String>,
    pub rdma: Option<bool>,
    pub source: Option<AttrSource>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EdgeKind {
    Pcie,
    Nvlink,
    Xgmi,
    Coherent,
    Numa,
    /// Between nodes of the P6 cluster graph.
    Network,
}

/// Link class between two devices, NVML topology levels plus the vendor fabrics.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PathClass {
    #[serde(rename = "self")]
    SelfPath,
    Pix,
    Pxb,
    Phb,
    Node,
    Sys,
    Nvlink,
    Xgmi,
}

impl PathClass {
    /// Planner preference, higher is better: self > NVLink = xGMI > PIX > PXB > PHB > NODE > SYS.
    pub fn rank(self) -> u8 {
        match self {
            PathClass::SelfPath => 7,
            PathClass::Nvlink | PathClass::Xgmi => 6,
            PathClass::Pix => 5,
            PathClass::Pxb => 4,
            PathClass::Phb => 3,
            PathClass::Node => 2,
            PathClass::Sys => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PathClass::SelfPath => "self",
            PathClass::Pix => "pix",
            PathClass::Pxb => "pxb",
            PathClass::Phb => "phb",
            PathClass::Node => "node",
            PathClass::Sys => "sys",
            PathClass::Nvlink => "nvlink",
            PathClass::Xgmi => "xgmi",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum P2pStatus {
    Enabled,
    Disabled,
    Unknown,
}

/// Where a value came from: `nominal` (derived from link type/speed/width or a default),
/// `vendor` (amd-smi / NVML) or `sysfs`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AttrSource {
    Nominal,
    Vendor,
    Sysfs,
}

/// A GPU↔GPU link as a vendor library reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VendorLink {
    pub kind: EdgeKind,
    pub path: PathClass,
    pub hops: Option<u32>,
    pub p2p: P2pStatus,
}

/// Vendor topology queries. `Err` means the value is unknown (library missing, call
/// unsupported); discovery then falls back to sysfs or `unknown` and logs one WARN.
pub trait TopologyVendor {
    fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String>;
    /// Peer access between `gpu` and the node's RDMA NICs; `Ok(None)` = the vendor cannot tell.
    fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String>;
    /// The cache-coherent GPU↔host interconnect (`nvlink_c2c`), `Ok(None)` when there is none.
    fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String>;
}

/// The topology sources of every registered discovery kind (`device_discovery` registry),
/// each opened on first use and asked only about its own vendor's devices. A kind whose
/// library cannot load answers `Err` (its values stay `unknown`); a GPU pair of two vendors
/// has no vendor link.
pub struct RegisteredTopology {
    opts: DiscoveryOptions,
    /// Per vendor: the opened source, or why it could not open.
    sources: BTreeMap<Vendor, Result<Box<dyn TopologyVendor>, String>>,
}

impl RegisteredTopology {
    pub fn new(opts: DiscoveryOptions) -> RegisteredTopology {
        RegisteredTopology {
            opts,
            sources: BTreeMap::new(),
        }
    }

    fn source(&mut self, vendor: Vendor) -> Result<&mut dyn TopologyVendor, String> {
        let opts = &self.opts;
        let entry =
            self.sources.entry(vendor).or_insert_with(|| {
                match crate::discovery::registry()
                    .iter()
                    .find(|k| k.vendor() == vendor)
                {
                    Some(kind) => kind.topology(opts),
                    None => Err(format!("no discovery kind is registered for {vendor:?}")),
                }
            });
        match entry {
            Ok(source) => Ok(source.as_mut()),
            Err(e) => Err(e.clone()),
        }
    }
}

impl TopologyVendor for RegisteredTopology {
    fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String> {
        if a.vendor != b.vendor {
            return Err(format!(
                "devices {} and {} have different vendors",
                a.index.0, b.index.0
            ));
        }
        self.source(a.vendor)?.link(a, b)
    }
    fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
        self.source(gpu.vendor)?.gpu_nic_p2p(gpu)
    }
    fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String> {
        self.source(gpu.vendor)?.coherent_host_link(gpu)
    }
}

/// PCI class code (upper 16 bits of `class`) of a PCI-to-PCI bridge.
const PCI_CLASS_BRIDGE_PCI: u32 = 0x0604;

/// Warns once per missing source (`numa`, `pci`, `net`, `infiniband`, `nvme`, `ipv4`,
/// `vendor_link`).
#[derive(Default)]
struct Warnings(BTreeSet<&'static str>);

impl Warnings {
    fn missing(&mut self, source: &'static str, detail: &str) {
        if self.0.insert(source) {
            tracing::warn!(
                event = "topology_source_missing",
                source,
                detail,
                "topology source unavailable; its values are reported as unknown"
            );
        }
    }
}

struct Builder {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    vertex_ids: BTreeSet<String>,
    edge_ids: BTreeSet<(String, String, EdgeKind)>,
}

impl Builder {
    fn vertex(&mut self, id: String, kind: VertexKind, attrs: VertexAttrs) -> bool {
        if !self.vertex_ids.insert(id.clone()) {
            return false;
        }
        self.vertices.push(Vertex { id, kind, attrs });
        true
    }

    fn edge(&mut self, e: Edge) {
        if self.edge_ids.insert((e.a.clone(), e.b.clone(), e.kind)) {
            self.edges.push(e);
        }
    }
}

fn edge(a: &str, b: &str, kind: EdgeKind) -> Edge {
    Edge {
        a: a.to_string(),
        b: b.to_string(),
        kind,
        path: None,
        hops: None,
        link_gts: None,
        width: None,
        p2p: P2pStatus::Unknown,
        vendor_interconnect: None,
        rdma: None,
        source: None,
    }
}

/// NUMA node of a PCI function: `-1` is attached to node 0 as `nominal` with a WARN naming
/// the vertex; an unreadable value stays unknown.
fn numa_of(f: Option<&PciFunction>, vertex: &str) -> (Option<u32>, Option<AttrSource>) {
    match f.and_then(|f| f.numa_node) {
        Some(n) if n >= 0 => (u32::try_from(n).ok(), Some(AttrSource::Sysfs)),
        Some(_) => {
            tracing::warn!(
                event = "topology_numa_unknown",
                vertex,
                "numa_node is -1; attached to numa0 (nominal)"
            );
            (Some(0), Some(AttrSource::Nominal))
        }
        None => (None, None),
    }
}

fn is_bridge(f: &PciFunction) -> bool {
    f.class.is_some_and(|c| c >> 8 == PCI_CLASS_BRIDGE_PCI)
}

/// Adds the root port / switch vertices above `dev` and the PCIe edges down to `endpoint`.
fn attach_chain(b: &mut Builder, dev: &PciDevice, endpoint: &str) {
    let mut parent: Option<String> = None;
    for (depth, f) in dev.chain.iter().enumerate() {
        let id = format!("pcie:{}", f.bdf);
        let kind = if depth == 0 {
            VertexKind::PcieRoot
        } else {
            VertexKind::PcieSwitch
        };
        if kind == VertexKind::PcieRoot || is_bridge(f) {
            if !b.vertex_ids.contains(&id) {
                let (numa, numa_source) = numa_of(Some(f), &id);
                b.vertex(
                    id.clone(),
                    kind,
                    VertexAttrs::Pcie(PcieAttrs {
                        pci_bus_id: f.bdf.clone(),
                        link_gts: f.link_gts,
                        width: f.width,
                        numa,
                        numa_source,
                    }),
                );
            }
            if let Some(p) = &parent {
                b.edge(pcie_edge(p, &id, f));
            }
            parent = Some(id);
        }
    }
    if let Some(p) = &parent {
        b.edge(pcie_edge(p, endpoint, &dev.function));
    }
}

fn pcie_edge(a: &str, b: &str, child: &PciFunction) -> Edge {
    Edge {
        link_gts: child.link_gts,
        width: child.width,
        source: Some(AttrSource::Sysfs),
        ..edge(a, b, EdgeKind::Pcie)
    }
}

/// The NVML-style level between two PCI functions derived from their sysfs chains.
fn nominal_path(a: &PciDevice, b: &PciDevice) -> PathClass {
    if a.bdf == b.bdf {
        return PathClass::SelfPath;
    }
    if a.host_bridge != b.host_bridge {
        let na = a.function.numa_node.filter(|n| *n >= 0);
        let nb = b.function.numa_node.filter(|n| *n >= 0);
        return match (na, nb) {
            (Some(x), Some(y)) if x == y => PathClass::Node,
            _ => PathClass::Sys,
        };
    }
    let common = a
        .chain
        .iter()
        .zip(&b.chain)
        .take_while(|(x, y)| x.bdf == y.bdf)
        .count();
    // Sharing at most the root port means crossing the host bridge.
    if common <= 1 {
        return PathClass::Phb;
    }
    let below = (a.chain.len() - common) + (b.chain.len() - common);
    if below <= 2 {
        PathClass::Pix
    } else {
        PathClass::Pxb
    }
}

/// Discovers the node-local topology. Never fails: every unreadable source leaves its values
/// `unknown`/`null` without a `source` and logs one WARN (`event = "topology_source_missing"`).
pub fn discover_topology(
    sysfs_root: &Path,
    inventory: &DeviceInventory,
    vendor: &mut dyn TopologyVendor,
) -> TopologyGraph {
    let fs = Sysfs::new(sysfs_root);
    let mut warn = Warnings::default();
    let mut b = Builder {
        vertices: Vec::new(),
        edges: Vec::new(),
        vertex_ids: BTreeSet::new(),
        edge_ids: BTreeSet::new(),
    };

    match fs.numa_nodes() {
        Ok(nodes) => {
            for n in &nodes {
                b.vertex(
                    format!("numa{}", n.id),
                    VertexKind::Numa,
                    VertexAttrs::Numa(NumaAttrs {
                        node: n.id,
                        cpus: n.cpus.clone(),
                        memory_bytes: n.memory_bytes,
                        distances: n.distances.clone(),
                    }),
                );
            }
            for (i, x) in nodes.iter().enumerate() {
                for y in &nodes[i + 1..] {
                    b.edge(Edge {
                        source: Some(AttrSource::Sysfs),
                        ..edge(
                            &format!("numa{}", x.id),
                            &format!("numa{}", y.id),
                            EdgeKind::Numa,
                        )
                    });
                }
            }
        }
        Err(e) => warn.missing("numa", &e),
    }

    let pci_ok = match fs.pci_available() {
        Ok(()) => true,
        Err(e) => {
            warn.missing("pci", &e);
            false
        }
    };
    let pci = |bdf: Option<&str>| -> Option<PciDevice> {
        pci_ok.then(|| bdf.and_then(|b| fs.pci_device(b))).flatten()
    };

    // GPUs
    let mut gpu_pci: Vec<Option<PciDevice>> = Vec::with_capacity(inventory.devices.len());
    for d in &inventory.devices {
        let id = gpu_id(d.index);
        let dev = pci(d.pci_bus_id.as_deref());
        let (numa, numa_source) = numa_of(dev.as_ref().map(|p| &p.function), &id);
        b.vertex(
            id.clone(),
            VertexKind::Gpu,
            VertexAttrs::Gpu(GpuAttrs {
                device_index: d.index.0,
                vendor: d.vendor,
                arch: d.arch.clone(),
                pci_bus_id: d.pci_bus_id.clone(),
                numa,
                numa_source,
            }),
        );
        if let Some(p) = &dev {
            attach_chain(&mut b, p, &id);
        }
        gpu_pci.push(dev);
    }

    // GPU↔GPU
    let devices = &inventory.devices;
    for i in 0..devices.len() {
        for j in i + 1..devices.len() {
            let (pa, pb) = (&gpu_pci[i], &gpu_pci[j]);
            let (link_gts, width) = match (pa, pb) {
                (Some(x), Some(y)) => (
                    min_opt(x.function.link_gts, y.function.link_gts),
                    min_opt(x.function.width, y.function.width),
                ),
                _ => (None, None),
            };
            let base = edge(
                &gpu_id(devices[i].index),
                &gpu_id(devices[j].index),
                EdgeKind::Pcie,
            );
            let e = match vendor.link(&devices[i], &devices[j]) {
                Ok(l) => Edge {
                    kind: l.kind,
                    path: Some(l.path),
                    hops: l.hops,
                    p2p: l.p2p,
                    link_gts,
                    width,
                    source: Some(AttrSource::Vendor),
                    ..base
                },
                Err(err) => {
                    warn.missing("vendor_link", &err);
                    match (pa, pb) {
                        (Some(x), Some(y)) => Edge {
                            path: Some(nominal_path(x, y)),
                            link_gts,
                            width,
                            source: Some(AttrSource::Nominal),
                            ..base
                        },
                        // Nothing known: the most conservative class, without a source.
                        _ => Edge {
                            path: Some(PathClass::Sys),
                            ..base
                        },
                    }
                }
            };
            b.edge(e);
        }
    }

    // Coherent GPU↔host links
    for d in devices {
        match vendor.coherent_host_link(d) {
            Ok(Some(interconnect)) => {
                let numa = b
                    .vertices
                    .iter()
                    .find_map(|v| match &v.attrs {
                        VertexAttrs::Gpu(g) if g.device_index == d.index.0 => g.numa,
                        _ => None,
                    })
                    .unwrap_or(0);
                b.edge(Edge {
                    vendor_interconnect: Some(interconnect),
                    source: Some(AttrSource::Vendor),
                    ..edge(&gpu_id(d.index), &format!("numa{numa}"), EdgeKind::Coherent)
                });
            }
            Ok(None) => {}
            Err(err) => warn.missing("vendor_link", &err),
        }
    }

    // NICs
    let nets = fs.net_interfaces().unwrap_or_else(|e| {
        warn.missing("net", &e);
        Vec::new()
    });
    let ib = fs.infiniband().unwrap_or_else(|e| {
        warn.missing("infiniband", &e);
        Vec::new()
    });
    let ipv4: BTreeMap<String, Vec<String>> = if nets.is_empty() {
        BTreeMap::new()
    } else {
        fs.ipv4_by_interface().unwrap_or_else(|e| {
            warn.missing("ipv4", &e);
            BTreeMap::new()
        })
    };
    let mut nic_pci: Vec<(String, Option<PciDevice>, bool)> = Vec::new();
    for n in &nets {
        let id = format!("nic:{}", n.name);
        let dev = pci(Some(&n.bdf));
        let (numa, numa_source) = numa_of(dev.as_ref().map(|p| &p.function), &id);
        let rdma = ib.iter().find(|r| r.bdf == n.bdf);
        let rate_gbps = match rdma {
            Some(r) if r.active != Some(false) && r.rate_gbps.is_some() => r.rate_gbps,
            _ => n.speed_mbps.map(|mbps| mbps as f64 / 1000.0),
        };
        let link_layer = rdma
            .and_then(|r| r.link_layer.clone())
            .or(match n.arp_type {
                Some(1) => Some("ethernet".to_string()),
                Some(32) => Some("infiniband".to_string()),
                _ => None,
            });
        b.vertex(
            id.clone(),
            VertexKind::Nic,
            VertexAttrs::Nic(NicAttrs {
                netdev: n.name.clone(),
                rdma_device: rdma.map(|r| r.name.clone()),
                link_layer,
                rate_gbps,
                ipv4: ipv4.get(&n.name).cloned().unwrap_or_default(),
                mac: n.mac.clone(),
                pci_bus_id: Some(n.bdf.clone()),
                numa,
                numa_source,
            }),
        );
        if let Some(p) = &dev {
            attach_chain(&mut b, p, &id);
        }
        nic_pci.push((id, dev, rdma.is_some()));
    }

    // GPU↔NIC
    if !nic_pci.is_empty() {
        for (d, gp) in devices.iter().zip(&gpu_pci) {
            let p2p = match vendor.gpu_nic_p2p(d) {
                Ok(s) => s.unwrap_or(P2pStatus::Unknown),
                Err(err) => {
                    warn.missing("vendor_link", &err);
                    P2pStatus::Unknown
                }
            };
            for (nic, np, rdma) in &nic_pci {
                let (path, source) = match (gp, np) {
                    (Some(x), Some(y)) => (nominal_path(x, y), Some(AttrSource::Nominal)),
                    _ => (PathClass::Sys, None),
                };
                b.edge(Edge {
                    path: Some(path),
                    p2p,
                    rdma: Some(*rdma),
                    source,
                    ..edge(&gpu_id(d.index), nic, EdgeKind::Pcie)
                });
            }
        }
    }

    // NVMe
    match fs.nvme() {
        Ok(ctrls) => {
            for c in &ctrls {
                let id = format!("nvme:{}", c.name);
                let dev = pci(Some(&c.bdf));
                let (numa, numa_source) = numa_of(dev.as_ref().map(|p| &p.function), &id);
                b.vertex(
                    id.clone(),
                    VertexKind::Nvme,
                    VertexAttrs::Nvme(NvmeAttrs {
                        controller: c.name.clone(),
                        model: c.model.clone(),
                        pci_bus_id: Some(c.bdf.clone()),
                        numa,
                        numa_source,
                    }),
                );
                if let Some(p) = &dev {
                    attach_chain(&mut b, p, &id);
                }
            }
        }
        Err(e) => warn.missing("nvme", &e),
    }

    let graph = TopologyGraph {
        node: TopologyNode {
            hostname: fs.hostname().unwrap_or_else(|| "unknown".to_string()),
            captured_at: rfc3339_utc(SystemTime::now()),
        },
        vertices: b.vertices,
        edges: b.edges,
    };
    tracing::info!(
        event = "topology_discovered",
        vertices = graph.vertices.len(),
        edges = graph.edges.len(),
        missing_sources = ?warn.0,
        "node topology captured"
    );
    graph
}

fn gpu_id(index: DeviceId) -> String {
    format!("gpu{}", index.0)
}

fn min_opt<T: PartialOrd + Copy>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if y < x { y } else { x }),
        _ => None,
    }
}

/// `YYYY-MM-DDTHH:MM:SSZ` (UTC, whole seconds).
fn rfc3339_utc(t: SystemTime) -> String {
    let secs = t
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Writes a capture file (see `tests/fixtures/topology/*/capture.txt`) as a directory tree
/// under `into`: `<path>\t<content>` lines become files (`\n` and `\\` unescaped) and
/// `<path>\t-> <target>` lines become symbolic links. Test support only.
#[doc(hidden)]
pub fn materialize_capture(capture: &Path, into: &Path) -> std::io::Result<()> {
    let text = std::fs::read_to_string(capture)?;
    for (lineno, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (rel, content) = line.split_once('\t').ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}:{}: no tab separator", capture.display(), lineno + 1),
            )
        })?;
        let path = into.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(target) = content.strip_prefix("-> ") {
            symlink(Path::new(target), &path)?;
        } else {
            std::fs::write(&path, unescape(content))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(not(unix))]
fn symlink(_: &Path, link: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("symbolic link {} needs a Unix host", link.display()),
    ))
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use serde::Deserialize;
    use turbine_core::types::{MemoryKind, Vendor};

    use super::*;
    use crate::inventory::{DeviceInfo, DeviceInventory, DeviceMemoryInfo};

    /// A `MakeWriter` target collecting every JSON log line.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Captured {
        fn lines(&self) -> Vec<serde_json::Value> {
            let bytes = self.0.lock().expect("capture lock").clone();
            String::from_utf8(bytes)
                .expect("utf-8 log")
                .lines()
                .map(|l| serde_json::from_str(l).expect("JSON log line"))
                .collect()
        }

        /// WARN lines whose `fields.event` equals `event`.
        fn warnings(&self, event: &str) -> Vec<serde_json::Value> {
            self.lines()
                .into_iter()
                .filter(|l| l["level"] == "WARN" && l["fields"]["event"] == event)
                .collect()
        }
    }

    fn with_logs<R>(f: impl FnOnce() -> R) -> (R, Captured) {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || writer.clone())
            .with_max_level(tracing::Level::INFO)
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        (out, captured)
    }

    #[derive(Deserialize)]
    struct RecordedDevice {
        index: u32,
        vendor: Vendor,
        arch: String,
        name: String,
        pci_bus_id: String,
        memory_bytes: u64,
        unified: bool,
    }

    #[derive(Deserialize)]
    struct RecordedLink {
        a: u32,
        b: u32,
        kind: EdgeKind,
        path: PathClass,
        hops: Option<u32>,
        p2p: P2pStatus,
    }

    /// The vendor answers recorded in a capture's `vendor.json`.
    #[derive(Deserialize)]
    struct Recorded {
        inventory: Vec<RecordedDevice>,
        links: Vec<RecordedLink>,
        gpu_nic_p2p: HashMap<String, Option<P2pStatus>>,
        coherent: HashMap<String, Option<String>>,
    }

    /// Replays `vendor.json`; a pair or device it does not mention is an error, like a
    /// vendor call that is unsupported.
    struct FixtureVendor(Recorded);

    impl TopologyVendor for FixtureVendor {
        fn link(&mut self, a: &DeviceInfo, b: &DeviceInfo) -> Result<VendorLink, String> {
            self.0
                .links
                .iter()
                .find(|l| {
                    (l.a, l.b) == (a.index.0, b.index.0) || (l.b, l.a) == (a.index.0, b.index.0)
                })
                .map(|l| VendorLink {
                    kind: l.kind,
                    path: l.path,
                    hops: l.hops,
                    p2p: l.p2p,
                })
                .ok_or_else(|| "no recorded link".to_string())
        }
        fn gpu_nic_p2p(&mut self, gpu: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
            self.0
                .gpu_nic_p2p
                .get(&gpu.index.0.to_string())
                .copied()
                .ok_or_else(|| "no recorded GPU-NIC P2P".to_string())
        }
        fn coherent_host_link(&mut self, gpu: &DeviceInfo) -> Result<Option<String>, String> {
            self.0
                .coherent
                .get(&gpu.index.0.to_string())
                .cloned()
                .ok_or_else(|| "no recorded coherent link".to_string())
        }
    }

    /// Every call fails, as when no vendor library is loaded.
    struct FailingVendor;

    impl TopologyVendor for FailingVendor {
        fn link(&mut self, _: &DeviceInfo, _: &DeviceInfo) -> Result<VendorLink, String> {
            Err("vendor topology unavailable".into())
        }
        fn gpu_nic_p2p(&mut self, _: &DeviceInfo) -> Result<Option<P2pStatus>, String> {
            Err("vendor topology unavailable".into())
        }
        fn coherent_host_link(&mut self, _: &DeviceInfo) -> Result<Option<String>, String> {
            Err("vendor topology unavailable".into())
        }
    }

    fn fixture(host: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/topology")
            .join(host)
            .join("capture.txt")
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "turbine-topology-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn inventory(recorded: &Recorded) -> DeviceInventory {
        DeviceInventory {
            devices: recorded
                .inventory
                .iter()
                .map(|d| DeviceInfo {
                    index: DeviceId(d.index),
                    vendor: d.vendor,
                    vendor_index: d.index,
                    name: d.name.clone(),
                    uuid: None,
                    pci_bus_id: Some(d.pci_bus_id.clone()),
                    arch: Some(d.arch.clone()),
                    driver_version: None,
                    memory: DeviceMemoryInfo {
                        kind: if d.unified {
                            MemoryKind::Unified
                        } else {
                            MemoryKind::Dedicated
                        },
                        total_bytes: d.memory_bytes,
                        shared_with_host: d.unified,
                    },
                })
                .collect(),
            backends: Vec::new(),
        }
    }

    /// The registry-backed source asks each vendor's discovery kind and never panics: missing
    /// libraries and vendor-mixed pairs are `Err` (their values stay unknown).
    #[test]
    fn registered_topology_dispatches_by_vendor() {
        let gpu = |i: u32, vendor: Vendor, bdf: &str| DeviceInfo {
            index: DeviceId(i),
            vendor,
            vendor_index: i,
            name: "gpu".into(),
            uuid: None,
            pci_bus_id: Some(bdf.into()),
            arch: None,
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 1 << 30,
                shared_with_host: false,
            },
        };
        let (a, b) = (
            gpu(0, Vendor::Amd, "0000:03:00.0"),
            gpu(1, Vendor::Amd, "0000:83:00.0"),
        );
        let n = gpu(2, Vendor::Nvidia, "0000:01:00.0");
        let opts = DiscoveryOptions {
            nvml_library: Some("/nonexistent/libnvidia-ml.so.1".into()),
            amd_smi_library: Some("/nonexistent/libamd_smi.so".into()),
            ..DiscoveryOptions::default()
        };
        let mut t = RegisteredTopology::new(opts);
        let err = t.link(&a, &b).expect_err("no amd-smi");
        assert!(err.contains("/nonexistent/libamd_smi.so"), "{err}");
        let err = t.coherent_host_link(&n).expect_err("no NVML");
        assert!(err.contains("/nonexistent/libnvidia-ml.so.1"), "{err}");
        let err = t.link(&a, &n).expect_err("vendor-mixed pair");
        assert!(err.contains("different vendors"), "{err}");
        // A second query reuses the recorded failure instead of reloading.
        assert!(t.gpu_nic_p2p(&a).is_err());
    }

    /// Materialises `host`'s capture and runs discovery on it with the recorded vendor answers.
    fn discover_fixture(host: &str) -> (TopologyGraph, Captured) {
        let root = scratch(host);
        materialize_capture(&fixture(host), &root).expect("materialise capture");
        let recorded: Recorded = serde_json::from_str(
            &std::fs::read_to_string(root.join("vendor.json")).expect("vendor.json"),
        )
        .expect("vendor.json parses");
        let inv = inventory(&recorded);
        let mut vendor = FixtureVendor(recorded);
        let out = with_logs(|| discover_topology(&root.join("sys"), &inv, &mut vendor));
        let _ = std::fs::remove_dir_all(&root);
        out
    }

    fn vertex<'a>(g: &'a TopologyGraph, id: &str) -> &'a Vertex {
        g.vertices
            .iter()
            .find(|v| v.id == id)
            .unwrap_or_else(|| panic!("no vertex {id} in {:?}", ids(g)))
    }

    fn ids(g: &TopologyGraph) -> Vec<&str> {
        g.vertices.iter().map(|v| v.id.as_str()).collect()
    }

    fn edges_between<'a>(g: &'a TopologyGraph, a: &str, b: &str) -> Vec<&'a Edge> {
        g.edges
            .iter()
            .filter(|e| (e.a == a && e.b == b) || (e.a == b && e.b == a))
            .collect()
    }

    fn gpu_attrs(v: &Vertex) -> &GpuAttrs {
        match &v.attrs {
            VertexAttrs::Gpu(a) => a,
            other => panic!("{} is not a GPU: {other:?}", v.id),
        }
    }

    #[test]
    fn novanas_fixture() {
        let (g, logs) = discover_fixture("novanas");
        assert_eq!(g.node.hostname, "novanas");

        let gpu_edges: Vec<&Edge> = g
            .edges
            .iter()
            .filter(|e| e.a.starts_with("gpu") && e.b.starts_with("gpu"))
            .collect();
        assert_eq!(gpu_edges.len(), 1, "{gpu_edges:?}");
        let e = gpu_edges[0];
        assert_eq!((e.a.as_str(), e.b.as_str()), ("gpu0", "gpu1"));
        assert_eq!(e.kind, EdgeKind::Pcie);
        assert_eq!(e.path, Some(PathClass::Sys));
        assert_eq!(e.p2p, P2pStatus::Disabled);
        assert_eq!(e.hops, Some(2));
        assert_eq!(e.source, Some(AttrSource::Vendor));
        assert_eq!((e.link_gts, e.width), (Some(32.0), Some(16)));

        for id in ["gpu0", "gpu1"] {
            let v = vertex(&g, id);
            assert_eq!(v.kind, VertexKind::Gpu);
            let a = gpu_attrs(v);
            assert_eq!(a.vendor, Vendor::Amd);
            assert_eq!(a.arch.as_deref(), Some("gfx1201"));
            assert_eq!(a.numa, Some(0), "{id}");
            assert_eq!(a.numa_source, Some(AttrSource::Nominal), "{id}");
        }
        let numa_warnings = logs.warnings("topology_numa_unknown");
        let warned: Vec<&str> = numa_warnings
            .iter()
            .map(|w| w["fields"]["vertex"].as_str().expect("vertex field"))
            .collect();
        // Every device with numa_node = -1 warns exactly once: both GPUs, the NIC, the NVMe
        // controller and the eight PCIe root ports and switch ports above them.
        for id in ["gpu0", "gpu1", "nic:enp10s0", "nvme:nvme0"] {
            assert_eq!(warned.iter().filter(|w| **w == id).count(), 1, "{warned:?}");
        }
        assert_eq!(warned.len(), 12, "{warned:?}");

        // JSON shape of the P5 §Data example.
        let json = serde_json::to_value(&g).expect("graph serialises");
        let gpu0 = json["vertices"]
            .as_array()
            .expect("vertices")
            .iter()
            .find(|v| v["id"] == "gpu0")
            .expect("gpu0 in JSON");
        assert_eq!(gpu0["kind"], "gpu");
        assert_eq!(gpu0["device_index"], 0);
        assert_eq!(gpu0["numa_source"], "nominal");
        let back: TopologyGraph = serde_json::from_value(json.clone()).expect("round trip");
        assert_eq!(serde_json::to_value(&back).expect("serialises"), json);
    }

    #[test]
    fn novanas_without_vendor_uses_the_pcie_tree() {
        let root = scratch("novanas-nominal");
        materialize_capture(&fixture("novanas"), &root).expect("materialise capture");
        let recorded: Recorded = serde_json::from_str(
            &std::fs::read_to_string(root.join("vendor.json")).expect("vendor.json"),
        )
        .expect("vendor.json parses");
        let (g, _) = with_logs(|| {
            discover_topology(&root.join("sys"), &inventory(&recorded), &mut FailingVendor)
        });
        let _ = std::fs::remove_dir_all(&root);
        let e = edges_between(&g, "gpu0", "gpu1");
        assert_eq!(e.len(), 1);
        // Different root ports under one host bridge: PHB, derived, P2P not known.
        assert_eq!(e[0].path, Some(PathClass::Phb));
        assert_eq!(e[0].source, Some(AttrSource::Nominal));
        assert_eq!(e[0].p2p, P2pStatus::Unknown);
        assert_eq!(vertex(&g, "pcie:0000:00:01.1").kind, VertexKind::PcieRoot);
        assert_eq!(vertex(&g, "pcie:0000:02:00.0").kind, VertexKind::PcieSwitch);
        let nic = vertex(&g, "nic:enp10s0");
        match &nic.attrs {
            VertexAttrs::Nic(a) => {
                assert_eq!(a.ipv4, ["192.168.10.203"]);
                assert_eq!(a.rate_gbps, Some(10.0));
                assert_eq!(a.rdma_device, None);
                assert_eq!(a.link_layer.as_deref(), Some("ethernet"));
            }
            other => panic!("not a NIC: {other:?}"),
        }
    }

    #[test]
    fn spark_fixture() {
        let (g, _) = discover_fixture("dgx-spark");
        let gpus: Vec<&Vertex> = g
            .vertices
            .iter()
            .filter(|v| v.kind == VertexKind::Gpu)
            .collect();
        assert_eq!(gpus.len(), 1);

        let nics: Vec<&NicAttrs> = g
            .vertices
            .iter()
            .filter_map(|v| match &v.attrs {
                VertexAttrs::Nic(a) => Some(a),
                _ => None,
            })
            .collect();
        assert_eq!(nics.len(), 4);
        let mut rdma: Vec<&str> = nics
            .iter()
            .map(|n| n.rdma_device.as_deref().expect("RDMA device"))
            .collect();
        rdma.sort_unstable();
        assert_eq!(
            rdma,
            ["roceP2p1s0f0", "roceP2p1s0f1", "rocep1s0f0", "rocep1s0f1"]
        );
        assert_eq!(
            nics.iter().filter(|n| n.rate_gbps == Some(200.0)).count(),
            2
        );
        assert!(
            nics.iter().all(|n| n.rate_gbps != Some(10.0)),
            "a down port's rate is unknown, never the idle rate"
        );

        let coherent = edges_between(&g, "gpu0", "numa0");
        assert_eq!(coherent.len(), 1, "{coherent:?}");
        assert_eq!(coherent[0].kind, EdgeKind::Coherent);
        assert_eq!(
            coherent[0].vendor_interconnect.as_deref(),
            Some("nvlink_c2c")
        );

        let nic_edges: Vec<&Edge> = g
            .edges
            .iter()
            .filter(|e| e.a == "gpu0" && e.b.starts_with("nic:"))
            .collect();
        assert_eq!(nic_edges.len(), 4);
        for e in nic_edges {
            assert_eq!(e.p2p, P2pStatus::Unknown, "{e:?}");
            assert_eq!(e.rdma, Some(true));
        }
    }

    #[test]
    fn missing_sources_degrade() {
        let root = scratch("empty");
        let inv = DeviceInventory {
            devices: (0..2)
                .map(|i| DeviceInfo {
                    index: DeviceId(i),
                    vendor: Vendor::Amd,
                    vendor_index: i,
                    name: "gpu".into(),
                    uuid: None,
                    pci_bus_id: Some(format!("0000:0{}:00.0", i + 3)),
                    arch: Some("gfx1201".into()),
                    driver_version: None,
                    memory: DeviceMemoryInfo {
                        kind: MemoryKind::Dedicated,
                        total_bytes: 1 << 30,
                        shared_with_host: false,
                    },
                })
                .collect(),
            backends: Vec::new(),
        };
        let (g, logs) =
            with_logs(|| discover_topology(&root.join("sys"), &inv, &mut FailingVendor));
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(ids(&g), ["gpu0", "gpu1"]);
        for v in &g.vertices {
            let a = gpu_attrs(v);
            assert_eq!((a.numa, a.numa_source), (None, None), "{v:?}");
        }
        let e = edges_between(&g, "gpu0", "gpu1");
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].path, Some(PathClass::Sys));
        assert_eq!(e[0].p2p, P2pStatus::Unknown);
        assert_eq!(e[0].source, None);
        assert_eq!((e[0].hops, e[0].link_gts, e[0].width), (None, None, None));

        let warned: Vec<String> = logs
            .warnings("topology_source_missing")
            .iter()
            .map(|w| w["fields"]["source"].as_str().expect("source").to_string())
            .collect();
        let mut sorted = warned.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            ["infiniband", "net", "numa", "nvme", "pci", "vendor_link"],
            "{warned:?}"
        );
    }

    #[test]
    fn path_class_ranks_and_names() {
        let order = [
            PathClass::Sys,
            PathClass::Node,
            PathClass::Phb,
            PathClass::Pxb,
            PathClass::Pix,
            PathClass::Nvlink,
        ];
        for pair in order.windows(2) {
            assert!(pair[0].rank() < pair[1].rank(), "{pair:?}");
        }
        assert_eq!(PathClass::Xgmi.rank(), PathClass::Nvlink.rank());
        assert_eq!(
            serde_json::to_value(PathClass::SelfPath).expect("serialises"),
            "self"
        );
        assert_eq!(
            serde_json::to_value(VertexKind::PcieRoot).expect("serialises"),
            "pcie_root"
        );
    }
}
