//! Parallel planner (P5 S-4): configuration + device inventory + topology graph + model shape →
//! a single-vendor [`ParallelPlan`] of TP groups and DP replicas, computed before the server
//! binds its port (a [`PlanError`] is exit 2).
//!
//! Rules: one vendor per plan (`devices: auto` picks the vendor with the most devices, ties to
//! the vendor of the lowest index, and lists the others as excluded); one architecture per TP
//! group; `tp` divides the attention heads and divides or is a multiple of the KV heads;
//! `tp: auto` is the smallest power of two whose per-rank weight shard plus one
//! maximum-length sequence of BF16 KV fits the smallest device budget; `dp: auto` uses every
//! remaining device. Automatic TP groups are built greedily from the best GPU↔GPU link class
//! between unassigned devices (never by index order); explicit device lists keep their order.
//! Every decision is logged with its reason code and returned in [`ParallelPlan::reasons`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use turbine_core::config::{DeviceSelection, ParallelConfig, RankMode, SizeOrAuto};
use turbine_core::registry::Module;
use turbine_core::types::{DeviceId, ModelShape, ReplicaId, Vendor};
use turbine_device::DeviceInventory;
use turbine_device::topology::{PathClass, TopologyGraph};

/// The process-lifetime multi-GPU plan (P5 §Data).
#[derive(Clone, Debug, PartialEq)]
pub struct ParallelPlan {
    pub tp: u32,
    pub dp: u32,
    /// The registered collective backend (`collective_backend` registry).
    pub backend: &'static str,
    pub mode: RankMode,
    /// The plan's one vendor; `None` for the `cpu` reference backend, which has no GPU.
    pub vendor: Option<Vendor>,
    pub excluded_devices: Vec<DeviceId>,
    pub groups: Vec<ReplicaGroup>,
    pub reasons: Vec<PlanReason>,
}

/// One DP replica: one TP group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaGroup {
    pub replica: ReplicaId,
    pub ranks: Vec<RankSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSlot {
    pub rank: u32,
    pub device: DeviceId,
    pub host: String,
}

/// Why the plan looks the way it does; `Display` is the reason code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanReason {
    FitsSingleDevice,
    TpRequiredForCapacity,
    GroupedByLink(PathClass),
    VendorHomogeneous,
    VendorExcluded(Vendor),
    ExplicitDevices,
    DeviceSharingEnabled,
    /// One engine per replica on `execution.device` (the single-GPU default, or the `cpu`
    /// reference backend): the inventory is not consulted.
    ExecutionDevice,
}

impl fmt::Display for PlanReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanReason::FitsSingleDevice => f.write_str("fits_single_device"),
            PlanReason::TpRequiredForCapacity => f.write_str("tp_required_for_capacity"),
            PlanReason::GroupedByLink(p) => write!(f, "grouped_by_link:{}", p.as_str()),
            PlanReason::VendorHomogeneous => f.write_str("vendor_homogeneous"),
            PlanReason::VendorExcluded(v) => write!(f, "vendor_excluded:{}", v.as_str()),
            PlanReason::ExplicitDevices => f.write_str("explicit_devices"),
            PlanReason::DeviceSharingEnabled => f.write_str("device_sharing_enabled"),
            PlanReason::ExecutionDevice => f.write_str("execution_device"),
        }
    }
}

/// An impossible plan; exit 2 before bind. `key` is the configuration key to change.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{key}: {reason}")]
pub struct PlanError {
    pub key: String,
    pub reason: String,
}

fn err(key: &str, reason: impl Into<String>) -> PlanError {
    PlanError {
        key: key.to_string(),
        reason: reason.into(),
    }
}

const TP_KEY: &str = "parallel.tensor_parallel_size";
const DP_KEY: &str = "parallel.data_parallel_size";
const DEVICES_KEY: &str = "parallel.devices";
const BACKEND_KEY: &str = "parallel.collective_backend";

/// A candidate device: index and architecture.
#[derive(Clone, Debug)]
struct Candidate {
    id: DeviceId,
    arch: Option<String>,
}

/// Best GPU↔GPU link class per unordered pair of device indices.
struct Links(BTreeMap<(u32, u32), PathClass>);

impl Links {
    fn from_graph(topo: &TopologyGraph) -> Links {
        let gpu = |id: &str| id.strip_prefix("gpu").and_then(|n| n.parse::<u32>().ok());
        let mut best: BTreeMap<(u32, u32), PathClass> = BTreeMap::new();
        for e in &topo.edges {
            let (Some(a), Some(b), Some(path)) = (gpu(&e.a), gpu(&e.b), e.path) else {
                continue;
            };
            let key = (a.min(b), a.max(b));
            let slot = best.entry(key).or_insert(path);
            if path.rank() > slot.rank() {
                *slot = path;
            }
        }
        Links(best)
    }

    /// Unknown links count as `sys`, the worst class (never assume a faster link).
    fn get(&self, a: DeviceId, b: DeviceId) -> PathClass {
        let key = (a.0.min(b.0), a.0.max(b.0));
        self.0.get(&key).copied().unwrap_or(PathClass::Sys)
    }

    /// The worst link between any two members.
    fn worst(&self, members: &[DeviceId]) -> Option<PathClass> {
        let mut worst: Option<PathClass> = None;
        for (i, a) in members.iter().enumerate() {
            for b in &members[i + 1..] {
                let p = self.get(*a, *b);
                if worst.is_none_or(|w| p.rank() < w.rank()) {
                    worst = Some(p);
                }
            }
        }
        worst
    }
}

/// `tp` divides the attention heads, and divides or is a multiple of the KV heads.
fn heads_ok(tp: u32, model: &ModelShape) -> Result<(), String> {
    let (heads, kv) = (model.num_attention_heads, model.num_kv_heads);
    if tp > heads || !heads.is_multiple_of(tp) {
        return Err(format!(
            "{tp} does not divide the model's {heads} attention heads"
        ));
    }
    if kv == 0 || !(kv.is_multiple_of(tp) || tp.is_multiple_of(kv)) {
        return Err(format!(
            "{tp} neither divides nor is a multiple of the model's {kv} KV heads"
        ));
    }
    Ok(())
}

/// Per-rank bytes of the weight shard plus one maximum-length BF16 KV sequence.
fn per_rank_bytes(tp: u32, model: &ModelShape) -> u64 {
    let kv_heads = if model.num_kv_heads.is_multiple_of(tp) {
        model.num_kv_heads / tp
    } else {
        1
    };
    let kv = u64::from(model.num_layers)
        * 2
        * u64::from(kv_heads)
        * u64::from(model.head_dim)
        * 2
        * u64::from(model.max_position_embeddings);
    model.weight_bytes.div_ceil(u64::from(tp)) + kv
}

/// Builds `count` TP groups of `tp` from `pool`, best link class first.
fn link_groups(
    pool: &[Candidate],
    tp: usize,
    count: usize,
    links: &Links,
) -> Result<Vec<Vec<DeviceId>>, PlanError> {
    let mut free: Vec<Candidate> = pool.to_vec();
    let mut groups = Vec::with_capacity(count);
    for _ in 0..count {
        if tp == 1 {
            groups.push(vec![free.remove(0).id]);
            continue;
        }
        // Best seed pair of one architecture: highest link rank, then lowest indices.
        let mut seed: Option<(u8, usize, usize)> = None;
        for i in 0..free.len() {
            for j in i + 1..free.len() {
                if free[i].arch != free[j].arch {
                    continue;
                }
                let r = links.get(free[i].id, free[j].id).rank();
                if seed.is_none_or(|(best, _, _)| r > best) {
                    seed = Some((r, i, j));
                }
            }
        }
        let Some((_, i, j)) = seed else {
            return Err(err(
                TP_KEY,
                format!("no {tp} devices of one architecture are left for a TP group"),
            ));
        };
        let arch = free[i].arch.clone();
        let mut members = vec![free[i].id, free[j].id];
        free.remove(j);
        free.remove(i);
        while members.len() < tp {
            // The device whose worst link to the group is best; ties to the lowest index.
            let next = free
                .iter()
                .enumerate()
                .filter(|(_, c)| c.arch == arch)
                .map(|(k, c)| {
                    let worst = members
                        .iter()
                        .map(|m| links.get(*m, c.id).rank())
                        .min()
                        .unwrap_or(0);
                    (worst, k)
                })
                .fold(None, |acc: Option<(u8, usize)>, (w, k)| match acc {
                    Some((bw, _)) if bw >= w => acc,
                    _ => Some((w, k)),
                });
            let Some((_, k)) = next else {
                return Err(err(
                    TP_KEY,
                    format!("no {tp} devices of one architecture are left for a TP group"),
                ));
            };
            members.push(free.remove(k).id);
        }
        members.sort();
        groups.push(members);
    }
    Ok(groups)
}

/// The collective backend of a plan on `vendor` with `tp` ranks per group in rank `mode`, from
/// the `collective_backend` registry: a configured name must serve the plan (a host-memory
/// backend only without tensor parallelism, a device backend only its vendors, a one-process
/// backend only in `local` mode); `auto` is `host` for tp = 1, else the first registered backend
/// serving `vendor` (in `static` mode: that also crosses processes).
fn choose_backend(
    name: &str,
    vendor: Vendor,
    tp: u32,
    mode: RankMode,
) -> Result<&'static str, PlanError> {
    let registry = crate::collective::registry();
    let crosses = |b: &dyn crate::collective::CollectiveBackend| {
        mode != RankMode::Static || !b.one_process_only()
    };
    if name == "auto" {
        if tp == 1 {
            return Ok(crate::collective::HostBackend.name());
        }
        return registry
            .iter()
            .find(|b| b.vendors().contains(&vendor) && crosses(*b))
            .map(|b| b.name())
            .ok_or_else(|| {
                err(
                    BACKEND_KEY,
                    format!(
                        "no registered collective backend serves {} devices (registered: {})",
                        vendor.as_str(),
                        registry.names().join(", ")
                    ),
                )
            });
    }
    let backend = registry
        .get(name)
        .ok_or_else(|| err(BACKEND_KEY, registry.unknown(name).to_string()))?;
    let vendors = backend.vendors();
    if vendors.is_empty() && tp > 1 {
        return Err(err(
            BACKEND_KEY,
            format!(
                "`{name}` is a host-memory backend for tensor_parallel_size 1 only; GPUs with \
                 tensor parallelism need a device backend"
            ),
        ));
    }
    if !vendors.is_empty() && !vendors.contains(&vendor) {
        return Err(err(
            BACKEND_KEY,
            format!("`{name}` cannot drive {} devices", vendor.as_str()),
        ));
    }
    if !crosses(backend) {
        return Err(err(
            BACKEND_KEY,
            format!(
                "`{name}` exchanges data through memory of one process, so its ranks must be \
                 threads of one process (parallel.ranks.mode local); static mode needs a \
                 backend that crosses processes"
            ),
        ));
    }
    Ok(backend.name())
}

/// Plans TP groups and DP replicas. `device_budget` is the per-device memory budget (phase-3
/// budget) used for `tp: auto`.
pub fn plan(
    inv: &DeviceInventory,
    topo: &TopologyGraph,
    cfg: &ParallelConfig,
    model: &ModelShape,
    device_budget: &dyn Fn(DeviceId) -> u64,
) -> Result<ParallelPlan, PlanError> {
    let mut reasons = Vec::new();
    let links = Links::from_graph(topo);

    // 1. Vendor and candidate devices.
    let (vendor, pool, excluded, explicit) = match &cfg.devices {
        DeviceSelection::List(list) => {
            let mut pool = Vec::with_capacity(list.len());
            for id in list {
                let d = inv.devices.iter().find(|d| d.index == *id).ok_or_else(|| {
                    err(
                        DEVICES_KEY,
                        format!("device {} is not in the inventory", id.0),
                    )
                })?;
                pool.push((
                    d.vendor,
                    Candidate {
                        id: d.index,
                        arch: d.arch.clone(),
                    },
                ));
            }
            let Some(&(vendor, _)) = pool.first() else {
                return Err(err(DEVICES_KEY, "must not be an empty list"));
            };
            if pool.iter().any(|(v, _)| *v != vendor) {
                return Err(err(
                    DEVICES_KEY,
                    "vendor-mixed plan: the listed devices mix vendors (supported from phase 7)",
                ));
            }
            reasons.push(PlanReason::ExplicitDevices);
            let pool: Vec<Candidate> = pool.into_iter().map(|(_, c)| c).collect();
            (vendor, pool, Vec::new(), true)
        }
        DeviceSelection::Auto => {
            let mut by_vendor: BTreeMap<Vendor, (usize, u32)> = BTreeMap::new();
            for d in &inv.devices {
                let e = by_vendor.entry(d.vendor).or_insert((0, d.index.0));
                e.0 += 1;
                e.1 = e.1.min(d.index.0);
            }
            // Most devices; ties to the vendor of the lowest device index.
            let Some((&vendor, _)) = by_vendor
                .iter()
                .max_by(|a, b| a.1.0.cmp(&b.1.0).then(b.1.1.cmp(&a.1.1)))
            else {
                let tp = cfg.tensor_parallel_size.fixed().unwrap_or(1);
                let key = if tp > 1 { TP_KEY } else { DEVICES_KEY };
                return Err(err(
                    key,
                    format!("needs {tp} GPU(s) of one vendor; the device inventory is empty"),
                ));
            };
            let mut pool = Vec::new();
            let mut excluded = Vec::new();
            for d in &inv.devices {
                if d.vendor == vendor {
                    pool.push(Candidate {
                        id: d.index,
                        arch: d.arch.clone(),
                    });
                } else {
                    excluded.push(d.index);
                }
            }
            pool.sort_by_key(|c| c.id);
            excluded.sort();
            if by_vendor.len() == 1 {
                reasons.push(PlanReason::VendorHomogeneous);
            }
            for &other in by_vendor.keys().filter(|v| **v != vendor) {
                reasons.push(PlanReason::VendorExcluded(other));
            }
            (vendor, pool, excluded, false)
        }
    };
    let distinct = pool.iter().map(|c| c.id).collect::<BTreeSet<_>>().len() as u32;

    // 2. Tensor-parallel size.
    let tp = match cfg.tensor_parallel_size {
        SizeOrAuto::Size(tp) => {
            heads_ok(tp, model).map_err(|reason| err(TP_KEY, reason))?;
            tp
        }
        SizeOrAuto::Auto => {
            let budget = pool.iter().map(|c| device_budget(c.id)).min().unwrap_or(0);
            let tp = [1u32, 2, 4, 8]
                .into_iter()
                .filter(|&p| p <= distinct.max(1) && heads_ok(p, model).is_ok())
                .find(|&p| per_rank_bytes(p, model) <= budget)
                .ok_or_else(|| {
                    err(
                        TP_KEY,
                        format!(
                            "auto: no tensor-parallel size up to {} fits the model's shard and \
                             one maximum-length sequence in a {budget}-byte device budget",
                            distinct.clamp(1, 8)
                        ),
                    )
                })?;
            reasons.push(if tp == 1 {
                PlanReason::FitsSingleDevice
            } else {
                PlanReason::TpRequiredForCapacity
            });
            tp
        }
    };
    if tp > distinct {
        return Err(err(
            TP_KEY,
            format!(
                "{tp} ranks need {tp} {} devices, {distinct} usable",
                vendor.as_str()
            ),
        ));
    }

    // 3. Data-parallel size and device sharing.
    let slots = if explicit {
        pool.len() as u32
    } else {
        distinct
    };
    let dp = match cfg.data_parallel_size {
        SizeOrAuto::Size(dp) => dp,
        SizeOrAuto::Auto => (slots / tp).max(1),
    };
    let distinct_groups = (slots / tp) as usize;
    let sharing = tp * dp > slots || (explicit && distinct < slots);
    if sharing && !cfg.allow_device_sharing {
        return Err(err(
            DP_KEY,
            format!(
                "{dp} replicas of {tp} ranks need {} devices, {slots} available (set \
                 parallel.allow_device_sharing to share devices between replicas)",
                tp * dp
            ),
        ));
    }
    if sharing {
        reasons.push(PlanReason::DeviceSharingEnabled);
    }

    // 4. TP groups.
    let base: Vec<Vec<DeviceId>> = if explicit {
        let chunks: Vec<Vec<DeviceId>> = pool
            .chunks(tp as usize)
            .map(|c| c.iter().map(|c| c.id).collect())
            .collect();
        if pool.len() % tp as usize != 0 {
            return Err(err(
                DEVICES_KEY,
                format!("{} devices do not split into TP groups of {tp}", pool.len()),
            ));
        }
        for (chunk, group) in pool.chunks(tp as usize).zip(&chunks) {
            if chunk.iter().any(|c| c.arch != chunk[0].arch) {
                return Err(err(TP_KEY, "a TP group mixes device architectures"));
            }
            if group.iter().collect::<BTreeSet<_>>().len() != group.len() {
                return Err(err(DEVICES_KEY, "a TP group lists one device twice"));
            }
        }
        chunks
    } else {
        link_groups(&pool, tp as usize, distinct_groups.min(dp as usize), &links)?
    };
    let host = topo.node.hostname.clone();
    let groups: Vec<ReplicaGroup> = (0..dp as usize)
        .map(|r| ReplicaGroup {
            replica: ReplicaId(r as u32),
            ranks: base[r % base.len()]
                .iter()
                .enumerate()
                .map(|(rank, device)| RankSlot {
                    rank: rank as u32,
                    device: *device,
                    host: host.clone(),
                })
                .collect(),
        })
        .collect();
    if tp > 1 {
        let mut seen = Vec::new();
        for g in &base {
            if let Some(worst) = links.worst(g)
                && !seen.contains(&worst)
            {
                seen.push(worst);
                reasons.push(PlanReason::GroupedByLink(worst));
            }
        }
    }

    // 5. Collective backend.
    let backend = choose_backend(cfg.collective_backend.as_str(), vendor, tp, cfg.ranks.mode)?;

    for r in &reasons {
        tracing::info!(event = "parallel_plan_decision", reason = %r, "parallel plan decision");
    }
    tracing::info!(
        event = "parallel_plan",
        tp,
        dp,
        backend,
        vendor = vendor.as_str(),
        mode = ?cfg.ranks.mode,
        excluded = ?excluded,
        "parallel plan"
    );
    Ok(ParallelPlan {
        tp,
        dp,
        backend,
        mode: cfg.ranks.mode,
        vendor: Some(vendor),
        excluded_devices: excluded,
        groups,
        reasons,
    })
}

/// The plan of a server whose replicas all run on `execution.device`, without consulting the
/// inventory: the `cpu` reference backend (`vendor` `None`; every replica shares the host) and
/// the single-GPU default (tp 1, dp 1, `devices: auto`), whose device and vendor the kernel
/// provider checks when it loads (exit 1). Tensor parallelism needs GPUs, and so do the `rccl`
/// and `nccl` backends; `auto` sizes resolve to 1.
pub fn plan_execution_device(
    cfg: &ParallelConfig,
    device: DeviceId,
    vendor: Option<Vendor>,
    host: &str,
) -> Result<ParallelPlan, PlanError> {
    let tp = cfg.tensor_parallel_size.fixed().unwrap_or(1);
    if tp != 1 {
        return Err(err(
            TP_KEY,
            format!("{tp} ranks need {tp} GPUs; execution.backend cpu runs tensor_parallel_size 1"),
        ));
    }
    let dp = cfg.data_parallel_size.fixed().unwrap_or(1);
    let mut reasons = vec![PlanReason::ExecutionDevice];
    if dp > 1 {
        if !cfg.allow_device_sharing {
            return Err(err(
                DP_KEY,
                format!(
                    "{dp} replicas would share execution.device {} (set \
                     parallel.allow_device_sharing to share it)",
                    device.0
                ),
            ));
        }
        reasons.push(PlanReason::DeviceSharingEnabled);
    }
    let name = cfg.collective_backend.as_str();
    let backend = match vendor {
        Some(v) => choose_backend(name, v, 1, RankMode::Local)?,
        // The cpu reference backend has no device memory: only a host-memory backend.
        None => match crate::collective::registry().get(name) {
            _ if name == "auto" => crate::collective::HostBackend.name(),
            Some(b) if b.vendors().is_empty() => b.name(),
            _ => {
                return Err(err(
                    BACKEND_KEY,
                    format!(
                        "`{name}` cannot drive the cpu reference backend (execution.device {})",
                        device.0
                    ),
                ));
            }
        },
    };
    let groups = (0..dp)
        .map(|r| ReplicaGroup {
            replica: ReplicaId(r),
            ranks: vec![RankSlot {
                rank: 0,
                device,
                host: host.to_string(),
            }],
        })
        .collect();
    for r in &reasons {
        tracing::info!(event = "parallel_plan_decision", reason = %r, "parallel plan decision");
    }
    tracing::info!(
        event = "parallel_plan",
        tp,
        dp,
        backend,
        vendor = vendor.map_or("none", |v| v.as_str()),
        device = device.0,
        "parallel plan"
    );
    Ok(ParallelPlan {
        tp,
        dp,
        backend,
        mode: cfg.ranks.mode,
        vendor,
        excluded_devices: Vec::new(),
        groups,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use turbine_core::config::{DeviceSelection, ParallelConfig, SizeOrAuto};
    use turbine_core::types::{DeviceId, MemoryKind, ModelShape, ReplicaId, Vendor};
    use turbine_device::topology::{
        AttrSource, Edge, EdgeKind, P2pStatus, PathClass, TopologyGraph, TopologyNode,
    };
    use turbine_device::{DeviceInfo, DeviceInventory, DeviceMemoryInfo};

    use super::*;

    const GIB: u64 = 1 << 30;

    fn gpu(index: u32, vendor: Vendor, arch: &str) -> DeviceInfo {
        DeviceInfo {
            index: DeviceId(index),
            vendor,
            vendor_index: index,
            name: format!("gpu{index}"),
            uuid: None,
            pci_bus_id: None,
            arch: Some(arch.to_string()),
            driver_version: None,
            memory: DeviceMemoryInfo {
                kind: MemoryKind::Dedicated,
                total_bytes: 32 * GIB,
                shared_with_host: false,
            },
        }
    }

    fn inventory(devices: Vec<DeviceInfo>) -> DeviceInventory {
        DeviceInventory {
            devices,
            backends: Vec::new(),
        }
    }

    fn link(a: u32, b: u32, kind: EdgeKind, path: PathClass) -> Edge {
        Edge {
            a: format!("gpu{a}"),
            b: format!("gpu{b}"),
            kind,
            path: Some(path),
            hops: None,
            link_gts: None,
            width: None,
            p2p: P2pStatus::Unknown,
            vendor_interconnect: None,
            rdma: None,
            source: Some(AttrSource::Nominal),
        }
    }

    fn graph(edges: Vec<Edge>) -> TopologyGraph {
        TopologyGraph {
            node: TopologyNode {
                hostname: "testhost".into(),
                captured_at: "2026-09-26T00:00:00Z".into(),
            },
            vertices: Vec::new(),
            edges,
        }
    }

    /// meta-llama/Llama-3.2-3B-Instruct.
    fn llama_3b() -> ModelShape {
        ModelShape {
            architecture: "LlamaForCausalLM".into(),
            num_layers: 28,
            hidden: 3072,
            num_attention_heads: 24,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate: 8192,
            vocab: 128_256,
            num_experts: 0,
            experts_per_token: 0,
            tied_embeddings: true,
            weight_bytes: 6_425_499_648,
            max_position_embeddings: 131_072,
        }
    }

    fn shape(heads: u32, kv_heads: u32) -> ModelShape {
        ModelShape {
            num_attention_heads: heads,
            num_kv_heads: kv_heads,
            weight_bytes: GIB,
            max_position_embeddings: 4096,
            ..llama_3b()
        }
    }

    fn cfg(tp: SizeOrAuto, dp: SizeOrAuto) -> ParallelConfig {
        ParallelConfig {
            tensor_parallel_size: tp,
            data_parallel_size: dp,
            ..ParallelConfig::default()
        }
    }

    fn budget_32g(_: DeviceId) -> u64 {
        32 * GIB
    }

    fn devices_of(plan: &ParallelPlan) -> Vec<Vec<u32>> {
        plan.groups
            .iter()
            .map(|g| g.ranks.iter().map(|r| r.device.0).collect())
            .collect()
    }

    fn codes(plan: &ParallelPlan) -> Vec<String> {
        plan.reasons.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn planner_cases() {
        // novanas: two gfx1201 over PCIe through the CPU (SYS), tp 2.
        let novanas = inventory(vec![
            gpu(0, Vendor::Amd, "gfx1201"),
            gpu(1, Vendor::Amd, "gfx1201"),
        ]);
        let novanas_graph = graph(vec![link(0, 1, EdgeKind::Pcie, PathClass::Sys)]);
        let p = plan(
            &novanas,
            &novanas_graph,
            &cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(1)),
            &llama_3b(),
            &budget_32g,
        )
        .expect("novanas tp 2");
        assert_eq!((p.tp, p.dp), (2, 1));
        assert_eq!(p.backend, "rccl");
        assert_eq!(p.vendor, Some(Vendor::Amd));
        assert_eq!(devices_of(&p), vec![vec![0, 1]]);
        assert_eq!(p.groups[0].replica, ReplicaId(0));
        assert_eq!(p.groups[0].ranks[1].rank, 1);
        assert_eq!(p.groups[0].ranks[0].host, "testhost");
        assert!(
            codes(&p).contains(&"grouped_by_link:sys".to_string()),
            "{:?}",
            codes(&p)
        );
        assert!(
            codes(&p).contains(&"vendor_homogeneous".to_string()),
            "{:?}",
            codes(&p)
        );

        // Four NVIDIA GPUs as two NVLink pairs {0,2} and {1,3}; everything else SYS.
        let nv = inventory((0..4).map(|i| gpu(i, Vendor::Nvidia, "sm_90")).collect());
        let mut edges = vec![
            link(0, 2, EdgeKind::Nvlink, PathClass::Nvlink),
            link(1, 3, EdgeKind::Nvlink, PathClass::Nvlink),
        ];
        for (a, b) in [(0, 1), (0, 3), (1, 2), (2, 3)] {
            edges.push(link(a, b, EdgeKind::Pcie, PathClass::Sys));
        }
        let p = plan(
            &nv,
            &graph(edges),
            &cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(2)),
            &llama_3b(),
            &budget_32g,
        )
        .expect("two NVLink pairs");
        assert_eq!(p.backend, "nccl");
        // Grouped by link, not by index order.
        assert_eq!(devices_of(&p), vec![vec![0, 2], vec![1, 3]]);
        assert_eq!(p.groups[1].replica, ReplicaId(1));
        assert!(
            codes(&p).contains(&"grouped_by_link:nvlink".to_string()),
            "{:?}",
            codes(&p)
        );

        // Explicit vendor-mixed lists are rejected, for TP and for DP.
        let mixed = inventory(vec![
            gpu(0, Vendor::Nvidia, "sm_121"),
            gpu(1, Vendor::Amd, "gfx1201"),
        ]);
        for (tp, dp) in [(2, 1), (1, 2)] {
            let mut c = cfg(SizeOrAuto::Size(tp), SizeOrAuto::Size(dp));
            c.devices = DeviceSelection::List(vec![DeviceId(0), DeviceId(1)]);
            let err = plan(&mixed, &graph(Vec::new()), &c, &llama_3b(), &budget_32g)
                .expect_err("vendor-mixed");
            assert_eq!(err.key, "parallel.devices");
            assert!(err.to_string().contains("vendor-mixed plan"), "{err}");
        }

        // Three AMD + one NVIDIA, devices auto, tp 1, dp auto: dp 3 on AMD, NVIDIA excluded.
        let three_one = inventory(vec![
            gpu(0, Vendor::Nvidia, "sm_121"),
            gpu(1, Vendor::Amd, "gfx1201"),
            gpu(2, Vendor::Amd, "gfx1201"),
            gpu(3, Vendor::Amd, "gfx1201"),
        ]);
        let p = plan(
            &three_one,
            &graph(Vec::new()),
            &cfg(SizeOrAuto::Size(1), SizeOrAuto::Auto),
            &llama_3b(),
            &budget_32g,
        )
        .expect("dp over the AMD devices");
        assert_eq!((p.tp, p.dp, p.vendor), (1, 3, Some(Vendor::Amd)));
        assert_eq!(p.excluded_devices, vec![DeviceId(0)]);
        assert_eq!(devices_of(&p), vec![vec![1], vec![2], vec![3]]);
        assert_eq!(p.backend, "host", "tp 1 needs no communicator");
        assert!(
            codes(&p).contains(&"vendor_excluded:nvidia".to_string()),
            "{:?}",
            codes(&p)
        );

        // tp auto on two 32 GiB R9700 budgets: the 3B model fits one device.
        let p = plan(
            &novanas,
            &novanas_graph,
            &cfg(SizeOrAuto::Auto, SizeOrAuto::Auto),
            &llama_3b(),
            &budget_32g,
        )
        .expect("auto sizes");
        assert_eq!((p.tp, p.dp), (1, 2));
        assert!(
            codes(&p).contains(&"fits_single_device".to_string()),
            "{:?}",
            codes(&p)
        );
        // ... but not a 12 GiB one: two ranks are required.
        let p = plan(
            &novanas,
            &novanas_graph,
            &cfg(SizeOrAuto::Auto, SizeOrAuto::Size(1)),
            &llama_3b(),
            &|_| 12 * GIB,
        )
        .expect("tp required");
        assert_eq!(p.tp, 2);
        assert!(
            codes(&p).contains(&"tp_required_for_capacity".to_string()),
            "{:?}",
            codes(&p)
        );

        // Head rules: tp 4 with 2 KV heads (replicated) is accepted; tp 8 with 3 KV heads is not.
        let four = inventory((0..4).map(|i| gpu(i, Vendor::Amd, "gfx1201")).collect());
        let p = plan(
            &four,
            &graph(Vec::new()),
            &cfg(SizeOrAuto::Size(4), SizeOrAuto::Size(1)),
            &shape(8, 2),
            &budget_32g,
        )
        .expect("kv heads replicated at tp 4");
        assert_eq!(p.tp, 4);
        let eight = inventory((0..8).map(|i| gpu(i, Vendor::Amd, "gfx1201")).collect());
        let err = plan(
            &eight,
            &graph(Vec::new()),
            &cfg(SizeOrAuto::Size(8), SizeOrAuto::Size(1)),
            &shape(24, 3),
            &budget_32g,
        )
        .expect_err("3 KV heads over 8 ranks");
        assert_eq!(err.key, "parallel.tensor_parallel_size");

        // Too few devices, and an empty inventory, name the TP size.
        let err = plan(
            &inventory(Vec::new()),
            &graph(Vec::new()),
            &cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(1)),
            &llama_3b(),
            &budget_32g,
        )
        .expect_err("no devices");
        assert_eq!(err.key, "parallel.tensor_parallel_size");
        let err = plan(
            &novanas,
            &novanas_graph,
            &cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(2)),
            &llama_3b(),
            &budget_32g,
        )
        .expect_err("four ranks on two devices");
        assert_eq!(err.key, "parallel.data_parallel_size");

        // TP groups never mix architectures.
        let archs = inventory(vec![
            gpu(0, Vendor::Amd, "gfx1201"),
            gpu(1, Vendor::Amd, "gfx942"),
        ]);
        let err = plan(
            &archs,
            &graph(vec![link(0, 1, EdgeKind::Pcie, PathClass::Sys)]),
            &cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(1)),
            &llama_3b(),
            &budget_32g,
        )
        .expect_err("mixed architectures");
        assert_eq!(err.key, "parallel.tensor_parallel_size");

        // Two replicas sharing one device with allow_device_sharing.
        let mut shared = cfg(SizeOrAuto::Size(1), SizeOrAuto::Size(2));
        shared.allow_device_sharing = true;
        let one = inventory(vec![gpu(0, Vendor::Amd, "gfx1201")]);
        let p = plan(&one, &graph(Vec::new()), &shared, &llama_3b(), &budget_32g)
            .expect("shared device");
        assert_eq!(devices_of(&p), vec![vec![0], vec![0]]);
        assert!(
            codes(&p).contains(&"device_sharing_enabled".to_string()),
            "{:?}",
            codes(&p)
        );

        // Explicit lists keep the listed grouping.
        let mut explicit = cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(1));
        explicit.devices = DeviceSelection::List(vec![DeviceId(1), DeviceId(0)]);
        let p = plan(
            &novanas,
            &novanas_graph,
            &explicit,
            &llama_3b(),
            &budget_32g,
        )
        .expect("explicit");
        assert_eq!(devices_of(&p), vec![vec![1, 0]]);
        assert!(
            codes(&p).contains(&"explicit_devices".to_string()),
            "{:?}",
            codes(&p)
        );
    }

    /// The backend comes from the `collective_backend` registry: explicit names must serve the
    /// plan, `auto` takes the first backend of the vendor (host for tp 1).
    #[test]
    fn backend_from_registry() {
        use super::choose_backend;
        let (local, stat) = (RankMode::Local, RankMode::Static);
        assert_eq!(choose_backend("auto", Vendor::Amd, 2, local), Ok("rccl"));
        assert_eq!(choose_backend("auto", Vendor::Amd, 2, stat), Ok("rccl"));
        assert_eq!(choose_backend("auto", Vendor::Nvidia, 4, local), Ok("nccl"));
        assert_eq!(choose_backend("auto", Vendor::Amd, 1, local), Ok("host"));
        assert_eq!(choose_backend("host", Vendor::Amd, 1, local), Ok("host"));
        assert_eq!(choose_backend("rccl", Vendor::Amd, 1, local), Ok("rccl"));
        assert_eq!(
            choose_backend("hostmem", Vendor::Amd, 2, local),
            Ok("hostmem")
        );
        for (name, vendor, tp, mode) in [
            ("host", Vendor::Amd, 2, local),
            ("rccl", Vendor::Nvidia, 2, local),
            ("nccl", Vendor::Amd, 2, local),
            ("gloo", Vendor::Amd, 2, local),
            ("hostmem", Vendor::Nvidia, 2, local),
            ("hostmem", Vendor::Amd, 2, stat),
        ] {
            let e = choose_backend(name, vendor, tp, mode).expect_err(name);
            assert_eq!(e.key, "parallel.collective_backend", "{name}: {e}");
        }
        let e = choose_backend("hostmem", Vendor::Amd, 2, stat).expect_err("static");
        assert!(e.reason.contains("parallel.ranks.mode local"), "{e}");
    }

    /// The server's single-device paths: the `cpu` reference backend (no vendor, every replica
    /// on `execution.device`) and the default GPU plan (tp 1, dp 1, `devices: auto` →
    /// `execution.device`), which never consult the inventory.
    #[test]
    fn execution_device_plans() {
        use turbine_core::config::ModuleName;
        let default = ParallelConfig::default();
        let p = plan_execution_device(&default, DeviceId(1), Some(Vendor::Amd), "h")
            .expect("default GPU plan");
        assert_eq!((p.tp, p.dp), (1, 1));
        assert_eq!(p.vendor, Some(Vendor::Amd));
        assert_eq!(p.backend, "host");
        assert_eq!(devices_of(&p), vec![vec![1]]);
        assert_eq!(p.groups[0].ranks[0].host, "h");
        assert_eq!(codes(&p), vec!["execution_device"]);

        // cpu: dp 2 on the one host "device" only with sharing.
        let dp2 = cfg(SizeOrAuto::Size(1), SizeOrAuto::Size(2));
        let e = plan_execution_device(&dp2, DeviceId(0), None, "h").expect_err("no sharing");
        assert_eq!(e.key, "parallel.data_parallel_size");
        assert!(e.reason.contains("allow_device_sharing"), "{e}");
        let mut shared = dp2.clone();
        shared.allow_device_sharing = true;
        let p = plan_execution_device(&shared, DeviceId(0), None, "h").expect("shared");
        assert_eq!(p.vendor, None);
        assert_eq!(devices_of(&p), vec![vec![0], vec![0]]);
        assert_eq!(p.groups[1].replica, ReplicaId(1));
        assert_eq!(
            codes(&p),
            vec!["execution_device", "device_sharing_enabled"]
        );
        let mut auto = shared.clone();
        auto.data_parallel_size = SizeOrAuto::Auto;
        auto.tensor_parallel_size = SizeOrAuto::Auto;
        let p = plan_execution_device(&auto, DeviceId(0), None, "h").expect("auto");
        assert_eq!((p.tp, p.dp), (1, 1));

        // Tensor parallelism and GPU communicators need GPUs.
        let tp2 = cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(1));
        let e = plan_execution_device(&tp2, DeviceId(0), None, "h").expect_err("cpu tp 2");
        assert_eq!(e.key, "parallel.tensor_parallel_size");
        let rccl = ParallelConfig {
            collective_backend: ModuleName::new("rccl").unwrap(),
            ..ParallelConfig::default()
        };
        let e = plan_execution_device(&rccl, DeviceId(0), None, "h").expect_err("cpu rccl");
        assert_eq!(e.key, "parallel.collective_backend");
        let p = plan_execution_device(&rccl, DeviceId(0), Some(Vendor::Amd), "h")
            .expect("rccl kept for an AMD device");
        assert_eq!(p.backend, "rccl");
        let nccl = ParallelConfig {
            collective_backend: ModuleName::new("nccl").unwrap(),
            ..ParallelConfig::default()
        };
        let e = plan_execution_device(&nccl, DeviceId(0), Some(Vendor::Amd), "h")
            .expect_err("nccl on AMD");
        assert_eq!(e.key, "parallel.collective_backend");
    }
}
