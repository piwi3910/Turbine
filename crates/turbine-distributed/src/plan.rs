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
//!
//! Expert parallelism (P5 S-11, S-12): `expert_parallel_size` ep > 1 needs a model with routed
//! experts (`ep_moe_only`), divides their count and runs with tp ∈ {1, ep}; a group is
//! max(tp, ep) ranks (attention replicated at tp = 1, tensor-parallel over the same ranks at
//! tp = ep) and the plan carries the expert placement ([`expert_placement`]). `ep: auto` is 1:
//! capacity is tensor parallelism's job (`tp: auto`).
//!
//! Pipeline parallelism (P5 S-10, S-12): `pipeline_parallel_size` pp > 1 runs with tp = ep = 1
//! in `local` ranks only (anything else is `combination_unsupported:<modes>`); a group is then
//! pp devices, one per stage. [`plan_stages`] splits the layers ([`crate::pipeline::partition`]
//! over [`PipelineCosts::from_shape`], reason `pp_partition_cost_balanced`, or the explicit
//! `parallel.pipeline.layer_split`, `pp_partition_explicit`) and places each group's stages by
//! measured host link ([`crate::pipeline::place_stages`], `pp_stage_host_traffic:<device>`; no
//! measurement: `pp_stage_device_order`); a group's ranks are its stages, in stage order.
//! `pp: auto` is 1 unless the model and one maximum-length sequence do not fit one device at
//! tp 1 (`pp_required_for_capacity`, the smallest pp that fits).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use turbine_core::config::{
    DeviceSelection, ExpertPlacementChoice, ParallelConfig, RankMode, SizeOrAuto,
};
use turbine_core::registry::Module;
use turbine_core::types::{DeviceId, ModelShape, ReplicaId, Vendor};
use turbine_device::DeviceInventory;
use turbine_device::topology::{PathClass, TopologyGraph};

use crate::expert::{EP_KEY, ExpertPlacement};
use crate::pipeline::{PP_KEY, PipelineCosts, StageReason, StageSpec, partition, place_stages};

/// The process-lifetime multi-GPU plan (P5 §Data).
#[derive(Clone, Debug, PartialEq)]
pub struct ParallelPlan {
    pub tp: u32,
    pub dp: u32,
    /// Pipeline stages per group (P5 S-10); > 1 only with tp = ep = 1.
    pub pp: u32,
    /// Expert-parallel ranks per group (P5 S-11); a group has max(tp, ep) ranks.
    pub ep: u32,
    /// The registered collective backend (`collective_backend` registry).
    pub backend: &'static str,
    pub mode: RankMode,
    /// The plan's one vendor; `None` for the `cpu` reference backend, which has no GPU.
    pub vendor: Option<Vendor>,
    pub excluded_devices: Vec<DeviceId>,
    pub groups: Vec<ReplicaGroup>,
    pub reasons: Vec<PlanReason>,
    /// ep > 1: which rank of a group holds each routed expert (the same for every group).
    pub experts: Option<Arc<ExpertPlacement>>,
    /// pp > 1: replica 0's stages in stage order (every group has the same layer ranges; group
    /// `g`'s stage `s` runs on `groups[g].ranks[s].device`); empty otherwise.
    pub stages: Vec<StageSpec>,
}

impl ParallelPlan {
    /// Ranks per tensor- or expert-parallel group: max(tp, ep).
    pub fn group_size(&self) -> u32 {
        self.tp.max(self.ep)
    }

    /// Devices one model instance (one DP replica) spans: pp × max(tp, ep).
    pub fn replica_size(&self) -> u32 {
        self.pp.max(1) * self.group_size()
    }
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
    /// A size set explicitly in the configuration (expert parallelism).
    Configured,
    /// Expert parallelism places only the routed experts; everything else is replicated (tp 1)
    /// or tensor-parallel over the same ranks (tp = ep). Also the refusal code of ep on a dense
    /// model.
    EpMoeOnly,
    /// The collective backend a multi-rank group runs on.
    Backend(&'static str),
    /// `pipeline_parallel_size: auto` and the model does not fit one device at tp 1.
    PpRequiredForCapacity,
    /// The layers were split by estimated stage cost.
    PpPartitionCostBalanced,
    /// The layers were split as `parallel.pipeline.layer_split` says.
    PpPartitionExplicit,
    /// The stage with the most host-side traffic (the last) is on this device, the fastest
    /// measured host link.
    PpStageHostTraffic(DeviceId),
    /// No host link was measured: the stages follow device order.
    PpStageDeviceOrder,
    /// A combination of parallel modes Phase 5 does not run (`pp+tp`, `pp+ep`, `pp+static`,
    /// `ep+tp`): the code of a refusal (a [`PlanError`]'s reason starts with it).
    CombinationUnsupported(&'static str),
}

impl From<&StageReason> for PlanReason {
    fn from(r: &StageReason) -> PlanReason {
        match r {
            StageReason::HostTraffic(d) => PlanReason::PpStageHostTraffic(*d),
            StageReason::NominalLinks => PlanReason::PpStageDeviceOrder,
        }
    }
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
            PlanReason::Configured => f.write_str("configured"),
            PlanReason::EpMoeOnly => f.write_str("ep_moe_only"),
            PlanReason::Backend(name) => write!(f, "backend:{name}"),
            PlanReason::PpRequiredForCapacity => f.write_str("pp_required_for_capacity"),
            PlanReason::PpPartitionCostBalanced => f.write_str("pp_partition_cost_balanced"),
            PlanReason::PpPartitionExplicit => f.write_str("pp_partition_explicit"),
            PlanReason::PpStageHostTraffic(d) => write!(f, "pp_stage_host_traffic:{}", d.0),
            PlanReason::PpStageDeviceOrder => f.write_str("pp_stage_device_order"),
            PlanReason::CombinationUnsupported(modes) => {
                write!(f, "combination_unsupported:{modes}")
            }
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

/// The expert-parallel size of `cfg` (`auto` is 1) checked against the model: ep > 1 needs
/// `num_experts` routed experts (`ep_moe_only`), divisible by ep, and `tp` of 1 or ep. The
/// reasons of an EP plan are `configured` and `ep_moe_only`.
pub fn expert_size(
    cfg: &ParallelConfig,
    num_experts: u32,
    architecture: &str,
    tp: u32,
    reasons: &mut Vec<PlanReason>,
) -> Result<u32, PlanError> {
    let ep = cfg.expert_parallel_size.fixed().unwrap_or(1);
    if ep <= 1 {
        return Ok(1);
    }
    if num_experts == 0 {
        return Err(err(
            EP_KEY,
            format!(
                "ep_moe_only: {ep} expert-parallel ranks need a mixture-of-experts model; \
                 {architecture} has no routed experts"
            ),
        ));
    }
    if !num_experts.is_multiple_of(ep) {
        return Err(err(
            EP_KEY,
            format!("{ep} does not divide the model's {num_experts} routed experts"),
        ));
    }
    if tp != 1 && tp != ep {
        return Err(err(
            EP_KEY,
            format!(
                "combination_unsupported:ep+tp: expert parallelism {ep} runs with \
                 tensor_parallel_size 1 or {ep}, not {tp}"
            ),
        ));
    }
    reasons.push(PlanReason::Configured);
    reasons.push(PlanReason::EpMoeOnly);
    Ok(ep)
}

/// The combinations of parallel modes Phase 5 runs with pipeline stages (S-12): pp > 1 needs
/// tp = ep = 1 and `local` ranks; anything else is refused naming the key and
/// `combination_unsupported:<modes>`.
pub fn check_pipeline_combination(
    pp: u32,
    tp: u32,
    ep: u32,
    mode: RankMode,
) -> Result<(), PlanError> {
    if pp <= 1 {
        return Ok(());
    }
    let refuse = |modes: &'static str, why: &str| {
        Err(err(
            PP_KEY,
            format!("{}: {why}", PlanReason::CombinationUnsupported(modes)),
        ))
    };
    if tp > 1 {
        return refuse(
            "pp+tp",
            "pipeline stages run one device each in Phase 5 (tensor_parallel_size 1)",
        );
    }
    if ep > 1 {
        return refuse(
            "pp+ep",
            "pipeline and expert parallelism do not combine in Phase 5",
        );
    }
    if mode != RankMode::Local {
        return refuse("pp+static", "pipeline stages run in `local` ranks only");
    }
    Ok(())
}

/// The pipeline-parallel size of `cfg` (module comment): a fixed size (reason `configured`
/// when > 1), or for `auto` 1 unless `fits(1)` fails at tp = ep = 1, then the smallest pp up to
/// `devices` that `fits` (`pp_required_for_capacity`). Checked against the other modes.
fn pipeline_size(
    cfg: &ParallelConfig,
    tp: u32,
    ep: u32,
    devices: u32,
    reasons: &mut Vec<PlanReason>,
    fits: impl Fn(u32) -> bool,
) -> Result<u32, PlanError> {
    let pp = match cfg.pipeline_parallel_size {
        SizeOrAuto::Size(pp) => {
            if pp > 1 && !reasons.contains(&PlanReason::Configured) {
                reasons.push(PlanReason::Configured);
            }
            pp
        }
        SizeOrAuto::Auto if tp == 1 && ep == 1 && !fits(1) => {
            let pp = (2..=devices.max(1)).find(|&p| fits(p)).ok_or_else(|| {
                err(
                    PP_KEY,
                    format!(
                        "auto: no pipeline of up to {devices} stages fits the model and one \
                         maximum-length sequence in the device budget"
                    ),
                )
            })?;
            reasons.push(PlanReason::PpRequiredForCapacity);
            pp
        }
        SizeOrAuto::Auto => 1,
    };
    check_pipeline_combination(pp, tp, ep, cfg.ranks.mode)?;
    Ok(pp)
}

/// The pipeline stages of every group of `plan` (P5 S-10; no-op for pp 1): the layers of
/// `model` split by estimated cost ([`PipelineCosts::from_shape`], `pp_partition_cost_balanced`)
/// or as `parallel.pipeline.layer_split` says (`pp_partition_explicit`), then each group's
/// stages placed on its devices by `host_link_gbps` (the measured host link, the slower
/// direction; `pp_stage_host_traffic:<device>`, or `pp_stage_device_order` unmeasured). Each
/// group's ranks become its stages in stage order; `plan.stages` holds replica 0's.
pub fn plan_stages(
    plan: &mut ParallelPlan,
    cfg: &ParallelConfig,
    model: &ModelShape,
    host_link_gbps: &dyn Fn(DeviceId) -> Option<f64>,
) -> Result<(), PlanError> {
    if plan.pp <= 1 {
        return Ok(());
    }
    let split = cfg.pipeline.layer_split.as_deref();
    let ranges = partition(&PipelineCosts::from_shape(model), plan.pp, split)?;
    let mut reasons = vec![if split.is_some() {
        PlanReason::PpPartitionExplicit
    } else {
        PlanReason::PpPartitionCostBalanced
    }];
    for g in &mut plan.groups {
        let devices: Vec<DeviceId> = g.ranks.iter().map(|r| r.device).collect();
        let host = g.ranks.first().map(|r| r.host.clone()).unwrap_or_default();
        let (stages, why) = place_stages(&ranges, &devices, host_link_gbps)?;
        let reason = PlanReason::from(&why);
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
        g.ranks = stages
            .iter()
            .map(|s| RankSlot {
                rank: s.stage,
                device: s.device,
                host: host.clone(),
            })
            .collect();
        if g.replica.0 == 0 {
            plan.stages = stages;
        }
    }
    for r in &reasons {
        tracing::info!(event = "parallel_plan_decision", reason = %r, "parallel plan decision");
    }
    tracing::info!(
        event = "pipeline_stages",
        pp = plan.pp,
        layers = ?ranges,
        devices = ?plan.stages.iter().map(|s| s.device.0).collect::<Vec<_>>(),
        "pipeline stages planned"
    );
    plan.reasons.extend(reasons);
    Ok(())
}

/// The placement of `num_experts` routed experts of every layer of `moe_layers` over `ep`
/// ranks (`parallel.expert.placement`: contiguous, or read from its file); `None` for ep 1.
pub fn expert_placement(
    cfg: &ParallelConfig,
    num_experts: u32,
    moe_layers: &[u32],
    ep: u32,
) -> Result<Option<Arc<ExpertPlacement>>, PlanError> {
    if ep <= 1 {
        return Ok(None);
    }
    let placement = match &cfg.expert.placement {
        ExpertPlacementChoice::Contiguous => {
            ExpertPlacement::contiguous(num_experts, moe_layers, ep)?
        }
        ExpertPlacementChoice::File(path) => {
            ExpertPlacement::from_file(path, num_experts, moe_layers, ep)?
        }
    };
    Ok(Some(Arc::new(placement)))
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
    // 2b. Expert-parallel size: a group is max(tp, ep) ranks.
    let ep = expert_size(
        cfg,
        model.num_experts,
        &model.architecture,
        tp,
        &mut reasons,
    )?;
    // 2c. Pipeline stages: a replica spans pp × max(tp, ep) devices.
    let pp = {
        let budget = pool.iter().map(|c| device_budget(c.id)).min().unwrap_or(0);
        let whole = per_rank_bytes(1, model);
        pipeline_size(cfg, tp, ep, distinct, &mut reasons, |pp| {
            whole.div_ceil(u64::from(pp)) <= budget
        })?
    };
    let group = tp.max(ep) * pp;
    let group_key = if pp > 1 {
        PP_KEY
    } else if ep > tp {
        EP_KEY
    } else {
        TP_KEY
    };
    if group > distinct {
        return Err(err(
            group_key,
            format!(
                "{group} ranks need {group} {} devices, {distinct} usable",
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
        SizeOrAuto::Auto => (slots / group).max(1),
    };
    let distinct_groups = (slots / group) as usize;
    let sharing = group * dp > slots || (explicit && distinct < slots);
    if sharing && !cfg.allow_device_sharing {
        return Err(err(
            DP_KEY,
            format!(
                "{dp} replicas of {group} ranks need {} devices, {slots} available (set \
                 parallel.allow_device_sharing to share devices between replicas)",
                group * dp
            ),
        ));
    }
    if sharing {
        reasons.push(PlanReason::DeviceSharingEnabled);
    }

    // 4. TP groups.
    let base: Vec<Vec<DeviceId>> = if explicit {
        let chunks: Vec<Vec<DeviceId>> = pool
            .chunks(group as usize)
            .map(|c| c.iter().map(|c| c.id).collect())
            .collect();
        if pool.len() % group as usize != 0 {
            return Err(err(
                DEVICES_KEY,
                format!("{} devices do not split into groups of {group}", pool.len()),
            ));
        }
        for (chunk, members) in pool.chunks(group as usize).zip(&chunks) {
            if chunk.iter().any(|c| c.arch != chunk[0].arch) {
                return Err(err(group_key, "a group mixes device architectures"));
            }
            if members.iter().collect::<BTreeSet<_>>().len() != members.len() {
                return Err(err(DEVICES_KEY, "a group lists one device twice"));
            }
        }
        chunks
    } else {
        link_groups(
            &pool,
            group as usize,
            distinct_groups.min(dp as usize),
            &links,
        )
        .map_err(|e| PlanError {
            key: group_key.to_string(),
            ..e
        })?
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
    if group > 1 {
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

    // 5. Collective backend, and the expert placement over every layer (a model shape with
    // routed experts has them in every layer; the server re-checks against the family).
    let backend = choose_backend(
        cfg.collective_backend.as_str(),
        vendor,
        group,
        cfg.ranks.mode,
    )?;
    if group > 1 {
        reasons.push(PlanReason::Backend(backend));
    }
    let moe_layers: Vec<u32> = (0..model.num_layers).collect();
    let experts = expert_placement(cfg, model.num_experts, &moe_layers, ep)?;

    for r in &reasons {
        tracing::info!(event = "parallel_plan_decision", reason = %r, "parallel plan decision");
    }
    tracing::info!(
        event = "parallel_plan",
        tp,
        dp,
        pp,
        ep,
        backend,
        vendor = vendor.as_str(),
        mode = ?cfg.ranks.mode,
        excluded = ?excluded,
        "parallel plan"
    );
    let mut plan = ParallelPlan {
        tp,
        dp,
        pp,
        ep,
        backend,
        mode: cfg.ranks.mode,
        vendor: Some(vendor),
        excluded_devices: excluded,
        groups,
        reasons,
        experts,
        stages: Vec::new(),
    };
    // 6. Pipeline stages on each group's devices by measured host link (S-10, S-13).
    plan_stages(&mut plan, cfg, model, &|d| topo.host_link_gbps(d))?;
    Ok(plan)
}

/// The plan of a server whose replicas all run on `execution.device`, without consulting the
/// inventory: the `cpu` reference backend (`vendor` `None`; every replica shares the host) and
/// the single-GPU default (tp 1, dp 1, `devices: auto`), whose device and vendor the kernel
/// provider checks when it loads (exit 1). Tensor parallelism needs GPUs, and so do the `rccl`
/// and `nccl` backends; `auto` sizes resolve to 1. The one exception is the host test path: the
/// cpu reference backend with an explicitly configured host-memory collective backend
/// (`parallel.collective_backend: host`) runs tp > 1 ranks (or pp > 1 stages) as threads, rank
/// `r` of every group on the host "device" `execution.device + r` (each rank its own host
/// memory). A pipeline's stages are placed afterwards by [`plan_stages`], which needs the model.
pub fn plan_execution_device(
    cfg: &ParallelConfig,
    device: DeviceId,
    vendor: Option<Vendor>,
    host: &str,
) -> Result<ParallelPlan, PlanError> {
    let tp = cfg.tensor_parallel_size.fixed().unwrap_or(1);
    let ep = cfg.expert_parallel_size.fixed().unwrap_or(1);
    let pp = cfg.pipeline_parallel_size.fixed().unwrap_or(1);
    check_pipeline_combination(pp, tp, ep, cfg.ranks.mode)?;
    let group = tp.max(ep) * pp;
    let host_backend = crate::collective::registry()
        .get(cfg.collective_backend.as_str())
        .is_some_and(|b| b.vendors().is_empty());
    if group != 1 && (vendor.is_some() || !host_backend) {
        return Err(err(
            if pp > 1 {
                PP_KEY
            } else if ep > tp {
                EP_KEY
            } else {
                TP_KEY
            },
            format!(
                "{group} ranks need {group} GPUs; execution.backend cpu runs one rank per \
                 group (or tensor- and expert-parallel ranks and pipeline stages as threads \
                 with parallel.collective_backend: host)"
            ),
        ));
    }
    if ep > 1 && tp != 1 && tp != ep {
        return Err(err(
            EP_KEY,
            format!(
                "combination_unsupported:ep+tp: expert parallelism {ep} runs with \
                 tensor_parallel_size 1 or {ep}, not {tp}"
            ),
        ));
    }
    let dp = cfg.data_parallel_size.fixed().unwrap_or(1);
    let mut reasons = vec![PlanReason::ExecutionDevice];
    if ep > 1 || pp > 1 {
        // The model's experts are checked (and placed) by the server against its config.
        reasons.push(PlanReason::Configured);
    }
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
            ranks: (0..group)
                .map(|rank| RankSlot {
                    rank,
                    device: DeviceId(device.0 + rank),
                    host: host.to_string(),
                })
                .collect(),
        })
        .collect();
    if group > 1 {
        reasons.push(PlanReason::Backend(backend));
    }
    for r in &reasons {
        tracing::info!(event = "parallel_plan_decision", reason = %r, "parallel plan decision");
    }
    tracing::info!(
        event = "parallel_plan",
        tp,
        dp,
        ep,
        backend,
        vendor = vendor.map_or("none", |v| v.as_str()),
        device = device.0,
        "parallel plan"
    );
    Ok(ParallelPlan {
        tp,
        dp,
        pp,
        ep,
        backend,
        mode: cfg.ranks.mode,
        vendor,
        excluded_devices: Vec::new(),
        groups,
        reasons,
        experts: None,
        stages: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use turbine_core::config::{
        DeviceSelection, ExpertPlacementChoice, ParallelConfig, SizeOrAuto,
    };
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
            measured_h2d_gbps: None,
            measured_d2h_gbps: None,
            cost_gbps: None,
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
        assert_eq!(p.backend, "hostmem", "auto: AMD, local ranks");
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

    /// allenai/OLMoE-1B-7B-0125-Instruct.
    fn olmoe() -> ModelShape {
        ModelShape {
            architecture: "OlmoeForCausalLM".into(),
            num_layers: 16,
            hidden: 2048,
            num_attention_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            intermediate: 1024,
            vocab: 50_304,
            num_experts: 64,
            experts_per_token: 8,
            tied_embeddings: false,
            weight_bytes: 13_838_000_000,
            max_position_embeddings: 4096,
        }
    }

    fn ep_cfg(tp: u32, ep: u32, dp: u32) -> ParallelConfig {
        ParallelConfig {
            expert_parallel_size: SizeOrAuto::Size(ep),
            ..cfg(SizeOrAuto::Size(tp), SizeOrAuto::Size(dp))
        }
    }

    /// P5 S-11, S-12: expert parallelism as a plan choice. On novanas OLMoE at ep 2 with tp 1
    /// (attention replicated) and with tp 2 (= ep) is one group of both GPUs, with reason codes
    /// `configured`, `ep_moe_only` and `backend:hostmem` and the contiguous placement (experts
    /// 0–31 on rank 0, 32–63 on rank 1, every layer); ep × dp builds ep-rank groups. Refused:
    /// ep on the dense Llama (`ep_moe_only`, naming the key), ep 3 (does not divide 64), tp 4
    /// with ep 2 (`combination_unsupported:ep+tp`), more ranks than devices, and a placement
    /// file missing an expert (naming `parallel.expert.placement`). Breaks if an unsupported
    /// combination plans or an EP group is sized by tp alone.
    #[test]
    fn expert_parallel_plans() {
        let novanas = inventory(vec![
            gpu(0, Vendor::Amd, "gfx1201"),
            gpu(1, Vendor::Amd, "gfx1201"),
        ]);
        let g = graph(vec![link(0, 1, EdgeKind::Pcie, PathClass::Sys)]);
        for tp in [1, 2] {
            let p = plan(&novanas, &g, &ep_cfg(tp, 2, 1), &olmoe(), &budget_32g)
                .unwrap_or_else(|e| panic!("ep 2 tp {tp}: {e}"));
            assert_eq!((p.tp, p.ep, p.dp, p.group_size()), (tp, 2, 1, 2));
            assert_eq!(devices_of(&p), vec![vec![0, 1]]);
            assert_eq!(p.backend, "hostmem", "auto: AMD, local ranks");
            for code in ["configured", "ep_moe_only", "backend:hostmem"] {
                assert!(codes(&p).contains(&code.to_string()), "{:?}", codes(&p));
            }
            let experts = p.experts.as_ref().expect("placement");
            assert_eq!(experts.ranks, 2);
            assert_eq!(experts.layers.len(), 16);
            assert_eq!(experts.local_experts(15, 0), (0..32).collect::<Vec<_>>());
            assert_eq!(experts.local_experts(0, 1), (32..64).collect::<Vec<_>>());
        }
        // tp 1 and ep 1 plan no placement and keep their reasons.
        let p = plan(&novanas, &g, &ep_cfg(1, 1, 2), &olmoe(), &budget_32g).expect("dp 2");
        assert_eq!((p.ep, p.experts.clone()), (1, None));
        assert!(!codes(&p).contains(&"ep_moe_only".to_string()));

        // ep × dp: two groups of two ranks on four GPUs.
        let four = inventory((0..4).map(|i| gpu(i, Vendor::Amd, "gfx1201")).collect());
        let p = plan(
            &four,
            &graph(Vec::new()),
            &ep_cfg(1, 2, 2),
            &olmoe(),
            &budget_32g,
        )
        .expect("ep 2 × dp 2");
        assert_eq!(devices_of(&p), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(p.group_size(), 2);

        let refused = |c: &ParallelConfig, model: &ModelShape| {
            plan(&novanas, &g, c, model, &budget_32g).expect_err("refused")
        };
        let e = refused(&ep_cfg(1, 2, 1), &llama_3b());
        assert_eq!(e.key, "parallel.expert_parallel_size");
        assert!(e.reason.starts_with("ep_moe_only"), "{e}");
        let e = refused(&ep_cfg(1, 3, 1), &olmoe());
        assert_eq!(e.key, "parallel.expert_parallel_size");
        assert!(e.reason.contains("does not divide"), "{e}");
        let e = refused(&ep_cfg(4, 2, 1), &olmoe());
        assert!(e.reason.starts_with("combination_unsupported:ep+tp"), "{e}");
        let e = refused(&ep_cfg(1, 4, 1), &olmoe());
        assert_eq!(e.key, "parallel.expert_parallel_size");
        assert!(e.reason.contains("4 ranks need 4"), "{e}");

        let dir = std::env::temp_dir().join(format!("turbine-plan-ep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("placement.yaml");
        let mut text = String::new();
        for layer in 0..16 {
            let n = if layer == 3 { 63 } else { 64 };
            let ranks: Vec<String> = (0..n).map(|e| (e % 2).to_string()).collect();
            text += &format!("{layer}: [{}]\n", ranks.join(", "));
        }
        std::fs::write(&file, text).unwrap();
        let mut from_file = ep_cfg(1, 2, 1);
        from_file.expert.placement = ExpertPlacementChoice::File(file.clone());
        let e = refused(&from_file, &olmoe());
        assert_eq!(e.key, "parallel.expert.placement");
        std::fs::remove_dir_all(&dir).ok();

        // The cpu host path: ep 2 as two rank threads on host devices 0 and 1.
        use turbine_core::config::ModuleName;
        let host = ParallelConfig {
            collective_backend: ModuleName::new("host").unwrap(),
            ..ep_cfg(1, 2, 1)
        };
        let p = plan_execution_device(&host, DeviceId(0), None, "h").expect("host ep 2");
        assert_eq!((p.tp, p.ep), (1, 2));
        assert_eq!(devices_of(&p), vec![vec![0, 1]]);
        assert_eq!(
            codes(&p),
            vec!["execution_device", "configured", "backend:host"]
        );
        let e = plan_execution_device(&ep_cfg(1, 2, 1), DeviceId(0), None, "h")
            .expect_err("cpu ep without the host backend");
        assert_eq!(e.key, "parallel.expert_parallel_size");
        let e = plan_execution_device(
            &ParallelConfig {
                collective_backend: ModuleName::new("host").unwrap(),
                ..ep_cfg(4, 2, 1)
            },
            DeviceId(0),
            None,
            "h",
        )
        .expect_err("tp 4 ep 2");
        assert!(e.reason.starts_with("combination_unsupported:ep+tp"), "{e}");
    }

    /// The backend comes from the `collective_backend` registry: explicit names must serve the
    /// plan, `auto` takes the first backend of the vendor (host for tp 1).
    #[test]
    fn backend_from_registry() {
        use super::choose_backend;
        let (local, stat) = (RankMode::Local, RankMode::Static);
        assert_eq!(choose_backend("auto", Vendor::Amd, 2, local), Ok("hostmem"));
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
        // … except the host test path: cpu ranks as threads over an explicit host backend,
        // rank r on host device `execution.device + r`, one group per replica.
        let host_tp = ParallelConfig {
            collective_backend: ModuleName::new("host").unwrap(),
            allow_device_sharing: true,
            ..cfg(SizeOrAuto::Size(2), SizeOrAuto::Size(2))
        };
        let p = plan_execution_device(&host_tp, DeviceId(3), None, "h").expect("host tp 2");
        assert_eq!((p.tp, p.dp, p.backend), (2, 2, "host"));
        let devices: Vec<Vec<u32>> = p
            .groups
            .iter()
            .map(|g| g.ranks.iter().map(|r| r.device.0).collect())
            .collect();
        assert_eq!(devices, [[3, 4], [3, 4]]);
        let e = plan_execution_device(&host_tp, DeviceId(0), Some(Vendor::Amd), "h")
            .expect_err("a GPU never takes the host path");
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

    /// A GPU's upstream PCIe edge carrying a measured host link.
    fn upstream(gpu: u32, gbps: f64) -> Edge {
        Edge {
            a: format!("pcie:0000:00:01.{gpu}"),
            b: format!("gpu{gpu}"),
            path: None,
            measured_h2d_gbps: Some(gbps),
            measured_d2h_gbps: Some(gbps),
            ..link(0, 0, EdgeKind::Pcie, PathClass::Sys)
        }
    }

    fn pp_cfg(pp: u32, tp: u32, ep: u32, dp: u32) -> ParallelConfig {
        ParallelConfig {
            pipeline_parallel_size: SizeOrAuto::Size(pp),
            ..ep_cfg(tp, ep, dp)
        }
    }

    /// P5 S-12 (plan Task 22): the parallel modes as plan choices on novanas's two GPUs. Refused
    /// with `combination_unsupported:<modes>` naming `parallel.pipeline_parallel_size`: pp 2
    /// with tp 2, pp 2 with ep 2 (on OLMoE), pp 2 in `static` ranks; ep 2 on the dense Llama is
    /// `ep_moe_only` and ep 3 on OLMoE does not divide its experts, both naming
    /// `parallel.expert_parallel_size`. Accepted with a reason code per choice: tp 2
    /// (`grouped_by_link:sys`), dp 2 (`vendor_homogeneous`), ep 2 at tp 1 and tp 2
    /// (`ep_moe_only`) and pp 2 (`configured`, `pp_partition_cost_balanced`, the last stage on
    /// the faster measured host link `pp_stage_host_traffic:0`, backend `hostmem`, its ranks in
    /// stage order, the head's stage with fewer layers); an explicit split is
    /// `pp_partition_explicit`, unmeasured links `pp_stage_device_order`, and pp × dp on four
    /// GPUs builds two 2-stage groups. Breaks if an unsupported combination plans or a PP plan
    /// loses its stages or reasons.
    #[test]
    fn parallel_modes() {
        let novanas = inventory(vec![
            gpu(0, Vendor::Amd, "gfx1201"),
            gpu(1, Vendor::Amd, "gfx1201"),
        ]);
        let measured = graph(vec![
            link(0, 1, EdgeKind::Pcie, PathClass::Sys),
            upstream(0, 13.1),
            upstream(1, 12.5),
        ]);
        let run = |c: &ParallelConfig, m: &ModelShape| plan(&novanas, &measured, c, m, &budget_32g);
        let refused = |c: &ParallelConfig, m: &ModelShape, key: &str, code: &str| {
            let e = run(c, m).expect_err(code);
            assert_eq!(e.key, key, "{code}: {e}");
            assert!(e.reason.starts_with(code), "{code}: {e}");
        };
        refused(
            &pp_cfg(2, 2, 1, 1),
            &llama_3b(),
            PP_KEY,
            "combination_unsupported:pp+tp",
        );
        refused(
            &pp_cfg(2, 1, 2, 1),
            &olmoe(),
            PP_KEY,
            "combination_unsupported:pp+ep",
        );
        let mut stat = pp_cfg(2, 1, 1, 1);
        stat.ranks.mode = RankMode::Static;
        refused(
            &stat,
            &llama_3b(),
            PP_KEY,
            "combination_unsupported:pp+static",
        );
        refused(&pp_cfg(1, 1, 2, 1), &llama_3b(), EP_KEY, "ep_moe_only");
        let e = run(&pp_cfg(1, 1, 3, 1), &olmoe()).expect_err("ep 3");
        assert_eq!(e.key, EP_KEY);
        assert!(e.reason.contains("does not divide"), "{e}");

        let accepted = |c: &ParallelConfig, m: &ModelShape, what: &str| {
            let p = run(c, m).unwrap_or_else(|e| panic!("{what}: {e}"));
            println!("parallel_modes {what}: {:?}", codes(&p));
            p
        };
        let p = accepted(&pp_cfg(1, 2, 1, 1), &llama_3b(), "tp 2");
        assert_eq!((p.tp, p.pp, p.replica_size()), (2, 1, 2));
        assert!(codes(&p).contains(&"grouped_by_link:sys".into()));
        let p = accepted(&pp_cfg(1, 1, 1, 2), &llama_3b(), "dp 2");
        assert_eq!((p.dp, devices_of(&p)), (2, vec![vec![0], vec![1]]));
        assert!(codes(&p).contains(&"vendor_homogeneous".into()));
        for tp in [1, 2] {
            let p = accepted(&pp_cfg(1, tp, 2, 1), &olmoe(), "ep 2");
            assert_eq!((p.ep, p.tp), (2, tp));
            assert!(codes(&p).contains(&"ep_moe_only".into()));
        }

        let p = accepted(&pp_cfg(2, 1, 1, 1), &llama_3b(), "pp 2");
        assert_eq!((p.pp, p.tp, p.ep, p.dp, p.replica_size()), (2, 1, 1, 1, 2));
        assert_eq!(p.backend, "hostmem");
        for code in [
            "configured",
            "pp_partition_cost_balanced",
            "pp_stage_host_traffic:0",
            "backend:hostmem",
        ] {
            assert!(codes(&p).contains(&code.to_string()), "{:?}", codes(&p));
        }
        assert_eq!(p.stages.len(), 2);
        assert_eq!(
            (p.stages[0].device, p.stages[1].device),
            (DeviceId(1), DeviceId(0))
        );
        assert_eq!(p.stages[0].layers.start, 0);
        assert_eq!(p.stages[0].layers.end, p.stages[1].layers.start);
        assert_eq!(p.stages[1].layers.end, 28);
        assert!(p.stages[1].layers.len() < p.stages[0].layers.len());
        assert!(p.stages[0].embedding && p.stages[1].lm_head);
        // The group's ranks are its stages, in stage order.
        assert_eq!(devices_of(&p), vec![vec![1, 0]]);
        assert_eq!(p.groups[0].ranks[1].rank, 1);

        let mut split = pp_cfg(2, 1, 1, 1);
        split.pipeline.layer_split = Some(vec![14, 14]);
        let p = plan(
            &novanas,
            &graph(Vec::new()),
            &split,
            &llama_3b(),
            &budget_32g,
        )
        .expect("explicit split");
        assert_eq!(
            p.stages
                .iter()
                .map(|s| s.layers.clone())
                .collect::<Vec<_>>(),
            vec![0..14, 14..28]
        );
        for code in ["pp_partition_explicit", "pp_stage_device_order"] {
            assert!(codes(&p).contains(&code.to_string()), "{:?}", codes(&p));
        }
        assert_eq!(devices_of(&p), vec![vec![0, 1]]);
        split.pipeline.layer_split = Some(vec![14, 13]);
        let e = run(&split, &llama_3b()).expect_err("[14, 13]");
        assert_eq!(e.key, "parallel.pipeline.layer_split");

        let four = inventory((0..4).map(|i| gpu(i, Vendor::Amd, "gfx1201")).collect());
        let p = plan(
            &four,
            &graph(Vec::new()),
            &pp_cfg(2, 1, 1, 2),
            &olmoe(),
            &budget_32g,
        )
        .expect("pp 2 × dp 2");
        assert_eq!(devices_of(&p), vec![vec![0, 1], vec![2, 3]]);

        // The cpu host path: two stage threads on host devices 0 and 1, placed by the model.
        use turbine_core::config::ModuleName;
        let host = ParallelConfig {
            collective_backend: ModuleName::new("host").unwrap(),
            ..pp_cfg(2, 1, 1, 1)
        };
        let mut p = plan_execution_device(&host, DeviceId(0), None, "h").expect("host pp 2");
        assert_eq!((p.pp, p.replica_size()), (2, 2));
        plan_stages(&mut p, &host, &llama_3b(), &|_| None).expect("stages");
        assert_eq!(devices_of(&p), vec![vec![0, 1]]);
        assert_eq!(
            codes(&p),
            vec![
                "execution_device",
                "configured",
                "backend:host",
                "pp_partition_cost_balanced",
                "pp_stage_device_order"
            ]
        );
        let e = plan_execution_device(&pp_cfg(2, 1, 1, 1), DeviceId(0), None, "h")
            .expect_err("cpu pp without the host backend");
        assert_eq!(e.key, PP_KEY);
        let e = plan_execution_device(
            &ParallelConfig {
                collective_backend: ModuleName::new("host").unwrap(),
                ..pp_cfg(2, 2, 1, 1)
            },
            DeviceId(0),
            None,
            "h",
        )
        .expect_err("pp 2 tp 2");
        assert!(e.reason.starts_with("combination_unsupported:pp+tp"), "{e}");
    }

    /// `pipeline_parallel_size: auto` is 1 while the model fits one device, else the smallest
    /// pp that fits (`pp_required_for_capacity`). Breaks if auto splits a fitting model.
    #[test]
    fn pipeline_auto_by_capacity() {
        let novanas = inventory(vec![
            gpu(0, Vendor::Amd, "gfx1201"),
            gpu(1, Vendor::Amd, "gfx1201"),
        ]);
        let g = graph(Vec::new());
        let auto = ParallelConfig {
            pipeline_parallel_size: SizeOrAuto::Auto,
            ..pp_cfg(1, 1, 1, 1)
        };
        let p = plan(&novanas, &g, &auto, &shape(24, 8), &budget_32g).expect("fits");
        assert_eq!(p.pp, 1);
        let big = ModelShape {
            weight_bytes: 40 * GIB,
            ..shape(24, 8)
        };
        let p = plan(&novanas, &g, &auto, &big, &budget_32g).expect("pp for capacity");
        assert_eq!(p.pp, 2);
        assert!(codes(&p).contains(&"pp_required_for_capacity".into()));
    }
}
