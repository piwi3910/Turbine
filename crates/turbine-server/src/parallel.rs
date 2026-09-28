//! Startup step 3 (P5 S-4, contract §16.3): the parallel plan, computed after device discovery
//! and the topology capture and before the kernel provider loads or the port binds, so an
//! impossible plan exits 2 with the key to change.
//!
//! - `execution.backend: cpu`: every replica runs on `execution.device` on the host
//!   ([`plan_execution_device`], no vendor); tensor parallelism is refused.
//! - GPU backends: `ParallelConfig::validate_devices` against the inventory, then either the
//!   single-GPU default (tp 1, dp 1, `devices: auto` → `execution.device`, as before Phase 5) or
//!   the topology-driven planner, which needs the model's head counts (read from `config.json`
//!   here, before the kernel provider) and a per-device budget for `tp: auto`.
//!
//! The plan is published as `turbine_parallel_info{tp,dp,backend,mode} 1` and as the `parallel`
//! object of `GET /turbine/v1/status`.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};
use turbine_core::config::{Config, DeviceSelection, ExpertPlacementChoice, RankMode, SizeOrAuto};
use turbine_core::types::{DeviceId, Vendor};
use turbine_device::DeviceInventory;
use turbine_device::topology::TopologyGraph;
use turbine_distributed::expert::{EP_KEY, ExpertPlacement};
use turbine_distributed::plan::{
    ParallelPlan, expert_placement, plan, plan_execution_device, plan_stages,
};

use turbine_distributed::collective::CollectiveLibrary;
use turbine_model::ep::{self, EpAttention, EpShard, ExpertCountsSnapshot, ExpertTokenCounts};
use turbine_model::load_model_config;
use turbine_model::tp::ShardSpec;
use turbine_observability::MetricsRegistry;

/// Why startup cannot use the configured plan.
#[derive(Debug, PartialEq, Eq)]
pub enum PlanFailure {
    /// Impossible or unsupported plan: exit 2, before bind.
    Config(String),
    /// The planner needs the model's shape and `config.json` cannot be read: exit 1.
    Startup(String),
}

/// The device vendor of the registered execution backend `name` (`None` for the host `cpu`
/// backend or an unknown name, which `Config::validate_modules` already refused).
fn backend_vendor(name: &str) -> Option<Vendor> {
    let vendor = turbine_kernels::backends::registry().get(name)?.vendor();
    [Vendor::Amd, Vendor::Nvidia]
        .into_iter()
        .find(|v| v.as_str() == vendor)
}

/// What the configured collective backend serves, for `ParallelConfig::validate_devices`.
fn collective_vendors(name: &str) -> Option<&'static [Vendor]> {
    turbine_distributed::collective::registry()
        .get(name)
        .map(|b| b.vendors())
}

/// Computes the plan (module comment).
pub fn plan_for(
    config: &Config,
    inventory: &DeviceInventory,
    topology: &TopologyGraph,
) -> Result<ParallelPlan, PlanFailure> {
    let p = &config.parallel;
    let exec = &config.execution;
    let host = topology.node.hostname.as_str();
    let plan_error = |e: turbine_distributed::plan::PlanError| PlanFailure::Config(e.to_string());
    let Some(vendor) = backend_vendor(exec.backend.as_str()) else {
        let mut plan = plan_execution_device(p, exec.device, None, host).map_err(plan_error)?;
        if plan.pp > 1 {
            // The cpu host path's stages (P5 S-10): split by the model's costs, device order.
            let shape = load_model_config(&config.model.path)
                .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?
                .shape();
            plan_stages(&mut plan, p, &shape, &|_| None).map_err(plan_error)?;
        }
        return Ok(plan);
    };
    let inv: Vec<(DeviceId, Vendor, Option<String>)> = inventory
        .devices
        .iter()
        .map(|d| (d.index, d.vendor, d.arch.clone()))
        .collect();
    p.validate_devices(&inv, collective_vendors(p.collective_backend.as_str()))
        .map_err(|e| PlanFailure::Config(e.to_string()))?;
    let single_default = p.tensor_parallel_size == SizeOrAuto::Size(1)
        && p.data_parallel_size == SizeOrAuto::Size(1)
        && p.expert_parallel_size.fixed().unwrap_or(1) == 1
        && p.pipeline_parallel_size == SizeOrAuto::Size(1)
        && p.devices == DeviceSelection::Auto;
    if single_default {
        return plan_execution_device(p, exec.device, Some(vendor), host).map_err(plan_error);
    }
    let shape = load_model_config(&config.model.path)
        .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?
        .shape();
    // Phase 3's device budget replaces this: total memory less the emergency reserve.
    let reserve = config.reliability.emergency_vram_reserve.0;
    let budget = |id: DeviceId| {
        inventory
            .devices
            .iter()
            .find(|d| d.index == id)
            .map_or(0, |d| d.memory.total_bytes.saturating_sub(reserve))
    };
    plan(inventory, topology, p, &shape, &budget).map_err(plan_error)
}

/// Refuses a plan the model cannot run (exit 2, like any other unusable configuration, before
/// anything is loaded or bound): with tensor parallelism the model's family must have
/// tensor-parallel hooks and the group size must split its attention heads, KV heads (or be a
/// multiple of them), intermediate and expert widths (`turbine_model::tp::check`, reason code
/// in the message). With expert parallelism (P5 S-11) the model must have routed experts
/// (`ep_moe_only`), and the placement (`parallel.expert.placement`, over the family's MoE
/// layers) must fit every rank (`turbine_model::ep::check`); it becomes `plan.experts`. With
/// pipeline stages (P5 S-10) the model's family must run as stages and the planned layer
/// ranges must cover its layers (`turbine_model::pp::check`).
/// `config.json` is read here; an unreadable one is exit 1.
pub fn check_executable(plan: &mut ParallelPlan, config: &Config) -> Result<(), PlanFailure> {
    if plan.ep > 1 {
        return check_expert_parallel(plan, config);
    }
    if plan.pp > 1 {
        let arch = load_model_config(&config.model.path)
            .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?;
        return turbine_model::pp::check(&arch, &plan.stages).map_err(|e| {
            tracing::error!(
                event = "parallel_plan_failed",
                reason = "pp_unsplittable_model",
                pp = plan.pp,
                error = %e,
                "the model cannot run as the planned pipeline stages"
            );
            PlanFailure::Config(format!(
                "parallel.pipeline_parallel_size: {} stages cannot run {}: {e}",
                plan.pp, arch.hf_architecture
            ))
        });
    }
    if plan.tp <= 1 {
        return Ok(());
    }
    if plan.mode == RankMode::Static && plan.vendor.is_none() {
        // The host collective's ranks are threads of one process; static ranks are processes.
        return Err(PlanFailure::Config(
            "parallel.ranks.mode: static rank processes need a device collective backend; the \
             cpu backend's host collective runs its ranks as threads (use local)"
                .into(),
        ));
    }
    let arch = load_model_config(&config.model.path)
        .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?;
    turbine_model::tp::check(
        &arch,
        ShardSpec {
            rank: 0,
            world: plan.tp,
        },
    )
    .map_err(|e| {
        tracing::error!(
            event = "parallel_plan_failed",
            reason = "tp_unsplittable_model",
            tp = plan.tp,
            error = %e,
            "the model cannot be split over the tensor-parallel group"
        );
        PlanFailure::Config(format!(
            "parallel.tensor_parallel_size: {} ranks cannot split {}: {e}",
            plan.tp, arch.hf_architecture
        ))
    })
}

/// [`check_executable`] of an expert-parallel plan.
fn check_expert_parallel(plan: &mut ParallelPlan, config: &Config) -> Result<(), PlanFailure> {
    let arch = load_model_config(&config.model.path)
        .map_err(|e| PlanFailure::Startup(format!("model config: {e}")))?;
    let refuse = |reason: &str, message: String| {
        tracing::error!(
            event = "parallel_plan_failed",
            reason,
            ep = plan.ep,
            tp = plan.tp,
            error = %message,
            "the model cannot be split over the expert-parallel group"
        );
        PlanFailure::Config(message)
    };
    let Some(moe) = arch.moe else {
        return Err(refuse(
            ep::EP_MOE_ONLY,
            format!(
                "{EP_KEY}: ep_moe_only: {} expert-parallel ranks need a mixture-of-experts \
                 model; {} has no routed experts",
                plan.ep, arch.hf_architecture
            ),
        ));
    };
    // Combinations the support matrix refuses for this architecture (P5 exit: OLMoE with
    // expert × tensor parallelism, `olmoe_ep_tp_drift`).
    if let Some(r) =
        turbine_core::support::parallel_refusal(&arch.hf_architecture, plan.tp, plan.ep)
    {
        let code = r.reason.split(':').next().unwrap_or(r.reason);
        return Err(refuse(
            code,
            format!(
                "{EP_KEY}: {} (expert_parallel_size {} with tensor_parallel_size {} on {})",
                r.reason, plan.ep, plan.tp, arch.hf_architecture
            ),
        ));
    }
    let placement = expert_placement(
        &config.parallel,
        moe.num_experts,
        &ep::moe_layers(&arch),
        plan.ep,
    )
    .map_err(|e| refuse("ep_placement", e.to_string()))?
    .ok_or_else(|| PlanFailure::Config(format!("{EP_KEY}: no placement for ep {}", plan.ep)))?;
    let attention = if plan.tp == plan.ep {
        EpAttention::TensorParallel
    } else {
        EpAttention::Replicated
    };
    for rank in 0..plan.ep {
        let s = EpShard {
            rank,
            world: plan.ep,
            attention,
        };
        ep::check(&arch, s, &placement).map_err(|e| {
            refuse(
                "ep_unsplittable_model",
                format!(
                    "{EP_KEY}: {} ranks cannot split {}: {e}",
                    plan.ep, arch.hf_architecture
                ),
            )
        })?;
    }
    plan.experts = Some(placement);
    Ok(())
}

/// Loads the plan's collective backend (`parallel.collective_backend`, `auto` resolved by the
/// planner; registry-driven, so a new backend needs no change here): its configured library
/// (`parallel.rccl_library`, …) or the default search. A failure names the backend and the
/// library (exit 1: tensor parallelism cannot run without it).
pub fn load_collective(
    config: &Config,
    plan: &ParallelPlan,
) -> Result<Arc<dyn CollectiveLibrary>, String> {
    let backend = turbine_distributed::collective::registry()
        .get(plan.backend)
        .ok_or_else(|| {
            format!(
                "parallel.collective_backend: `{}` is not registered",
                plan.backend
            )
        })?;
    // P5 Task 32: the hostmem copy-engine threshold, read when its library loads.
    turbine_distributed::collective::hostmem::set_dma_min_bytes(
        config
            .parallel
            .collective
            .hostmem_dma_min_bytes
            .map(|b| b.0),
    );
    let library = backend
        .load(backend.configured_library(&config.parallel))
        .map_err(|e| format!("collective backend `{}`: {e}", plan.backend))?;
    tracing::info!(
        event = "collective_backend_loaded",
        backend = library.backend(),
        version = library.version().as_deref().unwrap_or("none"),
        "collective backend loaded"
    );
    Ok(library)
}

/// The configuration fingerprint of the static-mode `Hello` (P5 S-5): BLAKE3 of the resolved
/// configuration with what legitimately differs per rank process cleared — the rank, its
/// local devices, its listen address and its execution device.
pub fn config_fingerprint(config: &Config) -> [u8; 32] {
    let mut c = config.clone();
    c.parallel.ranks.rank = 0;
    c.parallel.ranks.local_devices = Vec::new();
    c.server.listen = std::net::SocketAddr::from(([0, 0, 0, 0], 0));
    c.execution.device = DeviceId(0);
    let bytes = serde_json::to_vec(&c).unwrap_or_default();
    *blake3::hash(&bytes).as_bytes()
}

/// The device of the single engine: rank 0 of replica 0 — with pipeline stages the last stage
/// (the engine thread runs it and owns its KV pool, P5 S-10).
pub fn engine_device(plan: &ParallelPlan) -> DeviceId {
    leader_device(plan, &plan.groups[0])
}

/// The device of `group`'s engine: its rank 0, or its last stage under pipeline parallelism.
pub fn leader_device(
    plan: &ParallelPlan,
    group: &turbine_distributed::plan::ReplicaGroup,
) -> DeviceId {
    let rank = if plan.pp > 1 {
        group.ranks.last()
    } else {
        group.ranks.first()
    };
    rank.map_or(DeviceId(0), |r| r.device)
}

/// The startup host-link probe (P5 S-13): with `parallel.topology.measure_links` on a GPU
/// backend whose configuration spans more than the single-GPU default, every GPU of the
/// topology graph gets one bounded host↔device copy each way through a context of the
/// configured backend (its pinned memory and copy stream, kernel ABI v2.3 + v2.5), and the
/// graph's upstream PCIe edges carry the measured bandwidth; the planner places pipeline
/// stages by it. Bounded by `turbine_device::topology::PROBE_BUDGET`; a GPU whose context has no
/// copy stream (or fails to open) keeps its nominal edge with a WARN. The one-GPU default and
/// the cpu backend skip it (nothing reads the links).
pub fn measure_host_links(
    config: &Config,
    inventory: &DeviceInventory,
    topology: &mut TopologyGraph,
) {
    let p = &config.parallel;
    let exec = &config.execution;
    if !p.topology.measure_links || backend_vendor(exec.backend.as_str()).is_none() {
        return;
    }
    let single_default = p.tensor_parallel_size == SizeOrAuto::Size(1)
        && p.data_parallel_size == SizeOrAuto::Size(1)
        && p.expert_parallel_size.fixed().unwrap_or(1) == 1
        && p.pipeline_parallel_size == SizeOrAuto::Size(1)
        && p.devices == DeviceSelection::Auto;
    if single_default {
        return;
    }
    let Some(backend) = turbine_kernels::backends::registry().get(exec.backend.as_str()) else {
        return;
    };
    let started = Instant::now();
    let mut probe = turbine_kernels::link_probe::CopyLinkProbe::new(|device| {
        let opened = backend
            .open(&turbine_kernels::backends::BackendRequest {
                device,
                kernel_library: exec.kernel_library.as_deref(),
                inventory,
                meminfo: std::path::Path::new("/proc/meminfo"),
                card_profile: exec.card_profile.as_str(),
            })
            .map_err(|e| e.to_string())?;
        let ctx = opened
            .context
            .filter(|c| c.has_copy_engine())
            .ok_or("the kernel library has no copy streams (ABI v2.5)")?;
        Ok(turbine_kernels::link_probe::ProbeTarget {
            memory: opened.mem,
            copies: ctx.clone(),
            pinned: ctx,
        })
    });
    let results = turbine_device::topology::probe_links(topology, &mut probe);
    tracing::info!(
        event = "topology_link_probe",
        devices = results.len(),
        measured = results.values().filter(|r| r.is_ok()).count(),
        seconds = started.elapsed().as_secs_f64(),
        "host links probed"
    );
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct LinkLabels {
    device: String,
    direction: &'static str,
}

/// `turbine_topology_link_gbps{device,direction}` (P5 S-13): every measured host link of
/// `topology` (`h2d`, `d2h`); nothing when no link was measured.
pub fn register_links(reg: &MetricsRegistry, topology: &TopologyGraph) {
    let measured: Vec<(DeviceId, f64, f64)> = topology
        .gpu_devices()
        .into_iter()
        .filter_map(|d| {
            let id = format!("gpu{}", d.0);
            topology.edges.iter().find_map(|e| {
                (e.b == id && e.a.starts_with("pcie:"))
                    .then(|| e.measured_h2d_gbps.zip(e.measured_d2h_gbps))
                    .flatten()
                    .map(|(h, dd)| (d, h, dd))
            })
        })
        .collect();
    if measured.is_empty() {
        return;
    }
    let gauge = reg.register(
        "turbine_topology_link_gbps",
        "Measured host-link bandwidth of a GPU, GB/s",
        Family::<LinkLabels, Gauge<f64, AtomicU64>>::default(),
    );
    for (d, h2d, d2h) in measured {
        for (direction, v) in [("h2d", h2d), ("d2h", d2h)] {
            gauge
                .get_or_create(&LinkLabels {
                    device: d.0.to_string(),
                    direction,
                })
                .set(v);
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ParallelInfoLabels {
    tp: String,
    dp: String,
    backend: &'static str,
    mode: String,
}

/// `turbine_parallel_info{tp,dp,backend,mode}`, always 1.
pub fn register_info(reg: &MetricsRegistry, plan: &ParallelPlan) {
    let info = reg.register(
        "turbine_parallel_info",
        "The process's parallel plan (always 1)",
        Family::<ParallelInfoLabels, Gauge>::default(),
    );
    info.get_or_create(&ParallelInfoLabels {
        tp: plan.tp.to_string(),
        dp: plan.dp.to_string(),
        backend: plan.backend,
        mode: mode_str(plan),
    })
    .set(1);
}

/// `parallel.ranks.mode` as the configuration spells it (`local`, `static`).
fn mode_str(plan: &ParallelPlan) -> String {
    match serde_json::to_value(plan.mode) {
        Ok(Value::String(s)) => s,
        _ => format!("{:?}", plan.mode).to_lowercase(),
    }
}

/// The `parallel` object of `GET /turbine/v1/status`. With ep > 1 (P5 §Data) it also carries
/// `"pp"` and `"ep"`, and every group its `experts`: per rank `[first, last]` when the rank holds
/// one run of expert ids in every MoE layer (a contiguous placement), else its `layer_experts`
/// (`{layer: [expert ids]}`).
pub fn status(plan: &ParallelPlan) -> Value {
    let mut doc = json!({
        "tp": plan.tp,
        "dp": plan.dp,
        "backend": plan.backend,
        "mode": mode_str(plan),
        "groups": plan.groups.iter().map(|g| json!({
            "replica": g.replica.0,
            "ranks": g.ranks.iter().map(|r| json!({
                "rank": r.rank,
                "device": r.device.0,
                "host": r.host,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "plan_reasons": plan.reasons.iter().map(ToString::to_string).collect::<Vec<_>>(),
    });
    if plan.pp > 1 {
        doc["pp"] = json!(plan.pp);
        doc["ep"] = json!(plan.ep);
        let codes: Vec<String> = plan.reasons.iter().map(ToString::to_string).collect();
        if let Some(groups) = doc["groups"].as_array_mut() {
            for (g, doc) in plan.groups.iter().zip(groups) {
                doc["stages"] = stages_status(plan, g, &codes);
            }
        }
    }
    if plan.ep > 1 {
        doc["pp"] = json!(plan.pp);
        doc["ep"] = json!(plan.ep);
        if let (Some(placement), Some(groups)) = (&plan.experts, doc["groups"].as_array_mut()) {
            let experts = placement_status(placement);
            for g in groups {
                g["experts"] = experts.clone();
            }
        }
    }
    doc
}

/// Group `g`'s pipeline stages (see [`status`]): per stage its device, `layers: [first, last]`
/// and the placement reason of the group (`pp_stage_host_traffic:<device>` when one of its
/// devices got the host-traffic placement, else `pp_stage_device_order`).
fn stages_status(
    plan: &ParallelPlan,
    g: &turbine_distributed::plan::ReplicaGroup,
    codes: &[String],
) -> Value {
    let reason = g
        .ranks
        .iter()
        .map(|r| format!("pp_stage_host_traffic:{}", r.device.0))
        .find(|c| codes.contains(c))
        .unwrap_or_else(|| "pp_stage_device_order".to_string());
    Value::Array(
        plan.stages
            .iter()
            .zip(&g.ranks)
            .map(|(s, r)| {
                json!({
                    "stage": s.stage,
                    "device": r.device.0,
                    "layers": [s.layers.start, s.layers.end.saturating_sub(1)],
                    "reason": reason,
                })
            })
            .collect(),
    )
}

/// Every rank's experts (see [`status`]).
fn placement_status(p: &ExpertPlacement) -> Value {
    Value::Array(
        (0..p.ranks)
            .map(|rank| {
                let per_layer: Vec<(u32, Vec<u32>)> = p
                    .layers
                    .iter()
                    .map(|(l, _)| (*l, p.local_experts(*l, rank)))
                    .collect();
                let first = per_layer
                    .first()
                    .map(|(_, e)| e.clone())
                    .unwrap_or_default();
                let one_run = !first.is_empty()
                    && first.windows(2).all(|w| w[1] == w[0] + 1)
                    && per_layer.iter().all(|(_, e)| *e == first);
                if one_run {
                    json!({"rank": rank, "experts": [first[0], first[first.len() - 1]]})
                } else {
                    let layers: serde_json::Map<String, Value> = per_layer
                        .into_iter()
                        .map(|(l, e)| (l.to_string(), json!(e)))
                        .collect();
                    json!({"rank": rank, "layer_experts": layers})
                }
            })
            .collect(),
    )
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RankLabel {
    rank: String,
}

/// `turbine_expert_rank_tokens_total{rank}` and `turbine_expert_imbalance_ratio` (P5 S-11),
/// registered once; every data-parallel replica's EP group adds to the same series.
#[derive(Clone)]
pub struct ExpertMetrics {
    rank_tokens: Family<RankLabel, Counter>,
    imbalance: Gauge<f64, AtomicU64>,
}

impl ExpertMetrics {
    pub fn register(reg: &MetricsRegistry) -> ExpertMetrics {
        ExpertMetrics {
            rank_tokens: reg.register(
                "turbine_expert_rank_tokens",
                "Token-expert assignments computed by each expert-parallel rank",
                Family::<RankLabel, Counter>::default(),
            ),
            imbalance: reg.register(
                "turbine_expert_imbalance_ratio",
                "Largest over mean token-expert assignments per expert-parallel rank, last 10 s",
                Gauge::<f64, AtomicU64>::default(),
            ),
        }
    }
}

/// The window of `turbine_expert_imbalance_ratio` (spec §Metrics).
const IMBALANCE_WINDOW: Duration = Duration::from_secs(10);

/// One expert-parallel group's token counts as the server reports them: rank 0's
/// [`ExpertTokenCounts`] (every rank counts the same; the router is replicated), turned into the
/// metrics after every step ([`ExpertStats::observe`]) and into the `expert` section of the
/// replica's `/turbine/v1/scheduler` document ([`ExpertStats::document`]).
pub struct ExpertStats {
    counts: Arc<ExpertTokenCounts>,
    parallel_size: u32,
    /// `contiguous` or `file` (`parallel.expert.placement`).
    placement: &'static str,
    metrics: ExpertMetrics,
    state: Mutex<StatsWindow>,
}

#[derive(Default)]
struct StatsWindow {
    /// Per-rank totals already added to the counters.
    reported: Vec<u64>,
    /// `(when, per-rank totals)` samples of the last window, oldest first.
    samples: VecDeque<(Instant, Vec<u64>)>,
    imbalance: f64,
}

impl ExpertStats {
    pub fn new(
        parallel_size: u32,
        placement: &ExpertPlacementChoice,
        metrics: ExpertMetrics,
    ) -> ExpertStats {
        ExpertStats {
            counts: Arc::default(),
            parallel_size,
            placement: match placement {
                ExpertPlacementChoice::Contiguous => "contiguous",
                ExpertPlacementChoice::File(_) => "file",
            },
            metrics,
            state: Mutex::new(StatsWindow::default()),
        }
    }

    /// The counts rank 0's executor records into.
    pub fn counts(&self) -> &Arc<ExpertTokenCounts> {
        &self.counts
    }

    /// After a step: adds the new assignments of every rank to
    /// `turbine_expert_rank_tokens_total` and sets `turbine_expert_imbalance_ratio` (max / mean
    /// of the per-rank assignments over the last 10 s; unchanged while the window is empty).
    pub fn observe(&self) {
        self.observe_at(&self.counts.snapshot(), Instant::now());
    }

    fn observe_at(&self, snapshot: &ExpertCountsSnapshot, now: Instant) {
        let per_rank = &snapshot.per_rank;
        if per_rank.is_empty() {
            return;
        }
        let mut w = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if w.samples.is_empty() {
            // Nothing was counted before the first observation.
            w.samples.push_back((now, vec![0; per_rank.len()]));
        }
        w.reported.resize(per_rank.len(), 0);
        for (rank, (&total, reported)) in per_rank.iter().zip(w.reported.iter_mut()).enumerate() {
            if total > *reported {
                self.metrics
                    .rank_tokens
                    .get_or_create(&RankLabel {
                        rank: rank.to_string(),
                    })
                    .inc_by(total - *reported);
                *reported = total;
            }
        }
        w.samples.push_back((now, per_rank.clone()));
        // Keep the newest sample at or before the window's start as its baseline.
        while w.samples.len() > 2 && w.samples[1].0 + IMBALANCE_WINDOW <= now {
            w.samples.pop_front();
        }
        let base = &w.samples[0].1;
        let delta: Vec<u64> = per_rank
            .iter()
            .zip(base.iter().chain(std::iter::repeat(&0)))
            .map(|(a, b)| a.saturating_sub(*b))
            .collect();
        let sum: u64 = delta.iter().sum();
        if sum > 0 {
            let mean = sum as f64 / delta.len() as f64;
            let max = delta.iter().copied().max().unwrap_or(0) as f64;
            w.imbalance = max / mean;
            self.metrics.imbalance.set(w.imbalance);
        }
    }

    /// The `expert` section of `/turbine/v1/scheduler` (P5 §Data): `parallel_size`,
    /// `placement`, `tokens_per_rank`, the 10 largest `top_experts` (`{layer, expert,
    /// tokens}`), plus `imbalance_ratio` and the `steps` counted.
    pub fn document(&self) -> Value {
        self.document_of(&self.counts.snapshot())
    }

    fn document_of(&self, s: &ExpertCountsSnapshot) -> Value {
        let imbalance = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .imbalance;
        let tokens_per_rank = if s.per_rank.is_empty() {
            vec![0; self.parallel_size as usize]
        } else {
            s.per_rank.clone()
        };
        json!({
            "parallel_size": self.parallel_size,
            "placement": self.placement,
            "tokens_per_rank": tokens_per_rank,
            "top_experts": s.top_experts(10).into_iter().map(|(layer, expert, tokens)| json!({
                "layer": layer, "expert": expert, "tokens": tokens,
            })).collect::<Vec<_>>(),
            "imbalance_ratio": imbalance,
            "steps": s.steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use turbine_core::types::ReplicaId;
    use turbine_distributed::plan::{PlanReason, RankSlot, ReplicaGroup};

    use super::*;

    fn plan(tp: u32, dp: u32) -> ParallelPlan {
        ParallelPlan {
            tp,
            dp,
            pp: 1,
            ep: 1,
            stages: Vec::new(),
            backend: "host",
            mode: turbine_core::config::RankMode::Local,
            vendor: Some(Vendor::Amd),
            excluded_devices: Vec::new(),
            groups: vec![ReplicaGroup {
                replica: ReplicaId(0),
                ranks: vec![RankSlot {
                    rank: 0,
                    device: DeviceId(1),
                    host: "novanas".into(),
                }],
            }],
            reasons: vec![PlanReason::ExecutionDevice],
            experts: None,
        }
    }

    /// The status object has the contract's keys, and the info gauge renders its labels.
    #[test]
    fn status_and_info_gauge() {
        let p = plan(1, 1);
        assert_eq!(
            status(&p),
            json!({"tp": 1, "dp": 1, "backend": "host", "mode": "local",
                   "groups": [{"replica": 0, "ranks": [{"rank": 0, "device": 1, "host": "novanas"}]}],
                   "plan_reasons": ["execution_device"]})
        );
        assert_eq!(engine_device(&p), DeviceId(1));
        let reg = MetricsRegistry::new();
        register_info(&reg, &p);
        let text = reg.render().unwrap();
        assert!(
            text.contains(r#"turbine_parallel_info{tp="1",dp="1",backend="host",mode="local"} 1"#),
            "{text}"
        );
        // Without tensor parallelism nothing is read (the model path is not even looked at).
        let config = Config::default();
        assert!(check_executable(&mut p.clone(), &config).is_ok());
        assert!(check_executable(&mut plan(1, 2).clone(), &config).is_ok());
    }

    /// Tensor parallelism is checked against the model (P5 S-6): the tiny Llama (4 heads, 2 KV
    /// heads, intermediate 128) splits over 2 and 4 ranks, not over 8 (more ranks than heads),
    /// which is refused naming `parallel.tensor_parallel_size`; an unreadable `config.json` is
    /// a startup failure (exit 1), not a configuration error.
    #[test]
    fn tensor_parallel_checked_against_the_model() {
        let dir = turbine_model::testing::TempDir::new("parallel-tp-check");
        turbine_model::testing::tiny::write_tiny_llama(dir.path(), 1);
        let mut config = Config::default();
        config.model.path = dir.path().to_path_buf();
        for tp in [2, 4] {
            assert!(
                check_executable(&mut plan(tp, 1).clone(), &config).is_ok(),
                "tp {tp}"
            );
        }
        match check_executable(&mut plan(8, 1).clone(), &config) {
            Err(PlanFailure::Config(m)) => {
                assert!(m.starts_with("parallel.tensor_parallel_size: 8"), "{m}");
                assert!(m.contains("4 attention heads"), "{m}");
            }
            other => panic!("tp 8: {other:?}"),
        }
        config.model.path = dir.path().join("missing");
        assert!(matches!(
            check_executable(&mut plan(2, 1).clone(), &config),
            Err(PlanFailure::Startup(_))
        ));
    }

    /// P5 S-10, S-12 on the cpu host path: `pipeline_parallel_size: 2` over the host collective
    /// plans the tiny Llama's 2 layers as 2 stages on host devices 0 and 1 (checked against the
    /// model), the engine runs the last stage (device 1), and the status object carries `pp`,
    /// `ep` and every group's `stages` with their layers and placement reason; a split that
    /// does not cover the model is exit 2 naming `parallel.pipeline.layer_split`. Breaks if the
    /// stages or their status go missing, or the engine lands on another stage.
    #[test]
    fn pipeline_planned_checked_and_shown() {
        let dir = turbine_model::testing::TempDir::new("parallel-pp-check");
        turbine_model::testing::tiny::write_tiny_llama(dir.path(), 1);
        let mut config = Config::default();
        config.model.path = dir.path().to_path_buf();
        config.execution.backend = turbine_core::config::ModuleName::new("cpu").unwrap();
        config.parallel.pipeline_parallel_size = SizeOrAuto::Size(2);
        config.parallel.collective_backend = turbine_core::config::ModuleName::new("host").unwrap();
        let inventory = DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        let topology = TopologyGraph {
            node: turbine_device::topology::TopologyNode {
                hostname: "h".into(),
                captured_at: String::new(),
            },
            vertices: Vec::new(),
            edges: Vec::new(),
        };
        let mut p = plan_for(&config, &inventory, &topology).expect("pp 2 plan");
        check_executable(&mut p, &config).expect("the tiny Llama splits into 2 stages");
        assert_eq!((p.pp, p.tp, p.replica_size()), (2, 1, 2));
        assert_eq!(
            p.stages
                .iter()
                .map(|s| s.layers.clone())
                .collect::<Vec<_>>(),
            vec![0..1, 1..2]
        );
        assert_eq!(engine_device(&p), DeviceId(1));
        let doc = status(&p);
        assert_eq!((&doc["pp"], &doc["ep"]), (&json!(2), &json!(1)), "{doc}");
        assert_eq!(
            doc["groups"][0]["stages"],
            json!([
                {"stage": 0, "device": 0, "layers": [0, 0], "reason": "pp_stage_device_order"},
                {"stage": 1, "device": 1, "layers": [1, 1], "reason": "pp_stage_device_order"}
            ]),
            "{doc}"
        );
        config.parallel.pipeline.layer_split = Some(vec![2, 1]);
        match plan_for(&config, &inventory, &topology) {
            Err(PlanFailure::Config(m)) => {
                assert!(m.starts_with("parallel.pipeline.layer_split"), "{m}")
            }
            other => panic!("bad split: {other:?}"),
        }
    }

    fn ep_plan(tp: u32, ep: u32) -> ParallelPlan {
        let mut p = plan(tp, 1);
        p.ep = ep;
        p.groups[0].ranks = (0..tp.max(ep))
            .map(|rank| RankSlot {
                rank,
                device: DeviceId(rank),
                host: "h".into(),
            })
            .collect();
        p
    }

    /// Expert parallelism is checked against the model (P5 S-11, S-12): the tiny OLMoE (8
    /// experts, 2 layers) at ep 2 (tp 1 and tp 2) gets the contiguous placement over its MoE
    /// layers, shown per rank in the status object with `pp` and `ep`; a placement file shows
    /// `layer_experts`. The dense tiny Llama at ep 2 is refused with `ep_moe_only` naming
    /// `parallel.expert_parallel_size`, and so is ep 3 (does not divide 8 experts). Breaks if
    /// ep plans on a dense model or the status loses the placement.
    #[test]
    fn expert_parallel_checked_against_the_model() {
        let dir = turbine_model::testing::TempDir::new("parallel-ep-check");
        turbine_model::testing::tiny::write_tiny_olmoe(dir.path(), 1);
        let mut config = Config::default();
        config.model.path = dir.path().to_path_buf();
        // P5 exit (user decision 2026-09-28): OLMoE at ep 2 × tp 2 is refused with
        // `olmoe_ep_tp_drift` naming `parallel.expert_parallel_size`.
        match check_executable(&mut ep_plan(2, 2), &config) {
            Err(PlanFailure::Config(m)) => assert!(
                m.starts_with("parallel.expert_parallel_size: olmoe_ep_tp_drift"),
                "{m}"
            ),
            other => panic!("ep 2 tp 2: {other:?}"),
        }
        {
            let tp = 1;
            let mut p = ep_plan(tp, 2);
            check_executable(&mut p, &config).unwrap_or_else(|e| panic!("tp {tp}: {e:?}"));
            let placement = p.experts.as_ref().expect("placement");
            assert_eq!(placement.layers.len(), 2);
            assert_eq!(placement.local_experts(1, 1), [4, 5, 6, 7]);
            let doc = status(&p);
            assert_eq!((&doc["ep"], &doc["pp"]), (&json!(2), &json!(1)), "{doc}");
            assert_eq!(
                doc["groups"][0]["experts"],
                json!([{"rank": 0, "experts": [0, 3]}, {"rank": 1, "experts": [4, 7]}])
            );
        }
        match check_executable(&mut ep_plan(1, 3), &config) {
            Err(PlanFailure::Config(m)) => assert!(m.contains("does not divide"), "{m}"),
            other => panic!("ep 3: {other:?}"),
        }
        let file = dir.path().join("placement.yaml");
        std::fs::write(
            &file,
            "0: [0, 1, 0, 1, 0, 1, 0, 1]\n1: [0, 0, 0, 0, 1, 1, 1, 1]\n",
        )
        .unwrap();
        config.parallel.expert.placement = ExpertPlacementChoice::File(file);
        let mut p = ep_plan(1, 2);
        check_executable(&mut p, &config).expect("placement file");
        assert_eq!(
            status(&p)["groups"][0]["experts"][0],
            json!({"rank": 0, "layer_experts": {"0": [0, 2, 4, 6], "1": [0, 1, 2, 3]}})
        );

        let dense = turbine_model::testing::TempDir::new("parallel-ep-dense");
        turbine_model::testing::tiny::write_tiny_llama(dense.path(), 1);
        config.model.path = dense.path().to_path_buf();
        match check_executable(&mut ep_plan(1, 2), &config) {
            Err(PlanFailure::Config(m)) => {
                assert!(
                    m.starts_with("parallel.expert_parallel_size: ep_moe_only"),
                    "{m}"
                );
            }
            other => panic!("dense ep 2: {other:?}"),
        }
    }

    /// The expert metrics and the scheduler document's `expert` section follow rank 0's counts:
    /// every step adds the new assignments per rank to the counter (never twice), the
    /// imbalance is max / mean of the last 10 s, and `top_experts` lists the largest (layer,
    /// expert) pairs. Breaks if a rank's tokens are double-counted or the window never slides.
    #[test]
    fn expert_stats_feed_metrics_and_document() {
        let reg = MetricsRegistry::new();
        let stats = ExpertStats::new(
            2,
            &ExpertPlacementChoice::Contiguous,
            ExpertMetrics::register(&reg),
        );
        let doc = stats.document();
        assert_eq!(doc["tokens_per_rank"], json!([0, 0]), "{doc}");
        let t0 = Instant::now();
        // Four experts over two ranks; layer 0: experts 0 and 1 (rank 0) take 3 rows, layer 1:
        // expert 3 (rank 1) takes 1.
        let first = ExpertCountsSnapshot {
            steps: 1,
            per_rank: vec![3, 1],
            per_expert: vec![2, 1, 0, 1],
            per_layer: vec![(0, vec![2, 1, 0, 0]), (1, vec![0, 0, 0, 1])],
        };
        stats.observe_at(&first, t0);
        stats.observe_at(&first, t0);
        let text = reg.render().unwrap();
        assert!(
            text.contains(r#"turbine_expert_rank_tokens_total{rank="0"} 3"#),
            "{text}"
        );
        assert!(
            text.contains(r#"turbine_expert_rank_tokens_total{rank="1"} 1"#),
            "{text}"
        );
        assert!(
            text.contains("turbine_expert_imbalance_ratio 1.5"),
            "{text}"
        );
        // 20 s later only rank 1 computes: the window forgets the first step.
        let second = ExpertCountsSnapshot {
            steps: 2,
            per_rank: vec![3, 5],
            per_expert: vec![2, 1, 2, 3],
            per_layer: vec![(0, vec![2, 1, 0, 0]), (1, vec![0, 0, 2, 3])],
        };
        stats.observe_at(&second, t0 + Duration::from_secs(20));
        stats.observe_at(&second, t0 + Duration::from_secs(21));
        let text = reg.render().unwrap();
        assert!(
            text.contains(r#"turbine_expert_rank_tokens_total{rank="1"} 5"#),
            "{text}"
        );
        assert!(
            text.contains("turbine_expert_imbalance_ratio 2.0"),
            "{text}"
        );
        let doc = stats.document_of(&second);
        assert_eq!(doc["parallel_size"], 2);
        assert_eq!(doc["placement"], "contiguous");
        assert_eq!(doc["tokens_per_rank"], json!([3, 5]));
        assert_eq!(doc["steps"], 2);
        assert_eq!(
            doc["top_experts"][0],
            json!({"layer": 1, "expert": 3, "tokens": 3})
        );
    }
}
