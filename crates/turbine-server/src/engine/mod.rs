//! The Phase 2 engine (P2 S-1, contract §16.4): one dedicated OS thread owns the device context
//! (through the executor and the KV block pool), the scheduler, and every request's sampler and
//! detokenizer. The HTTP side reaches it only through bounded channels: [`EngineHandle`]'s
//! command channel (capacity `scheduler.max_queued_requests`) in, one channel of
//! [`EVENT_CHANNEL_CAPACITY`] events per request out. The engine never waits on a client: a
//! full output channel pauses the request, a closed one cancels it.
//!
//! Diagnostics read the documents the engine publishes after every step ([`EngineShared`]).
//!
//! Phase 3 builds the reliability side on the engine thread once the weights are loaded: the
//! re-measured memory budget, the reservation ledger and the emergency reserve
//! (`model::load`), the pressure controller and the admission gate in front of the scheduler
//! (`crate::reliability::build`), the telemetry sampler thread and the supervised controller
//! thread; in the `fault-injection` build the fault injector on the ledger, the executor and
//! the vendor telemetry.
//!
//! Phase 4 builds the KV hierarchy over the pool right after the weights load
//! (`crate::kv_orchestrator`: tiers and transfer calibration) and before the reliability side,
//! which drives it through the orchestrator's `KvReclaimer` and reads the L2 tier's storage
//! signals through the telemetry sampler.

pub(crate) mod deadlines;
pub(crate) mod grammar;
mod r#loop;
pub(crate) mod pp;
pub(crate) mod requests;
pub(crate) mod stages;
pub(crate) mod tp;
pub(crate) mod tp_tiers;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{mpsc, oneshot};
use turbine_core::clock::{Clock, SystemClock};
use turbine_core::config::{DevicesConfig, KvConfig};
use turbine_core::request::GenerationEvent;
use turbine_device::telemetry::proc::FsProc;
use turbine_device::telemetry::vendor::vendor_backends;
use turbine_device::telemetry::{
    SamplerCore, TelemetryConfig, TelemetryMetrics, TelemetrySampler, VendorTelemetry,
};
use turbine_device::{DeviceInventory, DiscoveryOptions};
use turbine_kv::tier::L2NvmeTier;
use turbine_kv::{KvDocument, KvMetrics};
use turbine_model::ModelMetrics;
use turbine_model::executor::ModelExecutor;
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_scheduler::{Scheduler, SchedulerMetrics, SchedulerSnapshot, SubmitError, policy};

use crate::backend::ModelBackend;
use crate::kv_orchestrator::{CopyDevice, KvHandle, KvOrchestrator, KvStart, L2StorageProbe};
use crate::metrics::ServerMetrics;
use crate::model::{self, PreparedModel};
use crate::reliability::{self as rel, EngineLedgerProbe, ReliabilityInputs};

pub(crate) use deadlines::Timeouts;
pub(crate) use r#loop::{EngineLoop, EngineParts};
pub(crate) use requests::{Submission, ToolOutput, ToolParser};

/// Events buffered per request between the engine and the HTTP response (P2 S-7); one slot
/// of it stays reserved for the error event that ends a stream the engine closes on a slow
/// client.
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

/// The engine's answer to a submission: queued, or refused by the scheduler's checks.
pub type SubmitAck = oneshot::Sender<Result<(), SubmitError>>;

/// What the HTTP side sends the engine.
pub enum EngineCommand {
    /// A new request, its output channel (capacity [`EVENT_CHANNEL_CAPACITY`]) and where the
    /// admission decision goes. The decision arrives before any event, so a refused request is
    /// a plain HTTP error even when it asked for a stream.
    Submit(Box<Submission>, mpsc::Sender<GenerationEvent>, SubmitAck),
    /// Refuse new submissions, cancel every request with reason `shutdown` and stop once the
    /// held events are delivered.
    Shutdown,
    /// The pressure controller changed the circuit state: look at it now, even when idle.
    Wake,
}

/// The HTTP side's end of the command channels.
#[derive(Clone)]
pub struct EngineHandle {
    pub submit_tx: mpsc::Sender<EngineCommand>,
    /// `POST /turbine/v1/kv/prefetch` (Phase 4).
    pub kv: KvHandle,
}

/// What the engine thread needs to build the KV hierarchy (Phase 4): the `kv` section and the
/// L2 tier opened before the listener bound.
pub struct KvSetup {
    pub cfg: KvConfig,
    pub l2: Option<Arc<L2NvmeTier>>,
}

/// Why the engine asks the server to exit.
#[derive(Debug)]
// `DeviceFatal` names the P3 circuit reason; the lint only notices it from three variants on.
#[allow(clippy::enum_variant_names)]
pub enum Fatal {
    /// Weight load, memory budget, KV allocation, emergency reserve or warm-up failed after
    /// the listener bound: exit 1.
    LoadFailed(String),
    /// The circuit breaker is fatal — a sticky device error, a controller failure, or an
    /// engine panic: exit 3 (P3 S-12, CONFLICT C-25).
    DeviceFatal(String),
    /// A `static`-mode worker rank process stopped (P5 S-5): `None` when its leader shut it down
    /// (exit 0), else why — a lost leader, a failed join, load or step (exit 1).
    RankStopped(Option<String>),
}

/// The documents behind `GET /turbine/v1/scheduler` and `GET /turbine/v1/kv`.
#[derive(Clone, Debug)]
pub struct EngineDocs {
    pub scheduler: SchedulerSnapshot,
    pub kv: KvDocument,
}

/// State the engine publishes for the HTTP side; `None` until the engine runs.
#[derive(Default)]
pub struct EngineShared {
    docs: Mutex<Option<EngineDocs>>,
    /// Tokens this engine still has to process for the requests it holds: per request its
    /// prompt plus `n × max_tokens`, less what it generated. Raised when a submission is
    /// accepted and recomputed after every step; the data-parallel router's load (P5 S-7).
    outstanding_tokens: AtomicU64,
}

impl EngineShared {
    pub fn outstanding_tokens(&self) -> u64 {
        self.outstanding_tokens.load(Ordering::Acquire)
    }

    pub(crate) fn add_outstanding(&self, tokens: u64) {
        self.outstanding_tokens.fetch_add(tokens, Ordering::AcqRel);
    }

    pub(crate) fn set_outstanding(&self, tokens: u64) {
        self.outstanding_tokens.store(tokens, Ordering::Release);
    }

    pub fn publish(&self, docs: EngineDocs) {
        *self.docs.lock().unwrap_or_else(PoisonError::into_inner) = Some(docs);
    }

    pub fn docs(&self) -> Option<EngineDocs> {
        self.docs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Metric handles the engine records into.
#[derive(Clone)]
pub struct EngineMetrics {
    pub server: ServerMetrics,
    pub model: ModelMetrics,
    pub scheduler: SchedulerMetrics,
    pub kv: KvMetrics,
}

/// What the engine thread needs besides the prepared model to start the reliability side.
pub struct ReliabilityStartup {
    pub inventory: DeviceInventory,
    /// `devices.nvml_library` / `devices.amd_smi_library` for the vendor telemetry.
    pub devices: DevicesConfig,
    pub metrics: ReliabilityMetrics,
    pub telemetry: TelemetryMetrics,
    /// The KV hierarchy's inputs (Phase 4).
    pub kv: KvSetup,
    /// The data-parallel replica this engine serves (P5 S-7; 0 with one replica).
    pub replica: u32,
}

/// The device side of `prepared`'s KV copies (Phase 4): its kernel library's copy stream and
/// pinned memory, or synchronous copies (no copy engine: L1 is disabled, logged).
pub(crate) fn copy_device(prepared: &PreparedModel) -> CopyDevice {
    match &prepared.provider.opened.context {
        Some(ctx) if ctx.has_copy_engine() => CopyDevice::Stream {
            engine: Arc::clone(ctx) as _,
            pinned: Arc::clone(ctx) as _,
        },
        Some(ctx) => {
            tracing::warn!(
                event = "kv_copy_engine_unavailable",
                library = %ctx.library().path().display(),
                minor = ctx.library().abi_minor(),
                "the kernel library has no copy streams (ABI v2.5); KV copies are \
                 synchronous and L1 is disabled"
            );
            CopyDevice::Sync {
                mem: Arc::clone(&prepared.provider.opened.mem),
            }
        }
        None => CopyDevice::Sync {
            mem: Arc::clone(&prepared.provider.opened.mem),
        },
    }
}

/// Starts the engine thread: it loads the weights, measures the memory budget, allocates the KV
/// pool and the emergency reserve, warms up, starts the telemetry sampler and the pressure
/// controller, marks the backend ready and then serves until the command channel closes or
/// `Shutdown`. With `group` (P5 S-6) `prepared` is rank 0 of a tensor-parallel group, loaded with
/// its workers ([`tp::load_group`]); with `pipeline` (P5 S-10) it is the last stage of a
/// pipeline, loaded with the earlier stages ([`pp::load_pipeline`]), and the engine keeps up to
/// `parallel.pipeline.micro_batches` micro-batches in flight. Failures are reported on `fatal`.
/// The thread logs `engine_stopped` once everything it owns is dropped; a clean shutdown joins
/// the returned handle ([`crate::startup`]) so that the process never exits (running the device
/// runtime's static destructors) while the engine still frees device memory.
#[allow(clippy::too_many_arguments)]
pub fn spawn(
    prepared: PreparedModel,
    group: Option<tp::TpGroupStart>,
    pipeline: Option<pp::PipelineStart>,
    backend: Arc<ModelBackend>,
    metrics: EngineMetrics,
    startup: ReliabilityStartup,
    queue_capacity: usize,
    timeouts: Timeouts,
    fatal: UnboundedSender<Fatal>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let replica = startup.replica;
    std::thread::Builder::new()
        .name(if replica == 0 {
            "turbine-engine".into()
        } else {
            format!("turbine-engine-{replica}")
        })
        .spawn(move || {
            // Everything the engine owns (executor, KV pool and tiers, sampler, the prepared
            // model and its kernel provider) is captured by `serve` and dropped when it returns,
            // so `engine_stopped` is logged only once the device resources are released.
            let serve = move || {
                // `scheduler.policy` was checked against the registry before any port was bound
                // (`Config::validate_modules`); `select` logs `module_selected`.
                let policy = match policy::registry()
                    .select(&prepared.modules.scheduling_policy, "scheduler.policy")
                {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                        return;
                    }
                };
                let warmup_token = prepared.generation.bos_token_id.unwrap_or(0);
                let stats = pipeline.as_ref().map(|p| Arc::clone(&p.stats));
                let loaded = match (group, pipeline) {
                    (_, Some(pipeline)) => {
                        let phase = |reason| backend.set_loading(reason);
                        pp::load_pipeline(
                            &prepared,
                            pipeline,
                            warmup_token,
                            &metrics.model,
                            &startup.metrics,
                            &phase,
                        )
                    }
                    (None, None) => {
                        model::load(&prepared, warmup_token, &metrics.model, &startup.metrics)
                    }
                    (Some(group), None) => {
                        let phase = |reason| backend.set_loading(reason);
                        tp::load_group(
                            &prepared,
                            group,
                            warmup_token,
                            &metrics.model,
                            &startup.metrics,
                            &phase,
                        )
                    }
                };
                let mut loaded = match loaded {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                        return;
                    }
                };
                let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
                // Phase 4: the KV hierarchy over the pool, before the reliability side that drives
                // it. L1 needs the kernel library's copy engine (ABI v2.3 + v2.5). Under tensor
                // parallelism every worker rank's pool is a shard of each block (P5).
                let shards = std::mem::take(&mut loaded.shards);
                let mut pool = loaded.pool;
                let device = copy_device(&prepared);
                let l2 = startup.kv.l2.clone();
                // A pipeline's pool is its last stage's; the earlier stages' shards come first.
                let start_kv = if stats.is_some() {
                    KvOrchestrator::start_pipeline
                } else {
                    KvOrchestrator::start
                };
                let started = start_kv(
                    KvStart {
                        cfg: &startup.kv.cfg,
                        memory_kind: prepared.provider.opened.memory_kind,
                        identity: prepared.identity,
                        device,
                        shards,
                        l2: startup.kv.l2,
                        clock: Arc::clone(&clock),
                        metrics: metrics.kv.clone(),
                        remote: loaded.remote_tiers.take(),
                        kv_scales: crate::kv_orchestrator::scale_hashes(&prepared.arch.kv_cache),
                    },
                    &mut pool,
                );
                let (mut kv, kv_handle) = match started {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                        return;
                    }
                };
                let reliability = &prepared.reliability;
                kv.set_ladder_dwell(reliability.pressure.deescalate_dwell.0);
                #[cfg(feature = "fault-injection")]
            let injector = reliability.fault_injection.clone().map(|cfg| {
                tracing::warn!(event = "fault_injection", config = ?cfg, "fault injection is on");
                Arc::new(turbine_reliability::fault::FaultInjector::new(cfg))
            });
                // Startup reservations are taken; injected allocation failures start with requests.
                #[cfg(feature = "fault-injection")]
                if let Some(injector) = &injector {
                    loaded.ledger.set_fault_injector(Arc::clone(injector));
                }
                let layout = *loaded.executor.kv_layout();
                let ledger = Arc::clone(&loaded.ledger);
                // P5 S-8: a tensor-parallel group admits against every rank's ledger.
                let group = std::mem::take(&mut loaded.group);
                let probe_group = group
                    .iter()
                    .map(|(b, l)| (b.device, Arc::clone(l)))
                    .collect();
                let parts = rel::build(ReliabilityInputs {
                    config: reliability,
                    budget: loaded.budget,
                    ledger: Arc::clone(&ledger),
                    reserve: loaded.reserve,
                    held: loaded.held,
                    params: &prepared.scheduler,
                    block_bytes: layout.block_bytes(),
                    workspace_bytes_per_token: prepared.workspace_bytes
                        / u64::from(prepared.scheduler.max_batch_tokens.max(1)),
                    metrics: startup.metrics.clone(),
                    clock: Arc::clone(&clock),
                    reclaimer: kv.reclaimer(),
                    replica,
                    group,
                });
                #[allow(unused_mut)]
                let mut executor: Box<dyn ModelExecutor> = loaded.executor;
                #[cfg(feature = "fault-injection")]
                if let Some(injector) = &injector {
                    // The backend in use names its sticky errors; one without any (the cpu backend)
                    // borrows the first registered backend's, which `KernelError::is_sticky` knows.
                    let sticky_name = std::iter::once(prepared.provider.backend)
                        .chain(turbine_kernels::backends::registry().iter())
                        .find_map(|b| b.sticky_error_prefixes().first().copied())
                        .unwrap_or("device error");
                    executor = Box::new(rel::FaultyExecutor::new(
                        executor,
                        Arc::clone(injector),
                        sticky_name,
                    ));
                }
                #[allow(unused_mut)]
                let mut vendor: Vec<Box<dyn VendorTelemetry>> =
                    vendor_backends(&DiscoveryOptions::from_config(&startup.devices));
                #[cfg(feature = "fault-injection")]
                if let Some(cfg) = reliability.fault_injection.as_ref()
                    && (cfg.telemetry_temperature_c.is_some() || cfg.telemetry_delay.is_some())
                {
                    vendor = vendor
                        .into_iter()
                        .map(|v| -> Box<dyn VendorTelemetry> {
                            Box::new(rel::FaultyVendor::new(
                                v,
                                cfg.telemetry_temperature_c,
                                cfg.telemetry_delay.map(|d| d.0),
                            ))
                        })
                        .collect();
                }
                let probe = EngineLedgerProbe::new(
                    ledger,
                    prepared.device,
                    Arc::clone(&parts.queue_len),
                    reliability.admission.max_queue,
                )
                .with_group(probe_group);
                let mut core = SamplerCore::new(
                    TelemetryConfig::from_config(&reliability.telemetry),
                    &startup.inventory,
                    vendor,
                    Box::new(FsProc::default()),
                    Arc::new(probe),
                    Arc::clone(&clock),
                )
                .with_metrics(startup.telemetry.clone());
                if let Some(l2) = l2 {
                    core = core.with_storage(Arc::new(L2StorageProbe {
                        l2,
                        max_queue_depth: startup.kv.cfg.nvme.max_queue_depth,
                    }));
                }
                let (sampler, latest) = TelemetrySampler::spawn_core(core);

                let PreparedModel {
                    tokenizer,
                    max_seq_len,
                    scheduler: params,
                    overlap_scheduling,
                    reliability,
                    ..
                } = prepared;
                let scheduler = Scheduler::new(params, Arc::clone(&clock))
                    .with_policy(policy)
                    .with_metrics(metrics.scheduler.clone())
                    .with_gate(parts.gate)
                    .with_micro_batches(stats.as_ref().map_or(1, |s| s.micro_batches));
                let (submit_tx, commands) = mpsc::channel(queue_capacity.max(1));
                let stop = Arc::new(AtomicBool::new(false));
                if let Err(e) = rel::spawn_controller(
                    parts.controller,
                    latest,
                    parts.stats,
                    reliability.telemetry.interval.0,
                    submit_tx.downgrade(),
                    Arc::clone(&stop),
                ) {
                    let _ = fatal.send(Fatal::LoadFailed(format!(
                        "cannot start the pressure controller thread: {e}"
                    )));
                    return;
                }
                let controller = parts.engine.handle.clone();
                let shared = Arc::new(EngineShared::default());
                let engine = EngineLoop::new(EngineParts {
                    executor,
                    pool,
                    kv,
                    scheduler,
                    clock,
                    commands,
                    shared: Arc::clone(&shared),
                    tokenizer,
                    max_seq_len,
                    metrics,
                    timeouts,
                    overlap: overlap_scheduling,
                    reliability: parts.engine,
                    pipeline: stats,
                });
                backend.set_ready(
                    replica,
                    EngineHandle {
                        submit_tx,
                        kv: kv_handle,
                    },
                    shared,
                    controller,
                    loaded.load_seconds,
                    loaded.weight_bytes,
                );
                tracing::info!(replica, "ready");
                drop(backend);
                let result = engine.run();
                stop.store(true, Ordering::Release);
                drop(sampler);
                if let Err(message) = result {
                    let _ = fatal.send(Fatal::DeviceFatal(message));
                }
            };
            serve();
            tracing::info!(
                event = "engine_stopped",
                replica,
                "engine stopped; its device resources are released"
            );
        })
}
