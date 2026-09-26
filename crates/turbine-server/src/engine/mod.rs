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

pub(crate) mod deadlines;
pub(crate) mod grammar;
mod r#loop;
pub(crate) mod requests;
pub(crate) mod stages;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{mpsc, oneshot};
use turbine_core::clock::{Clock, SystemClock};
use turbine_core::config::DevicesConfig;
use turbine_core::request::GenerationEvent;
use turbine_device::telemetry::proc::FsProc;
use turbine_device::telemetry::vendor::vendor_backends;
use turbine_device::telemetry::{
    SamplerCore, TelemetryConfig, TelemetryMetrics, TelemetrySampler, VendorTelemetry,
};
use turbine_device::{DeviceInventory, DiscoveryOptions};
use turbine_kv::{KvDocument, KvMetrics};
use turbine_model::ModelMetrics;
use turbine_model::executor::ModelExecutor;
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_scheduler::{Scheduler, SchedulerMetrics, SchedulerSnapshot, SubmitError, policy};

use crate::backend::ModelBackend;
use crate::metrics::ServerMetrics;
use crate::model::{self, PreparedModel};
use crate::reliability::{self as rel, EngineLedgerProbe, ReliabilityInputs};

pub(crate) use deadlines::Timeouts;
pub(crate) use r#loop::{EngineLoop, EngineParts};
pub(crate) use requests::{Submission, ToolOutput, ToolParser};

/// Events buffered per request between the engine and the HTTP response (P2 S-7).
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

/// The HTTP side's end of the command channel.
#[derive(Clone)]
pub struct EngineHandle {
    pub submit_tx: mpsc::Sender<EngineCommand>,
}

/// Why the engine asks the server to exit.
#[derive(Debug)]
pub enum Fatal {
    /// Weight load, memory budget, KV allocation, emergency reserve or warm-up failed after
    /// the listener bound: exit 1.
    LoadFailed(String),
    /// The circuit breaker is fatal — a sticky device error, a controller failure, or an
    /// engine panic: exit 3 (P3 S-12, CONFLICT C-25).
    DeviceFatal(String),
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
}

impl EngineShared {
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
}

/// Starts the engine thread: it loads the weights, measures the memory budget, allocates the KV
/// pool and the emergency reserve, warms up, starts the telemetry sampler and the pressure
/// controller, marks the backend ready and then serves until the command channel closes or
/// `Shutdown`. Failures are reported on `fatal`.
pub fn spawn(
    prepared: PreparedModel,
    backend: Arc<ModelBackend>,
    metrics: EngineMetrics,
    startup: ReliabilityStartup,
    queue_capacity: usize,
    timeouts: Timeouts,
    fatal: UnboundedSender<Fatal>,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("turbine-engine".into())
        .spawn(move || {
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
            let loaded =
                match model::load(&prepared, warmup_token, &metrics.model, &startup.metrics) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                        return;
                    }
                };
            let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
            let reliability = &prepared.reliability;
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
            );
            let (sampler, latest) = TelemetrySampler::spawn_core(
                SamplerCore::new(
                    TelemetryConfig::from_config(&reliability.telemetry),
                    &startup.inventory,
                    vendor,
                    Box::new(FsProc::default()),
                    Arc::new(probe),
                    Arc::clone(&clock),
                )
                .with_metrics(startup.telemetry.clone()),
            );

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
                .with_gate(parts.gate);
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
                pool: loaded.pool,
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
            });
            backend.set_ready(
                EngineHandle { submit_tx },
                shared,
                controller,
                loaded.load_seconds,
                loaded.weight_bytes,
            );
            tracing::info!("ready");
            drop(backend);
            let result = engine.run();
            stop.store(true, Ordering::Release);
            drop(sampler);
            if let Err(message) = result {
                let _ = fatal.send(Fatal::DeviceFatal(message));
            }
        })
        .map(|_| ())
}
