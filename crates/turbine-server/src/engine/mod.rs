//! The Phase 2 engine (P2 S-1, contract §16.4): one dedicated OS thread owns the device context
//! (through the executor and the KV block pool), the scheduler, and every request's sampler and
//! detokenizer. The HTTP side reaches it only through bounded channels: [`EngineHandle`]'s
//! command channel (capacity `scheduler.max_queued_requests`) in, one channel of
//! [`EVENT_CHANNEL_CAPACITY`] events per request out. The engine never waits on a client: a
//! full output channel pauses the request, a closed one cancels it.
//!
//! Diagnostics read the documents the engine publishes after every step ([`EngineShared`]).

pub(crate) mod deadlines;
pub(crate) mod grammar;
mod r#loop;
pub(crate) mod requests;
pub(crate) mod stages;

use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{mpsc, oneshot};
use turbine_core::clock::{Clock, SystemClock};
use turbine_core::request::GenerationEvent;
use turbine_kv::{KvDocument, KvMetrics};
use turbine_model::ModelMetrics;
use turbine_scheduler::{Scheduler, SchedulerMetrics, SchedulerSnapshot, SubmitError, policy};

use crate::backend::ModelBackend;
use crate::metrics::ServerMetrics;
use crate::model::{self, PreparedModel};

pub(crate) use deadlines::Timeouts;
pub(crate) use r#loop::{EngineLoop, EngineParts};
pub(crate) use requests::{Submission, ToolOutput, ToolParser};

/// Events buffered per request between the engine and the HTTP response (P2 S-7).
pub const EVENT_CHANNEL_CAPACITY: usize = 256;
/// Consecutive failed iterations after which the server reports `device_error` and exits 1
/// (CONFLICT C-25).
pub const MAX_CONSECUTIVE_FAILURES: u32 = 3;

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
}

/// The HTTP side's end of the command channel.
#[derive(Clone)]
pub struct EngineHandle {
    pub submit_tx: mpsc::Sender<EngineCommand>,
}

/// Why the engine asks the server to exit 1.
#[derive(Debug)]
pub enum Fatal {
    /// Weight load, KV allocation or warm-up failed after the listener bound.
    LoadFailed(String),
    /// [`MAX_CONSECUTIVE_FAILURES`] iterations in a row failed, or the engine panicked.
    DeviceError(String),
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

/// Starts the engine thread: it loads the weights, allocates the KV pool, warms up, marks the
/// backend ready and then serves until the command channel closes or `Shutdown`. Failures are
/// reported on `fatal`.
pub fn spawn(
    prepared: PreparedModel,
    backend: Arc<ModelBackend>,
    metrics: EngineMetrics,
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
            let loaded = match model::load(&prepared, warmup_token, &metrics.model) {
                Ok(l) => l,
                Err(e) => {
                    let _ = fatal.send(Fatal::LoadFailed(e.to_string()));
                    return;
                }
            };
            let PreparedModel {
                tokenizer,
                max_seq_len,
                scheduler: params,
                overlap_scheduling,
                ..
            } = prepared;
            let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
            let scheduler = Scheduler::new(params, Arc::clone(&clock))
                .with_policy(policy)
                .with_metrics(metrics.scheduler.clone());
            let (submit_tx, commands) = mpsc::channel(queue_capacity.max(1));
            let shared = Arc::new(EngineShared::default());
            let engine = EngineLoop::new(EngineParts {
                executor: loaded.executor,
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
            });
            backend.set_ready(
                EngineHandle { submit_tx },
                shared,
                loaded.load_seconds,
                loaded.weight_bytes,
            );
            tracing::info!("ready");
            drop(backend);
            if let Err(message) = engine.run() {
                let _ = fatal.send(Fatal::DeviceError(message));
            }
        })
        .map(|_| ())
}
