//! The server's side of P3 reliability: the emergency reserve's device allocation
//! ([`DeviceReserve`]), the ledger and admission-queue figures the telemetry sampler reads
//! ([`EngineLedgerProbe`]), the engine's handle on the pressure controller, recovery controller
//! and circuit breaker ([`EngineReliability`]), the supervised controller thread
//! ([`spawn_controller`]), the reject table as HTTP errors ([`api_error_for`]) and, in the
//! `fault-injection` build, the executor and vendor-telemetry wrappers that inject faults.
//!
//! Ownership: the [`PressureController`] sits behind one mutex shared by the controller thread
//! (one `tick` per telemetry fast tick) and the engine thread (device OOM, recovery outcomes,
//! device errors and probe results — never per token). Everyone else reads the controller's
//! atomic [`ControllerHandle`] snapshot. The engine publishes its per-iteration figures through
//! a lock-free [`EngineStats`] cell the controller thread reads.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::mpsc::WeakSender;
use turbine_api::ApiError;
use turbine_core::clock::Clock;
use turbine_core::config::ReliabilityConfig;
use turbine_core::request::ErrorCode;
use turbine_core::telemetry::LedgerProbe;
use turbine_core::types::{DeviceId, PressureSignal};
use turbine_device::telemetry::LatestSample;
use turbine_reliability::admission::{
    Admission, AdmissionParams, AdmissionQueue, Calibration, RejectionReason,
};
use turbine_reliability::budget::{DeviceBudget, PoolKind};
use turbine_reliability::circuit::{CircuitEvent, CircuitTransition};
use turbine_reliability::controller::{ControllerHandle, EngineStats, PressureController};
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::recovery::{RecoveryController, RecoveryOutcome, RecoveryStep};
use turbine_reliability::reserve::{EmergencyReserve, ReserveAllocator};
use turbine_reliability::signals::effective_thresholds;
use turbine_reliability::throttle::{KvReclaimer, SchedulerLimits};
use turbine_scheduler::{AdmissionGate, SchedulerParams};
use turbine_tensor::{DeviceBuffer, DeviceMemory};

use crate::engine::EngineCommand;

/// Throughput figures admission estimates use before its EWMAs are seeded by the first 32
/// iterations (P3 §Data): conservative figures for a single GPU; they only shape the
/// `Retry-After` estimate and the logged `est_*_seconds`, never a decision.
const DEFAULT_CALIBRATION: Calibration = Calibration {
    prefill_tokens_per_s: 2000.0,
    decode_step_s: 0.05,
};

/// The emergency reserve's real device allocation: one [`DeviceBuffer`] of the reserve's size,
/// dropped (freed) when the recovery controller releases the reserve.
pub struct DeviceReserve {
    mem: Arc<dyn DeviceMemory>,
    buffer: Option<DeviceBuffer>,
}

impl DeviceReserve {
    pub fn new(mem: Arc<dyn DeviceMemory>) -> DeviceReserve {
        DeviceReserve { mem, buffer: None }
    }
}

impl ReserveAllocator for DeviceReserve {
    fn allocate(&mut self, bytes: u64) -> Result<(), String> {
        let len = usize::try_from(bytes)
            .map_err(|_| format!("{bytes} bytes do not fit the address space"))?;
        let buffer = DeviceBuffer::alloc(&self.mem, len).map_err(|e| e.to_string())?;
        self.buffer = Some(buffer);
        Ok(())
    }

    fn free(&mut self) {
        self.buffer = None;
    }
}

/// What the telemetry sampler's fast tick reads besides `/proc`: the KV pool's utilisation
/// (used + reserved) from the ledger and the admission queue's fill, which the engine
/// publishes after every turn.
pub struct EngineLedgerProbe {
    ledger: Arc<Ledger>,
    device: DeviceId,
    queue_len: Arc<AtomicU32>,
    max_queue: u32,
    /// A tensor-parallel group's other ranks (P5 S-8): the KV utilisation is the worst rank's.
    group: Vec<(DeviceId, Arc<Ledger>)>,
}

impl EngineLedgerProbe {
    pub fn new(
        ledger: Arc<Ledger>,
        device: DeviceId,
        queue_len: Arc<AtomicU32>,
        max_queue: u32,
    ) -> EngineLedgerProbe {
        EngineLedgerProbe {
            ledger,
            device,
            queue_len,
            max_queue,
            group: Vec::new(),
        }
    }

    /// Reads a tensor-parallel group whose other ranks are `group` (device, ledger): the KV
    /// utilisation is then its worst member's (P5 S-8).
    pub fn with_group(mut self, group: Vec<(DeviceId, Arc<Ledger>)>) -> EngineLedgerProbe {
        self.group = group;
        self
    }
}

impl LedgerProbe for EngineLedgerProbe {
    fn kv_utilization(&self) -> f64 {
        self.group
            .iter()
            .map(|(d, l)| l.usage(*d, PoolKind::Kv).utilization())
            .fold(
                self.ledger.usage(self.device, PoolKind::Kv).utilization(),
                f64::max,
            )
    }

    fn queue_fill(&self) -> f64 {
        f64::from(self.queue_len.load(Ordering::Relaxed)) / f64::from(self.max_queue.max(1))
    }
}

/// The HTTP error of an admission rejection (P3 reject table): status, `type` and `code` from
/// the table, `Retry-After` = `retry_after_secs` (at least 1) when positive.
pub fn api_error_for(reason: RejectionReason, retry_after_secs: u64) -> ApiError {
    let code = match reason {
        RejectionReason::ContextExceedsKvCapacity => ErrorCode::ContextExceedsKvCapacity,
        RejectionReason::QueueFull => ErrorCode::QueueFull,
        RejectionReason::QueueTimeout => ErrorCode::QueueTimeout,
        RejectionReason::Survival => ErrorCode::Overloaded,
        RejectionReason::CircuitOpen => ErrorCode::CircuitOpen,
        _ => ErrorCode::InternalError,
    };
    ApiError::overload(code, (retry_after_secs > 0).then_some(retry_after_secs))
}

fn lock(controller: &Mutex<PressureController>) -> MutexGuard<'_, PressureController> {
    // A panic inside `tick` poisons the mutex; the controller then only takes the circuit
    // event that reports the failure, which it can apply whatever the state it panicked in.
    controller.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The engine thread's end of reliability: the shared controller, the recovery controller it
/// owns, the stats cell it publishes into and the startup reservations it keeps committed.
pub(crate) struct EngineReliability {
    controller: Arc<Mutex<PressureController>>,
    pub handle: ControllerHandle,
    recovery: RecoveryController,
    stats: Arc<ArcSwap<EngineStats>>,
    queue_len: Arc<AtomicU32>,
    /// `reliability.circuit.drain_timeout`: bounds the drain after a controller failure.
    pub drain_timeout: Duration,
    /// The committed `weights` and `workspace` reservations, held for the engine's lifetime.
    _held: Vec<Reservation>,
}

impl EngineReliability {
    /// Device out-of-memory: SURVIVAL now and the emergency reserve released (its bytes).
    pub fn on_oom(&self) -> u64 {
        lock(&self.controller).on_oom()
    }

    /// The iteration attempt that just failed with a device OOM had `batch_len` sequences.
    pub fn recovery_step(&mut self, batch_len: usize) -> RecoveryStep {
        self.recovery.on_oom(batch_len)
    }

    /// An iteration attempt succeeded: reports the end of a recovery to the controller.
    pub fn recovery_succeeded(&mut self) {
        if let Some(outcome) = self.recovery.on_success() {
            lock(&self.controller).on_recovery(outcome);
        }
    }

    /// Retries exhausted: the batch failed `resource_exhausted`.
    pub fn recovery_failed(&self) {
        lock(&self.controller).on_recovery(RecoveryOutcome::Failed);
    }

    pub fn circuit_event(&self, ev: CircuitEvent) -> Option<CircuitTransition> {
        lock(&self.controller).on_circuit_event(ev)
    }

    /// The engine's figures after a turn; the queue length also feeds `queue_fill`.
    pub fn publish(&self, stats: EngineStats) {
        self.queue_len.store(stats.queue_len, Ordering::Relaxed);
        self.stats.store(Arc::new(stats));
    }
}

/// Everything the reliability side is built from once the weights are loaded.
pub(crate) struct ReliabilityInputs<'a> {
    pub config: &'a ReliabilityConfig,
    pub budget: DeviceBudget,
    pub ledger: Arc<Ledger>,
    pub reserve: EmergencyReserve,
    /// Startup reservations to keep committed (weights, workspace).
    pub held: Vec<Reservation>,
    pub params: &'a SchedulerParams,
    pub block_bytes: u64,
    pub workspace_bytes_per_token: u64,
    pub metrics: ReliabilityMetrics,
    pub clock: Arc<dyn Clock>,
    /// What the controller's reclaim step drives (P3 S-10): the KV hierarchy's lock-free
    /// `KvReclaimHandle` from Phase 4 (demotion to lower tiers, freeing cached blocks).
    pub reclaimer: Arc<dyn KvReclaimer>,
    /// The data-parallel replica this engine serves (P5; 0 with one replica).
    pub replica: u32,
    /// A tensor-parallel group's other ranks, in rank order (P5 S-8): each rank's budget and
    /// ledger. Admission reserves every request's KV on each of them and on `ledger`, all or
    /// nothing; the pressure document lists them. Empty on one device.
    pub group: Vec<(DeviceBudget, Arc<Ledger>)>,
}

/// The pieces [`build`] returns: the engine's end, the admission gate for its scheduler, and
/// what the controller thread and the telemetry sampler need.
pub(crate) struct ReliabilityParts {
    pub engine: EngineReliability,
    pub gate: AdmissionGate,
    pub controller: Arc<Mutex<PressureController>>,
    pub stats: Arc<ArcSwap<EngineStats>>,
    pub queue_len: Arc<AtomicU32>,
}

/// The pressure controller over the budget, ledger and reserve, the admission gate in front of
/// the scheduler, and the recovery controller.
pub(crate) fn build(inp: ReliabilityInputs<'_>) -> ReliabilityParts {
    let cfg = inp.config;
    let device = inp.budget.device;
    let (controller, handle) = PressureController::new(
        cfg,
        SchedulerLimits {
            prefill_chunk_tokens: inp.params.prefill_chunk_tokens,
            block_tokens: inp.params.block_tokens,
        },
        inp.budget,
        Arc::clone(&inp.ledger),
        inp.reserve,
        Arc::clone(&inp.reclaimer),
        inp.metrics.clone(),
        Arc::clone(&inp.clock),
    );
    let controller = controller.with_replica(inp.replica);
    let controller = if inp.group.is_empty() {
        controller
    } else {
        controller.with_group_ranks(inp.group.clone())
    };
    let admission = Admission::new(
        AdmissionParams {
            device,
            adaptive: cfg.adaptive_admission,
            max_queue: cfg.admission.max_queue,
            large_prefill_tokens: cfg.admission.large_prefill_tokens,
            block_bytes: inp.block_bytes,
            prefill_chunk_tokens: inp.params.prefill_chunk_tokens,
            workspace_bytes_per_token: inp.workspace_bytes_per_token,
            calibration: DEFAULT_CALIBRATION,
        },
        Arc::clone(&inp.ledger),
        inp.metrics.clone(),
    )
    .with_kv_headroom(effective_thresholds(&cfg.pressure)[&PressureSignal::KvUtilization])
    .with_group_ledgers(
        inp.group
            .iter()
            .map(|(b, l)| (b.device, Arc::clone(l)))
            .collect(),
    );
    let queue = AdmissionQueue::new(
        cfg.admission.max_queue,
        cfg.admission.queue_timeout.0,
        cfg.admission.max_bypass,
    );
    let gate = AdmissionGate::new(
        admission,
        queue,
        handle.clone(),
        Arc::clone(&inp.clock),
        inp.params.max_running_requests,
    );
    let controller = Arc::new(Mutex::new(controller));
    let stats = Arc::new(ArcSwap::from_pointee(EngineStats {
        block_tokens: inp.params.block_tokens,
        ..EngineStats::default()
    }));
    let queue_len = Arc::new(AtomicU32::new(0));
    ReliabilityParts {
        engine: EngineReliability {
            controller: Arc::clone(&controller),
            handle,
            recovery: RecoveryController::new(&cfg.recovery, inp.metrics),
            stats: Arc::clone(&stats),
            queue_len: Arc::clone(&queue_len),
            drain_timeout: cfg.circuit.drain_timeout.0,
            _held: inp.held,
        },
        gate,
        controller,
        stats,
        queue_len,
    }
}

/// Runs the controller on its own thread: every `interval` it reports the engine's new
/// iterations to the circuit breaker and ticks on the latest telemetry sample and engine
/// figures. A circuit state change wakes the engine (it may be blocked waiting for work while
/// the circuit moves on to PROBING). A panic inside a tick is caught: the circuit opens with
/// `controller_failed` (fatal: the engine drains and the process exits 3), the engine is woken
/// and the thread stops — pressure control is never silently lost. The thread ends when `stop`
/// is set.
pub(crate) fn spawn_controller(
    controller: Arc<Mutex<PressureController>>,
    latest: LatestSample,
    stats: Arc<ArcSwap<EngineStats>>,
    interval: Duration,
    wake: WeakSender<EngineCommand>,
    stop: Arc<AtomicBool>,
) -> std::io::Result<JoinHandle<()>> {
    let wake_engine = move || {
        if let Some(tx) = wake.upgrade() {
            // A full channel wakes the engine anyway.
            let _ = tx.try_send(EngineCommand::Wake);
        }
    };
    std::thread::Builder::new()
        .name("turbine-pressure".into())
        .spawn(move || {
            let mut seen_iterations = 0u64;
            while !stop.load(Ordering::Acquire) {
                std::thread::sleep(interval);
                let sample = latest.load();
                let engine = stats.load_full();
                let ticked = catch_unwind(AssertUnwindSafe(|| {
                    let mut c = lock(&controller);
                    let before = c.handle().circuit();
                    if engine.iterations != seen_iterations {
                        c.on_circuit_event(CircuitEvent::Iteration);
                    }
                    c.tick(&sample, &engine);
                    before != c.handle().circuit()
                }));
                seen_iterations = engine.iterations;
                match ticked {
                    Ok(true) => wake_engine(),
                    Ok(false) => {}
                    Err(panic) => {
                        let what = panic
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic payload".into());
                        tracing::error!(
                            event = "circuit_transition",
                            reason = "controller_failed",
                            panic = %what,
                            "pressure controller panicked; draining, then exiting 3"
                        );
                        lock(&controller).on_circuit_event(CircuitEvent::ControllerFailed);
                        wake_engine();
                        return;
                    }
                }
            }
        })
}

#[cfg(feature = "fault-injection")]
pub use faults::{FaultyExecutor, FaultyVendor};

/// Fault wrappers (P3 S-16): only in the `fault-injection` build.
#[cfg(feature = "fault-injection")]
mod faults {
    use std::sync::Arc;
    use std::time::Duration;

    use turbine_core::telemetry::DeviceSample;
    use turbine_core::types::{BlockId, KvLayout, ModelShape, Vendor};
    use turbine_device::DeviceInfo;
    use turbine_device::telemetry::VendorTelemetry;
    use turbine_kernels::KernelError;
    use turbine_model::ModelError;
    use turbine_model::executor::{
        BatchInput, DecodeGraphs, ForwardTimings, GraphCounters, Logits, ModelExecutor, TokenFeed,
    };
    use turbine_reliability::fault::{FaultInjector, InjectedIterationFault};
    use turbine_tensor::KvPoolView;

    /// Raises `reliability.fault_injection.oom_at_iteration` (device out-of-memory) and
    /// `kernel_error_at_iteration` (a sticky device error named with the opened backend's first
    /// sticky error name, or a plain kernel error) on the configured forward pass or launch,
    /// counted from 1 after the warm-up; every other call goes to the wrapped executor.
    pub struct FaultyExecutor {
        inner: Box<dyn ModelExecutor>,
        injector: Arc<FaultInjector>,
        /// `ExecutionBackend::sticky_error_prefixes()[0]` of the backend in use (Phase 2m:
        /// vendor error names live behind the backend).
        sticky_name: String,
        forwards: u64,
    }

    impl FaultyExecutor {
        pub fn new(
            inner: Box<dyn ModelExecutor>,
            injector: Arc<FaultInjector>,
            sticky_name: &str,
        ) -> FaultyExecutor {
            FaultyExecutor {
                inner,
                injector,
                sticky_name: sticky_name.to_string(),
                forwards: 0,
            }
        }

        /// The fault of the next forward pass or launch, if one is configured for it.
        fn next_fault(&mut self) -> Option<ModelError> {
            self.forwards += 1;
            match self.injector.iteration_fault(self.forwards)? {
                InjectedIterationFault::DeviceOom => {
                    tracing::warn!(
                        event = "fault_injected",
                        fault = "device_oom",
                        iteration = self.forwards
                    );
                    Some(ModelError::Kernel(KernelError::OutOfMemory {
                        message: "injected device out-of-memory".into(),
                    }))
                }
                InjectedIterationFault::KernelError { sticky } => {
                    tracing::warn!(
                        event = "fault_injected",
                        fault = "kernel_error",
                        sticky,
                        iteration = self.forwards
                    );
                    let message = if sticky {
                        format!("{}: injected", self.sticky_name)
                    } else {
                        "injected kernel error".to_string()
                    };
                    Some(ModelError::Kernel(KernelError::Device { message }))
                }
            }
        }
    }

    impl ModelExecutor for FaultyExecutor {
        fn shape(&self) -> &ModelShape {
            self.inner.shape()
        }

        fn kv_layout(&self) -> &KvLayout {
            self.inner.kv_layout()
        }

        fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            match self.next_fault() {
                Some(e) => Err(e),
                None => self.inner.forward(batch),
            }
        }

        fn last_timings(&self) -> ForwardTimings {
            self.inner.last_timings()
        }

        fn overlaps(&self) -> bool {
            self.inner.overlaps()
        }

        fn launch(
            &mut self,
            batch: &BatchInput<'_>,
            feeds: &[TokenFeed],
        ) -> Result<(), ModelError> {
            match self.next_fault() {
                Some(e) => Err(e),
                None => self.inner.launch(batch, feeds),
            }
        }

        fn collect(&mut self) -> Result<Logits, ModelError> {
            self.inner.collect()
        }

        fn set_decode_graphs(&mut self, graphs: Option<DecodeGraphs>) {
            self.inner.set_decode_graphs(graphs);
        }

        fn graph_counters(&self) -> GraphCounters {
            self.inner.graph_counters()
        }

        fn reduces_logits(&self) -> bool {
            self.inner.reduces_logits()
        }

        fn copy_blocks(
            &mut self,
            kv: &KvPoolView<'_>,
            src: &[BlockId],
            dst: &[BlockId],
        ) -> Result<(), ModelError> {
            self.inner.copy_blocks(kv, src, dst)
        }
    }

    /// A vendor telemetry library whose calls are delayed by `telemetry_delay` and whose
    /// temperature reads `telemetry_temperature_c` when set.
    pub struct FaultyVendor {
        inner: Box<dyn VendorTelemetry>,
        temperature_c: Option<f64>,
        delay: Option<Duration>,
    }

    impl FaultyVendor {
        pub fn new(
            inner: Box<dyn VendorTelemetry>,
            temperature_c: Option<f64>,
            delay: Option<Duration>,
        ) -> FaultyVendor {
            FaultyVendor {
                inner,
                temperature_c,
                delay,
            }
        }
    }

    impl VendorTelemetry for FaultyVendor {
        fn vendor(&self) -> Vendor {
            self.inner.vendor()
        }

        fn sample(&mut self, device: &DeviceInfo) -> Result<DeviceSample, String> {
            if let Some(delay) = self.delay {
                std::thread::sleep(delay);
            }
            let mut sample = self.inner.sample(device)?;
            if let Some(t) = self.temperature_c {
                sample.temperature_c = Some(t);
            }
            Ok(sample)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of the P3 reject table becomes its HTTP error.
    #[test]
    fn reject_table_as_http_errors() {
        for (reason, retry, status, code, retry_header) in [
            (
                RejectionReason::ContextExceedsKvCapacity,
                0,
                400,
                "context_exceeds_kv_capacity",
                None,
            ),
            (RejectionReason::QueueFull, 7, 429, "queue_full", Some(7)),
            (
                RejectionReason::QueueTimeout,
                3,
                503,
                "queue_timeout",
                Some(3),
            ),
            (RejectionReason::Survival, 60, 503, "overloaded", Some(60)),
            (
                RejectionReason::CircuitOpen,
                1,
                503,
                "circuit_open",
                Some(1),
            ),
        ] {
            let e = api_error_for(reason, retry);
            assert_eq!(e.status.as_u16(), status, "{reason:?}");
            assert_eq!(e.code.as_str(), code, "{reason:?}");
            assert_eq!(e.retry_after, retry_header, "{reason:?}");
            assert_eq!(reason.http().0, status);
        }
    }

    /// The probe reads the ledger's KV utilisation (used + reserved) and the published queue
    /// length against `max_queue`.
    #[test]
    fn ledger_probe_reads_kv_and_queue() {
        use turbine_core::types::MemoryKind;
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: 1000,
            pools: vec![(PoolKind::Kv, 1000)],
        };
        let ledger = Ledger::new(&budget);
        let queue_len = Arc::new(AtomicU32::new(0));
        let probe =
            EngineLedgerProbe::new(Arc::clone(&ledger), DeviceId(0), Arc::clone(&queue_len), 4);
        assert_eq!(probe.kv_utilization(), 0.0);
        let mut r = ledger.reserve(DeviceId(0), PoolKind::Kv, 250).unwrap();
        assert_eq!(probe.kv_utilization(), 0.25);
        r.commit_bytes(100);
        assert_eq!(probe.kv_utilization(), 0.25);
        drop(r);
        assert_eq!(probe.kv_utilization(), 0.0);
        queue_len.store(3, Ordering::Relaxed);
        assert_eq!(probe.queue_fill(), 0.75);
    }
}
