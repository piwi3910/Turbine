//! Ties telemetry, signals, the state machine, the emergency reserve, the throttle plan and the circuit
//! breaker together. Runs on the telemetry fast tick (never per token); the scheduler reads one atomic
//! snapshot per iteration through `ControllerHandle`.
use crate::budget::DeviceBudget;
use crate::circuit::{CircuitBreaker, CircuitEvent, CircuitReason, CircuitTransition};
use crate::document::{
    AdmissionDoc, CircuitDoc, PressureDocument, ThrottleDoc, TransitionDoc, decisions_doc,
    memory_doc, rfc3339_millis,
};
use crate::horizon::ExhaustionHorizon;
use crate::ledger::Ledger;
use crate::metrics::{ReliabilityMetrics, SignalLabel};
use crate::reserve::EmergencyReserve;
use crate::signals::{
    PressureSignal, SignalEvaluator, SignalInputs, SignalValue, effective_thresholds,
};
use crate::state::{Gates, MachineConfig, PressureMachine, Transition};
use crate::throttle::{
    KvReclaimer, SchedulerLimits, ThrottlePlan, apply_reclaim, plan_with, publish_plan,
};
use arc_swap::ArcSwap;
use std::sync::Arc;
use std::time::Duration;
use turbine_core::clock::Clock;
use turbine_core::config::ReliabilityConfig;
use turbine_core::telemetry::TelemetrySample;
use turbine_core::types::{CircuitState, PressureState};

/// Engine-side figures published once per iteration (engine thread → controller).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineStats {
    /// Remaining `max_tokens` of every running sequence.
    pub running_remaining_tokens: Vec<u32>,
    pub free_kv_blocks: u32,
    pub block_tokens: u32,
    /// Observed per-sequence decode rate (tokens/s).
    pub decode_tokens_per_s: f64,
    /// Windowed p95 decode step time per sequence (seconds).
    pub step_time_p95_s: Option<f64>,
    pub queue_len: u32,
    pub iterations: u64,
}

/// What the scheduler and the HTTP side read; replaced atomically on every tick.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub state: PressureState,
    pub circuit: CircuitState,
    pub throttle: ThrottlePlan,
    pub circuit_retry_after_secs: u64,
    pub fatal: bool,
    pub drain_expired: bool,
    pub document: PressureDocument,
}

#[derive(Clone)]
pub struct ControllerHandle {
    snap: Arc<ArcSwap<Snapshot>>,
}
impl ControllerHandle {
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.snap.load_full()
    }
    pub fn throttle(&self) -> ThrottlePlan {
        self.snap.load().throttle
    }
    pub fn state(&self) -> PressureState {
        self.snap.load().state
    }
    pub fn circuit(&self) -> CircuitState {
        self.snap.load().circuit
    }
    pub fn document(&self) -> PressureDocument {
        self.snap.load().document.clone()
    }
}

/// A KvReclaimer that reclaims nothing (engines without a cache; tests).
pub struct NoReclaim;
impl KvReclaimer for NoReclaim {
    fn demote(&self, _: f64) -> u64 {
        0
    }
    fn free_unreferenced(&self, _: f64) -> u64 {
        0
    }
}

pub struct PressureController {
    cfg: ReliabilityConfig,
    limits: SchedulerLimits,
    clock: Arc<dyn Clock>,
    metrics: ReliabilityMetrics,
    evaluator: SignalEvaluator,
    machine: PressureMachine,
    circuit: CircuitBreaker,
    reserve: EmergencyReserve,
    budget: DeviceBudget,
    ledger: Arc<Ledger>,
    reclaimer: Arc<dyn KvReclaimer>,
    plan: ThrottlePlan,
    last_oom: Option<Duration>,
    baseline_step_s: Option<f64>,
    last_signals: Vec<SignalValue>,
    last_horizon: f64,
    handle: ControllerHandle,
}

impl PressureController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &ReliabilityConfig,
        limits: SchedulerLimits,
        budget: DeviceBudget,
        ledger: Arc<Ledger>,
        reserve: EmergencyReserve,
        reclaimer: Arc<dyn KvReclaimer>,
        metrics: ReliabilityMetrics,
        clock: Arc<dyn Clock>,
    ) -> (Self, ControllerHandle) {
        let thresholds = effective_thresholds(&cfg.pressure);
        let evaluator = SignalEvaluator::new(
            thresholds.clone(),
            cfg.telemetry.interval.0,
            cfg.telemetry.stale_after.0,
        );
        let machine = PressureMachine::new(
            MachineConfig::from_config(cfg),
            thresholds,
            metrics.clone(),
            Arc::clone(&clock),
        );
        let circuit = CircuitBreaker::new(&cfg.circuit, metrics.clone(), clock.now_mono());
        let survival = cfg.recovery.survival_liveness;
        let plan = plan_with(PressureState::Green, &limits, survival);
        publish_plan(&plan, &metrics);
        let placeholder = Snapshot {
            state: PressureState::Green,
            circuit: CircuitState::Healthy,
            throttle: plan,
            circuit_retry_after_secs: 1,
            fatal: false,
            drain_expired: false,
            document: empty_document(cfg.enabled, &plan),
        };
        let handle = ControllerHandle {
            snap: Arc::new(ArcSwap::from_pointee(placeholder)),
        };
        let mut c = Self {
            cfg: cfg.clone(),
            limits,
            clock,
            metrics,
            evaluator,
            machine,
            circuit,
            reserve,
            budget,
            ledger,
            reclaimer,
            plan,
            last_oom: None,
            baseline_step_s: None,
            last_signals: Vec::new(),
            last_horizon: f64::INFINITY,
            handle: ControllerHandle {
                snap: Arc::clone(&handle.snap),
            },
        };
        c.publish();
        (c, handle)
    }

    pub fn handle(&self) -> ControllerHandle {
        self.handle.clone()
    }

    /// One fast tick: signals → state machine → reserve → plan → circuit → snapshot.
    pub fn tick(&mut self, sample: &TelemetrySample, stats: &EngineStats) -> Option<Transition> {
        let now = self.clock.now_mono();
        self.last_horizon = ExhaustionHorizon::predict(
            &stats.running_remaining_tokens,
            stats.decode_tokens_per_s,
            stats.free_kv_blocks,
            stats.block_tokens.max(1),
        );
        let drift = match (stats.step_time_p95_s, self.baseline_step_s) {
            (Some(p95), Some(base)) if base > 0.0 => Some(p95 / base),
            _ => None,
        };
        // Baseline only from GREEN + HEALTHY iterations (P3 edge case "baseline learned under pressure").
        if let Some(p95) = stats.step_time_p95_s
            && self.machine.state() == PressureState::Green
            && self.circuit.state() == CircuitState::Healthy
        {
            self.baseline_step_s = Some(self.baseline_step_s.map_or(p95, |b| 0.9 * b + 0.1 * p95));
        }
        let dwell = self.cfg.pressure.deescalate_dwell.0;
        let devices = [(
            self.budget.device,
            self.budget.memory_kind,
            self.budget.budget_bytes,
        )];
        let inputs = SignalInputs {
            sample,
            host_reserve_bytes: self.cfg.memory.host_reserve_bytes.0,
            devices: &devices,
            exhaustion_horizon_seconds: self.last_horizon,
            step_time_drift: drift,
            allocation_failure_recent: self.last_oom.is_some_and(|t| now.saturating_sub(t) < dwell),
        };
        let signals = self.evaluator.evaluate(&inputs, now);

        if let Some(ratio) = drift {
            self.circuit_event(CircuitEvent::LatencyDrift { ratio }, now);
        }
        for s in &signals {
            match (s.signal, s.value) {
                (PressureSignal::Thermal, v) if v >= 2.0 => {
                    self.circuit_event(CircuitEvent::ThermalThrottle, now)
                }
                (PressureSignal::TelemetryStale, v) if v >= 1.0 => {
                    self.circuit_event(CircuitEvent::TelemetryStale, now)
                }
                _ => {}
            }
        }
        self.circuit_event(
            CircuitEvent::Tick {
                running: stats.running_remaining_tokens.len() as u32,
            },
            now,
        );

        if !self.reserve.held() && self.machine.state() <= PressureState::Red {
            self.reserve.try_reacquire();
        }
        let floor_signal = match self.circuit.last_reason() {
            Some(CircuitReason::LatencyDrift) => PressureSignal::StepTimeDrift,
            Some(CircuitReason::ThermalThrottle) => PressureSignal::Thermal,
            Some(CircuitReason::OomRecovered) => PressureSignal::AllocationFailure,
            _ => PressureSignal::TelemetryStale,
        };
        let floor = if self.circuit.state() == CircuitState::Degraded {
            PressureState::Yellow
        } else {
            PressureState::Green
        };
        let gates = Gates {
            floor,
            floor_signal,
            reserve_held: self.reserve.held(),
        };
        let transition = self.machine.evaluate(&signals, gates);
        self.last_signals = signals;
        self.apply_plan();
        self.publish();
        transition
    }

    /// Device out-of-memory from the executor or an allocator: SURVIVAL now, release the reserve.
    /// Returns the bytes released (0 when already released or disabled).
    pub fn on_oom(&mut self) -> u64 {
        let now = self.clock.now_mono();
        self.last_oom = Some(now);
        let failure = [SignalValue {
            signal: PressureSignal::AllocationFailure,
            value: 1.0,
            level: PressureState::Survival,
            stale: false,
        }];
        self.machine.evaluate(
            &failure,
            Gates {
                reserve_held: self.reserve.held(),
                ..Gates::default()
            },
        );
        let released = self
            .reserve
            .release_for_recovery(self.machine.state())
            .unwrap_or(0);
        if released > 0 {
            self.metrics
                .reclaim_bytes
                .get_or_create(&crate::metrics::ActionLabel {
                    action: "release_reserve",
                })
                .inc_by(released);
        }
        self.apply_plan();
        self.publish();
        released
    }

    /// Outcome of a bounded recovery (from the engine's `RecoveryController`).
    pub fn on_recovery(&mut self, outcome: crate::recovery::RecoveryOutcome) {
        let ev = match outcome {
            crate::recovery::RecoveryOutcome::Recovered { .. } => CircuitEvent::OomRecovered,
            crate::recovery::RecoveryOutcome::Failed => CircuitEvent::RecoveryFailed,
        };
        self.circuit_event(ev, self.clock.now_mono());
        self.publish();
    }

    /// Any other circuit input (iterations, device errors, probe results, controller failure).
    pub fn on_circuit_event(&mut self, ev: CircuitEvent) -> Option<CircuitTransition> {
        let t = self.circuit.on_event(ev, self.clock.now_mono());
        if t.is_some_and(|t| CircuitBreaker::baseline_reset_due(&t)) {
            self.baseline_step_s = None;
        }
        self.publish();
        t
    }

    fn circuit_event(&mut self, ev: CircuitEvent, now: Duration) {
        if let Some(t) = self.circuit.on_event(ev, now)
            && CircuitBreaker::baseline_reset_due(&t)
        {
            self.baseline_step_s = None;
        }
    }

    fn apply_plan(&mut self) {
        let plan = plan_with(
            self.machine.state(),
            &self.limits,
            self.cfg.recovery.survival_liveness,
        );
        if plan != self.plan {
            tracing::info!(
                event = "throttle_plan_changed",
                reason = plan.state.as_str(),
                batch_growth_limit = plan.batch_growth_limit.map_or(-1, i64::from),
                prefill_budget_fraction = plan.prefill_budget_fraction,
                prefill_chunk_tokens = plan.prefill_chunk_tokens.map_or(0, u64::from),
            );
            publish_plan(&plan, &self.metrics);
            self.plan = plan;
        }
        let kv = self.evaluator.thresholds()[&PressureSignal::KvUtilization];
        apply_reclaim(&self.plan, self.reclaimer.as_ref(), &kv, &self.metrics);
    }

    fn publish(&mut self) {
        for s in &self.last_signals {
            self.metrics
                .pressure_signal
                .get_or_create(&SignalLabel {
                    signal: s.signal.as_str(),
                })
                .set(s.value);
            self.metrics
                .pressure_signal_level
                .get_or_create(&SignalLabel {
                    signal: s.signal.as_str(),
                })
                .set(i64::from(s.level.as_u8()));
        }
        self.metrics
            .exhaustion_horizon_seconds
            .set(self.last_horizon);
        let state = self.machine.state();
        // The live signal with the highest level above GREEN (ties: first in PressureSignal
        // order); None when every live signal is GREEN.
        let dominant = self
            .last_signals
            .iter()
            .filter(|s| !s.stale && s.level > PressureState::Green)
            .max_by(|a, b| a.level.cmp(&b.level).then(b.signal.cmp(&a.signal)))
            .map(|s| s.signal);
        let document = PressureDocument {
            enabled: self.cfg.enabled,
            state,
            since: rfc3339_millis(self.machine.since_wall()),
            dominant_signal: dominant,
            exhaustion_horizon_seconds: self.last_horizon.is_finite().then_some(self.last_horizon),
            signals: self.last_signals.clone(),
            throttle: ThrottleDoc::from(&self.plan),
            memory: vec![memory_doc(&self.budget, &self.ledger, self.reserve.held())],
            admission: AdmissionDoc {
                queued: self.metrics.admission_queue_depth.get().max(0) as u32,
                max_queue: self.cfg.admission.max_queue,
                decisions: decisions_doc(&self.metrics),
            },
            circuit: CircuitDoc {
                state: self.circuit.state(),
                since: rfc3339_millis(
                    self.clock
                        .now_wall()
                        .checked_sub(self.clock.now_mono().saturating_sub(self.circuit.since()))
                        .unwrap_or(std::time::UNIX_EPOCH),
                ),
                last_reason: self.circuit.last_reason(),
            },
            transitions: self.machine.history().map(TransitionDoc::from).collect(),
        };
        self.handle.snap.store(Arc::new(Snapshot {
            state,
            circuit: self.circuit.state(),
            throttle: self.plan,
            circuit_retry_after_secs: self.circuit.retry_after_secs(),
            fatal: self.circuit.is_fatal(),
            drain_expired: self.circuit.drain_expired(),
            document,
        }));
    }
}

fn empty_document(enabled: bool, plan: &ThrottlePlan) -> PressureDocument {
    PressureDocument {
        enabled,
        state: PressureState::Green,
        since: rfc3339_millis(std::time::UNIX_EPOCH),
        dominant_signal: None,
        exhaustion_horizon_seconds: None,
        signals: Vec::new(),
        throttle: ThrottleDoc::from(plan),
        memory: Vec::new(),
        admission: AdmissionDoc::default(),
        circuit: CircuitDoc {
            state: CircuitState::Healthy,
            since: rfc3339_millis(std::time::UNIX_EPOCH),
            last_reason: None,
        },
        transitions: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetInputs, PoolKind, compute_budget};
    use crate::reserve::ReserveAllocator;
    use turbine_core::clock::FakeClock;
    use turbine_core::telemetry::{DeviceSample, HostSample, LedgerSample, SourceStatus};
    use turbine_core::types::{DeviceId, MemoryKind};

    struct Always;
    impl ReserveAllocator for Always {
        fn allocate(&mut self, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn free(&mut self) {}
    }

    const GIB: u64 = 1 << 30;

    fn controller(enabled: bool) -> (PressureController, ControllerHandle, FakeClock) {
        let cfg = ReliabilityConfig {
            enabled,
            ..ReliabilityConfig::default()
        };
        let inputs = BudgetInputs {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            measured_free_bytes: Some(30 * GIB),
            already_held_bytes: 0,
            host_mem_available_bytes: Some(100 * GIB),
            weights_bytes: 6 * GIB,
            kv_bytes_per_token: 114_688,
            max_seq_len: 8192,
            block_bytes: 1_835_008,
        };
        let budget = compute_budget(&inputs, &cfg, None).unwrap();
        let ledger = Ledger::new(&budget);
        let metrics = ReliabilityMetrics::unregistered();
        let reserve = EmergencyReserve::acquire(
            DeviceId(0),
            cfg.emergency_vram_reserve.0,
            &ledger,
            Box::new(Always),
            metrics.clone(),
        )
        .unwrap();
        let clock = FakeClock::new(Duration::ZERO);
        let limits = SchedulerLimits {
            prefill_chunk_tokens: 2048,
            block_tokens: 16,
        };
        let (c, h) = PressureController::new(
            &cfg,
            limits,
            budget,
            ledger,
            reserve,
            Arc::new(NoReclaim),
            metrics,
            Arc::new(clock.clone()),
        );
        (c, h, clock)
    }

    /// Deterministic simulated telemetry: a dedicated device at 40 °C, 64 GiB MemAvailable, no PSI or swap.
    fn sample(kv: f64, queue: f64) -> TelemetrySample {
        TelemetrySample {
            at_mono_ns: 0,
            host: HostSample {
                mem_available_bytes: Some(64 * GIB),
                swap_total_bytes: Some(0),
                swap_free_bytes: Some(0),
                pswpin_total: Some(0),
                psi_memory_some_avg10: Some(0.0),
                status: SourceStatus::Ok,
            },
            devices: vec![DeviceSample {
                memory_used_bytes: Some(20 * GIB),
                memory_free_bytes: Some(12 * GIB),
                temperature_c: Some(40.0),
                slowdown_temperature_c: Some(90.0),
                clock_mhz: Some(2350),
                ..DeviceSample::empty(DeviceId(0), SourceStatus::Ok)
            }],
            ledger: LedgerSample {
                kv_utilization: kv,
                queue_fill: queue,
            },
            storage: None,
        }
    }

    #[test]
    fn simulated_overload_cycle() {
        let (mut c, h, clock) = controller(true);
        let stats = EngineStats {
            block_tokens: 16,
            free_kv_blocks: 1000,
            ..EngineStats::default()
        };
        let run = |c: &mut PressureController, secs: u64, kv: f64, q: f64| {
            for _ in 0..secs * 10 {
                clock.advance(Duration::from_millis(100));
                c.tick(&sample(kv, q), &stats);
            }
        };
        run(&mut c, 2, 0.5, 0.0);
        assert_eq!(h.state(), PressureState::Green);
        run(&mut c, 1, 0.85, 0.3);
        assert_eq!(h.state(), PressureState::Orange);
        assert_eq!(h.throttle().prefill_chunk_tokens, Some(1024));
        assert_eq!(
            h.document().dominant_signal,
            Some(PressureSignal::KvUtilization)
        );
        // Device OOM: SURVIVAL at once, reserve released and visible in the document.
        assert_eq!(c.on_oom(), 2 * GIB);
        assert_eq!(h.state(), PressureState::Survival);
        let doc = h.document();
        assert!(!doc.memory[0].emergency_reserve_held);
        assert_eq!(
            doc.memory[0]
                .pools
                .iter()
                .find(|p| p.name == PoolKind::Reserve)
                .unwrap()
                .used_bytes,
            0
        );
        c.on_recovery(crate::recovery::RecoveryOutcome::Recovered { retries: 1 });
        assert_eq!(h.circuit(), CircuitState::Degraded);
        // Load gone: allocation_failure holds SURVIVAL for one dwell, then one level per dwell; DEGRADED keeps
        // the floor at YELLOW until the circuit window (60 s) passes without a trigger.
        run(&mut c, 45, 0.1, 0.0);
        assert_eq!(h.state(), PressureState::Yellow, "DEGRADED floor");
        assert!(
            h.document().memory[0].emergency_reserve_held,
            "re-acquired before dropping below RED"
        );
        run(&mut c, 40, 0.1, 0.0);
        assert_eq!(h.circuit(), CircuitState::Healthy);
        assert_eq!(h.state(), PressureState::Green);
        let doc = serde_json::to_value(h.document()).unwrap();
        for key in [
            "enabled",
            "state",
            "since",
            "dominant_signal",
            "exhaustion_horizon_seconds",
            "signals",
            "throttle",
            "memory",
            "admission",
            "circuit",
            "transitions",
        ] {
            assert!(doc.get(key).is_some(), "pressure document key {key}");
        }
        assert_eq!(doc["state"], "GREEN");
        assert!(doc["transitions"].as_array().unwrap().len() >= 6);
    }

    #[test]
    fn disabled_stays_green() {
        let (mut c, h, clock) = controller(false);
        for _ in 0..50 {
            clock.advance(Duration::from_millis(100));
            c.tick(
                &sample(0.99, 1.0),
                &EngineStats {
                    block_tokens: 16,
                    ..EngineStats::default()
                },
            );
        }
        assert_eq!(h.state(), PressureState::Green);
        let doc = serde_json::to_value(h.document()).unwrap();
        assert_eq!(doc["enabled"], false);
        assert_eq!(doc["state"], "GREEN");
    }
}
