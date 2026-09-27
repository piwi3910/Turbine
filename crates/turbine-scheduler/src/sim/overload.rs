//! Deterministic overload simulation (P3 S-17): the real `Scheduler` behind the P3
//! `AdmissionGate`, the `PressureController` and `RecoveryController`, a `BlockPool` over host
//! memory whose blocks are paid from the reservation ledger, the P2 cost-model executor and a
//! `FakeClock`. Telemetry is synthesised every `reliability.telemetry.interval` from the ledger
//! and the admission queue; device OOM is injected per iteration attempt and walks the same
//! `on_oom → RecoveryStep → on_recovery` path the engine uses. Circuit probes run as internal
//! 16-token requests that bypass the admission queue.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use smallvec::smallvec;
use turbine_core::clock::{Clock, FakeClock};
use turbine_core::config::ReliabilityConfig;
use turbine_core::request::FinishReason;
use turbine_core::telemetry::{
    DeviceSample, HostSample, LedgerSample, SourceStatus, TelemetrySample,
};
use turbine_core::types::{
    CircuitState, DType, DeviceId, KvLayout, MemoryKind, PressureSignal, PressureState, RequestId,
    SeqId,
};
use turbine_kv::{BlockPool, BlockPoolConfig, L0Reclaimer};
use turbine_reliability::admission::{Admission, AdmissionParams, AdmissionQueue, Calibration};
use turbine_reliability::budget::{DeviceBudget, PoolKind};
use turbine_reliability::circuit::CircuitEvent;
use turbine_reliability::controller::{ControllerHandle, EngineStats, PressureController};
use turbine_reliability::ledger::{Ledger, PoolUsage};
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::recovery::{RecoveryController, RecoveryOutcome, RecoveryStep};
use turbine_reliability::reserve::{EmergencyReserve, ReserveAllocator};
use turbine_reliability::signals::effective_thresholds;
use turbine_reliability::step_window::{DecodeStepWindow, StepSample};
use turbine_reliability::throttle::SchedulerLimits;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

use crate::gate::AdmissionGate;
use crate::request::{CancelReason, SchedRequest};
use crate::scheduler::{
    BatchKind, IterationFailure, IterationLimits, IterationOutcome, IterationPlan, Scheduler,
    SchedulerParams, SubmitError,
};
use crate::sim::executor::{CostModel, SimExecutor};

const DEVICE: DeviceId = DeviceId(0);
/// 2 layers × 2 KV heads × 64 dims, BF16: 1,024 bytes per token, 16,384 per 16-token block.
const LAYOUT: KvLayout = KvLayout {
    num_layers: 2,
    num_kv_heads: 2,
    head_dim: 64,
    dtype: DType::BF16,
    block_tokens: 16,
};
/// Circuit probe: a fixed short prompt and 16 greedy tokens.
const PROBE_PROMPT: u32 = 8;
const PROBE_TOKENS: u32 = 16;
/// Smoothing of the decode step-time estimates (baseline, per-sequence rate).
const STEP_ALPHA: f64 = 0.1;

/// Everything the overload harness is built from.
#[derive(Clone, Debug)]
pub struct OverloadConfig {
    pub seed: u64,
    /// L0 pool size; the ledger's `kv` pool holds exactly these blocks.
    pub pool_blocks: u32,
    pub max_seq_len: u32,
    pub params: SchedulerParams,
    pub cost: CostModel,
    pub reliability: ReliabilityConfig,
    /// Arrival rate of `run_load` as a multiple of `service_rate()`.
    pub rate_multiple: f64,
    /// Inclusive prompt-length range of `run_load` arrivals.
    pub prompt_range: (u32, u32),
    /// Inclusive `max_tokens` range of `run_load` arrivals (the model generates all of them).
    pub max_tokens_range: (u32, u32),
    /// A `scheduling_policy` registry name (Phase 2m), `default` by default: the overload tests
    /// run over every registered policy.
    pub policy: &'static str,
    /// Hold the circuit in DEGRADED while `run_load` submits (a `telemetry_stale` event on every
    /// tick, as a stale vendor library would).
    pub degraded_during_load: bool,
}

impl Default for OverloadConfig {
    fn default() -> Self {
        let reliability = ReliabilityConfig::default();
        OverloadConfig {
            seed: 1,
            pool_blocks: 4096,
            max_seq_len: 8192,
            params: SchedulerParams {
                max_running_requests: 64,
                max_batch_tokens: 8192,
                prefill_chunk_tokens: 2048,
                max_queued_requests: 256,
                chunked_prefill: true,
                block_tokens: LAYOUT.block_tokens,
                free_watermark: 0.01,
                max_seq_len: 8192,
                queue_timeout: reliability.admission.queue_timeout.0,
            },
            cost: CostModel {
                per_prefill_token_s: 0.000_2,
                per_decode_step_s: 0.02,
                per_seq_s: 0.000_5,
            },
            reliability,
            rate_multiple: 10.0,
            prompt_range: (64, 6000),
            max_tokens_range: (16, 1024),
            policy: "default",
            degraded_during_load: false,
        }
    }
}

impl OverloadConfig {
    /// Analytic service capacity (requests/s) for `run_load`'s length ranges: the requests the
    /// KV pool (worst-case reservations) and `max_running` let run at once, divided by the time
    /// one request occupies its slot (its prefill plus its decode steps in a full batch).
    pub fn service_rate(&self) -> f64 {
        let mean = |r: (u32, u32)| (f64::from(r.0) + f64::from(r.1)) / 2.0;
        let (prompt, output) = (mean(self.prompt_range), mean(self.max_tokens_range));
        let blocks = ((prompt + output) / f64::from(LAYOUT.block_tokens)).ceil();
        let concurrency = (f64::from(self.pool_blocks) / blocks)
            .floor()
            .clamp(1.0, f64::from(self.params.max_running_requests));
        let step = self.cost.per_decode_step_s + self.cost.per_seq_s * concurrency;
        let per_request = prompt * self.cost.per_prefill_token_s + output * step;
        concurrency / per_request
    }
}

/// How a simulated request ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Cancelled,
    /// Refused or dropped before running, with its reject-table `code`.
    Rejected(String),
    /// Failed while running, with its error `code`.
    Failed(String),
}

/// One `step_iteration`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IterationReport {
    /// Set when the iteration ended a recovery (after at least one device OOM).
    pub recovery: Option<RecoveryOutcome>,
    /// Sequences in each attempt of the iteration (one entry without an OOM).
    pub attempt_batch_sizes: Vec<usize>,
    /// Requests failed `resource_exhausted` because the retries were exhausted.
    pub failed_requests: Vec<RequestId>,
}

/// Summary of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct OverloadReport {
    pub kv_capacity_bytes: u64,
    /// Highest `used + reserved` of the `kv` ledger pool seen after any plan or completion.
    pub max_kv_committed_plus_reserved: u64,
    /// Sequences preempted while the pressure state was below SURVIVAL (must stay 0).
    pub preempted_below_survival: u32,
    /// Plans in RED, after a plan in RED, whose admitted count (running + waiting with a KV
    /// reservation) rose above the previous plan's (must stay 0: RED only refills finished
    /// slots).
    pub red_growth: u32,
    /// Outcome of every client request that has one, in submission order.
    pub outcomes: Vec<Outcome>,
    /// Client requests still queued or running.
    pub unfinished: usize,
    pub max_queue_len: usize,
    /// Every pressure state observed on a controller tick.
    pub states: BTreeSet<PressureState>,
    /// First time after `run_load`'s load stopped that the state was GREEN and the circuit
    /// HEALTHY, relative to the stop.
    pub green_after_stop: Option<Duration>,
    pub final_circuit: CircuitState,
    /// Longest stretch in which the engine ran nothing although requests waited in the
    /// admission queue, below SURVIVAL and with the circuit admitting: the device idling on
    /// queued work (the 2026-09-27 soak's stall).
    pub max_idle_with_queue: Duration,
}

/// The emergency reserve of a simulated device: nothing to allocate.
struct SimReserve;

impl ReserveAllocator for SimReserve {
    fn allocate(&mut self, _bytes: u64) -> Result<(), String> {
        Ok(())
    }
    fn free(&mut self) {}
}

struct Track {
    seq: SeqId,
    max_new: u32,
    generated: u32,
    finished: bool,
    probe: bool,
}

/// One executed iteration: start, duration and the requests it decoded.
struct Step {
    start: Duration,
    duration: Duration,
    decoded: Vec<RequestId>,
}

/// The harness. Every method advances only virtual time.
pub struct OverloadSim {
    cfg: OverloadConfig,
    clock: FakeClock,
    sched: Scheduler,
    pool: BlockPool,
    exec: SimExecutor,
    ledger: Arc<Ledger>,
    controller: PressureController,
    handle: ControllerHandle,
    recovery: RecoveryController,
    rng: ChaCha8Rng,
    next_id: u64,
    tracks: HashMap<RequestId, Track>,
    by_seq: HashMap<SeqId, RequestId>,
    outcomes: HashMap<RequestId, Outcome>,
    order: Vec<RequestId>,
    tails: HashMap<RequestId, Vec<String>>,
    steps: Vec<Step>,
    oom_attempts: u32,
    next_tick: Duration,
    probe: Option<RequestId>,
    baseline_step_s: Option<f64>,
    decode_step_s: f64,
    decode_steps: DecodeStepWindow,
    iterations: u64,
    load_stop: Option<Duration>,
    // report
    max_kv: u64,
    preempted_below_survival: u32,
    red_growth: u32,
    last_admitted: usize,
    last_plan_state: PressureState,
    max_queue_len: usize,
    states: BTreeSet<PressureState>,
    green_after_stop: Option<Duration>,
    idle_since: Option<Duration>,
    max_idle_with_queue: Duration,
    seen_iterations: u64,
}

impl OverloadSim {
    pub fn new(cfg: OverloadConfig) -> OverloadSim {
        let clock = FakeClock::new(Duration::ZERO);
        let clock_dyn: Arc<dyn Clock> = Arc::new(clock.clone());
        let rel = &cfg.reliability;
        let block_bytes = LAYOUT.block_bytes();
        let kv_bytes = u64::from(cfg.pool_blocks) * block_bytes;
        let reserve_bytes = rel.emergency_vram_reserve.0;
        let workspace = rel.memory.workspace_bytes.0;
        let budget = DeviceBudget {
            device: DEVICE,
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: kv_bytes + reserve_bytes + workspace,
            pools: vec![
                (PoolKind::Weights, 0),
                (PoolKind::Kv, kv_bytes),
                (PoolKind::Workspace, workspace),
                (PoolKind::Runtime, 0),
                (PoolKind::Reserve, reserve_bytes),
            ],
        };
        let metrics = ReliabilityMetrics::unregistered();
        let ledger = Ledger::new(&budget);
        ledger.set_metrics(metrics.clone());
        let reserve = EmergencyReserve::acquire(
            DEVICE,
            reserve_bytes,
            &ledger,
            Box::new(SimReserve),
            metrics.clone(),
        )
        .expect("the reserve pool holds the emergency reserve");
        let (controller, handle) = PressureController::new(
            rel,
            SchedulerLimits {
                prefill_chunk_tokens: cfg.params.prefill_chunk_tokens,
                block_tokens: cfg.params.block_tokens,
            },
            budget,
            Arc::clone(&ledger),
            reserve,
            Arc::new(L0Reclaimer),
            metrics.clone(),
            Arc::clone(&clock_dyn),
        );
        let admission = Admission::new(
            AdmissionParams {
                device: DEVICE,
                adaptive: rel.adaptive_admission,
                max_queue: rel.admission.max_queue,
                large_prefill_tokens: rel.admission.large_prefill_tokens,
                block_bytes,
                prefill_chunk_tokens: cfg.params.prefill_chunk_tokens,
                workspace_bytes_per_token: 0,
                calibration: Calibration {
                    prefill_tokens_per_s: 1.0 / cfg.cost.per_prefill_token_s.max(1e-9),
                    decode_step_s: cfg.cost.per_decode_step_s,
                },
            },
            Arc::clone(&ledger),
            metrics.clone(),
        )
        .with_kv_headroom(effective_thresholds(&rel.pressure)[&PressureSignal::KvUtilization]);
        let queue = AdmissionQueue::new(
            rel.admission.max_queue,
            rel.admission.queue_timeout.0,
            rel.admission.max_bypass,
        );
        let gate = AdmissionGate::new(
            admission,
            queue,
            handle.clone(),
            Arc::clone(&clock_dyn),
            cfg.params.max_running_requests,
        );
        let mut params = cfg.params;
        params.max_seq_len = cfg.max_seq_len;
        let sched = Scheduler::new(params, Arc::clone(&clock_dyn))
            .with_policy(
                crate::policy::registry()
                    .get(cfg.policy)
                    .unwrap_or_else(|| {
                        panic!("scheduling policy {:?} is not registered", cfg.policy)
                    }),
            )
            .with_gate(gate);
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DEVICE, kv_bytes);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: LAYOUT,
                num_blocks: cfg.pool_blocks,
            },
            mem,
        )
        .expect("the host pool fits its memory")
        .with_ledger(Arc::clone(&ledger), DEVICE);
        let recovery = RecoveryController::new(&rel.recovery, metrics);
        OverloadSim {
            rng: ChaCha8Rng::seed_from_u64(cfg.seed),
            exec: SimExecutor { cost: cfg.cost },
            decode_step_s: cfg.cost.per_decode_step_s,
            decode_steps: DecodeStepWindow::new(),
            cfg,
            clock,
            sched,
            pool,
            ledger,
            controller,
            handle,
            recovery,
            next_id: 0,
            tracks: HashMap::new(),
            by_seq: HashMap::new(),
            outcomes: HashMap::new(),
            order: Vec::new(),
            tails: HashMap::new(),
            steps: Vec::new(),
            oom_attempts: 0,
            next_tick: Duration::ZERO,
            probe: None,
            baseline_step_s: None,
            iterations: 0,
            load_stop: None,
            max_kv: 0,
            preempted_below_survival: 0,
            red_growth: 0,
            last_admitted: 0,
            last_plan_state: PressureState::Green,
            max_queue_len: 0,
            states: BTreeSet::new(),
            green_after_stop: None,
            idle_since: None,
            max_idle_with_queue: Duration::ZERO,
            seen_iterations: 0,
        }
    }

    pub fn now(&self) -> Duration {
        self.clock.now_mono()
    }

    pub fn config(&self) -> &OverloadConfig {
        &self.cfg
    }

    pub fn handle(&self) -> &ControllerHandle {
        &self.handle
    }

    /// Submit a client request now: `prompt` tokens, generating `max_tokens` (the whole
    /// remaining context when `None`). A refused request gets its `Rejected` outcome at once.
    pub fn submit_now(&mut self, prompt: u32, max_tokens: Option<u32>) -> RequestId {
        let max_new = max_tokens.unwrap_or(self.cfg.max_seq_len.saturating_sub(prompt));
        let (id, r) = self.new_request(prompt, max_new);
        self.order.push(id);
        self.track(id, max_new, false);
        match self.sched.submit(r, self.pool.total_blocks()) {
            Ok(()) => {}
            Err(e) => self.finish(id, Outcome::Rejected(reject_code(e).to_string())),
        }
        self.max_queue_len = self.max_queue_len.max(self.gate_len());
        id
    }

    /// The client disconnects.
    pub fn cancel(&mut self, id: RequestId) {
        self.sched.cancel(id, CancelReason::ClientDisconnect);
    }

    /// The next `n` iteration attempts fail with a device out-of-memory error.
    pub fn inject_oom_attempts(&mut self, n: u32) {
        self.oom_attempts += n;
    }

    /// Plan, execute (with bounded OOM recovery) and complete one iteration; when there is
    /// nothing to run, advance to the next telemetry tick instead.
    pub fn step_iteration(&mut self) -> IterationReport {
        let now = self.now();
        self.tick_until(now);
        self.maybe_probe();
        let state = self.handle.state();
        let limits = IterationLimits::from(&self.handle.throttle());
        let mut plan = self.sched.plan(&mut self.pool, &limits);
        if state < PressureState::Survival {
            self.preempted_below_survival += plan.preempted.len() as u32;
        }
        // Arrivals between two RED plans all queue, so the admitted count may only return to
        // the previous plan's after completions.
        let admitted = self.sched.admitted_count();
        // The work-conserving floor (0 → 1) is not growth.
        if state == PressureState::Red
            && self.last_plan_state == PressureState::Red
            && admitted > self.last_admitted.max(1)
        {
            self.red_growth += 1;
        }
        self.last_admitted = admitted;
        self.last_plan_state = state;
        self.observe_ledger();
        for (id, reason) in plan.dropped.clone() {
            let outcome = match reason {
                CancelReason::QueueTimeout => Outcome::Rejected("queue_timeout".into()),
                CancelReason::CircuitOpen => Outcome::Rejected("circuit_open".into()),
                CancelReason::Overloaded => Outcome::Rejected("overloaded".into()),
                _ => Outcome::Cancelled,
            };
            self.finish(id, outcome);
        }
        let mut report = IterationReport::default();
        let idle_on_queue = plan.is_empty()
            && self.gate_len() > 0
            && state < PressureState::Survival
            && !self.handle.circuit().blocks_readiness();
        if idle_on_queue {
            let since = *self.idle_since.get_or_insert(now);
            self.max_idle_with_queue = self.max_idle_with_queue.max(now - since);
        } else {
            self.idle_since = None;
        }
        if plan.is_empty() {
            self.sched.complete(
                &mut self.pool,
                IterationOutcome {
                    iteration: plan.iteration,
                    ..IterationOutcome::default()
                },
            );
            let next = self.next_tick.max(now + Duration::from_millis(1));
            self.clock.set(next);
            return report;
        }
        let batch: Vec<RequestId> = self.plan_requests(&plan);
        loop {
            report.attempt_batch_sizes.push(plan.items.len());
            if self.oom_attempts == 0 {
                break;
            }
            self.oom_attempts -= 1;
            self.controller.on_oom();
            match self.recovery.on_oom(plan.items.len()) {
                RecoveryStep::Retry {
                    backoff,
                    batch_limit,
                    ..
                } => {
                    self.clock.advance(backoff);
                    self.sched
                        .shrink_plan(&mut self.pool, &mut plan, batch_limit);
                }
                RecoveryStep::GiveUp => {
                    self.sched.complete(
                        &mut self.pool,
                        IterationOutcome {
                            iteration: plan.iteration,
                            failed: Some(IterationFailure {
                                message: "resource_exhausted".into(),
                            }),
                            ..IterationOutcome::default()
                        },
                    );
                    self.sched.fail_requests(&mut self.pool, &batch);
                    for id in &batch {
                        self.fail(*id);
                    }
                    self.controller.on_recovery(RecoveryOutcome::Failed);
                    report.recovery = Some(RecoveryOutcome::Failed);
                    report.failed_requests = batch;
                    self.observe_ledger();
                    return report;
                }
            }
        }
        let start = self.now();
        let duration = self.exec.duration(&plan);
        let outcome = self.execute(&plan);
        self.record_step(&plan, start, duration);
        self.clock.advance(duration);
        if let Some(g) = self.sched.gate_mut() {
            let prefill_tokens = plan.prefill_tokens();
            let prefill_s = f64::from(prefill_tokens) * self.cfg.cost.per_prefill_token_s;
            let decode = (plan.decode_tokens() > 0).then_some(duration.as_secs_f64());
            g.admission_mut()
                .observe_iteration(prefill_tokens, prefill_s, decode);
        }
        self.sched.complete(&mut self.pool, outcome);
        self.iterations += 1;
        if let Some(o) = self.recovery.on_success() {
            self.controller.on_recovery(o);
            report.recovery = Some(o);
        }
        self.observe_ledger();
        report
    }

    /// Step until `d` of virtual time has passed.
    pub fn run_for(&mut self, d: Duration) {
        let end = self.now() + d;
        while self.now() < end {
            self.step_iteration();
        }
    }

    /// Step until every client request has an outcome, or `limit` of virtual time has passed.
    pub fn run_until_done(&mut self, limit: Duration) {
        let end = self.now() + limit;
        while self.now() < end && self.unfinished() > 0 {
            self.step_iteration();
        }
    }

    /// Open-loop Poisson arrivals at `rate_multiple × service_rate()` for `load` (lengths drawn
    /// uniformly from the configured ranges with the seeded PRNG), then `quiet` with none.
    pub fn run_load(&mut self, load: Duration, quiet: Duration) -> OverloadReport {
        let rate = self.cfg.rate_multiple * self.cfg.service_rate();
        let start = self.now();
        let stop = start + load;
        let end = stop + quiet;
        self.load_stop = Some(stop);
        let mut next_arrival = start + self.inter_arrival(rate);
        while self.now() < end {
            while next_arrival <= self.now() && next_arrival < stop {
                let prompt = self.draw(self.cfg.prompt_range);
                let max_tokens = self.draw(self.cfg.max_tokens_range);
                self.submit_now(prompt, Some(max_tokens));
                next_arrival += self.inter_arrival(rate);
            }
            self.step_iteration();
        }
        self.report()
    }

    pub fn outcome(&self, id: RequestId) -> Option<&Outcome> {
        self.outcomes.get(&id)
    }

    /// The last SSE lines a streaming client of `id` received (empty while it runs or when it
    /// was refused before streaming).
    pub fn stream_tail(&self, id: RequestId) -> Vec<String> {
        self.tails.get(&id).cloned().unwrap_or_default()
    }

    pub fn running_ids(&self) -> Vec<RequestId> {
        self.sched.running_ids()
    }

    pub fn queued_ids(&self) -> Vec<RequestId> {
        self.sched.queued_ids()
    }

    /// Waiting in the admission gate's queue (not admitted, no reservation).
    pub fn is_queued(&self, id: RequestId) -> bool {
        self.sched.gate().is_some_and(|g| g.contains(id))
    }

    pub fn pool_used_blocks(&self) -> u32 {
        self.pool.used_blocks()
    }

    pub fn kv_usage(&self) -> PoolUsage {
        self.ledger.usage(DEVICE, PoolKind::Kv)
    }

    pub fn workspace_usage(&self) -> PoolUsage {
        self.ledger.usage(DEVICE, PoolKind::Workspace)
    }

    /// No `kv` or `workspace` bytes are reserved or committed.
    pub fn ledger_idle(&self) -> bool {
        let kv = self.kv_usage();
        let ws = self.workspace_usage();
        kv.used + kv.reserved + ws.used + ws.reserved == 0
    }

    /// Mean duration of the iterations started in `[from, to)` that decoded any of `ids`.
    pub fn mean_step_time(&self, ids: &[RequestId], from: Duration, to: Duration) -> Option<f64> {
        let (sum, n) = self
            .steps
            .iter()
            .filter(|s| s.start >= from && s.start < to)
            .filter(|s| s.decoded.iter().any(|d| ids.contains(d)))
            .fold((0.0, 0u32), |(sum, n), s| {
                (sum + s.duration.as_secs_f64(), n + 1)
            });
        (n > 0).then(|| sum / f64::from(n))
    }

    pub fn report(&self) -> OverloadReport {
        OverloadReport {
            kv_capacity_bytes: self.kv_usage().capacity,
            max_kv_committed_plus_reserved: self.max_kv,
            preempted_below_survival: self.preempted_below_survival,
            red_growth: self.red_growth,
            outcomes: self
                .order
                .iter()
                .filter_map(|id| self.outcomes.get(id).cloned())
                .collect(),
            unfinished: self.unfinished(),
            max_queue_len: self.max_queue_len,
            states: self.states.clone(),
            green_after_stop: self.green_after_stop,
            final_circuit: self.handle.circuit(),
            max_idle_with_queue: self.max_idle_with_queue,
        }
    }

    // ---- internals -------------------------------------------------------------------

    fn new_request(&mut self, prompt: u32, max_new: u32) -> (RequestId, SchedRequest) {
        let k = self.next_id;
        self.next_id += 1;
        let id = RequestId(uuid::Uuid::from_u128(u128::from(k) + 1));
        let mut r = SchedRequest::new(
            id,
            smallvec![SeqId(k)],
            prompt,
            max_new,
            self.cfg.params.block_tokens,
        );
        r.arrival = self.now();
        (id, r)
    }

    fn track(&mut self, id: RequestId, max_new: u32, probe: bool) {
        let seq = SeqId(self.next_id - 1);
        self.by_seq.insert(seq, id);
        self.tracks.insert(
            id,
            Track {
                seq,
                max_new,
                generated: 0,
                finished: false,
                probe,
            },
        );
    }

    fn unfinished(&self) -> usize {
        self.tracks.values().filter(|t| !t.probe).count()
    }

    fn gate_len(&self) -> usize {
        self.sched.gate().map_or(0, AdmissionGate::queue_len)
    }

    /// Record `id`'s outcome and forget it (probes report to the circuit breaker instead).
    fn finish(&mut self, id: RequestId, outcome: Outcome) {
        let Some(t) = self.tracks.remove(&id) else {
            return;
        };
        self.by_seq.remove(&t.seq);
        if t.probe {
            self.probe = None;
            let ev = match outcome {
                Outcome::Completed => CircuitEvent::ProbeSucceeded {
                    latency_ratio: self.probe_ratio(id),
                },
                _ => CircuitEvent::ProbeFailed,
            };
            self.controller.on_circuit_event(ev);
            return;
        }
        if outcome == Outcome::Completed {
            self.tails.insert(
                id,
                vec![
                    r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#
                        .to_string(),
                    "data: [DONE]".to_string(),
                ],
            );
        }
        self.outcomes.insert(id, outcome);
    }

    /// Retries exhausted: the stream ends with the error event, then `[DONE]` (C-3).
    fn fail(&mut self, id: RequestId) {
        let event = serde_json::json!({
            "error": {
                "message": "device out of memory: retries exhausted",
                "type": "server_error",
                "code": "resource_exhausted",
            }
        });
        let is_client = self.tracks.get(&id).is_some_and(|t| !t.probe);
        if is_client {
            self.tails.insert(
                id,
                vec![format!("data: {event}"), "data: [DONE]".to_string()],
            );
        }
        self.finish(id, Outcome::Failed("resource_exhausted".into()));
    }

    fn plan_requests(&self, plan: &IterationPlan) -> Vec<RequestId> {
        let mut ids: Vec<RequestId> = Vec::new();
        for item in &plan.items {
            if let Some(id) = self.by_seq.get(&item.seq)
                && !ids.contains(id)
            {
                ids.push(*id);
            }
        }
        ids
    }

    /// The "model": completed prefills and decodes sample one token; a sequence stops after
    /// `max_new` tokens.
    fn execute(&mut self, plan: &IterationPlan) -> IterationOutcome {
        let mut appended = Vec::new();
        let mut finished = Vec::new();
        let mut done = Vec::new();
        for item in &plan.items {
            let samples = match item.kind {
                BatchKind::Decode => true,
                BatchKind::Prefill { start, len } => {
                    self.sched.prefill_target(item.seq) == Some(start + len)
                }
            };
            if !samples {
                continue;
            }
            let Some(id) = self.by_seq.get(&item.seq).copied() else {
                continue;
            };
            let t = self.tracks.get_mut(&id).expect("tracked request");
            if t.finished {
                continue;
            }
            t.generated += 1;
            appended.push((item.seq, 1));
            if t.generated >= t.max_new {
                t.finished = true;
                finished.push((item.seq, FinishReason::Length));
                done.push(id);
            }
        }
        for id in done {
            self.finish(id, Outcome::Completed);
        }
        IterationOutcome {
            iteration: plan.iteration,
            finished,
            appended,
            failed: None,
        }
    }

    fn record_step(&mut self, plan: &IterationPlan, start: Duration, duration: Duration) {
        let decoded: Vec<RequestId> = plan
            .items
            .iter()
            .filter(|i| i.kind == BatchKind::Decode)
            .filter_map(|i| self.by_seq.get(&i.seq).copied())
            .collect();
        if decoded.is_empty() {
            return;
        }
        let secs = duration.as_secs_f64();
        self.decode_step_s = STEP_ALPHA * secs + (1.0 - STEP_ALPHA) * self.decode_step_s;
        let calm = self.handle.state() == PressureState::Green
            && self.handle.circuit() == CircuitState::Healthy
            && self.probe.is_none();
        self.decode_steps.observe(
            StepSample {
                prefill_tokens: plan.prefill_tokens(),
                rows: plan.decode_tokens(),
                context_tokens: plan.decode_context_tokens(),
                secs,
            },
            calm,
        );
        if calm {
            self.baseline_step_s = Some(
                self.baseline_step_s
                    .map_or(secs, |b| STEP_ALPHA * secs + (1.0 - STEP_ALPHA) * b),
            );
        }
        self.steps.push(Step {
            start,
            duration,
            decoded,
        });
    }

    /// Probe latency relative to the GREEN baseline decode step (1.0 without a baseline).
    fn probe_ratio(&self, probe: RequestId) -> f64 {
        let (sum, n) = self
            .steps
            .iter()
            .rev()
            .take_while(|s| s.decoded.contains(&probe) || s.decoded.is_empty())
            .filter(|s| s.decoded.contains(&probe))
            .fold((0.0, 0u32), |(sum, n), s| {
                (sum + s.duration.as_secs_f64(), n + 1)
            });
        match (self.baseline_step_s, n) {
            (Some(base), n) if n > 0 && base > 0.0 => sum / f64::from(n) / base,
            _ => 1.0,
        }
    }

    /// While the circuit is PROBING, keep one internal probe request in flight.
    fn maybe_probe(&mut self) {
        if self.handle.circuit() != CircuitState::Probing || self.probe.is_some() {
            return;
        }
        let (id, r) = self.new_request(PROBE_PROMPT, PROBE_TOKENS);
        self.track(id, PROBE_TOKENS, true);
        match self.sched.submit_probe(r, self.pool.total_blocks()) {
            Ok(()) => self.probe = Some(id),
            Err(_) => {
                self.probe = Some(id);
                self.finish(id, Outcome::Failed("probe_rejected".into()));
            }
        }
    }

    /// Controller ticks for every telemetry interval up to `now`.
    fn tick_until(&mut self, now: Duration) {
        let interval = self.cfg.reliability.telemetry.interval.0;
        while self.next_tick <= now {
            let sample = self.sample();
            let stats = EngineStats {
                running_remaining_tokens: self.sched.remaining_tokens(),
                free_kv_blocks: self.pool.free_blocks(),
                block_tokens: self.cfg.params.block_tokens,
                decode_tokens_per_s: 1.0 / self.decode_step_s.max(1e-9),
                step_time_p95: self.decode_steps.p95(),
                queue_len: self.gate_len() as u32,
                iterations: self.iterations,
            };
            // As the server's pressure thread: an iteration since the last tick marks the
            // engine busy (drift and thermal count only then).
            if self.iterations != self.seen_iterations {
                self.controller.on_circuit_event(CircuitEvent::Iteration);
                self.seen_iterations = self.iterations;
            }
            if self.cfg.degraded_during_load
                && self.load_stop.is_some_and(|stop| self.next_tick < stop)
            {
                self.controller
                    .on_circuit_event(CircuitEvent::TelemetryStale);
            }
            self.controller.tick(&sample, &stats);
            let (state, circuit) = (self.handle.state(), self.handle.circuit());
            self.states.insert(state);
            if let Some(stop) = self.load_stop
                && self.next_tick >= stop
                && self.green_after_stop.is_none()
                && state == PressureState::Green
                && circuit == CircuitState::Healthy
            {
                self.green_after_stop = Some(self.next_tick - stop);
            }
            self.next_tick += interval;
        }
    }

    /// Deterministic telemetry: a dedicated device at 40 °C of a 90 °C slowdown, 64 GiB
    /// `MemAvailable`, no PSI or swap activity; the ledger and the admission queue. Device
    /// memory in use is what a real device reports: the whole KV pool (allocated up front and
    /// filling the rest of the budget, as with `kv.gpu.max_bytes: null`), the emergency reserve
    /// while held and half the workspace pool.
    fn sample(&self) -> TelemetrySample {
        let max_queue = self.sched.gate().map_or(1, AdmissionGate::max_queue).max(1);
        let pool_bytes = |kind| self.ledger.usage(DEVICE, kind).capacity;
        let memory_used = pool_bytes(PoolKind::Kv)
            + self.ledger.usage(DEVICE, PoolKind::Reserve).used
            + pool_bytes(PoolKind::Workspace) / 2;
        TelemetrySample {
            at_mono_ns: u64::try_from(self.next_tick.as_nanos()).unwrap_or(u64::MAX),
            host: HostSample {
                mem_available_bytes: Some(64 << 30),
                swap_total_bytes: Some(0),
                swap_free_bytes: Some(0),
                pswpin_total: Some(0),
                psi_memory_some_avg10: Some(0.0),
                status: SourceStatus::Ok,
            },
            devices: vec![DeviceSample {
                temperature_c: Some(40.0),
                slowdown_temperature_c: Some(90.0),
                clock_mhz: Some(2350),
                memory_used_bytes: Some(memory_used),
                ..DeviceSample::empty(DEVICE, SourceStatus::Ok)
            }],
            ledger: LedgerSample {
                kv_utilization: self.kv_usage().utilization(),
                queue_fill: self.gate_len() as f64 / max_queue as f64,
            },
            storage: None,
        }
    }

    fn observe_ledger(&mut self) {
        let kv = self.kv_usage();
        self.max_kv = self.max_kv.max(kv.used + kv.reserved);
        self.max_queue_len = self.max_queue_len.max(self.gate_len());
    }

    fn inter_arrival(&mut self, rate: f64) -> Duration {
        // U in (0, 1]: -ln(U) / rate is exponential with mean 1 / rate.
        let u = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        Duration::from_secs_f64(-(1.0 - u).ln() / rate)
    }

    fn draw(&mut self, range: (u32, u32)) -> u32 {
        let span = u64::from(range.1.saturating_sub(range.0)) + 1;
        range.0 + (self.rng.next_u64() % span) as u32
    }
}

/// The reject-table code of a refused submission.
fn reject_code(e: SubmitError) -> &'static str {
    match e {
        SubmitError::Rejected { reason, .. } => reason.http().2,
        other => other.as_str(),
    }
}
