//! Deterministic scheduler simulator (P2 S-12): the real `Scheduler` and `BlockPool` driven by
//! a cost-model executor on virtual time (`FakeClock`, no sleeps). `KvSimDriver` (P4) adds the
//! real `KvHierarchy` with simulated tiers and transfers; `Simulation::run_pipelined` (P5) runs
//! pipeline micro-batches through simulated stages.

pub mod arrivals;
pub mod digests;
pub mod executor;
pub mod overload;
pub mod pipeline;
#[cfg(test)]
pub(crate) mod workloads;

pub use arrivals::{ArrivalProcess, LengthMix, SimArrival};
pub use executor::{CostModel, SimExecutor};
pub use pipeline::{MicroBatchTrace, PipelineReport};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use smallvec::SmallVec;
use turbine_core::clock::{Clock, FakeClock};
use turbine_core::request::{FinishReason, ResourceEstimate};
use turbine_core::types::{PressureSignal, PressureState, Priority, RequestId, SeqId};
use turbine_kv::BlockPool;
use turbine_kv::hierarchy::{
    AttachOutcome, AttachRequest, KvHierarchy, PrefetchAccepted, PrefetchError, PrefetchTarget,
    PrefixAttach,
};
use turbine_kv::transfer::SimTransferBackend;
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::signals::default_thresholds;
use turbine_reliability::throttle::{SchedulerLimits, ThrottlePlan, apply_reclaim, plan_for};

use crate::policy::SchedulingPolicy;
use crate::request::{CancelReason, RequestState, SchedRequest};
use crate::scheduler::{
    BatchKind, IterationLimits, IterationOutcome, IterationPlan, Scheduler, SchedulerParams,
    SubmitError,
};

/// Choices per request reserved by the simulator's sequence-id scheme.
const MAX_CHOICES: u64 = 16;

/// One batch item as the simulator saw it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TraceItem {
    pub seq: u64,
    pub kind: BatchKind,
    /// Context length after the iteration (`BatchItem::block_table.tokens`).
    pub tokens: u32,
}

/// One planned iteration (empty plans that dropped or preempted nothing are not recorded).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IterationTrace {
    pub iteration: u64,
    pub start_s: f64,
    pub duration_s: f64,
    pub items: Vec<TraceItem>,
    pub forks: Vec<(u64, u64)>,
    pub preempted: Vec<u64>,
    pub dropped: Vec<(RequestId, CancelReason)>,
    /// Queue length and running requests after the plan.
    pub waiting: u32,
    pub running: u32,
    /// Pool blocks in use after the plan.
    pub used_blocks: u32,
}

/// Everything a run produced; serialises byte-identically for equal inputs.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SimReport {
    pub iterations: Vec<IterationTrace>,
    pub rejected: Vec<(RequestId, SubmitError)>,
    pub completed: u32,
    pub max_waiting: u32,
    pub max_running: u32,
    /// Broken invariants: starved decodes, exceeded budgets or bounds, tokens sampled at the
    /// wrong position (duplicated or skipped after a recompute).
    pub violations: Vec<String>,
    /// Time to first token of every request that got one, in virtual seconds from its arrival
    /// to the end of the iteration that sampled it, in the order the first tokens came.
    pub ttft_s: Vec<f64>,
    /// Tokens sampled over the run (every choice counted).
    pub generated_tokens: u64,
}

impl SimReport {
    /// FNV-1a 64 over the `Debug` rendering of the planned iterations: equal digests mean the
    /// scheduler planned the same batches (Phase 2m S-9 pins the default policy to main's
    /// plans with it, `digests::MAIN_DIGESTS`).
    pub fn plan_digest(&self) -> u64 {
        fnv1a(format!("{:?}", self.iterations).as_bytes())
    }
}

/// FNV-1a 64-bit hash.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

struct SimSeq {
    request: RequestId,
    prompt: u32,
    output_len: u32,
    max_new: u32,
    generated: u32,
    finished: bool,
}

struct SimReq {
    seqs: SmallVec<[SeqId; 1]>,
    shared_done: bool,
    arrival: Duration,
    first_token: bool,
}

enum Hook {
    Cancel(RequestId, CancelReason),
    Pause(SeqId),
    Resume(SeqId),
}

/// The real scheduler and pool on virtual time.
pub struct Simulation {
    sched: Scheduler,
    pool: BlockPool,
    exec: SimExecutor,
    arrivals: ArrivalProcess,
    clock: FakeClock,
    limits: IterationLimits,
    arrived: u64,
    seqs: HashMap<SeqId, SimSeq>,
    reqs: HashMap<RequestId, SimReq>,
    hooks: BTreeMap<u64, Vec<Hook>>,
    report: SimReport,
}

impl Simulation {
    pub fn new(
        params: SchedulerParams,
        pool: BlockPool,
        exec: SimExecutor,
        arrivals: ArrivalProcess,
    ) -> Simulation {
        let clock = FakeClock::new(Duration::ZERO);
        Simulation {
            sched: Scheduler::new(params, Arc::new(clock.clone())),
            pool,
            exec,
            arrivals,
            clock,
            limits: IterationLimits::default(),
            arrived: 0,
            seqs: HashMap::new(),
            reqs: HashMap::new(),
            hooks: BTreeMap::new(),
            report: SimReport::default(),
        }
    }

    /// Run the scheduler under `policy` (default: `DefaultPolicy`); set before [`Self::run`].
    pub fn with_policy(mut self, policy: &'static dyn SchedulingPolicy) -> Simulation {
        self.sched = self.sched.with_policy(policy);
        self
    }

    /// Id of the `k`-th arrival (0-based).
    pub fn request_id(k: u64) -> RequestId {
        RequestId(uuid::Uuid::from_u128(u128::from(k) + 1))
    }

    /// Sequence of choice `j` of the `k`-th arrival.
    pub fn seq_id(k: u64, j: u64) -> SeqId {
        SeqId(k * MAX_CHOICES + j)
    }

    /// Cancel request `id` just before iteration `iteration` is planned.
    pub fn cancel_at(&mut self, iteration: u64, id: RequestId, reason: CancelReason) {
        self.hooks
            .entry(iteration)
            .or_default()
            .push(Hook::Cancel(id, reason));
    }

    /// Pause `seq` (full output channel) just before iteration `iteration` is planned.
    pub fn pause_at(&mut self, iteration: u64, seq: SeqId) {
        self.hooks
            .entry(iteration)
            .or_default()
            .push(Hook::Pause(seq));
    }

    /// Resume `seq` just before iteration `iteration` is planned.
    pub fn resume_at(&mut self, iteration: u64, seq: SeqId) {
        self.hooks
            .entry(iteration)
            .or_default()
            .push(Hook::Resume(seq));
    }

    pub fn now(&self) -> Duration {
        self.clock.now_mono()
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.sched
    }

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    /// Run until virtual time `until`, or until no arrival, hook or runnable work remains.
    pub fn run(&mut self, until: Duration) -> SimReport {
        loop {
            let now = self.clock.now_mono();
            if now >= until {
                break;
            }
            self.submit_arrivals(now);
            let next = self.sched.snapshot().iterations_total + 1;
            for hook in self.hooks.remove(&next).unwrap_or_default() {
                match hook {
                    Hook::Cancel(id, reason) => self.sched.cancel(id, reason),
                    Hook::Pause(seq) => self.sched.pause(seq),
                    Hook::Resume(seq) => self.sched.resume(seq),
                }
            }
            let mut decodable: Vec<SeqId> = self
                .seqs
                .keys()
                .copied()
                .filter(|s| self.sched.seq_state(*s) == Some(RequestState::Decoding))
                .collect();
            decodable.sort();

            let plan = self.sched.plan(&mut self.pool, &self.limits);
            self.check(&plan, &decodable);
            for (id, _) in &plan.dropped {
                self.forget(*id);
            }
            if plan.is_empty() {
                self.arrivals.on_done(plan.dropped.len() as u64, now);
                self.sched.complete(
                    &mut self.pool,
                    IterationOutcome {
                        iteration: plan.iteration,
                        ..IterationOutcome::default()
                    },
                );
                if !plan.dropped.is_empty() || !plan.preempted.is_empty() {
                    self.record(&plan, now, Duration::ZERO);
                }
                match self.arrivals.peek_time() {
                    Some(t) if t < until => self.clock.set(t.max(now)),
                    Some(_) => self.clock.set(until),
                    None if !self.hooks.is_empty() && !self.sched.is_idle() => {}
                    None => break,
                }
                continue;
            }
            let duration = self.exec.duration(&plan);
            let completed = self.report.completed;
            let outcome = self.execute(&plan, now + duration);
            self.record(&plan, now, duration);
            self.clock.advance(duration);
            self.sched.complete(&mut self.pool, outcome);
            let done = u64::from(self.report.completed - completed) + plan.dropped.len() as u64;
            self.arrivals.on_done(done, now + duration);
        }
        std::mem::take(&mut self.report)
    }

    fn submit_arrivals(&mut self, now: Duration) {
        let bt = self.sched.params().block_tokens;
        while let Some(a) = self.arrivals.next_before(now) {
            let k = self.arrived;
            self.arrived += 1;
            let id = Self::request_id(k);
            let n = u64::from(a.n.clamp(1, MAX_CHOICES as u32));
            let seqs: SmallVec<[SeqId; 1]> = (0..n).map(|j| Self::seq_id(k, j)).collect();
            let mut r = SchedRequest::new(id, seqs.clone(), a.prompt_len, a.max_new_tokens, bt);
            r.priority = a.priority;
            r.arrival = a.at;
            match self.sched.submit(r, self.pool.total_blocks()) {
                Ok(()) => {
                    for &s in &seqs {
                        self.seqs.insert(
                            s,
                            SimSeq {
                                request: id,
                                prompt: a.prompt_len,
                                output_len: a.output_len.clamp(1, a.max_new_tokens.max(1)),
                                max_new: a.max_new_tokens,
                                generated: 0,
                                finished: false,
                            },
                        );
                    }
                    self.reqs.insert(
                        id,
                        SimReq {
                            seqs,
                            shared_done: false,
                            arrival: a.at,
                            first_token: false,
                        },
                    );
                }
                Err(e) => {
                    self.report.rejected.push((id, e));
                    self.arrivals.on_done(1, now);
                }
            }
        }
    }

    /// The "model": completed prefills and decodes sample one token each (the shared prefill
    /// of `n > 1` one per choice); a sequence stops after `output_len` tokens. `end` is the
    /// virtual time the iteration finishes.
    fn execute(&mut self, plan: &IterationPlan, end: Duration) -> IterationOutcome {
        let mut appended = Vec::new();
        let mut finished = Vec::new();
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
            let Some(s) = self.seqs.get(&item.seq) else {
                continue;
            };
            // The KV now ends right before the token being sampled.
            let expected = s.prompt + s.generated;
            if item.block_table.tokens != expected {
                self.report.violations.push(format!(
                    "iteration {}: seq {} samples at context {} but prompt + generated = {expected}",
                    plan.iteration, item.seq.0, item.block_table.tokens
                ));
            }
            let id = s.request;
            let r = self.reqs.get_mut(&id).expect("tracked request");
            if !r.first_token {
                r.first_token = true;
                self.report
                    .ttft_s
                    .push(end.saturating_sub(r.arrival).as_secs_f64());
            }
            let targets: Vec<SeqId> = if matches!(item.kind, BatchKind::Prefill { .. })
                && !r.shared_done
                && r.seqs[0] == item.seq
            {
                r.shared_done = true;
                r.seqs.to_vec()
            } else {
                vec![item.seq]
            };
            for t in targets {
                let s = self.seqs.get_mut(&t).expect("tracked sequence");
                if s.finished {
                    continue;
                }
                s.generated += 1;
                self.report.generated_tokens += 1;
                appended.push((t, 1));
                if s.generated >= s.output_len {
                    s.finished = true;
                    let reason = if s.output_len >= s.max_new {
                        FinishReason::Length
                    } else {
                        FinishReason::Stop
                    };
                    finished.push((t, reason));
                }
            }
            let done = self.reqs[&id]
                .seqs
                .iter()
                .all(|q| self.seqs.get(q).is_none_or(|s| s.finished));
            if done {
                self.report.completed += 1;
                self.forget(id);
            }
        }
        IterationOutcome {
            iteration: plan.iteration,
            finished,
            appended,
            failed: None,
        }
    }

    fn forget(&mut self, id: RequestId) {
        if let Some(r) = self.reqs.remove(&id) {
            for s in r.seqs {
                self.seqs.remove(&s);
            }
        }
    }

    fn check(&mut self, plan: &IterationPlan, decodable: &[SeqId]) {
        let p = *self.sched.params();
        let decoded: HashSet<SeqId> = plan
            .items
            .iter()
            .filter(|i| i.kind == BatchKind::Decode)
            .map(|i| i.seq)
            .collect();
        let preempted: HashSet<SeqId> = plan.preempted.iter().map(|(s, _)| *s).collect();
        let dropped: HashSet<RequestId> = plan.dropped.iter().map(|(id, _)| *id).collect();
        for s in decodable {
            let request = self.seqs.get(s).map(|x| x.request);
            if !decoded.contains(s)
                && !preempted.contains(s)
                && !request.is_some_and(|r| dropped.contains(&r))
            {
                self.report.violations.push(format!(
                    "iteration {}: decodable seq {} was not decoded",
                    plan.iteration, s.0
                ));
            }
        }
        let cap = if p.chunked_prefill {
            p.prefill_chunk_tokens
        } else {
            p.max_batch_tokens
        };
        for i in &plan.items {
            if let BatchKind::Prefill { len, .. } = i.kind
                && len > cap
            {
                self.report.violations.push(format!(
                    "iteration {}: chunk of {len} tokens exceeds {cap}",
                    plan.iteration
                ));
            }
        }
        let tokens = plan.prefill_tokens() + plan.decode_tokens();
        if tokens > p.max_batch_tokens {
            self.report.violations.push(format!(
                "iteration {}: {tokens} tokens exceed max_batch_tokens {}",
                plan.iteration, p.max_batch_tokens
            ));
        }
        let snap = self.sched.snapshot();
        let running = snap.prefilling + snap.decoding + snap.paused;
        self.report.max_waiting = self.report.max_waiting.max(snap.waiting);
        self.report.max_running = self.report.max_running.max(running);
        if running > p.max_running_requests || snap.waiting > p.max_queued_requests {
            self.report.violations.push(format!(
                "iteration {}: {running} running / {} waiting exceed the bounds",
                plan.iteration, snap.waiting
            ));
        }
    }

    fn record(&mut self, plan: &IterationPlan, start: Duration, duration: Duration) {
        let snap = self.sched.snapshot();
        self.report.iterations.push(IterationTrace {
            iteration: plan.iteration,
            start_s: start.as_secs_f64(),
            duration_s: duration.as_secs_f64(),
            items: plan
                .items
                .iter()
                .map(|i| TraceItem {
                    seq: i.seq.0,
                    kind: i.kind,
                    tokens: i.block_table.tokens,
                })
                .collect(),
            forks: plan.forks.iter().map(|f| (f.src.0, f.dst.0)).collect(),
            preempted: plan.preempted.iter().map(|(s, _)| s.0).collect(),
            dropped: plan.dropped.clone(),
            waiting: snap.waiting,
            running: snap.prefilling + snap.decoding + snap.paused,
            used_blocks: self.pool.used_blocks(),
        });
    }
}

// ---- Phase 4: the KV hierarchy in the loop ------------------------------------------------

/// Virtual time an iteration with nothing to execute takes, so transfers keep progressing.
const IDLE_TICK: Duration = Duration::from_millis(1);

/// Where a driver request is: attaching (or waiting on a prefix another request computes),
/// promoting its prefix into L0, or handed to the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KvStage {
    Attaching,
    Promoting,
    Scheduled,
}

struct KvSimRequest {
    seq: SeqId,
    prompt: Vec<u32>,
    max_tokens: u32,
    /// Sampled tokens (deterministic stand-ins for the model's output).
    generated: Vec<u32>,
    stage: KvStage,
}

/// The real `Scheduler`, `BlockPool` and `KvHierarchy` against the simulated executor on
/// virtual time (P4 Task 11). One iteration (`step`): transfer completions → attach retries →
/// reclaim-order refresh → `Scheduler::plan` → `after_plan` → execution → `commit_progress` →
/// `Scheduler::complete` → `request_done` → `apply_reclaim` and `tick`.
pub struct KvSimDriver {
    sched: Scheduler,
    pool: BlockPool,
    kv: KvHierarchy,
    backend: SimTransferBackend,
    clock: FakeClock,
    exec: SimExecutor,
    limits: IterationLimits,
    pressure: PressureState,
    /// Phase 3's reclaim step for `pressure` (`plan_for` + `apply_reclaim` on the documented
    /// `kv_utilization` thresholds), as the pressure controller runs it every tick.
    reclaim_plan: ThrottlePlan,
    reclaim_metrics: ReliabilityMetrics,
    next_seq: u64,
    reqs: HashMap<RequestId, KvSimRequest>,
    seq_request: HashMap<SeqId, RequestId>,
    /// Requests waiting to attach, in arrival order.
    attaching: Vec<RequestId>,
    estimates: HashMap<RequestId, ResourceEstimate>,
    prefilled: HashMap<RequestId, u32>,
    violations: Vec<String>,
}

impl KvSimDriver {
    /// The executor of the P4 simulations: 8,000 prefill tokens/s and 5 ms per decode step.
    pub const EXECUTOR: SimExecutor = SimExecutor {
        cost: CostModel {
            per_prefill_token_s: 1.0 / 8_000.0,
            per_decode_step_s: 0.005,
            per_seq_s: 0.0,
        },
    };

    pub fn new(
        sched: Scheduler,
        pool: BlockPool,
        kv: KvHierarchy,
        backend: SimTransferBackend,
        clock: FakeClock,
    ) -> KvSimDriver {
        let mut kv = kv;
        kv.set_prefill_tps(1.0 / Self::EXECUTOR.cost.per_prefill_token_s);
        let sched_params = *sched.params();
        KvSimDriver {
            sched,
            pool,
            kv,
            backend,
            clock,
            exec: Self::EXECUTOR,
            limits: IterationLimits::default(),
            pressure: PressureState::Green,
            reclaim_plan: plan_for(PressureState::Green, &Self::limits_of(&sched_params)),
            reclaim_metrics: ReliabilityMetrics::unregistered(),
            next_seq: 0,
            reqs: HashMap::new(),
            seq_request: HashMap::new(),
            attaching: Vec::new(),
            estimates: HashMap::new(),
            prefilled: HashMap::new(),
            violations: Vec::new(),
        }
    }

    /// A request arrives; it attaches its cached prefix on the next `step`.
    pub fn submit(&mut self, id: RequestId, prompt: Vec<u32>, max_tokens: u32) {
        let seq = SeqId(self.next_seq);
        self.next_seq += 1;
        self.seq_request.insert(seq, id);
        self.reqs.insert(
            id,
            KvSimRequest {
                seq,
                prompt,
                max_tokens: max_tokens.max(1),
                generated: Vec::new(),
                stage: KvStage::Attaching,
            },
        );
        self.attaching.push(id);
    }

    /// The client went away. A request not yet handed to the scheduler releases its KV now;
    /// a scheduled one is dropped by the next plan.
    pub fn cancel(&mut self, id: RequestId) {
        let Some(r) = self.reqs.get(&id) else {
            return;
        };
        if r.stage == KvStage::Scheduled {
            self.sched.cancel(id, CancelReason::ClientDisconnect);
        } else {
            self.kv.request_done(&mut self.pool, id, true);
            self.forget(id);
        }
    }

    /// L0 pressure as the Phase 3 controller would report it; reclaim follows Phase 3's
    /// throttle plan for it every step (YELLOW demotes idle blocks to the YELLOW threshold,
    /// ORANGE frees and demotes to ORANGE's, RED and above free every unreferenced cached
    /// block — `turbine_reliability::throttle::plan_for`).
    pub fn set_pressure(&mut self, s: PressureState) {
        self.pressure = s;
        self.kv.set_l0_state(s);
        self.reclaim_plan = plan_for(s, &Self::limits_of(self.sched.params()));
    }

    fn limits_of(p: &SchedulerParams) -> SchedulerLimits {
        SchedulerLimits {
            prefill_chunk_tokens: p.prefill_chunk_tokens,
            block_tokens: p.block_tokens,
        }
    }

    /// `POST /turbine/v1/kv/prefetch` against the simulated tiers.
    pub fn prefetch(
        &mut self,
        target: PrefetchTarget<'_>,
    ) -> Result<PrefetchAccepted, PrefetchError> {
        self.kv.prefetch(&mut self.pool, target)
    }

    /// One iteration; returns the executed plan.
    pub fn step(&mut self) -> IterationPlan {
        // Transfer completions: requests whose promotions landed go to the scheduler.
        for (id, attach) in self.kv.poll(&mut self.pool, &mut self.backend) {
            self.schedule(id, attach);
        }
        self.retry_attaches();
        if self.pool.free_blocks() < self.pool.total_blocks().div_ceil(10) {
            self.kv.refresh_reclaim_order(&mut self.pool);
        }
        let handle = self.kv.reclaimer();
        let kv_thresholds = default_thresholds()[&PressureSignal::KvUtilization];
        apply_reclaim(
            &self.reclaim_plan,
            handle.as_ref(),
            &kv_thresholds,
            &self.reclaim_metrics,
        );

        let plan = self.sched.plan(&mut self.pool, &self.limits);
        self.kv.after_plan(&mut self.pool);
        self.check_writes(&plan);
        for (id, _) in &plan.dropped {
            self.kv.request_done(&mut self.pool, *id, true);
            self.forget(*id);
        }
        let duration = if plan.is_empty() {
            IDLE_TICK
        } else {
            self.exec.duration(&plan)
        };
        let outcome = self.execute(&plan);
        self.clock.advance(duration);
        // The KV the plan wrote is committed before `complete` releases finished tables.
        for item in &plan.items {
            let Some(id) = self.seq_request.get(&item.seq).copied() else {
                continue;
            };
            let Some(r) = self.reqs.get(&id) else {
                continue;
            };
            let tokens: Vec<u32> = r
                .prompt
                .iter()
                .chain(&r.generated)
                .copied()
                .take(item.block_table.tokens as usize)
                .collect();
            self.kv
                .commit_progress(&mut self.pool, id, &item.block_table.blocks, &tokens);
        }
        let finished: Vec<SeqId> = outcome.finished.iter().map(|(s, _)| *s).collect();
        self.record_samples(&outcome);
        self.sched.complete(&mut self.pool, outcome);
        for seq in finished {
            if let Some(id) = self.seq_request.get(&seq).copied() {
                self.kv.request_done(&mut self.pool, id, false);
                self.forget(id);
            }
        }
        self.kv.apply_reclaim(&mut self.pool);
        self.kv.tick(&mut self.pool);
        plan
    }

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    pub fn kv(&self) -> &KvHierarchy {
        &self.kv
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.sched
    }

    /// No request is attaching, promoting, queued or running.
    pub fn is_idle(&self) -> bool {
        self.reqs.is_empty() && self.sched.is_idle()
    }

    /// Prompt tokens prefilled for `id` (recomputes after a preemption included).
    pub fn prefilled_tokens(&self, id: RequestId) -> u32 {
        self.prefilled.get(&id).copied().unwrap_or(0)
    }

    /// The estimate `id` was submitted to the scheduler with.
    pub fn last_estimate(&self, id: RequestId) -> Option<ResourceEstimate> {
        self.estimates.get(&id).copied()
    }

    /// The request a sequence belongs to (also after it finished).
    pub fn request_of(&self, seq: SeqId) -> Option<RequestId> {
        self.seq_request.get(&seq).copied()
    }

    /// Broken invariants: a batch item writing a block another holder references.
    pub fn violations(&self) -> &[String] {
        &self.violations
    }

    fn retry_attaches(&mut self) {
        for id in std::mem::take(&mut self.attaching) {
            let Some(r) = self.reqs.get(&id) else {
                continue;
            };
            let req = AttachRequest {
                request: id,
                prompt: &r.prompt,
                cache_salt: "",
                session: None,
                priority: Priority::default(),
            };
            match self.kv.attach_prefix(&mut self.pool, &req) {
                AttachOutcome::Ready(a) => self.schedule(id, a),
                AttachOutcome::Promoting => {
                    if let Some(r) = self.reqs.get_mut(&id) {
                        r.stage = KvStage::Promoting;
                    }
                }
                AttachOutcome::WaitForPrefix => self.attaching.push(id),
            }
        }
    }

    /// Hands a request and its attached prefix to the scheduler.
    fn schedule(&mut self, id: RequestId, attach: PrefixAttach) {
        let Some(r) = self.reqs.get_mut(&id) else {
            // Cancelled while its promotions were in flight.
            self.pool.release(&attach.blocks);
            return;
        };
        r.stage = KvStage::Scheduled;
        let bt = self.sched.params().block_tokens;
        let prompt = r.prompt.len() as u32;
        let mut sr = SchedRequest::new(id, smallvec::smallvec![r.seq], prompt, r.max_tokens, bt);
        sr.arrival = self.clock.now_mono();
        sr.attach_prefix(attach, bt);
        self.estimates.insert(id, sr.estimate);
        let blocks = sr.cached_prefix.as_ref().map(|a| a.blocks.clone());
        if let Err(e) = self.sched.submit(sr, self.pool.total_blocks()) {
            self.violations
                .push(format!("request {} rejected at submission: {e}", id.0));
            if let Some(b) = blocks {
                self.pool.release(&b);
            }
            self.kv.request_done(&mut self.pool, id, true);
            self.forget(id);
        }
    }

    /// Every block a batch item writes must be held by that sequence alone.
    fn check_writes(&mut self, plan: &IterationPlan) {
        let bt = self.sched.params().block_tokens.max(1);
        for item in &plan.items {
            let (start, len) = match item.kind {
                BatchKind::Prefill { start, len } => (start, len),
                BatchKind::Decode => (item.block_table.tokens - 1, 1),
            };
            if len == 0 {
                continue;
            }
            for b in start / bt..=(start + len - 1) / bt {
                let Some(block) = item.block_table.blocks.get(b as usize) else {
                    continue;
                };
                let rc = self.pool.refcount(*block);
                if rc != 1 {
                    self.violations.push(format!(
                        "iteration {}: seq {} writes block {:?} with {rc} holders",
                        plan.iteration, item.seq.0, block
                    ));
                }
            }
        }
    }

    /// The "model": a completed prefill or a decode samples one deterministic token.
    fn execute(&mut self, plan: &IterationPlan) -> IterationOutcome {
        let mut appended = Vec::new();
        let mut finished = Vec::new();
        for item in &plan.items {
            let samples = match item.kind {
                BatchKind::Decode => true,
                BatchKind::Prefill { start, len } => {
                    let id = self.seq_request.get(&item.seq).copied();
                    if let Some(id) = id {
                        *self.prefilled.entry(id).or_insert(0) += len;
                    }
                    self.sched.prefill_target(item.seq) == Some(start + len)
                }
            };
            if !samples {
                continue;
            }
            let Some(r) = self
                .seq_request
                .get(&item.seq)
                .and_then(|id| self.reqs.get(id))
            else {
                continue;
            };
            appended.push((item.seq, 1));
            if r.generated.len() as u32 + 1 >= r.max_tokens {
                finished.push((item.seq, FinishReason::Length));
            }
        }
        IterationOutcome {
            iteration: plan.iteration,
            finished,
            appended,
            failed: None,
        }
    }

    fn record_samples(&mut self, outcome: &IterationOutcome) {
        for (seq, n) in &outcome.appended {
            let Some(r) = self
                .seq_request
                .get(seq)
                .and_then(|id| self.reqs.get_mut(id))
            else {
                continue;
            };
            for _ in 0..*n {
                let pos = (r.prompt.len() + r.generated.len()) as u32;
                r.generated
                    .push(seq.0 as u32 ^ pos.wrapping_mul(2_654_435_761));
            }
        }
    }

    fn forget(&mut self, id: RequestId) {
        self.reqs.remove(&id);
        self.attaching.retain(|x| *x != id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use turbine_core::registry::Module;
    use turbine_core::types::SeqId;

    use super::workloads::*;
    use super::*;
    use crate::policy::{self, DefaultPolicy};
    use crate::request::CancelReason;
    use crate::scheduler::{BatchKind, SchedulerParams, SubmitError};

    /// `(label, kind)` of an iteration's items, labels from arrival indices.
    fn composition(it: &IterationTrace, labels: &BTreeMap<u64, &str>) -> Vec<(String, BatchKind)> {
        let mut v: Vec<(String, BatchKind)> = it
            .items
            .iter()
            .map(|i| (labels[&i.seq].to_string(), i.kind))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    // ---- seeded workloads: shared by the invariant tests and the plan digests ------------

    /// Every registered policy: each invariant test runs over all of them (Phase 2m S-9).
    fn policies() -> impl Iterator<Item = Policy> {
        policy::registry().iter()
    }

    /// TS §7: C and D arrive first and are decoding by the time A (3 chunks of 32) arrives;
    /// B (one chunk) arrives after A's first chunk. One virtual second per sequence.
    fn ts_section7_run(policy: Policy) -> SimReport {
        let arrivals = vec![
            SimArrival::new(secs(0.0), 8, 100), // C
            SimArrival::new(secs(0.0), 8, 100), // D
            SimArrival::new(secs(2.0), 96, 10), // A: after C and D's prefill (2 s)
            SimArrival::new(secs(5.0), 20, 10), // B: after A's first chunk (3 s)
        ];
        let p = SchedulerParams {
            max_batch_tokens: 256,
            prefill_chunk_tokens: 32,
            ..params()
        };
        let mut sim = simulation(
            policy,
            p,
            pool(256),
            per_seq_cost(),
            ArrivalProcess::scripted(arrivals),
        );
        sim.run(secs(12.0))
    }

    /// The Poisson run without a pause, a sequence that decodes in iterations 199, 200 and
    /// 201 there, and the run that pauses it for iterations 200..260.
    fn decode_never_starved_runs(policy: Policy) -> (SimReport, u64, SimReport) {
        let free_run = poisson_run(policy, None);
        let victim = free_run
            .iterations
            .iter()
            .find(|it| it.iteration == 199)
            .and_then(|it| {
                it.items
                    .iter()
                    .filter(|i| i.kind == BatchKind::Decode)
                    .find(|i| decodes_of(&free_run, i.seq, 199..202) == 3)
            })
            .map(|i| i.seq)
            .expect("a long-running decode around iteration 200");
        let paused = poisson_run(policy, Some(SeqId(victim)));
        (free_run, victim, paused)
    }

    /// D and Z decode, P prefills 100 tokens in 16-token chunks, W waits for a slot; Z is
    /// paused at iteration 3 and all four are cancelled at iteration 5.
    fn cancellation_run(policy: Policy) -> SimReport {
        let p = SchedulerParams {
            max_running_requests: 3,
            max_batch_tokens: 64,
            prefill_chunk_tokens: 16,
            ..params()
        };
        let arrivals = vec![
            SimArrival::new(secs(0.0), 8, 500),  // D
            SimArrival::new(secs(0.0), 8, 500),  // Z
            SimArrival::new(secs(0.0), 100, 10), // P
            SimArrival::new(secs(0.0), 8, 10),   // W
        ];
        let mut sim = simulation(
            policy,
            p,
            pool(128),
            realistic_cost(),
            ArrivalProcess::scripted(arrivals),
        );
        sim.pause_at(3, Simulation::seq_id(1, 0));
        for (k, reason) in [
            (0, CancelReason::ClientDisconnect),
            (1, CancelReason::SlowClient),
            (2, CancelReason::RequestTimeout),
            (3, CancelReason::Shutdown),
        ] {
            sim.cancel_at(5, Simulation::request_id(k), reason);
        }
        sim.run(secs(100.0))
    }

    /// Service ≈ 8 running / (150 tokens × 0.5 s) ≈ 0.1 request/s; arrivals at 1/s = 10×.
    /// Returns the report and the virtual time the run ended at.
    fn overload_run(policy: Policy) -> (SimReport, Duration) {
        let mix = LengthMix {
            short_prompt: (4, 64),
            long_prompt: (65, 256),
            long_fraction: 0.2,
            output: (100, 200),
            max_new_tokens: 200,
        };
        let p = SchedulerParams {
            max_running_requests: 8,
            max_batch_tokens: 256,
            prefill_chunk_tokens: 128,
            max_queued_requests: 32,
            ..params()
        };
        let exec = SimExecutor {
            cost: CostModel {
                per_prefill_token_s: 0.000_5,
                per_decode_step_s: 0.5,
                per_seq_s: 0.001,
            },
        };
        let arrivals = ArrivalProcess::poisson(1.0, 7).with_mix(mix);
        let mut sim = simulation(policy, p, pool(512), exec, arrivals);
        let report = sim.run(secs(10_000.0));
        (report, sim.now())
    }

    /// Ten 100-token requests, four in flight at once.
    fn closed_loop_run(policy: Policy) -> SimReport {
        let template = SimArrival::new(secs(0.0), 100, 20);
        let arrivals = ArrivalProcess::closed_loop(template, 4, 10);
        let mut sim = simulation(policy, params(), pool(1024), realistic_cost(), arrivals);
        sim.run(secs(1_000.0))
    }

    /// Per Phase 2c lab config: its file, scheduler parameters, the starvation/budget run
    /// (1,000 seeded Poisson arrivals, prompts up to 3,000 tokens) and the overload run
    /// (1,000 arrivals at ~10× the service rate).
    fn phase2c_runs(policy: Policy) -> Vec<(&'static str, SchedulerParams, SimReport, SimReport)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut runs = Vec::new();
        for file in [
            "scripts/lab/phase2c-novanas-llama.yaml",
            "scripts/lab/phase2c-novanas-olmoe.yaml",
        ] {
            let cfg = turbine_core::config::load(&root.join(file), &[])
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            assert_eq!(
                (
                    cfg.scheduler.max_batch_tokens,
                    cfg.scheduler.prefill_chunk_tokens
                ),
                (PHASE2C_MAX_BATCH_TOKENS, PHASE2C_PREFILL_CHUNK_TOKENS),
                "{file}"
            );
            let p = SchedulerParams::from_config(&cfg, 8192);
            let blocks = 584;

            let mix = LengthMix {
                short_prompt: (4, 256),
                long_prompt: (257, 3000),
                long_fraction: 0.3,
                output: (1, 256),
                max_new_tokens: 256,
            };
            let arrivals = ArrivalProcess::poisson(2.0, 11)
                .with_mix(mix)
                .with_limit(1000);
            let mut sim = simulation(
                policy,
                p,
                pool_of(blocks, p.block_tokens),
                llama_r9700_cost(66e-6),
                arrivals,
            );
            let starvation = sim.run(secs(100_000.0));

            let mix = LengthMix {
                short_prompt: (4, 256),
                long_prompt: (257, 2048),
                long_fraction: 0.2,
                output: (200, 256),
                max_new_tokens: 256,
            };
            let arrivals = ArrivalProcess::poisson(200.0, 13)
                .with_mix(mix)
                .with_limit(1000);
            let mut sim = simulation(
                policy,
                p,
                pool_of(blocks, p.block_tokens),
                llama_r9700_cost(66e-6),
                arrivals,
            );
            let overload = sim.run(secs(100_000.0));
            runs.push((file, p, starvation, overload));
        }
        runs
    }

    /// One digest over several runs of a workload.
    fn digest_of(reports: &[&SimReport]) -> u64 {
        match reports {
            [one] => one.plan_digest(),
            many => {
                let digests: Vec<u64> = many.iter().map(|r| r.plan_digest()).collect();
                fnv1a(format!("{digests:?}").as_bytes())
            }
        }
    }

    /// `(workload, plan digest)` of every seeded workload of the invariant tests under the
    /// `default` policy.
    fn workload_digests() -> Vec<(&'static str, u64)> {
        let policy: Policy = &DefaultPolicy;
        let (free_run, _, paused) = decode_never_starved_runs(policy);
        let (chunked, unchunked) = chunk_budget_runs(policy);
        let phase2c = phase2c_runs(policy);
        let phase2c_reports: Vec<&SimReport> =
            phase2c.iter().flat_map(|(_, _, a, b)| [a, b]).collect();
        vec![
            (
                "ts_section7_iteration_pattern",
                digest_of(&[&ts_section7_run(policy)]),
            ),
            ("decode_never_starved", digest_of(&[&free_run, &paused])),
            ("chunk_budget_respected", digest_of(&[&chunked, &unchunked])),
            (
                "preemption_by_recompute",
                digest_of(&[&preemption_run(policy, 16, 12, 40, 100)]),
            ),
            (
                "preemption_by_recompute_at_128_token_pages",
                digest_of(&[&preemption_run(policy, 128, 6, 200, 300)]),
            ),
            (
                "cancellation_frees_within_one_iteration",
                digest_of(&[&cancellation_run(policy)]),
            ),
            (
                "bounded_under_overload",
                digest_of(&[&overload_run(policy).0]),
            ),
            (
                "closed_loop_keeps_concurrency",
                digest_of(&[&closed_loop_run(policy)]),
            ),
            (
                "phase2c_lab_configs_hold_invariants",
                digest_of(&phase2c_reports),
            ),
        ]
    }

    /// Prints the plan digest of every seeded workload (`digest <workload> <u64>`); run on
    /// main to fill `digests::MAIN_DIGESTS`.
    #[test]
    #[ignore = "records the digests; run with --ignored --nocapture"]
    fn record_plan_digests() {
        for (workload, digest) in workload_digests() {
            println!("digest {workload} {digest}");
        }
    }

    /// Phase 2m S-9: the `default` policy plans exactly what main planned. Catches any change
    /// of admission order, victim choice or chunk sizing that moves a single batch item.
    #[test]
    fn default_policy_plan_digests_match_main() {
        let digests = workload_digests();
        assert_eq!(digests.len(), digests::MAIN_DIGESTS.len());
        for ((workload, digest), (main_workload, main_digest)) in
            digests.iter().zip(digests::MAIN_DIGESTS)
        {
            assert_eq!(workload, main_workload);
            assert_eq!(
                digest, main_digest,
                "{workload}: the default policy's plans differ from main's"
            );
        }
    }

    /// Whether `policy` orders like the Phase 2 scheduler (priority, then arrival; preempted
    /// requests first; the worst-ranked victim). Assertions about who is preempted or admitted
    /// first hold only for such a policy; every other invariant holds for every policy.
    fn orders_like_default(policy: Policy) -> bool {
        policy.name() == DefaultPolicy.name()
    }

    // ---- invariant tests: every one runs over every registered policy -------------------

    #[test]
    fn ts_section7_iteration_pattern() {
        for policy in policies() {
            let name = policy.name();
            let report = ts_section7_run(policy);
            assert!(
                report.violations.is_empty(),
                "{name}: {:?}",
                report.violations
            );
            // Decodes precede prefill chunks inside every batch.
            for it in &report.iterations {
                let first_prefill = it.items.iter().position(|i| i.kind != BatchKind::Decode);
                if let Some(f) = first_prefill {
                    assert!(
                        it.items[f..].iter().all(|i| i.kind != BatchKind::Decode),
                        "{name}: a decode after a prefill chunk in {}",
                        it.iteration
                    );
                }
            }
            if !orders_like_default(policy) {
                continue;
            }
            let labels: BTreeMap<u64, &str> = [
                (Simulation::seq_id(0, 0).0, "C"),
                (Simulation::seq_id(1, 0).0, "D"),
                (Simulation::seq_id(2, 0).0, "A"),
                (Simulation::seq_id(3, 0).0, "B"),
            ]
            .into();
            let prefill = |start, len| BatchKind::Prefill { start, len };
            let d = BatchKind::Decode;
            let s = |x: &str| x.to_string();
            // TS §7 iterations 1–3 are simulator iterations 2–4.
            assert_eq!(
                composition(&report.iterations[1], &labels),
                [(s("A"), prefill(0, 32)), (s("C"), d), (s("D"), d)],
                "{name}"
            );
            assert_eq!(
                composition(&report.iterations[2], &labels),
                [
                    (s("A"), prefill(32, 32)),
                    (s("B"), prefill(0, 20)),
                    (s("C"), d),
                    (s("D"), d)
                ],
                "{name}"
            );
            assert_eq!(
                composition(&report.iterations[3], &labels),
                [
                    (s("A"), prefill(64, 32)),
                    (s("B"), d),
                    (s("C"), d),
                    (s("D"), d)
                ],
                "{name}"
            );
        }
    }

    fn decodes_of(report: &SimReport, seq: u64, iterations: std::ops::Range<u64>) -> usize {
        report
            .iterations
            .iter()
            .filter(|it| iterations.contains(&it.iteration))
            .filter(|it| {
                it.items
                    .iter()
                    .any(|i| i.seq == seq && i.kind == BatchKind::Decode)
            })
            .count()
    }

    /// Pausing and resuming around the starvation run: a paused sequence is never decoded and
    /// decodes again once resumed, and the run is deterministic. That no unpaused decode
    /// starves is `policies_suite`'s `decode_never_starved` (`registry_conformance::policies`).
    #[test]
    fn decode_never_starved() {
        for policy in policies() {
            let name = policy.name();
            // A sequence that decodes in iterations 199, 200 and 201 when nobody pauses it.
            let (_, victim, report) = decode_never_starved_runs(policy);
            // The pause starves nobody else either.
            assert!(report.violations.is_empty(), "{name}");
            assert!(report.iterations.len() > 1000, "{name}");
            assert_eq!(
                decodes_of(&report, victim, 200..260),
                0,
                "{name}: a paused sequence was decoded"
            );
            assert!(
                decodes_of(&report, victim, 260..270) > 0,
                "{name}: the resumed sequence decodes again"
            );
            let again = poisson_run(policy, Some(SeqId(victim)));
            assert_eq!(
                serde_json::to_vec(&report).unwrap(),
                serde_json::to_vec(&again).unwrap(),
                "{name}: same seed, different simulation"
            );
        }
    }

    /// The default policy chunks long prompts at exactly the chunk size. That every policy
    /// keeps the chunk and batch budgets (and rejects an over-budget prompt without chunked
    /// prefill) is `policies_suite`'s `chunk_budget` (`registry_conformance::policies`).
    #[test]
    fn chunk_budget_respected() {
        for policy in policies().filter(|p| orders_like_default(*p)) {
            let name = policy.name();
            let (report, _) = chunk_budget_runs(policy);
            assert!(
                report
                    .iterations
                    .iter()
                    .any(|it| it.items.iter().any(|i| i.kind
                        == BatchKind::Prefill {
                            start: 128,
                            len: 128
                        })),
                "{name}: long prompts are chunked at the chunk size"
            );
        }
    }

    #[test]
    fn preemption_by_recompute() {
        // 12 blocks of 16 tokens; each request needs up to 9 blocks, so both cannot finish
        // together.
        for policy in policies() {
            preemption_case(policy, preemption_run(policy, 16, 12, 40, 100));
        }
    }

    #[test]
    fn preemption_by_recompute_at_128_token_pages() {
        // The default page: 6 blocks of 128 tokens; each request needs up to 4 blocks.
        for policy in policies() {
            preemption_case(policy, preemption_run(policy, 128, 6, 200, 300));
        }
    }

    /// The default policy's choices in a [`preemption_run`]: R1 (the lower priority) is the
    /// only victim and is readmitted ahead of R2. That every policy preempts by recompute
    /// (from position 0, over prompt + generated, nothing duplicated or skipped, everything
    /// completing) is `policies_suite`'s `preemption_by_recompute`
    /// (`registry_conformance::policies`).
    fn preemption_case(policy: Policy, report: SimReport) {
        if !orders_like_default(policy) {
            return;
        }
        let name = policy.name();
        let r1_seq = Simulation::seq_id(1, 0).0;
        let r2_seq = Simulation::seq_id(2, 0).0;
        let preempted: Vec<(u64, u64)> = report
            .iterations
            .iter()
            .flat_map(|it| it.preempted.iter().map(move |s| (it.iteration, *s)))
            .collect();
        assert!(
            !preempted.is_empty(),
            "{name}: the pool is too small: someone is preempted"
        );
        let (first_preemption, victim) = preempted[0];
        let re_prefill: Vec<(u64, u32, u32)> = report
            .iterations
            .iter()
            .filter(|it| it.iteration > first_preemption)
            .flat_map(|it| {
                it.items
                    .iter()
                    .filter(|i| i.seq == victim)
                    .filter_map(move |i| match i.kind {
                        BatchKind::Prefill { start, len } => Some((it.iteration, start, len)),
                        BatchKind::Decode => None,
                    })
            })
            .collect();
        // R1 (the lower priority) is the only victim and is readmitted ahead of R2
        // (preempted requests go to the queue front).
        assert!(
            preempted.iter().all(|(_, s)| *s == r1_seq),
            "{name}: wrong victim: {preempted:?}"
        );
        let r2_first = report
            .iterations
            .iter()
            .find(|it| it.items.iter().any(|i| i.seq == r2_seq))
            .map(|it| it.iteration)
            .unwrap();
        assert!(
            re_prefill[0].0 <= r2_first,
            "{name}: the preempted request was not at the queue front"
        );
    }

    #[test]
    fn cancellation_frees_within_one_iteration() {
        for policy in policies() {
            let name = policy.name();
            let report = cancellation_run(policy);
            let z = Simulation::seq_id(1, 0);
            assert!(
                report.violations.is_empty(),
                "{name}: {:?}",
                report.violations
            );
            let it4 = report
                .iterations
                .iter()
                .find(|it| it.iteration == 4)
                .unwrap();
            assert!(
                it4.items.iter().all(|i| i.seq != z.0),
                "{name}: Z is paused"
            );
            assert_eq!(it4.waiting, 1, "{name}: one request waits for a slot");
            assert!(it4.used_blocks > 0, "{name}");
            if orders_like_default(policy) {
                let p_seq = Simulation::seq_id(2, 0).0;
                assert!(
                    it4.items.iter().any(|i| i.seq == p_seq
                        && matches!(i.kind, BatchKind::Prefill { start, .. } if start < 100)),
                    "{name}: P is between prefill chunks"
                );
            }
            let it5 = report
                .iterations
                .iter()
                .find(|it| it.iteration == 5)
                .unwrap();
            assert_eq!(it5.dropped.len(), 4, "{name}");
            assert_eq!(
                it5.used_blocks, 0,
                "{name}: every block is free before the next plan returns"
            );
            assert!(it5.items.is_empty(), "{name}");
            assert_eq!(report.completed, 0, "{name}");
        }
    }

    #[test]
    fn bounded_under_overload() {
        for policy in policies() {
            let name = policy.name();
            let (report, now) = overload_run(policy);
            assert!(
                report.violations.is_empty(),
                "{name}: {:?}",
                &report.violations[..report.violations.len().min(5)]
            );
            assert!(
                report.max_waiting <= 32,
                "{name}: waiting reached {}",
                report.max_waiting
            );
            assert!(
                report.max_running <= 8,
                "{name}: running reached {}",
                report.max_running
            );
            assert!(
                report.max_waiting == 32,
                "{name}: the queue fills under overload"
            );
            assert!(
                report.rejected.len() > 5_000,
                "{name}: {} rejected",
                report.rejected.len()
            );
            assert!(
                report
                    .rejected
                    .iter()
                    .all(|(_, e)| *e == SubmitError::QueueFull),
                "{name}"
            );
            assert!(
                report.completed > 500,
                "{name}: {} completed",
                report.completed
            );
            assert!(now >= secs(10_000.0), "{name}");
        }
    }

    /// The tuned batch shape (P2c S-12) the Phase 2c lab configs run
    /// (`scripts/lab/phase2c-novanas-*.yaml`).
    const PHASE2C_MAX_BATCH_TOKENS: u32 = 2048;
    const PHASE2C_PREFILL_CHUNK_TOKENS: u32 = 2048;

    #[test]
    fn closed_loop_keeps_concurrency() {
        for policy in policies() {
            let name = policy.name();
            let report = closed_loop_run(policy);
            assert!(
                report.violations.is_empty(),
                "{name}: {:?}",
                report.violations
            );
            assert_eq!(report.completed, 10, "{name}");
            assert_eq!(
                report.max_running, 4,
                "{name}: four requests in flight at once"
            );
            assert_eq!(report.ttft_s.len(), 10, "{name}");
            assert_eq!(report.generated_tokens, 200, "{name}");
            // The first four arrive at zero and prefill together (4 × 100 tokens ≤ 512).
            let first = report.iterations[0].duration_s;
            for t in &report.ttft_s[..4] {
                assert!((t - first).abs() < 1e-9, "{name}: ttft {t} vs {first}");
            }
            // Each later arrival comes when a request finishes, never before.
            assert!(report.ttft_s.iter().all(|&t| t > 0.0), "{name}");
        }
    }

    /// Iteration cost of Llama-3.2-3B on one R9700 (Phase 2c, fused projections, 128-token
    /// pages): a decode step of 16 sequences takes 17.2 ms (perf log, `decode fwd ms`) and a
    /// 2,048-token prefill chunk 135 ms (`forward_profile` prefill_2048), i.e. 66 µs a token.
    fn llama_r9700_cost(per_prefill_token_s: f64) -> SimExecutor {
        SimExecutor {
            cost: CostModel {
                per_prefill_token_s,
                per_decode_step_s: 0.0172,
                per_seq_s: 0.000_05,
            },
        }
    }

    /// The Phase 2c scheduler parameters with the given batch shape (the lab configs' other
    /// values: 64 running, 256 queued, 128-token pages).
    fn phase2c_params(max_batch_tokens: u32, prefill_chunk_tokens: u32) -> SchedulerParams {
        SchedulerParams {
            max_running_requests: 64,
            max_batch_tokens,
            prefill_chunk_tokens,
            max_queued_requests: 256,
            chunked_prefill: true,
            block_tokens: 128,
            free_watermark: 0.01,
            max_seq_len: 8192,
            queue_timeout: Duration::from_secs(60),
        }
    }

    /// The `p`-quantile (0..=1, nearest rank) of `v`.
    fn quantile(v: &[f64], p: f64) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[((s.len() as f64 * p).ceil() as usize).clamp(1, s.len()) - 1]
    }

    /// One closed-loop run of the Phase 2c benchmark workload (`turbine-bench --concurrency 16
    /// --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos`: ~680-token prompts):
    /// (TTFT p50 s, TTFT p90 s, output tok/s, ITL p50 s, worst decode stall s).
    fn bench_workload(p: SchedulerParams, exec: SimExecutor) -> (f64, f64, f64, f64, f64) {
        let template = SimArrival::new(secs(0.0), 680, 256);
        let arrivals = ArrivalProcess::closed_loop(template, 16, 200);
        // 8 GiB of Llama KV at 128-token pages (28 layers × 2 × 8 heads × 128 × BF16).
        let mut sim = Simulation::new(p, pool_of(584, 128), exec, arrivals);
        let report = sim.run(secs(100_000.0));
        assert!(
            report.violations.is_empty(),
            "{:?}",
            &report.violations[..report.violations.len().min(5)]
        );
        assert_eq!(report.completed, 200);
        assert_eq!(report.ttft_s.len(), 200);
        // Every decoded token waits for its iteration: token-weighted iteration durations.
        let mut itl = Vec::new();
        for it in &report.iterations {
            let decodes = it
                .items
                .iter()
                .filter(|i| i.kind == BatchKind::Decode)
                .count();
            itl.extend(std::iter::repeat_n(it.duration_s, decodes));
        }
        (
            quantile(&report.ttft_s, 0.5),
            quantile(&report.ttft_s, 0.9),
            report.generated_tokens as f64 / sim.now().as_secs_f64(),
            quantile(&itl, 0.5),
            itl.iter().copied().fold(0.0, f64::max),
        )
    }

    /// P2c S-12 (simulator part of the batch-shape sweep): the benchmark workload under the
    /// R9700 Llama cost model for every (`max_batch_tokens`, `prefill_chunk_tokens`) pair of the
    /// lab sweep. With 8,192 batch tokens the first wave admits 12 prompts into one ~0.55 s
    /// iteration, so the median request waits for the whole wave; a 2,048-token batch prefills
    /// about three prompts per iteration and interleaves the decodes, which roughly halves the
    /// median TTFT and the worst decode stall for the same throughput (the prefill work does not
    /// change, only its order). Prints one `sim_batch_shape` line per pair.
    #[test]
    fn phase2c_batch_shape_ttft() {
        let mut rows = Vec::new();
        for per_token in [66e-6, 57e-6] {
            for b in [1024, 2048, 4096, 8192] {
                for c in [512, 1024, 2048] {
                    if c > b {
                        continue;
                    }
                    let r = bench_workload(phase2c_params(b, c), llama_r9700_cost(per_token));
                    println!(
                        "sim_batch_shape prefill_us_per_token={:.0} max_batch_tokens={b} prefill_chunk_tokens={c} ttft_p50_ms={:.0} ttft_p90_ms={:.0} tok_s={:.1} itl_p50_ms={:.1} itl_max_ms={:.0}",
                        per_token * 1e6,
                        r.0 * 1e3,
                        r.1 * 1e3,
                        r.2,
                        r.3 * 1e3,
                        r.4 * 1e3
                    );
                    rows.push(((per_token * 1e6).round() as u32, b, c, r));
                }
            }
        }
        let get = |t: u32, b: u32, c: u32| {
            rows.iter()
                .find(|r| (r.0, r.1, r.2) == (t, b, c))
                .map(|r| r.3)
                .expect("simulated pair")
        };
        for t in [66, 57] {
            let base = get(t, 8192, 2048);
            let tuned = get(t, PHASE2C_MAX_BATCH_TOKENS, PHASE2C_PREFILL_CHUNK_TOKENS);
            assert!(
                tuned.0 <= 0.75 * base.0,
                "{t} µs/token: TTFT p50 {:.3} s vs {:.3} s at 8192/2048",
                tuned.0,
                base.0
            );
            assert!(
                tuned.2 >= 0.98 * base.2,
                "{t} µs/token: {:.1} tok/s vs {:.1} at 8192/2048",
                tuned.2,
                base.2
            );
            assert!(tuned.4 < base.4, "the worst decode stall shrinks");
        }
    }

    /// P2c S-12: both Phase 2c lab configs run the tuned batch shape, and the scheduler keeps its
    /// guarantees with it: 1,000 seeded Poisson arrivals with prompts up to 3,000 tokens (so
    /// prompts are chunked) never starve a decode and never exceed the chunk or batch budget,
    /// and 1,000 arrivals at ~10× the service rate stay within the running and queue bounds
    /// (the overflow is rejected `queue_full`).
    #[test]
    fn phase2c_lab_configs_hold_invariants() {
        let runs = policies().flat_map(|policy| {
            phase2c_runs(policy)
                .into_iter()
                .map(move |(file, p, a, b)| (format!("{}: {file}", policy.name()), p, a, b))
        });
        for (file, p, report, overload) in runs {
            // Starvation and budgets.
            assert!(
                report.violations.is_empty(),
                "{file}: {:?}",
                &report.violations[..report.violations.len().min(5)]
            );
            assert_eq!(
                report.completed as usize + report.rejected.len(),
                1000,
                "{file}"
            );
            let mut chunked = false;
            for it in &report.iterations {
                let mut tokens = 0;
                for i in &it.items {
                    tokens += match i.kind {
                        BatchKind::Prefill { start, len } => {
                            assert!(len <= p.prefill_chunk_tokens, "{file}: chunk {len}");
                            chunked |= start > 0;
                            len
                        }
                        BatchKind::Decode => 1,
                    };
                }
                assert!(tokens <= p.max_batch_tokens, "{file}: {tokens} tokens");
            }
            assert!(chunked, "{file}: long prompts are chunked");

            // Overload: ~10× the service rate.
            let report = overload;
            assert!(
                report.violations.is_empty(),
                "{file}: {:?}",
                &report.violations[..report.violations.len().min(5)]
            );
            assert!(report.max_running <= p.max_running_requests, "{file}");
            assert_eq!(
                report.max_waiting, p.max_queued_requests,
                "{file}: the queue fills"
            );
            assert!(!report.rejected.is_empty(), "{file}");
            assert!(
                report
                    .rejected
                    .iter()
                    .all(|(_, e)| *e == SubmitError::QueueFull),
                "{file}"
            );
            assert_eq!(
                report.completed as usize + report.rejected.len(),
                1000,
                "{file}"
            );
        }
    }
}
