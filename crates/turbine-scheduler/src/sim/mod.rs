//! Deterministic scheduler simulator (P2 S-12): the real `Scheduler` and `BlockPool` driven by
//! a cost-model executor on virtual time (`FakeClock`, no sleeps).

pub mod arrivals;
pub mod executor;

pub use arrivals::{ArrivalProcess, LengthMix, SimArrival};
pub use executor::{CostModel, SimExecutor};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use smallvec::SmallVec;
use turbine_core::clock::{Clock, FakeClock};
use turbine_core::request::FinishReason;
use turbine_core::types::{RequestId, SeqId};
use turbine_kv::BlockPool;

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    use turbine_core::types::{DType, DeviceId, KvLayout, Priority, SeqId};
    use turbine_kv::{BlockPool, BlockPoolConfig};
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::request::CancelReason;
    use crate::scheduler::{BatchKind, SchedulerParams, SubmitError};

    /// Accounting-only pool of 16-token blocks: zero bytes per block.
    fn pool(n: u32) -> BlockPool {
        pool_of(n, 16)
    }

    /// Accounting-only pool of `n` blocks of `block_tokens` tokens.
    fn pool_of(n: u32, block_tokens: u32) -> BlockPool {
        let layout = KvLayout {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: DType::BF16,
            block_tokens,
        };
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 0);
        BlockPool::new(
            BlockPoolConfig {
                layout,
                num_blocks: n,
            },
            mem,
        )
        .unwrap()
    }

    fn params() -> SchedulerParams {
        SchedulerParams {
            max_running_requests: 32,
            max_batch_tokens: 512,
            prefill_chunk_tokens: 128,
            max_queued_requests: 1024,
            chunked_prefill: true,
            block_tokens: 16,
            free_watermark: 0.01,
            max_seq_len: 4096,
            queue_timeout: Duration::from_secs(3600),
        }
    }

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    fn per_seq_cost() -> SimExecutor {
        SimExecutor {
            cost: CostModel {
                per_prefill_token_s: 0.0,
                per_decode_step_s: 0.0,
                per_seq_s: 1.0,
            },
        }
    }

    fn realistic_cost() -> SimExecutor {
        SimExecutor {
            cost: CostModel {
                per_prefill_token_s: 0.000_2,
                per_decode_step_s: 0.02,
                per_seq_s: 0.000_5,
            },
        }
    }

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

    #[test]
    fn ts_section7_iteration_pattern() {
        // C and D arrive first and are decoding by the time A (3 chunks of 32) arrives;
        // B (one chunk) arrives after A's first chunk. One virtual second per sequence.
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
        let mut sim = Simulation::new(
            p,
            pool(256),
            per_seq_cost(),
            ArrivalProcess::scripted(arrivals),
        );
        let report = sim.run(secs(12.0));
        assert!(report.violations.is_empty(), "{:?}", report.violations);
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
            [(s("A"), prefill(0, 32)), (s("C"), d), (s("D"), d)]
        );
        assert_eq!(
            composition(&report.iterations[2], &labels),
            [
                (s("A"), prefill(32, 32)),
                (s("B"), prefill(0, 20)),
                (s("C"), d),
                (s("D"), d)
            ]
        );
        assert_eq!(
            composition(&report.iterations[3], &labels),
            [
                (s("A"), prefill(64, 32)),
                (s("B"), d),
                (s("C"), d),
                (s("D"), d)
            ]
        );
        // Decodes precede prefill chunks inside every batch.
        for it in &report.iterations {
            let first_prefill = it.items.iter().position(|i| i.kind != BatchKind::Decode);
            if let Some(f) = first_prefill {
                assert!(it.items[f..].iter().all(|i| i.kind != BatchKind::Decode));
            }
        }
    }

    /// 1,000 seeded Poisson arrivals; `paused` is paused for iterations 200..260.
    fn poisson_run(paused: Option<SeqId>) -> SimReport {
        let arrivals = ArrivalProcess::poisson(4.0, 42).with_limit(1000);
        let mut sim = Simulation::new(params(), pool(1024), realistic_cost(), arrivals);
        if let Some(seq) = paused {
            sim.pause_at(200, seq);
            sim.resume_at(260, seq);
        }
        sim.run(secs(100_000.0))
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

    #[test]
    fn decode_never_starved() {
        // A sequence that decodes in iterations 199, 200 and 201 when nobody pauses it.
        let free_run = poisson_run(None);
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

        let report = poisson_run(Some(SeqId(victim)));
        // The simulator records a violation whenever a decoding, unpaused sequence that was
        // neither preempted nor dropped gets no decode in an iteration.
        assert!(
            report.violations.is_empty(),
            "{:?}",
            &report.violations[..report.violations.len().min(5)]
        );
        assert_eq!(report.completed as usize + report.rejected.len(), 1000);
        assert!(report.iterations.len() > 1000);
        assert_eq!(
            decodes_of(&report, victim, 200..260),
            0,
            "a paused sequence was decoded"
        );
        assert!(
            decodes_of(&report, victim, 260..270) > 0,
            "the resumed sequence decodes again"
        );
        let again = poisson_run(Some(SeqId(victim)));
        assert_eq!(
            serde_json::to_vec(&report).unwrap(),
            serde_json::to_vec(&again).unwrap(),
            "same seed, different simulation"
        );
    }

    #[test]
    fn chunk_budget_respected() {
        let report = poisson_run(None);
        let p = params();
        for it in &report.iterations {
            let tokens: u32 = it
                .items
                .iter()
                .map(|i| match i.kind {
                    BatchKind::Prefill { len, .. } => {
                        assert!(
                            len <= p.prefill_chunk_tokens,
                            "chunk {len} in {}",
                            it.iteration
                        );
                        len
                    }
                    BatchKind::Decode => 1,
                })
                .sum();
            assert!(
                tokens <= p.max_batch_tokens,
                "{tokens} tokens in {}",
                it.iteration
            );
        }
        assert!(
            report
                .iterations
                .iter()
                .any(|it| it.items.iter().any(|i| i.kind
                    == BatchKind::Prefill {
                        start: 128,
                        len: 128
                    })),
            "long prompts are chunked"
        );

        // Without chunked prefill: an over-budget prompt is rejected at submission; one that
        // fits runs whole.
        let p = SchedulerParams {
            chunked_prefill: false,
            ..params()
        };
        let arrivals = vec![
            SimArrival::new(secs(0.0), 600, 4),
            SimArrival::new(secs(0.0), 300, 4),
        ];
        let mut sim = Simulation::new(
            p,
            pool(256),
            realistic_cost(),
            ArrivalProcess::scripted(arrivals),
        );
        let report = sim.run(secs(100.0));
        assert_eq!(
            report.rejected,
            [(Simulation::request_id(0), SubmitError::PromptTooLong)]
        );
        assert_eq!(
            report.iterations[0].items[0].kind,
            BatchKind::Prefill { start: 0, len: 300 }
        );
        assert_eq!(report.completed, 1);
    }

    #[test]
    fn preemption_by_recompute() {
        // 12 blocks of 16 tokens; each request needs up to 9 blocks, so both cannot finish
        // together.
        preemption_case(16, 12, 40, 100);
    }

    #[test]
    fn preemption_by_recompute_at_128_token_pages() {
        // The default page: 6 blocks of 128 tokens; each request needs up to 4 blocks.
        preemption_case(128, 6, 200, 300);
    }

    /// Two requests of `prompt` + `max_new` tokens on a pool of `pool_blocks` blocks of
    /// `block_tokens`, too small for both to finish together. R1 has the lower priority (larger
    /// value) and is the only victim; R2 arrives later and waits for a free running slot.
    fn preemption_case(block_tokens: u32, pool_blocks: u32, prompt: u32, max_new: u32) {
        let mut r0 = SimArrival::new(secs(0.0), prompt, max_new);
        r0.priority = Priority(0);
        let mut r1 = SimArrival::new(secs(0.0), prompt, max_new);
        r1.priority = Priority(1);
        let r2 = SimArrival::new(secs(0.5), 20, 5);
        let p = SchedulerParams {
            max_running_requests: 2,
            max_batch_tokens: 256,
            prefill_chunk_tokens: 64,
            block_tokens,
            ..params()
        };
        let mut sim = Simulation::new(
            p,
            pool_of(pool_blocks, block_tokens),
            realistic_cost(),
            ArrivalProcess::scripted(vec![r0, r1, r2]),
        );
        let report = sim.run(secs(1000.0));
        // Every sampled token sits at the next position: no duplicate or skipped token.
        assert!(report.violations.is_empty(), "{:?}", report.violations);
        assert_eq!(
            report.completed, 3,
            "every request whose KV fits the empty pool completes"
        );

        let r1_seq = Simulation::seq_id(1, 0).0;
        let r2_seq = Simulation::seq_id(2, 0).0;
        let preempted: Vec<(u64, u64)> = report
            .iterations
            .iter()
            .flat_map(|it| it.preempted.iter().map(move |s| (it.iteration, *s)))
            .collect();
        assert!(
            !preempted.is_empty(),
            "the pool is too small: someone is preempted"
        );
        assert!(
            preempted.iter().all(|(_, s)| *s == r1_seq),
            "wrong victim: {preempted:?}"
        );
        let first_preemption = preempted[0].0;

        // R1 re-prefills from position 0 over prompt + generated tokens (more than its prompt),
        // and is readmitted ahead of R2 (preempted requests go to the queue front).
        let re_prefill: Vec<(u64, u32, u32)> = report
            .iterations
            .iter()
            .filter(|it| it.iteration > first_preemption)
            .flat_map(|it| {
                it.items
                    .iter()
                    .filter(|i| i.seq == r1_seq)
                    .filter_map(move |i| match i.kind {
                        BatchKind::Prefill { start, len } => Some((it.iteration, start, len)),
                        BatchKind::Decode => None,
                    })
            })
            .collect();
        assert_eq!(re_prefill[0].1, 0, "recompute starts at position 0");
        let recomputed: u32 = re_prefill.iter().map(|x| x.2).sum();
        assert!(
            recomputed > prompt,
            "recompute covers prompt + generated, got {recomputed}"
        );
        let r2_first = report
            .iterations
            .iter()
            .find(|it| it.items.iter().any(|i| i.seq == r2_seq))
            .map(|it| it.iteration)
            .unwrap();
        assert!(
            re_prefill[0].0 <= r2_first,
            "the preempted request was not at the queue front"
        );
    }

    #[test]
    fn cancellation_frees_within_one_iteration() {
        // D and Z decode, P prefills 100 tokens in 16-token chunks, W waits for a slot.
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
        let mut sim = Simulation::new(
            p,
            pool(128),
            realistic_cost(),
            ArrivalProcess::scripted(arrivals),
        );
        let z = Simulation::seq_id(1, 0);
        sim.pause_at(3, z);
        for (k, reason) in [
            (0, CancelReason::ClientDisconnect),
            (1, CancelReason::SlowClient),
            (2, CancelReason::RequestTimeout),
            (3, CancelReason::Shutdown),
        ] {
            sim.cancel_at(5, Simulation::request_id(k), reason);
        }
        let report = sim.run(secs(100.0));
        assert!(report.violations.is_empty(), "{:?}", report.violations);
        let it4 = report
            .iterations
            .iter()
            .find(|it| it.iteration == 4)
            .unwrap();
        let p_seq = Simulation::seq_id(2, 0).0;
        assert!(
            it4.items.iter().any(|i| i.seq == p_seq
                && matches!(i.kind, BatchKind::Prefill { start, .. } if start < 100)),
            "P is between prefill chunks"
        );
        assert!(it4.items.iter().all(|i| i.seq != z.0), "Z is paused");
        assert_eq!(it4.waiting, 1, "W is waiting");
        assert!(it4.used_blocks > 0);
        let it5 = report
            .iterations
            .iter()
            .find(|it| it.iteration == 5)
            .unwrap();
        assert_eq!(it5.dropped.len(), 4);
        assert_eq!(
            it5.used_blocks, 0,
            "every block is free before the next plan returns"
        );
        assert!(it5.items.is_empty());
        assert_eq!(report.completed, 0);
    }

    #[test]
    fn bounded_under_overload() {
        // Service ≈ 8 running / (150 tokens × 0.5 s) ≈ 0.1 request/s; arrivals at 1/s = 10×.
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
        let mut sim = Simulation::new(p, pool(512), exec, arrivals);
        let report = sim.run(secs(10_000.0));
        assert!(
            report.violations.is_empty(),
            "{:?}",
            &report.violations[..report.violations.len().min(5)]
        );
        assert!(
            report.max_waiting <= 32,
            "waiting reached {}",
            report.max_waiting
        );
        assert!(
            report.max_running <= 8,
            "running reached {}",
            report.max_running
        );
        assert!(report.max_waiting == 32, "the queue fills under overload");
        assert!(
            report.rejected.len() > 5_000,
            "{} rejected",
            report.rejected.len()
        );
        assert!(
            report
                .rejected
                .iter()
                .all(|(_, e)| *e == SubmitError::QueueFull)
        );
        assert!(report.completed > 500, "{} completed", report.completed);
        assert!(sim.now() >= secs(10_000.0));
    }

    /// The tuned batch shape (P2c S-12) the Phase 2c lab configs run
    /// (`scripts/lab/phase2c-novanas-*.yaml`).
    const PHASE2C_MAX_BATCH_TOKENS: u32 = 2048;
    const PHASE2C_PREFILL_CHUNK_TOKENS: u32 = 2048;

    #[test]
    fn closed_loop_keeps_concurrency() {
        let template = SimArrival::new(secs(0.0), 100, 20);
        let arrivals = ArrivalProcess::closed_loop(template, 4, 10);
        let mut sim = Simulation::new(params(), pool(1024), realistic_cost(), arrivals);
        let report = sim.run(secs(1_000.0));
        assert!(report.violations.is_empty(), "{:?}", report.violations);
        assert_eq!(report.completed, 10);
        assert_eq!(report.max_running, 4, "four requests in flight at once");
        assert_eq!(report.ttft_s.len(), 10);
        assert_eq!(report.generated_tokens, 200);
        // The first four arrive at zero and prefill together (4 × 100 tokens ≤ 512).
        let first = report.iterations[0].duration_s;
        for t in &report.ttft_s[..4] {
            assert!((t - first).abs() < 1e-9, "ttft {t} vs {first}");
        }
        // Each later arrival comes when a request finishes, never before.
        assert!(report.ttft_s.iter().all(|&t| t > 0.0));
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
}
