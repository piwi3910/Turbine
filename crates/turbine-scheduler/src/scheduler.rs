//! Per-iteration planning (P2 §Scheduling rules): drop cancelled requests, decode every
//! decodable sequence (preempting by recompute when blocks run short), then fill the token
//! budget with prefill chunks — continuing prefills oldest first, then admissions by priority
//! then arrival.
//!
//! Token accounting of one sequence with prompt `P` and `g` generated tokens: a prefill writes
//! the KV of positions `[0, P + g)` (after a preemption `g > 0`: recompute) and its last
//! position yields the next token; a decode writes the KV of the newest token and yields the
//! next one. `BlockTable::tokens` counts written positions, including the ones planned for the
//! iteration in flight. Preemption ranks requests by `(priority, admission order)`: the victim
//! is the worst-ranked (largest priority value, then most recently admitted), and a prefill or
//! fork may preempt only requests ranked below its own, so the best-ranked request always
//! progresses — no livelock while each request's full KV fits the empty pool (checked at
//! submission).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use turbine_core::clock::Clock;
use turbine_core::config::Config;
use turbine_core::request::FinishReason;
use turbine_core::types::{BlockId, Priority, RequestId, SeqId};
use turbine_kv::{BlockPool, BlockTable, blocks_for_tokens};

use crate::metrics::SchedulerMetrics;
use crate::queue::WaitingQueue;
use crate::request::{CancelReason, PreemptReason, RequestState, SchedRequest};

/// Scheduler bounds (P2 §Configuration). `max_seq_len` and `queue_timeout` are contract
/// additions used by the submission checks and rule 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerParams {
    pub max_running_requests: u32,
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub max_queued_requests: u32,
    pub chunked_prefill: bool,
    pub block_tokens: u32,
    /// Fraction of the pool kept free when admitting (0.01).
    pub free_watermark: f64,
    pub max_seq_len: u32,
    pub queue_timeout: Duration,
}

impl SchedulerParams {
    /// The Phase 2 keys of `cfg`; `max_seq_len` is the resolved `model.max_seq_len`.
    pub fn from_config(cfg: &Config, max_seq_len: u32) -> SchedulerParams {
        SchedulerParams {
            max_running_requests: cfg.effective_max_running(),
            max_batch_tokens: cfg.scheduler.max_batch_tokens,
            prefill_chunk_tokens: cfg.scheduler.prefill_chunk_tokens,
            max_queued_requests: cfg.scheduler.max_queued_requests,
            chunked_prefill: cfg.scheduler.chunked_prefill,
            block_tokens: cfg.kv.block_tokens,
            free_watermark: 0.01,
            max_seq_len,
            queue_timeout: cfg.scheduler.queue_timeout.0,
        }
    }
}

/// Per-iteration throttles. Phase 2 always passes `Default` (GREEN: no limit); Phase 3 derives
/// them from the pressure controller's throttle plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationLimits {
    /// At most this many admissions this iteration.
    pub batch_growth_limit: Option<u32>,
    /// No admissions at all.
    pub shrink_only: bool,
    /// Share of the remaining token budget prefills may use (0..=1).
    pub prefill_budget_fraction: f64,
    /// Chunk size override (never above `prefill_chunk_tokens`).
    pub prefill_chunk_tokens: Option<u32>,
    pub admit_new: bool,
    pub start_new_prefills: bool,
}

impl Default for IterationLimits {
    fn default() -> Self {
        IterationLimits {
            batch_growth_limit: None,
            shrink_only: false,
            prefill_budget_fraction: 1.0,
            prefill_chunk_tokens: None,
            admit_new: true,
            start_new_prefills: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchKind {
    /// Write the KV of positions `[start, start + len)`.
    Prefill { start: u32, len: u32 },
    /// Write the KV of the newest token and sample the next.
    Decode,
}

/// One sequence's work in an iteration. `block_table` covers every position written so far
/// including this iteration's (`block_table.tokens` = context length after the iteration).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchItem {
    pub seq: SeqId,
    pub kind: BatchKind,
    pub block_table: BlockTable,
}

/// A choice forked from the shared prefill (`n > 1`): `dst` shares `src`'s full blocks; the
/// engine copies the partial tail `copy = (from, to)` with `copy_blocks` before the forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForkOp {
    pub src: SeqId,
    pub dst: SeqId,
    pub copy: Option<(BlockId, BlockId)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IterationPlan {
    pub iteration: u64,
    /// Decodes first, then prefill chunks.
    pub items: Vec<BatchItem>,
    pub preempted: Vec<(SeqId, PreemptReason)>,
    pub dropped: Vec<(RequestId, CancelReason)>,
    pub forks: Vec<ForkOp>,
}

impl IterationPlan {
    /// Nothing to execute: the caller sleeps until a submission or cancellation (rule 4).
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.forks.is_empty()
    }

    pub fn prefill_tokens(&self) -> u32 {
        self.items
            .iter()
            .map(|i| match i.kind {
                BatchKind::Prefill { len, .. } => len,
                BatchKind::Decode => 0,
            })
            .sum()
    }

    pub fn decode_tokens(&self) -> u32 {
        self.items
            .iter()
            .filter(|i| i.kind == BatchKind::Decode)
            .count() as u32
    }
}

/// What the engine reports after executing a plan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IterationOutcome {
    pub iteration: u64,
    pub finished: Vec<(SeqId, FinishReason)>,
    /// Tokens sampled per sequence: 1 for a decode and for a completed prefill (on the shared
    /// prefill of `n > 1`, one for every choice), 0 or absent otherwise.
    pub appended: Vec<(SeqId, u32)>,
    pub failed: Option<IterationFailure>,
}

/// The iteration failed on the device: every request in it fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationFailure {
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum SubmitError {
    #[error("queue full")]
    QueueFull,
    #[error("context length exceeded")]
    ContextLengthExceeded,
    #[error("context exceeds KV capacity")]
    ContextExceedsKvCapacity,
    #[error("prompt exceeds batch budget")]
    PromptTooLong,
    #[error("shutting down")]
    ShuttingDown,
}

impl SubmitError {
    /// Metric `reason` label and log reason code.
    pub fn as_str(self) -> &'static str {
        match self {
            SubmitError::QueueFull => "queue_full",
            SubmitError::ContextLengthExceeded => "context_length_exceeded",
            SubmitError::ContextExceedsKvCapacity => "context_exceeds_kv_capacity",
            SubmitError::PromptTooLong => "prompt_too_long",
            SubmitError::ShuttingDown => "shutting_down",
        }
    }
}

/// Body of `GET /turbine/v1/scheduler` (P2 §Data).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SchedulerSnapshot {
    pub config: SnapshotConfig,
    pub waiting: u32,
    pub prefilling: u32,
    pub decoding: u32,
    pub paused: u32,
    pub constrained: u32,
    pub iterations_total: u64,
    pub preemptions_total: u64,
    pub last_iteration: LastIteration,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SnapshotConfig {
    pub max_running_requests: u32,
    pub max_batch_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub max_queued_requests: u32,
    pub chunked_prefill: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LastIteration {
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub requests: u32,
    pub duration_ms: f64,
}

struct ReqEntry {
    req: SchedRequest,
    /// Admission number of the current admission; `None` while waiting.
    admitted: Option<u64>,
    ever_admitted: bool,
    queued_since: Duration,
    /// `seqs[0]`'s first prefill finished; the other choices fork from it.
    shared_prefill_done: bool,
    cancel: Option<CancelReason>,
}

impl ReqEntry {
    /// Preemption rank: larger is worse (dropped first).
    fn rank(&self) -> (Priority, u64) {
        (self.req.priority, self.admitted.unwrap_or(u64::MAX))
    }
}

struct SeqEntry {
    request: RequestId,
    state: RequestState,
    table: BlockTable,
    generated: u32,
    /// Positions to prefill: `prompt + generated` at admission.
    target: u32,
    /// A choice waiting to fork from `seqs[0]`'s shared prefill.
    awaiting_fork: bool,
    /// Paused while not decoding: becomes `Paused` on reaching `Decoding`.
    pause_pending: bool,
}

impl SeqEntry {
    fn set_state(&mut self, to: RequestState, seq: SeqId) {
        if let Err(e) = self.state.transition(to) {
            tracing::error!(event = "scheduler_bug", seq = seq.0, error = %e, "illegal transition ignored");
            debug_assert!(false, "{e}");
        }
    }

    /// Enter `Decoding` (or `Paused` when a pause arrived earlier).
    fn start_decoding(&mut self, seq: SeqId) {
        self.set_state(RequestState::Decoding, seq);
        if self.pause_pending {
            self.pause_pending = false;
            self.set_state(RequestState::Paused, seq);
        }
    }
}

/// The Phase 2 continuous-batching scheduler. Single-threaded: owned by the engine thread.
pub struct Scheduler {
    params: SchedulerParams,
    clock: Arc<dyn Clock>,
    metrics: Option<SchedulerMetrics>,
    requests: HashMap<RequestId, ReqEntry>,
    /// Admitted requests in admission order.
    running: Vec<RequestId>,
    queue: WaitingQueue,
    seqs: HashMap<SeqId, SeqEntry>,
    /// Items of the plan awaiting `complete`.
    in_flight: Vec<(SeqId, BatchKind)>,
    in_flight_requests: Vec<RequestId>,
    iteration: u64,
    admissions: u64,
    preemptions_total: u64,
    shutting_down: bool,
    plan_started: Duration,
    last: LastIteration,
}

impl Scheduler {
    pub fn new(p: SchedulerParams, clock: Arc<dyn Clock>) -> Scheduler {
        Scheduler {
            params: p,
            clock,
            metrics: None,
            requests: HashMap::new(),
            running: Vec::new(),
            queue: WaitingQueue::new(),
            seqs: HashMap::new(),
            in_flight: Vec::new(),
            in_flight_requests: Vec::new(),
            iteration: 0,
            admissions: 0,
            preemptions_total: 0,
            shutting_down: false,
            plan_started: Duration::ZERO,
            last: LastIteration::default(),
        }
    }

    /// Publish metrics to `m` (the simulator runs without).
    pub fn with_metrics(mut self, m: SchedulerMetrics) -> Scheduler {
        self.metrics = Some(m);
        self.publish_gauges();
        self
    }

    pub fn params(&self) -> &SchedulerParams {
        &self.params
    }

    /// Submission checks (P2 §Scheduling rules), then queue by priority and arrival.
    pub fn submit(&mut self, r: SchedRequest, pool_total_blocks: u32) -> Result<(), SubmitError> {
        let result = self.check_submission(&r, pool_total_blocks);
        match result {
            Err(e) => {
                tracing::info!(event = "reject", request_id = %r.id.0, reason = e.as_str(), "request rejected");
                if let Some(m) = &self.metrics {
                    m.admission("rejected", e.as_str());
                }
                Err(e)
            }
            Ok(()) => {
                let now = self.clock.now_mono();
                self.queue.push(r.id, r.priority, r.arrival);
                for &seq in &r.seqs {
                    self.seqs.insert(
                        seq,
                        SeqEntry {
                            request: r.id,
                            state: RequestState::Waiting,
                            table: BlockTable::new(),
                            generated: 0,
                            target: r.prompt_len,
                            awaiting_fork: false,
                            pause_pending: false,
                        },
                    );
                }
                self.requests.insert(
                    r.id,
                    ReqEntry {
                        req: r,
                        admitted: None,
                        ever_admitted: false,
                        queued_since: now,
                        shared_prefill_done: false,
                        cancel: None,
                    },
                );
                if let Some(m) = &self.metrics {
                    m.admission("queued", "ok");
                }
                self.publish_gauges();
                Ok(())
            }
        }
    }

    fn check_submission(
        &self,
        r: &SchedRequest,
        pool_total_blocks: u32,
    ) -> Result<(), SubmitError> {
        let p = &self.params;
        if self.shutting_down {
            return Err(SubmitError::ShuttingDown);
        }
        let total = u64::from(r.prompt_len) + u64::from(r.max_new_tokens);
        if total > u64::from(p.max_seq_len) {
            return Err(SubmitError::ContextLengthExceeded);
        }
        if !p.chunked_prefill && r.prompt_len > p.max_batch_tokens {
            return Err(SubmitError::PromptTooLong);
        }
        // Every choice's KV at completion; choices share the prompt's full blocks.
        let bt = p.block_tokens.max(1);
        let one = u64::from(blocks_for_tokens(
            r.prompt_len.saturating_add(r.max_new_tokens),
            bt,
        ));
        let shared = u64::from(r.prompt_len / bt);
        let extra_choices = r.seqs.len().saturating_sub(1) as u64;
        if one + extra_choices * (one - shared.min(one)) > u64::from(pool_total_blocks) {
            return Err(SubmitError::ContextExceedsKvCapacity);
        }
        if self.queue.len() >= p.max_queued_requests as usize {
            return Err(SubmitError::QueueFull);
        }
        Ok(())
    }

    /// Mark a request cancelled; rule 1 of the next `plan` drops it and frees its blocks.
    pub fn cancel(&mut self, id: RequestId, reason: CancelReason) {
        if let Some(e) = self.requests.get_mut(&id) {
            e.cancel.get_or_insert(reason);
        }
    }

    /// The client's output channel is full: stop decoding `seq`, keep its KV.
    pub fn pause(&mut self, seq: SeqId) {
        let Some(e) = self.seqs.get_mut(&seq) else {
            return;
        };
        match e.state {
            RequestState::Decoding => {
                e.set_state(RequestState::Paused, seq);
                tracing::info!(event = "pause", request_id = %e.request.0, seq = seq.0, reason = "output_channel_full", "sequence paused");
            }
            s if s.is_live() && s != RequestState::Paused => e.pause_pending = true,
            _ => {}
        }
        self.publish_gauges();
    }

    /// The output channel drained: `seq` decodes again.
    pub fn resume(&mut self, seq: SeqId) {
        let Some(e) = self.seqs.get_mut(&seq) else {
            return;
        };
        e.pause_pending = false;
        if e.state == RequestState::Paused {
            e.set_state(RequestState::Decoding, seq);
        }
        self.publish_gauges();
    }

    /// New submissions get `ShuttingDown`; admitted and queued requests continue.
    pub fn begin_shutdown(&mut self) {
        self.shutting_down = true;
    }

    /// No request is queued or running.
    pub fn is_idle(&self) -> bool {
        self.requests.is_empty()
    }

    /// State of a sequence the scheduler still tracks.
    pub fn seq_state(&self, seq: SeqId) -> Option<RequestState> {
        self.seqs.get(&seq).map(|e| e.state)
    }

    /// Positions `seq`'s current prefill must reach; `Some` only while it prefills. A prefill
    /// item ending at this position completes the prefill and yields a token.
    pub fn prefill_target(&self, seq: SeqId) -> Option<u32> {
        self.seqs
            .get(&seq)
            .filter(|e| e.state == RequestState::Prefilling && !e.awaiting_fork)
            .map(|e| e.target)
    }

    /// One iteration's batch (P2 §Scheduling rules 1–3).
    pub fn plan(&mut self, pool: &mut BlockPool, limits: &IterationLimits) -> IterationPlan {
        self.iteration += 1;
        let now = self.clock.now_mono();
        self.plan_started = now;
        let mut plan = IterationPlan {
            iteration: self.iteration,
            ..IterationPlan::default()
        };
        let mut in_plan: HashSet<RequestId> = HashSet::new();
        let mut preempted_now: HashSet<RequestId> = HashSet::new();

        // (1) Drop cancelled and queue-timed-out requests; their blocks return first.
        self.drop_cancelled(pool, now, &mut plan);

        // (2) Every decodable sequence decodes; preempt until the pool covers them.
        self.plan_decodes(pool, &mut plan, &mut in_plan, &mut preempted_now);

        // Forks of finished shared prefills (n > 1) wait for blocks, never re-prefill.
        self.plan_forks(pool, &mut plan, &mut in_plan, &mut preempted_now);

        // (3) Prefill: continuing prefills oldest first, then admissions.
        let decode_tokens = plan.decode_tokens();
        let fraction = limits.prefill_budget_fraction.clamp(0.0, 1.0);
        let mut budget = (f64::from(self.params.max_batch_tokens.saturating_sub(decode_tokens))
            * fraction)
            .floor() as u32;
        let chunk_cap = self.chunk_cap(limits);
        for id in self.running.clone() {
            for seq in self.prefill_seqs(id) {
                if budget == 0 || !self.is_running(id) {
                    break;
                }
                self.schedule_chunk(
                    pool,
                    seq,
                    &mut budget,
                    chunk_cap,
                    true,
                    &mut plan,
                    &mut in_plan,
                    &mut preempted_now,
                );
            }
        }
        if limits.admit_new && limits.start_new_prefills && !limits.shrink_only {
            let mut admitted_now = 0u32;
            while budget > 0
                && (self.running.len() as u32) < self.params.max_running_requests
                && limits.batch_growth_limit.is_none_or(|g| admitted_now < g)
            {
                let Some(head) = self.queue.peek() else {
                    break;
                };
                if preempted_now.contains(&head) || !self.admissible(pool, head, budget, chunk_cap)
                {
                    break;
                }
                self.admit(head, now);
                admitted_now += 1;
                for seq in self.prefill_seqs(head) {
                    if budget == 0 {
                        break;
                    }
                    self.schedule_chunk(
                        pool,
                        seq,
                        &mut budget,
                        chunk_cap,
                        false,
                        &mut plan,
                        &mut in_plan,
                        &mut preempted_now,
                    );
                }
            }
        }

        self.in_flight = plan.items.iter().map(|i| (i.seq, i.kind)).collect();
        self.in_flight_requests = self
            .running
            .iter()
            .copied()
            .filter(|id| in_plan.contains(id))
            .collect();
        self.last = LastIteration {
            prefill_tokens: plan.prefill_tokens(),
            decode_tokens,
            requests: plan.items.len() as u32,
            duration_ms: 0.0,
        };
        self.publish_gauges();
        plan
    }

    /// Apply an executed plan: prefill completion, appended tokens, finished sequences, or
    /// the failure of every request in the iteration.
    pub fn complete(&mut self, pool: &mut BlockPool, outcome: IterationOutcome) {
        if outcome.iteration != self.iteration {
            tracing::warn!(
                event = "scheduler_outcome_mismatch",
                expected = self.iteration,
                got = outcome.iteration,
                "outcome for another iteration"
            );
        }
        let elapsed = self.clock.now_mono().saturating_sub(self.plan_started);
        self.last.duration_ms = elapsed.as_secs_f64() * 1000.0;
        if !self.in_flight.is_empty()
            && let Some(m) = &self.metrics
        {
            m.iteration(
                elapsed.as_secs_f64(),
                self.last.prefill_tokens,
                self.last.decode_tokens,
                self.last.requests,
            );
        }
        let in_flight = std::mem::take(&mut self.in_flight);
        let in_flight_requests = std::mem::take(&mut self.in_flight_requests);

        if let Some(failure) = outcome.failed {
            for id in in_flight_requests {
                if self.requests.contains_key(&id) {
                    tracing::warn!(event = "fail", request_id = %id.0, reason = "iteration_failed", message = %failure.message, "request failed with its iteration");
                    self.remove_request(pool, id, RequestState::Failed);
                }
            }
            self.publish_gauges();
            return;
        }

        for (seq, n) in outcome.appended {
            if let Some(e) = self.seqs.get_mut(&seq) {
                e.generated += n;
            }
        }
        for (seq, kind) in in_flight {
            if !matches!(kind, BatchKind::Prefill { .. }) {
                continue;
            }
            let Some(e) = self.seqs.get(&seq) else {
                continue;
            };
            if e.state != RequestState::Prefilling || e.table.tokens < e.target {
                continue;
            }
            let id = e.request;
            let Some(r) = self.requests.get_mut(&id) else {
                continue;
            };
            if r.req.seqs.first() == Some(&seq) && !r.shared_prefill_done {
                r.shared_prefill_done = true;
                let forks_waiting = r.req.seqs.iter().skip(1).any(|s| {
                    self.seqs
                        .get(s)
                        .is_some_and(|c| c.awaiting_fork && c.state.is_live())
                });
                if forks_waiting {
                    continue; // decodes once its choices have forked
                }
            }
            if let Some(e) = self.seqs.get_mut(&seq) {
                e.start_decoding(seq);
            }
        }
        for (seq, _reason) in outcome.finished {
            self.finish_seq(pool, seq);
        }
        self.publish_gauges();
    }

    /// Body of `GET /turbine/v1/scheduler`.
    pub fn snapshot(&self) -> SchedulerSnapshot {
        let (mut prefilling, mut decoding, mut paused) = (0, 0, 0);
        for id in &self.running {
            match self.request_state(*id) {
                RequestState::Prefilling => prefilling += 1,
                RequestState::Decoding => decoding += 1,
                RequestState::Paused => paused += 1,
                _ => {}
            }
        }
        let p = &self.params;
        SchedulerSnapshot {
            config: SnapshotConfig {
                max_running_requests: p.max_running_requests,
                max_batch_tokens: p.max_batch_tokens,
                prefill_chunk_tokens: p.prefill_chunk_tokens,
                max_queued_requests: p.max_queued_requests,
                chunked_prefill: p.chunked_prefill,
            },
            waiting: self.queue.len() as u32,
            prefilling,
            decoding,
            paused,
            constrained: self.requests.values().filter(|r| r.req.constrained).count() as u32,
            iterations_total: self.iteration,
            preemptions_total: self.preemptions_total,
            last_iteration: self.last,
        }
    }

    // ---- rules -------------------------------------------------------------------------

    fn drop_cancelled(&mut self, pool: &mut BlockPool, now: Duration, plan: &mut IterationPlan) {
        let candidates: Vec<RequestId> = self
            .running
            .iter()
            .copied()
            .chain(self.queue.iter())
            .collect();
        for id in candidates {
            let Some(e) = self.requests.get(&id) else {
                continue;
            };
            let timed_out = e.admitted.is_none()
                && !e.ever_admitted
                && now.saturating_sub(e.queued_since) > self.params.queue_timeout;
            let reason = e
                .cancel
                .or_else(|| {
                    e.req
                        .cancel
                        .is_cancelled()
                        .then_some(CancelReason::ClientDisconnect)
                })
                .or_else(|| timed_out.then_some(CancelReason::QueueTimeout));
            let Some(reason) = reason else {
                continue;
            };
            self.remove_request(pool, id, RequestState::Cancelled);
            plan.dropped.push((id, reason));
            if reason == CancelReason::QueueTimeout {
                tracing::info!(event = "reject", request_id = %id.0, reason = reason.as_str(), "queued request timed out");
                if let Some(m) = &self.metrics {
                    m.admission("rejected", reason.as_str());
                }
            } else {
                tracing::info!(event = "cancel", request_id = %id.0, reason = reason.as_str(), "request cancelled");
                if let Some(m) = &self.metrics {
                    m.cancelled(reason.as_str());
                }
            }
        }
    }

    fn plan_decodes(
        &mut self,
        pool: &mut BlockPool,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        preempted_now: &mut HashSet<RequestId>,
    ) {
        let bt = self.params.block_tokens;
        loop {
            let need: u32 = self
                .decode_set()
                .iter()
                .map(|s| self.seqs[s].table.blocks_needed(1, bt))
                .sum();
            if need <= pool.free_blocks() {
                break;
            }
            let Some(victim) = self.worst_running(|_| true) else {
                break;
            };
            self.preempt(pool, victim, plan, preempted_now);
        }
        for seq in self.decode_set() {
            let e = self
                .seqs
                .get_mut(&seq)
                .expect("decode set holds tracked sequences");
            let need = e.table.blocks_needed(1, bt);
            let Ok(blocks) = pool.allocate(need) else {
                // Unreachable: the loop above made the pool cover every decode.
                tracing::error!(
                    event = "scheduler_bug",
                    seq = seq.0,
                    "decode lacks blocks after preemption"
                );
                continue;
            };
            e.table.blocks.extend(blocks);
            e.table.tokens += 1;
            in_plan.insert(e.request);
            plan.items.push(BatchItem {
                seq,
                kind: BatchKind::Decode,
                block_table: e.table.clone(),
            });
        }
    }

    fn plan_forks(
        &mut self,
        pool: &mut BlockPool,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        preempted_now: &mut HashSet<RequestId>,
    ) {
        for id in self.running.clone() {
            let Some(r) = self.requests.get(&id) else {
                continue;
            };
            if !r.shared_prefill_done || r.admitted.is_none() {
                continue;
            }
            let seqs = r.req.seqs.clone();
            let rank = r.rank();
            let parent = seqs[0];
            let children: Vec<SeqId> = seqs[1..]
                .iter()
                .copied()
                .filter(|s| {
                    self.seqs
                        .get(s)
                        .is_some_and(|c| c.awaiting_fork && c.state.is_live())
                })
                .collect();
            let mut all_forked = true;
            for child in children {
                let parent_table = self.seqs[&parent].table.clone();
                let mut forked = pool.fork(&parent_table);
                if forked.is_err() {
                    // Make room from requests ranked below this one that are not in the plan.
                    while let Some(victim) =
                        self.worst_running(|v| v.0 != id && !in_plan.contains(&v.0) && v.1 > rank)
                    {
                        self.preempt(pool, victim, plan, preempted_now);
                        if pool.free_blocks() > 0 {
                            break;
                        }
                    }
                    forked = pool.fork(&parent_table);
                }
                let Ok((table, copy)) = forked else {
                    all_forked = false;
                    break; // waits for blocks; the prompt is not re-prefilled
                };
                let c = self.seqs.get_mut(&child).expect("child tracked");
                c.table = table;
                c.awaiting_fork = false;
                c.start_decoding(child);
                in_plan.insert(id);
                plan.forks.push(ForkOp {
                    src: parent,
                    dst: child,
                    copy,
                });
            }
            if all_forked {
                let p = self.seqs.get_mut(&parent).expect("parent tracked");
                if p.state == RequestState::Prefilling {
                    p.start_decoding(parent);
                } else if !p.state.is_live() && !p.table.blocks.is_empty() {
                    // The parent finished at its prefill; its table was held for the forks.
                    let blocks = std::mem::take(&mut p.table);
                    pool.release(&blocks.blocks);
                    self.retire_if_done(id);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn schedule_chunk(
        &mut self,
        pool: &mut BlockPool,
        seq: SeqId,
        budget: &mut u32,
        chunk_cap: u32,
        may_preempt: bool,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        preempted_now: &mut HashSet<RequestId>,
    ) -> bool {
        let bt = self.params.block_tokens;
        let e = &self.seqs[&seq];
        let id = e.request;
        let remaining = e.target.saturating_sub(e.table.tokens);
        let len = remaining.min(chunk_cap).min(*budget);
        if len == 0 || (self.whole_prefill_required(remaining) && len < remaining) {
            return false;
        }
        let need = e.table.blocks_needed(len, bt);
        if need > pool.free_blocks() && may_preempt {
            let rank = self.requests[&id].rank();
            while need > pool.free_blocks() {
                let Some(victim) =
                    self.worst_running(|v| v.0 != id && !in_plan.contains(&v.0) && v.1 > rank)
                else {
                    break;
                };
                self.preempt(pool, victim, plan, preempted_now);
            }
        }
        let Ok(blocks) = pool.allocate(need) else {
            return false; // new prefills wait for blocks
        };
        let e = self.seqs.get_mut(&seq).expect("sequence tracked");
        e.table.blocks.extend(blocks);
        let start = e.table.tokens;
        e.table.tokens += len;
        *budget -= len;
        in_plan.insert(id);
        plan.items.push(BatchItem {
            seq,
            kind: BatchKind::Prefill { start, len },
            block_table: e.table.clone(),
        });
        true
    }

    /// Would admitting `id` now fit its first chunk plus the free-block watermark?
    fn admissible(&self, pool: &BlockPool, id: RequestId, budget: u32, chunk_cap: u32) -> bool {
        let Some(r) = self.requests.get(&id) else {
            return false;
        };
        let first = r.req.seqs.iter().find_map(|s| {
            let e = self.seqs.get(s)?;
            let forking = !r.shared_prefill_done && Some(s) != r.req.seqs.first();
            (e.state.is_live() && !forking).then_some(r.req.prompt_len + e.generated)
        });
        let Some(target) = first else {
            return false;
        };
        let len = target.min(chunk_cap).min(budget);
        if len == 0 || (self.whole_prefill_required(target) && len < target) {
            return false;
        }
        let watermark = (f64::from(pool.total_blocks()) * self.params.free_watermark).ceil() as u32;
        pool.free_blocks() >= blocks_for_tokens(len, self.params.block_tokens) + watermark
    }

    fn admit(&mut self, id: RequestId, now: Duration) {
        self.queue.remove(id);
        let admission = self.admissions;
        self.admissions += 1;
        self.running.push(id);
        let r = self.requests.get_mut(&id).expect("queued request tracked");
        r.admitted = Some(admission);
        if !r.ever_admitted {
            r.ever_admitted = true;
            if let Some(m) = &self.metrics {
                m.queue_wait(now.saturating_sub(r.queued_since).as_secs_f64());
            }
        }
        let shared_done = r.shared_prefill_done;
        let prompt = r.req.prompt_len;
        let seqs = r.req.seqs.clone();
        for (i, seq) in seqs.iter().enumerate() {
            let e = self.seqs.get_mut(seq).expect("sequence tracked");
            if e.state != RequestState::Waiting {
                continue;
            }
            e.set_state(RequestState::Prefilling, *seq);
            e.awaiting_fork = i > 0 && !shared_done;
            e.target = prompt + e.generated;
        }
    }

    fn preempt(
        &mut self,
        pool: &mut BlockPool,
        id: RequestId,
        plan: &mut IterationPlan,
        preempted_now: &mut HashSet<RequestId>,
    ) {
        self.running.retain(|r| *r != id);
        let r = self.requests.get_mut(&id).expect("running request tracked");
        r.admitted = None;
        let seqs = r.req.seqs.clone();
        for seq in seqs {
            let e = self.seqs.get_mut(&seq).expect("sequence tracked");
            let table = std::mem::take(&mut e.table);
            pool.release(&table.blocks);
            if matches!(
                e.state,
                RequestState::Prefilling | RequestState::Decoding | RequestState::Paused
            ) {
                e.set_state(RequestState::Waiting, seq);
                plan.preempted.push((seq, PreemptReason::KvExhausted));
            }
        }
        self.queue.push_front(id);
        preempted_now.insert(id);
        self.preemptions_total += 1;
        let reason = PreemptReason::KvExhausted.as_str();
        tracing::info!(event = "preempt", request_id = %id.0, reason, "request preempted by recompute");
        if let Some(m) = &self.metrics {
            m.preempted(reason);
        }
    }

    fn finish_seq(&mut self, pool: &mut BlockPool, seq: SeqId) {
        let Some(e) = self.seqs.get(&seq) else {
            return;
        };
        if !e.state.is_live() {
            return;
        }
        let id = e.request;
        let hold_for_forks = self.requests.get(&id).is_some_and(|r| {
            r.req.seqs.first() == Some(&seq)
                && r.req.seqs[1..].iter().any(|s| {
                    self.seqs
                        .get(s)
                        .is_some_and(|c| c.awaiting_fork && c.state.is_live())
                })
        });
        let e = self.seqs.get_mut(&seq).expect("sequence tracked");
        e.awaiting_fork = false;
        e.set_state(RequestState::Finished, seq);
        if !hold_for_forks {
            let table = std::mem::take(&mut e.table);
            pool.release(&table.blocks);
        }
        self.retire_if_done(id);
    }

    /// Forget a request once every sequence is terminal and holds no block.
    fn retire_if_done(&mut self, id: RequestId) {
        let Some(r) = self.requests.get(&id) else {
            return;
        };
        let done = r.req.seqs.iter().all(|s| {
            self.seqs
                .get(s)
                .is_none_or(|e| !e.state.is_live() && e.table.blocks.is_empty())
        });
        if done {
            let r = self.requests.remove(&id).expect("checked above");
            for s in &r.req.seqs {
                self.seqs.remove(s);
            }
            self.running.retain(|x| *x != id);
            self.queue.remove(id);
        }
    }

    /// Release every block of `id`, move its live sequences to `terminal` and forget it.
    fn remove_request(&mut self, pool: &mut BlockPool, id: RequestId, terminal: RequestState) {
        let Some(r) = self.requests.remove(&id) else {
            return;
        };
        for seq in &r.req.seqs {
            if let Some(mut e) = self.seqs.remove(seq) {
                pool.release(&e.table.blocks);
                if e.state.is_live() {
                    e.set_state(terminal, *seq);
                }
            }
        }
        self.running.retain(|x| *x != id);
        self.queue.remove(id);
    }

    // ---- helpers -----------------------------------------------------------------------

    fn is_running(&self, id: RequestId) -> bool {
        self.requests.get(&id).is_some_and(|r| r.admitted.is_some())
    }

    /// Decoding (not paused) sequences of running requests, in admission order.
    fn decode_set(&self) -> Vec<SeqId> {
        self.running
            .iter()
            .flat_map(|id| self.requests[id].req.seqs.iter().copied())
            .filter(|s| {
                self.seqs
                    .get(s)
                    .is_some_and(|e| e.state == RequestState::Decoding)
            })
            .collect()
    }

    /// Sequences of `id` with prompt positions left to prefill (not forking).
    fn prefill_seqs(&self, id: RequestId) -> Vec<SeqId> {
        let Some(r) = self.requests.get(&id) else {
            return Vec::new();
        };
        r.req
            .seqs
            .iter()
            .copied()
            .filter(|s| {
                self.seqs.get(s).is_some_and(|e| {
                    e.state == RequestState::Prefilling
                        && !e.awaiting_fork
                        && e.table.tokens < e.target
                })
            })
            .collect()
    }

    /// The worst-ranked running request accepted by `eligible((id, rank))`.
    fn worst_running(
        &self,
        eligible: impl Fn((RequestId, (Priority, u64))) -> bool,
    ) -> Option<RequestId> {
        self.running
            .iter()
            .map(|id| (*id, self.requests[id].rank()))
            .filter(|c| eligible(*c))
            .max_by_key(|(_, rank)| *rank)
            .map(|(id, _)| id)
    }

    fn chunk_cap(&self, limits: &IterationLimits) -> u32 {
        let p = &self.params;
        let cap = if p.chunked_prefill {
            p.prefill_chunk_tokens
        } else {
            p.max_batch_tokens
        };
        limits
            .prefill_chunk_tokens
            .map_or(cap, |c| c.min(cap))
            .max(1)
    }

    /// Without chunked prefill a prompt that fits one iteration's budget runs whole; longer
    /// recomputes after a preemption (prompt + generated) are chunked by necessity.
    fn whole_prefill_required(&self, remaining: u32) -> bool {
        !self.params.chunked_prefill && remaining <= self.params.max_batch_tokens
    }

    /// Aggregate state of a running request: prefilling, else decoding, else paused.
    fn request_state(&self, id: RequestId) -> RequestState {
        let states: Vec<RequestState> = self.requests[&id]
            .req
            .seqs
            .iter()
            .filter_map(|s| self.seqs.get(s).map(|e| e.state))
            .collect();
        [
            RequestState::Prefilling,
            RequestState::Decoding,
            RequestState::Paused,
        ]
        .into_iter()
        .find(|s| states.contains(s))
        .unwrap_or(RequestState::Finished)
    }

    fn publish_gauges(&self) {
        let Some(m) = &self.metrics else {
            return;
        };
        let s = self.snapshot();
        m.set_active("prefilling", s.prefilling);
        m.set_active("decoding", s.decoding);
        m.set_active("paused", s.paused);
        m.set_queued(s.waiting);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use smallvec::smallvec;
    use turbine_core::clock::FakeClock;
    use turbine_core::types::{DType, DeviceId, KvLayout};
    use turbine_kv::{BlockPool, BlockPoolConfig};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;

    fn params() -> SchedulerParams {
        SchedulerParams {
            max_running_requests: 8,
            max_batch_tokens: 64,
            prefill_chunk_tokens: 32,
            max_queued_requests: 2,
            chunked_prefill: true,
            block_tokens: 16,
            free_watermark: 0.01,
            max_seq_len: 4096,
            queue_timeout: Duration::from_secs(60),
        }
    }

    /// A pool of `n` accounting-only blocks (zero bytes per block).
    fn pool(n: u32) -> BlockPool {
        let layout = KvLayout {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: DType::BF16,
            block_tokens: 16,
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

    fn request(n: u128, seq: u64, prompt: u32, max_new: u32) -> SchedRequest {
        SchedRequest::new(
            RequestId(uuid::Uuid::from_u128(n)),
            smallvec![SeqId(seq)],
            prompt,
            max_new,
            16,
        )
    }

    fn kinds(plan: &IterationPlan) -> Vec<(u64, BatchKind)> {
        plan.items.iter().map(|i| (i.seq.0, i.kind)).collect()
    }

    /// Completes `plan`: every finished prefill and every decode appends one token.
    fn complete_all(s: &mut Scheduler, pool: &mut BlockPool, plan: &IterationPlan, done: &[u64]) {
        let appended = plan
            .items
            .iter()
            .filter(|i| match i.kind {
                BatchKind::Decode => true,
                BatchKind::Prefill { start, len } => s.prefill_target(i.seq) == Some(start + len),
            })
            .map(|i| (i.seq, 1))
            .collect();
        let finished = done
            .iter()
            .map(|s| (SeqId(*s), FinishReason::Stop))
            .collect();
        s.complete(
            pool,
            IterationOutcome {
                iteration: plan.iteration,
                finished,
                appended,
                failed: None,
            },
        );
    }

    #[test]
    fn decode_first_then_chunked_prefill() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(params(), Arc::new(clock.clone()));
        let mut p = pool(64);

        // Two short requests prefill in one iteration and then decode.
        s.submit(request(1, 1, 8, 50), p.total_blocks()).unwrap();
        s.submit(request(2, 2, 8, 50), p.total_blocks()).unwrap();
        let first = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&first),
            [
                (1, BatchKind::Prefill { start: 0, len: 8 }),
                (2, BatchKind::Prefill { start: 0, len: 8 })
            ]
        );
        complete_all(&mut s, &mut p, &first, &[]);

        // Two waiting 100-token prompts: budget 64 − 2 decodes = 62 → chunks of 32 and 30.
        s.submit(request(3, 3, 100, 10), p.total_blocks()).unwrap();
        s.submit(request(4, 4, 100, 10), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Decode),
                (2, BatchKind::Decode),
                (3, BatchKind::Prefill { start: 0, len: 32 }),
                (4, BatchKind::Prefill { start: 0, len: 30 }),
            ]
        );
        // Block tables cover every token of the iteration.
        assert_eq!(plan.items[0].block_table.tokens, 9);
        assert_eq!(plan.items[2].block_table.blocks.len(), 2);
        assert_eq!(plan.items[3].block_table.blocks.len(), 2);
        assert_eq!(p.used_blocks(), 1 + 1 + 2 + 2);
        complete_all(&mut s, &mut p, &plan, &[]);

        // Next iteration: request 3 continues first (oldest), request 4 gets the rest.
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Decode),
                (2, BatchKind::Decode),
                (3, BatchKind::Prefill { start: 32, len: 32 }),
                (4, BatchKind::Prefill { start: 30, len: 30 }),
            ]
        );
        complete_all(&mut s, &mut p, &plan, &[1]);
        assert_eq!(
            p.used_blocks(),
            1 + 4 + 4,
            "finished request 1 freed its block"
        );

        // The queue holds 2: the third submission is refused.
        s.submit(request(5, 5, 10, 10), p.total_blocks()).unwrap();
        s.submit(request(6, 6, 10, 10), p.total_blocks()).unwrap();
        assert!(matches!(
            s.submit(request(7, 7, 10, 10), p.total_blocks()),
            Err(SubmitError::QueueFull)
        ));
        // KV at completion larger than the whole pool (64 blocks = 1024 tokens).
        assert!(matches!(
            s.submit(request(8, 8, 1000, 100), p.total_blocks()),
            Err(SubmitError::ContextExceedsKvCapacity)
        ));
        assert!(matches!(
            s.submit(request(9, 9, 4000, 100), p.total_blocks()),
            Err(SubmitError::ContextLengthExceeded)
        ));
        s.begin_shutdown();
        assert!(matches!(
            s.submit(request(10, 10, 10, 10), p.total_blocks()),
            Err(SubmitError::ShuttingDown)
        ));
    }

    #[test]
    fn cancel_pause_preempt_and_observability() {
        let clock = FakeClock::new(Duration::ZERO);
        let reg = MetricsRegistry::new();
        let mut s = Scheduler::new(
            SchedulerParams {
                max_queued_requests: 8,
                ..params()
            },
            Arc::new(clock.clone()),
        )
        .with_metrics(SchedulerMetrics::register(&reg));
        // 5 blocks of 16 tokens; the 1 % watermark rounds up to 1 block.
        let mut p = pool(5);

        let mut low = request(1, 1, 30, 30);
        low.priority = Priority(1);
        s.submit(low, p.total_blocks()).unwrap();
        s.submit(request(2, 2, 30, 30), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.items.len(), 2, "both admitted (2 blocks each)");
        assert_eq!(p.free_blocks(), 1);
        complete_all(&mut s, &mut p, &plan, &[]);

        // Both write positions 30 and 31 inside their second block: no new block needed.
        // Decodes run in admission order; priority 0 (request 2) was admitted first.
        for _ in 0..2 {
            let plan = s.plan(&mut p, &IterationLimits::default());
            assert_eq!(
                kinds(&plan),
                [(2, BatchKind::Decode), (1, BatchKind::Decode)]
            );
            complete_all(&mut s, &mut p, &plan, &[]);
        }

        // Position 32 needs a third block each and the pool is empty: the lower-priority
        // request (priority 1) is preempted back to the queue; the other decodes.
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.preempted, [(SeqId(1), PreemptReason::KvExhausted)]);
        assert_eq!(kinds(&plan), [(2, BatchKind::Decode)]);
        assert_eq!(s.snapshot().waiting, 1);
        complete_all(&mut s, &mut p, &plan, &[]);

        // Pause keeps the blocks and removes the sequence from the decode set.
        s.pause(SeqId(2));
        let snap = s.snapshot();
        assert_eq!((snap.paused, snap.decoding), (1, 0));
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert!(
            plan.items.iter().all(|i| i.seq != SeqId(2)),
            "a paused sequence is not decoded"
        );
        complete_all(&mut s, &mut p, &plan, &[]);
        s.resume(SeqId(2));

        // Cancel frees every block before the next plan returns.
        s.cancel(
            RequestId(uuid::Uuid::from_u128(2)),
            CancelReason::SlowClient,
        );
        s.cancel(
            RequestId(uuid::Uuid::from_u128(1)),
            CancelReason::ClientDisconnect,
        );
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.dropped.len(), 2);
        assert!(plan.items.is_empty());
        assert_eq!(p.used_blocks(), 0);
        assert!(s.is_idle());

        let snap = serde_json::to_value(s.snapshot()).unwrap();
        for key in [
            "config",
            "waiting",
            "prefilling",
            "decoding",
            "paused",
            "constrained",
            "iterations_total",
            "preemptions_total",
            "last_iteration",
        ] {
            assert!(snap.get(key).is_some(), "snapshot lacks {key}: {snap}");
        }
        assert_eq!(snap["preemptions_total"], 1);
        assert_eq!(snap["config"]["max_batch_tokens"], 64);
        assert!(snap["last_iteration"].get("duration_ms").is_some());

        let text = reg.render().unwrap();
        for needle in [
            "turbine_admission_total{outcome=\"queued\",reason=\"ok\"} 2",
            "turbine_preemptions_total{reason=\"kv_exhausted\"} 1",
            "turbine_requests_cancelled_total{reason=\"slow_client\"} 1",
            "turbine_requests_cancelled_total{reason=\"client_disconnect\"} 1",
            "turbine_requests_queued 0",
            "turbine_requests_active{state=\"decoding\"} 0",
            "turbine_queue_wait_seconds_count",
            "turbine_iteration_seconds_count",
            "turbine_iteration_tokens_count{phase=\"prefill\"}",
            "turbine_batch_requests_count",
        ] {
            assert!(text.contains(needle), "missing {needle} in\n{text}");
        }
    }

    #[test]
    fn queue_timeout_and_rejections_are_counted() {
        let clock = FakeClock::new(Duration::ZERO);
        let reg = MetricsRegistry::new();
        let mut s = Scheduler::new(
            SchedulerParams {
                max_running_requests: 1,
                ..params()
            },
            Arc::new(clock.clone()),
        )
        .with_metrics(SchedulerMetrics::register(&reg));
        let mut p = pool(64);
        s.submit(request(1, 1, 8, 500), p.total_blocks()).unwrap();
        s.submit(request(2, 2, 8, 5), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.items.len(), 1, "max_running_requests 1");
        complete_all(&mut s, &mut p, &plan, &[]);
        clock.advance(Duration::from_secs(61));
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            plan.dropped,
            [(
                RequestId(uuid::Uuid::from_u128(2)),
                CancelReason::QueueTimeout
            )]
        );
        assert!(s.submit(request(3, 3, 5000, 1), p.total_blocks()).is_err());
        let text = reg.render().unwrap();
        assert!(
            text.contains(
                "turbine_admission_total{outcome=\"rejected\",reason=\"queue_timeout\"} 1"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "turbine_admission_total{outcome=\"rejected\",reason=\"context_length_exceeded\"} 1"
            ),
            "{text}"
        );
    }

    #[test]
    fn prompt_too_long_without_chunked_prefill() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(
            SchedulerParams {
                chunked_prefill: false,
                ..params()
            },
            Arc::new(clock),
        );
        let mut p = pool(64);
        assert!(matches!(
            s.submit(request(1, 1, 65, 1), p.total_blocks()),
            Err(SubmitError::PromptTooLong)
        ));
        s.submit(request(2, 2, 64, 1), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [(2, BatchKind::Prefill { start: 0, len: 64 })]
        );
    }

    #[test]
    fn choices_fork_from_one_prefill() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(params(), Arc::new(clock));
        let mut p = pool(64);
        // n = 3, a 20-token prompt: one full block shared, the partial tail copied.
        let mut r = request(1, 10, 20, 10);
        r.seqs = smallvec![SeqId(10), SeqId(11), SeqId(12)];
        s.submit(r, p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [(10, BatchKind::Prefill { start: 0, len: 20 })]
        );
        let parent_table = plan.items[0].block_table.clone();
        // The shared prefill samples one token for every choice.
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: plan.iteration,
                appended: vec![(SeqId(10), 1), (SeqId(11), 1), (SeqId(12), 1)],
                ..IterationOutcome::default()
            },
        );
        assert_eq!(s.seq_state(SeqId(10)), Some(RequestState::Prefilling));

        let plan = s.plan(&mut p, &IterationLimits::default());
        assert!(plan.items.is_empty(), "no decode before the forks exist");
        assert_eq!(plan.forks.len(), 2);
        for (fork, child) in plan.forks.iter().zip([11, 12]) {
            assert_eq!((fork.src, fork.dst), (SeqId(10), SeqId(child)));
            let (from, to) = fork.copy.expect("partial tail is copied");
            assert_eq!(from, parent_table.blocks[1]);
            assert_ne!(to, from);
        }
        // 2 parent blocks + one private tail per child; the full block is shared.
        assert_eq!(p.used_blocks(), 4);
        complete_all(&mut s, &mut p, &plan, &[]);

        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (10, BatchKind::Decode),
                (11, BatchKind::Decode),
                (12, BatchKind::Decode)
            ]
        );
        assert_eq!(plan.items[1].block_table.blocks[0], parent_table.blocks[0]);
        complete_all(&mut s, &mut p, &plan, &[10, 11, 12]);
        assert_eq!(p.used_blocks(), 0);
        assert!(s.is_idle());

        // A parent that finishes at its prefill keeps its table until its choices forked.
        let mut r = request(2, 20, 16, 4);
        r.seqs = smallvec![SeqId(20), SeqId(21)];
        s.submit(r, p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: plan.iteration,
                appended: vec![(SeqId(20), 1), (SeqId(21), 1)],
                finished: vec![(SeqId(20), FinishReason::Stop)],
                failed: None,
            },
        );
        assert_eq!(p.used_blocks(), 1, "held for the fork");
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.forks.len(), 1);
        assert_eq!(
            plan.forks[0].copy, None,
            "a 16-token prompt has no partial tail"
        );
        assert_eq!(
            p.used_blocks(),
            1,
            "the child now holds the shared block alone"
        );
        complete_all(&mut s, &mut p, &plan, &[21]);
        assert_eq!(p.used_blocks(), 0);
        assert!(s.is_idle());
    }
}
