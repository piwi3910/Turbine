//! Per-iteration planning (P2 §Scheduling rules): drop cancelled requests, decode every
//! decodable sequence (preempting by recompute when blocks run short), then fill the token
//! budget with prefill chunks — continuing prefills oldest first, then admissions by priority
//! then arrival.
//!
//! Token accounting of one sequence with prompt `P` and `g` generated tokens: a prefill writes
//! the KV of positions `[0, P + g)` (after a preemption `g > 0`: recompute) and its last
//! position yields the next token; a decode writes the KV of the newest token and yields the
//! next one. `BlockTable::tokens` counts written positions, including the ones planned for the
//! iteration in flight. The scheduling policy (`crate::policy`, Phase 2m S-9) orders the
//! waiting queue, ranks running requests for preemption, picks the victim and sizes prefill
//! chunks; this module is the mechanism. Under the `default` policy the rank is
//! `(priority, admission order)` and the victim the worst-ranked (largest priority value, then
//! most recently admitted). Whatever the policy, a prefill or fork may preempt only requests
//! ranked below its own, so the best-ranked request always progresses — no livelock while each
//! request's full KV fits the empty pool (checked at submission).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use smallvec::SmallVec;
use turbine_core::clock::Clock;
use turbine_core::config::Config;
use turbine_core::request::FinishReason;
use turbine_core::types::{BlockId, PressureState, RequestId, SeqId};
use turbine_kv::{BlockPool, BlockTable, PoolError, blocks_for_tokens};
use turbine_reliability::admission::RejectionReason;
use turbine_reliability::ledger::Reservation;
use turbine_reliability::throttle::{AdmissionMode, ThrottlePlan};

use crate::gate::{AdmissionGate, GateOutcome};
use crate::metrics::SchedulerMetrics;
use crate::policy::{AdmissionInfo, DefaultPolicy, PreemptionRank, RunningInfo, SchedulingPolicy};
use crate::queue::WaitingQueue;
use crate::request::{CancelReason, PreemptReason, RequestState, SchedRequest};

/// Scheduler bounds (P2 §Configuration). `max_seq_len` and `queue_timeout` are contract
/// additions used by the submission checks and rule 1; from Phase 3 `queue_timeout` is
/// `reliability.admission.queue_timeout` (CONFLICT C-1).
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
            queue_timeout: cfg.reliability.admission.queue_timeout.0,
        }
    }
}

/// Per-iteration throttles. Phase 2 always passes `Default` (GREEN: no limit); Phase 3 derives
/// them from the pressure controller's throttle plan (`From<&ThrottlePlan>`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationLimits {
    /// At most this many admissions this iteration. With an admission gate: growth of the
    /// admitted count (running + waiting with a KV reservation) over the previous iteration's
    /// (`Some(0)`: queued requests may replace finished ones, the batch does not grow).
    pub batch_growth_limit: Option<u32>,
    /// No admissions at all.
    pub shrink_only: bool,
    /// Share of the remaining token budget prefills may use (0..=1).
    pub prefill_budget_fraction: f64,
    /// Chunk size override (never above `prefill_chunk_tokens`).
    pub prefill_chunk_tokens: Option<u32>,
    pub admit_new: bool,
    pub start_new_prefills: bool,
    /// Running requests may be preempted to make room. Phase 2: always; with an admission gate
    /// (Phase 3) only in SURVIVAL, and only when the next decode step cannot allocate.
    pub allow_preempt: bool,
    /// SURVIVAL, `survival_liveness: requeue_unstarted` (P3 decision "SURVIVAL liveness fix",
    /// option A): admitted requests that have not started return to the admission gate's
    /// queue and drop their KV reservations.
    pub requeue_unstarted: bool,
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
            allow_preempt: true,
            requeue_unstarted: false,
        }
    }
}

impl From<&ThrottlePlan> for IterationLimits {
    /// The P3 throttle-plan table as scheduler limits. SURVIVAL has no prefill chunk and a
    /// prefill budget of 0, so no prefill runs at all.
    fn from(p: &ThrottlePlan) -> Self {
        IterationLimits {
            batch_growth_limit: p.batch_growth_limit,
            shrink_only: p.shrink_only,
            prefill_budget_fraction: p.prefill_budget_fraction,
            prefill_chunk_tokens: p.prefill_chunk_tokens,
            // RED (`AllQueued`) still pumps the gate's queue into finished slots.
            admit_new: p.admission != AdmissionMode::Stopped,
            start_new_prefills: p.start_new_prefills,
            allow_preempt: p.state == PressureState::Survival,
            requeue_unstarted: p.requeue_unstarted,
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

    /// Context tokens the decode items attend over (their context lengths after the step).
    pub fn decode_context_tokens(&self) -> u64 {
        self.items
            .iter()
            .filter(|i| i.kind == BatchKind::Decode)
            .map(|i| u64::from(i.block_table.tokens))
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
#[non_exhaustive]
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
    /// The admission gate refused the request (P3 reject table); `retry_after_secs` is the
    /// `Retry-After` hint (0 = none).
    #[error("rejected: {}", reason.as_str())]
    Rejected {
        reason: RejectionReason,
        retry_after_secs: u64,
    },
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
            SubmitError::Rejected { reason, .. } => reason.as_str(),
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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LastIteration {
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub requests: u32,
    pub duration_ms: f64,
    /// Milliseconds of the iteration per engine stage (P2c S-1: `schedule`, `prepare`,
    /// `launch`, `device_wait`, `sample`, `detokenize`, `emit`, `complete`). The scheduler
    /// leaves it empty; the engine fills it in the document it publishes.
    pub stages_ms: BTreeMap<String, f64>,
}

struct ReqEntry {
    req: SchedRequest,
    /// Admission number of the current admission; `None` while waiting.
    admitted: Option<u64>,
    ever_admitted: bool,
    /// The queue counter at submission (`AdmissionInfo::submit_no`).
    submit_no: u64,
    queued_since: Duration,
    /// `seqs[0]`'s first prefill finished; the other choices fork from it.
    shared_prefill_done: bool,
    cancel: Option<CancelReason>,
    /// Worst-case KV reservation from the admission gate (P3); every block of the request is
    /// paid from it, and dropping the entry releases it.
    reservation: Option<Reservation>,
    /// An admitted prefill above `large_prefill_tokens` that has not finished yet.
    expensive: bool,
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

/// Every choice's KV blocks at completion; choices share the prompt's full blocks.
fn request_kv_blocks(r: &SchedRequest, block_tokens: u32) -> u64 {
    let bt = block_tokens.max(1);
    let one = u64::from(blocks_for_tokens(
        r.prompt_len.saturating_add(r.max_new_tokens),
        bt,
    ));
    let shared = u64::from(r.prompt_len / bt);
    let extra_choices = r.seqs.len().saturating_sub(1) as u64;
    one + extra_choices * (one - shared.min(one))
}

/// The blocks a request's worst-case KV reservation covers: `request_kv_blocks` minus the full
/// blocks of its attached cached prefix (P4 S-3: `cached_prefix_tokens` feeds the estimate), which
/// other requests' reservations already paid for or which were cached unreferenced.
fn reserved_kv_blocks(r: &SchedRequest, block_tokens: u32) -> u64 {
    let cached = r
        .cached_prefix
        .as_ref()
        .map_or(0, |a| a.blocks.len() as u64);
    request_kv_blocks(r, block_tokens).saturating_sub(cached)
}

/// The Phase 2 continuous-batching scheduler. Single-threaded: owned by the engine thread.
pub struct Scheduler {
    params: SchedulerParams,
    clock: Arc<dyn Clock>,
    metrics: Option<SchedulerMetrics>,
    policy: &'static dyn SchedulingPolicy,
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
    /// P3 admission gate; `None` keeps the Phase 2 behaviour.
    gate: Option<AdmissionGate>,
    /// Requests that left the gate's queue on `cancel`, reported by the next plan.
    gate_dropped: Vec<(RequestId, CancelReason)>,
    /// Phase 4: attached prefix blocks of requests that left the admission queue without a
    /// pool at hand (`cancel`); released at the next `plan`.
    prefix_release: Vec<SmallVec<[BlockId; 16]>>,
    /// Admitted requests (running + waiting with a reservation) after the previous plan: the
    /// reference for batch growth with a gate.
    prev_admitted: usize,
}

impl Scheduler {
    pub fn new(p: SchedulerParams, clock: Arc<dyn Clock>) -> Scheduler {
        Scheduler {
            params: p,
            clock,
            metrics: None,
            policy: &DefaultPolicy,
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
            gate: None,
            gate_dropped: Vec::new(),
            prefix_release: Vec::new(),
            prev_admitted: 0,
        }
    }

    /// Put the P3 admission gate in front of the scheduler: it owns the single waiting queue
    /// (`scheduler.max_queued_requests` then bounds only the HTTP→engine channel, C-1), every
    /// admitted request carries its worst-case KV reservation, and preemption happens only when
    /// `IterationLimits::allow_preempt` (SURVIVAL).
    pub fn with_gate(mut self, gate: AdmissionGate) -> Scheduler {
        self.gate = Some(gate);
        self
    }

    pub fn gate(&self) -> Option<&AdmissionGate> {
        self.gate.as_ref()
    }

    pub fn gate_mut(&mut self) -> Option<&mut AdmissionGate> {
        self.gate.as_mut()
    }

    /// Waiting requests: the gate's queue plus admitted requests not yet started (and
    /// preempted ones).
    pub fn queue_len(&self) -> usize {
        self.queue.len() + self.gate.as_ref().map_or(0, AdmissionGate::queue_len)
    }

    /// Publish metrics to `m` (the simulator runs without).
    pub fn with_metrics(mut self, m: SchedulerMetrics) -> Scheduler {
        self.metrics = Some(m);
        self.publish_gauges();
        self
    }

    /// Schedule under `policy` (default: `DefaultPolicy`). Set it before the first submission:
    /// queued requests keep the admission keys of the policy they were queued under.
    pub fn with_policy(mut self, policy: &'static dyn SchedulingPolicy) -> Scheduler {
        debug_assert!(
            self.requests.is_empty(),
            "policy changed with requests queued"
        );
        self.policy = policy;
        self
    }

    /// The scheduling policy in use.
    pub fn policy(&self) -> &'static dyn SchedulingPolicy {
        self.policy
    }

    pub fn params(&self) -> &SchedulerParams {
        &self.params
    }

    /// Submission checks (P2 §Scheduling rules), then queue by priority and arrival. With an
    /// admission gate the gate decides: admitted (with its worst-case KV reservation) into the
    /// scheduler's queue, held in the gate's queue, or rejected (P3 S-9).
    pub fn submit(
        &mut self,
        mut r: SchedRequest,
        pool_total_blocks: u32,
    ) -> Result<(), SubmitError> {
        if let Err(e) = self.check_submission(&r, pool_total_blocks) {
            self.count_rejection(r.id, e);
            return Err(e);
        }
        let block_tokens = self.params.block_tokens;
        let submit_no = self.queue.next_order();
        let Some(gate) = self.gate.as_mut() else {
            self.enqueue(r, None, submit_no);
            return Ok(());
        };
        // The reservation covers every choice's KV at completion, less the attached prefix.
        r.estimate.projected_kv_blocks =
            u32::try_from(reserved_kv_blocks(&r, block_tokens)).unwrap_or(u32::MAX);
        let id = r.id;
        // An immediate admission takes an admitted slot: within `max_running_requests` and the
        // throttle plan's batch growth over the previous plan's admitted count, like the pump.
        let limits = IterationLimits::from(&gate.controller().throttle());
        let admitted = self.running.len() + self.queue.len();
        let slot_free = admitted < self.params.max_running_requests as usize
            && limits
                .batch_growth_limit
                .is_none_or(|g| admitted < self.prev_admitted + g as usize);
        // The gate's queue is ordered by the policy's key, like the scheduler's own queue.
        let key = self.policy.admission_key(&AdmissionInfo {
            priority: r.priority,
            arrival: r.arrival,
            submit_no,
            preempted: false,
            push_no: submit_no,
        });
        match gate.offer(r, key, submit_no, slot_free) {
            Ok(GateOutcome::Admitted(r, reservation)) => {
                self.enqueue(*r, Some(reservation), submit_no);
                Ok(())
            }
            Ok(GateOutcome::Queued) => {
                if let Some(m) = &self.metrics {
                    m.admission("queued", "gated");
                }
                self.publish_gauges();
                Ok(())
            }
            Err(e) => {
                self.count_rejection(id, e);
                Err(e)
            }
        }
    }

    /// An internal request (circuit probe) that bypasses the admission queue but still takes
    /// its worst-case KV reservation; without a gate it is a plain submission.
    pub fn submit_probe(
        &mut self,
        mut r: SchedRequest,
        pool_total_blocks: u32,
    ) -> Result<(), SubmitError> {
        self.check_submission(&r, pool_total_blocks)?;
        let block_tokens = self.params.block_tokens;
        let reservation = match self.gate.as_ref() {
            Some(gate) => {
                r.estimate.projected_kv_blocks =
                    u32::try_from(reserved_kv_blocks(&r, block_tokens)).unwrap_or(u32::MAX);
                let res = gate
                    .reserve_direct(&r.estimate)
                    .map_err(|_| SubmitError::Rejected {
                        reason: RejectionReason::QueueFull,
                        retry_after_secs: 1,
                    })?;
                Some(res)
            }
            None => None,
        };
        let submit_no = self.queue.next_order();
        self.enqueue(r, reservation, submit_no);
        Ok(())
    }

    fn count_rejection(&self, id: RequestId, e: SubmitError) {
        tracing::info!(event = "reject", request_id = %id.0, reason = e.as_str(), "request rejected");
        if let Some(m) = &self.metrics {
            m.admission("rejected", e.as_str());
        }
    }

    /// Track `r` in the scheduler's own queue: Phase 2 queued, or admitted by the gate with
    /// its reservation. `submit_no` is the queue counter taken at submission.
    fn enqueue(&mut self, r: SchedRequest, reservation: Option<Reservation>, submit_no: u64) {
        let expensive = reservation.is_some()
            && self
                .gate
                .as_ref()
                .is_some_and(|g| g.is_expensive(&r.estimate));
        let now = self.clock.now_mono();
        let key = self.policy.admission_key(&AdmissionInfo {
            priority: r.priority,
            arrival: r.arrival,
            submit_no,
            preempted: false,
            push_no: submit_no,
        });
        self.queue.push(r.id, key);
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
                submit_no,
                queued_since: now,
                shared_prefill_done: false,
                cancel: None,
                reservation,
                expensive,
            },
        );
        if let Some(m) = &self.metrics {
            m.admission("queued", "ok");
        }
        self.publish_gauges();
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
        if request_kv_blocks(r, p.block_tokens) > u64::from(pool_total_blocks) {
            return Err(SubmitError::ContextExceedsKvCapacity);
        }
        // With a gate the admission queue has its own bound (C-1).
        if self.gate.is_none() && self.queue.len() >= p.max_queued_requests as usize {
            return Err(SubmitError::QueueFull);
        }
        Ok(())
    }

    /// Mark a request cancelled; rule 1 of the next `plan` drops it and frees its blocks.
    pub fn cancel(&mut self, id: RequestId, reason: CancelReason) {
        if let Some(e) = self.requests.get_mut(&id) {
            e.cancel.get_or_insert(reason);
        } else if let Some(gate) = self.gate.as_mut()
            && let Some(r) = gate.remove(id)
        {
            // Still in the admission queue: it never took a reservation.
            self.gate_dropped.push((id, reason));
            if let Some(a) = r.cached_prefix {
                self.prefix_release.push(a.blocks);
            }
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
        self.requests.is_empty() && self.gate.as_ref().is_none_or(|g| g.queue_len() == 0)
    }

    /// Admitted requests in admission order.
    pub fn running_ids(&self) -> Vec<RequestId> {
        self.running.clone()
    }

    /// Waiting requests: admitted but not started (preempted first), then the gate's queue.
    pub fn queued_ids(&self) -> Vec<RequestId> {
        let mut ids: Vec<RequestId> = self.queue.iter().collect();
        if let Some(g) = &self.gate {
            ids.extend(g.queued_ids());
        }
        ids
    }

    /// Tokens `seq` may still generate (`max_new_tokens − generated`).
    pub fn token_limit(&self, seq: SeqId) -> Option<u32> {
        let e = self.seqs.get(&seq)?;
        let r = self.requests.get(&e.request)?;
        Some(r.req.max_new_tokens.saturating_sub(e.generated))
    }

    /// Remaining tokens of every live sequence of the running requests (exhaustion horizon).
    pub fn remaining_tokens(&self) -> Vec<u32> {
        self.running
            .iter()
            .flat_map(|id| self.requests[id].req.seqs.iter())
            .filter(|s| self.seqs.get(s).is_some_and(|e| e.state.is_live()))
            .filter_map(|s| self.token_limit(*s))
            .collect()
    }

    /// Recovery (P3 S-11): keep only the first `limit` items of the plan in flight and undo
    /// the others — their positions are un-written and blocks allocated for them return to
    /// the pool — so the iteration can be retried with a smaller batch.
    pub fn shrink_plan(&mut self, pool: &mut BlockPool, plan: &mut IterationPlan, limit: usize) {
        if plan.items.len() <= limit {
            return;
        }
        let bt = self.params.block_tokens;
        for item in plan.items.drain(limit..) {
            let Some(e) = self.seqs.get_mut(&item.seq) else {
                continue;
            };
            let written = match item.kind {
                BatchKind::Decode => 1,
                BatchKind::Prefill { len, .. } => len,
            };
            e.table.tokens = e.table.tokens.saturating_sub(written);
            let keep = blocks_for_tokens(e.table.tokens, bt) as usize;
            if e.table.blocks.len() > keep {
                let extra: SmallVec<[BlockId; 16]> = e.table.blocks.drain(keep..).collect();
                pool.release(&extra);
            }
        }
        self.in_flight = plan.items.iter().map(|i| (i.seq, i.kind)).collect();
        let kept: HashSet<RequestId> = plan
            .items
            .iter()
            .filter_map(|i| self.seqs.get(&i.seq).map(|e| e.request))
            .chain(
                plan.forks
                    .iter()
                    .filter_map(|f| self.seqs.get(&f.dst).map(|e| e.request)),
            )
            .collect();
        self.in_flight_requests.retain(|id| kept.contains(id));
        self.last.prefill_tokens = plan.prefill_tokens();
        self.last.decode_tokens = plan.decode_tokens();
        self.last.requests = plan.items.len() as u32;
    }

    /// Fail `ids` (retries exhausted, P3 S-11): their blocks and reservations are released.
    /// Returns the ones the scheduler still tracked.
    pub fn fail_requests(&mut self, pool: &mut BlockPool, ids: &[RequestId]) -> Vec<RequestId> {
        let mut failed = Vec::new();
        for &id in ids {
            if self.requests.contains_key(&id) {
                tracing::warn!(event = "fail", request_id = %id.0, reason = "resource_exhausted", "request failed after recovery retries");
                self.remove_request(pool, id, RequestState::Failed);
                failed.push(id);
            }
        }
        self.in_flight_requests.retain(|id| !failed.contains(id));
        self.publish_gauges();
        failed
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

    /// Requests holding a running slot or waiting to start. With a gate every one of them
    /// holds its worst-case KV reservation; batch growth limits this count.
    pub fn admitted_count(&self) -> usize {
        self.running.len() + self.queue.len()
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
        // SURVIVAL, option A: admitted requests that have not started give their KV
        // reservations back and wait in the admission queue again.
        if limits.requeue_unstarted {
            self.requeue_unstarted(pool, &mut plan);
        }

        // (2) Every decodable sequence decodes; preempt until the pool covers them.
        self.plan_decodes(pool, limits, &mut plan, &mut in_plan, &mut preempted_now);

        // Forks of finished shared prefills (n > 1) wait for blocks, never re-prefill.
        self.plan_forks(pool, &mut plan, &mut in_plan, &mut preempted_now);

        // (3) Prefill: continuing prefills oldest first, then admissions.
        let decode_tokens = plan.decode_tokens();
        let fraction = limits.prefill_budget_fraction.clamp(0.0, 1.0);
        let mut budget = (f64::from(self.params.max_batch_tokens.saturating_sub(decode_tokens))
            * fraction)
            .floor() as u32;
        let chunk_cap = self.policy.chunk_cap(&self.params, limits);
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
        // With a gate every request in the scheduler's queue already holds its worst-case KV
        // reservation: the admitted count is running + waiting, and it grows by at most
        // `batch_growth_limit` over the previous iteration's (`Some(0)`: queued requests only
        // refill slots of finished ones — RED included). The gate pumps its queue into the
        // freed slots; admitted requests start as the prefill budget allows.
        let gated = self.gate.is_some();
        let may_start = limits.admit_new && limits.start_new_prefills && !limits.shrink_only;
        if gated && may_start {
            let admitted = self.admitted_count();
            let allowance = limits.batch_growth_limit.map_or(usize::MAX, |g| {
                (self.prev_admitted + g as usize).saturating_sub(admitted)
            });
            let mut slots = (self.params.max_running_requests as usize)
                .saturating_sub(admitted)
                .min(allowance);
            // Work-conserving floor (P3 S-10): a growth limit never holds the admitted count at
            // 0 while requests wait, or ORANGE/RED freeze an idle engine behind a full queue.
            let idle = admitted == 0;
            if idle {
                slots = slots.max(1);
            }
            let pumped = self
                .gate
                .as_mut()
                .map(|g| g.pump(slots, idle))
                .unwrap_or_default();
            for (r, reservation, submit_no) in pumped {
                self.enqueue(r, Some(reservation), submit_no);
            }
        }
        if may_start {
            let mut admitted_now = 0u32;
            while budget > 0
                && (self.running.len() as u32) < self.params.max_running_requests
                && (gated || limits.batch_growth_limit.is_none_or(|g| admitted_now < g))
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
            stages_ms: BTreeMap::new(),
        };
        self.prev_admitted = self.admitted_count();
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
            if r.expensive {
                r.expensive = false;
                if let Some(g) = self.gate.as_mut() {
                    g.admission_mut().expensive_prefill_finished();
                }
            }
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
            waiting: self.queue_len() as u32,
            prefilling,
            decoding,
            paused,
            constrained: self.requests.values().filter(|r| r.req.constrained).count() as u32,
            iterations_total: self.iteration,
            preemptions_total: self.preemptions_total,
            last_iteration: self.last.clone(),
        }
    }

    // ---- rules -------------------------------------------------------------------------

    fn drop_cancelled(&mut self, pool: &mut BlockPool, now: Duration, plan: &mut IterationPlan) {
        self.drop_gated(pool, plan);
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
            // With a gate, admitted requests never time out: the gate's queue owns the wait.
            let timed_out = self.gate.is_none()
                && e.admitted.is_none()
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

    /// SURVIVAL under `survival_liveness: requeue_unstarted` (P3 decision "SURVIVAL liveness
    /// fix", option A). SURVIVAL starts no prefill, so an admitted request that has not started
    /// (never given a running slot: no KV written) only holds its worst-case KV reservation;
    /// held by many such requests, those reservations can keep `kv_utilization` above
    /// SURVIVAL's exit threshold with nothing left to release them. Each goes back to the
    /// admission gate's queue at its original turn and drops its reservation, so the running
    /// requests drain the pool; when the queue is full it is answered `overloaded`. Requests
    /// that started (running, or preempted with KV to recompute) keep theirs.
    fn requeue_unstarted(&mut self, pool: &mut BlockPool, plan: &mut IterationPlan) {
        let Some(gate) = self.gate.as_mut() else {
            return;
        };
        let unstarted: Vec<RequestId> = self
            .queue
            .iter()
            .filter(|id| {
                self.requests
                    .get(id)
                    .is_some_and(|r| !r.ever_admitted && r.reservation.is_some())
            })
            .collect();
        for id in unstarted {
            self.queue.remove(id);
            let Some(entry) = self.requests.remove(&id) else {
                continue;
            };
            for seq in &entry.req.seqs {
                self.seqs.remove(seq);
            }
            if entry.expensive {
                gate.admission_mut().expensive_prefill_finished();
            }
            let ReqEntry {
                mut req,
                submit_no,
                reservation,
                ..
            } = entry;
            // The worst-case KV reservation goes back to the pool now, and so do the blocks of
            // an attached prefix (P4): the request recomputes it, like after a preemption.
            drop(reservation);
            if let Some(a) = req.cached_prefix.take() {
                pool.release(&a.blocks);
            }
            let key = self.policy.admission_key(&AdmissionInfo {
                priority: req.priority,
                arrival: req.arrival,
                submit_no,
                preempted: false,
                push_no: submit_no,
            });
            if gate.requeue(req, key, submit_no) {
                tracing::info!(event = "survival_requeue", request_id = %id.0, "admitted request returned to the admission queue");
                if let Some(m) = &self.metrics {
                    m.admission("queued", "survival_requeue");
                }
            } else {
                let reason = CancelReason::Overloaded;
                plan.dropped.push((id, reason));
                tracing::info!(event = "reject", request_id = %id.0, reason = reason.as_str(), "admitted request rejected: the admission queue is full in SURVIVAL");
                if let Some(m) = &self.metrics {
                    m.admission("rejected", reason.as_str());
                }
            }
        }
        self.publish_gauges();
    }

    /// Requests leaving the gate's queue: cancelled while queued, timed out, or rejected
    /// because the circuit opened. None of them holds a reservation; the blocks of an attached
    /// prefix (P4) are released here.
    fn drop_gated(&mut self, pool: &mut BlockPool, plan: &mut IterationPlan) {
        for blocks in std::mem::take(&mut self.prefix_release) {
            pool.release(&blocks);
        }
        let Some(gate) = self.gate.as_mut() else {
            return;
        };
        let mut gone: Vec<(RequestId, CancelReason)> = std::mem::take(&mut self.gate_dropped);
        let mut left: Vec<(SchedRequest, CancelReason)> = gate
            .take_cancelled()
            .into_iter()
            .map(|r| (r, CancelReason::ClientDisconnect))
            .collect();
        left.extend(
            gate.expire()
                .into_iter()
                .map(|r| (r, CancelReason::QueueTimeout)),
        );
        if gate.controller().circuit().blocks_readiness() {
            left.extend(
                gate.reject_all()
                    .into_iter()
                    .map(|r| (r, CancelReason::CircuitOpen)),
            );
        }
        for (r, reason) in left {
            if let Some(a) = &r.cached_prefix {
                pool.release(&a.blocks);
            }
            gone.push((r.id, reason));
        }
        for (id, reason) in gone {
            plan.dropped.push((id, reason));
            if matches!(
                reason,
                CancelReason::QueueTimeout | CancelReason::CircuitOpen
            ) {
                tracing::info!(event = "reject", request_id = %id.0, reason = reason.as_str(), "queued request rejected");
                if let Some(m) = &self.metrics {
                    m.admission("rejected", reason.as_str());
                }
            } else {
                tracing::info!(event = "cancel", request_id = %id.0, reason = reason.as_str(), "queued request cancelled");
                if let Some(m) = &self.metrics {
                    m.cancelled(reason.as_str());
                }
            }
        }
    }

    fn plan_decodes(
        &mut self,
        pool: &mut BlockPool,
        limits: &IterationLimits,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        preempted_now: &mut HashSet<RequestId>,
    ) {
        let bt = self.params.block_tokens;
        // Preempt until the pool covers every decode (never, without allow_preempt).
        while limits.allow_preempt && self.decode_blocks_needed() > pool.available_blocks() {
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
            let Ok(blocks) = Self::allocate_for(pool, &mut self.requests, e.request, need) else {
                // With preemption allowed the loop above made the pool cover every decode;
                // without it (a gate below SURVIVAL) the sequence waits for a block.
                if limits.allow_preempt {
                    tracing::error!(
                        event = "scheduler_bug",
                        seq = seq.0,
                        "decode lacks blocks after preemption"
                    );
                } else {
                    tracing::warn!(
                        event = "decode_deferred",
                        seq = seq.0,
                        reason = "kv_exhausted",
                        "decode waits for a KV block (no preemption below SURVIVAL)"
                    );
                }
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
            let rank = self.rank(id);
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
                if forked.is_err() && self.gate.is_none() {
                    // Make room from requests ranked below this one that are not in the plan.
                    while let Some(victim) =
                        self.worst_running(|v| v.0 != id && !in_plan.contains(&v.0) && v.1 > rank)
                    {
                        self.preempt(pool, victim, plan, preempted_now);
                        if pool.available_blocks() > 0 {
                            break;
                        }
                    }
                    forked = pool.fork(&parent_table);
                }
                let Ok((table, copy)) = forked else {
                    all_forked = false;
                    break; // waits for blocks; the prompt is not re-prefilled
                };
                if copy.is_some()
                    && let Some(res) = self
                        .requests
                        .get_mut(&id)
                        .and_then(|r| r.reservation.as_mut())
                {
                    // The fresh tail block is paid like any other block of the request.
                    res.commit_bytes(pool.layout().block_bytes());
                }
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
                // Only a parent whose shared prefill is complete waits here; a parent that
                // re-prefills after a preemption is promoted by `complete` when it finishes.
                if p.state == RequestState::Prefilling && p.table.tokens >= p.target {
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
        // With a gate every request's KV is reserved: prefills never preempt (P3 §Throttle).
        if need > pool.available_blocks() && may_preempt && self.gate.is_none() {
            let rank = self.rank(id);
            while need > pool.available_blocks() {
                let Some(victim) =
                    self.worst_running(|v| v.0 != id && !in_plan.contains(&v.0) && v.1 > rank)
                else {
                    break;
                };
                self.preempt(pool, victim, plan, preempted_now);
            }
        }
        let Ok(blocks) = Self::allocate_for(pool, &mut self.requests, id, need) else {
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
        // A cached prefix is attached, not prefilled: the first chunk starts after it.
        let cached = r.req.cached_prefix.as_ref().map_or(0, |a| a.cached_tokens);
        let target = target.saturating_sub(cached);
        let len = target.min(chunk_cap).min(budget);
        if len == 0 || (self.whole_prefill_required(target) && len < target) {
            return false;
        }
        // A request holding its KV reservation needs no free-block watermark.
        let watermark = if r.reservation.is_some() {
            0
        } else {
            (f64::from(pool.total_blocks()) * self.params.free_watermark).ceil() as u32
        };
        pool.available_blocks() >= blocks_for_tokens(len, self.params.block_tokens) + watermark
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
        // The first admission hands the attached prefix to the first sequence: its table
        // starts with the shared blocks and prefill starts at `cached_tokens`, so no shared
        // block is ever written. A preempted request recomputes from scratch.
        let mut prefix = r.req.cached_prefix.take();
        for (i, seq) in seqs.iter().enumerate() {
            let e = self.seqs.get_mut(seq).expect("sequence tracked");
            if e.state != RequestState::Waiting {
                continue;
            }
            e.set_state(RequestState::Prefilling, *seq);
            e.awaiting_fork = i > 0 && !shared_done;
            e.target = prompt + e.generated;
            if i == 0
                && let Some(a) = prefix.take()
            {
                e.table = BlockTable {
                    blocks: a.blocks,
                    tokens: a.cached_tokens,
                };
            }
        }
        debug_assert!(
            prefix.is_none(),
            "an attached prefix outlived its admission"
        );
    }

    fn preempt(
        &mut self,
        pool: &mut BlockPool,
        id: RequestId,
        plan: &mut IterationPlan,
        preempted_now: &mut HashSet<RequestId>,
    ) {
        // With a gate preemption happens only in SURVIVAL (P3); the reservation is kept, so
        // the recompute is guaranteed its KV.
        let preempt_reason = if self.gate.is_some() {
            PreemptReason::SurvivalDecodeAlloc
        } else {
            PreemptReason::KvExhausted
        };
        self.running.retain(|r| *r != id);
        let push_no = self.queue.next_order();
        let r = self.requests.get_mut(&id).expect("running request tracked");
        r.admitted = None;
        let key = self.policy.admission_key(&AdmissionInfo {
            priority: r.req.priority,
            arrival: r.req.arrival,
            submit_no: r.submit_no,
            preempted: true,
            push_no,
        });
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
                plan.preempted.push((seq, preempt_reason));
            }
        }
        self.queue.push(id, key);
        preempted_now.insert(id);
        self.preemptions_total += 1;
        let reason = preempt_reason.as_str();
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
            if r.expensive
                && let Some(g) = self.gate.as_mut()
            {
                g.admission_mut().expensive_prefill_finished();
            }
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
        if r.expensive
            && let Some(g) = self.gate.as_mut()
        {
            g.admission_mut().expensive_prefill_finished();
        }
        if let Some(a) = &r.req.cached_prefix {
            pool.release(&a.blocks);
        }
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

    /// `n` blocks for request `id`, paid from its reservation when it has one.
    fn allocate_for(
        pool: &mut BlockPool,
        requests: &mut HashMap<RequestId, ReqEntry>,
        id: RequestId,
        n: u32,
    ) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        match requests.get_mut(&id).and_then(|r| r.reservation.as_mut()) {
            Some(res) => pool.allocate_reserved(n, res),
            None => pool.allocate(n),
        }
    }

    fn is_running(&self, id: RequestId) -> bool {
        self.requests.get(&id).is_some_and(|r| r.admitted.is_some())
    }

    /// Blocks the decode set needs for one more token each.
    fn decode_blocks_needed(&self) -> u32 {
        let bt = self.params.block_tokens;
        self.decode_set()
            .iter()
            .map(|s| self.seqs[s].table.blocks_needed(1, bt))
            .sum()
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

    /// The policy's preemption rank of tracked request `id`: larger is dropped first.
    fn rank(&self, id: RequestId) -> PreemptionRank {
        let r = &self.requests[&id];
        self.policy.preemption_rank(&RunningInfo {
            id,
            priority: r.req.priority,
            admitted: r.admitted,
        })
    }

    /// The policy's victim among the running requests accepted by `eligible((id, rank))`,
    /// offered in admission order.
    fn worst_running(
        &self,
        eligible: impl Fn((RequestId, PreemptionRank)) -> bool,
    ) -> Option<RequestId> {
        let candidates: Vec<(RequestId, PreemptionRank)> = self
            .running
            .iter()
            .map(|id| (*id, self.rank(*id)))
            .filter(|c| eligible(*c))
            .collect();
        let victim = self.policy.pick_victim(&candidates)?;
        if candidates.iter().any(|(id, _)| *id == victim) {
            Some(victim)
        } else {
            tracing::error!(event = "scheduler_bug", request_id = %victim.0, policy = self.policy.name(), "policy picked an ineligible victim");
            debug_assert!(false, "ineligible victim {victim:?}");
            None
        }
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
    use turbine_core::types::{DType, DeviceId, KvLayout, Priority};
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

    /// A pool of `n` accounting-only 16-token blocks (zero bytes per block).
    fn pool(n: u32) -> BlockPool {
        pool_of(n, 16)
    }

    /// A pool of `n` accounting-only blocks of `block_tokens` tokens.
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

    /// At the default 128-token page: completion KV is counted in whole pages, decodes fill a
    /// prefill's partial last page before taking another, and admission keeps the free-block
    /// watermark.
    #[test]
    fn default_page_block_accounting() {
        const BT: u32 = 128;
        let clock = FakeClock::new(Duration::ZERO);
        let params = SchedulerParams {
            max_batch_tokens: 512,
            prefill_chunk_tokens: 256,
            max_queued_requests: 8,
            block_tokens: BT,
            ..params()
        };
        let mut s = Scheduler::new(params, Arc::new(clock));
        // 5 blocks of 128 tokens; the 1 % watermark rounds up to 1 block.
        let mut p = pool_of(5, BT);
        let req = |n: u64, prompt, max_new| {
            SchedRequest::new(
                RequestId(uuid::Uuid::from_u128(u128::from(n))),
                smallvec![SeqId(n)],
                prompt,
                max_new,
                BT,
            )
        };

        // 700 tokens at completion need 6 pages of the 5; 640 fill exactly 5.
        assert!(matches!(
            s.submit(req(9, 600, 100), p.total_blocks()),
            Err(SubmitError::ContextExceedsKvCapacity)
        ));
        s.submit(req(1, 200, 440), p.total_blocks()).unwrap();
        s.submit(req(2, 100, 300), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Prefill { start: 0, len: 200 }),
                (2, BatchKind::Prefill { start: 0, len: 100 })
            ]
        );
        assert_eq!(plan.items[0].block_table.blocks.len(), 2);
        assert_eq!(plan.items[1].block_table.blocks.len(), 1);
        assert_eq!(p.used_blocks(), 3);
        complete_all(&mut s, &mut p, &plan, &[]);

        // Positions 100..=127 of request 2 and 200..=227 of request 1 land in the partial
        // last pages: 28 decode steps take no block.
        for _ in 0..28 {
            let plan = s.plan(&mut p, &IterationLimits::default());
            assert_eq!(
                kinds(&plan),
                [(1, BatchKind::Decode), (2, BatchKind::Decode)]
            );
            assert_eq!(p.used_blocks(), 3);
            complete_all(&mut s, &mut p, &plan, &[]);
        }
        // Position 128 opens request 2's second page.
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(plan.items[0].block_table.blocks.len(), 2);
        assert_eq!(plan.items[0].block_table.tokens, 229);
        assert_eq!(plan.items[1].block_table.blocks.len(), 2);
        assert_eq!(plan.items[1].block_table.tokens, 129);
        assert_eq!(p.used_blocks(), 4);
        complete_all(&mut s, &mut p, &plan, &[]);

        // One free page covers a 100-token prompt but not the watermark on top: it waits.
        s.submit(req(4, 100, 10), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [(1, BatchKind::Decode), (2, BatchKind::Decode)]
        );
        assert_eq!(s.snapshot().waiting, 1);
        complete_all(&mut s, &mut p, &plan, &[2]);
        assert_eq!(p.used_blocks(), 2, "finished request 2 freed both pages");

        // Three free pages: admitted into one page.
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Decode),
                (4, BatchKind::Prefill { start: 0, len: 100 })
            ]
        );
        assert_eq!(plan.items[1].block_table.blocks.len(), 1);
        assert_eq!(p.used_blocks(), 3);
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
