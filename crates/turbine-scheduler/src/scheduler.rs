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
//!
//! Micro-batches (Phase 5 S-10, pipeline parallelism): with `with_micro_batches(m)` up to `m`
//! plans are in flight at once, each matched by its `iteration` in `complete`, in any order.
//! A plan never touches a request of another plan in flight: no decode, prefill chunk or fork,
//! never a preemption victim, and a cancellation is applied by the first plan after its plan
//! completed — so no block is freed under an executing plan. Each plan holds at most
//! `ceil(live / m)` sequences (`live`: the running sequences, in flight or not, plus the waiting
//! requests a free slot could start) and, with chunked prefill, `max_batch_tokens / m` tokens,
//! so the running set spreads over the micro-batches; decodes go least recently stepped first.
//! With `m = 1` (the default) planning is unchanged.
//!
//! Multi-token appends (P5p S-9): `extend(seq, tokens)` asks the next plan to feed
//! `1 + tokens` positions of the decoding sequence as one `BatchKind::Prefill { start, len }`
//! item with `logits_rows = len` (the verified jump-forward of S-10 feeds the sampled-but-
//! unfed token and the grammar's forced tokens; the last row yields the next token). The item
//! takes `len` from the token budget left after the decodes and is planned before the prefill
//! chunks; not coverable — the budget, the executor's logits-row capacity (the plan keeps an
//! iteration's total rows at `max_running_requests`, which sizes the executor's head buffers)
//! or the pool — falls back to a plain decode and drops the extension. The engine reports the
//! tokens it kept as `appended` and the rest as `IterationOutcome::rolled_back`, which
//! shortens the table and frees blocks left empty. A preemption, finish or removal drops the
//! pending extension with the table it would have extended. Callers bound `tokens` by
//! `structured_output.jump_forward_max_tokens` (spec S-9); the plan additionally caps the item
//! by the policy's chunk cap and the request's own `prompt_len + max_new_tokens`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use smallvec::SmallVec;
use turbine_core::clock::Clock;
use turbine_core::config::Config;
use turbine_core::request::FinishReason;
use turbine_core::types::{BlockId, PressureState, RequestId, SeqId, StateLayout};
use turbine_kv::hierarchy::PrefixAttach;
use turbine_kv::{BlockPool, BlockTable, PoolError, blocks_for_tokens};
use turbine_reliability::admission::RejectionReason;
use turbine_reliability::ledger::Reservation;
use turbine_reliability::throttle::{AdmissionMode, ThrottlePlan};

use crate::gate::{AdmissionGate, GateOutcome};
use crate::metrics::SchedulerMetrics;
use crate::policy::{AdmissionInfo, DefaultPolicy, PreemptionRank, RunningInfo, SchedulingPolicy};
use crate::queue::WaitingQueue;
use crate::request::{CancelReason, PreemptReason, RequestState, SchedError, SchedRequest};

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
    /// The recent window (P6b S-5): the newest full blocks of each live sequence are allocated
    /// in the BF16 page class while the L0 base format is window-eligible (TurboQuant); a
    /// block that leaves the window is recompressed by the KV hierarchy. `None` with the
    /// window off.
    pub recent_window: Option<RecentWindow>,
    /// The `cache_aware` policy's ordering window (`scheduler.cache_aware_window`, P5p S-7),
    /// read by the policy at push; unused by `default`.
    pub cache_aware_window: Duration,
}

/// The recent window (P6b S-5, `kv.recent_window_blocks` with a window-eligible `kv.dtype`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecentWindow {
    /// How many of a sequence's newest full blocks hold BF16 pages.
    pub blocks: u32,
    /// The page class the window blocks are allocated in (the BF16 class).
    pub format: &'static str,
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
            recent_window: (cfg.kv.dtype.recent_window_base() && cfg.kv.recent_window_blocks > 0)
                .then_some(RecentWindow {
                    blocks: cfg.kv.recent_window_blocks,
                    format: "bf16",
                }),
            cache_aware_window: cfg.scheduler.cache_aware_window.0,
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
    /// The verified jump-forward's forced-token run this item feeds after its sampled token
    /// (P5p S-10): empty for every other item. The engine verifies the run against the model
    /// in one multi-row step and reports the kept tokens and the rollback.
    pub draft: Vec<u32>,
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
    /// Copy-on-write tail copies (P5p S-5), one per request that reuses a token-granular tail,
    /// planned with the sequence's first prefill chunk: `(seq, src, dst)` executed before the
    /// forward, with the fork copies.
    pub tail_copies: Vec<(SeqId, BlockId, BlockId)>,
}

impl IterationPlan {
    /// Nothing to execute: the caller sleeps until a submission or cancellation (rule 4).
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.forks.is_empty() && self.tail_copies.is_empty()
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
    /// Positions the engine fed but rejected (P5p S-9): the table is shortened by that many
    /// tokens and blocks left empty are freed. Kept + rolled-back positions = every extension
    /// item's `len`.
    pub rolled_back: Vec<(SeqId, u32)>,
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
    /// Pipeline micro-batches (Phase 5 S-10); present only with more than one micro-batch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineSnapshot>,
}

/// The `pipeline` section of `GET /turbine/v1/scheduler` (Phase 5 S-10). The scheduler fills
/// the micro-batch counts; the engine fills `stages` (placement and busy ratios) in the
/// document it publishes, as it does `LastIteration::stages_ms`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PipelineSnapshot {
    pub stages: Vec<StageSnapshot>,
    pub micro_batches: u32,
    pub micro_batches_in_flight: u32,
}

/// One pipeline stage in the `pipeline` section.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageSnapshot {
    pub stage: u32,
    pub device: u32,
    /// First and last layer of the stage.
    pub layers: [u32; 2],
    /// Fraction of the last 10 s the stage was executing.
    pub busy_ratio: f64,
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
    /// `launch`, `device_wait`, `sample`, `detokenize`, `emit`, `complete`); they sum to
    /// `duration_ms`. The scheduler leaves it empty; the engine fills it in the document it
    /// publishes. Under overlap scheduling the iteration's window (its plan to its completion)
    /// spans two engine turns, and its stages are the host's time over that window — which
    /// includes waiting for and finishing the iteration before it.
    pub stages_ms: BTreeMap<String, f64>,
    /// Milliseconds of `duration_ms` after the iteration's launch returned during which its
    /// forward pass was in flight on the device while the host was in other stages (the
    /// previous iteration's wait and host work, the next plan: the device time overlap
    /// scheduling hides; user decision 2026-10-07 2 B). A part of
    /// `duration_ms`, already counted in `stages_ms`; 0 on the serial loop. The engine fills it.
    pub overlap_ms: f64,
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
    /// Iteration of the last plan with an item of this sequence (0: none). With micro-batches
    /// the least recently stepped decodes go first, so a capped micro-batch starves none.
    last_step: u64,
    /// The sequence's pending copy-on-write tail copy `(src, dst)` (P5p S-5), set at admission
    /// from the prefix's [`turbine_kv::hierarchy::TailCopy`]; planned into the iteration that
    /// carries the sequence's first prefill chunk, and cleared wherever the table drops before
    /// it could be planned (preemption, finish, removal) — the copy must never fire against a
    /// stale table.
    tail_pending: Option<(BlockId, BlockId)>,
    /// Pending multi-token append (P5p S-9): the next plan feeds `1 + tokens` positions of
    /// this decoding sequence as one prefill-kind item. Cleared wherever the table drops
    /// (preemption, finish, removal) and decided by every plan: planned, or dropped for a
    /// plain decode when the budget, the logits-row capacity or the pool cannot cover it.
    extend_pending: Option<Vec<u32>>,
}

/// A plan awaiting `complete`.
#[derive(Debug, Default)]
struct InFlight {
    items: Vec<(SeqId, BatchKind)>,
    /// Running requests with an item or a fork in the plan.
    requests: Vec<RequestId>,
    started: Duration,
    last: LastIteration,
    /// Sequences of this plan reported finished by another plan's outcome; finished when this
    /// plan completes, so their blocks are not freed under it.
    deferred_finished: Vec<SeqId>,
    /// The plan has work (`!IterationPlan::is_empty`); only these count as micro-batches.
    executes: bool,
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
/// other requests' reservations already paid for or which were cached unreferenced. Only full
/// blocks are excluded (P5p S-5): the private tail block counts against the reservation like any
/// block the request writes.
fn reserved_kv_blocks(r: &SchedRequest, block_tokens: u32) -> u64 {
    let bt = block_tokens.max(1);
    let cached = r
        .cached_prefix
        .as_ref()
        .map_or(0, |a| u64::from(a.cached_tokens / bt));
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
    /// Plans awaiting `complete`, by iteration (empty plans too, until the next `plan`).
    in_flight: BTreeMap<u64, InFlight>,
    /// Requests of a plan in flight, with its iteration: no other plan may touch them.
    busy: HashMap<RequestId, u64>,
    /// Plans with work that may be in flight at once (`parallel.pipeline.micro_batches`).
    micro_batches: u32,
    /// The model's per-sequence state (P7 S-1): every request's worst-case reservation holds
    /// it beside the KV blocks (`ResourceEstimate::state_bytes`).
    state_bytes: u64,
    iteration: u64,
    admissions: u64,
    preemptions_total: u64,
    shutting_down: bool,
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
            in_flight: BTreeMap::new(),
            busy: HashMap::new(),
            micro_batches: 1,
            state_bytes: 0,
            iteration: 0,
            admissions: 0,
            preemptions_total: 0,
            shutting_down: false,
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

    /// Keep up to `m` plans in flight (pipeline micro-batches, Phase 5 S-10; `m` ≥ 1, default
    /// 1): each plan skips the requests of the others and is capped to its share of the
    /// running set (module docs).
    pub fn with_micro_batches(mut self, m: u32) -> Scheduler {
        self.micro_batches = m.max(1);
        self
    }

    pub fn micro_batches(&self) -> u32 {
        self.micro_batches
    }

    /// The model's per-sequence state (P7 S-1): each submitted request's estimate carries
    /// `state.bytes_per_sequence()`, so its worst-case reservation holds the state beside its
    /// KV blocks and releases both together (finish, cancel, preemption).
    pub fn with_state_layout(mut self, state: StateLayout) -> Scheduler {
        self.state_bytes = state.bytes_per_sequence();
        self
    }

    /// The per-sequence state bytes each request reserves (0 for families without state).
    pub fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    /// The worst-case reservation an admitted request holds (`None` without a gate, or once it
    /// left the scheduler).
    pub fn reservation(&self, id: RequestId) -> Option<&Reservation> {
        self.requests.get(&id)?.reservation.as_ref()
    }

    /// Plans with work that were planned and not completed yet.
    pub fn micro_batches_in_flight(&self) -> usize {
        self.in_flight.values().filter(|f| f.executes).count()
    }

    /// Iterations of the plans with work awaiting `complete`, oldest first.
    pub fn in_flight_iterations(&self) -> Vec<u64> {
        self.in_flight
            .iter()
            .filter(|(_, f)| f.executes)
            .map(|(i, _)| *i)
            .collect()
    }

    /// The iteration of the plan in flight holding request `id`, if any.
    pub fn in_flight_iteration_of(&self, id: RequestId) -> Option<u64> {
        self.busy.get(&id).copied()
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
        r.estimate.state_bytes = self.state_bytes;
        let block_tokens = self.params.block_tokens;
        let submit_no = self.queue.next_order();
        let window = self.params.cache_aware_window;
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
        let key = self.policy.admission_key(&Self::admission_info(
            window, &r, submit_no, false, submit_no,
        ));
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
        r.estimate.state_bytes = self.state_bytes;
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

    /// The admission info the policy sees for `r` (Phase 5p S-7): the estimate's prompt and
    /// attached-prefix lengths and the configured `cache_aware` window. An associated
    /// function without `&self`, so a call site can use it while the gate or the request map
    /// is mutably borrowed; copy `self.params.cache_aware_window` before that borrow.
    fn admission_info(
        window: Duration,
        r: &SchedRequest,
        submit_no: u64,
        preempted: bool,
        push_no: u64,
    ) -> AdmissionInfo {
        AdmissionInfo {
            priority: r.priority,
            arrival: r.arrival,
            submit_no,
            preempted,
            push_no,
            prompt_tokens: r.estimate.prompt_tokens,
            cached_prefix_tokens: r.estimate.cached_prefix_tokens,
            window,
            state_bytes: r.estimate.state_bytes,
        }
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
        let key = self.policy.admission_key(&Self::admission_info(
            self.params.cache_aware_window,
            &r,
            submit_no,
            false,
            submit_no,
        ));
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
                    last_step: 0,
                    tail_pending: None,
                    extend_pending: None,
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

    /// Queued requests release their attached prefixes for pressure reclaim
    /// ([`AdmissionGate::detach_prefixes`]; the engine caps `want` per controller tick and gives
    /// the references back). Without a gate nothing is released.
    pub fn detach_queued_prefixes(
        &mut self,
        pool: &BlockPool,
        want: usize,
    ) -> Vec<(RequestId, PrefixAttach, usize)> {
        let bt = self.params.block_tokens;
        match self.gate.as_mut() {
            Some(g) if want > 0 => g.detach_prefixes(pool, want, |r| {
                u32::try_from(request_kv_blocks(r, bt)).unwrap_or(u32::MAX)
            }),
            _ => Vec::new(),
        }
    }

    /// Admitted requests that released their prefix in the admission queue and wait for the
    /// engine to attach it again before they start.
    pub fn awaiting_reattach(&self) -> Vec<RequestId> {
        self.queue
            .iter()
            .filter(|id| self.requests.get(id).is_some_and(|r| r.req.reattach))
            .collect()
    }

    /// The prefix a released request attached again (or an empty attach: recompute). Its KV
    /// reservation keeps covering the whole request, and the attached blocks are committed
    /// against it (`block_bytes` each): they are referenced, so the ledger's `held` counts them,
    /// and an uncommitted reservation would count them a second time (on the lab, 1.6 GB of such
    /// double counting pushed `kv_utilization` from 0.58 to 0.99 and into SURVIVAL). Returns the
    /// attach when `id` is not waiting for one (cancelled meanwhile): the caller releases the
    /// blocks.
    pub fn reattach(
        &mut self,
        id: RequestId,
        attach: PrefixAttach,
        block_bytes: u64,
    ) -> Option<PrefixAttach> {
        if !self.requests.get(&id).is_some_and(|r| r.req.reattach) {
            return Some(attach);
        }
        self.commit_reattached(id, attach.blocks.len(), block_bytes);
        let r = self.requests.get_mut(&id).expect("checked above");
        let cached = attach.cached_tokens;
        let e = &mut r.req.estimate;
        e.cached_prefix_tokens = cached;
        e.new_prefill_tokens = r.req.prompt_len - cached.min(r.req.prompt_len);
        r.req.reattach = false;
        r.req.cached_prefix = (!attach.blocks.is_empty()).then_some(attach);
        None
    }

    /// A released request's new attach holds `blocks` L0 blocks (also while its promotions are
    /// in flight: their targets are allocated and referenced at once): its reservation has
    /// `blocks × block_bytes` committed in all, so the ledger counts them once (as committed, not
    /// also as `held`). Nothing else has committed against the reservation of a request that
    /// has not started.
    pub fn commit_reattached(&mut self, id: RequestId, blocks: usize, block_bytes: u64) {
        let Some(res) = self
            .requests
            .get_mut(&id)
            .filter(|r| r.req.reattach)
            .and_then(|r| r.reservation.as_mut())
        else {
            return;
        };
        let target = (blocks as u64).saturating_mul(block_bytes);
        res.commit_bytes(target.saturating_sub(res.committed()));
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

    /// Asks the next plan to feed `1 + tokens` positions of the decoding sequence `seq` as one
    /// prefill-kind item (`BatchKind::Prefill { start, len }` with `len = 1 + tokens`,
    /// `logits_rows = len`; P5p S-9): the engine's verified multi-token step feeds the
    /// sampled-but-unfed token and the grammar's forced tokens, and its `complete` reports the
    /// kept tokens as `appended` and the rest as `rolled_back`. The next plan takes the item's
    /// tokens from the budget left after the decodes and plans it before the prefill chunks;
    /// an extension the budget, the executor's logits-row capacity (`max_running_requests`
    /// rows per iteration) or the pool cannot cover falls back to a plain decode and is
    /// dropped. Callers bound `tokens` by `structured_output.jump_forward_max_tokens`; the
    /// plan additionally caps the item by the policy's chunk cap and the request's own
    /// `prompt_len + max_new_tokens`. A sequence that is not decoding (prefilling, waiting,
    /// paused, finished, untracked) is `SchedError::NotDecoding`; `tokens` 0 is a no-op.
    pub fn extend(&mut self, seq: SeqId, tokens: Vec<u32>) -> Result<(), SchedError> {
        let Some(e) = self.seqs.get_mut(&seq) else {
            return Err(SchedError::NotDecoding);
        };
        if e.state != RequestState::Decoding {
            return Err(SchedError::NotDecoding);
        }
        if !tokens.is_empty() {
            e.extend_pending = Some(tokens);
        }
        Ok(())
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
        for item in plan.items.drain(limit..) {
            let written = match item.kind {
                BatchKind::Decode => 1,
                BatchKind::Prefill { len, .. } => len,
            };
            self.shorten_table(pool, item.seq, written);
        }
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
        self.last.prefill_tokens = plan.prefill_tokens();
        self.last.decode_tokens = plan.decode_tokens();
        self.last.requests = plan.items.len() as u32;
        let Some(f) = self.in_flight.get_mut(&plan.iteration) else {
            tracing::warn!(
                event = "scheduler_outcome_mismatch",
                iteration = plan.iteration,
                "shrunk a plan that is not in flight"
            );
            return;
        };
        f.items = plan.items.iter().map(|i| (i.seq, i.kind)).collect();
        f.requests.retain(|id| kept.contains(id));
        f.last.prefill_tokens = self.last.prefill_tokens;
        f.last.decode_tokens = self.last.decode_tokens;
        f.last.requests = self.last.requests;
        let iteration = plan.iteration;
        self.busy
            .retain(|id, it| *it != iteration || kept.contains(id));
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
        for f in self.in_flight.values_mut() {
            f.requests.retain(|id| !failed.contains(id));
        }
        self.publish_gauges();
        failed
    }

    /// State of a sequence the scheduler still tracks.
    pub fn seq_state(&self, seq: SeqId) -> Option<RequestState> {
        self.seqs.get(&seq).map(|e| e.state)
    }

    /// Positions of `seq` the plans so far feed (its block table's `tokens`): one past the
    /// host's last sampled token while an iteration planned after that sample still has to
    /// be completed (overlap scheduling launches it before the host samples; P5p S-10).
    pub fn seq_tokens(&self, seq: SeqId) -> Option<u32> {
        self.seqs.get(&seq).map(|e| e.table.tokens)
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

    /// An admitted request waits for its first prefill chunk (P5p S-3): one in the scheduler's
    /// own queue — admitted by the gate (or, without a gate, submitted) and not started, or
    /// preempted and waiting to prefill again. The gate's queue does not count: those requests
    /// wait for a slot, not for the next plan. Overlap scheduling reads this before launching
    /// an iteration ahead.
    pub fn has_unstarted_admitted(&self) -> bool {
        !self.queue.is_empty()
    }

    /// One iteration's batch (P2 §Scheduling rules 1–3).
    ///
    /// With micro-batches, the requests of the plans in flight are skipped and the plan is
    /// capped to its share of the running set; with every micro-batch in flight the plan is
    /// empty.
    pub fn plan(&mut self, pool: &mut BlockPool, limits: &IterationLimits) -> IterationPlan {
        self.iteration += 1;
        let now = self.clock.now_mono();
        // Empty plans hold nothing; an engine need not complete them.
        self.in_flight.retain(|_, f| f.executes);
        let mut plan = IterationPlan {
            iteration: self.iteration,
            ..IterationPlan::default()
        };
        if self.micro_batches_in_flight() >= self.micro_batches as usize {
            tracing::debug!(
                event = "micro_batches_full",
                in_flight = self.micro_batches_in_flight(),
                "every micro-batch is in flight: empty plan"
            );
            self.in_flight.insert(
                plan.iteration,
                InFlight {
                    started: now,
                    ..InFlight::default()
                },
            );
            return plan;
        }
        let (seq_cap, token_cap) = self.micro_batch_caps();
        let mut in_plan: HashSet<RequestId> = HashSet::new();
        let mut preempted_now: HashSet<RequestId> = HashSet::new();

        // (1) Drop cancelled and queue-timed-out requests; their blocks return first.
        self.drop_cancelled(pool, now, &mut plan);
        // SURVIVAL, option A: admitted requests that have not started give their KV
        // reservations back and wait in the admission queue again.
        if limits.requeue_unstarted {
            self.requeue_unstarted(pool, &mut plan);
        }

        // (2) Multi-token appends (P5p S-9) and then every decodable sequence decodes; preempt
        // until the pool covers the decodes.
        let cap = seq_cap.min(token_cap as usize);
        let candidates = self.decode_candidates(cap);
        let mut rows = 0u32;
        let (extended, ext_reserved) = self.plan_extensions(
            pool,
            limits,
            &candidates,
            token_cap,
            &mut plan,
            &mut in_plan,
            &mut rows,
        );

        self.plan_decodes(
            pool,
            limits,
            cap,
            &candidates,
            &extended,
            &mut plan,
            &mut in_plan,
            &mut preempted_now,
            &mut rows,
        );

        // Forks of finished shared prefills (n > 1) wait for blocks, never re-prefill.
        self.plan_forks(pool, &mut plan, &mut in_plan, &mut preempted_now);

        // (3) Prefill: continuing prefills oldest first, then admissions. An iteration's total
        // logits rows stay at `max_running_requests` (the executor's head capacity), so an
        // extension's extra rows shrink what the prefills may take.
        let decode_tokens = plan.decode_tokens();
        let fraction = limits.prefill_budget_fraction.clamp(0.0, 1.0);
        let mut budget = (f64::from(token_cap.saturating_sub(decode_tokens)) * fraction).floor()
            as u32
            - ext_reserved;
        let chunk_cap = self.policy.chunk_cap(&self.params, limits);
        let rows_cap = self.params.max_running_requests;
        for id in self.running.clone() {
            if self.busy.contains_key(&id) {
                continue;
            }
            for seq in self.prefill_seqs(id) {
                if budget == 0 || !self.is_running(id) || plan.items.len() >= seq_cap {
                    break;
                }
                self.schedule_chunk(
                    pool,
                    seq,
                    &mut budget,
                    chunk_cap,
                    true,
                    rows_cap,
                    &mut rows,
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
                && plan.items.len() < seq_cap
                && rows < rows_cap
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
                    if budget == 0 || plan.items.len() >= seq_cap {
                        break;
                    }
                    self.schedule_chunk(
                        pool,
                        seq,
                        &mut budget,
                        chunk_cap,
                        false,
                        rows_cap,
                        &mut rows,
                        &mut plan,
                        &mut in_plan,
                        &mut preempted_now,
                    );
                }
            }
        }

        let requests: Vec<RequestId> = self
            .running
            .iter()
            .copied()
            .filter(|id| in_plan.contains(id))
            .collect();
        for &id in &requests {
            self.busy.insert(id, plan.iteration);
        }
        for item in &plan.items {
            if let Some(e) = self.seqs.get_mut(&item.seq) {
                e.last_step = plan.iteration;
            }
        }
        self.last = LastIteration {
            prefill_tokens: plan.prefill_tokens(),
            decode_tokens,
            requests: plan.items.len() as u32,
            duration_ms: 0.0,
            stages_ms: BTreeMap::new(),
            overlap_ms: 0.0,
        };
        self.in_flight.insert(
            plan.iteration,
            InFlight {
                items: plan.items.iter().map(|i| (i.seq, i.kind)).collect(),
                requests,
                started: now,
                last: self.last.clone(),
                deferred_finished: Vec::new(),
                executes: !plan.is_empty(),
            },
        );
        self.prev_admitted = self.admitted_count();
        self.publish_gauges();
        plan
    }

    /// Per-plan caps `(sequences, tokens)`: `max_running_requests` sequences (the executor's
    /// batch is sized for that many; with `n` > 1 the running requests hold more sequences, which
    /// then take turns) and `max_batch_tokens` with one micro-batch; with `m`, at most
    /// `ceil(live / m)` sequences and (chunked prefill only — a whole prompt must still fit one
    /// plan) `max_batch_tokens / m` tokens, where `live` counts the running sequences that
    /// prefill or decode (in flight or not) plus the waiting requests the free running slots
    /// could start.
    fn micro_batch_caps(&self) -> (usize, u32) {
        let m = self.micro_batches.max(1);
        let max_seqs = self.params.max_running_requests as usize;
        if m == 1 {
            return (max_seqs, self.params.max_batch_tokens);
        }
        let live = self
            .running
            .iter()
            .flat_map(|id| self.requests[id].req.seqs.iter())
            .filter(|s| {
                self.seqs.get(s).is_some_and(|e| {
                    matches!(e.state, RequestState::Prefilling | RequestState::Decoding)
                })
            })
            .count();
        let free_slots =
            (self.params.max_running_requests as usize).saturating_sub(self.running.len());
        let startable = self.queue_len().min(free_slots);
        let seqs = (live + startable)
            .div_ceil(m as usize)
            .clamp(1, max_seqs.max(1));
        let tokens = if self.params.chunked_prefill {
            (self.params.max_batch_tokens / m).max(1)
        } else {
            self.params.max_batch_tokens
        };
        (seqs, tokens)
    }

    /// The snapshot's `last_iteration` (no stages): right after [`Scheduler::complete`], the
    /// iteration it completed, `duration_ms` included (a `plan` replaces it).
    pub fn last_iteration(&self) -> &LastIteration {
        &self.last
    }

    /// Apply an executed plan: prefill completion, appended tokens, finished sequences, or
    /// the failure of every request in the iteration.
    ///
    /// The plan is matched by `outcome.iteration`; plans may complete in any order. A failure
    /// fails only that plan's requests. `finished` and `appended` may name sequences of
    /// earlier, already completed plans (late finishes); a sequence held by another plan in
    /// flight finishes when that plan completes.
    pub fn complete(&mut self, pool: &mut BlockPool, outcome: IterationOutcome) {
        let Some(entry) = self.in_flight.remove(&outcome.iteration) else {
            tracing::warn!(
                event = "scheduler_outcome_mismatch",
                expected = ?self.in_flight.keys().collect::<Vec<_>>(),
                got = outcome.iteration,
                "outcome for a plan that is not in flight"
            );
            self.apply_samples(pool, outcome);
            self.publish_gauges();
            return;
        };
        let iteration = outcome.iteration;
        self.busy.retain(|_, it| *it != iteration);
        let elapsed = self.clock.now_mono().saturating_sub(entry.started);
        self.last = entry.last;
        self.last.duration_ms = elapsed.as_secs_f64() * 1000.0;
        if !entry.items.is_empty()
            && let Some(m) = &self.metrics
        {
            m.iteration(
                elapsed.as_secs_f64(),
                self.last.prefill_tokens,
                self.last.decode_tokens,
                self.last.requests,
            );
        }
        let in_flight = entry.items;
        let in_flight_requests = entry.requests;

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

        for &(seq, n) in &outcome.appended {
            if let Some(e) = self.seqs.get_mut(&seq) {
                e.generated += n;
            }
        }
        for (seq, n) in outcome.rolled_back {
            self.shorten_table(pool, seq, n);
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
        for &(seq, _reason) in &outcome.finished {
            self.finish_or_defer(pool, seq);
        }
        for seq in entry.deferred_finished {
            self.finish_seq(pool, seq);
        }
        self.publish_gauges();
    }

    /// Tokens and finishes of an outcome without a plan in flight.
    fn apply_samples(&mut self, pool: &mut BlockPool, outcome: IterationOutcome) {
        if outcome.failed.is_some() {
            return;
        }
        for (seq, n) in outcome.appended {
            if let Some(e) = self.seqs.get_mut(&seq) {
                e.generated += n;
            }
        }
        for (seq, n) in outcome.rolled_back {
            self.shorten_table(pool, seq, n);
        }
        for (seq, _reason) in outcome.finished {
            self.finish_or_defer(pool, seq);
        }
    }

    /// Removes `n` tokens from the end of `seq`'s table (`IterationOutcome::rolled_back`, P5p
    /// S-9; also `shrink_plan`), freeing blocks left empty.
    fn shorten_table(&mut self, pool: &mut BlockPool, seq: SeqId, n: u32) {
        let bt = self.params.block_tokens;
        let Some(e) = self.seqs.get_mut(&seq) else {
            return;
        };
        e.table.tokens = e.table.tokens.saturating_sub(n);
        let keep = blocks_for_tokens(e.table.tokens, bt) as usize;
        if e.table.blocks.len() > keep {
            let extra: SmallVec<[BlockId; 16]> = e.table.blocks.drain(keep..).collect();
            pool.release(&extra);
        }
    }

    /// Finish `seq` now, or when the plan in flight holding its request completes.
    fn finish_or_defer(&mut self, pool: &mut BlockPool, seq: SeqId) {
        if let Some(e) = self.seqs.get(&seq)
            && let Some(it) = self.busy.get(&e.request)
            && let Some(f) = self.in_flight.get_mut(it)
        {
            f.deferred_finished.push(seq);
            return;
        }
        self.finish_seq(pool, seq);
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
            pipeline: (self.micro_batches > 1).then(|| PipelineSnapshot {
                stages: Vec::new(),
                micro_batches: self.micro_batches,
                micro_batches_in_flight: self.micro_batches_in_flight() as u32,
            }),
        }
    }

    // ---- rules -------------------------------------------------------------------------

    fn drop_cancelled(&mut self, pool: &mut BlockPool, now: Duration, plan: &mut IterationPlan) {
        self.drop_gated(pool, plan);
        // A request of a plan in flight is dropped by the first plan after that one completed.
        let candidates: Vec<RequestId> = self
            .running
            .iter()
            .copied()
            .chain(self.queue.iter())
            .filter(|id| !self.busy.contains_key(id))
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
        let window = self.params.cache_aware_window;
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
            let key = self.policy.admission_key(&Self::admission_info(
                window, &req, submit_no, false, submit_no,
            ));
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

    #[allow(clippy::too_many_arguments)]
    fn plan_decodes(
        &mut self,
        pool: &mut BlockPool,
        limits: &IterationLimits,
        cap: usize,
        candidates: &[SeqId],
        extended: &HashSet<SeqId>,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        preempted_now: &mut HashSet<RequestId>,
        rows: &mut u32,
    ) {
        let bt = self.params.block_tokens;
        // Preempt until the pool covers every decode (never, without allow_preempt). With
        // micro-batches the victim is still the worst-ranked running request; when it is in
        // flight nothing is preempted, and the decodes lacking a block wait for its plan.
        let mut victim_in_flight = false;
        while limits.allow_preempt
            && self.decode_blocks_needed(cap, extended) > pool.available_blocks()
        {
            let Some(victim) = self.worst_running(|_| true) else {
                break;
            };
            if self.busy.contains_key(&victim) {
                victim_in_flight = true;
                break;
            }
            self.preempt(pool, victim, plan, preempted_now);
        }
        for seq in candidates {
            if extended.contains(seq) {
                continue; // this plan's step for the sequence is the extension item
            }
            // A preemption of this plan's loop may have returned the sequence to the queue
            // after `candidates` was taken.
            let Some(e) = self
                .seqs
                .get_mut(seq)
                .filter(|e| e.state == RequestState::Decoding)
            else {
                continue;
            };
            let need = e.table.blocks_needed(1, bt);
            let Ok(blocks) = Self::allocate_for(
                pool,
                &mut self.requests,
                e.request,
                need,
                self.params.recent_window,
            ) else {
                // With preemption allowed the loop above made the pool cover every decode;
                // without it (a gate below SURVIVAL) the sequence waits for a block.
                if victim_in_flight {
                    tracing::debug!(
                        event = "decode_deferred",
                        seq = seq.0,
                        reason = "victim_in_flight",
                        "decode waits for the micro-batch holding the preemption victim"
                    );
                } else if limits.allow_preempt {
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
            *rows += 1;
            in_plan.insert(e.request);
            plan.items.push(BatchItem {
                seq: *seq,
                kind: BatchKind::Decode,
                block_table: e.table.clone(),
                draft: Vec::new(),
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
            if !r.shared_prefill_done || r.admitted.is_none() || self.busy.contains_key(&id) {
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
                    while let Some(victim) = self.worst_running(|v| {
                        v.0 != id
                            && !in_plan.contains(&v.0)
                            && !self.busy.contains_key(&v.0)
                            && v.1 > rank
                    }) {
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

    /// Plans a decoding sequence's pending extension (P5p S-9): the item feeds `1 + tokens`
    /// positions from the budget left after the decodes, before the prefill chunks. Not
    /// coverable — the budget, the plan's logits-row capacity (`rows_cap` =
    /// `max_running_requests`, the executor's head bound) or the pool — the extension is
    /// dropped and the sequence decodes; a request whose plan is in flight keeps its pending
    /// extension for the next plan. Returns the extended sequences and the token budget the
    /// extension items consumed.
    #[allow(clippy::too_many_arguments)]
    fn plan_extensions(
        &mut self,
        pool: &mut BlockPool,
        limits: &IterationLimits,
        candidates: &[SeqId],
        token_cap: u32,
        plan: &mut IterationPlan,
        in_plan: &mut HashSet<RequestId>,
        rows: &mut u32,
    ) -> (HashSet<SeqId>, u32) {
        let bt = self.params.block_tokens;
        let fraction = limits.prefill_budget_fraction.clamp(0.0, 1.0);
        let rows_cap = self.params.max_running_requests;
        let mut extended = HashSet::new();
        let mut reserved = 0u32;
        let mut replaced_decodes = 0u32;
        for &seq in candidates {
            let Some(e) = self.seqs.get_mut(&seq) else {
                continue;
            };
            let request = e.request;
            if self.busy.contains_key(&request) {
                continue; // no plan may touch an in-flight request; the extension waits
            }
            let Some(tokens) = e.extend_pending.take() else {
                continue;
            };
            let (table_tokens, ceiling) = {
                let r = &self.requests[&request];
                (
                    e.table.tokens,
                    r.req.prompt_len.saturating_add(r.req.max_new_tokens),
                )
            };
            // Nothing left to append (the engine extended past `prompt_len + max_new_tokens`):
            // a plain decode. The plan may cap the run (the request's own ceiling): the item
            // carries the truncated draft.
            let Some(len) = u32::try_from(tokens.len() + 1)
                .ok()
                .map(|l| l.min(ceiling.saturating_sub(table_tokens)))
                .filter(|&l| l > 1)
            else {
                continue;
            };
            let budget =
                (f64::from(token_cap.saturating_sub(candidates.len() as u32 - replaced_decodes))
                    * fraction)
                    .floor() as u32;
            // The plan's logits rows also cover the decodes planned after this (`rows_cap` =
            // `max_running_requests`): every other candidate still decodes unless its
            // extension was planned, so reserve a row per remaining candidate.
            let decodes_left = (candidates.len() as u32).saturating_sub(1 + replaced_decodes);
            if len > budget || *rows + len + decodes_left > rows_cap {
                continue; // dropped: the sequence decodes
            }
            let need = e.table.blocks_needed(len, bt);
            let Ok(blocks) = Self::allocate_for(
                pool,
                &mut self.requests,
                request,
                need,
                self.params.recent_window,
            ) else {
                continue; // dropped: the pool cannot cover the extension's blocks
            };
            let e = self.seqs.get_mut(&seq).expect("sequence tracked");
            e.table.blocks.extend(blocks);
            let start = e.table.tokens;
            e.table.tokens += len;
            *rows += len;
            reserved += len;
            replaced_decodes += 1;
            in_plan.insert(request);
            extended.insert(seq);
            plan.items.push(BatchItem {
                seq,
                kind: BatchKind::Prefill { start, len },
                block_table: e.table.clone(),
                draft: tokens[..(len - 1) as usize].to_vec(),
            });
        }
        (extended, reserved)
    }

    #[allow(clippy::too_many_arguments)]
    fn schedule_chunk(
        &mut self,
        pool: &mut BlockPool,
        seq: SeqId,
        budget: &mut u32,
        chunk_cap: u32,
        may_preempt: bool,
        rows_cap: u32,
        rows: &mut u32,
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
        if *rows + 1 > rows_cap {
            // The iteration's logits rows are at `max_running_requests` (P5p S-9); a prefill
            // chunk's last position is one of them.
            return false;
        }
        let need = e.table.blocks_needed(len, bt);
        // With a gate every request's KV is reserved: prefills never preempt (P3 §Throttle).
        if need > pool.available_blocks() && may_preempt && self.gate.is_none() {
            let rank = self.rank(id);
            while need > pool.available_blocks() {
                let Some(victim) = self.worst_running(|v| {
                    v.0 != id
                        && !in_plan.contains(&v.0)
                        && !self.busy.contains_key(&v.0)
                        && v.1 > rank
                }) else {
                    break;
                };
                self.preempt(pool, victim, plan, preempted_now);
            }
        }
        let Ok(blocks) = Self::allocate_for(
            pool,
            &mut self.requests,
            id,
            need,
            self.params.recent_window,
        ) else {
            return false; // new prefills wait for blocks
        };
        let e = self.seqs.get_mut(&seq).expect("sequence tracked");
        e.table.blocks.extend(blocks);
        let start = e.table.tokens;
        e.table.tokens += len;
        *budget -= len;
        *rows += 1;
        in_plan.insert(id);
        // The copy-on-write tail copy (P5p S-5) rides with the sequence's first prefill chunk;
        // the engine executes it with the fork copies before the forward.
        if let Some((src, dst)) = e.tail_pending.take() {
            plan.tail_copies.push((seq, src, dst));
        }
        plan.items.push(BatchItem {
            seq,
            kind: BatchKind::Prefill { start, len },
            draft: Vec::new(),
            block_table: e.table.clone(),
        });
        true
    }

    /// Would admitting `id` now fit its first chunk plus the free-block watermark?
    fn admissible(&self, pool: &BlockPool, id: RequestId, budget: u32, chunk_cap: u32) -> bool {
        let Some(r) = self.requests.get(&id) else {
            return false;
        };
        // It released its prefix while queued: it starts once the engine attached it again.
        if r.req.reattach {
            return false;
        }
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
                // The tail copy is planned with the sequence's first prefill chunk; wherever
                // the table drops before that, the pending copy is dropped with it.
                e.tail_pending = a.tail_copy.map(|t| (t.src, t.dst));
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
        let window = self.params.cache_aware_window;
        let r = self.requests.get_mut(&id).expect("running request tracked");
        r.admitted = None;
        let key = self.policy.admission_key(&Self::admission_info(
            window,
            &r.req,
            r.submit_no,
            true,
            push_no,
        ));
        let seqs = r.req.seqs.clone();
        for seq in seqs {
            let e = self.seqs.get_mut(&seq).expect("sequence tracked");
            let table = std::mem::take(&mut e.table);
            pool.release(&table.blocks);
            // The pending tail copy dies with the table it copied into.
            e.tail_pending = None;
            e.extend_pending = None;
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
            e.tail_pending = None;
            e.extend_pending = None;
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
                e.tail_pending = None;
                e.extend_pending = None;
                if e.state.is_live() {
                    e.set_state(terminal, *seq);
                }
            }
        }
        self.running.retain(|x| *x != id);
        self.queue.remove(id);
        self.busy.remove(&id);
    }

    // ---- helpers -----------------------------------------------------------------------

    /// `n` blocks for request `id`, paid from its reservation when it has one: the trailing
    /// `min(n, window)` from the recent window's BF16 page class, the rest from the base class
    /// (table order: base blocks first, window blocks last).
    fn allocate_for(
        pool: &mut BlockPool,
        requests: &mut HashMap<RequestId, ReqEntry>,
        id: RequestId,
        n: u32,
        window: Option<RecentWindow>,
    ) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        let Some(w) = window else {
            return match requests.get_mut(&id).and_then(|r| r.reservation.as_mut()) {
                Some(res) => pool.allocate_reserved(n, res),
                None => pool.allocate(n),
            };
        };
        let win = (w.blocks as usize).min(n as usize);
        let base = n as usize - win;
        let mut out = match requests.get_mut(&id).and_then(|r| r.reservation.as_mut()) {
            Some(res) => pool.allocate_reserved(base as u32, res)?,
            None => pool.allocate(base as u32)?,
        };
        if win > 0 {
            let window_blocks = match requests.get_mut(&id).and_then(|r| r.reservation.as_mut()) {
                Some(res) => pool.allocate_in_reserved(w.format, win as u32, res)?,
                None => pool.allocate_in(w.format, win as u32)?,
            };
            out.extend(window_blocks);
        }
        Ok(out)
    }

    fn is_running(&self, id: RequestId) -> bool {
        self.requests.get(&id).is_some_and(|r| r.admitted.is_some())
    }

    /// Blocks this plan's decodes need for one more token each. The sequences of `extended`
    /// step through their extension item instead and are not counted.
    fn decode_blocks_needed(&self, cap: usize, extended: &HashSet<SeqId>) -> u32 {
        let bt = self.params.block_tokens;
        self.decode_candidates(cap)
            .iter()
            .filter(|s| !extended.contains(s))
            .map(|s| self.seqs[s].table.blocks_needed(1, bt))
            .sum()
    }

    /// The decodes of this plan: the whole decode set with one micro-batch while it fits `cap`;
    /// with more micro-batches, or more decoding sequences than `cap` (forks of `n` > 1), the
    /// sequences of requests not in flight, least recently stepped first (admission order
    /// among equals), at most `cap`.
    fn decode_candidates(&self, cap: usize) -> Vec<SeqId> {
        let mut set = self.decode_set();
        if self.micro_batches > 1 {
            set.retain(|s| !self.busy.contains_key(&self.seqs[s].request));
        }
        if self.micro_batches > 1 || set.len() > cap {
            set.sort_by_key(|s| self.seqs[s].last_step);
            set.truncate(cap);
        }
        set
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

    use super::*;
    use crate::SchedError;

    use smallvec::smallvec;
    use turbine_core::clock::FakeClock;
    use turbine_core::types::{DType, DeviceId, KvLayout, Priority};
    use turbine_kv::{BlockPool, BlockPoolConfig};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

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
            cache_aware_window: Duration::from_secs(1),
            recent_window: None,
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
                rolled_back: Vec::new(),
                failed: None,
            },
        );
    }

    /// P5p S-3: `has_unstarted_admitted` is true from submission until the request's first
    /// prefill chunk is planned, and again while one waits for a running slot. Breaks if the
    /// engine's overlap rule would launch ahead of a request that the next plan could start.
    #[test]
    fn has_unstarted_admitted_until_first_chunk() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(
            SchedulerParams {
                max_running_requests: 1,
                ..params()
            },
            Arc::new(clock.clone()),
        );
        let mut p = pool(64);
        assert!(!s.has_unstarted_admitted());
        s.submit(request(1, 1, 8, 2), p.total_blocks()).unwrap();
        assert!(s.has_unstarted_admitted());
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert!(!s.has_unstarted_admitted(), "the first chunk is planned");
        complete_all(&mut s, &mut p, &plan, &[]);
        // A second request waits behind the only running slot.
        s.submit(request(2, 2, 8, 2), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&plan), [(1, BatchKind::Decode)]);
        assert!(s.has_unstarted_admitted());
        complete_all(&mut s, &mut p, &plan, &[1]);
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&plan), [(2, BatchKind::Prefill { start: 0, len: 8 })]);
        assert!(!s.has_unstarted_admitted());
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

    /// S-9: `Scheduler::extend` plans a decoding sequence's next step as one `1 + tokens`
    /// prefill-kind item, taken from the budget left after the decodes and before the prefill
    /// chunks; `rolled_back` shortens the table and frees a block left empty; an extension the
    /// budget cannot cover falls back to a plain decode and is dropped; another state is
    /// `NotDecoding`; a preemption drops the pending extension.
    #[test]
    fn extend_and_roll_back() {
        let clock = FakeClock::new(Duration::ZERO);

        // A decoding sequence extended by 5 tokens is planned as one 6-token item, before the
        // iteration's prefill chunk and from the budget left after the decodes (none here: the
        // extension replaces the sequence's decode).
        let mut s = Scheduler::new(params(), Arc::new(clock.clone()));
        let mut p = pool(64);
        s.submit(request(1, 1, 12, 50), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [(1, BatchKind::Prefill { start: 0, len: 12 })]
        );
        complete_all(&mut s, &mut p, &plan, &[]);
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&plan), [(1, BatchKind::Decode)]);
        complete_all(&mut s, &mut p, &plan, &[]); // table 13 tokens (1 block), generated 2

        // A 100-token prompt waits behind the extension's 6 tokens (budget 64 − 6 = 58, the
        // chunk cap keeps its first chunk at 32).
        s.submit(request(2, 2, 100, 10), p.total_blocks()).unwrap();
        s.extend(SeqId(1), vec![1, 2, 3, 4, 5]).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Prefill { start: 13, len: 6 }),
                (2, BatchKind::Prefill { start: 0, len: 32 }),
            ]
        );
        assert_eq!(plan.items[0].block_table.tokens, 19);
        assert_eq!(plan.items[0].block_table.blocks.len(), 2);
        // The engine accepts all 5 forced tokens and row 5 yields the next one: 6 appended.
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: plan.iteration,
                finished: vec![],
                appended: vec![(SeqId(1), 6)],
                rolled_back: Vec::new(),
                failed: None,
            },
        );
        assert_eq!(s.token_limit(SeqId(1)), Some(50 - 8));
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Decode),
                (2, BatchKind::Prefill { start: 32, len: 32 })
            ]
        );
        complete_all(&mut s, &mut p, &plan, &[]);

        // Extending a prefilling, paused or unknown sequence is refused.
        s.extend(SeqId(2), vec![1, 2, 3]).unwrap_err(); // still between prefill chunks
        s.pause(SeqId(1));
        assert!(matches!(
            s.extend(SeqId(1), vec![1, 2, 3]),
            Err(SchedError::NotDecoding)
        ));
        s.resume(SeqId(1));
        assert!(matches!(
            s.extend(SeqId(99), vec![1, 2, 3]),
            Err(SchedError::NotDecoding)
        ));

        // An extension the budget cannot cover falls back to a plain decode and is dropped,
        // not retried on a later plan.
        let no_prefill = IterationLimits {
            prefill_budget_fraction: 0.0,
            ..IterationLimits::default()
        };
        s.extend(SeqId(1), vec![1, 2, 3, 4, 5]).unwrap();
        let plan = s.plan(&mut p, &no_prefill);
        assert_eq!(kinds(&plan), [(1, BatchKind::Decode)]);
        complete_all(&mut s, &mut p, &plan, &[]);
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Decode),
                (2, BatchKind::Prefill { start: 64, len: 32 })
            ],
            "the dropped extension is not retried"
        );
        complete_all(&mut s, &mut p, &plan, &[]);

        // A roll-back of 3 shortens the table by 3 tokens and frees the block it emptied.
        let mut s = Scheduler::new(params(), Arc::new(clock.clone()));
        let mut p = pool(8);
        s.submit(request(1, 1, 12, 50), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        complete_all(&mut s, &mut p, &plan, &[]);
        let plan = s.plan(&mut p, &IterationLimits::default());
        complete_all(&mut s, &mut p, &plan, &[]); // table 13 tokens, 1 block
        s.extend(SeqId(1), vec![1, 2, 3, 4, 5]).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [(1, BatchKind::Prefill { start: 13, len: 6 })]
        );
        assert_eq!(plan.items[0].block_table.blocks.len(), 2);
        assert_eq!(p.used_blocks(), 2);
        // Rows 0..3 verify (3 forced tokens kept), row 3 disagrees: the model's token is
        // taken (appended) and the 3 rejected positions are rolled back.
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: plan.iteration,
                finished: vec![],
                appended: vec![(SeqId(1), 4)],
                rolled_back: vec![(SeqId(1), 3)],
                failed: None,
            },
        );
        assert_eq!(s.token_limit(SeqId(1)), Some(50 - 6));
        assert_eq!(
            p.used_blocks(),
            1,
            "the emptied block was freed (16 tokens = 1 block of 16)"
        );
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&plan), [(1, BatchKind::Decode)]);
        assert_eq!(
            plan.items[0].block_table.tokens, 17,
            "16 kept tokens + the decode"
        );
        assert_eq!(
            plan.items[0].block_table.blocks.len(),
            2,
            "the decode refills a second block"
        );
        complete_all(&mut s, &mut p, &plan, &[]);

        // A preemption drops the pending extension: after the request is recomputed and
        // decodes again, no extension item appears.
        let mut s = Scheduler::new(params(), Arc::new(clock.clone()));
        let mut p = pool(3);
        s.submit(request(1, 1, 8, 40), p.total_blocks()).unwrap();
        s.submit(request(2, 2, 8, 40), p.total_blocks()).unwrap();
        let plan = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&plan),
            [
                (1, BatchKind::Prefill { start: 0, len: 8 }),
                (2, BatchKind::Prefill { start: 0, len: 8 }),
            ]
        );
        complete_all(&mut s, &mut p, &plan, &[]);
        s.extend(SeqId(2), vec![1, 2, 3]).unwrap();
        // Both decode until they cross their first block boundary together; the pool of 3
        // blocks cannot cover the two new blocks, so the worse-ranked request 2 is preempted.
        let mut preempted = None;
        for _ in 0..12 {
            let plan = s.plan(&mut p, &IterationLimits::default());
            if !plan.preempted.is_empty() {
                assert_eq!(plan.preempted, [(SeqId(2), PreemptReason::KvExhausted)]);
                assert_eq!(
                    kinds(&plan),
                    [(1, BatchKind::Decode)],
                    "only the survivor decodes; the pending extension went with the preemption"
                );
                preempted = Some(plan);
                break;
            }
            complete_all(&mut s, &mut p, &plan, &[]);
        }
        let plan = preempted.expect("the pool forced a preemption");
        complete_all(&mut s, &mut p, &plan, &[]);
        assert!(matches!(
            s.extend(SeqId(2), vec![1, 2, 3]),
            Err(SchedError::NotDecoding)
        ));
        // Request 2 is recomputed from scratch and decodes again — without its extension.
        for _ in 0..6 {
            let plan = s.plan(&mut p, &IterationLimits::default());
            for item in &plan.items {
                let ok = item.seq != SeqId(2)
                    || match item.kind {
                        BatchKind::Decode => true,
                        BatchKind::Prefill { start, .. } => start < 8,
                    };
                assert!(
                    ok,
                    "the preempted request's extension survived: {:?}",
                    item.kind
                );
            }
            complete_all(&mut s, &mut p, &plan, &[]);
        }
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

    /// `max_running_requests` bounds requests, and the executor is sized for that many
    /// sequences per step: with `n` > 1 the running requests hold more decodable sequences than
    /// that, so a plan takes at most `max_running_requests` of them, least recently stepped
    /// first, and every sequence keeps decoding in turn. Breaks if a plan holds more sequences
    /// than the executor was built for ("7 sequences exceed max_seqs 6") or a fork starves.
    #[test]
    fn forks_never_exceed_the_sequence_cap() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(
            SchedulerParams {
                max_running_requests: 2,
                ..params()
            },
            Arc::new(clock),
        );
        let mut p = pool(64);
        let mut a = request(1, 10, 20, 50);
        a.seqs = smallvec![SeqId(10), SeqId(11), SeqId(12)];
        s.submit(a, p.total_blocks()).unwrap();
        s.submit(request(2, 20, 8, 50), p.total_blocks()).unwrap();
        let mut steps: HashMap<u64, u32> = HashMap::new();
        for i in 0..12 {
            let plan = s.plan(&mut p, &IterationLimits::default());
            assert!(plan.items.len() <= 2, "iteration {i}: {:?}", kinds(&plan));
            for item in &plan.items {
                if item.kind == BatchKind::Decode {
                    *steps.entry(item.seq.0).or_default() += 1;
                }
            }
            let appended = plan
                .items
                .iter()
                .map(|i| i.seq)
                .chain(
                    // The shared prefill samples one token for every choice.
                    plan.items
                        .iter()
                        .filter(|i| i.seq == SeqId(10) && i.kind != BatchKind::Decode)
                        .flat_map(|_| [SeqId(11), SeqId(12)]),
                )
                .map(|q| (q, 1))
                .collect();
            s.complete(
                &mut p,
                IterationOutcome {
                    iteration: plan.iteration,
                    appended,
                    ..IterationOutcome::default()
                },
            );
        }
        let counts: Vec<u32> = [10, 11, 12, 20].iter().map(|q| steps[q]).collect();
        let (min, max) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
        assert!(max - min <= 1, "every sequence decodes in turn: {steps:?}");
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
                rolled_back: Vec::new(),
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

    fn seqs_of(plan: &IterationPlan) -> Vec<u64> {
        plan.items.iter().map(|i| i.seq.0).collect()
    }

    /// Two micro-batches (Phase 5 S-10): each plan takes half the running set and skips the
    /// requests of the plan in flight; with both in flight the plan is empty; plans complete
    /// out of order; a late finish of an in-flight sequence waits for its plan; a failed plan
    /// fails only its own requests. Catches a sequence planned twice, `complete` consuming the
    /// wrong plan, blocks freed under an executing plan, or a failure spilling over.
    #[test]
    fn micro_batches_in_flight_out_of_order() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(
            SchedulerParams {
                max_queued_requests: 8,
                ..params()
            },
            Arc::new(clock),
        )
        .with_micro_batches(2);
        let mut p = pool(64);
        for n in 1..=4 {
            s.submit(request(n, n as u64, 8, 50), p.total_blocks())
                .unwrap();
        }
        // Four startable requests over two micro-batches: two each.
        let a = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(seqs_of(&a), [1, 2]);
        let b = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(seqs_of(&b), [3, 4]);
        assert_eq!(s.micro_batches_in_flight(), 2);
        assert_eq!(s.in_flight_iterations(), [a.iteration, b.iteration]);
        let snap = s.snapshot().pipeline.expect("pipeline section");
        assert_eq!((snap.micro_batches, snap.micro_batches_in_flight), (2, 2));

        // Both micro-batches in flight: nothing more may start.
        let c = s.plan(&mut p, &IterationLimits::default());
        assert!(c.is_empty());
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: c.iteration,
                ..IterationOutcome::default()
            },
        );

        // `b` completes first: its requests decode while `a` is still in flight.
        complete_all(&mut s, &mut p, &b, &[]);
        assert_eq!(s.in_flight_iterations(), [a.iteration]);
        let d = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(
            kinds(&d),
            [(3, BatchKind::Decode), (4, BatchKind::Decode)],
            "the requests of the plan in flight are skipped"
        );
        assert_eq!(s.seq_state(SeqId(1)), Some(RequestState::Prefilling));

        // `a` completes and reports seq 3 finished late: seq 3 is in `d`, so its block stays.
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: a.iteration,
                appended: vec![(SeqId(1), 1), (SeqId(2), 1)],
                rolled_back: Vec::new(),
                finished: vec![(SeqId(3), FinishReason::Stop)],
                failed: None,
            },
        );
        assert_eq!(
            p.used_blocks(),
            4,
            "no block freed under the executing plan"
        );
        let e = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&e), [(1, BatchKind::Decode), (2, BatchKind::Decode)]);
        complete_all(&mut s, &mut p, &d, &[]);
        assert_eq!(p.used_blocks(), 3, "seq 3 finished with its plan");

        // A failed micro-batch fails only its own requests.
        let f = s.plan(&mut p, &IterationLimits::default());
        assert_eq!(kinds(&f), [(4, BatchKind::Decode)]);
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: f.iteration,
                failed: Some(IterationFailure {
                    message: "device".into(),
                }),
                ..IterationOutcome::default()
            },
        );
        assert_eq!(s.seq_state(SeqId(4)), None);
        assert_eq!(s.seq_state(SeqId(1)), Some(RequestState::Decoding));
        assert_eq!(p.used_blocks(), 2);
        complete_all(&mut s, &mut p, &e, &[1, 2]);
        assert_eq!(p.used_blocks(), 0);
        assert!(s.is_idle());
        assert_eq!(s.micro_batches_in_flight(), 0);
    }

    /// A request of a plan in flight is never a preemption victim: when the worst-ranked
    /// request is in flight, the better one's decode waits for that plan instead of being
    /// preempted, and the next plan preempts the victim. Catches blocks freed under an
    /// executing plan, or the best-ranked request preempted for lack of an eligible victim.
    #[test]
    fn micro_batches_never_preempt_in_flight() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut s = Scheduler::new(
            SchedulerParams {
                max_queued_requests: 8,
                free_watermark: 0.0,
                ..params()
            },
            Arc::new(clock),
        )
        .with_micro_batches(2);
        // Two 30-token prompts fill the 4 blocks exactly.
        let mut p = pool(4);
        let mut low = request(1, 1, 30, 30);
        low.priority = Priority(1);
        s.submit(low, p.total_blocks()).unwrap();
        s.submit(request(2, 2, 30, 30), p.total_blocks()).unwrap();
        let lim = IterationLimits::default();

        let a = s.plan(&mut p, &lim);
        assert_eq!(kinds(&a), [(2, BatchKind::Prefill { start: 0, len: 30 })]);
        let b = s.plan(&mut p, &lim);
        assert_eq!(kinds(&b), [(1, BatchKind::Prefill { start: 0, len: 30 })]);
        complete_all(&mut s, &mut p, &a, &[]);
        let c = s.plan(&mut p, &lim);
        assert_eq!(kinds(&c), [(2, BatchKind::Decode)]);
        complete_all(&mut s, &mut p, &b, &[]);
        complete_all(&mut s, &mut p, &c, &[]);
        // Least recently stepped first: seq 1, then seq 2; positions 30 and 31 need no block.
        let d = s.plan(&mut p, &lim);
        assert_eq!(kinds(&d), [(1, BatchKind::Decode)]);
        let e = s.plan(&mut p, &lim);
        assert_eq!(kinds(&e), [(2, BatchKind::Decode)]);
        complete_all(&mut s, &mut p, &d, &[]);
        complete_all(&mut s, &mut p, &e, &[]);
        let f = s.plan(&mut p, &lim);
        assert_eq!(kinds(&f), [(1, BatchKind::Decode)]);

        // Seq 2 needs a third block; the victim (request 1) is in `f`: nothing is preempted.
        let g = s.plan(&mut p, &lim);
        assert!(g.is_empty() && g.preempted.is_empty(), "{g:?}");
        assert_eq!(p.used_blocks(), 4);
        s.complete(
            &mut p,
            IterationOutcome {
                iteration: g.iteration,
                ..IterationOutcome::default()
            },
        );
        complete_all(&mut s, &mut p, &f, &[]);
        // `f` completed: request 1 is preempted and seq 2 decodes.
        let h = s.plan(&mut p, &lim);
        assert_eq!(h.preempted, [(SeqId(1), PreemptReason::KvExhausted)]);
        assert_eq!(kinds(&h), [(2, BatchKind::Decode)]);
    }

    mod queued_prefix {
        use turbine_core::config::{ByteSize, ReliabilityConfig};
        use turbine_core::types::MemoryKind;
        use turbine_kv::L0Reclaimer;
        use turbine_kv::planner::{KvPlan, PlanReason};
        use turbine_reliability::admission::{
            Admission, AdmissionParams, AdmissionQueue, Calibration,
        };
        use turbine_reliability::budget::{DeviceBudget, PoolKind};
        use turbine_reliability::controller::PressureController;
        use turbine_reliability::ledger::Ledger;
        use turbine_reliability::metrics::ReliabilityMetrics;
        use turbine_reliability::reserve::{EmergencyReserve, ReserveAllocator};
        use turbine_reliability::throttle::SchedulerLimits;

        use super::*;

        struct NoReserve;

        impl ReserveAllocator for NoReserve {
            fn allocate(&mut self, _bytes: u64) -> Result<(), String> {
                Ok(())
            }
            fn free(&mut self) {}
        }

        /// A scheduler behind an admission gate (GREEN, one running slot) whose ledger counts
        /// one byte per reserved block, so `reserved` reads the pumped estimates.
        fn gated() -> (Scheduler, Arc<Ledger>, PressureController) {
            let clock: Arc<dyn Clock> = Arc::new(FakeClock::new(Duration::ZERO));
            let kv_bytes = 1 << 20;
            let budget = DeviceBudget {
                device: DeviceId(0),
                memory_kind: MemoryKind::Dedicated,
                budget_bytes: kv_bytes,
                pools: vec![(PoolKind::Kv, kv_bytes), (PoolKind::Reserve, 0)],
            };
            let metrics = ReliabilityMetrics::unregistered();
            let ledger = Ledger::new(&budget);
            let reserve = EmergencyReserve::acquire(
                DeviceId(0),
                0,
                &ledger,
                Box::new(NoReserve),
                metrics.clone(),
            )
            .unwrap();
            let cfg = ReliabilityConfig {
                emergency_vram_reserve: ByteSize(0),
                ..ReliabilityConfig::default()
            };
            let (controller, handle) = PressureController::new(
                &cfg,
                SchedulerLimits {
                    prefill_chunk_tokens: 32,
                    block_tokens: 16,
                },
                budget,
                Arc::clone(&ledger),
                reserve,
                Arc::new(L0Reclaimer),
                metrics.clone(),
                Arc::clone(&clock),
            );
            let admission = Admission::new(
                AdmissionParams {
                    device: DeviceId(0),
                    adaptive: true,
                    max_queue: 16,
                    large_prefill_tokens: 1 << 20,
                    block_bytes: 1,
                    prefill_chunk_tokens: 32,
                    workspace_bytes_per_token: 0,
                    calibration: Calibration {
                        prefill_tokens_per_s: 1000.0,
                        decode_step_s: 0.01,
                    },
                },
                Arc::clone(&ledger),
                metrics,
            );
            let queue = AdmissionQueue::new(16, Duration::from_secs(60), 0);
            let gate = AdmissionGate::new(admission, queue, handle, Arc::clone(&clock), 1);
            let mut p = params();
            p.max_running_requests = 1;
            (Scheduler::new(p, clock).with_gate(gate), ledger, controller)
        }

        /// `blocks` attached as a cached prefix (one reference each, taken by the caller).
        fn attached(blocks: &[BlockId]) -> PrefixAttach {
            PrefixAttach {
                blocks: blocks.iter().copied().collect(),
                cached_tokens: blocks.len() as u32 * 16,
                lossy_tokens: 0,
                plan: KvPlan {
                    reuse_l0: blocks.len() as u32,
                    promote: Vec::new(),
                    recompute_tokens: 0,
                    partial_tokens: 0,
                    reason: PlanReason::AllL0,
                },
                tail_copy: None,
            }
        }

        fn with_prefix(n: u128, prompt: u32, blocks: &[BlockId]) -> SchedRequest {
            let mut r = request(n, n as u64, prompt, 8);
            r.attach_prefix(attached(blocks), 16);
            r
        }

        fn id(n: u128) -> RequestId {
            RequestId(uuid::Uuid::from_u128(n))
        }

        /// The queued-prefix release (decision "6b: queued-prefix demotion — granularity and
        /// scope", 1 A, 2 A): requests behind the admission queue's head release whole prefixes,
        /// the last to be admitted first, until the blocks only they hold reach the demand; a
        /// request holding nothing alone is skipped and the head keeps its prefix. Breaks if the
        /// walk runs head first, releases the head, counts shared blocks, overshoots the demand,
        /// or releases tail blocks only.
        #[test]
        fn released_last_queued_first_up_to_the_demand() {
            let (mut s, _ledger, _c) = gated();
            let mut p = pool(64);
            s.submit(request(1, 1, 16, 8), 64).unwrap();
            assert_eq!(kinds(&s.plan(&mut p, &IterationLimits::default())).len(), 1);
            let b = p.allocate(8).unwrap();
            // Head Q2 (2 alone), Q3 [b2 shared, b3 b4 alone], Q4 [b2 shared, b5 alone], Q5 [b6
            // shared with a running holder].
            p.incref(b[2]);
            p.incref(b[6]);
            s.submit(with_prefix(2, 40, &b[0..2]), 64).unwrap();
            s.submit(with_prefix(3, 60, &b[2..5]), 64).unwrap();
            s.submit(with_prefix(4, 40, &[b[2], b[5]]), 64).unwrap();
            s.submit(with_prefix(5, 30, &b[6..7]), 64).unwrap();
            assert_eq!(s.queued_ids(), [id(2), id(3), id(4), id(5)]);

            assert!(s.detach_queued_prefixes(&p, 0).is_empty(), "no demand");
            let first = s.detach_queued_prefixes(&p, 1);
            let got: Vec<(RequestId, usize, usize)> = first
                .iter()
                .map(|(i, a, alone)| (*i, a.blocks.len(), *alone))
                .collect();
            assert_eq!(got, [(id(4), 2, 1)], "Q5 holds nothing alone; Q4 covers 1");
            p.release(&first[0].1.blocks);
            // b2 is now Q3's alone: Q3's whole prefix (3 blocks) covers the demand of 3.
            let second = s.detach_queued_prefixes(&p, 3);
            let got: Vec<(RequestId, usize, usize)> = second
                .iter()
                .map(|(i, a, alone)| (*i, a.blocks.len(), *alone))
                .collect();
            assert_eq!(got, [(id(3), 3, 3)]);
            p.release(&second[0].1.blocks);
            assert!(
                s.detach_queued_prefixes(&p, 32).is_empty(),
                "the head keeps its prefix"
            );
        }

        /// A released request covers its whole KV again (the pump reserves the queue's copy of
        /// its estimate) and, once admitted, does not start until the engine attached its prefix
        /// again; it then prefills after the re-attached blocks, which are committed against its
        /// reservation. Breaks if the queue's estimate keeps the prefix discount, a released
        /// request starts without `reattach`, the re-attached prefix is not used, or its blocks
        /// stay reserved (counted twice with the ledger's `held`).
        #[test]
        fn released_request_waits_for_its_reattach() {
            let (mut s, ledger, _c) = gated();
            let mut p = pool(64);
            let lim = IterationLimits::default();
            s.submit(request(1, 1, 16, 8), 64).unwrap();
            let r = s.plan(&mut p, &lim);
            complete_all(&mut s, &mut p, &r, &[]);
            let b = p.allocate(2).unwrap();
            s.submit(request(2, 2, 16, 8), 64).unwrap();
            // 40 prompt + 8 new = 3 blocks; the 2 attached ones are left out until released.
            s.submit(with_prefix(3, 40, &b), 64).unwrap();
            let released = s.detach_queued_prefixes(&p, 2);
            assert_eq!(released.len(), 1);
            p.release(&released[0].1.blocks);
            assert!(
                s.awaiting_reattach().is_empty(),
                "still in the gate's queue"
            );

            // R and then Q2 finish; Q3 is pumped with a 3-block reservation.
            let r = s.plan(&mut p, &lim);
            complete_all(&mut s, &mut p, &r, &[1]);
            for _ in 0..4 {
                let r = s.plan(&mut p, &lim);
                complete_all(&mut s, &mut p, &r, &[2]);
            }
            let plan = s.plan(&mut p, &lim);
            assert!(plan.items.is_empty(), "Q3 waits for its prefix");
            assert_eq!(s.awaiting_reattach(), [id(3)]);
            assert_eq!(ledger.usage(DeviceId(0), PoolKind::Kv).reserved, 3);
            complete_all(&mut s, &mut p, &plan, &[]);

            let again = p.allocate(2).unwrap();
            assert!(
                s.reattach(id(9), attached(&again), 1).is_some(),
                "not waiting"
            );
            // While its promotions are in flight their targets are already held (here 1 of the
            // 2 blocks: a block reused from L0 needs none); the landed attach commits the rest.
            s.commit_reattached(id(3), 1, 1);
            let kv = ledger.usage(DeviceId(0), PoolKind::Kv);
            assert_eq!((kv.reserved, kv.used), (2, 1));
            assert!(s.reattach(id(3), attached(&again), 1).is_none());
            // The re-attached blocks are committed against the reservation (the ledger's `held`
            // counts them as referenced blocks; reserved would count them again).
            let kv = ledger.usage(DeviceId(0), PoolKind::Kv);
            assert_eq!((kv.reserved, kv.used), (1, 2));
            assert!(s.awaiting_reattach().is_empty());
            let plan = s.plan(&mut p, &lim);
            assert_eq!(
                kinds(&plan),
                [(3, BatchKind::Prefill { start: 32, len: 8 })],
                "prefill starts after the re-attached blocks"
            );
            assert_eq!(&plan.items[0].block_table.blocks[..2], &again[..]);
        }
    }
}
