//! The engine's iteration loop (P2 §Scheduling rules, S-3, S-7, S-8, S-9). One turn:
//!
//! 1. Take commands: block on the channel only while nothing is queued, running or waiting to
//!    be delivered (rule 4); otherwise drain it without waiting, after a short sleep when the
//!    previous plan had nothing to run (every running request paused, say).
//! 2. Retry the events held behind full output channels; a drained channel resumes its request.
//! 3. Cancel requests whose client dropped the stream (`client_disconnect`).
//! 4. `Scheduler::plan` drops cancelled requests (their blocks return first), decodes every
//!    decodable sequence and fills the token budget with prefill chunks.
//! 5. Fork copies (`n` > 1) through `ModelExecutor::copy_blocks`, then one ragged `forward` over
//!    the plan's items with the pool's `KvPoolView`.
//! 6. Every item whose step yields a token samples it with its request's sampler and emits the
//!    events with `try_send`: a full channel holds the event on the host and pauses the request
//!    (`Scheduler::pause`, `turbine_stream_paused_total`), a closed one cancels it.
//! 7. `Scheduler::complete`, then the diagnostics documents and `turbine_kv_blocks` are
//!    published.
//!
//! Overlap scheduling (P2c, `execution.overlap_scheduling`, [`EngineLoop::turn_overlap`]): when
//! the executor can launch a step without waiting for it (host staging, kernel ABI v2.3) and
//! reduces logits on the device, steps 5–7 are split around the device: the iteration launched
//! last turn is completed in the scheduler before its tokens are known (each yielding row
//! appends one token; `max_tokens` and context finishes follow from counts), the next plan is
//! launched at once — each decode whose newest token the device chose (greedy, or an unseeded
//! draw) takes it on the device ([`TokenFeed`]) — and only then is the previous iteration
//! waited for and sampled, detokenised and emitted, while the next one runs. Token-dependent
//! finishes (EOS, stop ids and strings, grammars, failures) reach the scheduler one iteration
//! late; the finished sequence's extra row is computed and dropped. A batch that needs a token
//! the device did not choose (a constraint, a seeded draw, `top_k`, a full row, a fork's first
//! token) first finishes the iteration in flight, as the serial loop would. Sampling, stop and
//! length semantics are those of the serial loop: on the CPU reference every request gets the
//! same tokens either way; on a GPU the dropped extra rows change batch compositions, whose
//! BF16 rounding may differ as with any other batch.
//!
//! Pipeline parallelism (P5 S-10, [`EngineLoop::turn_pipelined`]): with a pipeline's executor
//! the scheduler keeps up to `parallel.pipeline.micro_batches` plans in flight (disjoint
//! sequences). Each turn plans and launches micro-batches into the stages before the last until
//! the pipeline is full, then completes the oldest: its last stage runs on the engine thread
//! while the earlier stages run the next ones, and its tokens are sampled, emitted and
//! completed as in the serial loop. Overlap scheduling is off under pipelining.
//!
//! Phase 4 threads the KV orchestrator through the turn (`crate::kv_orchestrator`): prefetch
//! commands after step 1; before the plan, completed transfers admit the submissions whose
//! prefixes were promoted, submissions waiting on a prefix another request computes attach
//! again, and the reclaim order and the controller's pressure state reach the hierarchy;
//! `after_plan` follows the plan; an iteration that ran successfully commits the full blocks
//! it wrote (under overlap scheduling the blocks are held from the ahead completion until the
//! iteration is collected, and committed only when it succeeded); finished and dropped requests
//! are reported after `complete`; session TTLs run last. A new request attaches its cached
//! prefix before the scheduler sees it: `Ready` enters the admission gate at once with the
//! prefix; `Promoting` / `WaitForPrefix` hold the submission (and its admission answer) until
//! the prefix is in L0 — at most the directory's pending wait plus the copies.
//!
//! Every turn is timed in the eight stages of [`Stage`] (P2c S-1) by marks around these steps;
//! an executed iteration records them in `turbine_engine_iteration_seconds{stage}`, and every
//! published scheduler document carries the last turn's `stages_ms`.
//!
//! Reliability (P3 S-9 … S-12; CONFLICT C-25 retires the Phase 2 exit after three failed
//! iterations): each turn reads the pressure controller's snapshot once — its throttle plan
//! becomes the iteration's `IterationLimits`, the admission gate in front of the scheduler
//! decides against it — and publishes the engine's figures ([`EngineStats`]) for the
//! controller. A device out-of-memory error enters SURVIVAL (the emergency reserve is released)
//! and retries the iteration with a halved batch after a backoff, at most
//! `reliability.recovery.max_retries` times; when the retries are exhausted the batch's
//! requests fail `resource_exhausted` and the engine keeps serving. Errors are classified by
//! the execution backend that raised them (`KernelError::is_oom` / `is_sticky`, whose sticky
//! error names each registered backend declares). Any other device error fails the
//! iteration's requests with `internal_error` and opens the circuit breaker: the admission
//! queue is rejected `circuit_open`, running requests continue until they finish or
//! `reliability.circuit.drain_timeout` passes (then they fail `circuit_open`), and after the
//! cooldown internal 16-token greedy probes, which bypass the admission queue and are not
//! client requests, decide whether the circuit closes. A sticky (context-corrupting) device
//! error stops device work at once: every live request fails `resource_exhausted` and the
//! engine ends so the server exits 3. A panic inside a turn is caught, logged with the
//! iteration's request ids, fails every request with `internal_error` and is treated like a
//! sticky device error. Under overlap scheduling an out-of-memory launch first finishes the
//! iteration in flight, then runs the plan through the serial recovery path; an error when
//! collecting an iteration that the scheduler already completed ahead cannot be retried, so
//! its requests fail (`resource_exhausted` for out-of-memory, counted as a failed recovery).

use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use tokio::sync::mpsc::{self, error::TryRecvError};
use turbine_core::clock::Clock;
use turbine_core::request::{
    Endpoint, ErrorCode, FinishReason, GenerationEvent, GenerationRequest, SamplingParams,
    StopConditions,
};
use turbine_core::types::{BlockId, CircuitState, PressureState, Priority, RequestId, SeqId};
use turbine_kv::BlockPool;
use turbine_kv::hierarchy::{AttachOutcome, AttachRequest, CAPACITY_BATCH, PrefixAttach};
use turbine_model::executor::{
    BatchInput, GraphCounters, Logits, LogitsSlot, ModelExecutor, ReducedRow, RowReduce, SeqSlice,
    TokenFeed,
};
use turbine_model::{ForwardPhase, ModelError, SampleJob, SampledToken, Tokenizer, sample_rows};
use turbine_reliability::budget::PoolKind;
use turbine_reliability::circuit::{CircuitEvent, CircuitReason};
use turbine_reliability::controller::{EngineStats, Snapshot};
use turbine_reliability::recovery::RecoveryStep;
use turbine_reliability::step_window::{DecodeStepWindow, StepSample};
use turbine_scheduler::{
    BatchKind, CancelReason, IterationFailure, IterationLimits, IterationOutcome, IterationPlan,
    SchedRequest, Scheduler, SubmitError,
};

use super::deadlines::{Deadlines, Timeouts};
use super::pp::PipelineStats;
use super::requests::{ActiveRequest, Delivery, Flush, Submission};
use super::stages::{Stage, StageClock};
use super::{
    EVENT_CHANNEL_CAPACITY, EngineCommand, EngineDocs, EngineMetrics, EngineShared, SubmitAck,
};
use crate::kv_orchestrator::KvOrchestrator;
use crate::metrics::{Outcome, TokenKind};
use crate::reliability::EngineReliability;

/// Sleep between turns while requests exist but the last plan had nothing to run.
const IDLE_POLL: Duration = Duration::from_millis(2);
/// Smoothing of the decode step-time estimate and the GREEN baseline.
const STEP_ALPHA: f64 = 0.1;
/// Circuit probe (P3 S-12): a fixed prompt and 16 greedy tokens.
const PROBE_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
const PROBE_TOKENS: u32 = 16;

/// Everything the engine owns.
pub(crate) struct EngineParts {
    pub executor: Box<dyn ModelExecutor>,
    pub pool: BlockPool,
    pub scheduler: Scheduler,
    pub clock: Arc<dyn Clock>,
    pub commands: mpsc::Receiver<EngineCommand>,
    pub shared: Arc<EngineShared>,
    pub tokenizer: Arc<Tokenizer>,
    pub max_seq_len: u32,
    pub metrics: EngineMetrics,
    /// `server.request_timeout` and `server.slow_client_timeout`.
    pub timeouts: Timeouts,
    /// `execution.overlap_scheduling`: launch each iteration before the host work of the one
    /// before it (P2c), when the executor can ([`ModelExecutor::overlaps`] and a device logits
    /// reduction); otherwise the serial loop.
    pub overlap: bool,
    /// The pressure controller's snapshot, recovery and circuit inputs (P3).
    pub reliability: EngineReliability,
    /// The KV hierarchy over `pool` (Phase 4).
    pub kv: KvOrchestrator,
    /// Pipeline parallelism (P5 S-10): `executor` is a pipeline's
    /// ([`super::pp::PpExecutor`]) and the engine keeps up to its micro-batches in flight
    /// ([`EngineLoop::turn_pipelined`]); `None` otherwise.
    pub pipeline: Option<Arc<PipelineStats>>,
}

/// The message of the `slow_client` error event.
const SLOW_CLIENT_MESSAGE: &str =
    "the client did not read the stream within server.slow_client_timeout";

enum Turn {
    Continue,
    Stop,
}

pub(crate) struct EngineLoop {
    exec: Box<dyn ModelExecutor>,
    pool: BlockPool,
    sched: Scheduler,
    clock: Arc<dyn Clock>,
    commands: mpsc::Receiver<EngineCommand>,
    shared: Arc<EngineShared>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: u32,
    metrics: EngineMetrics,
    deadlines: Deadlines,
    requests: HashMap<RequestId, ActiveRequest>,
    /// Owner request and choice index of every sequence.
    seqs: HashMap<SeqId, (RequestId, usize)>,
    next_seq: u64,
    rel: EngineReliability,
    /// The controller snapshot read at the start of this turn.
    snap: Arc<Snapshot>,
    /// The circuit probe in flight and its (unread) event channel.
    probe: Option<(RequestId, mpsc::Receiver<GenerationEvent>)>,
    probe_prompt: Vec<u32>,
    /// Pure decode step times behind `step_time_drift`.
    decode_steps: DecodeStepWindow,
    /// Smoothed decode step time (s); 0 before the first decode.
    decode_step_s: f64,
    /// Smoothed decode step time of GREEN + HEALTHY iterations without a probe: what a probe's
    /// per-token latency is compared with.
    baseline_step_s: Option<f64>,
    /// Iterations executed (the circuit breaker's `Iteration` events).
    iterations: u64,
    /// When the engine first saw a fatal circuit (a controller failure drains from then on).
    fatal_since: Option<Instant>,
    /// Requests of the iteration being executed (named in the log if it panics).
    iteration_requests: Vec<RequestId>,
    shutting_down: bool,
    /// The previous plan had nothing to execute.
    idle_turn: bool,
    /// Times the current turn's stages.
    stages: StageClock,
    /// The executor's decode graph counters at the last forward
    /// (`turbine_decode_graph_total{outcome}` adds what changed since).
    graph_counters: GraphCounters,
    /// Overlap scheduling is on (see [`EngineParts::overlap`]).
    overlap: bool,
    /// Overlap scheduling: the iteration launched last turn, completed in the scheduler ahead of
    /// its tokens, whose host work is still to do.
    in_flight: Option<InFlight>,
    /// Overlap scheduling: finishes the host learned (EOS, stop strings, failures) that the
    /// scheduler gets at its next completion.
    late_finished: Vec<(SeqId, FinishReason)>,
    /// Iterations launched while the previous one was still in flight.
    overlapped: u64,
    kv: KvOrchestrator,
    /// Submissions waiting for their prefix (Phase 4): promotions in flight, or another request
    /// computing it (`attach_again`). Their admission answer is sent once they enter the
    /// scheduler.
    held: HashMap<RequestId, HeldSubmission>,
    /// Requests that ended this turn, reported to the KV hierarchy after `complete`
    /// (`true`: cancelled or failed).
    kv_done: Vec<(RequestId, bool)>,
    /// Admitted requests that released their prefix while queued and whose new attach waits for
    /// promotions ([`EngineLoop::reattach_released`]).
    reattaching: HashSet<RequestId>,
    /// Pipeline parallelism: the stages' timing and placement, and the micro-batch count.
    pipeline: Option<Arc<PipelineStats>>,
    /// Pipeline parallelism: the micro-batches launched into the stages, oldest first.
    pipe: VecDeque<PipeFlight>,
    /// Pipeline parallelism: when the last micro-batch was collected (the step cadence).
    last_collect: Option<Instant>,
}

/// A micro-batch launched into the pipeline's stages before the last, waiting for its last
/// stage (and its host work) in a later turn.
struct PipeFlight {
    plan: IterationPlan,
    built: Built,
    /// Requests of the plan, failed together if its step fails.
    requests: Vec<RequestId>,
    launched: Instant,
}

/// A submission whose cached prefix is on its way to L0.
struct HeldSubmission {
    submission: Submission,
    events: mpsc::Sender<GenerationEvent>,
    ack: SubmitAck,
    /// Another request computes the next prefix block: attach again next turn.
    attach_again: bool,
}

/// The KV one plan item wrote, committed when the iteration succeeds: request, table, tokens.
type KvCommit = (RequestId, SmallVec<[BlockId; 16]>, Vec<u32>);

/// What `sample` needs of one launched row, fixed when its batch is built (under overlap
/// scheduling the scheduler has moved on by the time the row is sampled).
#[derive(Clone, Copy, Debug)]
struct RowInfo {
    seq: SeqId,
    /// A prefill chunk (else a decode).
    prefill: bool,
    /// The step yields a token: a decode, or the chunk that completes a prefill.
    yields: bool,
    /// What the row asked the device for.
    reduce: Option<RowReduce>,
    /// The device's choice for this row is its token (greedy, or an unseeded draw over the
    /// vocabulary), so the next launch may take it on the device.
    feedable: bool,
    /// The next launch took this row's token on the device: the host's token must match it.
    fed_next: bool,
}

/// One batch built from a plan: token ids (placeholders where fed), positions, the plan item
/// and `(q_start, q_len, kv_len)` of each row, what `sample` needs per row, and the feeds.
struct Built {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    slices: Vec<(usize, u32, u32, u32)>,
    rows: Vec<RowInfo>,
    feeds: Vec<TokenFeed>,
}

enum Build {
    Ready(Built),
    /// A token is neither on the host nor chosen on the device by the launch in flight: that
    /// launch must be finished first.
    NeedsHost,
}

/// An iteration launched ahead of the host work of the one before it (overlap scheduling).
struct InFlight {
    plan: IterationPlan,
    rows: Vec<RowInfo>,
    /// Sequences that get a token once `rows` are sampled.
    yielders: HashSet<SeqId>,
    /// Requests of the plan, failed together if its step fails.
    requests: Vec<RequestId>,
    /// The forward was launched; false when nothing ran or the launch failed.
    launched: bool,
    phase: ForwardPhase,
    /// Host time of the launch call.
    launch_time: Duration,
    /// The full blocks the iteration writes (Phase 4), held (one pool reference each) from its
    /// ahead completion until it is collected: committed on success, then released.
    commits: Vec<KvCommit>,
}

impl EngineLoop {
    /// The engine over `p`; its diagnostics documents are published at once.
    pub fn new(p: EngineParts) -> EngineLoop {
        let overlap = p.overlap && p.executor.overlaps() && p.executor.reduces_logits();
        tracing::info!(
            event = "overlap_scheduling",
            enabled = overlap,
            reason = if overlap {
                "enabled"
            } else if !p.overlap {
                "disabled_by_config"
            } else if !p.executor.overlaps() {
                "kernel_library_without_host_staging"
            } else {
                "no_device_logits_reduction"
            },
            "overlap scheduling"
        );
        let probe_prompt = p
            .tokenizer
            .encode(PROBE_TEXT, true)
            .ok()
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| vec![0]);
        let snap = p.reliability.handle.snapshot();
        let mut engine = EngineLoop {
            deadlines: Deadlines::new(Arc::clone(&p.clock), p.timeouts),
            exec: p.executor,
            pool: p.pool,
            sched: p.scheduler,
            clock: p.clock,
            commands: p.commands,
            shared: p.shared,
            tokenizer: p.tokenizer,
            max_seq_len: p.max_seq_len,
            metrics: p.metrics,
            requests: HashMap::new(),
            seqs: HashMap::new(),
            next_seq: 1,
            rel: p.reliability,
            snap,
            probe: None,
            probe_prompt,
            decode_steps: DecodeStepWindow::new(),
            decode_step_s: 0.0,
            baseline_step_s: None,
            iterations: 0,
            fatal_since: None,
            iteration_requests: Vec::new(),
            shutting_down: false,
            idle_turn: false,
            stages: StageClock::start(),
            graph_counters: GraphCounters::default(),
            overlap,
            in_flight: None,
            late_finished: Vec::new(),
            overlapped: 0,
            kv: p.kv,
            held: HashMap::new(),
            kv_done: Vec::new(),
            reattaching: HashSet::new(),
            pipeline: p.pipeline,
            pipe: VecDeque::new(),
            last_collect: None,
        };
        engine.publish(false);
        engine.publish_stats();
        engine
    }

    /// Serves until the command channel closes or `Shutdown` completes. `Err` carries the
    /// reason the server must exit 3: a fatal circuit (sticky device error, controller
    /// failure) or a panic.
    pub fn run(mut self) -> Result<(), String> {
        loop {
            let turn = catch_unwind(AssertUnwindSafe(|| {
                if self.overlap {
                    self.turn_overlap()
                } else if self.pipeline.is_some() {
                    self.turn_pipelined()
                } else {
                    self.turn()
                }
            }));
            let message = match turn {
                Ok(Ok(Turn::Continue)) => continue,
                Ok(Ok(Turn::Stop)) => return Ok(()),
                Ok(Err(message)) => message,
                Err(panic) => {
                    let what = panic
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic payload".into());
                    let ids: Vec<String> = self
                        .iteration_requests
                        .iter()
                        .map(|id| id.0.to_string())
                        .collect();
                    tracing::error!(
                        event = "engine_panic",
                        request_ids = %ids.join(","),
                        panic = %what,
                        "engine thread panicked"
                    );
                    let message = format!("engine thread panicked: {what}");
                    // The device state is unknown: treat it like a sticky device error.
                    self.rel
                        .circuit_event(CircuitEvent::DeviceError { sticky: true });
                    self.fail_all(ErrorCode::InternalError, &message);
                    message
                }
            };
            return Err(message);
        }
    }

    fn turn(&mut self) -> Result<Turn, String> {
        if !self.receive_commands() {
            return Ok(Turn::Stop);
        }
        self.kv.serve_commands(&mut self.pool);
        self.stages.mark(Stage::Schedule);
        self.flush_outputs();
        self.stages.mark(Stage::Emit);
        self.detect_disconnects();
        self.expire_deadlines();

        let limits = self.read_snapshot()?;
        self.kv_before_plan();
        let mut plan = self.sched.plan(&mut self.pool, &limits);
        self.kv.after_plan(&mut self.pool);
        for &(id, reason) in &plan.dropped {
            self.on_dropped(id, reason);
        }
        self.stages.mark(Stage::Schedule);
        if plan.is_empty() {
            self.sched.complete(
                &mut self.pool,
                IterationOutcome {
                    iteration: plan.iteration,
                    ..IterationOutcome::default()
                },
            );
            self.kv_end_turn();
            self.idle_turn = true;
            self.publish(false);
            self.publish_stats();
            return Ok(Turn::Continue);
        }
        self.idle_turn = false;
        self.iteration_requests = plan_requests(&plan, &self.seqs);
        let started = Instant::now();
        let outcome = self.execute(&mut plan);
        let failed = outcome.failed.is_some();
        if !failed {
            // Only KV the forward pass wrote successfully enters the cache; the tables are
            // still held (before `complete` releases finished sequences). A recovery retry may
            // have shrunk `plan`: what ran is what it holds now.
            for (id, blocks, tokens) in self.kv_commits(&plan) {
                self.kv.commit(&mut self.pool, id, &blocks, &tokens);
            }
        }
        self.sched.complete(&mut self.pool, outcome);
        if !failed {
            self.observe(&plan, started.elapsed().as_secs_f64());
        }
        self.kv_end_turn();
        self.publish(true);
        self.publish_stats();
        self.check_fatal()?;
        Ok(Turn::Continue)
    }

    /// One turn with pipeline micro-batches (P5 S-10): the same steps as [`EngineLoop::turn`],
    /// but while fewer than `parallel.pipeline.micro_batches` micro-batches are in the pipeline
    /// the scheduler plans another (its plans in flight hold disjoint sequences, so a sequence
    /// enters stage 0 again only after its token was sampled) and it is launched into the
    /// stages before the last ([`EngineLoop::launch_pipe`]); then the oldest micro-batch is
    /// completed: its last stage runs on this thread, its tokens are sampled and emitted, its
    /// KV committed and the scheduler completes it ([`EngineLoop::collect_pipe`]), while the
    /// earlier stages already run the next ones. A failed micro-batch fails only its own
    /// requests; a stage failure fails every micro-batch in the pipeline (`replica_failed`).
    fn turn_pipelined(&mut self) -> Result<Turn, String> {
        if !self.receive_commands() {
            return Ok(Turn::Stop);
        }
        self.kv.serve_commands(&mut self.pool);
        self.stages.mark(Stage::Schedule);
        self.flush_outputs();
        self.stages.mark(Stage::Emit);
        self.detect_disconnects();
        self.expire_deadlines();

        let limits = self.read_snapshot()?;
        self.kv_before_plan();
        let micro_batches = self.pipeline.as_ref().map_or(1, |p| p.micro_batches) as usize;
        while self.pipe.len() < micro_batches {
            let plan = self.sched.plan(&mut self.pool, &limits);
            self.kv.after_plan(&mut self.pool);
            for &(id, reason) in &plan.dropped {
                self.on_dropped(id, reason);
            }
            if plan.is_empty() {
                self.sched.complete(
                    &mut self.pool,
                    IterationOutcome {
                        iteration: plan.iteration,
                        ..IterationOutcome::default()
                    },
                );
                break;
            }
            self.stages.mark(Stage::Schedule);
            self.launch_pipe(plan);
            self.stages.mark(Stage::Launch);
        }
        let executed = match self.pipe.pop_front() {
            Some(flight) => {
                self.collect_pipe(flight);
                true
            }
            None => false,
        };
        self.idle_turn = !executed;
        self.iteration_requests = self
            .pipe
            .iter()
            .flat_map(|f| f.requests.iter().copied())
            .collect();
        self.kv_end_turn();
        self.publish(executed);
        self.publish_stats();
        self.check_fatal()?;
        Ok(Turn::Continue)
    }

    /// Fork copies, the batch and its launch into the pipeline's stages before the last; a
    /// plan with forks only completes at once, a failed launch fails the plan's requests.
    fn launch_pipe(&mut self, plan: IterationPlan) {
        let requests = plan_requests(&plan, &self.seqs);
        self.iteration_requests = requests.clone();
        let launched = Instant::now();
        match self.try_launch(&plan) {
            Ok(Some(built)) => self.pipe.push_back(PipeFlight {
                plan,
                built,
                requests,
                launched,
            }),
            Ok(None) => self.sched.complete(
                &mut self.pool,
                IterationOutcome {
                    iteration: plan.iteration,
                    ..IterationOutcome::default()
                },
            ),
            Err(error) => {
                let outcome = self.pipe_failure(plan.iteration, error);
                self.sched.complete(&mut self.pool, outcome);
            }
        }
    }

    /// [`EngineLoop::launch_pipe`]'s device work: `None` when the plan has only forks.
    fn try_launch(&mut self, plan: &IterationPlan) -> Result<Option<Built>, IterationError> {
        self.fork_copies(plan)?;
        self.stages.mark(Stage::Prepare);
        if plan.items.is_empty() {
            return Ok(None);
        }
        let built = match self
            .build_batch(plan, None)
            .map_err(IterationError::other)?
        {
            Build::Ready(built) => built,
            Build::NeedsHost => {
                return Err(IterationError::other(
                    "a batch token is not on the host".into(),
                ));
            }
        };
        let slices = seq_slices(plan, &built);
        let view = self.pool.view();
        self.exec
            .launch(
                &BatchInput {
                    tokens: &built.tokens,
                    positions: &built.positions,
                    seqs: &slices,
                    kv: &view,
                },
                &[],
            )
            .map_err(|e| IterationError::model("pipeline launch failed", e))?;
        Ok(Some(built))
    }

    /// Completes the oldest micro-batch (see [`EngineLoop::turn_pipelined`]).
    fn collect_pipe(&mut self, f: PipeFlight) {
        self.iteration_requests = f.requests.clone();
        let phase = phase_of(&f.plan);
        let result = {
            let slices = seq_slices(&f.plan, &f.built);
            let view = self.pool.view();
            self.exec.forward(&BatchInput {
                tokens: &f.built.tokens,
                positions: &f.built.positions,
                seqs: &slices,
                kv: &view,
            })
        };
        self.stages.mark(Stage::DeviceWait);
        let now = Instant::now();
        // The pipeline completes one micro-batch per cadence: the step time the estimates see.
        let cadence = self
            .last_collect
            .map_or(now - f.launched, |t| (now - t).min(now - f.launched));
        self.last_collect = Some(now);
        self.metrics
            .model
            .observe_forward(phase, (now - f.launched).as_secs_f64());
        let logits = result
            .map_err(|e| {
                IterationError::model(&format!("{} forward pass failed", phase.as_str()), e)
            })
            .and_then(|logits| {
                self.check_logits(&logits, f.built.rows.len())
                    .map(|()| logits)
                    .map_err(IterationError::other)
            });
        match logits {
            Ok(logits) => {
                let mut outcome = IterationOutcome {
                    iteration: f.plan.iteration,
                    ..IterationOutcome::default()
                };
                self.sample(&f.built.rows, logits, &mut outcome);
                self.rel.recovery_succeeded();
                for (id, blocks, tokens) in self.kv_commits(&f.plan) {
                    self.kv.commit(&mut self.pool, id, &blocks, &tokens);
                }
                self.sched.complete(&mut self.pool, outcome);
                self.observe(&f.plan, cadence.as_secs_f64());
            }
            Err(error) => {
                let outcome = self.pipe_failure(f.plan.iteration, error);
                self.sched.complete(&mut self.pool, outcome);
            }
        }
    }

    /// A micro-batch failed: its requests (`iteration_requests`) end — a sticky device error
    /// turns the circuit fatal (exit 3), out of memory is `resource_exhausted`, a failed stage
    /// or collective `replica_failed` with the circuit open (`collective_failed`).
    fn pipe_failure(&mut self, iteration: u64, error: IterationError) -> IterationOutcome {
        let mut outcome = IterationOutcome {
            iteration,
            ..IterationOutcome::default()
        };
        if error.sticky {
            tracing::error!(event = "iteration_failed", reason = "device_fatal", iteration, error = %error.message, "sticky device error");
            self.rel
                .circuit_event(CircuitEvent::DeviceError { sticky: true });
            outcome.failed = Some(IterationFailure {
                message: error.message,
            });
        } else if error.oom {
            self.rel.on_oom();
            self.rel.recovery_failed();
            self.fail_iteration(ErrorCode::ResourceExhausted, error.message, &mut outcome);
        } else {
            let (code, event) = error.failure();
            // The circuit opens before the requests hear of the failure, so a client that
            // reads its error already finds `/ready` at 503 `circuit_open`.
            self.rel.circuit_event(event);
            self.fail_iteration(code, error.message, &mut outcome);
        }
        outcome
    }

    /// Reads the controller's snapshot for this turn (P3): a fatal circuit ends the engine, an
    /// expired drain fails the running requests, PROBING keeps a probe in flight, and the
    /// throttle plan becomes the iteration's limits.
    fn read_snapshot(&mut self) -> Result<IterationLimits, String> {
        let before = self.snap.circuit;
        self.snap = self.rel.handle.snapshot();
        let snap = Arc::clone(&self.snap);
        // Back from PROBING: the device may have changed, relearn the drift baselines (S-12).
        if before == CircuitState::Probing && snap.circuit == CircuitState::Healthy {
            self.decode_steps.reset();
        }
        if snap.fatal {
            self.fatal(&snap)?;
        }
        if snap.drain_expired {
            self.fail_running(
                ErrorCode::CircuitOpen,
                "the circuit breaker is open and reliability.circuit.drain_timeout has passed",
            );
        }
        self.maybe_probe(snap.circuit);
        Ok(IterationLimits::from(&snap.throttle))
    }

    /// After an executed iteration: a circuit that turned fatal ends the engine.
    fn check_fatal(&mut self) -> Result<(), String> {
        let snap = self.rel.handle.snapshot();
        if snap.fatal {
            self.fatal(&snap)?;
        }
        Ok(())
    }

    /// The circuit is fatal. A sticky device error (or a panic) stops device work at once:
    /// every live request fails `resource_exhausted` and `Err` ends the engine (exit 3). A
    /// controller failure first lets the running requests finish, up to `drain_timeout`
    /// (admission already refuses everything with `circuit_open`).
    fn fatal(&mut self, snap: &Snapshot) -> Result<(), String> {
        let reason = snap.document.circuit.last_reason;
        if reason == Some(CircuitReason::ControllerFailed) {
            let since = *self.fatal_since.get_or_insert_with(Instant::now);
            if !self.sched.running_ids().is_empty() && since.elapsed() < self.rel.drain_timeout {
                return Ok(());
            }
        }
        let reason = reason.map_or("device_fatal", CircuitReason::as_str);
        let message = format!("circuit open ({reason}): no further device work; exiting");
        tracing::error!(
            event = "circuit_transition",
            reason,
            "fatal circuit: failing every request and exiting"
        );
        self.fail_all(ErrorCode::ResourceExhausted, &message);
        Err(message)
    }

    /// Fails every running request with `code` (the circuit's drain timed out).
    fn fail_running(&mut self, code: ErrorCode, message: &str) {
        let running = self.sched.running_ids();
        let failed = self.sched.fail_requests(&mut self.pool, &running);
        self.fail_requests_with(&failed, code, message);
    }

    /// `ids` end with an error event carrying `code`, counted as failed.
    fn fail_requests_with(&mut self, ids: &[RequestId], code: ErrorCode, message: &str) {
        for &id in ids {
            if self.requests.get(&id).is_some_and(|r| !r.done) {
                self.account(id, Outcome::Failed, message);
                self.deliver(id, ActiveRequest::error_event(code, message));
                self.retire(id);
            }
        }
    }

    /// While the circuit is PROBING, keeps one internal probe in flight: a fixed prompt and
    /// [`PROBE_TOKENS`] greedy tokens, bypassing the admission queue (with its worst-case KV
    /// reservation), never counted as a client request. Its outcome goes to the circuit
    /// breaker when it ends ([`EngineLoop::account`]).
    fn maybe_probe(&mut self, circuit: CircuitState) {
        if circuit != CircuitState::Probing || self.probe.is_some() || self.shutting_down {
            return;
        }
        let id = RequestId::new_v4();
        let seq = SeqId(self.next_seq);
        self.next_seq += 1;
        let prompt_len = u32::try_from(self.probe_prompt.len()).unwrap_or(u32::MAX);
        let mut r = SchedRequest::new(
            id,
            smallvec::smallvec![seq],
            prompt_len,
            PROBE_TOKENS,
            self.pool.layout().block_tokens,
        );
        r.arrival = self.clock.now_mono();
        self.sync_kv_held();
        if let Err(e) = self.sched.submit_probe(r, self.pool.total_blocks()) {
            tracing::warn!(
                event = "circuit_probe",
                reason = e.as_str(),
                "circuit probe refused"
            );
            self.rel.circuit_event(CircuitEvent::ProbeFailed);
            return;
        }
        let request = GenerationRequest {
            id,
            n: 1,
            priority: Priority::default(),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            session: None,
            cache_salt: None,
            kv_policy: None,
            endpoint: Endpoint::Completions,
            http_request_id: "circuit-probe".into(),
            prompt_tokens: self.probe_prompt.clone(),
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: StopConditions {
                max_tokens: PROBE_TOKENS,
                ignore_eos: true,
                ..StopConditions::default()
            },
        };
        let (events, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let active = ActiveRequest::new(request.into(), events, &[seq], &self.tokenizer);
        self.seqs.insert(seq, (id, 0));
        self.requests.insert(id, active);
        self.deadlines.track(id, u64::MAX);
        self.probe = Some((id, rx));
        tracing::info!(event = "circuit_probe", request_id = %id.0, "circuit probe started");
    }

    /// One executed iteration of `secs`: the decode step window and estimates, and the
    /// admission throughput EWMAs.
    fn observe(&mut self, plan: &IterationPlan, secs: f64) {
        self.iterations += 1;
        let prefill = plan.prefill_tokens();
        let decodes = plan.decode_tokens();
        let calm = self.snap.state == PressureState::Green
            && self.snap.circuit == CircuitState::Healthy
            && self.probe.is_none();
        let context_tokens = plan.decode_context_tokens();
        let judged = self.decode_steps.observe(
            StepSample {
                prefill_tokens: prefill,
                rows: decodes,
                context_tokens,
                secs,
            },
            calm,
        );
        if prefill == 0 && decodes > 0 {
            tracing::debug!(
                event = "decode_step",
                rows = decodes,
                context_tokens,
                secs,
                calm,
                ratio = judged,
                "decode step against its shape bucket's calm baseline"
            );
        }
        if decodes > 0 {
            self.decode_step_s = if self.decode_step_s > 0.0 {
                STEP_ALPHA * secs + (1.0 - STEP_ALPHA) * self.decode_step_s
            } else {
                secs
            };
            if self.snap.state == PressureState::Green
                && self.snap.circuit == CircuitState::Healthy
                && self.probe.is_none()
            {
                self.baseline_step_s = Some(
                    self.baseline_step_s
                        .map_or(secs, |b| STEP_ALPHA * secs + (1.0 - STEP_ALPHA) * b),
                );
            }
        }
        // A mixed batch's time is split between prefill and decode by token count.
        let prefill_s = if prefill > 0 {
            secs * f64::from(prefill) / f64::from(prefill + decodes)
        } else {
            0.0
        };
        if let Some(g) = self.sched.gate_mut() {
            g.admission_mut()
                .observe_iteration(prefill, prefill_s, (decodes > 0).then_some(secs));
        }
        // The planner's recompute cost follows the same prefill rate (P4 S-9).
        self.kv.record_prefill(prefill, prefill_s);
    }

    /// Reports the L0 blocks requests hold to the ledger's `kv` pool (P6b): a request's KV
    /// reservation leaves out the cached prefix blocks it attached (P4 S-3), so without this the
    /// ledger missed every block a request took from the cache. Admission and `kv_utilization`
    /// then count them; called before each submission and plan and with the stats.
    fn sync_kv_held(&self) {
        if let Some((ledger, device)) = self.pool.ledger() {
            let bytes = u64::from(self.pool.referenced_blocks())
                .saturating_mul(self.pool.layout().block_bytes());
            ledger.set_held(*device, PoolKind::Kv, bytes);
        }
    }

    /// The figures the pressure controller reads on its next tick.
    fn publish_stats(&self) {
        self.sync_kv_held();
        let p95 = self.decode_steps.p95();
        self.rel.publish(EngineStats {
            running_remaining_tokens: self.sched.remaining_tokens(),
            // Phase 4: cached-but-unreferenced L0 blocks (finished prompts kept for prefix reuse)
            // are handed out by the next allocation, so the exhaustion horizon counts them as
            // free, as `kv_utilization` and `device_memory` do (no reservation covers them).
            free_kv_blocks: self.pool.available_blocks(),
            block_tokens: self.pool.layout().block_tokens,
            decode_tokens_per_s: if self.decode_step_s > 0.0 {
                1.0 / self.decode_step_s
            } else {
                0.0
            },
            step_time_p95: p95,
            queue_len: self
                .sched
                .gate()
                .map_or(0, |g| u32::try_from(g.queue_len()).unwrap_or(u32::MAX)),
            iterations: self.iterations,
        });
    }

    /// Nothing queued, running, in flight or waiting to be delivered, and no circuit probe to
    /// start.
    fn quiet(&self) -> bool {
        let probe_due = !self.shutting_down
            && self.probe.is_none()
            && self.rel.handle.circuit() == CircuitState::Probing;
        self.sched.is_idle()
            && self.requests.is_empty()
            && self.in_flight.is_none()
            && self.pipe.is_empty()
            && self.held.is_empty()
            && self.kv.transfers_idle()
            && !probe_due
    }

    /// Step 1. Returns false when the engine should stop. The turn's stage clock starts after
    /// the wait for work.
    fn receive_commands(&mut self) -> bool {
        if self.quiet() {
            if self.shutting_down {
                return false;
            }
            let Some(cmd) = self.commands.blocking_recv() else {
                return false;
            };
            self.stages = StageClock::start();
            self.command(cmd);
        } else {
            if self.idle_turn {
                std::thread::sleep(IDLE_POLL);
            }
            self.stages = StageClock::start();
        }
        loop {
            match self.commands.try_recv() {
                Ok(cmd) => self.command(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // The HTTP side is gone: finish what is running as in a shutdown.
                    if !self.shutting_down {
                        self.begin_shutdown();
                    }
                    break;
                }
            }
        }
        true
    }

    fn command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Submit(request, events, ack) => self.submit(*request, events, ack),
            EngineCommand::Shutdown => self.begin_shutdown(),
            // The pressure controller changed the circuit: the turn reads the new snapshot.
            EngineCommand::Wake => {}
        }
    }

    fn begin_shutdown(&mut self) {
        self.shutting_down = true;
        self.sched.begin_shutdown();
        let live: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| !r.done)
            .map(|(id, _)| *id)
            .collect();
        for id in live {
            self.sched.cancel(id, CancelReason::Shutdown);
        }
        // Held submissions were never admitted: they are refused like any new one.
        for (id, h) in std::mem::take(&mut self.held) {
            self.kv.request_done(&mut self.pool, id, true);
            let _ = h.ack.send(Err(SubmitError::ShuttingDown));
        }
        tracing::info!(event = "engine_shutdown", "engine shutting down");
    }

    /// A new request attaches its cached prefix (P4 S-3), then goes through the scheduler's
    /// checks and admission ([`EngineLoop::admit_submission`]); one whose prefix is still on its
    /// way to L0 is held, its admission answer pending, until it is there.
    fn submit(
        &mut self,
        submission: Submission,
        events: mpsc::Sender<GenerationEvent>,
        ack: SubmitAck,
    ) {
        let request = &submission.request;
        let prompt_len = u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX);
        let zero_tokens = request.stop.max_tokens == 0 && prompt_len <= self.max_seq_len;
        if zero_tokens || self.shutting_down {
            self.admit_submission(submission, events, ack, None);
            return;
        }
        let id = request.id;
        match self.attach(id, &submission.request) {
            AttachOutcome::Ready(a) => self.admit_submission(submission, events, ack, Some(a)),
            outcome => {
                let held = HeldSubmission {
                    submission,
                    events,
                    ack,
                    attach_again: outcome == AttachOutcome::WaitForPrefix,
                };
                self.held.insert(id, held);
            }
        }
    }

    /// `KvHierarchy::attach_prefix` for `request` (P4 S-3).
    fn attach(&mut self, id: RequestId, request: &GenerationRequest) -> AttachOutcome {
        self.kv.attach(
            &mut self.pool,
            &AttachRequest {
                request: id,
                prompt: &request.prompt_tokens,
                cache_salt: request.cache_salt.as_deref().unwrap_or(""),
                session: request.session.as_ref(),
                priority: request.priority,
                allow_lossy: request.kv_policy.map(|p| p.allow_lossy),
            },
        )
    }

    /// Scheduler submission checks (with the attached prefix, whose blocks the admission
    /// estimate and the KV reservation leave out), then the request is queued and its choices
    /// start. A refused request gives its prefix back.
    fn admit_submission(
        &mut self,
        submission: Submission,
        events: mpsc::Sender<GenerationEvent>,
        ack: SubmitAck,
        attach: Option<PrefixAttach>,
    ) {
        let request = &submission.request;
        let id = request.id;
        let prompt_len = u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX);
        let seqs: SmallVec<[SeqId; 1]> = (0..request.n.max(1))
            .map(|_| {
                let seq = SeqId(self.next_seq);
                self.next_seq += 1;
                seq
            })
            .collect();
        // `max_tokens: 0` (explicit, or the default for a prompt that fills the context) has
        // nothing to generate; a prompt beyond the context still goes to the scheduler's checks.
        let zero_tokens = request.stop.max_tokens == 0 && prompt_len <= self.max_seq_len;
        if zero_tokens && self.shutting_down {
            let _ = ack.send(Err(SubmitError::ShuttingDown));
            return;
        }
        let cached_tokens = attach.as_ref().map_or(0, |a| a.cached_tokens);
        let lossy_cached_tokens = attach.as_ref().map_or(0, |a| a.lossy_tokens);
        if !zero_tokens {
            let bt = self.pool.layout().block_tokens;
            let mut r =
                SchedRequest::new(id, seqs.clone(), prompt_len, request.stop.max_tokens, bt);
            r.priority = request.priority;
            r.arrival = self.clock.now_mono();
            r.constrained = request.constraint.is_some();
            let blocks = attach.as_ref().map(|a| a.blocks.clone());
            if let Some(a) = attach {
                r.attach_prefix(a, bt);
            }
            self.sync_kv_held();
            if let Err(e) = self.sched.submit(r, self.pool.total_blocks()) {
                if let Some(blocks) = blocks {
                    self.pool.release(&blocks);
                }
                self.kv.request_done(&mut self.pool, id, true);
                let _ = ack.send(Err(e));
                return;
            }
        } else if let Some(a) = attach {
            self.pool.release(&a.blocks);
            self.kv.request_done(&mut self.pool, id, true);
        }
        let submitter_gone = ack.send(Ok(())).is_err();
        self.metrics
            .server
            .add_tokens(TokenKind::Prompt, u64::from(prompt_len));
        self.shared.add_outstanding(outstanding(
            prompt_len,
            submission.request.n,
            submission.request.stop.max_tokens,
            0,
        ));
        let mut active = ActiveRequest::new(submission, events, &seqs, &self.tokenizer);
        active.cached_tokens = cached_tokens;
        active.lossy_cached_tokens = lossy_cached_tokens;
        for (i, &seq) in seqs.iter().enumerate() {
            self.seqs.insert(seq, (id, i));
        }
        self.requests.insert(id, active);
        self.deadlines
            .track(id, self.requests[&id].request.deadline_ms);
        for choice in 0..seqs.len() as u32 {
            self.deliver(id, GenerationEvent::Started { choice });
        }
        if zero_tokens {
            // `max_tokens: 0`: nothing to generate, the request never enters the scheduler.
            let finished: Vec<GenerationEvent> = {
                let r = &self.requests[&id];
                (0..r.choices.len())
                    .map(|c| r.finished_event(c, FinishReason::Length))
                    .collect()
            };
            if let Some(r) = self.requests.get_mut(&id) {
                for c in &mut r.choices {
                    c.finish = Some(FinishReason::Length);
                }
            }
            self.account(id, Outcome::Ok, "");
            for event in finished {
                self.deliver(id, event);
            }
            self.retire(id);
        } else if submitter_gone {
            self.sched.cancel(id, CancelReason::ClientDisconnect);
        }
    }

    /// Before the plan (Phase 4): held submissions whose promotions landed are admitted, those
    /// waiting on another request's prefix attach again, one whose client went away is dropped
    /// (its KV released), then the reclaim order and the controller's pressure state.
    fn kv_before_plan(&mut self) {
        for (id, attach) in self.kv.poll(&mut self.pool) {
            if self.reattaching.remove(&id) {
                self.reattached(id, attach);
                continue;
            }
            match self.held.remove(&id) {
                Some(h) => self.admit_submission(h.submission, h.events, h.ack, Some(attach)),
                None => {
                    // Its submission is gone: the landed prefix goes back to the cache.
                    self.pool.release(&attach.blocks);
                    self.kv.request_done(&mut self.pool, id, true);
                }
            }
        }
        let gone: Vec<RequestId> = self
            .held
            .iter()
            .filter(|(_, h)| h.events.is_closed() || h.ack.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in gone {
            self.held.remove(&id);
            self.kv.request_done(&mut self.pool, id, true);
        }
        let again: Vec<RequestId> = self
            .held
            .iter()
            .filter(|(_, h)| h.attach_again)
            .map(|(id, _)| *id)
            .collect();
        for id in again {
            let Some(h) = self.held.remove(&id) else {
                continue;
            };
            match self.attach(id, &h.submission.request) {
                AttachOutcome::Ready(a) => {
                    self.admit_submission(h.submission, h.events, h.ack, Some(a))
                }
                outcome => {
                    let attach_again = outcome == AttachOutcome::WaitForPrefix;
                    self.held.insert(id, HeldSubmission { attach_again, ..h });
                }
            }
        }
        self.reattach_released();
        self.kv.before_plan(&mut self.pool, self.snap.state);
        // The plan's refills from the admission queue reserve against what the pool holds now.
        self.sync_kv_held();
    }

    /// Admitted requests that released their prefix in the admission queue
    /// ([`EngineLoop::release_queued_prefixes`]) attach it again through the planner before
    /// they start: resident blocks now, promotions when they land, or a recompute; one waiting on
    /// another request's block tries again next turn. Their KV reservation stays whole.
    fn reattach_released(&mut self) {
        for id in self.sched.awaiting_reattach() {
            if self.reattaching.contains(&id) {
                continue;
            }
            let Some(r) = self.requests.get(&id) else {
                continue;
            };
            let request = &r.request;
            let outcome = self.kv.attach_again(
                &mut self.pool,
                &AttachRequest {
                    request: id,
                    prompt: &request.prompt_tokens,
                    cache_salt: request.cache_salt.as_deref().unwrap_or(""),
                    session: request.session.as_ref(),
                    priority: request.priority,
                    allow_lossy: request.kv_policy.map(|p| p.allow_lossy),
                },
            );
            match outcome {
                AttachOutcome::Ready(a) => self.reattached(id, a),
                AttachOutcome::Promoting => {
                    // The promotion targets are referenced now: commit them against the
                    // reservation, or the ledger counts them twice until they land.
                    let blocks = self.kv.pending_blocks(id);
                    let block_bytes = self.pool.layout().block_bytes();
                    self.sched.commit_reattached(id, blocks, block_bytes);
                    self.reattaching.insert(id);
                }
                AttachOutcome::WaitForPrefix => {}
            }
        }
    }

    /// A released request's prefix is attached again: the scheduler may start it, and its
    /// `usage.cached_tokens` are the new attach's. One the scheduler no longer waits for
    /// (cancelled meanwhile) gives the blocks back.
    fn reattached(&mut self, id: RequestId, attach: PrefixAttach) {
        let (cached, lossy) = (attach.cached_tokens, attach.lossy_tokens);
        let block_bytes = self.pool.layout().block_bytes();
        match self.sched.reattach(id, attach, block_bytes) {
            None => {
                if let Some(r) = self.requests.get_mut(&id) {
                    r.cached_tokens = cached;
                    r.lossy_cached_tokens = lossy;
                }
            }
            Some(attach) => self.pool.release(&attach.blocks),
        }
    }

    /// At YELLOW and ORANGE the pressure reclaim may want more blocks than L0 holds
    /// unreferenced (decision "6b: after the held-prefix ledger fix", 1 B): requests waiting in
    /// the admission queue then release their whole prefixes, the last to be admitted first, up
    /// to that shortfall and at most [`CAPACITY_BATCH`] blocks per controller tick (decision "6b:
    /// queued-prefix demotion — granularity and scope", 1 A, 2 A). The released blocks leave the
    /// ledger's `held` now and the next reclaim demotes them; each request attaches again once
    /// admitted ([`EngineLoop::reattach_released`]).
    fn release_queued_prefixes(&mut self) {
        let want = self.kv.take_queued_prefix_demand().min(CAPACITY_BATCH);
        if want == 0 {
            return;
        }
        let released = self.sched.detach_queued_prefixes(&self.pool, want);
        if released.is_empty() {
            return;
        }
        for (id, attach, alone) in released {
            self.kv.detach_prefix(&mut self.pool, id, &attach, alone);
        }
        self.sync_kv_held();
    }

    /// The full blocks each item of `plan` completes, with their tokens (choice 0 of a request
    /// holds its prompt blocks; forks share them). Every token whose KV the iteration writes is
    /// on the host by the time its iteration is committed.
    fn kv_commits(&self, plan: &IterationPlan) -> Vec<KvCommit> {
        let bt = self.pool.layout().block_tokens.max(1);
        let mut out = Vec::new();
        for item in &plan.items {
            let Some(&(id, 0)) = self.seqs.get(&item.seq) else {
                continue;
            };
            let end = item.block_table.tokens;
            let start = match item.kind {
                BatchKind::Prefill { start, .. } => start,
                BatchKind::Decode => end.saturating_sub(1),
            };
            if end / bt <= start / bt {
                continue;
            }
            let Some(r) = self.requests.get(&id) else {
                continue;
            };
            let tokens: Option<Vec<u32>> = (0..end).map(|p| r.token_at(0, p)).collect();
            if let Some(tokens) = tokens {
                out.push((id, item.block_table.blocks.clone(), tokens));
            }
        }
        out
    }

    /// After `complete`: the requests that ended this turn, then pending reclaim requests and
    /// session TTLs.
    fn kv_end_turn(&mut self) {
        for (id, cancelled) in std::mem::take(&mut self.kv_done) {
            self.reattaching.remove(&id);
            self.kv.request_done(&mut self.pool, id, cancelled);
        }
        self.kv.end_turn(&mut self.pool);
        self.release_queued_prefixes();
    }

    /// Step 2.
    fn flush_outputs(&mut self) {
        let held: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| r.has_held())
            .map(|(id, _)| *id)
            .collect();
        for id in held {
            let Some(r) = self.requests.get_mut(&id) else {
                continue;
            };
            match r.flush() {
                Flush::Held => {}
                Flush::Drained if r.done => self.forget(id),
                Flush::Drained => {
                    self.deadlines.resumed(id);
                    let seqs: Vec<SeqId> = r.live_seqs().collect();
                    for seq in seqs {
                        self.sched.resume(seq);
                    }
                    tracing::debug!(event = "resume", request_id = %id.0, "output channel drained");
                }
                Flush::Closed => self.client_gone(id),
            }
        }
    }

    /// Step 3.
    fn detect_disconnects(&mut self) {
        let closed: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| r.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in closed {
            self.client_gone(id);
        }
    }

    /// Request deadlines and slow-client timers (P2 S-7, S-8): a live request is cancelled
    /// (the next plan frees its blocks and sends the error event); a finished one whose final
    /// events stay unread past `server.slow_client_timeout` is dropped, which closes its stream
    /// with the `slow_client` error event in the slot reserved for it.
    fn expire_deadlines(&mut self) {
        for (id, reason) in self.deadlines.expired() {
            match self.requests.get(&id) {
                Some(r) if r.done => {
                    if reason == CancelReason::SlowClient {
                        tracing::info!(event = "cancel", request_id = %id.0, reason = reason.as_str(), "closing a stream whose final events stay unread");
                        if let Some(r) = self.requests.get_mut(&id) {
                            r.close_with(ActiveRequest::error_event(
                                ErrorCode::SlowClient,
                                SLOW_CLIENT_MESSAGE,
                            ));
                        }
                        self.forget(id);
                    }
                }
                Some(_) => self.sched.cancel(id, reason),
                None => {}
            }
        }
    }

    /// The client dropped the stream: a live request is cancelled (the next plan frees its
    /// blocks), a finished one is forgotten.
    fn client_gone(&mut self, id: RequestId) {
        match self.requests.get(&id) {
            Some(r) if r.done => self.forget(id),
            Some(_) => self.sched.cancel(id, CancelReason::ClientDisconnect),
            None => {}
        }
    }

    /// A request the plan dropped (rule 1): its blocks are already free.
    fn on_dropped(&mut self, id: RequestId, reason: CancelReason) {
        if self.requests.get(&id).is_none_or(|r| r.done) {
            return;
        }
        let (outcome, error) = match reason {
            CancelReason::ClientDisconnect => (Outcome::Cancelled, None),
            CancelReason::QueueTimeout => (
                Outcome::Rejected,
                Some((
                    ErrorCode::QueueTimeout,
                    "the request waited longer than reliability.admission.queue_timeout",
                )),
            ),
            CancelReason::CircuitOpen => (
                Outcome::Rejected,
                Some((
                    ErrorCode::CircuitOpen,
                    "the circuit breaker opened while the request was queued",
                )),
            ),
            CancelReason::Overloaded => (
                Outcome::Rejected,
                Some((
                    ErrorCode::Overloaded,
                    "the engine entered SURVIVAL before the request started and the admission \
                     queue is full",
                )),
            ),
            CancelReason::RequestTimeout => (
                Outcome::Cancelled,
                Some((
                    ErrorCode::RequestTimeout,
                    "the request exceeded server.request_timeout",
                )),
            ),
            CancelReason::SlowClient => (
                Outcome::Cancelled,
                Some((ErrorCode::SlowClient, SLOW_CLIENT_MESSAGE)),
            ),
            CancelReason::Shutdown => (
                Outcome::Cancelled,
                Some((ErrorCode::ShuttingDown, "the server is shutting down")),
            ),
            other => (
                Outcome::Cancelled,
                Some((ErrorCode::InternalError, other.as_str())),
            ),
        };
        self.account(id, outcome, reason.as_str());
        match error {
            Some((code, message)) => {
                self.deliver(id, ActiveRequest::error_event(code, message));
                self.retire(id);
            }
            None => self.forget(id),
        }
    }

    /// Steps 5 and 6 with bounded out-of-memory recovery (module comment): returns the outcome
    /// for `Scheduler::complete`. `plan` shrinks when a retry runs a smaller batch.
    fn execute(&mut self, plan: &mut IterationPlan) -> IterationOutcome {
        let mut outcome = IterationOutcome {
            iteration: plan.iteration,
            ..IterationOutcome::default()
        };
        let batch = self.iteration_requests.clone();
        loop {
            let error = match self.attempt(plan) {
                Ok(None) => return outcome,
                Ok(Some((rows, logits))) => {
                    self.sample(&rows, logits, &mut outcome);
                    self.rel.recovery_succeeded();
                    return outcome;
                }
                Err(e) => e,
            };
            if !error.oom {
                if error.sticky {
                    tracing::error!(event = "iteration_failed", reason = "device_fatal", iteration = plan.iteration, error = %error.message, "sticky device error");
                    // The fatal path after `complete` fails every live request.
                    self.rel
                        .circuit_event(CircuitEvent::DeviceError { sticky: true });
                    outcome.failed = Some(IterationFailure {
                        message: error.message,
                    });
                } else {
                    let (code, event) = error.failure();
                    // The circuit opens first (see `pipe_failure`).
                    self.rel.circuit_event(event);
                    self.fail_iteration(code, error.message, &mut outcome);
                }
                return outcome;
            }
            let released = self.rel.on_oom();
            tracing::warn!(event = "allocation_failure", reason = "device_oom", iteration = plan.iteration, batch = plan.items.len(), reserve_released = released, error = %error.message, "device out of memory");
            match self.rel.recovery_step(plan.items.len()) {
                RecoveryStep::Retry {
                    backoff,
                    batch_limit,
                    ..
                } => {
                    std::thread::sleep(backoff);
                    self.sched.shrink_plan(&mut self.pool, plan, batch_limit);
                    self.iteration_requests = plan_requests(plan, &self.seqs);
                }
                RecoveryStep::GiveUp => {
                    self.rel.recovery_failed();
                    // Requests shrunk out of the plan leave the scheduler here; the ones still
                    // in flight fail with the iteration in `complete`.
                    let in_plan = self.iteration_requests.clone();
                    let rest: Vec<RequestId> = batch
                        .iter()
                        .filter(|id| !in_plan.contains(id))
                        .copied()
                        .collect();
                    self.sched.fail_requests(&mut self.pool, &rest);
                    self.fail_requests_with(
                        &batch,
                        ErrorCode::ResourceExhausted,
                        "device memory was exhausted and recovery retries failed",
                    );
                    outcome.failed = Some(IterationFailure {
                        message: error.message,
                    });
                    return outcome;
                }
            }
        }
    }

    /// One attempt at the plan: the fork copies, the batch, then the forward pass (`None` when
    /// the plan has only forks).
    fn attempt(
        &mut self,
        plan: &IterationPlan,
    ) -> Result<Option<(Vec<RowInfo>, Logits)>, IterationError> {
        self.fork_copies(plan)?;
        self.stages.mark(Stage::Prepare);
        if plan.items.is_empty() {
            return Ok(None);
        }
        let built = match self
            .build_batch(plan, None)
            .map_err(IterationError::other)?
        {
            Build::Ready(built) => built,
            // Unreachable: without a launch in flight every token is on the host.
            Build::NeedsHost => {
                return Err(IterationError::other(
                    "a batch token is not on the host".into(),
                ));
            }
        };
        let logits = self.forward(plan, &built)?;
        Ok(Some((built.rows, logits)))
    }

    /// The `n` > 1 fork copies of `plan` (`ModelExecutor::copy_blocks`), ordered before its
    /// forward on the stream.
    fn fork_copies(&mut self, plan: &IterationPlan) -> Result<(), IterationError> {
        let (src, dst): (Vec<_>, Vec<_>) = plan.forks.iter().filter_map(|f| f.copy).unzip();
        if src.is_empty() {
            return Ok(());
        }
        self.exec
            .copy_blocks(&self.pool.view(), &src, &dst)
            .map_err(|e| IterationError::model("copy_blocks failed", e))
    }

    /// Packs `plan` into one ragged batch. Each item's tokens come from its request's history;
    /// with `prev` (a launch whose host work is not done: overlap scheduling), a decode's newest
    /// token that is not on the host yet is taken on the device from `prev`'s row of that
    /// sequence when that row's device choice is its token ([`TokenFeed`]) — otherwise the batch
    /// is `NeedsHost` and `prev` must be finished first. Under overlap scheduling an item whose
    /// choice already ended (the scheduler learns an EOS or stop-string finish one iteration
    /// late) is left out. Samplers are only touched once the batch is known to be buildable,
    /// so a `NeedsHost` build changes nothing.
    fn build_batch(
        &mut self,
        plan: &IterationPlan,
        prev: Option<&InFlight>,
    ) -> Result<Build, String> {
        struct Item {
            index: usize,
            id: RequestId,
            choice: usize,
            start: u32,
            len: u32,
            kv_len: u32,
            prefill: bool,
            yields: bool,
            /// `prev`'s slot whose device choice is this item's newest token.
            feed: Option<usize>,
        }
        let feedable: HashMap<SeqId, usize> = prev
            .map(|p| {
                p.rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.feedable)
                    .map(|(slot, r)| (r.seq, slot))
                    .collect()
            })
            .unwrap_or_default();
        let mut items = Vec::with_capacity(plan.items.len());
        for (index, item) in plan.items.iter().enumerate() {
            let kv_len = item.block_table.tokens;
            let (start, len, prefill) = match item.kind {
                BatchKind::Prefill { start, len } => (start, len, true),
                BatchKind::Decode => (kv_len.saturating_sub(1), 1, false),
            };
            let Some(&(id, choice)) = self.seqs.get(&item.seq) else {
                if self.overlap {
                    continue;
                }
                return Err(format!("sequence {} has no request", item.seq.0));
            };
            let r = self
                .requests
                .get(&id)
                .ok_or_else(|| format!("request {} is not tracked", id.0))?;
            if self.overlap && (r.done || r.choices[choice].finish.is_some()) {
                continue;
            }
            let yields = !prefill || self.sched.prefill_target(item.seq) == Some(start + len);
            let mut feed = None;
            for p in start..start + len {
                if r.token_at(choice, p).is_some() {
                    continue;
                }
                match feedable.get(&item.seq) {
                    Some(&slot) if !prefill && p + 1 == kv_len => feed = Some(slot),
                    _ if prev.is_some() => return Ok(Build::NeedsHost),
                    _ => {
                        return Err(format!(
                            "sequence {} has no token at position {p}",
                            item.seq.0
                        ));
                    }
                }
            }
            items.push(Item {
                index,
                id,
                choice,
                start,
                len,
                kv_len,
                prefill,
                yields,
                feed,
            });
        }

        let total: usize = items.iter().map(|i| i.len as usize).sum();
        let mut built = Built {
            tokens: Vec::with_capacity(total),
            positions: Vec::with_capacity(total),
            slices: Vec::with_capacity(items.len()),
            rows: Vec::with_capacity(items.len()),
            feeds: Vec::new(),
        };
        let reduces = self.exec.reduces_logits();
        for it in items {
            let seq = plan.items[it.index].seq;
            // A token of `prev` for this choice is not observed yet: its sampler asks one ahead.
            let ahead = usize::from(prev.is_some_and(|p| p.yielders.contains(&seq)));
            let reduce = if reduces && it.yields {
                self.device_reduce(it.prefill, it.id, it.choice, ahead)
            } else {
                None
            };
            let r = &self.requests[&it.id];
            let q_start = built.tokens.len() as u32;
            for p in it.start..it.start + it.len {
                built.tokens.push(r.token_at(it.choice, p).unwrap_or(0));
                built.positions.push(p);
            }
            if let Some(slot) = it.feed {
                built.feeds.push(TokenFeed {
                    token: q_start + it.len - 1,
                    prev_slot: slot as u32,
                });
            }
            let seeded = r.choices[it.choice].is_seeded();
            let feedable =
                reduce.is_some_and(|q| q.temperature <= 0.0 || (q.uniform.is_some() && !seeded));
            built.slices.push((it.index, q_start, it.len, it.kv_len));
            built.rows.push(RowInfo {
                seq,
                prefill: it.prefill,
                yields: it.yields,
                reduce,
                feedable,
                fed_next: false,
            });
        }
        Ok(Build::Ready(built))
    }

    /// Runs the forward pass over `built` (one ragged batch of `plan`'s items).
    fn forward(&mut self, plan: &IterationPlan, built: &Built) -> Result<Logits, IterationError> {
        let slices = seq_slices(plan, built);
        let phase = phase_of(plan);
        let started = Instant::now();
        let view = self.pool.view();
        let result = self.exec.forward(&BatchInput {
            tokens: &built.tokens,
            positions: &built.positions,
            seqs: &slices,
            kv: &view,
        });
        if result.is_ok() {
            let t = self.exec.last_timings();
            self.stages.add(Stage::Launch, t.launch);
            self.stages.add(Stage::DeviceWait, t.device_wait);
        }
        self.record_graph_counters();
        self.stages.mark(Stage::Prepare);
        self.metrics
            .model
            .observe_forward(phase, started.elapsed().as_secs_f64());
        let logits = result.map_err(|e| {
            IterationError::model(&format!("{} forward pass failed", phase.as_str()), e)
        })?;
        self.check_logits(&logits, slices.len())
            .map_err(IterationError::other)?;
        Ok(logits)
    }

    /// Adds the executor's decode graph outcomes since the last call to
    /// `turbine_decode_graph_total{outcome}`.
    fn record_graph_counters(&mut self) {
        let graphs = self.exec.graph_counters();
        self.metrics
            .server
            .decode_graphs(&graphs.since(&self.graph_counters));
        self.graph_counters = graphs;
    }

    /// The forward returned one logits slot per sequence; records the rows' paths.
    fn check_logits(&self, logits: &Logits, seqs: usize) -> Result<(), String> {
        if logits.slots() != seqs {
            return Err(format!(
                "forward returned logits for {} of {seqs} sequences",
                logits.slots(),
            ));
        }
        self.metrics
            .server
            .logits_rows(logits.reduced.len(), logits.rows);
        Ok(())
    }

    /// The device reduction a yielding row of choice `choice` of request `id` asks for (P2c
    /// S-4): only a live choice whose sampler finds the step eligible (a constrained choice's
    /// grammar mask keeps it on the host), and not the shared
    /// prefill row that forks further choices (each fork samples its own copy of the whole
    /// row). `ahead` is 1 when the choice's previous token is still on the device (overlap
    /// scheduling): its sampler then asks for the step after the next one it observes.
    fn device_reduce(
        &mut self,
        prefill: bool,
        id: RequestId,
        choice: usize,
        ahead: usize,
    ) -> Option<RowReduce> {
        let r = self.requests.get_mut(&id)?;
        if r.done {
            return None;
        }
        if choice == 0 && prefill && !r.forking_choices().is_empty() {
            return None;
        }
        let c = r.choices.get_mut(choice)?;
        if c.finish.is_some() {
            return None;
        }
        let constrained = c.is_constrained();
        c.sampler_mut().device_request_ahead(ahead, constrained)
    }

    /// Samples every row whose step yields a token and emits its events. The completed shared
    /// prefill of an `n` > 1 request also gives every forking choice its first token, each
    /// from its own copy of the prompt's last logits row. A request whose token the next launch
    /// took on the device but the host chose differently (a row without a finite logit) fails.
    fn sample(&mut self, rows: &[RowInfo], mut logits: Logits, outcome: &mut IterationOutcome) {
        let (mut drawn, diverged) = self.draw_decode_tokens(rows, &mut logits);
        for id in diverged {
            self.token_diverged(id, outcome);
        }
        let vocab = logits.vocab;
        for (slot, row) in rows.iter().enumerate() {
            if !row.yields {
                continue;
            }
            let Some(&(id, choice)) = self.seqs.get(&row.seq) else {
                continue;
            };
            let Some(r) = self.requests.get(&id) else {
                continue;
            };
            if r.done {
                continue;
            }
            let forks = if choice == 0 && row.prefill {
                r.forking_choices()
            } else {
                Vec::new()
            };
            let row_logits = match logits.full_row_index(slot) {
                Some(i) => &mut logits.data[i * vocab..(i + 1) * vocab],
                // A reduced row: its token was drawn from the reduction.
                None => &mut [][..],
            };
            let shared = (!forks.is_empty()).then(|| row_logits.to_vec());
            let token = drawn[slot].take();
            if !self.sample_choice(id, choice, row_logits, token, outcome) {
                continue;
            }
            for fork in forks {
                let mut own = shared.clone().unwrap_or_default();
                if !self.sample_choice(id, fork, &mut own, None, outcome) {
                    break;
                }
            }
        }
    }

    /// Draws the tokens of every decode row of an unconstrained live choice at once, across
    /// threads (`turbine_model::sample_rows`: each sampler only touches its own row, so the
    /// tokens are those of sampling row by row). Indexed by row; `None` rows (prefill
    /// completions with their forks, constrained choices) are sampled by
    /// [`ActiveRequest::step`] on the engine thread. Also returns the requests whose token the
    /// next launch took on the device although the host drew another.
    fn draw_decode_tokens(
        &mut self,
        rows: &[RowInfo],
        logits: &mut Logits,
    ) -> (Vec<Option<SampledToken>>, Vec<RequestId>) {
        let mut drawn: Vec<Option<SampledToken>> = vec![None; rows.len()];
        let mut diverged = Vec::new();
        // Reduced rows (decode or the last prefill chunk) finish from their reduction.
        for (slot, row) in rows.iter().enumerate() {
            let LogitsSlot::Reduced(reduced) = logits.slot(slot) else {
                continue;
            };
            let Some(&(id, choice)) = self.seqs.get(&row.seq) else {
                continue;
            };
            if let Some(sampler) = self
                .requests
                .get_mut(&id)
                .filter(|r| !r.done)
                .and_then(|r| r.choices.get_mut(choice))
                .and_then(|c| c.unconstrained_sampler())
            {
                let token = sampler.finish_reduced(reduced);
                if row.fed_next && device_choice(reduced, row.reduce) != Some(token.token) {
                    diverged.push(id);
                }
                drawn[slot] = Some(token);
            }
        }
        let decode_rows: HashMap<SeqId, usize> = rows
            .iter()
            .enumerate()
            .filter(|&(slot, row)| !row.prefill && logits.full_row_index(slot).is_some())
            .map(|(slot, row)| (row.seq, slot))
            .collect();
        if decode_rows.is_empty() {
            return (drawn, diverged);
        }
        let full_index: Vec<Option<usize>> = (0..logits.slots())
            .map(|row| logits.full_row_index(row))
            .collect();
        let mut full_rows: Vec<Option<&mut [f32]>> = logits
            .data
            .chunks_exact_mut(logits.vocab)
            .map(Some)
            .collect();
        let mut job_rows = Vec::with_capacity(decode_rows.len());
        let mut jobs = Vec::with_capacity(decode_rows.len());
        for r in self.requests.values_mut().filter(|r| !r.done) {
            for c in r.choices.iter_mut() {
                let Some(&row) = decode_rows.get(&c.seq) else {
                    continue;
                };
                if let Some(sampler) = c.unconstrained_sampler()
                    && let Some(row_logits) = full_index[row]
                        .and_then(|i| full_rows.get_mut(i))
                        .and_then(Option::take)
                {
                    job_rows.push(row);
                    jobs.push(SampleJob {
                        sampler,
                        logits: row_logits,
                        mask: None,
                    });
                }
            }
        }
        for (row, token) in job_rows.into_iter().zip(sample_rows(jobs)) {
            drawn[row] = Some(token);
        }
        (drawn, diverged)
    }

    /// One turn under overlap scheduling (P2c): the iteration launched last turn (`in_flight`)
    /// runs on the device while this turn takes commands, completes it in the scheduler from
    /// what is known before its tokens (the tokens it appends, and `max_tokens` and context
    /// finishes, which depend only on counts), plans and launches the next iteration — its
    /// decodes taking their newest tokens on the device from the one in flight — and only then
    /// waits for the one in flight and does its host work (sampling, detokenisation, events),
    /// while the next runs. Finishes that depend on the tokens (EOS, stop ids and strings,
    /// grammars) reach the scheduler one iteration late: such a sequence's row in the next
    /// iteration is computed and dropped. A batch that needs a token the device did not choose
    /// (a constrained, seeded-draw, `top_k` or full-row choice, a fork's first token) waits for
    /// the iteration in flight first, so it runs as the serial loop does.
    fn turn_overlap(&mut self) -> Result<Turn, String> {
        if !self.receive_commands() {
            return Ok(Turn::Stop);
        }
        self.kv.serve_commands(&mut self.pool);
        self.stages.mark(Stage::Schedule);
        self.flush_outputs();
        self.stages.mark(Stage::Emit);
        self.detect_disconnects();
        self.expire_deadlines();
        if let Some(mut prev) = self.in_flight.take() {
            // The KV the iteration writes is held until it is collected (Phase 4): the ahead
            // completion may release a finished sequence's blocks before then. Its input tokens
            // are all on the host now (the iteration before it was collected last turn).
            if prev.launched {
                prev.commits = self.kv_commits(&prev.plan);
                for (_, blocks, _) in &prev.commits {
                    for &b in blocks {
                        self.pool.incref(b);
                    }
                }
            }
            let outcome = self.ahead_outcome(&prev);
            self.sched.complete(&mut self.pool, outcome);
            self.in_flight = Some(prev);
        }
        let limits = self.read_snapshot()?;
        self.kv_before_plan();
        let plan = self.sched.plan(&mut self.pool, &limits);
        self.kv.after_plan(&mut self.pool);
        for &(id, reason) in &plan.dropped {
            self.on_dropped(id, reason);
        }
        self.stages.mark(Stage::Schedule);
        let mut prev = self.in_flight.take();
        if plan.is_empty() {
            let executed = prev.is_some();
            if let Some(p) = prev {
                self.finish(p)?;
            }
            let finished = std::mem::take(&mut self.late_finished);
            self.sched.complete(
                &mut self.pool,
                IterationOutcome {
                    iteration: plan.iteration,
                    finished,
                    ..IterationOutcome::default()
                },
            );
            self.kv_end_turn();
            self.idle_turn = !executed;
            self.publish(executed);
            self.publish_stats();
            self.check_fatal()?;
            return Ok(Turn::Continue);
        }
        self.idle_turn = false;
        self.iteration_requests = plan_requests(&plan, &self.seqs);
        let next = self.launch_next(plan, &mut prev)?;
        if let Some(p) = prev {
            self.finish(p)?;
        }
        self.in_flight = next;
        self.kv_end_turn();
        self.publish(true);
        self.publish_stats();
        self.check_fatal()?;
        Ok(Turn::Continue)
    }

    /// The outcome of the launched iteration `f` before its tokens are known: one token per
    /// yielding row of a live choice (and per forking choice at a completed shared prefill),
    /// the `length` finishes those tokens cause (`max_tokens`, the context), and the finishes
    /// the host learned since the last completion.
    fn ahead_outcome(&mut self, f: &InFlight) -> IterationOutcome {
        let mut outcome = IterationOutcome {
            iteration: f.plan.iteration,
            finished: std::mem::take(&mut self.late_finished),
            ..IterationOutcome::default()
        };
        if !f.launched {
            return outcome;
        }
        for row in f.rows.iter().filter(|r| r.yields) {
            let Some(&(id, choice)) = self.seqs.get(&row.seq) else {
                continue;
            };
            let Some(r) = self.requests.get(&id).filter(|r| !r.done) else {
                continue;
            };
            let mut choices = vec![choice];
            if choice == 0 && row.prefill {
                choices.extend(r.forking_choices());
            }
            for c in choices {
                let ch = &r.choices[c];
                if ch.finish.is_some() {
                    continue;
                }
                outcome.appended.push((ch.seq, 1));
                let generated = ch.generated.len() as u32 + 1;
                if generated >= r.request.stop.max_tokens
                    || r.prompt_len() + generated >= self.max_seq_len
                {
                    outcome.finished.push((ch.seq, FinishReason::Length));
                }
            }
        }
        outcome
    }

    /// Launches `plan` (overlap scheduling): with the iteration in flight `prev` still on the
    /// device when every token not on the host is one it chose, else after finishing `prev`.
    /// A failed launch is classified ([`EngineLoop::launch_failed`]): out of memory runs the
    /// plan through the serial recovery path (`None`: nothing left in flight), any other error
    /// fails the plan's requests (their sequences are reported finished at the next
    /// completion) and goes to the circuit breaker.
    fn launch_next(
        &mut self,
        plan: IterationPlan,
        prev: &mut Option<InFlight>,
    ) -> Result<Option<InFlight>, String> {
        let requests = plan_requests(&plan, &self.seqs);
        let mut next = InFlight {
            plan,
            rows: Vec::new(),
            yielders: HashSet::new(),
            requests,
            launched: false,
            phase: ForwardPhase::Decode,
            launch_time: Duration::ZERO,
            commits: Vec::new(),
        };
        next.phase = phase_of(&next.plan);
        if let Err(error) = self.fork_copies(&next.plan) {
            return self.launch_failed(next, prev, error);
        }
        let built = match self.build_batch(&next.plan, prev.as_ref()) {
            Ok(Build::Ready(built)) => built,
            Ok(Build::NeedsHost) => {
                if let Some(p) = prev.take() {
                    self.finish(p)?;
                }
                match self.build_batch(&next.plan, None) {
                    Ok(Build::Ready(built)) => built,
                    Ok(Build::NeedsHost) => {
                        let error =
                            IterationError::other("a batch token is not on the host".into());
                        return self.launch_failed(next, prev, error);
                    }
                    Err(message) => {
                        return self.launch_failed(next, prev, IterationError::other(message));
                    }
                }
            }
            Err(message) => return self.launch_failed(next, prev, IterationError::other(message)),
        };
        self.stages.mark(Stage::Prepare);
        if built.rows.is_empty() {
            return Ok(Some(next));
        }
        if let Some(p) = prev.as_mut() {
            for f in &built.feeds {
                p.rows[f.prev_slot as usize].fed_next = true;
            }
        }
        let (launched, launch_time) = {
            let slices = seq_slices(&next.plan, &built);
            let view = self.pool.view();
            let started = Instant::now();
            let launched = self.exec.launch(
                &BatchInput {
                    tokens: &built.tokens,
                    positions: &built.positions,
                    seqs: &slices,
                    kv: &view,
                },
                &built.feeds,
            );
            (launched, started.elapsed())
        };
        next.launch_time = launch_time;
        self.record_graph_counters();
        self.stages.mark(Stage::Launch);
        if let Err(e) = launched {
            let error =
                IterationError::model(&format!("{} forward pass failed", next.phase.as_str()), e);
            if let Some(p) = prev.as_mut() {
                for row in &mut p.rows {
                    row.fed_next = false;
                }
            }
            return self.launch_failed(next, prev, error);
        }
        if prev.as_ref().is_some_and(|p| p.launched) {
            self.overlapped += 1;
        }
        next.yielders = self.yielders(&built.rows);
        next.rows = built.rows;
        next.launched = true;
        Ok(Some(next))
    }

    /// Sequences that get a token from `rows` once they are sampled: each yielding row's own,
    /// and the forking choices of a completed shared prefill.
    fn yielders(&self, rows: &[RowInfo]) -> HashSet<SeqId> {
        let mut out = HashSet::new();
        for row in rows.iter().filter(|r| r.yields) {
            out.insert(row.seq);
            if !row.prefill {
                continue;
            }
            if let Some(&(id, 0)) = self.seqs.get(&row.seq)
                && let Some(r) = self.requests.get(&id)
            {
                out.extend(r.forking_choices().into_iter().map(|c| r.choices[c].seq));
            }
        }
        out
    }

    /// `next` could not be launched. Out of memory: the iteration in flight `prev` is finished
    /// first, then `next`'s plan runs through the serial recovery path ([`EngineLoop::execute`])
    /// and is completed in the scheduler at once, so nothing is left in flight. Any other
    /// error fails `next`'s requests and goes to the circuit breaker.
    fn launch_failed(
        &mut self,
        next: InFlight,
        prev: &mut Option<InFlight>,
        error: IterationError,
    ) -> Result<Option<InFlight>, String> {
        if !error.oom {
            self.device_failure(&next.requests, next.plan.iteration, error);
            return Ok(Some(next));
        }
        if let Some(p) = prev.take() {
            self.finish(p)?;
        }
        let mut plan = next.plan;
        let started = Instant::now();
        let mut outcome = self.recover_after(&mut plan, error);
        outcome
            .finished
            .extend(std::mem::take(&mut self.late_finished));
        let failed = outcome.failed.is_some();
        self.sched.complete(&mut self.pool, outcome);
        if !failed {
            self.observe(&plan, started.elapsed().as_secs_f64());
        }
        Ok(None)
    }

    /// The serial recovery loop of [`EngineLoop::execute`] for a plan whose first attempt
    /// already failed with out-of-memory `error`.
    fn recover_after(
        &mut self,
        plan: &mut IterationPlan,
        error: IterationError,
    ) -> IterationOutcome {
        let released = self.rel.on_oom();
        tracing::warn!(event = "allocation_failure", reason = "device_oom", iteration = plan.iteration, batch = plan.items.len(), reserve_released = released, error = %error.message, "device out of memory (overlap launch)");
        self.iteration_requests = plan_requests(plan, &self.seqs);
        match self.rel.recovery_step(plan.items.len()) {
            RecoveryStep::Retry {
                backoff,
                batch_limit,
                ..
            } => {
                std::thread::sleep(backoff);
                self.sched.shrink_plan(&mut self.pool, plan, batch_limit);
                self.iteration_requests = plan_requests(plan, &self.seqs);
                self.execute(plan)
            }
            RecoveryStep::GiveUp => {
                self.rel.recovery_failed();
                let batch = self.iteration_requests.clone();
                self.fail_requests_with(
                    &batch,
                    ErrorCode::ResourceExhausted,
                    "device memory was exhausted and recovery retries failed",
                );
                IterationOutcome {
                    iteration: plan.iteration,
                    failed: Some(IterationFailure {
                        message: error.message,
                    }),
                    ..IterationOutcome::default()
                }
            }
        }
    }

    /// A device error the overlap path cannot retry (a failed launch that is not out of
    /// memory, or an error collecting an iteration the scheduler already completed ahead):
    /// the requests fail — `resource_exhausted` for out of memory (a failed recovery),
    /// `internal_error` otherwise — and the circuit breaker hears of it (a sticky error turns
    /// it fatal).
    fn device_failure(&mut self, ids: &[RequestId], iteration: u64, error: IterationError) {
        if error.oom {
            self.rel.on_oom();
            self.rel.recovery_failed();
            self.fail_requests(ids, iteration, ErrorCode::ResourceExhausted, &error.message);
        } else {
            let (code, event) = error.failure();
            // The circuit opens first (see `pipe_failure`).
            self.rel.circuit_event(event);
            self.fail_requests(ids, iteration, code, &error.message);
        }
    }

    /// Waits for the launched iteration `p` (only it; the next keeps running) and does its host
    /// work: sampling, detokenisation, events. Finishes it learns go to the next completion.
    fn finish(&mut self, mut p: InFlight) -> Result<(), String> {
        let commits = std::mem::take(&mut p.commits);
        let ok = self.collect_in_flight(p);
        // The held KV enters the cache only when the iteration that wrote it succeeded; the
        // hold is dropped either way.
        for (id, blocks, tokens) in commits {
            if ok {
                self.kv.commit(&mut self.pool, id, &blocks, &tokens);
            }
            self.pool.release(&blocks);
        }
        Ok(())
    }

    /// [`EngineLoop::finish`]'s device wait and host work; true when the iteration succeeded.
    fn collect_in_flight(&mut self, p: InFlight) -> bool {
        if !p.launched {
            return false;
        }
        let started = Instant::now();
        let collected = self.exec.collect();
        let waited = started.elapsed();
        self.stages.mark(Stage::DeviceWait);
        self.metrics
            .model
            .observe_forward(p.phase, (p.launch_time + waited).as_secs_f64());
        let logits = collected
            .map_err(|e| {
                IterationError::model(&format!("{} forward pass failed", p.phase.as_str()), e)
            })
            .and_then(|logits| {
                self.check_logits(&logits, p.rows.len())
                    .map(|()| logits)
                    .map_err(IterationError::other)
            });
        match logits {
            Ok(logits) => {
                let mut outcome = IterationOutcome::default();
                self.sample(&p.rows, logits, &mut outcome);
                self.late_finished.extend(outcome.finished);
                self.observe(&p.plan, (p.launch_time + waited).as_secs_f64());
                true
            }
            Err(error) => {
                self.device_failure(&p.requests, p.plan.iteration, error);
                false
            }
        }
    }

    /// Every live request of `ids` (iteration `iteration`) fails with `code`; its live
    /// sequences are reported finished at the next completion (overlap scheduling).
    fn fail_requests(&mut self, ids: &[RequestId], iteration: u64, code: ErrorCode, message: &str) {
        tracing::error!(event = "iteration_failed", iteration, error = %message, "iteration failed");
        for &id in ids {
            let Some(r) = self.requests.get(&id).filter(|r| !r.done) else {
                continue;
            };
            let live: Vec<SeqId> = r.live_seqs().collect();
            self.late_finished
                .extend(live.into_iter().map(|s| (s, FinishReason::Stop)));
            self.account(id, Outcome::Failed, message);
            self.deliver(id, ActiveRequest::error_event(code, message));
            self.retire(id);
        }
    }

    /// The next launch took request `id`'s token on the device, but the host chose another (a
    /// row without a finite logit): its KV no longer matches its tokens, so it ends with
    /// `internal_error`; every live sequence of it is finished.
    fn token_diverged(&mut self, id: RequestId, outcome: &mut IterationOutcome) {
        let Some(r) = self.requests.get_mut(&id).filter(|r| !r.done) else {
            return;
        };
        let live: Vec<SeqId> = r.live_seqs().collect();
        for c in &mut r.choices {
            c.finish.get_or_insert(FinishReason::Stop);
        }
        outcome
            .finished
            .extend(live.into_iter().map(|s| (s, FinishReason::Stop)));
        let message = "the device's token choice differs from the host's (non-finite logits)";
        tracing::warn!(
            event = "fail",
            request_id = %r.request.http_request_id,
            reason = "token_diverged",
            "overlap scheduling: {message}"
        );
        self.account(id, Outcome::Failed, &format!("token_diverged: {message}"));
        self.deliver(
            id,
            ActiveRequest::error_event(ErrorCode::InternalError, message),
        );
        self.retire(id);
    }

    /// Samples choice `choice` of request `id` from `row` — or takes `drawn`, the token its
    /// sampler already drew from that row — and delivers its events. Returns false when the
    /// request ended with a constraint failure.
    fn sample_choice(
        &mut self,
        id: RequestId,
        choice: usize,
        row: &mut [f32],
        drawn: Option<SampledToken>,
        outcome: &mut IterationOutcome,
    ) -> bool {
        let Some(r) = self.requests.get_mut(&id) else {
            return false;
        };
        if r.done || r.choices[choice].finish.is_some() {
            return true;
        }
        let seq = r.choices[choice].seq;
        let metrics = Some(&self.metrics.model);
        let stepped = match drawn {
            Some(token) => r.step_sampled(choice, token, self.max_seq_len, metrics),
            None => r.step(choice, row, self.max_seq_len, metrics),
        };
        let step = match stepped {
            Ok(step) => step,
            Err(e) => {
                self.constraint_failed(id, &e.to_string(), outcome);
                return false;
            }
        };
        let now = Instant::now();
        let probe = self.probe.as_ref().is_some_and(|(p, _)| *p == id);
        let c = &mut r.choices[choice];
        match c.last_token_at {
            // A circuit probe is not a client request: no latency samples.
            _ if probe => {}
            None => self
                .metrics
                .server
                .ttft
                .observe((now - r.arrived).as_secs_f64()),
            Some(prev) => self.metrics.server.itl.observe((now - prev).as_secs_f64()),
        }
        c.last_token_at = Some(now);
        outcome.appended.push((seq, 1));
        let finished_event = step.finish.map(|reason| r.finished_event(choice, reason));
        let all_finished = r.all_finished();
        if let Some(reason) = step.finish {
            outcome.finished.push((seq, reason));
        }
        self.stages.add(Stage::Detokenize, step.detokenize);
        self.stages.mark(Stage::Sample);
        // The request is accounted before its last events go out, so a client that has read
        // the whole response sees its metrics.
        if all_finished {
            self.account(id, Outcome::Ok, "");
            self.stages.mark(Stage::Complete);
        }
        for event in step.events {
            self.deliver(id, event);
        }
        if let Some(event) = finished_event {
            self.deliver(id, event);
        }
        self.stages.mark(Stage::Emit);
        if all_finished {
            self.retire(id);
            self.stages.mark(Stage::Complete);
        }
        true
    }

    /// A matcher failed mid-generation (llguidance step limit, a stuck constraint): that
    /// request alone ends with `internal_error` (reason `constraint_error`) and every live
    /// sequence of it is finished, so the scheduler frees its blocks; the batch continues.
    fn constraint_failed(&mut self, id: RequestId, message: &str, outcome: &mut IterationOutcome) {
        let Some(r) = self.requests.get_mut(&id) else {
            return;
        };
        let live: Vec<SeqId> = r.live_seqs().collect();
        for c in &mut r.choices {
            c.finish.get_or_insert(FinishReason::Stop);
        }
        for seq in live {
            outcome.finished.push((seq, FinishReason::Stop));
        }
        tracing::warn!(
            event = "fail",
            request_id = %r.request.http_request_id,
            reason = "constraint_error",
            error = %message,
            "constrained decoding failed"
        );
        self.account(id, Outcome::Failed, &format!("constraint_error: {message}"));
        self.deliver(
            id,
            ActiveRequest::error_event(ErrorCode::InternalError, message),
        );
        self.retire(id);
    }

    /// Every request of the iteration fails with `code` (`internal_error`, or `replica_failed`
    /// for a failed collective); the scheduler frees them.
    fn fail_iteration(&mut self, code: ErrorCode, message: String, outcome: &mut IterationOutcome) {
        tracing::error!(event = "iteration_failed", iteration = outcome.iteration, code = code.as_str(), error = %message, "iteration failed");
        for id in self.iteration_requests.clone() {
            if self.requests.get(&id).is_some_and(|r| !r.done) {
                self.account(id, Outcome::Failed, &message);
                self.deliver(id, ActiveRequest::error_event(code, message.clone()));
                self.retire(id);
            }
        }
        outcome.failed = Some(IterationFailure { message });
    }

    /// The engine stops: every request not yet accounted gets `code`.
    fn fail_all(&mut self, code: ErrorCode, message: &str) {
        let live: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| !r.done)
            .map(|(id, _)| *id)
            .collect();
        for id in live {
            self.account(id, Outcome::Failed, message);
            self.deliver(id, ActiveRequest::error_event(code, message));
            self.retire(id);
        }
    }

    /// Sends `event` to request `id`'s stream without waiting; see the module comment.
    fn deliver(&mut self, id: RequestId, event: GenerationEvent) {
        let Some(r) = self.requests.get_mut(&id) else {
            return;
        };
        match r.emit(event) {
            Delivery::Sent => {}
            Delivery::Held { now_full } => {
                self.deadlines.paused(id);
                if now_full && !r.done {
                    let seqs: Vec<SeqId> = r.live_seqs().collect();
                    for seq in seqs {
                        self.sched.pause(seq);
                    }
                    self.metrics.server.stream_paused.inc();
                }
            }
            Delivery::Closed => self.client_gone(id),
        }
    }

    /// Records the end of request `id` once (tokens, latency, outcome, log line). The circuit
    /// probe's end goes to the circuit breaker instead: `ProbeSucceeded` with its per-token
    /// latency over the GREEN baseline decode step (1.0 before a baseline exists), else
    /// `ProbeFailed`.
    fn account(&mut self, id: RequestId, outcome: Outcome, detail: &str) {
        let Some(r) = self.requests.get_mut(&id) else {
            return;
        };
        if r.done {
            return;
        }
        r.done = true;
        if self.probe.as_ref().is_some_and(|(p, _)| *p == id) {
            let per_token = r.arrived.elapsed().as_secs_f64() / r.generated_tokens().max(1) as f64;
            self.probe = None;
            let event = if outcome == Outcome::Ok {
                let latency_ratio = self
                    .baseline_step_s
                    .filter(|b| *b > 0.0)
                    .map_or(1.0, |b| per_token / b);
                CircuitEvent::ProbeSucceeded { latency_ratio }
            } else {
                CircuitEvent::ProbeFailed
            };
            tracing::info!(event = "circuit_probe", request_id = %id.0, outcome = outcome.as_str(), per_token_seconds = per_token, reason = detail, "circuit probe finished");
            self.rel.circuit_event(event);
            return;
        }
        self.kv_done.push((id, outcome != Outcome::Ok));
        let generated = r.generated_tokens();
        let m = &self.metrics.server;
        m.add_tokens(TokenKind::Generated, generated);
        let e2e = r.arrived.elapsed().as_secs_f64();
        if outcome == Outcome::Ok || outcome == Outcome::Failed {
            m.e2e.observe(e2e);
        }
        m.request(r.request.endpoint, outcome);
        tracing::info!(
            event = "request_finished",
            request_id = %r.request.http_request_id,
            endpoint = r.request.endpoint.as_str(),
            outcome = outcome.as_str(),
            reason = detail,
            prompt_tokens = r.request.prompt_tokens.len(),
            generated_tokens = generated,
            e2e_seconds = e2e,
            "request finished"
        );
        if outcome == Outcome::Failed {
            tracing::error!(request_id = %r.request.http_request_id, error = %detail, "request failed");
        }
    }

    /// After a done request's last events: forget it unless some are still held.
    fn retire(&mut self, id: RequestId) {
        if self
            .requests
            .get(&id)
            .is_some_and(|r| r.done && !r.has_held())
        {
            self.forget(id);
        }
    }

    /// Drops the host state of request `id`.
    fn forget(&mut self, id: RequestId) {
        self.deadlines.forget(id);
        if let Some(r) = self.requests.remove(&id) {
            for c in &r.choices {
                self.seqs.remove(&c.seq);
            }
        }
    }

    /// Step 7. Closes the turn's stage clock: the documents carry its `stages_ms`, and an
    /// `executed` iteration (a non-empty plan) records its stages.
    fn publish(&mut self, executed: bool) {
        let mut scheduler = self.sched.snapshot();
        if let Some(p) = &self.pipeline {
            scheduler.pipeline = Some(p.snapshot(self.pipe.len() as u32));
        }
        let kv = self.kv.document(&self.pool);
        self.metrics.kv.record(&self.pool);
        self.stages.mark(Stage::Complete);
        let stages = std::mem::replace(&mut self.stages, StageClock::start()).finish();
        if executed {
            self.metrics.server.observe_stages(&stages);
        }
        scheduler.last_iteration.stages_ms = stages
            .to_ms_map()
            .into_iter()
            .map(|(stage, ms)| (stage.to_string(), ms))
            .collect();
        self.shared.publish(EngineDocs { scheduler, kv });
        let pending = self
            .requests
            .values()
            .filter(|r| !r.done)
            .map(|r| {
                outstanding(
                    r.prompt_len(),
                    r.request.n,
                    r.request.stop.max_tokens,
                    r.generated_tokens(),
                )
            })
            .sum();
        self.shared.set_outstanding(pending);
    }
}

/// The [`SeqSlice`] of each row of `built`, a batch of `plan`.
fn seq_slices<'a>(plan: &'a IterationPlan, built: &Built) -> Vec<SeqSlice<'a>> {
    built
        .slices
        .iter()
        .zip(&built.rows)
        .map(|(&(index, q_start, q_len, kv_len), row)| SeqSlice {
            seq: row.seq,
            q_start,
            q_len,
            kv_len,
            block_table: &plan.items[index].block_table.blocks,
            reduce: row.reduce,
        })
        .collect()
}

/// `prefill` when the plan writes any prefill chunk, else `decode`.
fn phase_of(plan: &IterationPlan) -> ForwardPhase {
    if plan.prefill_tokens() > 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    }
}

/// The token the device chose for a row reduced as `reduce` asked (the one a launch fed on
/// the device takes): its draw, or its best candidate for a greedy row.
fn device_choice(reduced: &ReducedRow, reduce: Option<RowReduce>) -> Option<u32> {
    match reduce? {
        q if q.uniform.is_some() => reduced.sampled.map(|(id, _)| id),
        q if q.temperature <= 0.0 => reduced.top.first().map(|&(id, _)| id),
        _ => None,
    }
}

/// Why an iteration attempt failed, as the execution backend classifies it
/// (`KernelError::is_oom`, `KernelError::is_sticky`): out of memory is recovered from, a sticky
/// device error is fatal, anything else opens the circuit. A tensor-parallel group's failed
/// collective (P5: a timeout, a rank's abort or a backend error) opens it with reason
/// `collective_failed`, and the iteration's requests end with `replica_failed`.
#[derive(Debug)]
struct IterationError {
    message: String,
    oom: bool,
    sticky: bool,
    collective: bool,
}

impl IterationError {
    fn model(context: &str, e: ModelError) -> IterationError {
        let (oom, sticky, collective) = match &e {
            ModelError::Kernel(k) => (k.is_oom(), k.is_sticky(), false),
            ModelError::Collective(_) => (false, false, true),
            _ => (false, false, false),
        };
        IterationError {
            message: format!("{context}: {e}"),
            oom,
            sticky,
            collective,
        }
    }

    fn other(message: String) -> IterationError {
        IterationError {
            message,
            oom: false,
            sticky: false,
            collective: false,
        }
    }

    /// The error code the iteration's requests end with, and the circuit event.
    fn failure(&self) -> (ErrorCode, CircuitEvent) {
        if self.collective {
            (ErrorCode::ReplicaFailed, CircuitEvent::CollectiveFailed)
        } else {
            (
                ErrorCode::InternalError,
                CircuitEvent::DeviceError {
                    sticky: self.sticky,
                },
            )
        }
    }
}

/// Tokens a request still has to process: its prompt plus `n × max_tokens`, less what it
/// generated (the data-parallel router's load measure, P5 S-7).
fn outstanding(prompt_len: u32, n: u32, max_tokens: u32, generated: u64) -> u64 {
    (u64::from(prompt_len) + u64::from(n.max(1)) * u64::from(max_tokens)).saturating_sub(generated)
}

/// Distinct requests of the plan's items and forks, in plan order.
fn plan_requests(
    plan: &IterationPlan,
    seqs: &HashMap<SeqId, (RequestId, usize)>,
) -> Vec<RequestId> {
    let mut ids: Vec<RequestId> = Vec::new();
    let all = plan
        .items
        .iter()
        .map(|i| i.seq)
        .chain(plan.forks.iter().flat_map(|f| [f.src, f.dst]));
    for seq in all {
        if let Some(&(id, _)) = seqs.get(&seq)
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc::error::TryRecvError;
    use tokio::sync::oneshot;
    use turbine_core::clock::{FakeClock, SystemClock};
    use turbine_core::config::{ByteSize, KvConfig, ReliabilityConfig};
    use turbine_core::pressure::PressureSignal;
    use turbine_core::request::CancelFlag;
    use turbine_core::telemetry::{
        DeviceSample, HostSample, LedgerSample, SourceStatus, TelemetrySample,
    };
    use turbine_core::types::{DeviceId, KvLayout, MemoryKind, ModelIdentity, ModelShape};
    use turbine_kernels::test_support::plain_device_error;
    use turbine_kernels::{KernelError, KernelMetrics, KernelRegistry, cpu_reference_provider};
    use turbine_kv::identity::KvFormat;
    use turbine_kv::tier::L2NvmeTier;
    use turbine_kv::{BlockPoolConfig, KvMetrics};
    use turbine_model::executor::{self, ExecutorOptions, SequenceKv};
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::{TinySpec, write_tiny_llama};
    use turbine_model::{
        GenerateOptions, MAX_STAGING_BYTES, ModelError, ModelMetrics, SafetensorsIndex,
        WeightLoader, generate, llama_slots,
    };
    use turbine_observability::MetricsRegistry;
    use turbine_reliability::budget::{DeviceBudget, PoolKind};
    use turbine_reliability::controller::{ControllerHandle, PressureController};
    use turbine_reliability::ledger::Ledger;
    use turbine_reliability::metrics::ReliabilityMetrics;
    use turbine_reliability::reserve::EmergencyReserve;
    use turbine_scheduler::{SchedulerMetrics, SchedulerParams};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceMemory, KvPoolView};

    use super::*;
    use crate::kv_orchestrator::{CopyDevice, KvStart, kv_format};
    use crate::metrics::ServerMetrics;
    use crate::reliability::{DeviceReserve, ReliabilityInputs};

    const BLOCK_TOKENS: u32 = 16;

    type Admitted = oneshot::Receiver<Result<(), SubmitError>>;

    fn params(max_running: u32, prefill_chunk: u32) -> SchedulerParams {
        SchedulerParams {
            max_running_requests: max_running,
            max_batch_tokens: 64,
            prefill_chunk_tokens: prefill_chunk,
            max_queued_requests: 8,
            chunked_prefill: true,
            block_tokens: BLOCK_TOKENS,
            free_watermark: 0.01,
            max_seq_len: 128,
            queue_timeout: Duration::from_secs(60),
        }
    }

    fn tiny() -> (TempDir, TinySpec, Arc<Tokenizer>) {
        let dir = TempDir::new("turbine-engine-loop");
        let spec = write_tiny_llama(dir.path(), 7);
        let tokenizer = Arc::new(Tokenizer::from_file(&spec.dir.join("tokenizer.json")).unwrap());
        (dir, spec, tokenizer)
    }

    fn mem() -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(0), 1 << 30)
    }

    /// The tiny Llama on the cpu reference provider, for batches of `max_seqs` sequences.
    fn tiny_executor(spec: &TinySpec, max_seqs: u32) -> Box<dyn ModelExecutor> {
        let cfg = &spec.config;
        let mem = mem();
        let index = SafetensorsIndex::open(&spec.dir).unwrap();
        let weights =
            WeightLoader::load(&index, &llama_slots(cfg), &mem, MAX_STAGING_BYTES).expect("load");
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let registry = KernelRegistry::build(
            vec![provider],
            &order,
            &executor::requirements(cfg, BLOCK_TOKENS, ExecutorOptions::default()),
            &KernelMetrics::register(&MetricsRegistry::new()),
            None,
        )
        .unwrap();
        executor::build_executor(
            cfg,
            weights,
            Arc::new(registry),
            mem,
            BLOCK_TOKENS,
            64,
            max_seqs,
            ExecutorOptions::default(),
        )
        .expect("executor")
    }

    fn request(prompt: &[u32], max_tokens: u32) -> GenerationRequest {
        GenerationRequest {
            id: RequestId::new_v4(),
            n: 1,
            priority: Priority::default(),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            session: None,
            cache_salt: None,
            kv_policy: None,
            endpoint: Endpoint::Completions,
            http_request_id: "t".into(),
            prompt_tokens: prompt.to_vec(),
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: StopConditions {
                max_tokens,
                ignore_eos: true,
                ..StopConditions::default()
            },
        }
    }

    struct TestEngine {
        engine: EngineLoop,
        tx: mpsc::Sender<EngineCommand>,
        shared: Arc<EngineShared>,
        reg: MetricsRegistry,
        controller: ControllerHandle,
        /// The pressure controller itself: a test ticks it (no controller thread runs).
        pressure: Arc<std::sync::Mutex<PressureController>>,
        clock: Arc<dyn Clock>,
    }

    /// An engine over `exec` with a 64-block pool, overlap scheduling off.
    fn engine(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
    ) -> TestEngine {
        engine_with(exec, tokenizer, params, false)
    }

    /// [`engine`] with `execution.overlap_scheduling` = `overlap` and the default `kv` section
    /// (L0 only: the cpu backend has no pinned memory and L2 is off).
    fn engine_with(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
        overlap: bool,
    ) -> TestEngine {
        engine_with_kv(
            exec,
            tokenizer,
            params,
            overlap,
            KvConfig::default(),
            |_, _, _| None,
        )
    }

    /// An engine over `exec` with a 64-block pool, the `kv` section `kv` (its block size set
    /// to the executor's) and the L2 tier `l2` builds from the KV format and metrics.
    fn engine_with_kv(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
        overlap: bool,
        kv: KvConfig,
        l2: impl FnOnce(&KvConfig, &KvFormat, KvMetrics) -> Option<Arc<L2NvmeTier>>,
    ) -> TestEngine {
        engine_with_pipeline(exec, tokenizer, params, overlap, kv, l2, None)
    }

    /// [`engine_with_kv`] with a pipeline's stats (`exec` a pipeline's executor; the scheduler
    /// keeps its micro-batches in flight).
    fn engine_with_pipeline(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
        overlap: bool,
        kv: KvConfig,
        l2: impl FnOnce(&KvConfig, &KvFormat, KvMetrics) -> Option<Arc<L2NvmeTier>>,
        pipeline: Option<Arc<PipelineStats>>,
    ) -> TestEngine {
        let config = ReliabilityConfig {
            emergency_vram_reserve: ByteSize(0),
            ..ReliabilityConfig::default()
        };
        engine_full(exec, tokenizer, params, overlap, kv, l2, pipeline, config)
    }

    /// [`engine_with_pipeline`] under the reliability section `config`.
    #[allow(clippy::too_many_arguments)]
    fn engine_full(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
        overlap: bool,
        mut kv: KvConfig,
        l2: impl FnOnce(&KvConfig, &KvFormat, KvMetrics) -> Option<Arc<L2NvmeTier>>,
        pipeline: Option<Arc<PipelineStats>>,
        config: ReliabilityConfig,
    ) -> TestEngine {
        let reg = MetricsRegistry::new();
        let metrics = EngineMetrics {
            server: ServerMetrics::register(&reg),
            model: ModelMetrics::register(&reg),
            scheduler: SchedulerMetrics::register(&reg),
            kv: KvMetrics::register(&reg),
        };
        let layout = *exec.kv_layout();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        // The P3 reliability side over a budget whose kv pool is exactly the 64 blocks, with
        // no emergency reserve.
        let kv_bytes = 64 * layout.block_bytes();
        let budget = DeviceBudget {
            device: DeviceId(0),
            memory_kind: MemoryKind::Dedicated,
            budget_bytes: kv_bytes,
            pools: vec![(PoolKind::Kv, kv_bytes), (PoolKind::Reserve, 0)],
        };
        let reliability_metrics = ReliabilityMetrics::register(&reg);
        let ledger = Ledger::new(&budget);
        ledger.set_metrics(reliability_metrics.clone());
        let reserve = EmergencyReserve::acquire(
            DeviceId(0),
            0,
            &ledger,
            Box::new(DeviceReserve::new(mem())),
            reliability_metrics.clone(),
        )
        .unwrap();
        let mut pool = BlockPool::new(
            BlockPoolConfig {
                layout,
                num_blocks: 64,
            },
            mem(),
        )
        .unwrap()
        .with_ledger(Arc::clone(&ledger), DeviceId(0));
        kv.block_tokens = layout.block_tokens;
        let l2 = l2(&kv, &kv_format(layout), metrics.kv.clone());
        let (kv, _handle) = KvOrchestrator::start(
            KvStart {
                cfg: &kv,
                memory_kind: MemoryKind::Dedicated,
                identity: ModelIdentity::from_bytes(b"tiny config", b"tiny index"),
                device: CopyDevice::Sync {
                    mem: pool_mem(&pool),
                },
                shards: Vec::new(),
                l2,
                clock: Arc::clone(&clock),
                metrics: metrics.kv.clone(),
                remote: None,
                kv_scales: None,
            },
            &mut pool,
        )
        .expect("the KV hierarchy starts");
        let parts = crate::reliability::build(ReliabilityInputs {
            config: &config,
            budget,
            ledger: Arc::clone(&ledger),
            reserve,
            held: Vec::new(),
            params: &params,
            block_bytes: layout.block_bytes(),
            workspace_bytes_per_token: 0,
            metrics: reliability_metrics,
            clock: Arc::clone(&clock),
            reclaimer: kv.reclaimer(),
            replica: 0,
            group: Vec::new(),
        });
        let scheduler = Scheduler::new(params, Arc::clone(&clock))
            .with_metrics(metrics.scheduler.clone())
            .with_gate(parts.gate)
            .with_micro_batches(pipeline.as_ref().map_or(1, |p| p.micro_batches));
        let controller = parts.engine.handle.clone();
        let pressure = Arc::clone(&parts.controller);
        let test_clock = Arc::clone(&clock);
        let (tx, commands) = mpsc::channel(8);
        let shared = Arc::new(EngineShared::default());
        let engine = EngineLoop::new(EngineParts {
            executor: exec,
            pool,
            scheduler,
            clock,
            commands,
            shared: Arc::clone(&shared),
            tokenizer,
            max_seq_len: 128,
            metrics,
            timeouts: Timeouts {
                request: Duration::from_secs(600),
                slow_client: Duration::from_secs(30),
            },
            overlap,
            reliability: parts.engine,
            kv,
            pipeline,
        });
        TestEngine {
            engine,
            tx,
            shared,
            reg,
            controller,
            pressure,
            clock: test_clock,
        }
    }

    /// L0 blocks requests hold (`referenced_blocks`): from Phase 4 finished requests leave their
    /// full blocks cached, allocated but unreferenced.
    fn held_blocks(docs: &EngineDocs) -> u64 {
        docs.kv.tiers[0]
            .state
            .as_ref()
            .expect("the hierarchy's document")
            .referenced_blocks
    }

    /// The host memory the test pools live in.
    fn pool_mem(pool: &BlockPool) -> Arc<dyn DeviceMemory> {
        Arc::clone(pool.view().storage.memory())
    }

    /// Greedy tokens and `usage.cached_tokens` of one request run to completion.
    fn run_one(tx: &mpsc::Sender<EngineCommand>, req: GenerationRequest) -> (Vec<u32>, u32) {
        let (mut rx, admitted) = submit(tx, req);
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let mut tokens = Vec::new();
        loop {
            match rx.blocking_recv().expect("stream ended early") {
                GenerationEvent::Token { token_id, .. } => tokens.push(token_id),
                GenerationEvent::Finished { usage, .. } => {
                    return (tokens, usage.expect("usage").cached_tokens);
                }
                GenerationEvent::Error { code, message } => panic!("{code:?}: {message}"),
                _ => {}
            }
        }
    }

    /// The value of the Prometheus series `series` (0 when absent).
    fn metric(reg: &MetricsRegistry, series: &str) -> f64 {
        reg.render()
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix(series)?.trim().parse().ok())
            .unwrap_or(0.0)
    }

    /// Prefix sharing in the engine (P4 S-3, S-17 on the cpu backend), serial and overlapped:
    /// a repeated prompt attaches its two full 16-token blocks instead of prefilling them,
    /// reports 32 `cached_tokens` and yields exactly the cold run's greedy tokens; the same
    /// prompt under a cache salt shares nothing. Breaks if reuse changes the output, is not
    /// used, or crosses salts.
    #[test]
    fn prefix_reuse_reports_cached_tokens_and_matches_cold() {
        for overlap in [false, true] {
            let (_dir, spec, tokenizer) = tiny();
            let exec = if overlap {
                tiny_reducing_executor(&spec, 4)
            } else {
                tiny_executor(&spec, 4)
            };
            let t = engine_with(exec, Arc::clone(&tokenizer), params(4, 64), overlap);
            let prompt: Vec<u32> = std::iter::once(256).chain(97..136).collect();
            assert_eq!(prompt.len(), 40);
            let TestEngine {
                engine,
                tx,
                shared,
                reg,
                ..
            } = t;
            let handle = std::thread::spawn(move || engine.run());

            let (cold, cached) = run_one(&tx, request(&prompt, 8));
            assert_eq!(cached, 0, "nothing is cached before the first run");
            let (warm, cached) = run_one(&tx, request(&prompt, 8));
            assert_eq!(cached, 32, "two full blocks are reused (overlap {overlap})");
            assert_eq!(
                warm, cold,
                "reused KV gives the cold run's tokens (overlap {overlap})"
            );
            let mut salted = request(&prompt, 8);
            salted.cache_salt = Some("a".into());
            let (tokens, cached) = run_one(&tx, salted);
            assert_eq!(
                cached, 0,
                "a salted request shares nothing with unsalted ones"
            );
            assert_eq!(tokens, cold);

            let kv = serde_json::to_value(shared.docs().unwrap().kv).unwrap();
            assert_eq!(kv["hit_rate"]["prompt_tokens"], 120, "{kv}");
            assert_eq!(kv["hit_rate"]["cached_tokens"], 32, "{kv}");
            assert_eq!(kv["policy"], "cost_aware", "{kv}");
            assert!(metric(&reg, "turbine_kv_prefix_cached_tokens_total") >= 32.0);
            drop(tx);
            handle.join().unwrap().unwrap();
        }
    }

    /// L0 → L2 → L0 on the cpu backend (P4 S-8, S-11, S-17): after a demotion request every
    /// cached block of a finished prompt leaves L0 for the NVMe tier; the prompt sent again
    /// promotes them back before its prefill and yields the cold run's greedy tokens, with no
    /// checksum eviction. Breaks if the round trip corrupts KV, the blocks never leave L0, or
    /// the promotion is skipped.
    #[test]
    fn l2_round_trip_matches_cold() {
        let (dir, spec, tokenizer) = tiny();
        let mut kv = KvConfig::default();
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = ByteSize(16 << 20);
        kv.nvme.slab_bytes = ByteSize(1 << 20);
        let t = engine_with_kv(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 64),
            false,
            kv,
            |cfg, format, metrics| {
                crate::kv_orchestrator::open_l2(
                    cfg,
                    format,
                    &ModelIdentity::from_bytes(b"tiny config", b"tiny index"),
                    Arc::new(SystemClock::new()),
                    metrics,
                )
                .expect("L2 opens in the temp directory")
            },
        );
        let reclaim = t.engine.kv.reclaimer();
        let TestEngine {
            engine, tx, reg, ..
        } = t;
        let handle = std::thread::spawn(move || engine.run());

        let prompt: Vec<u32> = std::iter::once(256).chain(97..136).collect();
        let (cold, _) = run_one(&tx, request(&prompt, 8));
        // A second run hits the cached blocks: the reuse evidence demotion needs.
        let (again, cached) = run_one(&tx, request(&prompt, 8));
        assert_eq!((again.as_slice(), cached), (cold.as_slice(), 32));
        // Everything unreferenced leaves L0 at the end of the next turn.
        reclaim.demote(0.0);
        let _ = run_one(&tx, request(&[256, 1, 2], 2));
        let demoted = r#"turbine_kv_demotions_total{from="l0",to="l2"}"#;
        let started = Instant::now();
        while metric(&reg, demoted) < 2.0 {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the prompt's blocks never reached L2"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let (warm, cached) = run_one(&tx, request(&prompt, 8));
        assert_eq!(cached, 32, "both blocks came back from L2");
        assert_eq!(
            warm, cold,
            "KV that went through L2 gives the cold run's tokens"
        );
        assert!(metric(&reg, r#"turbine_kv_promotions_total{from="l2",to="l0"}"#) >= 2.0);
        assert_eq!(
            metric(
                &reg,
                r#"turbine_kv_evictions_total{tier="l2",reason="checksum"}"#
            ),
            0.0
        );
        drop(tx);
        handle.join().unwrap().unwrap();
    }

    /// The sum of every Prometheus series starting with `prefix` (labels included).
    fn metric_sum(reg: &MetricsRegistry, prefix: &str) -> f64 {
        reg.render()
            .unwrap()
            .lines()
            .filter(|l| l.starts_with(prefix))
            .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
            .sum()
    }

    /// One run of [`ladder_rewrites_l2_copies_like_a_direct_fp8_tier`]: a prompt's two blocks go
    /// to L2 at GREEN (stored `l0` with the ladder, `fp8_e4m3` without), the controller is held
    /// at ORANGE while small requests drive iterations, then the prompt runs again. Returns its
    /// tokens, its cached tokens, the ladder's l0 → fp8 rewrites in L2 and L2's bytes.
    fn ladder_run(ladder: bool) -> (Vec<u32>, u32, f64, u64) {
        let (dir, spec, tokenizer) = tiny();
        let mut kv = KvConfig::default();
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = ByteSize(16 << 20);
        kv.nvme.slab_bytes = ByteSize(1 << 20);
        // Every full block of the prompt is lossy in both runs (no lossless tail at demotion).
        kv.lossless_tail_blocks = 0;
        let fp8 = turbine_core::config::ModuleName::new("fp8_e4m3").unwrap();
        if ladder {
            kv.ladder.enabled = true;
            kv.ladder.l0 = false;
            kv.ladder.max_format = fp8.clone();
        } else {
            kv.nvme.format = fp8;
        }
        let mut config = ReliabilityConfig {
            emergency_vram_reserve: ByteSize(0),
            ..ReliabilityConfig::default()
        };
        // ORANGE from 2 % KV utilisation: the ladder compresses the lowest tier at ORANGE
        // whatever its fill; tiny prompts stay cheap enough to be admitted there.
        config.pressure.thresholds.insert(
            PressureSignal::KvUtilization,
            [Some(0.01), Some(0.02), Some(0.95), Some(0.99)],
        );
        let l2_tier = Arc::new(std::sync::Mutex::new(None));
        let opened = Arc::clone(&l2_tier);
        let mut t = engine_full(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 64),
            false,
            kv.clone(),
            move |cfg, format, metrics| {
                let l2 = crate::kv_orchestrator::open_l2(
                    cfg,
                    format,
                    &ModelIdentity::from_bytes(b"tiny config", b"tiny index"),
                    Arc::new(SystemClock::new()),
                    metrics,
                )
                .expect("L2 opens in the temp directory");
                *opened.lock().unwrap() = l2.clone();
                l2
            },
            None,
            config,
        );
        // As the engine thread does at startup (the cpu backend's transcode is the cpu
        // reference provider).
        let mem = pool_mem(&t.engine.pool);
        kv.block_tokens = BLOCK_TOKENS;
        assert!(
            t.engine
                .kv
                .enable_device_transcode(&kv, cpu_reference_provider(), &mem),
            "the fp8 rung runs on the device transcode"
        );
        let reclaim = t.engine.kv.reclaimer();
        let ctl = (Arc::clone(&t.pressure), Arc::clone(&t.clock));
        let TestEngine {
            engine,
            tx,
            reg,
            controller,
            ..
        } = t;
        let handle = std::thread::spawn(move || engine.run());
        let demoted = r#"turbine_kv_demotions_total{from="l0",to="l2"}"#;
        let rewrites = r#"turbine_kv_ladder_actions_total{tier="l2",from="l0",to="fp8_e4m3""#;

        let prompt: Vec<u32> = std::iter::once(256).chain(97..136).collect();
        let _ = run_one(&tx, request(&prompt, 8));
        let (_, cached) = run_one(&tx, request(&prompt, 8));
        assert_eq!(cached, 32, "the reuse evidence demotion needs");
        reclaim.demote(0.0);
        let _ = run_one(&tx, request(&[256, 1, 2], 2));
        let deadline = Instant::now() + Duration::from_secs(20);
        while metric(&reg, demoted) < 2.0 {
            assert!(
                Instant::now() < deadline,
                "the prompt's blocks never reached L2 (ladder {ladder}): {}",
                metric(&reg, demoted)
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        for _ in 0..3 {
            tick(&ctl, 0.5);
        }
        assert_eq!(controller.state(), PressureState::Orange);
        // Iterations at ORANGE: the ladder rewrites L2's l0 copies (with the ladder on).
        let deadline = Instant::now() + Duration::from_secs(20);
        for i in 0.. {
            tick(&ctl, 0.5);
            let _ = run_one(&tx, request(&[256, 3, 4 + i % 50], 2));
            if (ladder && metric_sum(&reg, rewrites) >= 2.0) || (!ladder && i >= 5) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the ladder never rewrote the L2 copies: {} rewrites\n{}",
                metric_sum(&reg, rewrites),
                reg.render()
                    .unwrap()
                    .lines()
                    .filter(|l| l.contains("ladder")
                        || l.contains("pressure_state")
                        || l.contains("tier_blocks")
                        || l.contains("demotions_total")
                        || l.contains("evictions_total"))
                    .filter(|l| !l.starts_with('#') && !l.ends_with(" 0"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        assert_eq!(controller.state(), PressureState::Orange);
        let (tokens, cached) = run_one(&tx, request(&prompt, 8));
        let l2_bytes = l2_tier
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |t| turbine_kv::tier::KvTier::used_bytes(t.as_ref()));
        let actions = metric_sum(&reg, rewrites);
        drop(tx);
        handle.join().unwrap().unwrap();
        (tokens, cached, actions, l2_bytes)
    }

    /// P6b S-6, the L1/L2 ladder on the server (cpu backend): with `kv.ladder.enabled` and an
    /// `l0` L2, ORANGE pressure rewrites the L2 copies of a prompt's blocks one rung down to
    /// `fp8_e4m3` through the device transcode (the cpu reference provider here), and the prompt
    /// sent again reuses them from L2 with exactly the tokens it gets from an L2 that stored them
    /// as `fp8_e4m3` in the first place (the same bytes), at the same L2 bytes. Breaks if the
    /// ladder is refused or never acts on the server, a rewrite stores other bytes than a direct
    /// fp8 demotion (a missing decode, the source's bytes), or a rewritten copy is not reused.
    #[test]
    fn ladder_rewrites_l2_copies_like_a_direct_fp8_tier() {
        let (ladder, cached, rewrites, l2_ladder) = ladder_run(true);
        let (direct, cached_direct, none, l2_direct) = ladder_run(false);
        assert!(rewrites >= 2.0, "both blocks were rewritten: {rewrites}");
        assert_eq!(none, 0.0, "no ladder, no rewrite");
        assert_eq!(
            (cached, cached_direct),
            (32, 32),
            "both reuse the L2 blocks"
        );
        assert_eq!(
            ladder, direct,
            "rewritten copies give the direct fp8 copies' tokens"
        );
        assert_eq!(l2_ladder, l2_direct, "L2 holds fp8 copies in both runs");
    }

    /// Queues `req` with an output channel of `capacity` events.
    fn submit_with(
        tx: &mpsc::Sender<EngineCommand>,
        req: GenerationRequest,
        capacity: usize,
    ) -> (mpsc::Receiver<GenerationEvent>, Admitted) {
        let (events, rx) = mpsc::channel(capacity);
        let (ack, admitted) = oneshot::channel();
        if tx
            .try_send(EngineCommand::Submit(Box::new(req.into()), events, ack))
            .is_err()
        {
            panic!("command channel full");
        }
        (rx, admitted)
    }

    fn submit(
        tx: &mpsc::Sender<EngineCommand>,
        req: GenerationRequest,
    ) -> (mpsc::Receiver<GenerationEvent>, Admitted) {
        submit_with(tx, req, EVENT_CHANNEL_CAPACITY)
    }

    /// The events already in a stream's channel.
    fn drain(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    fn internal_error(events: &[GenerationEvent]) -> bool {
        events.iter().any(|e| {
            matches!(
                e,
                GenerationEvent::Error {
                    code: ErrorCode::InternalError,
                    ..
                }
            )
        })
    }

    fn token_count(events: &[GenerationEvent]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, GenerationEvent::Token { .. }))
            .count()
    }

    /// The tiny Llama on the cpu reference provider with the device logits reduction (the
    /// executor overlap scheduling needs), for batches of `max_seqs` sequences.
    fn tiny_reducing_executor(spec: &TinySpec, max_seqs: u32) -> Box<dyn ModelExecutor> {
        let cfg = &spec.config;
        let mem = mem();
        let index = SafetensorsIndex::open(&spec.dir).unwrap();
        let weights =
            WeightLoader::load(&index, &llama_slots(cfg), &mem, MAX_STAGING_BYTES).expect("load");
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let mut reqs = executor::requirements(cfg, BLOCK_TOKENS, ExecutorOptions::default());
        reqs.push(executor::logits::reduce_requirement(cfg));
        let registry = KernelRegistry::build(
            vec![provider],
            &order,
            &reqs,
            &KernelMetrics::register(&MetricsRegistry::new()),
            None,
        )
        .unwrap();
        executor::build_executor(
            cfg,
            weights,
            Arc::new(registry),
            mem,
            BLOCK_TOKENS,
            64,
            max_seqs,
            ExecutorOptions::default(),
        )
        .expect("executor")
    }

    /// One choice's stream: its tokens, its text and how it ended.
    #[derive(Debug, Default, PartialEq)]
    struct ChoiceStream {
        tokens: Vec<u32>,
        logprobs: Vec<Option<f32>>,
        text: String,
        finish: Option<(FinishReason, u32)>,
    }

    /// Runs `requests` (all queued before the engine starts) on a fresh engine over the tiny
    /// Llama with the device reduction, with overlap scheduling on or off, and returns every
    /// request's choices and how many iterations launched ahead of one in flight.
    fn run_streams(
        spec: &TinySpec,
        tokenizer: &Arc<Tokenizer>,
        requests: &[GenerationRequest],
        overlap: bool,
    ) -> (Vec<Vec<ChoiceStream>>, u64) {
        let t = engine_with(
            tiny_reducing_executor(spec, 8),
            Arc::clone(tokenizer),
            params(4, 8),
            overlap,
        );
        assert_eq!(
            t.engine.overlap, overlap,
            "the reducing CPU executor can overlap"
        );
        let streams: Vec<_> = requests.iter().map(|r| submit(&t.tx, r.clone())).collect();
        let engine = t.engine;
        let handle = std::thread::spawn(move || {
            let mut engine = engine;
            let result = loop {
                let turn = if engine.overlap {
                    engine.turn_overlap()
                } else {
                    engine.turn()
                };
                match turn {
                    Ok(Turn::Continue) => {}
                    Ok(Turn::Stop) => break Ok(()),
                    Err(e) => break Err(e),
                }
            };
            (result, engine.overlapped)
        });
        let mut out = Vec::new();
        for ((mut rx, admitted), req) in streams.into_iter().zip(requests) {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            let mut choices: Vec<ChoiceStream> =
                (0..req.n).map(|_| ChoiceStream::default()).collect();
            let mut finished = 0;
            while finished < req.n {
                match rx.blocking_recv().expect("stream ended early") {
                    GenerationEvent::Token {
                        choice,
                        token_id,
                        text,
                        logprob,
                        ..
                    } => {
                        let c = &mut choices[choice as usize];
                        c.tokens.push(token_id);
                        c.logprobs.push(logprob);
                        c.text.push_str(&text);
                    }
                    GenerationEvent::Finished {
                        choice,
                        reason,
                        usage,
                    } => {
                        choices[choice as usize].finish =
                            Some((reason, usage.unwrap().completion_tokens));
                        finished += 1;
                    }
                    GenerationEvent::Started { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            out.push(choices);
        }
        drop(t.tx);
        let (result, overlapped) = handle.join().unwrap();
        assert_eq!(result, Ok(()));
        let docs = t.shared.docs().unwrap();
        assert_eq!(held_blocks(&docs), 0, "overlap {overlap}");
        assert_eq!(docs.scheduler.waiting, 0);
        (out, overlapped)
    }

    /// P2c overlap scheduling: with overlap on, every request gets exactly the stream it gets
    /// with overlap off — greedy requests (their decodes fed on the device), greedy with
    /// logprobs, a seeded draw (its token is the host's: those iterations wait), an `n` = 2
    /// request (its forks' first tokens are the host's), a chunked 20-token prompt, a stop
    /// string, an EOS the scheduler learns one iteration late and `max_tokens` from 1 to 12 —
    /// and iterations did launch ahead. Breaks if a fed token, a late finish, the sampler's
    /// uniform order or the scheduler's token accounting differs from the serial loop.
    #[test]
    fn overlap_scheduling_matches_serial() {
        let (_dir, spec, tokenizer) = tiny();
        let prompts: [Vec<u32>; 4] = [
            vec![256, 72, 101, 108, 108, 111],
            // 20 tokens: prefilled in chunks of 8.
            std::iter::once(256).chain(97..116).collect(),
            vec![256, 79],
            vec![256, 1, 2, 3],
        ];
        // A greedy run of prompt 0 alone, to pick a stop string and an EOS id it will meet.
        let (probe, _) = run_streams(&spec, &tokenizer, &[request(&prompts[0], 12)], false);
        let probe = &probe[0][0];
        assert_eq!(probe.tokens.len(), 12);
        let stop: String = probe.text.chars().skip(4).take(2).collect();
        assert!(
            !stop.is_empty(),
            "the probe produced text: {:?}",
            probe.text
        );
        let eos = probe.tokens[5];

        let mut requests = Vec::new();
        requests.push(request(&prompts[0], 12));
        let mut with_stop = request(&prompts[0], 12);
        with_stop.stop.stop_strings = vec![stop.clone()];
        requests.push(with_stop);
        let mut with_eos = request(&prompts[0], 12);
        with_eos.stop.eos_token_ids = [eos].into_iter().collect();
        with_eos.stop.ignore_eos = false;
        requests.push(with_eos);
        let mut logprobs = request(&prompts[1], 9);
        logprobs.sampling.logprobs = Some(3);
        requests.push(logprobs);
        let mut seeded = request(&prompts[2], 7);
        seeded.sampling = SamplingParams {
            temperature: 0.8,
            seed: Some(42),
            ..SamplingParams::default()
        };
        requests.push(seeded);
        let mut forked = request(&prompts[3], 5);
        forked.n = 2;
        requests.push(forked);
        requests.push(request(&prompts[2], 1));
        requests.push(request(&prompts[3], 10));

        let (serial, none) = run_streams(&spec, &tokenizer, &requests, false);
        assert_eq!(none, 0);
        let (overlapped, ahead) = run_streams(&spec, &tokenizer, &requests, true);
        assert!(ahead > 0, "no iteration launched ahead");
        assert_eq!(overlapped, serial);
        // The cases did what they are here for.
        assert_eq!(serial[1][0].finish.map(|f| f.0), Some(FinishReason::Stop));
        assert!(!serial[1][0].text.contains(&stop));
        let eos_at = probe.tokens.iter().position(|&t| t == eos).unwrap() as u32 + 1;
        assert_eq!(serial[2][0].finish, Some((FinishReason::Stop, eos_at)));
        assert!(serial[3][0].logprobs.iter().all(Option::is_some));
        assert_eq!(serial[5].len(), 2);
        assert_eq!(serial[6][0].finish, Some((FinishReason::Length, 1)));
    }

    /// Continuous batching with chunked prefill gives every request exactly the greedy tokens
    /// the Phase 1 single-request loop gives it alone.
    #[test]
    fn batched_greedy_matches_single_request_generate() {
        let (_dir, spec, tokenizer) = tiny();
        let prompts: [Vec<u32>; 3] = [
            vec![256, 72, 101, 108, 108, 111],
            // 20 tokens: prefilled in chunks of 8 over three iterations.
            std::iter::once(256).chain(97..116).collect(),
            vec![256, 79],
        ];
        let max_tokens = 12;

        let mut reference = Vec::new();
        for prompt in &prompts {
            let mut exec = tiny_executor(&spec, 1);
            let mut kv = SequenceKv::new(&mem(), *exec.kv_layout(), 128).unwrap();
            let req = request(prompt, max_tokens);
            let cancel = CancelFlag::default();
            let tokens: Vec<u32> = generate(
                exec.as_mut(),
                &mut kv,
                Arc::clone(&tokenizer),
                &req,
                &cancel,
                GenerateOptions {
                    max_seq_len: 128,
                    metrics: None,
                },
            )
            .filter_map(|e| match e {
                GenerationEvent::Token { token_id, .. } => Some(token_id),
                _ => None,
            })
            .collect();
            assert_eq!(tokens.len(), max_tokens as usize);
            reference.push(tokens);
        }

        let t = engine(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 8),
        );
        let streams: Vec<_> = prompts
            .iter()
            .map(|p| submit(&t.tx, request(p, max_tokens)))
            .collect();
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        let mut got = Vec::new();
        for (mut rx, admitted) in streams {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            let mut tokens = Vec::new();
            loop {
                match rx.blocking_recv().expect("stream ended early") {
                    GenerationEvent::Token { token_id, .. } => tokens.push(token_id),
                    GenerationEvent::Finished { reason, usage, .. } => {
                        assert_eq!(reason, FinishReason::Length);
                        assert_eq!(usage.unwrap().completion_tokens, max_tokens);
                        break;
                    }
                    GenerationEvent::Started { choice: 0 } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            got.push(tokens);
        }
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
        assert_eq!(got, reference);

        let docs = t.shared.docs().unwrap();
        assert_eq!(held_blocks(&docs), 0);
        assert_eq!(docs.scheduler.waiting, 0);
        assert!(docs.scheduler.iterations_total >= u64::from(max_tokens));
        let text = t.reg.render().unwrap();
        // The used gauge counts the blocks left cached for prefix reuse, as the document does.
        let used = format!(
            r#"turbine_kv_blocks{{tier="l0",state="used"}} {}"#,
            docs.kv.tiers[0].blocks_used
        );
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 3"#,
            used.as_str(),
            r#"turbine_tokens_total{kind="generated"} 36"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    /// P2 S-10: `n: 3` prefills the prompt once; the choices fork from it (the partial tail
    /// block copied) and each continues exactly like a single greedy request.
    #[test]
    fn n_choices_fork_after_one_prefill() {
        let (_dir, spec, tokenizer) = tiny();
        // 20 tokens: one full block and a partial tail block that the forks copy.
        let prompt: Vec<u32> = std::iter::once(256).chain(97..116).collect();
        let max_tokens = 12;
        let mut exec = tiny_executor(&spec, 1);
        let mut kv = SequenceKv::new(&mem(), *exec.kv_layout(), 128).unwrap();
        let single = request(&prompt, max_tokens);
        let cancel = CancelFlag::default();
        let reference: Vec<u32> = generate(
            exec.as_mut(),
            &mut kv,
            Arc::clone(&tokenizer),
            &single,
            &cancel,
            GenerateOptions {
                max_seq_len: 128,
                metrics: None,
            },
        )
        .filter_map(|e| match e {
            GenerationEvent::Token { token_id, .. } => Some(token_id),
            _ => None,
        })
        .collect();

        let t = engine(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 32),
        );
        let mut req = request(&prompt, max_tokens);
        req.n = 3;
        let (mut rx, admitted) = submit(&t.tx, req);
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let mut tokens = vec![Vec::new(); 3];
        let mut finished = 0;
        while finished < 3 {
            match rx.blocking_recv().expect("stream ended early") {
                GenerationEvent::Token {
                    choice, token_id, ..
                } => tokens[choice as usize].push(token_id),
                GenerationEvent::Finished { usage, .. } => {
                    assert_eq!(usage.unwrap().prompt_tokens, 20);
                    finished += 1;
                }
                GenerationEvent::Started { .. } => {}
                other => panic!("unexpected {other:?}"),
            }
        }
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
        for (choice, got) in tokens.iter().enumerate() {
            assert_eq!(got, &reference, "choice {choice}");
        }
        let docs = t.shared.docs().unwrap();
        assert_eq!(held_blocks(&docs), 0);
        let text = t.reg.render().unwrap();
        for line in [
            r#"turbine_tokens_total{kind="prompt"} 20"#,
            r#"turbine_tokens_total{kind="generated"} 36"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1"#,
        ] {
            assert!(
                text.contains(line),
                "missing {line:?} in
{text}"
            );
        }
    }

    /// A working executor whose `fail_launches`-th launches and `fail_collects`-th collects
    /// (1-based) fail like a (non-sticky) device error, and whose `oom_launches`-th launches
    /// fail with device out-of-memory.
    struct Flaky {
        inner: Box<dyn ModelExecutor>,
        launches: usize,
        collects: usize,
        fail_launches: Vec<usize>,
        fail_collects: Vec<usize>,
        oom_launches: Vec<usize>,
    }

    impl Flaky {
        fn device_error() -> ModelError {
            ModelError::Kernel(plain_device_error("injected"))
        }
    }

    impl ModelExecutor for Flaky {
        fn shape(&self) -> &ModelShape {
            self.inner.shape()
        }
        fn kv_layout(&self) -> &KvLayout {
            self.inner.kv_layout()
        }
        fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            self.inner.forward(batch)
        }
        fn reduces_logits(&self) -> bool {
            self.inner.reduces_logits()
        }
        fn overlaps(&self) -> bool {
            self.inner.overlaps()
        }
        fn launch(
            &mut self,
            batch: &BatchInput<'_>,
            feeds: &[TokenFeed],
        ) -> Result<(), ModelError> {
            self.launches += 1;
            if self.fail_launches.contains(&self.launches) {
                return Err(Flaky::device_error());
            }
            if self.oom_launches.contains(&self.launches) {
                return Err(ModelError::Kernel(KernelError::OutOfMemory {
                    message: "injected".into(),
                }));
            }
            self.inner.launch(batch, feeds)
        }
        fn collect(&mut self) -> Result<Logits, ModelError> {
            self.collects += 1;
            let logits = self.inner.collect();
            if self.fail_collects.contains(&self.collects) {
                return Err(Flaky::device_error());
            }
            logits
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

    /// Runs `prompts` (6 tokens each, one running request at a time) on an overlapping engine
    /// over `flaky` until every stream ends; returns each stream's events and the engine.
    fn run_flaky(
        flaky: Flaky,
        tokenizer: &Arc<Tokenizer>,
        prompts: &[Vec<u32>],
    ) -> (Vec<Vec<GenerationEvent>>, TestEngineDone) {
        let t = engine_with(Box::new(flaky), Arc::clone(tokenizer), params(1, 8), true);
        assert!(t.engine.overlap);
        let streams: Vec<_> = prompts
            .iter()
            .map(|p| submit(&t.tx, request(p, 6)))
            .collect();
        let (shared, engine) = (Arc::clone(&t.shared), t.engine);
        let handle = std::thread::spawn(move || engine.run());
        let mut outcomes = Vec::new();
        for (mut rx, admitted) in streams {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            outcomes.push(read_to_end(&mut rx));
        }
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
        (
            outcomes,
            TestEngineDone {
                shared,
                reg: t.reg,
                controller: t.controller,
            },
        )
    }

    /// What is left of a [`TestEngine`] after its engine ran.
    struct TestEngineDone {
        shared: Arc<EngineShared>,
        reg: MetricsRegistry,
        controller: ControllerHandle,
    }

    fn generated(events: &[GenerationEvent]) -> Vec<u32> {
        events
            .iter()
            .filter_map(|e| match e {
                GenerationEvent::Token { token_id, .. } => Some(*token_id),
                _ => None,
            })
            .collect()
    }

    /// Overlap scheduling after failed steps (P3: the circuit breaker replaces the Phase 2 exit
    /// after three failed iterations, CONFLICT C-25): a request in a step whose launch or whose
    /// collect fails with a (non-sticky) device error ends with `internal_error`, the circuit
    /// opens and the requests still in the admission queue are rejected `circuit_open`; a
    /// request that finished before keeps the tokens it gets when nothing fails, the engine
    /// keeps running, and no block stays used. Breaks if a failed step's sequences are never
    /// reported finished to the scheduler (their blocks leak, or the engine never idles).
    #[test]
    fn overlap_failed_steps_fail_their_requests_only() {
        let (_dir, spec, tokenizer) = tiny();
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| vec![256, 40 + i, 41 + i]).collect();
        let (want, _) = run_streams(
            &spec,
            &tokenizer,
            &prompts.iter().map(|p| request(p, 6)).collect::<Vec<_>>(),
            false,
        );
        // Each request is one prefill and five decodes: request 0 runs launches 1–6, request 1
        // launches 7–12 and fails at its third (launch 9, never collected).
        let flaky = Flaky {
            inner: tiny_reducing_executor(&spec, 8),
            launches: 0,
            collects: 0,
            fail_launches: vec![9],
            fail_collects: Vec::new(),
            oom_launches: Vec::new(),
        };
        let (outcomes, done) = run_flaky(flaky, &tokenizer, &prompts);
        assert_eq!(generated(&outcomes[0]), want[0][0].tokens);
        assert_eq!(error_code(&outcomes[1]), Some(ErrorCode::InternalError));
        assert_eq!(error_code(&outcomes[2]), Some(ErrorCode::CircuitOpen));
        assert_eq!(error_code(&outcomes[3]), Some(ErrorCode::CircuitOpen));
        assert!(done.controller.circuit().blocks_readiness());
        assert_eq!(held_blocks(&done.shared.docs().unwrap()), 0);
        let text = done.reg.render().unwrap();
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="failed"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="rejected"} 2"#,
            r#"turbine_circuit_transitions_total{from="HEALTHY",to="CIRCUIT_OPEN",reason="device_error"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }

        // A failed collect: request 0's third step (launch 4, fed from it, is dropped with it).
        let flaky = Flaky {
            inner: tiny_reducing_executor(&spec, 8),
            launches: 0,
            collects: 0,
            fail_launches: Vec::new(),
            fail_collects: vec![3],
            oom_launches: Vec::new(),
        };
        let (outcomes, done) = run_flaky(flaky, &tokenizer, &prompts);
        assert_eq!(error_code(&outcomes[0]), Some(ErrorCode::InternalError));
        for events in &outcomes[1..] {
            assert_eq!(error_code(events), Some(ErrorCode::CircuitOpen));
        }
        assert!(done.controller.circuit().blocks_readiness());
        assert_eq!(held_blocks(&done.shared.docs().unwrap()), 0);
    }

    /// P3 S-11 under overlap scheduling: a launch that fails with device out-of-memory
    /// finishes the iteration in flight, then runs the plan through the serial recovery path;
    /// the request keeps every token it gets when nothing fails, the recovery is counted, and
    /// the pressure state is SURVIVAL (the controller thread, absent here, de-escalates it).
    /// Breaks if an out-of-memory launch fails its request or leaks the in-flight iteration.
    #[test]
    fn overlap_oom_launch_recovers_serially() {
        let (_dir, spec, tokenizer) = tiny();
        let prompts = vec![vec![256, 40, 41]];
        let (want, _) = run_streams(
            &spec,
            &tokenizer,
            &prompts.iter().map(|p| request(p, 6)).collect::<Vec<_>>(),
            false,
        );
        let flaky = Flaky {
            inner: tiny_reducing_executor(&spec, 8),
            launches: 0,
            collects: 0,
            fail_launches: Vec::new(),
            fail_collects: Vec::new(),
            oom_launches: vec![3],
        };
        let (outcomes, done) = run_flaky(flaky, &tokenizer, &prompts);
        assert_eq!(error_code(&outcomes[0]), None, "{:?}", outcomes[0]);
        assert_eq!(generated(&outcomes[0]), want[0][0].tokens);
        assert_eq!(done.controller.state(), PressureState::Survival);
        assert_eq!(held_blocks(&done.shared.docs().unwrap()), 0);
        let text = done.reg.render().unwrap();
        for line in [
            r#"turbine_recoveries_total{outcome="recovered"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    /// Fails every forward like a device error (or, with `collective`, like a tensor-parallel
    /// group's failed collective), or panics.
    struct BrokenExecutor {
        shape: ModelShape,
        kv: KvLayout,
        panic: bool,
        collective: bool,
    }

    impl ModelExecutor for BrokenExecutor {
        fn shape(&self) -> &ModelShape {
            &self.shape
        }
        fn kv_layout(&self) -> &KvLayout {
            &self.kv
        }
        fn forward(&mut self, _batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            assert!(!self.panic, "injected engine panic");
            if self.collective {
                return Err(ModelError::Collective(
                    turbine_distributed::collective::CollectiveError::RemoteAbort { rank: 1 },
                ));
            }
            Err(ModelError::Kernel(plain_device_error("injected")))
        }
        fn copy_blocks(
            &mut self,
            _kv: &KvPoolView<'_>,
            _src: &[BlockId],
            _dst: &[BlockId],
        ) -> Result<(), ModelError> {
            Ok(())
        }
    }

    fn broken(spec: &TinySpec, panic: bool) -> Box<dyn ModelExecutor> {
        Box::new(BrokenExecutor {
            shape: spec.config.shape(),
            kv: spec.config.kv_layout(BLOCK_TOKENS),
            panic,
            collective: false,
        })
    }

    /// The events a stream receives until it ends (its sender dropped or an end event read).
    fn read_to_end(rx: &mut mpsc::Receiver<GenerationEvent>) -> Vec<GenerationEvent> {
        let mut events = Vec::new();
        while let Some(e) = rx.blocking_recv() {
            let end = matches!(
                e,
                GenerationEvent::Error { .. } | GenerationEvent::Finished { .. }
            );
            events.push(e);
            if end {
                break;
            }
        }
        events
    }

    fn error_code(events: &[GenerationEvent]) -> Option<ErrorCode> {
        events.iter().find_map(|e| match e {
            GenerationEvent::Error { code, .. } => Some(*code),
            _ => None,
        })
    }

    /// P3 S-12 (CONFLICT C-25 retires the Phase 2 exit after three failed iterations): a
    /// non-OOM device error fails the running request with `internal_error` and opens the
    /// circuit; the requests waiting in the admission queue are rejected `circuit_open`; the
    /// engine keeps running and no block stays used.
    #[test]
    fn device_error_opens_the_circuit() {
        let (_dir, spec, tokenizer) = tiny();
        // One running request at a time: the first runs, the other three wait in the queue.
        let t = engine(broken(&spec, false), tokenizer, params(1, 64));
        let streams: Vec<_> = (0..4)
            .map(|_| submit(&t.tx, request(&[256, 1, 2], 4)))
            .collect();
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        for (i, (mut rx, admitted)) in streams.into_iter().enumerate() {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            let expected = if i == 0 {
                ErrorCode::InternalError
            } else {
                ErrorCode::CircuitOpen
            };
            assert_eq!(
                error_code(&read_to_end(&mut rx)),
                Some(expected),
                "request {i}"
            );
        }
        assert!(t.controller.circuit().blocks_readiness());
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
        assert_eq!(held_blocks(&t.shared.docs().unwrap()), 0);
        let text = t.reg.render().unwrap();
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="failed"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="rejected"} 3"#,
            r#"turbine_circuit_transitions_total{from="HEALTHY",to="CIRCUIT_OPEN",reason="device_error"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    /// A failed iteration opens the circuit before its requests hear of the failure: the
    /// moment a client reads its `replica_failed` (collective) or `internal_error` (device) event,
    /// the circuit already blocks readiness (`/ready` 503 `circuit_open`). Breaks if the engine
    /// answers the requests first, which lets a client see `/ready` 200 after its error.
    #[test]
    fn circuit_opens_before_the_failed_request_hears() {
        for collective in [true, false] {
            let (_dir, spec, tokenizer) = tiny();
            let exec = Box::new(BrokenExecutor {
                shape: spec.config.shape(),
                kv: spec.config.kv_layout(BLOCK_TOKENS),
                panic: false,
                collective,
            });
            let t = engine(exec, tokenizer, params(1, 64));
            let (mut rx, admitted) = submit(&t.tx, request(&[256, 1, 2], 4));
            let controller = t.controller.clone();
            let engine = t.engine;
            let handle = std::thread::spawn(move || engine.run());
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            loop {
                match rx.blocking_recv().expect("an error event") {
                    GenerationEvent::Error { code, .. } => {
                        let want = if collective {
                            ErrorCode::ReplicaFailed
                        } else {
                            ErrorCode::InternalError
                        };
                        assert_eq!(code, want);
                        assert!(
                            controller.circuit().blocks_readiness(),
                            "collective {collective}: the circuit is open when the error arrives"
                        );
                        break;
                    }
                    _ => continue,
                }
            }
            drop(t.tx);
            assert_eq!(handle.join().unwrap(), Ok(()));
        }
    }

    /// `n` > 1 on an executor sized for `max_running_requests` sequences: two running requests,
    /// one with 3 choices, hold 4 decoding sequences on an executor built for 2; the scheduler
    /// steps at most 2 per iteration, in turn, and both requests complete every choice. Breaks
    /// if a step hands the executor more sequences than it holds ("sequences exceed max_seqs").
    #[test]
    fn choices_beyond_the_executor_batch_take_turns() {
        let (_dir, spec, tokenizer) = tiny();
        let t = engine(tiny_executor(&spec, 2), tokenizer, params(2, 64));
        let mut three = request(&[256, 1, 2, 3], 6);
        three.n = 3;
        let streams = [
            submit(&t.tx, three),
            submit(&t.tx, request(&[256, 4, 5], 6)),
        ];
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        for (i, (mut rx, admitted)) in streams.into_iter().enumerate() {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            // Every choice ends with its own event: read until the request drops its channel.
            let events: Vec<GenerationEvent> = std::iter::from_fn(|| rx.blocking_recv()).collect();
            assert_eq!(error_code(&events), None, "request {i}: {events:?}");
            let tokens = events
                .iter()
                .filter(|e| matches!(e, GenerationEvent::Token { .. }))
                .count();
            assert_eq!(tokens, if i == 0 { 18 } else { 6 }, "request {i}");
        }
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
    }

    /// P5 S-6: a tensor-parallel group's failed collective (a rank aborted) ends the running
    /// request with `replica_failed`, opens the circuit with reason `collective_failed` (queued
    /// requests are rejected `circuit_open`), keeps the engine running and frees every block.
    /// Breaks if a collective failure is reported as a plain device error or turns fatal.
    #[test]
    fn collective_failure_ends_requests_with_replica_failed() {
        let (_dir, spec, tokenizer) = tiny();
        let exec = Box::new(BrokenExecutor {
            shape: spec.config.shape(),
            kv: spec.config.kv_layout(BLOCK_TOKENS),
            panic: false,
            collective: true,
        });
        let t = engine(exec, tokenizer, params(1, 64));
        let streams: Vec<_> = (0..3)
            .map(|_| submit(&t.tx, request(&[256, 1, 2], 4)))
            .collect();
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        for (i, (mut rx, admitted)) in streams.into_iter().enumerate() {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            let expected = if i == 0 {
                ErrorCode::ReplicaFailed
            } else {
                ErrorCode::CircuitOpen
            };
            assert_eq!(
                error_code(&read_to_end(&mut rx)),
                Some(expected),
                "request {i}"
            );
        }
        assert!(t.controller.circuit().blocks_readiness());
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()), "not fatal: no exit 3");
        assert_eq!(held_blocks(&t.shared.docs().unwrap()), 0);
        let text = t.reg.render().unwrap();
        let line = r#"turbine_circuit_transitions_total{from="HEALTHY",to="CIRCUIT_OPEN",reason="collective_failed"} 1"#;
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }

    /// Reads a stream to its end: its tokens, and whether it finished (else it failed).
    fn tokens_of(rx: &mut mpsc::Receiver<GenerationEvent>) -> (Vec<u32>, Option<ErrorCode>) {
        let events = read_to_end(rx);
        (generated(&events), error_code(&events))
    }

    /// P5 S-10 in the engine loop (plan Task 24): the tiny Llama as a 2-stage pipeline (stage 0
    /// on its own thread, the last on the engine thread, over the host collective) with 1 and
    /// 2 micro-batches serves 4 concurrent greedy requests with exactly one device's tokens,
    /// frees every block, and publishes the scheduler document's `pipeline` section (2 stages
    /// with their layers, the micro-batch count). Breaks if a micro-batch reuses a sequence in
    /// flight, a token is sampled from another micro-batch's logits or blocks leak.
    #[test]
    fn pipeline_micro_batches_serve_one_device_tokens() {
        use super::super::pp::testing::{pipeline, stats};
        let (_dir, spec, tokenizer) = tiny();
        let prompts: [Vec<u32>; 4] = [
            vec![256, 72, 101, 108, 108, 111],
            std::iter::once(256).chain(97..116).collect(),
            vec![256, 79],
            vec![256, 1, 2, 3, 4],
        ];
        let run = |exec: Box<dyn ModelExecutor>, m: Option<Arc<PipelineStats>>| {
            let t = engine_with_pipeline(
                exec,
                Arc::clone(&tokenizer),
                params(4, 8),
                false,
                KvConfig::default(),
                |_, _, _| None,
                m,
            );
            let streams: Vec<_> = prompts
                .iter()
                .map(|p| submit(&t.tx, request(p, 10)))
                .collect();
            let engine = t.engine;
            let handle = std::thread::spawn(move || engine.run());
            let mut got = Vec::new();
            for (mut rx, admitted) in streams {
                assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
                let (tokens, error) = tokens_of(&mut rx);
                assert_eq!(error, None);
                got.push(tokens);
            }
            drop(t.tx);
            assert_eq!(handle.join().unwrap(), Ok(()));
            let docs = t.shared.docs().unwrap();
            assert_eq!(held_blocks(&docs), 0, "every block freed");
            (got, docs)
        };
        let (want, _) = run(tiny_executor(&spec, 4), None);
        assert!(want.iter().all(|t| t.len() == 10));
        for m in [1, 2] {
            let s = stats(m);
            let (exec, _pool) = pipeline(&spec, Arc::clone(&s), 64, |e| e);
            let (got, docs) = run(Box::new(exec), Some(s));
            assert_eq!(got, want, "micro-batches {m}");
            let p = docs.scheduler.pipeline.expect("the pipeline section");
            assert_eq!((p.micro_batches, p.stages.len()), (m, 2));
            assert_eq!(p.stages[0].layers, [0, 0]);
        }
    }

    /// P5 S-10 failure mode "pipeline stage fails": stage 0 fails, so every request with a
    /// micro-batch in the pipeline ends with `replica_failed`, the circuit opens with reason
    /// `collective_failed` (later requests are rejected `circuit_open`), the engine keeps
    /// running and no block stays used. Breaks if a stage failure passes for a device error,
    /// hangs the engine or leaks the micro-batches' blocks.
    #[test]
    fn pipeline_stage_failure_ends_requests_with_replica_failed() {
        use super::super::pp::testing::{FailingStage, pipeline, stats};
        let (_dir, spec, tokenizer) = tiny();
        let s = stats(2);
        let (exec, _pool) = pipeline(&spec, Arc::clone(&s), 64, |inner| {
            Box::new(FailingStage {
                inner,
                sticky: false,
            }) as Box<dyn ModelExecutor>
        });
        let t = engine_with_pipeline(
            Box::new(exec),
            tokenizer,
            params(2, 64),
            false,
            KvConfig::default(),
            |_, _, _| None,
            Some(s),
        );
        let streams: Vec<_> = (0..3)
            .map(|_| submit(&t.tx, request(&[256, 1, 2], 4)))
            .collect();
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        let mut codes = Vec::new();
        for (mut rx, admitted) in streams {
            assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
            codes.push(tokens_of(&mut rx).1);
        }
        // Both running requests were in the pipeline (one micro-batch each); the third waited.
        assert_eq!(
            codes,
            [
                Some(ErrorCode::ReplicaFailed),
                Some(ErrorCode::ReplicaFailed),
                Some(ErrorCode::CircuitOpen)
            ]
        );
        assert!(t.controller.circuit().blocks_readiness());
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()), "not fatal");
        assert_eq!(held_blocks(&t.shared.docs().unwrap()), 0);
        let text = t.reg.render().unwrap();
        let line = r#"turbine_circuit_transitions_total{from="HEALTHY",to="CIRCUIT_OPEN",reason="collective_failed"} 1"#;
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }

    /// An engine panic is caught: every in-flight request gets `internal_error`, the circuit
    /// turns fatal and the engine reports the panic (the server exits 3).
    #[test]
    fn engine_panic_fails_every_request() {
        let (_dir, spec, tokenizer) = tiny();
        let t = engine(broken(&spec, true), tokenizer, params(4, 64));
        let mut streams: Vec<_> = (0..2)
            .map(|_| submit(&t.tx, request(&[256, 1, 2], 4)))
            .collect();
        let err = t.engine.run().unwrap_err();
        assert!(
            err.contains("engine thread panicked: injected engine panic"),
            "{err}"
        );
        for (rx, _) in &mut streams {
            assert!(internal_error(&drain(rx)));
        }
        assert!(t.controller.snapshot().fatal);
    }

    /// A stream that stops reading pauses its request (events held, KV kept) without stalling
    /// the others; reading on resumes it with every token once, in order; a dropped stream is
    /// cancelled with `client_disconnect` and its blocks are freed.
    #[test]
    fn full_channel_pauses_and_closed_channel_cancels() {
        let (_dir, spec, tokenizer) = tiny();
        let t = engine(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 64),
        );
        // Four slots, one reserved for a closing event: `Started` and two tokens fill the slow
        // stream's channel.
        let (mut slow, admitted) = submit_with(&t.tx, request(&[256, 1, 2], 40), 4);
        let (dropped, _dropped_admitted) = submit(&t.tx, request(&[256, 3], 100));
        let (mut fast, _fast_admitted) = submit(&t.tx, request(&[256, 4], 20));
        let (shared, reg) = (Arc::clone(&t.shared), t.reg.clone());
        let engine = t.engine;
        let handle = std::thread::spawn(move || engine.run());
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));

        let fast_events: Vec<_> = std::iter::from_fn(|| fast.blocking_recv()).collect();
        assert_eq!(token_count(&fast_events), 20);
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.docs().unwrap().scheduler.paused != 1 {
            assert!(Instant::now() < deadline, "the slow request never paused");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(held_blocks(&shared.docs().unwrap()) > 0);
        let text = reg.render().unwrap();
        assert!(text.contains("turbine_stream_paused_total 1"), "{text}");

        drop(dropped);
        let slow_events: Vec<_> = std::iter::from_fn(|| slow.blocking_recv()).collect();
        assert_eq!(token_count(&slow_events), 40);
        assert!(matches!(
            slow_events.last(),
            Some(GenerationEvent::Finished {
                reason: FinishReason::Length,
                ..
            })
        ));
        drop(t.tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
        assert_eq!(held_blocks(&shared.docs().unwrap()), 0);
        let text = reg.render().unwrap();
        for line in [
            r#"turbine_requests_cancelled_total{reason="client_disconnect"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 2"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="cancelled"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    /// P6b SURVIVAL at 20+ multi-turn sessions: a request that attaches a cached prefix holds
    /// its blocks without a reservation for them (the reservation leaves the attached prefix
    /// out), so the ledger must still count them. A 100-token prompt runs once (6 full
    /// 16-token blocks cached), then again with a stream that stops reading, so it pauses
    /// holding the 6 attached blocks plus its tail. The ledger's kv `used` + `reserved` then
    /// covers every block the pool holds. Breaks if the engine does not report the pool's
    /// referenced blocks to the ledger: the ledger showed 2 of the 7 held blocks, and on the lab
    /// L0 filled to 585 of 585 blocks while `kv_utilization` read 0.70, until admitted decodes
    /// waited for a block (`decode_deferred`) and the horizon jumped GREEN → SURVIVAL.
    #[test]
    fn attached_prefix_blocks_count_in_the_ledger() {
        let (_dir, spec, tokenizer) = tiny();
        let t = engine(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 64),
        );
        let block_bytes = t.engine.pool.layout().block_bytes() as f64;
        let prompt: Vec<u32> = std::iter::once(256).chain(1..100).collect();
        assert_eq!(prompt.len(), 100);
        let TestEngine {
            engine,
            tx,
            shared,
            reg,
            ..
        } = t;
        let handle = std::thread::spawn(move || engine.run());
        let (_, cached) = run_one(&tx, request(&prompt, 8));
        assert_eq!(cached, 0);
        // `Started` and two tokens fill the four slots; the request pauses with its KV.
        let (mut slow, admitted) = submit_with(&tx, request(&prompt, 20), 4);
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.docs().unwrap().scheduler.paused != 1 {
            assert!(Instant::now() < deadline, "the request never paused");
            std::thread::sleep(Duration::from_millis(5));
        }
        let held = held_blocks(&shared.docs().unwrap()) as f64;
        assert!(held >= 7.0, "6 attached blocks and a tail: {held}");
        let gauge = |kind: &str| {
            metric(
                &reg,
                &format!(r#"turbine_memory_pool_bytes{{device="0",pool="kv",kind="{kind}"}}"#),
            )
        };
        let ledger_blocks = (gauge("used") + gauge("reserved")) / block_bytes;
        assert!(
            ledger_blocks >= held,
            "the ledger counts {ledger_blocks} blocks, the pool holds {held}"
        );
        let events: Vec<_> = std::iter::from_fn(|| slow.blocking_recv()).collect();
        assert_eq!(token_count(&events), 20);
        drop(tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
    }

    /// A telemetry sample at `clock`'s now with KV utilisation `kv` (host and device calm).
    fn calm_sample(clock: &Arc<dyn Clock>, kv: f64) -> TelemetrySample {
        TelemetrySample {
            at_mono_ns: clock.now_mono().as_nanos() as u64,
            host: HostSample {
                mem_available_bytes: Some(64 << 30),
                swap_total_bytes: Some(0),
                swap_free_bytes: Some(0),
                pswpin_total: Some(0),
                psi_memory_some_avg10: Some(0.0),
                status: SourceStatus::Ok,
            },
            // No device memory figure: the test budget is only the 64-block pool.
            devices: vec![DeviceSample {
                temperature_c: Some(40.0),
                slowdown_temperature_c: Some(90.0),
                clock_mhz: Some(2350),
                ..DeviceSample::empty(DeviceId(0), SourceStatus::Ok)
            }],
            ledger: LedgerSample {
                kv_utilization: kv,
                queue_fill: 0.0,
            },
            storage: None,
        }
    }

    /// One controller tick at KV utilisation `kv` (the test is the controller thread).
    fn tick(t: &(Arc<std::sync::Mutex<PressureController>>, Arc<dyn Clock>), kv: f64) {
        let stats = EngineStats {
            block_tokens: BLOCK_TOKENS,
            free_kv_blocks: 64,
            ..EngineStats::default()
        };
        t.0.lock().unwrap().tick(&calm_sample(&t.1, kv), &stats);
    }

    /// Queued-prefix demotion on the cpu backend (decisions "6b: after the held-prefix ledger
    /// fix", 1 B, and "6b: queued-prefix demotion — granularity and scope", 1 A, 2 A): a request
    /// waiting in the admission queue behind its head holds its 6 attached blocks at GREEN, even when the
    /// controller asks for more than L0 holds unreferenced. At YELLOW the reclaim's shortfall
    /// makes it release them: they leave the referenced blocks and the next reclaim demotes them
    /// to L2. Admitted once the running request ends, it attaches again through the planner (L2
    /// or recompute) and yields the cold run's greedy tokens. Breaks if a queued prefix is
    /// released at GREEN, never released at YELLOW, not demoted afterwards, or if the released
    /// request never re-attaches (it would wait forever) or produces other tokens.
    #[test]
    fn yellow_releases_queued_prefixes_and_they_reattach() {
        let (dir, spec, tokenizer) = tiny();
        let mut kv = KvConfig::default();
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = ByteSize(16 << 20);
        kv.nvme.slab_bytes = ByteSize(1 << 20);
        let mut config = ReliabilityConfig {
            emergency_vram_reserve: ByteSize(0),
            ..ReliabilityConfig::default()
        };
        // YELLOW from 1 % KV utilisation: its reclaim target (1 % of 64 blocks) is below what
        // the requests hold, so the reclaim always falls short.
        config.pressure.thresholds.insert(
            PressureSignal::KvUtilization,
            [Some(0.01), Some(0.9), Some(0.95), Some(0.99)],
        );
        let t = engine_full(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(1, 64),
            false,
            kv,
            |cfg, format, metrics| {
                crate::kv_orchestrator::open_l2(
                    cfg,
                    format,
                    &ModelIdentity::from_bytes(b"tiny config", b"tiny index"),
                    Arc::new(SystemClock::new()),
                    metrics,
                )
                .expect("L2 opens in the temp directory")
            },
            None,
            config,
        );
        let reclaim = t.engine.kv.reclaimer();
        let ctl = (Arc::clone(&t.pressure), Arc::clone(&t.clock));
        let TestEngine {
            engine,
            tx,
            shared,
            reg,
            controller,
            ..
        } = t;
        let handle = std::thread::spawn(move || engine.run());
        let detached = "turbine_kv_queued_prefix_detached_blocks_total";
        let demoted = r#"turbine_kv_demotions_total{from="l0",to="l2"}"#;

        let prompt: Vec<u32> = std::iter::once(256).chain(1..100).collect();
        let (cold, _) = run_one(&tx, request(&prompt, 8));
        // R holds the only running slot: paused behind a full channel, with its KV.
        let other: Vec<u32> = std::iter::once(256).chain(150..189).collect();
        let (mut slow, admitted) = submit_with(&tx, request(&other, 20), 4);
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.docs().unwrap().scheduler.paused != 1 {
            assert!(Instant::now() < deadline, "R never paused");
            std::thread::sleep(Duration::from_millis(5));
        }
        // The admission queue's head H (it keeps whatever it holds), then Q with the prompt's 6
        // cached blocks attached.
        let (mut head, admitted) = submit(&tx, request(&[256, 5, 6, 7], 4));
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let (mut queued, admitted) = submit(&tx, request(&prompt, 8));
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.docs().unwrap().scheduler.waiting != 2 {
            assert!(Instant::now() < deadline, "H and Q are not both queued");
            std::thread::sleep(Duration::from_millis(5));
        }
        let held = held_blocks(&shared.docs().unwrap());
        assert!(held >= 9, "R's 3 blocks and Q's 6: {held}");

        // GREEN: a reclaim that falls short releases nothing.
        reclaim.demote(0.01);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(metric(&reg, detached), 0.0, "nothing released at GREEN");
        assert_eq!(held_blocks(&shared.docs().unwrap()), held);

        // YELLOW: the controller's reclaim falls short, Q's blocks are released and demoted.
        for _ in 0..3 {
            tick(&ctl, 0.5);
        }
        assert_eq!(controller.state(), PressureState::Yellow);
        let deadline = Instant::now() + Duration::from_secs(20);
        while metric(&reg, detached) < 6.0 || metric(&reg, demoted) < 6.0 {
            assert!(
                Instant::now() < deadline,
                "Q's prefix was not released and demoted: {} released, {} demoted",
                metric(&reg, detached),
                metric(&reg, demoted)
            );
            tick(&ctl, 0.5);
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(metric(&reg, detached), 6.0, "only Q's own blocks");
        assert_eq!(
            held_blocks(&shared.docs().unwrap()),
            held - 6,
            "the released blocks left the referenced ones"
        );

        // R and H end; Q is admitted, attaches again (from L2 or recomputed) and matches the
        // cold run.
        let events: Vec<_> = std::iter::from_fn(|| slow.blocking_recv()).collect();
        assert_eq!(token_count(&events), 20);
        let events: Vec<_> = std::iter::from_fn(|| head.blocking_recv()).collect();
        assert_eq!(token_count(&events), 4);
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut events = Vec::new();
        loop {
            match queued.try_recv() {
                Ok(e) => events.push(e),
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {
                    assert!(Instant::now() < deadline, "Q never ran: {events:?}");
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        assert!(!internal_error(&events), "{events:?}");
        assert_eq!(generated(&events), cold, "Q yields the cold run's tokens");
        drop(tx);
        assert_eq!(handle.join().unwrap(), Ok(()));
    }

    /// P2 S-7 slow client, deterministic on a fake clock: a stream that stops reading pauses,
    /// is cancelled with `slow_client` one `server.slow_client_timeout` later (its error event
    /// held behind the full channel), and when the client still reads nothing for another
    /// timeout the engine closes the stream. The client reading on afterwards sees what the
    /// channel buffered, then the `slow_client` error, then the end of the stream — never the
    /// bare close the API turns into `internal_error`. Breaks if closing a finished request
    /// drops its terminal event (the tiny_server `slow_client_paused_then_cancelled` failure on
    /// a loaded host) or if the cancellation is not accounted as `slow_client`.
    #[test]
    fn slow_client_closed_after_cancel_ends_with_slow_client() {
        let (_dir, spec, tokenizer) = tiny();
        let mut t = engine(
            tiny_executor(&spec, 4),
            Arc::clone(&tokenizer),
            params(4, 64),
        );
        let clock = FakeClock::new(Duration::from_secs(100));
        let timeout = Duration::from_secs(1);
        t.engine.deadlines = Deadlines::new(
            Arc::new(clock.clone()),
            Timeouts {
                request: Duration::from_secs(600),
                slow_client: timeout,
            },
        );
        let req = request(&[256, 1, 2], 40);
        let id = req.id;
        let (mut slow, admitted) = submit_with(&t.tx, req, 4);
        let mut engine = t.engine;
        let mut turns = 0;
        while !engine
            .requests
            .get(&id)
            .is_some_and(ActiveRequest::has_held)
        {
            assert!(matches!(engine.turn(), Ok(Turn::Continue)));
            turns += 1;
            assert!(turns < 100, "the slow request never paused");
        }
        assert_eq!(admitted.blocking_recv().unwrap(), Ok(()));

        // One timeout paused: cancelled, its error event held behind the full channel.
        clock.advance(timeout);
        assert!(matches!(engine.turn(), Ok(Turn::Continue)));
        assert!(
            engine
                .requests
                .get(&id)
                .is_some_and(|r| r.done && r.has_held())
        );
        // Another timeout unread: the engine lets the request go and closes the stream.
        clock.advance(timeout);
        assert!(matches!(engine.turn(), Ok(Turn::Continue)));
        assert!(!engine.requests.contains_key(&id), "the stream is closed");

        let events: Vec<_> = std::iter::from_fn(|| slow.try_recv().ok()).collect();
        assert!(token_count(&events) > 0, "{events:?}");
        assert!(
            matches!(
                events.last(),
                Some(GenerationEvent::Error {
                    code: ErrorCode::SlowClient,
                    ..
                })
            ),
            "{events:?}"
        );
        assert_eq!(slow.try_recv().err(), Some(TryRecvError::Disconnected));
        let text = t.reg.render().unwrap();
        for line in [
            r#"turbine_requests_cancelled_total{reason="slow_client"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="cancelled"} 1"#,
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    /// Scheduler refusals come back on the admission answer before any event.
    #[test]
    fn refused_submissions_answer_before_any_event() {
        let (_dir, spec, tokenizer) = tiny();
        let t = engine(broken(&spec, false), tokenizer, params(4, 64));
        // 100 prompt + 40 new tokens > max_seq_len 128.
        let (mut rx, admitted) = submit(&t.tx, request(&[1; 100], 40));
        let (mut shut, shut_admitted) = {
            let (events, rx) = mpsc::channel(4);
            let (ack, admitted) = oneshot::channel();
            if t.tx.try_send(EngineCommand::Shutdown).is_err()
                || t.tx
                    .try_send(EngineCommand::Submit(
                        Box::new(request(&[1, 2], 4).into()),
                        events,
                        ack,
                    ))
                    .is_err()
            {
                panic!("command channel full");
            }
            (rx, admitted)
        };
        drop(t.tx);
        assert_eq!(t.engine.run(), Ok(()));
        assert_eq!(
            admitted.blocking_recv().unwrap(),
            Err(SubmitError::ContextLengthExceeded)
        );
        assert_eq!(
            shut_admitted.blocking_recv().unwrap(),
            Err(SubmitError::ShuttingDown)
        );
        assert!(drain(&mut rx).is_empty());
        assert!(drain(&mut shut).is_empty());
    }
}
