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
//! Every turn is timed in the eight stages of [`Stage`] (P2c S-1) by marks around these steps;
//! an executed iteration records them in `turbine_engine_iteration_seconds{stage}`, and every
//! published scheduler document carries the last turn's `stages_ms`.
//!
//! A failed iteration fails every request in it with `internal_error`; three in a row stop the
//! engine (CONFLICT C-25). A panic inside a turn is caught, logged with the iteration's request
//! ids, and fails every request.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use tokio::sync::mpsc::{self, error::TryRecvError};
use turbine_core::clock::Clock;
use turbine_core::request::{ErrorCode, FinishReason, GenerationEvent};
use turbine_core::types::{RequestId, SeqId};
use turbine_kv::{BlockPool, KvDocument};
use turbine_model::executor::{BatchInput, Logits, ModelExecutor, SeqSlice};
use turbine_model::{ForwardPhase, SampleJob, SampledToken, Tokenizer, sample_rows};
use turbine_scheduler::{
    BatchKind, CancelReason, IterationFailure, IterationLimits, IterationOutcome, IterationPlan,
    SchedRequest, Scheduler, SubmitError,
};

use super::deadlines::{Deadlines, Timeouts};
use super::requests::{ActiveRequest, Delivery, Flush, Submission};
use super::stages::{Stage, StageClock};
use super::{
    EngineCommand, EngineDocs, EngineMetrics, EngineShared, MAX_CONSECUTIVE_FAILURES, SubmitAck,
};
use crate::metrics::{Outcome, TokenKind};

/// Sleep between turns while requests exist but the last plan had nothing to run.
const IDLE_POLL: Duration = Duration::from_millis(2);

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
}

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
    consecutive_failures: u32,
    /// Requests of the iteration being executed (named in the log if it panics).
    iteration_requests: Vec<RequestId>,
    shutting_down: bool,
    /// The previous plan had nothing to execute.
    idle_turn: bool,
    /// Times the current turn's stages.
    stages: StageClock,
}

impl EngineLoop {
    /// The engine over `p`; its diagnostics documents are published at once.
    pub fn new(p: EngineParts) -> EngineLoop {
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
            consecutive_failures: 0,
            iteration_requests: Vec::new(),
            shutting_down: false,
            idle_turn: false,
            stages: StageClock::start(),
        };
        engine.publish(false);
        engine
    }

    /// Serves until the command channel closes or `Shutdown` completes. `Err` carries the
    /// reason the server must exit 1: consecutive failed iterations or a panic.
    pub fn run(mut self) -> Result<(), String> {
        loop {
            let turn = catch_unwind(AssertUnwindSafe(|| self.turn()));
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
                    format!("engine thread panicked: {what}")
                }
            };
            self.fail_all(&message);
            return Err(message);
        }
    }

    fn turn(&mut self) -> Result<Turn, String> {
        if !self.receive_commands() {
            return Ok(Turn::Stop);
        }
        self.stages.mark(Stage::Schedule);
        self.flush_outputs();
        self.stages.mark(Stage::Emit);
        self.detect_disconnects();
        self.expire_deadlines();

        let plan = self.sched.plan(&mut self.pool, &IterationLimits::default());
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
            self.idle_turn = true;
            self.publish(false);
            return Ok(Turn::Continue);
        }
        self.idle_turn = false;
        self.iteration_requests = plan_requests(&plan, &self.seqs);
        let outcome = self.execute(&plan);
        let failure = outcome.failed.clone();
        self.sched.complete(&mut self.pool, outcome);
        self.publish(true);
        match failure {
            None => self.consecutive_failures = 0,
            Some(f) => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(format!(
                        "{} consecutive iterations failed; last error: {}",
                        self.consecutive_failures, f.message
                    ));
                }
            }
        }
        Ok(Turn::Continue)
    }

    /// Nothing queued, running or waiting to be delivered.
    fn quiet(&self) -> bool {
        self.sched.is_idle() && self.requests.is_empty()
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
        tracing::info!(event = "engine_shutdown", "engine shutting down");
    }

    /// Scheduler submission checks, then the request is queued and its choices start.
    fn submit(
        &mut self,
        submission: Submission,
        events: mpsc::Sender<GenerationEvent>,
        ack: SubmitAck,
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
        if !zero_tokens {
            let mut r = SchedRequest::new(
                id,
                seqs.clone(),
                prompt_len,
                request.stop.max_tokens,
                self.pool.layout().block_tokens,
            );
            r.priority = request.priority;
            r.arrival = self.clock.now_mono();
            r.constrained = request.constraint.is_some();
            if let Err(e) = self.sched.submit(r, self.pool.total_blocks()) {
                let _ = ack.send(Err(e));
                return;
            }
        }
        let submitter_gone = ack.send(Ok(())).is_err();
        self.metrics
            .server
            .add_tokens(TokenKind::Prompt, u64::from(prompt_len));
        let active = ActiveRequest::new(submission, events, &seqs, &self.tokenizer);
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
    /// events stay unread past `server.slow_client_timeout` is dropped, which closes its stream.
    fn expire_deadlines(&mut self) {
        for (id, reason) in self.deadlines.expired() {
            match self.requests.get(&id) {
                Some(r) if r.done => {
                    if reason == CancelReason::SlowClient {
                        tracing::info!(event = "cancel", request_id = %id.0, reason = reason.as_str(), "closing a stream whose final events stay unread");
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
                    "the request waited longer than scheduler.queue_timeout",
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
                Some((
                    ErrorCode::SlowClient,
                    "the client did not read the stream within server.slow_client_timeout",
                )),
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

    /// Steps 5 and 6: returns the outcome for `Scheduler::complete`.
    fn execute(&mut self, plan: &IterationPlan) -> IterationOutcome {
        let mut outcome = IterationOutcome {
            iteration: plan.iteration,
            ..IterationOutcome::default()
        };
        let (src, dst): (Vec<_>, Vec<_>) = plan.forks.iter().filter_map(|f| f.copy).unzip();
        if !src.is_empty()
            && let Err(e) = self.exec.copy_blocks(&self.pool.view(), &src, &dst)
        {
            self.fail_iteration(format!("copy_blocks failed: {e}"), &mut outcome);
            return outcome;
        }
        self.stages.mark(Stage::Prepare);
        if plan.items.is_empty() {
            return outcome;
        }
        let logits = match self.forward(plan) {
            Ok(logits) => logits,
            Err(message) => {
                self.fail_iteration(message, &mut outcome);
                return outcome;
            }
        };
        self.sample(plan, logits, &mut outcome);
        outcome
    }

    /// Packs the plan's items into one ragged batch and runs the forward pass.
    fn forward(&mut self, plan: &IterationPlan) -> Result<Logits, String> {
        let total: usize = plan
            .items
            .iter()
            .map(|i| match i.kind {
                BatchKind::Prefill { len, .. } => len as usize,
                BatchKind::Decode => 1,
            })
            .sum();
        let mut tokens = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        let mut slices = Vec::with_capacity(plan.items.len());
        for item in &plan.items {
            let kv_len = item.block_table.tokens;
            let (start, len) = match item.kind {
                BatchKind::Prefill { start, len } => (start, len),
                BatchKind::Decode => (kv_len.saturating_sub(1), 1),
            };
            let (id, choice) = *self
                .seqs
                .get(&item.seq)
                .ok_or_else(|| format!("sequence {} has no request", item.seq.0))?;
            let r = self
                .requests
                .get(&id)
                .ok_or_else(|| format!("request {} is not tracked", id.0))?;
            let q_start = tokens.len() as u32;
            for p in start..start + len {
                tokens.push(r.token_at(choice, p).ok_or_else(|| {
                    format!("sequence {} has no token at position {p}", item.seq.0)
                })?);
                positions.push(p);
            }
            slices.push(SeqSlice {
                seq: item.seq,
                q_start,
                q_len: len,
                kv_len,
                block_table: &item.block_table.blocks,
            });
        }
        let phase = if plan.prefill_tokens() > 0 {
            ForwardPhase::Prefill
        } else {
            ForwardPhase::Decode
        };
        let started = Instant::now();
        let view = self.pool.view();
        let result = self.exec.forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &slices,
            kv: &view,
        });
        if result.is_ok() {
            let t = self.exec.last_timings();
            self.stages.add(Stage::Launch, t.launch);
            self.stages.add(Stage::DeviceWait, t.device_wait);
        }
        self.stages.mark(Stage::Prepare);
        self.metrics
            .model
            .observe_forward(phase, started.elapsed().as_secs_f64());
        let logits = result.map_err(|e| format!("{} forward pass failed: {e}", phase.as_str()))?;
        if logits.rows != slices.len() {
            return Err(format!(
                "forward returned {} logits rows for {} sequences",
                logits.rows,
                slices.len()
            ));
        }
        Ok(logits)
    }

    /// Samples every item whose step yields a token and emits its events. The completed shared
    /// prefill of an `n` > 1 request also gives every forking choice its first token, each
    /// from its own copy of the prompt's last logits row.
    fn sample(&mut self, plan: &IterationPlan, mut logits: Logits, outcome: &mut IterationOutcome) {
        let mut drawn = self.draw_decode_tokens(plan, &mut logits);
        let vocab = logits.vocab;
        for (row, item) in plan.items.iter().enumerate() {
            let yields = match item.kind {
                BatchKind::Decode => true,
                BatchKind::Prefill { start, len } => {
                    self.sched.prefill_target(item.seq) == Some(start + len)
                }
            };
            if !yields {
                continue;
            }
            let Some(&(id, choice)) = self.seqs.get(&item.seq) else {
                continue;
            };
            let Some(r) = self.requests.get(&id) else {
                continue;
            };
            if r.done {
                continue;
            }
            let forks = if choice == 0 && matches!(item.kind, BatchKind::Prefill { .. }) {
                r.forking_choices()
            } else {
                Vec::new()
            };
            let row_logits = &mut logits.data[row * vocab..(row + 1) * vocab];
            let shared = (!forks.is_empty()).then(|| row_logits.to_vec());
            let token = drawn[row].take();
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
    /// tokens are those of sampling row by row). Indexed by plan row; `None` rows (prefill
    /// completions with their forks, constrained choices) are sampled by
    /// [`ActiveRequest::step`] on the engine thread.
    fn draw_decode_tokens(
        &mut self,
        plan: &IterationPlan,
        logits: &mut Logits,
    ) -> Vec<Option<SampledToken>> {
        let mut drawn: Vec<Option<SampledToken>> = vec![None; plan.items.len()];
        let decode_rows: HashMap<SeqId, usize> = plan
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches!(item.kind, BatchKind::Decode))
            .map(|(row, item)| (item.seq, row))
            .collect();
        if decode_rows.is_empty() {
            return drawn;
        }
        let mut rows: Vec<Option<&mut [f32]>> = logits
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
                    && let Some(row_logits) = rows.get_mut(row).and_then(Option::take)
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
        drawn
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
        let c = &mut r.choices[choice];
        match c.last_token_at {
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

    /// Every request of the iteration fails with `internal_error`; the scheduler frees them.
    fn fail_iteration(&mut self, message: String, outcome: &mut IterationOutcome) {
        tracing::error!(event = "iteration_failed", iteration = outcome.iteration, error = %message, "iteration failed");
        for id in self.iteration_requests.clone() {
            if self.requests.get(&id).is_some_and(|r| !r.done) {
                self.account(id, Outcome::Failed, &message);
                self.deliver(
                    id,
                    ActiveRequest::error_event(ErrorCode::InternalError, message.clone()),
                );
                self.retire(id);
            }
        }
        outcome.failed = Some(IterationFailure { message });
    }

    /// The engine stops: every request not yet accounted gets `internal_error`.
    fn fail_all(&mut self, message: &str) {
        let live: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| !r.done)
            .map(|(id, _)| *id)
            .collect();
        for id in live {
            self.account(id, Outcome::Failed, message);
            self.deliver(
                id,
                ActiveRequest::error_event(ErrorCode::InternalError, message),
            );
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

    /// Records the end of request `id` once (tokens, latency, outcome, log line).
    fn account(&mut self, id: RequestId, outcome: Outcome, detail: &str) {
        let Some(r) = self.requests.get_mut(&id) else {
            return;
        };
        if r.done {
            return;
        }
        r.done = true;
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
        let kv = KvDocument::from_pool(&self.pool);
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
    }
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

    use tokio::sync::oneshot;
    use turbine_core::clock::SystemClock;
    use turbine_core::request::{
        CancelFlag, Endpoint, GenerationRequest, SamplingParams, StopConditions,
    };
    use turbine_core::types::{BlockId, DeviceId, KvLayout, ModelShape, Priority};
    use turbine_kernels::{KernelError, KernelMetrics, KernelRegistry, cpu_reference_provider};
    use turbine_kv::{BlockPoolConfig, KvMetrics};
    use turbine_model::executor::{self, ExecutorOptions, SequenceKv};
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::{TinySpec, write_tiny_llama};
    use turbine_model::{
        GenerateOptions, MAX_STAGING_BYTES, ModelError, ModelMetrics, SafetensorsIndex,
        WeightLoader, generate, llama_slots,
    };
    use turbine_observability::MetricsRegistry;
    use turbine_scheduler::{SchedulerMetrics, SchedulerParams};
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{DeviceMemory, KvPoolView};

    use super::*;
    use crate::engine::EVENT_CHANNEL_CAPACITY;
    use crate::metrics::ServerMetrics;

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
    }

    /// An engine over `exec` with a 64-block pool.
    fn engine(
        exec: Box<dyn ModelExecutor>,
        tokenizer: Arc<Tokenizer>,
        params: SchedulerParams,
    ) -> TestEngine {
        let reg = MetricsRegistry::new();
        let metrics = EngineMetrics {
            server: ServerMetrics::register(&reg),
            model: ModelMetrics::register(&reg),
            scheduler: SchedulerMetrics::register(&reg),
            kv: KvMetrics::register(&reg),
        };
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: *exec.kv_layout(),
                num_blocks: 64,
            },
            mem(),
        )
        .unwrap();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let scheduler =
            Scheduler::new(params, Arc::clone(&clock)).with_metrics(metrics.scheduler.clone());
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
        });
        TestEngine {
            engine,
            tx,
            shared,
            reg,
        }
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
        assert_eq!(docs.kv.tiers[0].blocks_used, 0);
        assert_eq!(docs.scheduler.waiting, 0);
        assert!(docs.scheduler.iterations_total >= u64::from(max_tokens));
        let text = t.reg.render().unwrap();
        for line in [
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 3"#,
            r#"turbine_kv_blocks{tier="l0",state="used"} 0"#,
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
        assert_eq!(docs.kv.tiers[0].blocks_used, 0);
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

    /// Fails every forward like a device error, or panics.
    struct BrokenExecutor {
        shape: ModelShape,
        kv: KvLayout,
        panic: bool,
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
            Err(ModelError::Kernel(KernelError::Device {
                message: "hipErrorOutOfMemory: injected".into(),
            }))
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
        })
    }

    /// CONFLICT C-25: three consecutive failed iterations stop the engine; every request in
    /// them, and every request still queued, gets `internal_error`, and no block stays used.
    #[test]
    fn three_consecutive_failed_iterations_stop_the_engine() {
        let (_dir, spec, tokenizer) = tiny();
        // One running request at a time: each iteration fails exactly one request.
        let t = engine(broken(&spec, false), tokenizer, params(1, 64));
        let mut streams: Vec<_> = (0..4)
            .map(|_| submit(&t.tx, request(&[256, 1, 2], 4)))
            .collect();
        let err = t.engine.run().unwrap_err();
        assert!(err.contains("3 consecutive iterations failed"), "{err}");
        assert!(err.contains("hipErrorOutOfMemory"), "{err}");
        for (rx, admitted) in &mut streams {
            assert_eq!(admitted.try_recv().unwrap(), Ok(()));
            assert!(internal_error(&drain(rx)));
        }
        assert_eq!(t.shared.docs().unwrap().kv.tiers[0].blocks_used, 0);
        let text = t.reg.render().unwrap();
        assert!(
            text.contains(
                r#"turbine_requests_total{endpoint="/v1/completions",outcome="failed"} 4"#
            ),
            "{text}"
        );
    }

    /// An engine panic is caught: every in-flight request gets `internal_error` and the engine
    /// reports the panic (the server exits 1).
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
        // Four events: `Started` and three tokens fill the slow stream's channel.
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
        assert!(shared.docs().unwrap().kv.tiers[0].blocks_used > 0);
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
        assert_eq!(shared.docs().unwrap().kv.tiers[0].blocks_used, 0);
        let text = reg.render().unwrap();
        for line in [
            r#"turbine_requests_cancelled_total{reason="client_disconnect"} 1"#,
            r#"turbine_requests_total{endpoint="/v1/completions",outcome="ok"} 2"#,
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
