//! Pipeline-parallel execution in the engine (P5 S-10; plan Task 24, the engine half).
//!
//! A pipeline of `pp` stages runs one stage per thread: every stage before the last on a thread
//! of its own ([`StageWorker`]), the last stage on the engine thread inside [`PpExecutor`], a
//! [`ModelExecutor`] like any other. Each stage holds its layers' weights
//! ([`turbine_model::pp`]) and its own paged KV pool of its layers, with the leader pool's block
//! count and indexed by the leader's logical block ids: the engine's pool is the last stage's,
//! so the engine's block manager (scheduler, prefix cache, KV orchestrator) is the only one and
//! a freed or cancelled sequence's blocks are free on every stage at once.
//!
//! Micro-batches: [`ModelExecutor::launch`] enters a batch into the stages before the last (each
//! stage thread takes its steps in order from a bounded queue) and returns at once; the next
//! [`ModelExecutor::forward`] of that same batch runs the last stage on the engine thread, which
//! receives the hidden state from the stage before it, and returns the logits. So the engine
//! keeps up to `parallel.pipeline.micro_batches` batches launched (the scheduler's micro-batch
//! plans, disjoint sets of sequences) and stage `s` runs micro-batch `k + 1` while stage
//! `s + 1` runs `k`; `send` blocks until the peer receives, so every stage runs concurrently
//! with its neighbours. A `forward` with nothing launched is one whole step (warm-up, probes).
//! Every stage is fed the same batch, in the same order; only the last returns logits.
//!
//! Fork copies (`copy_blocks`) are queued to every stage before the next launch and run on the
//! last stage at once (their blocks belong to no batch in flight).
//!
//! Failures: a stage whose executor (or hand-off) fails aborts the group's communicator, so no
//! stage waits out the op timeout; the step of that batch and of every other batch in the
//! pipeline then fail as [`ModelError::Collective`] (`replica_failed`, circuit
//! `collective_failed`), except a stage's sticky device error, returned as it is (exit 3). The
//! communicator is not re-created (plan Task 28): the replica stays failed until a restart.
//!
//! Timing (`turbine_pipeline_stage_duration_seconds{stage}`, `turbine_pipeline_bubble_ratio`,
//! the `/turbine/v1/scheduler` `pipeline` section): a stage is busy from the later of its start
//! and the previous stage's end (a stage waiting for its input is idle) to the end of its step,
//! which includes a blocked hand-off.
//!
//! Loading ([`load_pipeline`]): every stage loads on its own thread at once — the communicator
//! init is collective — measures its communicator's device memory, loads its layers, re-measures
//! its budget and proposes the block count its budget holds; the pipeline takes the smallest.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use turbine_api::backend::NotReadyReason;
use turbine_core::clock::Clock;
use turbine_core::config::Config;
use turbine_core::types::{BlockId, DeviceId, KvLayout, ModelShape, SeqId};
use turbine_device::DeviceInventory;
use turbine_distributed::collective::{
    Collective, CollectiveError, CollectiveInit, CollectiveLibrary, CollectiveMetrics,
};
use turbine_distributed::plan::{ParallelPlan, ReplicaGroup};
use turbine_kernels::KernelError;
use turbine_kv::BlockPool;
use turbine_kv::identity::KvFormat;
use turbine_model::executor::{
    BatchInput, ExecutorLimits, ForwardTimings, Logits, ModelExecutor, RowReduce, SeqSlice,
    TokenFeed,
};
use turbine_model::pp::{self, PpContext, StageSpec};
use turbine_model::{ModelError, ModelMetrics};
use turbine_observability::MetricsRegistry;
use turbine_reliability::budget::DeviceBudget;
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::reserve::EmergencyReserve;
use turbine_scheduler::{PipelineMetrics, PipelineSnapshot, StageSnapshot, StageTimeline};

use crate::kv_orchestrator::{BlockAddresses, KvShard, kv_format};
use crate::model::{self, LoadedModel, PpStage, PreparedModel, RankPart, StartupError};

/// A stage's failure, kept for the engine: the stage and its error (a sticky device error is
/// returned to the engine as it is).
type Fault = Mutex<Option<(u32, ModelError)>>;

/// A stage that stopped loading because another one failed (the other's error is the cause).
const ANOTHER_STAGE_FAILED: &str = "another stage of the pipeline failed to load";

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn invalid(message: String) -> ModelError {
    ModelError::Kernel(KernelError::InvalidArgument { message })
}

/// One sequence of a batch, owned so it can cross to the stage threads.
#[derive(Clone, Debug)]
struct OwnedSeq {
    seq: SeqId,
    q_start: u32,
    q_len: u32,
    kv_len: u32,
    block_table: Vec<BlockId>,
    reduce: Option<RowReduce>,
}

/// A batch every stage runs (tokens, positions and sequence slices; each stage its own pool).
#[derive(Debug)]
struct OwnedBatch {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    seqs: Vec<OwnedSeq>,
}

impl OwnedBatch {
    fn of(batch: &BatchInput<'_>) -> OwnedBatch {
        OwnedBatch {
            tokens: batch.tokens.to_vec(),
            positions: batch.positions.to_vec(),
            seqs: batch
                .seqs
                .iter()
                .map(|s| OwnedSeq {
                    seq: s.seq,
                    q_start: s.q_start,
                    q_len: s.q_len,
                    kv_len: s.kv_len,
                    block_table: s.block_table.to_vec(),
                    reduce: s.reduce,
                })
                .collect(),
        }
    }

    fn slices(&self) -> Vec<SeqSlice<'_>> {
        self.seqs
            .iter()
            .map(|s| SeqSlice {
                seq: s.seq,
                q_start: s.q_start,
                q_len: s.q_len,
                kv_len: s.kv_len,
                block_table: &s.block_table,
                reduce: s.reduce,
            })
            .collect()
    }

    /// The batch is `batch` (the one a `forward` completes must be the one launched).
    fn same_as(&self, batch: &BatchInput<'_>) -> bool {
        self.tokens == batch.tokens
            && self.positions == batch.positions
            && self.seqs.len() == batch.seqs.len()
            && self
                .seqs
                .iter()
                .zip(batch.seqs)
                .all(|(a, b)| a.seq == b.seq && a.kv_len == b.kv_len && a.q_len == b.q_len)
    }
}

/// What a stage thread takes from its queue.
enum StageJob {
    Step {
        step: u64,
        batch: Arc<OwnedBatch>,
    },
    Copy {
        src: Vec<BlockId>,
        dst: Vec<BlockId>,
    },
}

/// A stage thread's report of one step.
#[derive(Clone, Copy, Debug)]
struct StageDone {
    stage: u32,
    step: u64,
    start: Instant,
    end: Instant,
    ok: bool,
}

/// One stage before the last: its executor over its layers, its KV pool and what must live as
/// long as they do (its communicator, ledger reservations and emergency reserve).
pub(crate) struct StageWorker {
    stage: u32,
    exec: Box<dyn ModelExecutor>,
    pool: BlockPool,
    collective: Arc<dyn Collective>,
    /// Its ledger, reservations and emergency reserve, held while the stage runs.
    _keep: Box<dyn Send>,
}

impl StageWorker {
    pub(crate) fn new(
        stage: u32,
        exec: Box<dyn ModelExecutor>,
        pool: BlockPool,
        collective: Arc<dyn Collective>,
        keep: Box<dyn Send>,
    ) -> StageWorker {
        StageWorker {
            stage,
            exec,
            pool,
            collective,
            _keep: keep,
        }
    }

    /// Records a failure: the other stages may be waiting for a hand-off this stage will never
    /// make, so the communicator is aborted.
    fn fail(&self, faults: &Fault, what: &str, e: ModelError) {
        self.collective.abort();
        tracing::error!(event = "pp_stage_failed", stage = self.stage, what, error = %e, "pipeline stage failed");
        lock(faults).get_or_insert((self.stage, e));
    }

    /// The stage thread: takes steps and copies in order until the queue closes. After a
    /// failure nothing runs on the device any more; every step is reported failed.
    fn run(mut self, jobs: Receiver<StageJob>, done: SyncSender<StageDone>, faults: Arc<Fault>) {
        let mut failed = false;
        while let Ok(job) = jobs.recv() {
            match job {
                StageJob::Copy { src, dst } => {
                    if failed {
                        continue;
                    }
                    let r = self.exec.copy_blocks(&self.pool.view(), &src, &dst);
                    if let Err(e) = r {
                        failed = true;
                        self.fail(&faults, "copy_blocks", e);
                    }
                }
                StageJob::Step { step, batch } => {
                    let start = Instant::now();
                    let ok = !failed && {
                        let view = self.pool.view();
                        let slices = batch.slices();
                        let r = self.exec.forward(&BatchInput {
                            tokens: &batch.tokens,
                            positions: &batch.positions,
                            seqs: &slices,
                            kv: &view,
                        });
                        match r {
                            Ok(_) => true,
                            Err(e) => {
                                failed = true;
                                self.fail(&faults, "forward", e);
                                false
                            }
                        }
                    };
                    let report = StageDone {
                        stage: self.stage,
                        step,
                        start,
                        end: Instant::now(),
                        ok,
                    };
                    if done.send(report).is_err() {
                        break;
                    }
                }
            }
        }
    }
}

/// A stage thread and its queue.
struct StageHandle {
    stage: u32,
    jobs: Option<SyncSender<StageJob>>,
    thread: Option<JoinHandle<()>>,
}

/// The pipeline's timing and placement, shared by the executor (which records every step) and
/// the engine loop (which publishes the `/turbine/v1/scheduler` `pipeline` section).
pub(crate) struct PipelineStats {
    origin: Instant,
    timeline: Mutex<StageTimeline>,
    metrics: PipelineMetrics,
    /// Per stage: its device and first and last layer.
    placement: Vec<(u32, [u32; 2])>,
    /// `parallel.pipeline.micro_batches`, resolved.
    pub micro_batches: u32,
}

impl PipelineStats {
    pub(crate) fn new(
        placement: &[(DeviceId, StageSpec)],
        micro_batches: u32,
        metrics: PipelineMetrics,
    ) -> PipelineStats {
        PipelineStats {
            origin: Instant::now(),
            timeline: Mutex::new(StageTimeline::new(placement.len())),
            metrics,
            placement: placement
                .iter()
                .map(|(d, s)| (d.0, [s.layers.start, s.layers.end.saturating_sub(1)]))
                .collect(),
            micro_batches: micro_batches.max(1),
        }
    }

    pub(crate) fn stages(&self) -> u32 {
        self.placement.len() as u32
    }

    /// One step's busy interval on every stage, in stage order.
    fn record(&self, intervals: &[(Instant, Instant)]) {
        let since = |t: Instant| t.saturating_duration_since(self.origin);
        let mut timeline = lock(&self.timeline);
        let mut last = Instant::now();
        for (stage, &(start, end)) in intervals.iter().enumerate() {
            timeline.record(stage, since(start), since(end));
            self.metrics.observe_stage(
                stage as u32,
                end.saturating_duration_since(start).as_secs_f64(),
            );
            last = end;
        }
        self.metrics
            .set_bubble_ratio(timeline.bubble_ratio(since(last)));
    }

    /// The `pipeline` section of `/turbine/v1/scheduler` with `in_flight` micro-batches.
    pub(crate) fn snapshot(&self, in_flight: u32) -> PipelineSnapshot {
        let now = Instant::now().saturating_duration_since(self.origin);
        let timeline = lock(&self.timeline);
        PipelineSnapshot {
            stages: self
                .placement
                .iter()
                .enumerate()
                .map(|(s, &(device, layers))| StageSnapshot {
                    stage: s as u32,
                    device,
                    layers,
                    busy_ratio: timeline.busy_ratio(s, now),
                })
                .collect(),
            micro_batches: self.micro_batches,
            micro_batches_in_flight: in_flight,
        }
    }
}

/// A batch entered into the stages before the last, waiting for its last-stage `forward`.
struct Fed {
    step: u64,
    batch: Arc<OwnedBatch>,
}

/// The engine's executor of a pipeline (module comment).
pub(crate) struct PpExecutor {
    last: Box<dyn ModelExecutor>,
    /// The last stage's communicator: aborted when the last stage fails.
    collective: Arc<dyn Collective>,
    stages: Vec<StageHandle>,
    done: Receiver<StageDone>,
    /// Reports of later steps, received while waiting for an earlier one.
    early: Vec<StageDone>,
    faults: Arc<Fault>,
    step: u64,
    fed: VecDeque<Fed>,
    stats: Arc<PipelineStats>,
    /// How long the engine waits for a stage's report of a step the last stage finished.
    op_timeout: Duration,
}

impl PpExecutor {
    /// Starts one thread per stage of `workers` (stages 0..pp-1 in order) around `last`, the
    /// last stage, whose communicator is `collective`. Each stage queue holds at most `depth`
    /// jobs (steps and copies).
    pub(crate) fn start(
        last: Box<dyn ModelExecutor>,
        collective: Arc<dyn Collective>,
        workers: Vec<StageWorker>,
        stats: Arc<PipelineStats>,
        op_timeout: Duration,
    ) -> Result<PpExecutor, StartupError> {
        let depth = 2 * stats.micro_batches as usize + 2;
        // At most `micro_batches` steps are launched, each reported once by every stage.
        let (done_tx, done) = sync_channel(workers.len() * (depth + 1) + 1);
        let faults: Arc<Fault> = Arc::new(Mutex::new(None));
        let mut stages = Vec::with_capacity(workers.len());
        for w in workers {
            let stage = w.stage;
            let (tx, rx) = sync_channel(depth);
            let (done_tx, faults) = (done_tx.clone(), Arc::clone(&faults));
            let thread = std::thread::Builder::new()
                .name(format!("turbine-pp-stage-{stage}"))
                .spawn(move || w.run(rx, done_tx, faults))
                .map_err(|e| {
                    StartupError::new(format!("cannot start pipeline stage thread {stage}: {e}"))
                })?;
            stages.push(StageHandle {
                stage,
                jobs: Some(tx),
                thread: Some(thread),
            });
        }
        Ok(PpExecutor {
            last,
            collective,
            stages,
            done,
            early: Vec::new(),
            faults,
            step: 0,
            fed: VecDeque::new(),
            stats,
            op_timeout,
        })
    }

    /// A stage's sticky device error, if one was reported (the process must exit 3).
    fn sticky_fault(&self) -> Option<ModelError> {
        let mut slot = lock(&self.faults);
        let sticky = matches!(&*slot, Some((_, ModelError::Kernel(k))) if k.is_sticky());
        if sticky {
            return slot.take().map(|(_, e)| e);
        }
        None
    }

    /// The pipeline failed: every stage stops waiting, and the engine gets a stage's sticky
    /// device error as it is, anything else as a failed collective naming the stage.
    fn failed(&self, stage: u32, e: Option<ModelError>) -> ModelError {
        self.collective.abort();
        if let Some(sticky) = self.sticky_fault() {
            return sticky;
        }
        if let Some(ModelError::Kernel(k)) = &e
            && k.is_sticky()
        {
            return e.expect("a sticky error");
        }
        let (stage, message) = match (&*lock(&self.faults), e) {
            (Some((s, first)), _) => (*s, first.to_string()),
            (None, Some(e)) => (stage, e.to_string()),
            (None, None) => (stage, "the stage stopped".to_string()),
        };
        tracing::error!(event = "pp_pipeline_failed", stage, error = %message, "a pipeline stage failed");
        ModelError::Collective(CollectiveError::RemoteAbort {
            rank: stage as usize,
        })
    }

    /// Sends `job` to every stage before the last.
    fn send_to_stages(&mut self, job: impl Fn() -> StageJob) -> Result<(), ModelError> {
        for i in 0..self.stages.len() {
            let stage = self.stages[i].stage;
            let sent = self.stages[i]
                .jobs
                .as_ref()
                .is_some_and(|tx| tx.send(job()).is_ok());
            if !sent {
                return Err(self.failed(stage, None));
            }
        }
        Ok(())
    }

    /// Every earlier stage's report of `step`, in stage order (a stage that failed or does not
    /// report within the op timeout fails the pipeline).
    fn reports(&mut self, step: u64) -> Result<Vec<StageDone>, ModelError> {
        let want = self.stages.len();
        let mut got: Vec<StageDone> = Vec::with_capacity(want);
        self.early.retain(|d| {
            if d.step == step {
                got.push(*d);
                false
            } else {
                true
            }
        });
        let deadline = Instant::now() + self.op_timeout;
        while got.len() < want {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.done.recv_timeout(left) {
                Ok(d) if d.step == step => got.push(d),
                Ok(d) => self.early.push(d),
                Err(RecvTimeoutError::Timeout) => {
                    let missing = (0..want as u32)
                        .find(|s| !got.iter().any(|d| d.stage == *s))
                        .unwrap_or(0);
                    return Err(self.failed(
                        missing,
                        Some(ModelError::Collective(CollectiveError::Timeout {
                            op: "pipeline_stage",
                            after: self.op_timeout,
                        })),
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => return Err(self.failed(0, None)),
            }
        }
        got.sort_by_key(|d| d.stage);
        if let Some(bad) = got.iter().find(|d| !d.ok) {
            return Err(self.failed(bad.stage, None));
        }
        Ok(got)
    }

    /// Runs the last stage on `batch` (entered as `step`) and completes the step.
    fn finish(&mut self, step: u64, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
        let start = Instant::now();
        let result = self.last.forward(batch);
        let end = Instant::now();
        let logits = match result {
            Ok(l) => l,
            Err(e) => {
                let last = self.stats.stages().saturating_sub(1);
                // Wake stages still waiting on the last one before reading their reports.
                self.collective.abort();
                let _ = self.reports(step);
                return Err(self.failed(last, Some(e)));
            }
        };
        let reports = self.reports(step)?;
        let mut intervals = Vec::with_capacity(reports.len() + 1);
        let mut previous: Option<Instant> = None;
        for d in reports
            .iter()
            .map(|d| (d.start, d.end))
            .chain([(start, end)])
        {
            let begin = previous.map_or(d.0, |p| d.0.max(p)).min(d.1);
            intervals.push((begin, d.1));
            previous = Some(d.1);
        }
        self.stats.record(&intervals);
        Ok(logits)
    }
}

impl ModelExecutor for PpExecutor {
    fn shape(&self) -> &ModelShape {
        self.last.shape()
    }

    /// The last stage's layout: its layers (every stage's pool has the same block count).
    fn kv_layout(&self) -> &KvLayout {
        self.last.kv_layout()
    }

    /// Completes the oldest launched batch, which must be `batch`, by running the last stage;
    /// with nothing launched, launches `batch` first (one whole step).
    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
        if self.fed.is_empty() {
            self.launch(batch, &[])?;
        }
        let fed = self.fed.pop_front().expect("a launched batch");
        if !fed.batch.same_as(batch) {
            // Never run the last stage on another batch than the one its input belongs to.
            let e = invalid(format!(
                "pipeline step {}: forward of a batch that is not the oldest launched one",
                fed.step
            ));
            return Err(self.failed(self.stats.stages().saturating_sub(1), Some(e)));
        }
        self.finish(fed.step, batch)
    }

    fn last_timings(&self) -> ForwardTimings {
        self.last.last_timings()
    }

    /// Enters `batch` into the stages before the last and returns (module comment); up to
    /// `parallel.pipeline.micro_batches` batches may be launched before their `forward`.
    fn launch(&mut self, batch: &BatchInput<'_>, feeds: &[TokenFeed]) -> Result<(), ModelError> {
        if !feeds.is_empty() {
            return Err(invalid(
                "a pipeline does not take tokens chosen on the device".into(),
            ));
        }
        self.step += 1;
        let step = self.step;
        let owned = Arc::new(OwnedBatch::of(batch));
        self.send_to_stages(|| StageJob::Step {
            step,
            batch: Arc::clone(&owned),
        })?;
        self.fed.push_back(Fed { step, batch: owned });
        Ok(())
    }

    fn reduces_logits(&self) -> bool {
        self.last.reduces_logits()
    }

    /// Queues the copies to every earlier stage (ahead of the next launch) and runs them on the
    /// last stage now.
    fn copy_blocks(
        &mut self,
        kv: &turbine_tensor::KvPoolView<'_>,
        src: &[BlockId],
        dst: &[BlockId],
    ) -> Result<(), ModelError> {
        self.send_to_stages(|| StageJob::Copy {
            src: src.to_vec(),
            dst: dst.to_vec(),
        })?;
        self.last.copy_blocks(kv, src, dst).map_err(|e| {
            let last = self.stats.stages().saturating_sub(1);
            self.failed(last, Some(e))
        })
    }
}

impl Drop for PpExecutor {
    fn drop(&mut self) {
        if !self.fed.is_empty() {
            // Earlier stages may be blocked handing off a batch the last stage never takes.
            self.collective.abort();
        }
        for s in &mut self.stages {
            s.jobs = None;
        }
        for s in &mut self.stages {
            if let Some(t) = s.thread.take() {
                let _ = t.join();
            }
        }
    }
}

/// What the engine thread needs to load a pipeline besides its last stage's prepared model.
pub(crate) struct PipelineStart {
    /// Stages 0..pp-1 in stage order, each prepared on its device.
    pub stages: Vec<PreparedModel>,
    /// The loaded collective backend (`parallel.collective_backend`).
    pub library: Arc<dyn CollectiveLibrary>,
    pub init_timeout: Duration,
    pub op_timeout: Duration,
    /// `parallel.collective.hostmem_max_bytes` (`None` = `auto`).
    pub route_max_bytes: Option<u64>,
    pub metrics: CollectiveMetrics,
    pub clock: Arc<dyn Clock>,
    /// Timing and placement, shared with the engine loop.
    pub stats: Arc<PipelineStats>,
    /// The whole model's KV format: a tier copy of a block is every stage's shard.
    pub format: KvFormat,
}

/// Prepares replica `group`'s pipeline before the listener binds (P5 S-10): every stage of
/// `plan` on its device, for its layers (`base`: the first replica's first model, whose
/// grammar compiler and kernel metrics every other model shares). Returns the last stage (the
/// engine's model) and the rest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_stages(
    config: &Config,
    inventory: &DeviceInventory,
    registry: &MetricsRegistry,
    base: Option<&PreparedModel>,
    plan: &ParallelPlan,
    group: &ReplicaGroup,
    collective: &(Arc<dyn CollectiveLibrary>, CollectiveMetrics),
    metrics: &PipelineMetrics,
) -> Result<(PreparedModel, PipelineStart), StartupError> {
    let pp = plan.pp;
    let slots_on = |device: DeviceId| {
        plan.groups
            .iter()
            .flat_map(|g| &g.ranks)
            .filter(|s| s.device == device)
            .count() as u32
    };
    let mut prepared: Vec<PreparedModel> = Vec::with_capacity(pp as usize);
    let mut placement = Vec::with_capacity(pp as usize);
    for (spec, slot) in plan.stages.iter().zip(&group.ranks) {
        let mut cfg = config.clone();
        cfg.execution.device = slot.device;
        let part = Some(RankPart::Stage(PpStage {
            spec: StageSpec {
                device: slot.device,
                ..spec.clone()
            },
            stages: pp,
        }));
        let mut p = match base.or(prepared.first()) {
            None => model::prepare_rank(&cfg, inventory, registry, part)?,
            Some(b) => model::prepare_replica(&cfg, inventory, b, part)?,
        };
        p.share_device(slots_on(slot.device))?;
        placement.push((slot.device, spec.clone()));
        prepared.push(p);
    }
    let Some(last) = prepared.pop() else {
        return Err(StartupError::new("a pipeline without stages"));
    };
    let micro_batches = config
        .parallel
        .pipeline
        .micro_batches
        .fixed()
        .unwrap_or(pp)
        .max(1);
    let arch = &last.arch;
    let format = kv_format(arch.kv_layout(last.block_tokens));
    tracing::info!(
        event = "pipeline_prepared",
        stages = pp,
        micro_batches,
        devices = ?placement.iter().map(|(d, _)| d.0).collect::<Vec<_>>(),
        "pipeline stages prepared"
    );
    let (library, cmetrics) = collective;
    Ok((
        last,
        PipelineStart {
            stages: prepared,
            library: Arc::clone(library),
            init_timeout: config.parallel.collective.init_timeout.0,
            op_timeout: config.parallel.collective.op_timeout.0,
            route_max_bytes: config.parallel.collective.hostmem_max_bytes.fixed(),
            metrics: cmetrics.clone(),
            clock: Arc::new(turbine_core::clock::SystemClock::new()),
            stats: Arc::new(PipelineStats::new(
                &placement,
                micro_batches,
                metrics.clone(),
            )),
            format,
        },
    ))
}

/// A stage after its load, before the pipeline's warm-up.
struct StageLoaded {
    executor: Box<dyn ModelExecutor>,
    pool: BlockPool,
    weight_bytes: u64,
    budget: DeviceBudget,
    ledger: Arc<Ledger>,
    held: Vec<Reservation>,
    reserve: EmergencyReserve,
    collective: Arc<dyn Collective>,
}

/// The pipeline's block-count agreement: every stage proposes the blocks its budget holds and
/// all take the smallest; a failed stage releases the others.
struct Agreement {
    state: Mutex<(Vec<Option<u32>>, bool)>,
    cv: Condvar,
}

impl Agreement {
    fn new(stages: usize) -> Agreement {
        Agreement {
            state: Mutex::new((vec![None; stages], false)),
            cv: Condvar::new(),
        }
    }

    fn agree(&self, stage: usize, blocks: u32) -> Option<u32> {
        let mut s = lock(&self.state);
        s.0[stage] = Some(blocks);
        self.cv.notify_all();
        loop {
            if s.1 {
                return None;
            }
            if s.0.iter().all(Option::is_some) {
                return s.0.iter().flatten().copied().min();
            }
            s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn fail(&self) {
        lock(&self.state).1 = true;
        self.cv.notify_all();
    }
}

/// Everything one stage's loader thread shares with the others.
struct PipelineLoad<'a> {
    library: &'a Arc<dyn CollectiveLibrary>,
    unique_id: [u8; 128],
    stages: u32,
    init_timeout: Duration,
    op_timeout: Duration,
    route_max_bytes: Option<u64>,
    clock: &'a Arc<dyn Clock>,
    metrics: &'a CollectiveMetrics,
    reliability: &'a ReliabilityMetrics,
    agreement: &'a Agreement,
    phase: &'a (dyn Fn(NotReadyReason) + Sync),
}

/// Loads stage `p.stage`: communicator, layers, budget, the agreed pool.
fn load_stage(p: &PreparedModel, g: &PipelineLoad<'_>) -> Result<StageLoaded, StartupError> {
    let Some(stage) = p.stage.as_ref() else {
        g.agreement.fail();
        return Err(StartupError::new(
            "a pipeline stage prepared without its stage",
        ));
    };
    let spec = &stage.spec;
    let stage_error = |what: &str, e: &dyn std::fmt::Display| {
        StartupError::new(format!(
            "stage {} of {} (device {}, layers {:?}): {what}: {e}",
            spec.stage, g.stages, p.device.0, spec.layers
        ))
    };
    let mem = &p.provider.opened.mem;
    // Reading the free memory also makes this thread's current device the stage's (the kernel
    // library enters its context on every call), which a device communicator's init uses.
    let free = |what: &str| {
        mem.mem_info()
            .map(|m| m.free_bytes)
            .map_err(|e| stage_error(what, &e))
    };
    let before = free("device memory info")?;
    let collective = Arc::clone(g.library)
        .open(CollectiveInit {
            rank: spec.stage as usize,
            world: g.stages as usize,
            unique_id: g.unique_id,
            init_timeout: g.init_timeout,
            op_timeout: g.op_timeout,
            clock: Arc::clone(g.clock),
            metrics: Some(g.metrics.clone()),
            memory: Some(Arc::clone(mem)),
            route_max_bytes: g.route_max_bytes,
        })
        .map_err(|e| {
            g.agreement.fail();
            stage_error("collective init", &e)
        })?;
    let collective_bytes = before.saturating_sub(free("device memory info")?);
    tracing::info!(
        event = "pp_stage_collective",
        stage = spec.stage,
        stages = g.stages,
        device = p.device.0,
        backend = collective.backend(),
        collective_bytes,
        "pipeline stage joined its communicator"
    );
    (g.phase)(NotReadyReason::LoadingWeights);
    let loaded = (|| {
        let weights = model::load_weights(p)?;
        let weight_bytes = weights.weight_bytes;
        let (budget, ledger, held) =
            model::post_load_budget(p, weight_bytes, collective_bytes, g.reliability)?;
        let mine = model::pool_blocks(p, &budget)?;
        let blocks = g
            .agreement
            .agree(spec.stage as usize, mine)
            .ok_or_else(|| stage_error("load", &ANOTHER_STAGE_FAILED))?;
        if blocks < mine {
            tracing::info!(
                event = "pp_pool_agreed",
                stage = spec.stage,
                proposed = mine,
                blocks,
                "the pipeline's smallest KV pool sizes this stage's pool"
            );
        }
        let executor = pp::build_executor(
            &p.arch,
            weights,
            Arc::clone(&p.registry),
            Arc::clone(mem),
            ExecutorLimits {
                block_tokens: p.block_tokens,
                max_batch_tokens: p.scheduler.max_batch_tokens,
                max_seqs: p.scheduler.max_running_requests,
            },
            p.executor_options,
            PpContext {
                stage: spec.stage,
                stages: g.stages,
                layers: spec.layers.clone(),
                collective: Arc::clone(&collective),
                stream: mem.compute_stream(),
            },
        )
        .map_err(|e| stage_error("executor", &e))?;
        let pool = model::allocate_pool(p, blocks, &ledger)?;
        let reserve = model::acquire_reserve(p, &ledger, g.reliability)?;
        Ok(StageLoaded {
            executor,
            pool,
            weight_bytes,
            budget,
            ledger,
            held,
            reserve,
            collective: Arc::clone(&collective),
        })
    })();
    if loaded.is_err() {
        g.agreement.fail();
        collective.abort();
    }
    loaded
}

/// Loads a pipeline whose last stage is `last` and warms it up (module comment): the engine's
/// [`LoadedModel`] with a [`PpExecutor`] and every earlier stage's KV shard for the tier copies
/// (in stage order). `phase` reports the loading step for `/ready` (`collective_init`,
/// `loading_weights`, then `loading_model` for the warm-up).
pub(crate) fn load_pipeline(
    last: &PreparedModel,
    start: PipelineStart,
    warmup_token: u32,
    metrics: &ModelMetrics,
    reliability: &ReliabilityMetrics,
    phase: &(dyn Fn(NotReadyReason) + Sync),
) -> Result<LoadedModel, StartupError> {
    let started = Instant::now();
    let stages = start.stages.len() as u32 + 1;
    let unique_id = start
        .library
        .unique_id()
        .map_err(|e| StartupError::new(format!("collective unique id: {e}")))?;
    phase(NotReadyReason::CollectiveInit);
    let agreement = Agreement::new(stages as usize);
    let load = PipelineLoad {
        library: &start.library,
        unique_id,
        stages,
        init_timeout: start.init_timeout,
        op_timeout: start.op_timeout,
        route_max_bytes: start.route_max_bytes,
        clock: &start.clock,
        metrics: &start.metrics,
        reliability,
        agreement: &agreement,
        phase,
    };
    let all: Vec<&PreparedModel> = start.stages.iter().chain(std::iter::once(last)).collect();
    let results: Vec<Result<StageLoaded, StartupError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .iter()
            .enumerate()
            .map(|(s, p)| {
                let load = &load;
                std::thread::Builder::new()
                    .name(format!("turbine-pp-stage-{s}-load"))
                    .spawn_scoped(scope, move || load_stage(p, load))
            })
            .collect();
        handles
            .into_iter()
            .enumerate()
            .map(|(s, h)| match h {
                Ok(h) => h
                    .join()
                    .unwrap_or_else(|_| Err(StartupError::new(format!("stage {s} load panicked")))),
                Err(e) => {
                    agreement.fail();
                    Err(StartupError::new(format!(
                        "cannot start the load thread of stage {s}: {e}"
                    )))
                }
            })
            .collect()
    });
    let mut loaded = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for r in results {
        match r {
            Ok(stage) => loaded.push(stage),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        let at = errors
            .iter()
            .position(|e| !e.to_string().contains(ANOTHER_STAGE_FAILED))
            .unwrap_or(0);
        return Err(errors.swap_remove(at));
    }
    let Some(tail) = loaded.pop() else {
        return Err(StartupError::new("a pipeline without its last stage"));
    };
    let mut shards = Vec::with_capacity(loaded.len());
    let mut workers = Vec::with_capacity(loaded.len());
    for (s, (stage, p)) in loaded.into_iter().zip(&start.stages).enumerate() {
        shards.push(KvShard {
            device: super::copy_device(p),
            addresses: BlockAddresses::of(&stage.pool),
        });
        workers.push(StageWorker::new(
            s as u32,
            stage.executor,
            stage.pool,
            stage.collective,
            Box::new((stage.held, stage.reserve, stage.ledger)),
        ));
    }
    let mut executor = PpExecutor::start(
        tail.executor,
        tail.collective,
        workers,
        Arc::clone(&start.stats),
        start.op_timeout,
    )?;
    let mut pool = tail.pool;
    phase(NotReadyReason::LoadingModel);
    model::warm_up(&mut executor, &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    metrics.record_load(
        load_seconds,
        last.arch.weight_format.0.name(),
        tail.weight_bytes,
    );
    tracing::info!(
        event = "pp_pipeline_ready",
        stages,
        micro_batches = start.stats.micro_batches,
        load_seconds,
        last_stage_weight_bytes = tail.weight_bytes,
        budget_bytes = tail.budget.budget_bytes,
        kv_blocks = pool.total_blocks(),
        "pipeline loaded and warmed up"
    );
    Ok(LoadedModel {
        executor: Box::new(executor),
        pool,
        weight_bytes: tail.weight_bytes,
        load_seconds,
        budget: tail.budget,
        ledger: tail.ledger,
        reserve: tail.reserve,
        held: tail.held,
        shards,
        remote_tiers: None,
        group: Vec::new(),
    })
}

/// A 2-stage pipeline of the tiny Llama on host devices, for the engine's tests.
#[cfg(test)]
pub(crate) mod testing {
    use turbine_distributed::collective::HostCollective;
    use turbine_kernels::test_support::sticky_device_error;
    use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
    use turbine_kv::BlockPoolConfig;
    use turbine_model::executor::{self, ExecutorOptions};
    use turbine_model::testing::tiny::TinySpec;
    use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, llama_slots};
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;

    pub(crate) const BLOCK_TOKENS: u32 = 16;
    const LIMITS: ExecutorLimits = ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens: 64,
        max_seqs: 4,
    };

    fn host(device: u32) -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(100 + device), 1 << 30)
    }

    fn registry(reqs: &[turbine_kernels::OpRequirement]) -> Arc<KernelRegistry> {
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        Arc::new(KernelRegistry::build(vec![provider], &order, reqs, &metrics, None).unwrap())
    }

    fn pool(mem: &Arc<dyn DeviceMemory>, layout: KvLayout, blocks: u32) -> BlockPool {
        BlockPool::new(
            BlockPoolConfig {
                layout,
                num_blocks: blocks,
            },
            Arc::clone(mem),
        )
        .unwrap()
    }

    /// The tiny Llama on one host device, with a pool of `blocks` blocks.
    pub(crate) fn one_device(spec: &TinySpec, blocks: u32) -> (Box<dyn ModelExecutor>, BlockPool) {
        let (cfg, mem) = (&spec.config, host(9));
        let index = SafetensorsIndex::open(&spec.dir).unwrap();
        let weights = WeightLoader::load(&index, &llama_slots(cfg), &mem, MAX_STAGING_BYTES)
            .expect("weights");
        let reqs = executor::requirements(cfg, BLOCK_TOKENS, ExecutorOptions::default());
        let exec = executor::build_executor(
            cfg,
            weights,
            registry(&reqs),
            Arc::clone(&mem),
            BLOCK_TOKENS,
            64,
            4,
            ExecutorOptions::default(),
        )
        .unwrap();
        let pool = pool(&mem, cfg.kv_layout(BLOCK_TOKENS), blocks);
        (exec, pool)
    }

    /// Stage `stage` of the tiny Llama split into its 2 layers, on host device `stage`.
    fn stage(
        spec: &TinySpec,
        stage: u32,
        collective: Arc<dyn Collective>,
        blocks: u32,
    ) -> (Box<dyn ModelExecutor>, BlockPool) {
        let (cfg, mem) = (&spec.config, host(stage));
        let layers = stage..stage + 1;
        let s = StageSpec {
            stage,
            device: DeviceId(stage),
            layers: layers.clone(),
            embedding: stage == 0,
            lm_head: stage == 1,
        };
        let opts = ExecutorOptions::default();
        let index = SafetensorsIndex::open(&spec.dir).unwrap();
        let slots = pp::weight_slots(cfg, &s).unwrap();
        let weights = WeightLoader::load_part(
            cfg.weight_format.get(),
            &index,
            &slots,
            &llama_slots(cfg),
            &mem,
            MAX_STAGING_BYTES,
        )
        .expect("stage weights");
        let reqs = pp::requirements(cfg, &layers, BLOCK_TOKENS, opts).unwrap();
        let exec = pp::build_executor(
            cfg,
            weights,
            registry(&reqs),
            Arc::clone(&mem),
            LIMITS,
            opts,
            PpContext {
                stage,
                stages: 2,
                layers: layers.clone(),
                collective,
                stream: mem.compute_stream(),
            },
        )
        .unwrap();
        let pool = pool(
            &mem,
            pp::kv_layout(cfg, &layers, BLOCK_TOKENS).unwrap(),
            blocks,
        );
        (exec, pool)
    }

    pub(crate) fn stats(micro_batches: u32) -> Arc<PipelineStats> {
        let spec = |s: u32| StageSpec {
            stage: s,
            device: DeviceId(s),
            layers: s..s + 1,
            embedding: s == 0,
            lm_head: s == 1,
        };
        Arc::new(PipelineStats::new(
            &[(DeviceId(0), spec(0)), (DeviceId(1), spec(1))],
            micro_batches,
            PipelineMetrics::register(&MetricsRegistry::new()),
        ))
    }

    /// A 2-stage pipeline of the tiny Llama over the host collective, stage 0's executor
    /// wrapped by `wrap`, every stage pool of `blocks` blocks; returns the executor and the last
    /// stage's (the engine's) pool.
    pub(crate) fn pipeline(
        spec: &TinySpec,
        stats: Arc<PipelineStats>,
        blocks: u32,
        wrap: impl FnOnce(Box<dyn ModelExecutor>) -> Box<dyn ModelExecutor>,
    ) -> (PpExecutor, BlockPool) {
        let mut comms = HostCollective::group(2, Duration::from_secs(20)).into_iter();
        let c0: Arc<dyn Collective> = Arc::new(comms.next().unwrap());
        let c1: Arc<dyn Collective> = Arc::new(comms.next().unwrap());
        let (e0, p0) = stage(spec, 0, Arc::clone(&c0), blocks);
        let (e1, p1) = stage(spec, 1, Arc::clone(&c1), blocks);
        let worker = StageWorker::new(0, wrap(e0), p0, c0, Box::new(()));
        let exec = PpExecutor::start(e1, c1, vec![worker], stats, Duration::from_secs(20))
            .expect("stage threads");
        (exec, p1)
    }

    /// Stage 0 failing on every step (with a sticky device error when `sticky`).
    pub(crate) struct FailingStage {
        pub inner: Box<dyn ModelExecutor>,
        pub sticky: bool,
    }

    impl ModelExecutor for FailingStage {
        fn shape(&self) -> &ModelShape {
            self.inner.shape()
        }
        fn kv_layout(&self) -> &KvLayout {
            self.inner.kv_layout()
        }
        fn forward(&mut self, _batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            Err(ModelError::Kernel(if self.sticky {
                sticky_device_error("injected sticky stage failure")
            } else {
                KernelError::InvalidArgument {
                    message: "injected stage failure".into(),
                }
            }))
        }
        fn copy_blocks(
            &mut self,
            kv: &turbine_tensor::KvPoolView<'_>,
            src: &[BlockId],
            dst: &[BlockId],
        ) -> Result<(), ModelError> {
            self.inner.copy_blocks(kv, src, dst)
        }
    }
}

#[cfg(test)]
mod tests {
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::write_tiny_llama;

    use super::testing::{FailingStage, one_device, pipeline, stats};
    use super::*;

    const BLOCKS: u32 = 8;

    fn argmax(row: &[f32]) -> u32 {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap() as u32
    }

    /// Two sequences' prompts, then decode steps, run as two micro-batches (sequence 1 alone,
    /// sequence 2 alone) launched together and completed in order when `pipelined`, else one
    /// batch at a time; the greedy tokens of both.
    fn greedy_two(
        exec: &mut dyn ModelExecutor,
        pool: &BlockPool,
        pipelined: bool,
    ) -> [Vec<u32>; 2] {
        let prompts: [Vec<u32>; 2] = [
            (0..9u32).map(|i| (i * 7 + 1) % 200).collect(),
            (0..5u32).map(|i| (i * 13 + 5) % 200).collect(),
        ];
        let tables = [[BlockId(0)], [BlockId(1)]];
        let mut next: [Vec<u32>; 2] = [prompts[0].clone(), prompts[1].clone()];
        let mut lens = [0u32; 2];
        let mut out: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        let view = pool.view();
        for _ in 0..6 {
            let batches: Vec<(Vec<u32>, Vec<u32>, [SeqSlice<'_>; 1])> = (0..2)
                .map(|s| {
                    let tokens = next[s].clone();
                    let positions: Vec<u32> = (lens[s]..lens[s] + tokens.len() as u32).collect();
                    let slice = SeqSlice {
                        seq: SeqId(s as u64 + 1),
                        q_start: 0,
                        q_len: tokens.len() as u32,
                        kv_len: lens[s] + tokens.len() as u32,
                        block_table: &tables[s],
                        reduce: None,
                    };
                    (tokens, positions, [slice])
                })
                .collect();
            if pipelined {
                for b in &batches {
                    exec.launch(
                        &BatchInput {
                            tokens: &b.0,
                            positions: &b.1,
                            seqs: &b.2,
                            kv: &view,
                        },
                        &[],
                    )
                    .expect("launch");
                }
            }
            for (s, b) in batches.iter().enumerate() {
                let logits = exec
                    .forward(&BatchInput {
                        tokens: &b.0,
                        positions: &b.1,
                        seqs: &b.2,
                        kv: &view,
                    })
                    .expect("forward");
                let t = argmax(logits.row(0));
                lens[s] += b.0.len() as u32;
                next[s] = vec![t];
                out[s].push(t);
            }
        }
        out
    }

    /// P5 S-10 in the engine's executor: the tiny Llama as 2 stages on their own threads over
    /// the host collective gives one device's greedy tokens for two sequences, stepped one
    /// batch at a time and as two micro-batches launched together (stage 0 runs the second
    /// while the last stage runs the first); the stage timeline records every step. Breaks if
    /// a micro-batch's hand-off is matched to the wrong batch or a stage runs out of order.
    #[test]
    fn pipeline_matches_one_device() {
        let dir = TempDir::new("engine-pp");
        let spec = write_tiny_llama(dir.path(), 5);
        let (mut one, one_pool) = one_device(&spec, BLOCKS);
        let want = greedy_two(one.as_mut(), &one_pool, false);
        for pipelined in [false, true] {
            let (mut pp, pool) = pipeline(&spec, stats(2), BLOCKS, |e| e);
            assert_eq!(*pp.kv_layout(), pool.layout());
            let got = greedy_two(&mut pp, &pool, pipelined);
            assert_eq!(got, want, "pipelined {pipelined}");
            let snap = pp.stats.snapshot(0);
            assert_eq!(snap.stages.len(), 2);
            assert_eq!(snap.stages[1].layers, [1, 1]);
            assert!(snap.stages.iter().all(|s| s.busy_ratio > 0.0), "{snap:?}");
        }
    }

    /// P5 S-10 failure mode "pipeline stage fails": stage 0's error aborts the communicator,
    /// so the last stage stops waiting at once and every launched batch fails as a collective
    /// error naming stage 0 (the engine ends their requests with `replica_failed`); a stage's
    /// sticky device error comes back as that error (exit 3). Breaks if the last stage waits
    /// out the op timeout or a stage failure passes for success.
    #[test]
    fn stage_failure_fails_every_launched_batch() {
        let dir = TempDir::new("engine-pp-fail");
        let spec = write_tiny_llama(dir.path(), 5);
        for sticky in [false, true] {
            let (mut pp, pool) = pipeline(&spec, stats(2), BLOCKS, |inner| {
                Box::new(FailingStage { inner, sticky }) as Box<dyn ModelExecutor>
            });
            let view = pool.view();
            let tables = [[BlockId(0)], [BlockId(1)]];
            let started = Instant::now();
            let batch = |s: usize| {
                [SeqSlice {
                    seq: SeqId(s as u64),
                    q_start: 0,
                    q_len: 2,
                    kv_len: 2,
                    block_table: &tables[s],
                    reduce: None,
                }]
            };
            let (b0, b1) = (batch(0), batch(1));
            for seqs in [&b0, &b1] {
                pp.launch(
                    &BatchInput {
                        tokens: &[1, 2],
                        positions: &[0, 1],
                        seqs,
                        kv: &view,
                    },
                    &[],
                )
                .expect("launch");
            }
            for seqs in [&b0, &b1] {
                let e = pp
                    .forward(&BatchInput {
                        tokens: &[1, 2],
                        positions: &[0, 1],
                        seqs,
                        kv: &view,
                    })
                    .expect_err("the stage failed");
                match (&e, sticky) {
                    (ModelError::Collective(CollectiveError::RemoteAbort { rank: 0 }), false) => {}
                    (ModelError::Kernel(k), true) if k.is_sticky() => {}
                    // After the sticky error was returned once, the rest is a failed pipeline.
                    (ModelError::Collective(_), true) => {}
                    other => panic!("sticky {sticky}: {other:?}"),
                }
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "no stage waited out the op timeout"
            );
        }
    }
}
