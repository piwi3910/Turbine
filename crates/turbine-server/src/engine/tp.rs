//! Tensor-parallel execution in the engine (P5 S-5, S-6; plan Task 17).
//!
//! One engine thread drives a whole tensor-parallel group through [`TpExecutor`], a
//! [`ModelExecutor`] like any other: rank 0's executor runs on the engine thread and every other
//! rank on its own thread of a [`RankRuntime`] (`local` mode), each rank holding its weight
//! shard and its own paged KV pool — the leader pool's block count and layout per rank
//! ([`turbine_model::tp::kv_layout`]), indexed by the leader's logical block ids, so the leader's
//! block manager (scheduler, prefix cache, KV orchestrator) is the only one. Each `forward`
//! becomes a [`StepPlan`] handed to the workers before rank 0 runs its own part; the collectives
//! inside the forward keep the ranks in lockstep, every rank ends with the full logits, and the
//! leader returns rank 0's. Workers ask their executor for a one-candidate device reduction of
//! each row when it reduces logits, so they never copy full rows to the host, and discard the
//! result. After the step the leader waits until every worker is idle, so the leader may issue
//! tier copies on the workers' contexts between steps (the KV orchestrator's shards).
//!
//! Fork copies (`copy_blocks`) run on the leader's pool and travel to the workers in a plan of
//! their own ([`StepPlan::copies`]), ahead of the next forward.
//!
//! Failures: a failed collective or a failed rank aborts the group's communicator, so no rank
//! waits out the op timeout, and the step returns [`ModelError::Collective`]: the engine fails
//! the iteration's requests with `replica_failed` and opens the circuit (`collective_failed`).
//! A worker's sticky device error is returned as that error, so the process still exits 3.
//! Anything else marks the group broken, and its next step — the circuit's probe after the
//! cooldown — first re-creates the communicator (P5 Task 28, decision "P5: collective failure
//! recovery" B): a fresh unique id, `Reinit` to every worker (a failed worker thread or process
//! stays up for it), every rank opening its rank of the new group at once, each bounded by the
//! init timeout, and every executor on its new communicator (`ModelExecutor::set_collective`).
//! A failed re-creation fails the probe; the next probe tries again.
//!
//! Loading ([`load_group`]): every rank loads on its own thread at once — the communicator init
//! is collective — measures the communicator's device memory (the drop of free memory across the
//! init, P5 S-8), loads its weight shard, re-measures its budget and proposes the block count its
//! budget holds; the group takes the smallest, so every pool has the same block count. The
//! leader's warm-up forward then runs the whole group once.
//!
//! `static` rank mode (`parallel.ranks.mode: static`): rank 0 is the leader process serving
//! HTTP; it waits for every worker process to join over the rank transport
//! (`RankRuntime::static_leader`, `rank_missing` meanwhile), then loads like `local` mode with
//! only its own rank, the pool agreement going through the communicator (an all-reduce); plans
//! travel as frames. Each worker process ([`run_static_worker`]) joins, loads its rank and
//! executes plans until the leader shuts it down or is lost (exit 1). KV tiers: each process
//! keeps its own rank's L1/L2 shards, the leader's orchestrator driving the workers' copies
//! over the rank link (P5 Task 30, [`super::tp_tiers`]).
//!
//! Group admission in `static` mode (P5 Task 33, decision "P5: group KV admission in static
//! rank mode" B): each worker reports its memory budget once it loaded (`Loaded`), and the
//! leader keeps an exact mirror of that worker's ledger ([`Ledger::mirror`]) in the group's
//! ledgers, so admission reserves on every rank as in `local` mode. Every step plan carries the
//! mirrors' changes since the previous plan ([`StepPlan::ledger`]); the worker replays them on
//! its real ledger before the step ([`LedgerReplica`]), so the two follow the leader's
//! reservations in step order. Every [`MIRROR_CHECK_STEPS`] steps, and in the last plan before
//! shutdown, a plan also carries the mirror's digest, which the worker compares with its real
//! ledger's: a mismatch logs `event="ledger_mirror_divergence"` (WARN, reason
//! `mirror_digest_mismatch`) and counts `turbine_ledger_mirror_divergence_total{rank}`.

use std::net::SocketAddr;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use turbine_api::backend::NotReadyReason;
use turbine_core::clock::Clock;
use turbine_core::types::{BlockId, DeviceId, KvLayout, MemoryKind, ModelShape};
use turbine_distributed::collective::{
    Collective, CollectiveError, CollectiveInit, CollectiveLibrary, CollectiveMetrics, ReduceOp,
};
use turbine_distributed::rank::{
    ExecError, HelloExpect, LedgerChange, RankBudget, RankError, RankLedger, RankMessage,
    RankRuntime, StepExecutor, StepOutput, StepPlan, StepSeq,
};
use turbine_distributed::transport::Transport;
use turbine_kv::BlockPool;
use turbine_model::ep::{self, EpAttention, EpContext};
use turbine_model::executor::{
    BatchInput, ExecutorLimits, ForwardTimings, Logits, ModelExecutor, RowReduce, SeqSlice,
};
use turbine_model::tp::{self, ShardSpec, TpContext};
use turbine_model::{ModelError, ModelMetrics};
use turbine_reliability::budget::DeviceBudget;
use turbine_reliability::budget::PoolKind;
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_reliability::ledger::{LedgerOp, LedgerReplica};
use turbine_reliability::metrics::RankLabel;
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::reserve::EmergencyReserve;
use turbine_tensor::{DType, DeviceBuffer, DeviceMemory};

use crate::kv_orchestrator::{BlockAddresses, KvShard};
use crate::model::{self, LoadedModel, PreparedModel, StartupError};
use crate::parallel::ExpertStats;

/// A worker rank's failure, kept for the leader: its rank and its error (a sticky device error
/// is returned to the engine as it is).
type Fault = Mutex<Option<(u32, ModelError)>>;

/// A rank that stopped loading because another one failed (the other's error is the cause).
const ANOTHER_RANK_FAILED: &str = "another rank of the group failed to load";

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The step plan of `batch`: every sequence's new tokens, positions and block table.
fn step_plan(step: u64, batch: &BatchInput<'_>) -> StepPlan {
    StepPlan {
        step,
        sequences: batch
            .seqs
            .iter()
            .map(|s| {
                let r = s.q_start as usize..(s.q_start + s.q_len) as usize;
                StepSeq {
                    seq_id: s.seq,
                    tokens: batch.tokens[r.clone()].to_vec(),
                    positions: batch.positions[r].to_vec(),
                    block_table: s.block_table.to_vec(),
                    is_prefill: s.q_len > 1,
                }
            })
            .collect(),
        copies: Vec::new(),
        ledger: Vec::new(),
    }
}

/// Steps between two mirror-ledger digest checks in `static` mode (P5 Task 33); the last plan
/// before shutdown always carries the digests.
pub(crate) const MIRROR_CHECK_STEPS: u64 = 64;

/// The leader's mirror of one `static` worker rank's ledger (module comment).
struct Mirror {
    rank: u32,
    device: DeviceId,
    ledger: Arc<Ledger>,
}

/// Fault injection (the `fault-injection` build only, P5 Task 28 lab test): when this
/// environment variable names a file, the first worker-rank step that finds the file deletes it
/// and aborts that rank's communicator before its forward — a collective failure mid-generation
/// on any backend.
#[cfg(feature = "fault-injection")]
pub(crate) const ABORT_FILE_ENV: &str = "TURBINE_FAULT_TP_ABORT_FILE";

/// True once per appearance of the [`ABORT_FILE_ENV`] file.
#[cfg(feature = "fault-injection")]
fn abort_injected() -> bool {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var_os(ABORT_FILE_ENV).map(Into::into))
        .as_ref()
        .is_some_and(|p| std::fs::remove_file(p).is_ok())
}

/// A failure at the rank runtime as the collective failure the engine reports
/// (`replica_failed`, circuit `collective_failed`).
fn collective_error(e: RankError) -> ModelError {
    ModelError::Collective(match e {
        RankError::Closed { rank } | RankError::Executor { rank, .. } => {
            CollectiveError::RemoteAbort {
                rank: rank as usize,
            }
        }
        other => CollectiveError::Backend {
            code: -1,
            message: other.to_string(),
        },
    })
}

/// A mirror's KV-pool change on the rank link (the mirror reserves only KV: group admission).
fn wire_change(op: LedgerOp) -> Option<LedgerChange> {
    match op {
        LedgerOp::Reserve {
            id,
            pool: PoolKind::Kv,
            bytes,
        } => Some(LedgerChange::Reserve { id, bytes }),
        LedgerOp::Reserve { .. } => None,
        LedgerOp::Commit { id, bytes } => Some(LedgerChange::Commit { id, bytes }),
        LedgerOp::Release { id } => Some(LedgerChange::Release { id }),
    }
}

/// A change from the rank link as a ledger op on the worker's KV pool.
fn ledger_op(c: LedgerChange) -> LedgerOp {
    match c {
        LedgerChange::Reserve { id, bytes } => LedgerOp::Reserve {
            id,
            pool: PoolKind::Kv,
            bytes,
        },
        LedgerChange::Commit { id, bytes } => LedgerOp::Commit { id, bytes },
        LedgerChange::Release { id } => LedgerOp::Release { id },
    }
}

/// The leader's executor of a tensor-parallel group (module comment).
pub(crate) struct TpExecutor {
    leader: Box<dyn ModelExecutor>,
    runtime: RankRuntime,
    /// Rank 0's communicator: aborted when rank 0 fails, so the workers stop waiting.
    collective: Arc<dyn Collective>,
    faults: Arc<Fault>,
    step: u64,
    /// Expert parallelism: the group's token counts, turned into metrics after every step.
    experts: Option<Arc<ExpertStats>>,
    /// `static` mode: the mirrors of the workers' ledgers (module comment), the steps between
    /// digest checks, the mirror changes a failed hand-off did not deliver, and the workers'
    /// startup commitments on the mirrors (held, never sent: the workers hold their own).
    mirrors: Vec<Mirror>,
    check_every: u64,
    unsent: Vec<RankLedger>,
    _mirror_held: Vec<Reservation>,
    /// How rank 0 opens its communicator again; `None`: the group cannot be re-created.
    comm: Option<CommSpec>,
    /// A collective failed: the next step re-creates the communicator first (P5 Task 28).
    broken: bool,
}

/// How long the leader waits for the workers' answers to a re-init beyond the communicator
/// init timeout (an NCCL-API init that never returns is abandoned 5 s after it).
const REINIT_MARGIN: Duration = Duration::from_secs(10);

impl TpExecutor {
    pub(crate) fn new(
        leader: Box<dyn ModelExecutor>,
        runtime: RankRuntime,
        collective: Arc<dyn Collective>,
        faults: Arc<Fault>,
    ) -> TpExecutor {
        TpExecutor {
            leader,
            runtime,
            collective,
            faults,
            step: 0,
            experts: None,
            mirrors: Vec::new(),
            check_every: MIRROR_CHECK_STEPS,
            unsent: Vec::new(),
            _mirror_held: Vec::new(),
            comm: None,
            broken: false,
        }
    }

    /// Re-creates a failed communicator with `comm` (P5 Task 28).
    fn with_comm(mut self, comm: CommSpec) -> TpExecutor {
        self.comm = Some(comm);
        self
    }

    /// Reports `experts` after every forward (expert parallelism).
    pub(crate) fn with_experts(mut self, experts: Option<Arc<ExpertStats>>) -> TpExecutor {
        self.experts = experts;
        self
    }

    /// `static` mode: sends `mirrors`' changes with every plan and their digests every
    /// `check_every` steps (module comment); `held` are the workers' startup commitments on them.
    fn with_mirrors(
        mut self,
        mirrors: Vec<Mirror>,
        check_every: u64,
        held: Vec<Reservation>,
    ) -> TpExecutor {
        self.mirrors = mirrors;
        self.check_every = check_every.max(1);
        self._mirror_held = held;
        self
    }

    /// Every mirror's changes since the last plan (after any undelivered ones), with its digest
    /// when `check`.
    fn ledger_part(&mut self, check: bool) -> Vec<RankLedger> {
        let mut unsent = std::mem::take(&mut self.unsent);
        self.mirrors
            .iter()
            .map(|m| {
                let (ops, digest) = m.ledger.drain_journal_with_digest(m.device, PoolKind::Kv);
                let mut changes = unsent
                    .iter_mut()
                    .find(|u| u.rank == m.rank)
                    .map(|u| std::mem::take(&mut u.changes))
                    .unwrap_or_default();
                changes.extend(ops.into_iter().filter_map(wire_change));
                RankLedger {
                    rank: m.rank,
                    changes,
                    digest: check.then_some(digest),
                }
            })
            .collect()
    }

    /// A worker's sticky device error, if one was reported (the process must exit 3).
    fn sticky_fault(&self) -> Option<ModelError> {
        let mut slot = lock(&self.faults);
        let sticky = matches!(&*slot, Some((_, ModelError::Kernel(k))) if k.is_sticky());
        if sticky {
            return slot.take().map(|(_, e)| e);
        }
        None
    }

    /// The group failed at the rank runtime: a worker's sticky device error as it is, anything
    /// else as a failed collective.
    fn rank_failed(&self, e: RankError) -> ModelError {
        self.collective.abort();
        if let Some(sticky) = self.sticky_fault() {
            return sticky;
        }
        tracing::error!(event = "tp_rank_failed", error = %e, "a tensor-parallel rank failed");
        collective_error(e)
    }

    /// Re-creates the group's failed communicator (P5 Task 28, decision "P5: collective failure
    /// recovery" B): a fresh unique id, every worker opening its rank of the new group while
    /// rank 0 opens its own — each bounded by the init timeout — and every rank's executor on
    /// its new communicator. The circuit breaker's probe is the step that runs this; a failure
    /// leaves the group broken for the next probe.
    fn recover(&mut self) -> Result<(), ModelError> {
        if let Some(sticky) = self.sticky_fault() {
            return Err(sticky);
        }
        let Some(comm) = self.comm.clone() else {
            return Err(ModelError::Collective(CollectiveError::Backend {
                code: -1,
                message: "this group cannot re-create its communicator".into(),
            }));
        };
        let started = Instant::now();
        let failed = |why: &str, e: &dyn std::fmt::Display| {
            tracing::warn!(
                event = "collective_reinit",
                outcome = "failed",
                reason = why,
                error = %e,
                "re-creating the tensor-parallel communicator failed; the next probe retries"
            );
        };
        let unique_id = comm.library.unique_id().map_err(|e| {
            failed("unique_id", &e);
            ModelError::Collective(e)
        })?;
        if let Err(e) = self.runtime.reinit_begin(unique_id) {
            failed("rank_gone", &e);
            return Err(collective_error(e));
        }
        let opened = comm.open(unique_id);
        let answered = self.runtime.reinit_wait(comm.init_timeout + REINIT_MARGIN);
        let c = match (opened, answered) {
            (Ok(c), Ok(())) => c,
            (Ok(c), Err(e)) => {
                c.abort();
                failed("worker_init", &e);
                return Err(collective_error(e));
            }
            (Err(e), _) => {
                failed("leader_init", &e);
                return Err(ModelError::Collective(e));
            }
        };
        if let Err(e) = self.leader.set_collective(Arc::clone(&c)) {
            c.abort();
            failed("executor", &e);
            return Err(e);
        }
        self.collective = c;
        lock(&self.faults).take();
        self.broken = false;
        tracing::info!(
            event = "collective_reinit",
            outcome = "recreated",
            backend = self.collective.backend(),
            seconds = started.elapsed().as_secs_f64(),
            "tensor-parallel communicator re-created on every rank"
        );
        Ok(())
    }

    /// Hands `plan` to the workers, runs `own` on rank 0 and waits until every worker is idle;
    /// a group whose collective failed is re-created first. A failed collective marks it
    /// broken.
    fn run<T>(
        &mut self,
        plan: StepPlan,
        own: impl FnOnce(&mut dyn ModelExecutor) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        if self.broken {
            self.recover()?;
        }
        let result = self.run_once(plan, own);
        if matches!(result, Err(ModelError::Collective(_))) {
            self.broken = true;
        }
        result
    }

    fn run_once<T>(
        &mut self,
        mut plan: StepPlan,
        own: impl FnOnce(&mut dyn ModelExecutor) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        self.step += 1;
        if !self.mirrors.is_empty() {
            plan.ledger = self.ledger_part(self.step.is_multiple_of(self.check_every));
        }
        let sent_ledger = (!plan.ledger.is_empty()).then(|| plan.ledger.clone());
        if let Err(e) = self.runtime.step(plan) {
            // The workers never saw these changes: they go with the next plan.
            self.unsent = sent_ledger.unwrap_or_default();
            return Err(self.rank_failed(e));
        }
        let out = own(self.leader.as_mut());
        if out.is_err() {
            // The workers may be waiting in a collective rank 0 will never reach.
            self.collective.abort();
        }
        let idle = self.runtime.wait_idle();
        match (out, idle) {
            (Ok(v), Ok(())) => Ok(v),
            (Ok(_), Err(e)) => Err(self.rank_failed(e)),
            (Err(e), _) => Err(self.sticky_fault().unwrap_or(e)),
        }
    }
}

impl ModelExecutor for TpExecutor {
    fn shape(&self) -> &ModelShape {
        self.leader.shape()
    }

    /// Rank 0's layout: its KV heads (every rank's pool has the same block count and bytes).
    fn kv_layout(&self) -> &KvLayout {
        self.leader.kv_layout()
    }

    fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
        let plan = step_plan(self.step, batch);
        let logits = self.run(plan, |leader| leader.forward(batch))?;
        if let Some(experts) = &self.experts {
            experts.observe();
        }
        Ok(logits)
    }

    fn last_timings(&self) -> ForwardTimings {
        self.leader.last_timings()
    }

    fn reduces_logits(&self) -> bool {
        self.leader.reduces_logits()
    }

    /// Rank 0's decode graphs (every rank captures and replays the same keys).
    fn graph_counters(&self) -> turbine_model::executor::GraphCounters {
        self.leader.graph_counters()
    }

    /// Copies on rank 0's pool now and on every worker's before anything else runs there.
    fn copy_blocks(
        &mut self,
        kv: &turbine_tensor::KvPoolView<'_>,
        src: &[BlockId],
        dst: &[BlockId],
    ) -> Result<(), ModelError> {
        let plan = StepPlan {
            step: self.step,
            sequences: Vec::new(),
            copies: src.iter().copied().zip(dst.iter().copied()).collect(),
            ledger: Vec::new(),
        };
        self.run(plan, |leader| leader.copy_blocks(kv, src, dst))
    }
}

impl Drop for TpExecutor {
    /// `static` mode: the mirrors' last changes and digests go to the workers before the
    /// shutdown (a final check).
    fn drop(&mut self) {
        if !self.mirrors.is_empty() {
            let ledger = self.ledger_part(true);
            let _ = self.runtime.step(StepPlan {
                step: self.step,
                sequences: Vec::new(),
                copies: Vec::new(),
                ledger,
            });
        }
        self.runtime.shutdown("engine stopped");
    }
}

/// One worker rank: its executor over its shard, its KV pool and what must live as long as they
/// do (its communicator, ledger reservations and emergency reserve).
pub(crate) struct WorkerRank {
    rank: u32,
    exec: Box<dyn ModelExecutor>,
    pool: BlockPool,
    collective: Arc<dyn Collective>,
    faults: Arc<Fault>,
    /// The executor reduces logits on the device: ask for one candidate per row instead of
    /// copying whole rows to the host.
    reduce: bool,
    /// `static` mode: its real ledger following the leader's mirror (module comment).
    mirror: Option<MirrorCheck>,
    /// How it opens its communicator again (P5 Task 28); `None`: it cannot.
    comm: Option<CommSpec>,
    /// Its ledger, reservations and emergency reserve, held while the rank runs.
    _keep: Box<dyn Send>,
}

/// A `static` worker's side of the mirror ledger: its real ledger replaying the mirror's
/// changes, and where divergences are counted.
struct MirrorCheck {
    replica: LedgerReplica,
    metrics: ReliabilityMetrics,
}

impl MirrorCheck {
    /// Applies the plan's changes for `rank` and, when it carries a digest, compares.
    fn apply(&mut self, rank: u32, plan: &StepPlan) {
        for part in plan.ledger.iter().filter(|p| p.rank == rank) {
            let ops: Vec<LedgerOp> = part.changes.iter().map(|&c| ledger_op(c)).collect();
            if let Err(detail) = self.replica.apply(&ops) {
                self.diverged(rank, plan.step, "mirror_replay_failed", &detail);
            }
            if let Some(want) = part.digest {
                let have = self.replica.digest(PoolKind::Kv);
                if have != want {
                    let detail =
                        format!("leader's mirror digest {want:#018x}, ledger {have:#018x}");
                    self.diverged(rank, plan.step, "mirror_digest_mismatch", &detail);
                }
            }
        }
    }

    fn diverged(&self, rank: u32, step: u64, reason: &'static str, detail: &str) {
        let usage = self
            .replica
            .ledger()
            .usage(self.replica.device(), PoolKind::Kv);
        tracing::warn!(
            event = "ledger_mirror_divergence",
            reason,
            rank,
            step,
            used = usage.used,
            reserved = usage.reserved,
            detail,
            "this rank's KV ledger differs from the leader's mirror of it"
        );
        self.metrics
            .ledger_mirror_divergence
            .get_or_create(&RankLabel { rank })
            .inc();
    }
}

impl WorkerRank {
    fn new(
        rank: u32,
        exec: Box<dyn ModelExecutor>,
        pool: BlockPool,
        collective: Arc<dyn Collective>,
        faults: Arc<Fault>,
        keep: Box<dyn Send>,
    ) -> WorkerRank {
        WorkerRank {
            rank,
            reduce: exec.reduces_logits(),
            exec,
            pool,
            collective,
            faults,
            mirror: None,
            comm: None,
            _keep: keep,
        }
    }

    fn run(&mut self, plan: &StepPlan) -> Result<usize, ModelError> {
        let view = self.pool.view();
        if !plan.copies.is_empty() {
            let (src, dst): (Vec<BlockId>, Vec<BlockId>) = plan.copies.iter().copied().unzip();
            self.exec.copy_blocks(&view, &src, &dst)?;
        }
        if plan.sequences.is_empty() {
            return Ok(0);
        }
        #[cfg(feature = "fault-injection")]
        if abort_injected() {
            // The step's collectives then fail on the aborted communicator, as after a real
            // mid-generation failure of the backend.
            tracing::warn!(
                event = "fault_injection",
                rank = self.rank,
                "aborting this rank's communicator"
            );
            self.collective.abort();
        }
        let reduce = self.reduce.then_some(RowReduce {
            top_n: 1,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        });
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut slices = Vec::with_capacity(plan.sequences.len());
        for s in &plan.sequences {
            let (Some(&last), true) = (s.positions.last(), s.positions.len() == s.tokens.len())
            else {
                return Err(ModelError::Kernel(
                    turbine_kernels::KernelError::InvalidArgument {
                        message: format!(
                            "step {}: sequence {} has {} tokens and {} positions",
                            plan.step,
                            s.seq_id.0,
                            s.tokens.len(),
                            s.positions.len()
                        ),
                    },
                ));
            };
            slices.push(SeqSlice {
                seq: s.seq_id,
                q_start: tokens.len() as u32,
                q_len: s.tokens.len() as u32,
                kv_len: last + 1,
                block_table: &s.block_table,
                block_formats: &[],
                reduce,
            });
            tokens.extend_from_slice(&s.tokens);
            positions.extend_from_slice(&s.positions);
        }
        let logits = self.exec.forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &slices,
            kv: &view,
        })?;
        Ok(logits.slots())
    }
}

impl StepExecutor for WorkerRank {
    fn execute(&mut self, plan: &StepPlan) -> Result<StepOutput, ExecError> {
        // The mirror's changes first: they are the leader's reservations up to this step.
        if let Some(m) = &mut self.mirror {
            m.apply(self.rank, plan);
        }
        match self.run(plan) {
            Ok(rows) => Ok(StepOutput {
                logits: None,
                rows,
                vocab: self.exec.shape().vocab as usize,
            }),
            Err(e) => {
                // The other ranks may be waiting in a collective this rank will never reach.
                self.collective.abort();
                let message = format!("rank {}: {e}", self.rank);
                tracing::error!(event = "tp_rank_failed", rank = self.rank, step = plan.step, error = %e, "tensor-parallel worker rank failed");
                let sticky = matches!(&e, ModelError::Kernel(k) if k.is_sticky());
                lock(&self.faults).get_or_insert((self.rank, e));
                Err(if sticky {
                    ExecError::DeviceFatal(message)
                } else {
                    ExecError::Executor(message)
                })
            }
        }
    }

    /// Opens this rank of the new group and puts the executor on it (P5 Task 28).
    fn reinit(&mut self, unique_id: [u8; 128]) -> Result<(), ExecError> {
        let Some(comm) = &self.comm else {
            return Err(ExecError::Executor(format!(
                "rank {} cannot re-create its communicator",
                self.rank
            )));
        };
        let c = comm.open(unique_id).map_err(ExecError::Collective)?;
        if let Err(e) = self.exec.set_collective(Arc::clone(&c)) {
            c.abort();
            return Err(ExecError::Executor(format!("rank {}: {e}", self.rank)));
        }
        self.collective = c;
        // A static worker's own fault slot: the failure is over.
        lock(&self.faults).take();
        Ok(())
    }

    fn abort(&mut self) {
        self.collective.abort();
    }
}

/// What the engine thread needs to load a tensor-parallel group besides rank 0's prepared model.
pub(crate) struct TpGroupStart {
    /// Ranks 1.. in rank order, each prepared on its device (`local` mode).
    pub workers: Vec<PreparedModel>,
    /// The loaded collective backend (`parallel.collective_backend`).
    pub library: Arc<dyn CollectiveLibrary>,
    pub init_timeout: Duration,
    pub op_timeout: Duration,
    /// `parallel.collective.hostmem_max_bytes` (`None` = `auto`).
    pub route_max_bytes: Option<u64>,
    /// `parallel.plan_queue_depth`.
    pub depth: usize,
    pub metrics: CollectiveMetrics,
    pub clock: Arc<dyn Clock>,
    /// `static` rank mode (P5 S-5): the worker ranks are other processes that join this leader
    /// over the rank transport; `workers` is empty.
    pub remote: Option<StaticLeader>,
    /// Expert parallelism (P5 S-11): rank 0's token counts, reported after every step.
    pub experts: Option<Arc<ExpertStats>>,
}

/// The leader's side of a `static` group: where it listens and what every joining rank must
/// match (its `Hello`).
pub(crate) struct StaticLeader {
    pub transport: &'static dyn Transport,
    pub listen: SocketAddr,
    pub expect: HelloExpect,
    pub world: u32,
    /// Steps between two mirror-ledger digest checks ([`MIRROR_CHECK_STEPS`]).
    pub mirror_check_steps: u64,
}

/// The mirror of a worker's ledger from its reported `budget` (`Loaded`), with the worker's
/// startup commitments (weights, workspace, …) taken on it and kept out of its journal: the
/// worker holds its own.
fn mirror_of(
    rank: u32,
    b: &RankBudget,
    memory_kind: MemoryKind,
) -> Result<(DeviceBudget, Mirror, Vec<Reservation>), StartupError> {
    let pool = |name: &str| {
        PoolKind::ALL
            .into_iter()
            .find(|k| k.as_str() == name)
            .ok_or_else(|| StartupError::new(format!("rank {rank}: unknown pool {name:?}")))
    };
    let budget = DeviceBudget {
        device: DeviceId(b.device),
        memory_kind,
        budget_bytes: b.budget_bytes,
        pools: b
            .pools
            .iter()
            .map(|(name, bytes)| Ok((pool(name)?, *bytes)))
            .collect::<Result<_, StartupError>>()?,
    };
    let ledger = Ledger::mirror(&budget);
    let mut held = Vec::with_capacity(b.committed.len());
    for (name, bytes) in &b.committed {
        let mut r = ledger
            .reserve(budget.device, pool(name)?, *bytes)
            .map_err(|e| StartupError::new(format!("rank {rank}'s mirror ledger: {e}")))?;
        r.commit();
        held.push(r);
    }
    let _ = ledger.drain_journal();
    let mirror = Mirror {
        rank,
        device: budget.device,
        ledger,
    };
    Ok((budget, mirror, held))
}

/// What a worker reports to its leader once it loaded (`Loaded`): its budget, the bytes committed
/// outside the KV pool and its pool's geometry.
fn rank_budget(budget: &DeviceBudget, ledger: &Ledger, pool: &BlockPool) -> RankBudget {
    RankBudget {
        device: budget.device.0,
        budget_bytes: budget.budget_bytes,
        pools: budget
            .pools
            .iter()
            .map(|(k, bytes)| (k.as_str().to_string(), *bytes))
            .collect(),
        committed: budget
            .pools
            .iter()
            .filter(|(k, _)| *k != PoolKind::Kv)
            .map(|(k, _)| (k.as_str().to_string(), ledger.usage(budget.device, *k).used))
            .filter(|(_, used)| *used > 0)
            .collect(),
        block_bytes: pool.layout().block_bytes(),
        blocks: pool.total_blocks(),
    }
}

/// A `static`-mode worker rank process (P5 S-5): how it joins its leader and loads its shard.
pub(crate) struct StaticWorker {
    pub transport: &'static dyn Transport,
    pub leader: SocketAddr,
    /// This rank's `Hello` (rank, world size, fingerprints, vendor, architecture).
    pub hello: RankMessage,
    pub library: Arc<dyn CollectiveLibrary>,
    pub init_timeout: Duration,
    pub op_timeout: Duration,
    /// `parallel.collective.hostmem_max_bytes` (`None` = `auto`).
    pub route_max_bytes: Option<u64>,
    pub metrics: CollectiveMetrics,
    pub clock: Arc<dyn Clock>,
    /// Wraps the loaded executor (`std::convert::identity` in the server; tests inject
    /// failures).
    pub wrap: fn(Box<dyn ModelExecutor>) -> Box<dyn ModelExecutor>,
    /// Its share of the KV tiers (P5 Task 30).
    pub tiers: super::tp_tiers::WorkerTierStart,
}

impl StaticWorker {
    /// This worker's own rank number (from its `Hello`), for `startup::serve` to name it when
    /// its thread does not join in time at shutdown.
    pub(crate) fn rank(&self) -> u32 {
        match &self.hello {
            RankMessage::Hello { rank, .. } => *rank,
            _ => 0,
        }
    }
}

/// A rank after its load, before the group's warm-up.
struct RankLoaded {
    executor: Box<dyn ModelExecutor>,
    pool: BlockPool,
    weight_bytes: u64,
    budget: DeviceBudget,
    ledger: Arc<Ledger>,
    held: Vec<Reservation>,
    reserve: EmergencyReserve,
    collective: Arc<dyn Collective>,
    /// How to open the rank's communicator again (P5 Task 28).
    comm: CommSpec,
}

/// How one rank opens its communicator of a group: at load and again when a failed group is
/// re-created (P5 Task 28), each time with the group's new unique id.
#[derive(Clone)]
struct CommSpec {
    library: Arc<dyn CollectiveLibrary>,
    rank: usize,
    world: usize,
    init_timeout: Duration,
    op_timeout: Duration,
    route_max_bytes: Option<u64>,
    clock: Arc<dyn Clock>,
    metrics: CollectiveMetrics,
    /// The rank's device memory: its context is entered on the opening thread first (the
    /// NCCL-API init uses the thread's current device).
    memory: Arc<dyn DeviceMemory>,
}

impl CommSpec {
    /// Opens this rank of the group `unique_id`, bounded by the init timeout.
    fn open(&self, unique_id: [u8; 128]) -> Result<Arc<dyn Collective>, CollectiveError> {
        // Reading the free memory makes this thread's current device the rank's.
        let _ = self.memory.mem_info();
        Arc::clone(&self.library).open(CollectiveInit {
            rank: self.rank,
            world: self.world,
            unique_id,
            init_timeout: self.init_timeout,
            op_timeout: self.op_timeout,
            clock: Arc::clone(&self.clock),
            metrics: Some(self.metrics.clone()),
            memory: Some(Arc::clone(&self.memory)),
            route_max_bytes: self.route_max_bytes,
        })
    }
}

/// The group's block-count agreement: every rank proposes the blocks its budget holds and all
/// take the smallest; a failed rank releases the others.
struct Agreement {
    state: Mutex<(Vec<Option<u32>>, bool)>,
    cv: Condvar,
}

impl Agreement {
    fn new(world: usize) -> Agreement {
        Agreement {
            state: Mutex::new((vec![None; world], false)),
            cv: Condvar::new(),
        }
    }

    /// Rank `rank` proposes `blocks`; the smallest proposal once every rank made one, `None`
    /// when a rank failed.
    fn agree(&self, rank: usize, blocks: u32) -> Option<u32> {
        let mut s = lock(&self.state);
        s.0[rank] = Some(blocks);
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

/// How the ranks agree on the pool's block count: in this process (`local` rank mode), or
/// across processes through the communicator (`static`: an all-reduce of the negated proposals
/// with `Max`, exact in F32 below 2^24 blocks).
enum Agree<'a> {
    Local(&'a Agreement),
    Collective,
}

impl Agree<'_> {
    fn fail(&self) {
        if let Agree::Local(a) = self {
            a.fail();
        }
    }
}

/// The smallest `blocks` over every rank of `c`, by an all-reduce on `mem`'s compute stream.
fn agree_over(c: &dyn Collective, mem: &Arc<dyn DeviceMemory>, blocks: u32) -> Result<u32, String> {
    let mut buf = DeviceBuffer::alloc(mem, 4).map_err(|e| e.to_string())?;
    buf.copy_from_host(0, &(-(blocks as f32)).to_le_bytes())
        .map_err(|e| e.to_string())?;
    let stream = mem.compute_stream();
    c.step_begin();
    let mut slice = buf.whole();
    let reduced = c.all_reduce(&mut slice, DType::F32, ReduceOp::Max, &stream);
    let synced = mem.synchronize();
    let ended = c.step_end();
    reduced.map_err(|e| e.to_string())?;
    synced.map_err(|e| e.to_string())?;
    ended.map_err(|e| e.to_string())?;
    let mut out = [0u8; 4];
    buf.copy_to_host(0, &mut out).map_err(|e| e.to_string())?;
    Ok((-f32::from_le_bytes(out)) as u32)
}

/// Everything one rank's loader thread shares with the others.
struct GroupLoad<'a> {
    library: &'a Arc<dyn CollectiveLibrary>,
    unique_id: [u8; 128],
    world: u32,
    init_timeout: Duration,
    op_timeout: Duration,
    route_max_bytes: Option<u64>,
    clock: &'a Arc<dyn Clock>,
    metrics: &'a CollectiveMetrics,
    reliability: &'a ReliabilityMetrics,
    agreement: Agree<'a>,
    phase: &'a (dyn Fn(NotReadyReason) + Sync),
}

/// Loads rank `p.shard` (module comment): communicator, weight shard, budget, the agreed pool.
fn load_rank(p: &PreparedModel, g: &GroupLoad<'_>) -> Result<RankLoaded, StartupError> {
    let s = p.shard.unwrap_or(ShardSpec {
        rank: 0,
        world: g.world,
    });
    let rank_error = |what: &str, e: &dyn std::fmt::Display| {
        StartupError::new(format!(
            "rank {} of {} (device {}): {what}: {e}",
            s.rank, s.world, p.device.0
        ))
    };
    let mem = &p.provider.opened.mem;
    // Reading the free memory also makes this thread's current device the rank's (the kernel
    // library enters its context on every call), which the NCCL-API communicator init uses.
    let free = |what: &str| {
        mem.mem_info()
            .map(|m| m.free_bytes)
            .map_err(|e| rank_error(what, &e))
    };
    let before = free("device memory info")?;
    let comm = CommSpec {
        library: Arc::clone(g.library),
        rank: s.rank as usize,
        world: s.world as usize,
        init_timeout: g.init_timeout,
        op_timeout: g.op_timeout,
        route_max_bytes: g.route_max_bytes,
        clock: Arc::clone(g.clock),
        metrics: g.metrics.clone(),
        memory: Arc::clone(mem),
    };
    let collective = comm.open(g.unique_id).map_err(|e| {
        g.agreement.fail();
        rank_error("collective init", &e)
    })?;
    let collective_bytes = before.saturating_sub(free("device memory info")?);
    tracing::info!(
        event = "tp_rank_collective",
        rank = s.rank,
        world = s.world,
        device = p.device.0,
        backend = collective.backend(),
        collective_bytes,
        "tensor-parallel rank joined its communicator"
    );
    (g.phase)(NotReadyReason::LoadingWeights);
    let loaded = (|| {
        let weights = model::load_weights(p)?;
        let weight_bytes = weights.weight_bytes;
        let (budget, ledger, held) =
            model::post_load_budget(p, weight_bytes, collective_bytes, g.reliability)?;
        let mine = model::pool_blocks(p, &budget)?;
        let blocks = match &g.agreement {
            Agree::Local(a) => a
                .agree(s.rank as usize, mine)
                .ok_or_else(|| rank_error("load", &ANOTHER_RANK_FAILED))?,
            Agree::Collective => agree_over(collective.as_ref(), mem, mine)
                .map_err(|e| rank_error("KV pool agreement", &e))?,
        };
        if blocks < mine {
            tracing::info!(
                event = "tp_pool_agreed",
                rank = s.rank,
                proposed = mine,
                blocks,
                "the group's smallest KV pool sizes this rank's pool"
            );
        }
        let limits = ExecutorLimits {
            block_tokens: p.block_tokens,
            max_batch_tokens: p.scheduler.max_batch_tokens,
            max_seqs: p.scheduler.max_running_requests,
        };
        let tp_context = TpContext {
            rank: s.rank,
            world: s.world,
            collective: Arc::clone(&collective),
            stream: mem.compute_stream(),
        };
        let executor = match &p.expert {
            // Expert parallelism (P5 S-11): the rank's experts over the same communicator;
            // tensor-parallel attention at tp = ep, whose FFN all-reduce is the combine.
            Some(e) => ep::build_executor(
                &p.arch,
                weights,
                Arc::clone(&p.registry),
                Arc::clone(mem),
                limits,
                p.executor_options,
                EpContext {
                    rank: s.rank,
                    world: s.world,
                    placement: Arc::clone(&e.placement),
                    collective: Arc::clone(&collective),
                    stream: mem.compute_stream(),
                    counts: Arc::clone(&e.counts),
                },
                (e.shard.attention == EpAttention::TensorParallel).then_some(tp_context),
            ),
            None => tp::build_executor(
                &p.arch,
                weights,
                Arc::clone(&p.registry),
                Arc::clone(mem),
                limits,
                p.executor_options,
                tp_context,
            ),
        }
        .map_err(|e| rank_error("executor", &e))?;
        let mut executor = executor;
        model::install_decode_graphs(p, executor.as_mut());
        let pool = model::allocate_pool(p, blocks, &ledger)?;
        let reserve = model::acquire_reserve(p, &ledger, g.reliability)?;
        Ok(RankLoaded {
            executor,
            pool,
            weight_bytes,
            budget,
            ledger,
            held,
            reserve,
            collective: Arc::clone(&collective),
            comm: comm.clone(),
        })
    })();
    if loaded.is_err() {
        g.agreement.fail();
        collective.abort();
    }
    loaded
}

/// Loads a tensor-parallel group whose rank 0 is `leader` and warms it up (module comment):
/// the engine's [`LoadedModel`] with a [`TpExecutor`] and every worker's KV shard for the tier
/// copies. `phase` reports the loading step for `/ready` (`collective_init`,
/// `loading_weights`, then `loading_model` for the warm-up).
pub(crate) fn load_group(
    leader: &PreparedModel,
    group: TpGroupStart,
    warmup_token: u32,
    metrics: &ModelMetrics,
    reliability: &ReliabilityMetrics,
    phase: &(dyn Fn(NotReadyReason) + Sync),
) -> Result<LoadedModel, StartupError> {
    let started = Instant::now();
    let world = match &group.remote {
        Some(s) => s.world,
        None => group.workers.len() as u32 + 1,
    };
    let unique_id = group
        .library
        .unique_id()
        .map_err(|e| StartupError::new(format!("collective unique id: {e}")))?;
    // `static` mode: every worker process joins before anything is loaded (`rank_missing`
    // meanwhile; exit 1 naming the missing ranks after `parallel.collective.init_timeout`).
    let check_every = group
        .remote
        .as_ref()
        .map_or(MIRROR_CHECK_STEPS, |s| s.mirror_check_steps);
    let remote = match group.remote {
        Some(s) => {
            phase(NotReadyReason::RankMissing);
            tracing::info!(event = "tp_static_leader", listen = %s.listen, world, "waiting for the static worker ranks");
            let runtime = RankRuntime::static_leader(
                s.transport,
                s.listen,
                s.expect,
                world as usize,
                group.init_timeout,
                unique_id,
                group.depth,
            )
            .map_err(|e| StartupError::new(format!("static ranks: {e}")))?;
            Some(runtime)
        }
        None => None,
    };
    phase(NotReadyReason::CollectiveInit);
    let local = Agreement::new(group.workers.len() + 1);
    let load = GroupLoad {
        library: &group.library,
        unique_id,
        world,
        init_timeout: group.init_timeout,
        op_timeout: group.op_timeout,
        route_max_bytes: group.route_max_bytes,
        clock: &group.clock,
        metrics: &group.metrics,
        reliability,
        agreement: if remote.is_some() {
            Agree::Collective
        } else {
            Agree::Local(&local)
        },
        phase,
    };
    let ranks: Vec<&PreparedModel> = std::iter::once(leader).chain(&group.workers).collect();
    let results: Vec<Result<RankLoaded, StartupError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranks
            .iter()
            .enumerate()
            .map(|(r, p)| {
                let load = &load;
                std::thread::Builder::new()
                    .name(format!("turbine-rank-{r}-load"))
                    .spawn_scoped(scope, move || load_rank(p, load))
            })
            .collect();
        handles
            .into_iter()
            .enumerate()
            .map(|(r, h)| match h {
                Ok(h) => h
                    .join()
                    .unwrap_or_else(|_| Err(StartupError::new(format!("rank {r} load panicked")))),
                Err(e) => Err(StartupError::new(format!(
                    "cannot start the load thread of rank {r}: {e}"
                ))),
            })
            .collect()
    });
    let mut loaded = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for r in results {
        match r {
            Ok(rank) => loaded.push(rank),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        // The first failure that is not just another rank's echo names the cause.
        let at = errors
            .iter()
            .position(|e| !e.to_string().contains(ANOTHER_RANK_FAILED))
            .unwrap_or(0);
        return Err(errors.swap_remove(at));
    }
    let mut loaded = loaded.into_iter();
    let Some(rank0) = loaded.next() else {
        return Err(StartupError::new("tensor-parallel group without rank 0"));
    };
    let faults: Arc<Fault> = Arc::new(Mutex::new(None));
    let mut shards = Vec::with_capacity(group.workers.len());
    let mut ledgers = Vec::with_capacity(group.workers.len());
    let mut workers: Vec<Box<dyn StepExecutor>> = Vec::with_capacity(group.workers.len());
    for (i, (rank, p)) in loaded.zip(&group.workers).enumerate() {
        ledgers.push((rank.budget.clone(), Arc::clone(&rank.ledger)));
        shards.push(KvShard {
            device: super::copy_device(p),
            addresses: BlockAddresses::of(&rank.pool),
        });
        let mut worker = WorkerRank::new(
            i as u32 + 1,
            rank.executor,
            rank.pool,
            rank.collective,
            Arc::clone(&faults),
            Box::new((rank.held, rank.reserve, rank.ledger)),
        );
        worker.comm = Some(rank.comm);
        workers.push(Box::new(worker));
    }
    // `static` mode: a mirror of every worker's ledger from the budget it reports once loaded
    // (P5 Task 33), admitted through like a `local` rank's ledger.
    let mut mirrors = Vec::new();
    let mut mirror_held = Vec::new();
    if let Some(runtime) = &remote {
        let budgets = runtime
            .budgets(group.init_timeout)
            .map_err(|e| StartupError::new(format!("static ranks' budgets: {e}")))?;
        for (rank, b) in budgets {
            if b.blocks != rank0.pool.total_blocks() {
                return Err(StartupError::new(format!(
                    "rank {rank} reports {} KV blocks, rank 0 has {}",
                    b.blocks,
                    rank0.pool.total_blocks()
                )));
            }
            let (budget, mirror, held) = mirror_of(rank, &b, rank0.budget.memory_kind)?;
            tracing::info!(
                event = "ledger_mirror",
                rank,
                device = b.device,
                kv_bytes = budget.pool(PoolKind::Kv),
                "the leader mirrors this static rank's ledger for group admission"
            );
            ledgers.push((budget, Arc::clone(&mirror.ledger)));
            mirrors.push(mirror);
            mirror_held.extend(held);
        }
    }
    // `static` mode: the workers' own KV tiers, driven by the orchestrator (P5 Task 30).
    let remote_tiers = match &remote {
        Some(runtime) => {
            super::tp_tiers::leader_driver(runtime, group.init_timeout, group.op_timeout)?
        }
        None => None,
    };
    let runtime = match remote {
        Some(runtime) => runtime,
        None => RankRuntime::local(workers, group.depth),
    };
    let mut executor = TpExecutor::new(rank0.executor, runtime, rank0.collective, faults)
        .with_experts(group.experts.clone())
        .with_mirrors(mirrors, check_every, mirror_held)
        .with_comm(rank0.comm);
    let mut pool = rank0.pool;
    phase(NotReadyReason::LoadingModel);
    model::warm_up(&mut executor, &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    metrics.record_load(
        load_seconds,
        leader.arch.weight_format.0.name(),
        rank0.weight_bytes,
    );
    model::record_quantization(metrics, &leader.arch, rank0.weight_bytes);
    tracing::info!(
        event = "tp_group_ready",
        world,
        load_seconds,
        rank0_weight_bytes = rank0.weight_bytes,
        budget_bytes = rank0.budget.budget_bytes,
        kv_blocks = pool.total_blocks(),
        "tensor-parallel group loaded and warmed up"
    );
    Ok(LoadedModel {
        executor: Box::new(executor),
        pool,
        weight_bytes: rank0.weight_bytes,
        load_seconds,
        budget: rank0.budget,
        ledger: rank0.ledger,
        reserve: rank0.reserve,
        held: rank0.held,
        shards,
        remote_tiers,
        group: ledgers,
    })
}

/// A `static`-mode worker rank process (P5 S-5): joins its leader (`rank_missing` until the
/// leader welcomes it), opens the communicator with the leader's unique id, loads its shard,
/// agrees on the pool with the group, reports its budget (`Loaded`), calls `ready` with its ledger
/// and then executes every step plan (replaying the leader's mirror of its ledger) until the
/// leader shuts it down (`Ok`). A lost leader aborts the communicator and returns `Err` (the
/// process exits 1 and releases its device memory), as does a failed load or step.
pub(crate) fn run_static_worker(
    prepared: &PreparedModel,
    start: StaticWorker,
    reliability: &ReliabilityMetrics,
    phase: &(dyn Fn(NotReadyReason) + Sync),
    ready: impl FnOnce(&Arc<Ledger>),
) -> Result<(), String> {
    let s = prepared
        .shard
        .ok_or_else(|| "a static worker rank needs its shard".to_string())?;
    phase(NotReadyReason::RankMissing);
    let link = RankRuntime::static_worker(
        start.transport,
        start.leader,
        start.hello,
        start.init_timeout,
    )
    .map_err(|e| format!("rank {}: joining the leader {}: {e}", s.rank, start.leader))?;
    tracing::info!(event = "tp_static_joined", rank = s.rank, leader = %start.leader, "joined the leader");
    phase(NotReadyReason::CollectiveInit);
    let load = GroupLoad {
        library: &start.library,
        unique_id: link.unique_id(),
        world: s.world,
        init_timeout: start.init_timeout,
        op_timeout: start.op_timeout,
        route_max_bytes: start.route_max_bytes,
        clock: &start.clock,
        metrics: &start.metrics,
        reliability,
        agreement: Agree::Collective,
        phase,
    };
    let rank = load_rank(prepared, &load).map_err(|e| e.to_string())?;
    let mut link = link;
    // P5 Task 33: the leader mirrors this rank's ledger from the budget it reports now.
    link.loaded(rank_budget(&rank.budget, &rank.ledger, &rank.pool))
        .map_err(|e| format!("rank {}: reporting its budget: {e}", s.rank))?;
    let tiers = super::tp_tiers::start_worker_tiers(prepared, start.tiers, &rank.pool, &mut link)?;
    let ledger = Arc::clone(&rank.ledger);
    let mut worker = WorkerRank::new(
        s.rank,
        (start.wrap)(rank.executor),
        rank.pool,
        rank.collective,
        Arc::new(Mutex::new(None)),
        Box::new((rank.held, rank.reserve, rank.ledger)),
    );
    worker.comm = Some(rank.comm);
    worker.mirror = Some(MirrorCheck {
        replica: LedgerReplica::new(Arc::clone(&ledger), rank.budget.device),
        metrics: reliability.clone(),
    });
    tracing::info!(
        event = "tp_static_worker_ready",
        rank = s.rank,
        "worker rank loaded; executing step plans"
    );
    ready(&ledger);
    link.run_with_tiers(&mut worker, tiers)
        .map_err(|e| format!("rank {}: {e}", s.rank))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use turbine_core::types::{DeviceId, SeqId};
    use turbine_distributed::collective::{CollectiveBackend, HostCollective};
    use turbine_kernels::test_support::{plain_device_error, sticky_device_error};
    use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
    use turbine_kv::BlockPoolConfig;
    use turbine_model::executor::{self, ExecutorOptions};
    use turbine_model::testing::TempDir;
    use turbine_model::testing::tiny::{TinySpec, write_tiny_llama};
    use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, llama_slots};
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;

    const BLOCK_TOKENS: u32 = 16;
    const BLOCKS: u32 = 4;
    const LIMITS: ExecutorLimits = ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens: 64,
        max_seqs: 4,
    };

    fn host(device: u32) -> Arc<dyn DeviceMemory> {
        HostMemory::new(DeviceId(device), 1 << 30)
    }

    fn registry(reqs: &[turbine_kernels::OpRequirement]) -> Arc<KernelRegistry> {
        let provider = cpu_reference_provider();
        let order = [provider.id()];
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        Arc::new(KernelRegistry::build(vec![provider], &order, reqs, &metrics, None).unwrap())
    }

    /// The tiny Llama on one host device.
    fn one_device(spec: &TinySpec) -> (Box<dyn ModelExecutor>, BlockPool) {
        let (cfg, mem) = (&spec.config, host(0));
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
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: cfg.kv_layout(BLOCK_TOKENS),
                num_blocks: BLOCKS,
            },
            mem,
        )
        .unwrap();
        (exec, pool)
    }

    /// Rank `rank` of a `world`-rank group of the tiny Llama on its own host device.
    fn rank(
        spec: &TinySpec,
        rank: u32,
        world: u32,
        collective: Arc<dyn Collective>,
    ) -> (Box<dyn ModelExecutor>, BlockPool) {
        let (cfg, mem) = (&spec.config, host(rank));
        let s = ShardSpec { rank, world };
        let opts = ExecutorOptions::default();
        let index = SafetensorsIndex::open(&spec.dir).unwrap();
        let slots = tp::weight_slots(cfg, s).unwrap();
        let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("shard");
        let reqs = tp::requirements(cfg, s, BLOCK_TOKENS, opts).unwrap();
        let exec = tp::build_executor(
            cfg,
            weights,
            registry(&reqs),
            Arc::clone(&mem),
            LIMITS,
            opts,
            TpContext {
                rank,
                world,
                collective,
                stream: mem.compute_stream(),
            },
        )
        .unwrap();
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: tp::kv_layout(cfg, s, BLOCK_TOKENS).unwrap(),
                num_blocks: BLOCKS,
            },
            mem,
        )
        .unwrap();
        (exec, pool)
    }

    /// A worker executor whose `fail_at`-th forward fails with `error()` before it reaches its
    /// collectives.
    struct FailAt {
        inner: Box<dyn ModelExecutor>,
        forwards: usize,
        fail_at: usize,
        error: fn() -> ModelError,
    }

    impl ModelExecutor for FailAt {
        fn shape(&self) -> &ModelShape {
            self.inner.shape()
        }
        fn kv_layout(&self) -> &KvLayout {
            self.inner.kv_layout()
        }
        fn forward(&mut self, batch: &BatchInput<'_>) -> Result<Logits, ModelError> {
            self.forwards += 1;
            if self.forwards == self.fail_at {
                return Err((self.error)());
            }
            self.inner.forward(batch)
        }
        fn set_collective(&mut self, c: Arc<dyn Collective>) -> Result<(), ModelError> {
            self.inner.set_collective(c)
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

    /// A two-rank group over the host collective (op timeout `op_timeout`), rank 1 wrapped by
    /// `wrap`; returns the leader's executor and pool.
    fn group(
        spec: &TinySpec,
        op_timeout: Duration,
        wrap: impl FnOnce(Box<dyn ModelExecutor>) -> Box<dyn ModelExecutor>,
    ) -> (TpExecutor, BlockPool) {
        let mut comms = HostCollective::group(2, op_timeout).into_iter();
        let c0: Arc<dyn Collective> = Arc::new(comms.next().unwrap());
        let c1: Arc<dyn Collective> = Arc::new(comms.next().unwrap());
        let (e0, p0) = rank(spec, 0, 2, Arc::clone(&c0));
        let (e1, p1) = rank(spec, 1, 2, Arc::clone(&c1));
        let faults: Arc<Fault> = Arc::new(Mutex::new(None));
        let worker = WorkerRank::new(1, wrap(e1), p1, c1, Arc::clone(&faults), Box::new(()));
        let runtime = RankRuntime::local(vec![Box::new(worker)], 2);
        (TpExecutor::new(e0, runtime, c0, faults), p0)
    }

    fn argmax(row: &[f32]) -> u32 {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap() as u32
    }

    /// One sequence's step: `tokens` at positions `start..` over `table`; its greedy token.
    fn step(
        exec: &mut dyn ModelExecutor,
        view: &turbine_tensor::KvPoolView<'_>,
        tokens: &[u32],
        start: u32,
        table: &[BlockId],
    ) -> Result<u32, ModelError> {
        let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
        let seqs = [SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: tokens.len() as u32,
            kv_len: start + tokens.len() as u32,
            block_table: table,
            block_formats: &[],
            reduce: None,
        }];
        let logits = exec.forward(&BatchInput {
            tokens,
            positions: &positions,
            seqs: &seqs,
            kv: view,
        })?;
        Ok(argmax(logits.row(0)))
    }

    /// Prefills `prompt` into block 0, forks block 0 into block 2 with `copy_blocks` and decodes
    /// `steps` greedy tokens there (the prompt fits one block, so the fork holds the prefix).
    fn greedy_with_fork(
        exec: &mut dyn ModelExecutor,
        pool: &BlockPool,
        prompt: &[u32],
        steps: usize,
    ) -> Vec<u32> {
        let view = pool.view();
        let mut next = step(exec, &view, prompt, 0, &[BlockId(0)]).expect("prefill");
        exec.copy_blocks(&view, &[BlockId(0)], &[BlockId(2)])
            .expect("fork copy");
        let mut out = vec![next];
        for i in 0..steps {
            let at = (prompt.len() + i) as u32;
            next = step(exec, &view, &[next], at, &[BlockId(2)]).expect("decode");
            out.push(next);
        }
        out
    }

    /// P5 S-6 in the engine: the tiny Llama split over two ranks behind [`TpExecutor`] gives one
    /// device's greedy tokens, including after a fork copy — which must reach the worker's pool
    /// (the decode after it reads the copied block on every rank). Breaks if the step plan
    /// drops or misplaces tokens, positions or block tables, or if copies stay on rank 0.
    #[test]
    fn tp_executor_matches_one_device_with_fork_copies() {
        let dir = TempDir::new("engine-tp-match");
        let spec = write_tiny_llama(dir.path(), 11);
        let prompt: Vec<u32> = (0..10).map(|i| (i * 17 + 3) % spec.vocab).collect();
        let (mut one, one_pool) = one_device(&spec);
        let want = greedy_with_fork(one.as_mut(), &one_pool, &prompt, 5);
        let (mut tp, pool) = group(&spec, Duration::from_secs(60), |e| e);
        assert_eq!(*tp.kv_layout(), pool.layout());
        let got = greedy_with_fork(&mut tp, &pool, &prompt, 5);
        assert_eq!(got, want);
    }

    /// Rank `rank` of a tp 2 group of the tiny Llama prepared as the server prepares it: the
    /// cpu backend on host device `rank`, a 16 MiB pool, then `edit` on the configuration.
    fn prepared_with(
        spec: &TinySpec,
        rank: u32,
        edit: impl Fn(&mut turbine_core::config::Config),
    ) -> PreparedModel {
        let mut config = turbine_core::config::Config::default();
        config.model.path = spec.dir.clone();
        config.execution.backend = turbine_core::config::ModuleName::new("cpu").unwrap();
        config.execution.device = DeviceId(rank);
        config.kv.gpu.max_bytes = Some(turbine_core::config::ByteSize(16 << 20));
        config.reliability.emergency_vram_reserve = turbine_core::config::ByteSize(1 << 20);
        edit(&mut config);
        let inventory = turbine_device::DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        model::prepare_rank(
            &config,
            &inventory,
            &MetricsRegistry::new(),
            Some(model::RankPart::Tensor(ShardSpec { rank, world: 2 })),
        )
        .expect("prepare")
    }

    /// A tp 2 `static` group in one process: the leader's loaded model (rank 0) and the worker
    /// rank 1 on its own thread, joined over the `tcp` rank transport and the host collective.
    struct StaticRun {
        leader: PreparedModel,
        loaded: LoadedModel,
        /// The worker's `run_static_worker` result.
        worker: std::thread::JoinHandle<Result<(), String>>,
        /// The worker's real ledger, once it loaded.
        worker_ledger: Arc<Ledger>,
        /// The worker's metrics (its own registry, as in its own process).
        worker_metrics: MetricsRegistry,
        /// The leader's metrics.
        metrics: MetricsRegistry,
    }

    /// Starts a [`StaticRun`] over `spec` with `edit` on both ranks' configuration, a
    /// mirror-ledger check every `check_steps` steps, the worker's executor wrapped by `wrap`
    /// and its KV tiers `tiers`.
    fn start_static(
        spec: &TinySpec,
        edit: impl Fn(&mut turbine_core::config::Config) + Copy,
        check_steps: u64,
        wrap: fn(Box<dyn ModelExecutor>) -> Box<dyn ModelExecutor>,
        tiers: super::super::tp_tiers::WorkerTierStart,
    ) -> StaticRun {
        let (leader, worker) = (prepared_with(spec, 0, edit), prepared_with(spec, 1, edit));
        let library = turbine_distributed::collective::HostBackend
            .load(None)
            .expect("host library");
        let transport = turbine_distributed::transport::select("tcp").unwrap();
        let listen = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let expect = HelloExpect {
            model_fingerprint: leader.identity.fingerprint(),
            config_fingerprint: [7; 32],
            device_vendor: turbine_core::types::Vendor::Amd,
            device_arch: "test".into(),
        };
        let hello = RankMessage::Hello {
            protocol: turbine_distributed::rank::PROTOCOL_VERSION,
            rank: 1,
            world_size: 2,
            model_fingerprint: expect.model_fingerprint,
            config_fingerprint: expect.config_fingerprint,
            device_vendor: expect.device_vendor,
            device_arch: expect.device_arch.clone(),
        };
        let (reg, worker_reg) = (MetricsRegistry::new(), MetricsRegistry::new());
        let (metrics, reliability) = (
            CollectiveMetrics::register(&reg),
            ReliabilityMetrics::register(&reg),
        );
        let clock: Arc<dyn Clock> = Arc::new(turbine_core::clock::SystemClock::new());
        let worker_start = StaticWorker {
            transport,
            leader: listen,
            hello,
            library: Arc::clone(&library),
            init_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            route_max_bytes: None,
            metrics: CollectiveMetrics::register(&worker_reg),
            clock: Arc::clone(&clock),
            wrap,
            tiers,
        };
        let worker_reliability = ReliabilityMetrics::register(&worker_reg);
        let (ledger_tx, ledger_rx) = std::sync::mpsc::channel();
        let worker_thread = std::thread::spawn(move || {
            let phase = |_| {};
            run_static_worker(&worker, worker_start, &worker_reliability, &phase, |l| {
                let _ = ledger_tx.send(Arc::clone(l));
            })
        });
        let group = TpGroupStart {
            workers: Vec::new(),
            library,
            init_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            route_max_bytes: None,
            depth: 2,
            metrics,
            clock,
            remote: Some(StaticLeader {
                transport,
                listen,
                expect,
                world: 2,
                mirror_check_steps: check_steps,
            }),
            experts: None,
        };
        let model_metrics = ModelMetrics::register(&reg);
        let loaded = load_group(&leader, group, 0, &model_metrics, &reliability, &|_| {})
            .expect("static group");
        let worker_ledger = ledger_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the worker loaded");
        StaticRun {
            leader,
            loaded,
            worker: worker_thread,
            worker_ledger,
            worker_metrics: worker_reg,
            metrics: reg,
        }
    }

    /// P5 S-5, `static` rank mode in one process: a leader group with a remote rank that joins
    /// over the `tcp` rank transport (Hello → Welcome with the leader's unique id), both
    /// loading, agreeing on the pool through the communicator and warming up; the leader's
    /// executor then gives one device's greedy tokens (fork copy included), and dropping it
    /// shuts the worker down cleanly (`Ok`). Breaks if the handshake, the collective pool
    /// agreement or the step plans over the socket are wrong.
    #[test]
    fn static_group_over_tcp_matches_one_device() {
        let dir = TempDir::new("engine-tp-static");
        let spec = write_tiny_llama(dir.path(), 11);
        let prompt: Vec<u32> = (0..10).map(|i| (i * 17 + 3) % spec.vocab).collect();
        let (mut one, one_pool) = one_device(&spec);
        let want = greedy_with_fork(one.as_mut(), &one_pool, &prompt, 5);

        let run = start_static(
            &spec,
            |_| {},
            MIRROR_CHECK_STEPS,
            std::convert::identity,
            super::super::tp_tiers::WorkerTierStart::off(),
        );
        assert!(
            run.loaded.shards.is_empty(),
            "no tier shards across processes"
        );
        assert_eq!(run.loaded.group.len(), 1, "the worker's mirror ledger");
        let LoadedModel {
            mut executor, pool, ..
        } = run.loaded;
        let got = greedy_with_fork(executor.as_mut(), &pool, &prompt, 5);
        assert_eq!(got, want);
        drop(executor);
        let stopped = run.worker.join().expect("worker thread");
        assert_eq!(
            stopped,
            Ok(()),
            "the leader's shutdown stops the worker cleanly"
        );
    }

    /// The engine over a loaded group, built as `engine::spawn` builds it (KV hierarchy, the
    /// reliability side admitting through the group's ledgers, the scheduler behind its gate),
    /// its command channel and published documents.
    fn engine_over(
        prepared: &PreparedModel,
        loaded: LoadedModel,
    ) -> (
        crate::engine::EngineLoop,
        tokio::sync::mpsc::Sender<crate::engine::EngineCommand>,
        Arc<crate::engine::EngineShared>,
        MetricsRegistry,
    ) {
        let kv_cfg = turbine_core::config::KvConfig {
            block_tokens: loaded.pool.layout().block_tokens,
            ..turbine_core::config::KvConfig::default()
        };
        engine_over_kv(prepared, loaded, &kv_cfg, |_| None, |_| {})
    }

    /// [`engine_over`] with the `kv` section `kv_cfg`, the L2 tier `open_l2` opens with the
    /// engine's KV metrics, and `on_kv` shown the KV orchestrator before the engine takes it.
    fn engine_over_kv(
        prepared: &PreparedModel,
        mut loaded: LoadedModel,
        kv_cfg: &turbine_core::config::KvConfig,
        open_l2: impl FnOnce(turbine_kv::KvMetrics) -> Option<Arc<turbine_kv::tier::L2NvmeTier>>,
        on_kv: impl FnOnce(&crate::kv_orchestrator::KvOrchestrator),
    ) -> (
        crate::engine::EngineLoop,
        tokio::sync::mpsc::Sender<crate::engine::EngineCommand>,
        Arc<crate::engine::EngineShared>,
        MetricsRegistry,
    ) {
        let reg = MetricsRegistry::new();
        let metrics = crate::engine::EngineMetrics {
            server: crate::metrics::ServerMetrics::register(&reg),
            model: ModelMetrics::register(&reg),
            scheduler: turbine_scheduler::SchedulerMetrics::register(&reg),
            kv: turbine_kv::KvMetrics::register(&reg),
        };
        let l2 = open_l2(metrics.kv.clone());
        let clock: Arc<dyn Clock> = Arc::new(turbine_core::clock::SystemClock::new());
        let mut pool = loaded.pool;
        let (kv, _handle) = crate::kv_orchestrator::KvOrchestrator::start(
            crate::kv_orchestrator::KvStart {
                cfg: kv_cfg,
                memory_kind: MemoryKind::Dedicated,
                identity: prepared.identity,
                device: crate::engine::copy_device(prepared),
                shards: std::mem::take(&mut loaded.shards),
                l2,
                clock: Arc::clone(&clock),
                metrics: metrics.kv.clone(),
                remote: loaded.remote_tiers.take(),
                kv_scales: None,
            },
            &mut pool,
        )
        .expect("the KV hierarchy starts");
        on_kv(&kv);
        let parts = crate::reliability::build(crate::reliability::ReliabilityInputs {
            config: &prepared.reliability,
            budget: loaded.budget,
            ledger: Arc::clone(&loaded.ledger),
            reserve: loaded.reserve,
            held: loaded.held,
            params: &prepared.scheduler,
            block_bytes: pool.layout().block_bytes(),
            workspace_bytes_per_token: 0,
            metrics: ReliabilityMetrics::register(&reg),
            clock: Arc::clone(&clock),
            reclaimer: kv.reclaimer(),
            replica: 0,
            group: loaded.group,
        });
        let scheduler = turbine_scheduler::Scheduler::new(prepared.scheduler, Arc::clone(&clock))
            .with_metrics(metrics.scheduler.clone())
            .with_gate(parts.gate);
        let (tx, commands) = tokio::sync::mpsc::channel(64);
        let shared = Arc::new(crate::engine::EngineShared::default());
        let engine = crate::engine::EngineLoop::new(crate::engine::EngineParts {
            executor: loaded.executor,
            pool,
            scheduler,
            clock,
            commands,
            shared: Arc::clone(&shared),
            tokenizer: Arc::clone(&prepared.tokenizer),
            max_seq_len: prepared.max_seq_len,
            metrics,
            timeouts: crate::engine::Timeouts {
                request: Duration::from_secs(600),
                slow_client: Duration::from_secs(600),
            },
            overlap: false,
            reliability: parts.engine,
            kv,
            pipeline: None,
        });
        (engine, tx, shared, reg)
    }

    /// A greedy request for `max_tokens` tokens of `prompt` with `n` choices.
    fn greedy_request(
        prompt: Vec<u32>,
        max_tokens: u32,
        n: u32,
    ) -> turbine_core::request::GenerationRequest {
        use turbine_core::request::{Endpoint, GenerationRequest, SamplingParams, StopConditions};
        GenerationRequest {
            id: turbine_core::types::RequestId::new_v4(),
            n,
            priority: turbine_core::types::Priority::default(),
            echo: false,
            constraint: None,
            deadline_ms: u64::MAX,
            session: None,
            cache_salt: None,
            kv_policy: None,
            endpoint: Endpoint::Completions,
            http_request_id: "t".into(),
            prompt_tokens: prompt,
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

    type Events = tokio::sync::mpsc::Receiver<turbine_core::request::GenerationEvent>;

    /// Queues `req` with an output channel of `capacity` events; waits for its admission ack.
    fn submit(
        tx: &tokio::sync::mpsc::Sender<crate::engine::EngineCommand>,
        req: turbine_core::request::GenerationRequest,
        capacity: usize,
    ) -> Events {
        let (events, rx) = tokio::sync::mpsc::channel(capacity);
        let (ack, admitted) = tokio::sync::oneshot::channel();
        tx.blocking_send(crate::engine::EngineCommand::Submit(
            Box::new(req.into()),
            events,
            ack,
        ))
        .unwrap_or_else(|_| panic!("engine gone"));
        assert_eq!(admitted.blocking_recv().expect("ack"), Ok(()));
        rx
    }

    /// Reads a stream to its end: the tokens, or the error code.
    fn read_all(rx: &mut Events) -> Result<usize, String> {
        use turbine_core::request::GenerationEvent;
        let mut tokens = 0;
        while let Some(e) = rx.blocking_recv() {
            match e {
                GenerationEvent::Token { .. } => tokens += 1,
                GenerationEvent::Finished { .. } => return Ok(tokens),
                GenerationEvent::Error { code, message } => {
                    return Err(format!("{code:?}: {message}"));
                }
                _ => {}
            }
        }
        Err("stream closed without an end".into())
    }

    /// The sum of `series`' samples in a registry's rendering.
    fn sum_series(reg: &MetricsRegistry, series: &str) -> f64 {
        reg.render()
            .unwrap()
            .lines()
            .filter(|l| l.starts_with(series))
            .filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok())
            .sum()
    }

    /// Polls `cond` every 10 ms; panics naming `what` after `limit`.
    fn wait_until(limit: Duration, what: &str, mut cond: impl FnMut() -> bool) {
        let started = Instant::now();
        while !cond() {
            assert!(started.elapsed() < limit, "{what}: not within {limit:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// P5 Task 33 (decision "P5: group KV admission in static rank mode" B), the plan's
    /// `tiny_server tp2_static_mirror_matches_worker_ledger` in one process (the host collective
    /// joins threads of one process only): a tp 2 `static` group on the cpu backend over
    /// loopback `tcp` runs a mixed workload through the engine — 18 concurrent greedy requests,
    /// `n` = 2 forks among them, 4 cancelled mid-stream, on a 32-block pool whose worst-case
    /// reservations queue them (`kv_reservation`; worst-case admission leaves no decode short of
    /// a block, so nothing is preempted) — with the mirror's digest checked at every step: no
    /// divergence is counted; then two held streams pause holding KV, and the worker's real
    /// ledger equals the leader's mirror exactly (live reservations, nothing in flight). A
    /// deliberately skewed worker ledger makes the final check (the last plan before shutdown)
    /// log `mirror_digest_mismatch` once. Breaks if admission skips the mirror, a reservation,
    /// commit or release misses the worker or arrives out of order, or a divergence goes unseen.
    #[test]
    fn static_mirror_matches_worker_ledger() {
        let dir = TempDir::new("engine-tp-mirror");
        let spec = write_tiny_llama(dir.path(), 11);
        let edit = |c: &mut turbine_core::config::Config| {
            c.kv.block_tokens = 16;
            // 32 blocks of 2 KiB per rank (one of the tiny model's two KV heads), one per running
            // request at most (forks included: the executor holds `max_running_requests`
            // sequences).
            c.kv.gpu.max_bytes = Some(turbine_core::config::ByteSize(64 << 10));
            c.scheduler.max_running_requests = 32;
            c.model.max_seq_len = Some(256);
        };
        let run = start_static(
            &spec,
            edit,
            1,
            std::convert::identity,
            super::super::tp_tiers::WorkerTierStart::off(),
        );
        let StaticRun {
            leader,
            loaded,
            worker,
            worker_ledger,
            worker_metrics,
            metrics: _leader_metrics,
        } = run;
        assert_eq!(loaded.group.len(), 1, "one worker, one mirror");
        let mirror = Arc::clone(&loaded.group[0].1);
        let device = loaded.group[0].0.device;
        assert_eq!(loaded.pool.total_blocks(), 32, "the small pool");
        let (engine, tx, shared, engine_metrics) = engine_over(&leader, loaded);
        let engine = std::thread::spawn(move || engine.run());
        let divergences = || sum_series(&worker_metrics, "turbine_ledger_mirror_divergence_total{");

        // Phase 1: the mixed workload, to the end.
        let streams: Vec<(usize, Events)> = (0..18usize)
            .map(|i| {
                let prompt: Vec<u32> = std::iter::once(256)
                    .chain((0..(6 + (i * 5) % 24) as u32).map(|t| 97 + (t * 7 + i as u32) % 26))
                    .collect();
                let n = if i % 5 == 2 { 2 } else { 1 };
                let max_tokens = 8 + ((i * 11) % 40) as u32;
                (i, submit(&tx, greedy_request(prompt, max_tokens, n), 256))
            })
            .collect();
        let readers: Vec<_> = streams
            .into_iter()
            .map(|(i, mut rx)| {
                std::thread::spawn(move || {
                    if i % 4 == 1 {
                        // Cancelled mid-stream: the client goes away after a few tokens.
                        let mut seen = 0;
                        while seen < 3 {
                            match rx.blocking_recv() {
                                Some(turbine_core::request::GenerationEvent::Token { .. }) => {
                                    seen += 1;
                                }
                                Some(_) => {}
                                None => break,
                            }
                        }
                        drop(rx);
                        return Ok(0);
                    }
                    read_all(&mut rx)
                })
            })
            .collect();
        for (i, r) in readers.into_iter().enumerate() {
            let result = r.join().expect("reader");
            assert!(result.is_ok(), "request {i}: {result:?}");
        }
        wait_until(Duration::from_secs(30), "the workload drained", || {
            shared.docs().is_some_and(|d| {
                d.scheduler.waiting + d.scheduler.prefilling + d.scheduler.decoding == 0
            })
        });
        let snapshot = shared.docs().unwrap().scheduler;
        let queued = sum_series(
            &engine_metrics,
            r#"turbine_admission_decisions_total{decision="queue",reason="kv_reservation"}"#,
        );
        eprintln!(
            "mirror workload: {} iterations, {} preemptions, {queued} kv_reservation queueings",
            snapshot.iterations_total, snapshot.preemptions_total
        );
        assert!(queued > 0.0, "the small pool queued admissions");
        assert_eq!(divergences(), 0.0, "no divergence at any step");

        // Phase 2: two held streams pause holding KV; the step that ran them carried every
        // change before them, and nothing changes while they are paused.
        let held: Vec<Events> = (0..2)
            .map(|i| {
                let prompt: Vec<u32> = std::iter::once(256).chain(100..110 + i).collect();
                submit(&tx, greedy_request(prompt, 120, 1), 4)
            })
            .collect();
        wait_until(Duration::from_secs(30), "both held streams paused", || {
            shared.docs().is_some_and(|d| {
                d.scheduler.paused == 2
                    && d.scheduler.waiting + d.scheduler.prefilling + d.scheduler.decoding == 0
            })
        });
        let live = mirror.usage(device, PoolKind::Kv);
        assert!(
            live.used + live.reserved > 0,
            "the held streams hold KV: {live:?}"
        );
        assert_eq!(
            worker_ledger.usage(device, PoolKind::Kv),
            live,
            "the worker's real ledger is the mirror, exactly"
        );
        assert_eq!(divergences(), 0.0);

        // Phase 3: skew the worker's ledger; the final plan's check sees it once.
        let skew = worker_ledger
            .reserve(device, PoolKind::Kv, 1)
            .expect("skew the worker's ledger");
        drop(held);
        drop(tx);
        assert_eq!(engine.join().expect("engine thread"), Ok(()));
        assert_eq!(
            worker.join().expect("worker thread"),
            Ok(()),
            "the leader's shutdown stops the worker cleanly"
        );
        assert_eq!(
            divergences(),
            1.0,
            "the skew is seen by the final check, once"
        );
        drop(skew);
        assert_eq!(
            worker_ledger.usage(device, PoolKind::Kv),
            mirror.usage(device, PoolKind::Kv),
            "every release reached the worker"
        );
    }

    /// A worker executor whose third forward (after the warm-up and a prefill) fails with a plain
    /// device error before its collectives.
    fn fail_third(inner: Box<dyn ModelExecutor>) -> Box<dyn ModelExecutor> {
        Box::new(FailAt {
            inner,
            forwards: 0,
            fail_at: 3,
            error: || ModelError::Kernel(plain_device_error("injected")),
        })
    }

    /// P5 Task 28 in `static` rank mode: a worker process whose step fails stays up (it reports
    /// the failure and aborts its communicator); the leader's step fails as a collective
    /// failure, and its next step re-creates the communicator on both ranks (fresh unique id,
    /// `Reinit` over the rank link) and then runs: the greedy tokens are one device's, the retried
    /// decode included, and the worker still shuts down cleanly. Breaks if a failed worker exits,
    /// the group is not re-created or a rank stays on its aborted communicator.
    #[test]
    fn static_group_recovers_from_a_failed_step() {
        let dir = TempDir::new("engine-tp-static-recover");
        let spec = write_tiny_llama(dir.path(), 11);
        let prompt: Vec<u32> = (0..10).map(|i| (i * 17 + 3) % spec.vocab).collect();
        let table = [BlockId(0)];
        let plen = prompt.len() as u32;
        let (mut one, one_pool) = one_device(&spec);
        let want = {
            let view = one_pool.view();
            let t0 = step(one.as_mut(), &view, &prompt, 0, &table).unwrap();
            let t1 = step(one.as_mut(), &view, &[t0], plen, &table).unwrap();
            let t2 = step(one.as_mut(), &view, &[t1], plen + 1, &table).unwrap();
            vec![t0, t1, t2]
        };

        let run = start_static(
            &spec,
            |_| {},
            MIRROR_CHECK_STEPS,
            fail_third,
            super::super::tp_tiers::WorkerTierStart::off(),
        );
        let LoadedModel {
            mut executor, pool, ..
        } = run.loaded;
        let view = pool.view();
        let t0 = step(executor.as_mut(), &view, &prompt, 0, &table).expect("prefill");
        let err = step(executor.as_mut(), &view, &[t0], plen, &table).expect_err("rank 1 fails");
        assert!(matches!(err, ModelError::Collective(_)), "{err}");
        let t1 = step(executor.as_mut(), &view, &[t0], plen, &table).expect("re-created");
        let t2 = step(executor.as_mut(), &view, &[t1], plen + 1, &table).expect("decode");
        assert_eq!(vec![t0, t1, t2], want);
        drop(executor);
        assert_eq!(
            run.worker.join().expect("worker thread"),
            Ok(()),
            "the failed worker stayed up and shut down cleanly"
        );
    }

    /// A worker that fails before its collectives aborts the group at once (well inside the
    /// 60 s op timeout): the leader's step fails as a collective failure (`replica_failed`), and
    /// every later step fails too. A worker's sticky device error comes back as that error, so
    /// the process still exits 3. Breaks if a failed rank leaves the leader waiting out the op
    /// timeout or hides a sticky error.
    #[test]
    fn failed_worker_aborts_the_group() {
        let dir = TempDir::new("engine-tp-fail");
        let spec = write_tiny_llama(dir.path(), 11);
        let plain: fn() -> ModelError = || ModelError::Kernel(plain_device_error("injected"));
        let sticky: fn() -> ModelError = || ModelError::Kernel(sticky_device_error("injected"));
        for (error, is_sticky) in [(plain, false), (sticky, true)] {
            let (mut tp, pool) = group(&spec, Duration::from_secs(60), |inner| {
                Box::new(FailAt {
                    inner,
                    forwards: 0,
                    fail_at: 2,
                    error,
                })
            });
            let view = pool.view();
            let table = [BlockId(0)];
            step(&mut tp, &view, &[5, 6, 7], 0, &table).expect("first step");
            let started = Instant::now();
            let err = step(&mut tp, &view, &[1], 3, &table).expect_err("rank 1 fails");
            assert!(started.elapsed() < Duration::from_secs(20), "{err}");
            match (&err, is_sticky) {
                (ModelError::Kernel(k), true) => assert!(k.is_sticky(), "{err}"),
                (ModelError::Collective(_), false) => {}
                _ => panic!("sticky {is_sticky}: {err}"),
            }
            let again = step(&mut tp, &view, &[1], 3, &table).expect_err("stays failed");
            assert!(matches!(again, ModelError::Collective(_)), "{again}");
        }
    }

    /// A request's greedy tokens with their logprobs, and `usage.cached_tokens`, read to its
    /// end.
    fn tokens_and_cached(rx: &mut Events) -> (Vec<(u32, f32)>, u32) {
        use turbine_core::request::GenerationEvent;
        let mut tokens = Vec::new();
        while let Some(e) = rx.blocking_recv() {
            match e {
                GenerationEvent::Token {
                    token_id, logprob, ..
                } => tokens.push((token_id, logprob.expect("logprobs requested"))),
                GenerationEvent::Finished { usage, .. } => {
                    return (tokens, usage.expect("usage").cached_tokens);
                }
                GenerationEvent::Error { code, message } => panic!("{code:?}: {message}"),
                _ => {}
            }
        }
        panic!("stream closed without an end");
    }

    /// P5 Task 30 (decision "P5: KV tiers in static rank mode" B), the plan's tiny-server check
    /// in one process (the host collective joins threads of one process only): a tp 2 `static`
    /// group on the cpu backend over loopback `tcp`, each rank with its own L2 directory
    /// (`rank-<r>`), serves a multi-turn session whose first turn's prefix went to L2 on both
    /// ranks in between: the second turn reports the prefix's 32 `cached_tokens`, its blocks come
    /// back from L2 (`promotions{l2,l0}`) and its greedy tokens and logprobs equal a cold run of
    /// the same prompt (under a cache salt, so nothing is shared) bit for bit. Fillers run
    /// through every block in between, so stale bytes cannot pass for restored ones. Breaks if a
    /// rank's shard does not go to or come back from its own L2 (the worker would attend over
    /// other bytes and the all-reduced logprobs would drift), or the leader reuses blocks the
    /// worker never restored.
    #[test]
    fn static_tiers_serve_a_demoted_prefix() {
        use super::super::tp_tiers::{self, WorkerTierStart};

        let dir = TempDir::new("engine-tp-static-tiers");
        let spec = write_tiny_llama(dir.path(), 11);
        let edit = |c: &mut turbine_core::config::Config| {
            c.kv.block_tokens = 16;
            // 32 blocks of 2 KiB per rank: the fillers below overwrite every block.
            c.kv.gpu.max_bytes = Some(turbine_core::config::ByteSize(64 << 10));
            c.scheduler.max_running_requests = 32;
            c.model.max_seq_len = Some(256);
        };
        let mut kv = turbine_core::config::KvConfig {
            block_tokens: 16,
            ..turbine_core::config::KvConfig::default()
        };
        // The cpu backend has no copy stream: L1 is off, L2 goes through synchronous copies.
        kv.cpu.enabled = false;
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = turbine_core::config::ByteSize(16 << 20);
        kv.nvme.slab_bytes = turbine_core::config::ByteSize(1 << 20);
        let (mut kv0, mut kv1) = (kv.clone(), kv);
        tp_tiers::static_rank_kv(&mut kv0, 0, 2);
        tp_tiers::static_rank_kv(&mut kv1, 1, 2);
        let rank1 = prepared_with(&spec, 1, edit);
        let clock: Arc<dyn Clock> = Arc::new(turbine_core::clock::SystemClock::new());
        let worker_l2 = tp_tiers::open_rank_l2(
            &kv1,
            rank1.pool.layout,
            2,
            &rank1.identity,
            Arc::clone(&clock),
            turbine_kv::KvMetrics::register(&MetricsRegistry::new()),
        )
        .expect("the worker's L2 opens");
        let run = start_static(
            &spec,
            edit,
            MIRROR_CHECK_STEPS,
            std::convert::identity,
            WorkerTierStart {
                kv: kv1,
                l2: worker_l2,
            },
        );
        let StaticRun {
            leader,
            loaded,
            worker,
            ..
        } = run;
        assert!(loaded.remote_tiers.is_some(), "the leader drives the tiers");
        let mut reclaim = None;
        let (engine, tx, _shared, reg) = engine_over_kv(
            &leader,
            loaded,
            &kv0,
            |metrics| {
                tp_tiers::open_rank_l2(
                    &kv0,
                    leader.pool.layout,
                    2,
                    &leader.identity,
                    Arc::clone(&clock),
                    metrics,
                )
                .expect("the leader's L2 opens")
            },
            |kv| reclaim = Some(kv.reclaimer()),
        );
        let reclaim = reclaim.expect("the KV orchestrator started");
        let engine = std::thread::spawn(move || engine.run());
        let run_one = |prompt: &[u32], salt: Option<&str>| {
            let mut req = greedy_request(prompt.to_vec(), 8, 1);
            req.cache_salt = salt.map(str::to_string);
            req.sampling.logprobs = Some(0);
            tokens_and_cached(&mut submit(&tx, req, 64))
        };

        // Turn 1: 40 tokens (two full blocks), twice for the reuse evidence demotion needs.
        let turn1: Vec<u32> = std::iter::once(256).chain(97..136).collect();
        let (answer, cached) = run_one(&turn1, None);
        assert_eq!(cached, 0);
        let (again, cached) = run_one(&turn1, None);
        assert_eq!((again.as_slice(), cached), (answer.as_slice(), 32));
        // Turn 2 of the session: the whole conversation so far and a new user message; its cold
        // tokens under a cache salt first.
        let turn2: Vec<u32> = turn1
            .iter()
            .chain(answer.iter().map(|(t, _)| t))
            .copied()
            .chain(140..150)
            .collect();
        let (cold, cached) = run_one(&turn2, Some("cold"));
        assert_eq!(cached, 0, "a salted run shares nothing");

        // Every unreferenced block leaves L0 for each rank's own L2 at the end of a turn.
        reclaim.demote(0.0);
        let _ = run_one(&[256, 1, 2], None);
        let demoted = r#"turbine_kv_demotions_total{from="l0",to="l2"}"#;
        wait_until(
            Duration::from_secs(20),
            "turn 1's blocks reached L2",
            || sum_series(&reg, demoted) >= 2.0,
        );
        // Other prompts then run through every block of both pools, so a rank that did not
        // restore turn 1's blocks from its L2 would attend over their bytes.
        for i in 0..8u32 {
            let filler: Vec<u32> = std::iter::once(256)
                .chain((0..100).map(|t| 97 + (t * 7 + i * 3) % 26))
                .collect();
            let _ = run_one(&filler, None);
        }
        let promoted = r#"turbine_kv_promotions_total{from="l2",to="l0"}"#;
        let before = sum_series(&reg, promoted);

        let (warm, cached) = run_one(&turn2, None);
        assert_eq!(cached, 32, "turn 1's prefix came back from L2");
        assert!(
            sum_series(&reg, promoted) >= before + 2.0,
            "both blocks were promoted from L2"
        );
        // The cpu provider's rows are independent of the batch: the warm run's logprobs are the
        // cold run's bit for bit (a worker attending over stale blocks drifts them by ~1e-3
        // without changing a greedy token of the tiny model).
        assert_eq!(
            warm, cold,
            "KV that went through both ranks' L2 gives the cold run's tokens and logprobs"
        );

        drop(tx);
        assert_eq!(engine.join().expect("engine thread"), Ok(()));
        assert_eq!(
            worker.join().expect("worker thread"),
            Ok(()),
            "the leader's shutdown stops the worker cleanly"
        );
    }

    /// What [`flood_scenario`] saw: every later turn's `cached_tokens` in the first pass, each
    /// resumed session's, the L2 -> L0 promotions during the resume, and each resumed session's
    /// greedy tokens.
    #[derive(Debug)]
    struct Flood {
        later: Vec<u32>,
        resumed: Vec<u32>,
        promoted: f64,
        /// Admission plans `retrieve_cheaper` and `recompute_cheaper` over the whole scenario.
        plans: (f64, f64),
        /// Each resumed session's greedy tokens.
        answers: Vec<Vec<u32>>,
    }

    /// Run 8 of the Task 30 lab on the host, over an engine's command channel and its metrics:
    /// 4 multi-turn sessions of 3 turns, one-off prompts that cycle a 32-block L0 many times,
    /// two prompts that need the whole pool at once, then each session's next turn (its whole
    /// history, which only the first pass wrote).
    fn flood_scenario(
        label: &str,
        tx: &tokio::sync::mpsc::Sender<crate::engine::EngineCommand>,
        shared: &crate::engine::EngineShared,
        reg: &MetricsRegistry,
    ) -> Flood {
        let run_one = |prompt: &[u32]| {
            let mut rx = submit(tx, greedy_request(prompt.to_vec(), 8, 1), 64);
            tokens_and_cached_plain(&mut rx)
        };
        let counters = |stage: &str| {
            let text = reg.render().unwrap();
            for l in text.lines().filter(|l| {
                [
                    "promotions",
                    "demotions",
                    "drops",
                    "plans",
                    "transfer_bandwidth_bytes_per_second",
                    "transfer_seconds_sum",
                    "transfer_seconds_count",
                ]
                .iter()
                .any(|m| l.starts_with(&format!("turbine_kv_{m}")))
            }) {
                if !l.ends_with(" 0") {
                    eprintln!("{label} {stage}: {l}");
                }
            }
        };
        let system: Vec<u32> = std::iter::once(256).chain(97..128).collect();
        let mut histories = Vec::new();
        let mut later = Vec::new();
        for s in 0..4u32 {
            let mut history = system.clone();
            for turn in 0..3u32 {
                history.extend((0..12).map(|t| 97 + (t * 5 + s * 7 + turn * 3) % 26));
                let (answer, cached, _) = run_one(&history);
                if turn > 0 {
                    later.push(cached);
                }
                history.extend(answer);
            }
            histories.push(history);
        }
        counters("after the sessions");
        for i in 0..24u32 {
            let filler: Vec<u32> = std::iter::once(256)
                .chain((0..100).map(|t| 97 + (t * 11 + i * 13) % 26))
                .collect();
            let _ = run_one(&filler);
        }
        counters("after the flood");
        let wide: Vec<_> = (0..2u32)
            .map(|i| {
                let prompt: Vec<u32> = std::iter::once(256)
                    .chain((0..230).map(|t| 97 + (t * 17 + i * 5) % 26))
                    .collect();
                submit(tx, greedy_request(prompt, 8, 1), 64)
            })
            .collect();
        for mut rx in wide {
            let _ = tokens_and_cached_plain(&mut rx);
        }
        counters("after the wide prompts");
        let promoted = r#"turbine_kv_promotions_total{from="l2",to="l0"}"#;
        let before = sum_series(reg, promoted);
        let resumed_pairs: Vec<(Vec<u32>, u32, Duration)> = histories
            .iter_mut()
            .enumerate()
            .map(|(s, history)| {
                history.extend((0..12).map(|t| 97 + (t * 3 + s as u32) % 26));
                run_one(history)
            })
            .collect();
        let resumed = resumed_pairs.iter().map(|(_, c, _)| *c).collect();
        let answers: Vec<Vec<u32>> = resumed_pairs.into_iter().map(|(t, _, _)| t).collect();
        counters("after the resume");
        // Nothing stays in flight once the engine is idle: every copy, every rank's part of it
        // included, completes (a lost acknowledgement would hold its bytes forever).
        wait_until(Duration::from_secs(10), "every tier copy completed", || {
            shared.docs().is_some_and(|d| {
                d.kv.summary
                    .as_ref()
                    .is_some_and(|k| k.transfers.inflight_bytes == 0)
            })
        });
        let plan = |reason: &str| {
            sum_series(
                reg,
                &format!("turbine_kv_plans_total{{reason=\"{reason}\"}}"),
            )
        };
        let out = Flood {
            later,
            resumed,
            promoted: sum_series(reg, promoted) - before,
            plans: (plan("retrieve_cheaper"), plan("recompute_cheaper")),
            answers,
        };
        eprintln!("{label}: {out:?}");
        out
    }

    /// The `kv` section of the flood tests: L2 only (the cpu backend has no copy stream).
    fn flood_kv(dir: &TempDir) -> turbine_core::config::KvConfig {
        let mut kv = turbine_core::config::KvConfig {
            block_tokens: 16,
            ..turbine_core::config::KvConfig::default()
        };
        kv.cpu.enabled = false;
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = turbine_core::config::ByteSize(16 << 20);
        kv.nvme.slab_bytes = turbine_core::config::ByteSize(1 << 20);
        kv
    }

    fn flood_edit(c: &mut turbine_core::config::Config) {
        c.kv.block_tokens = 16;
        c.kv.gpu.max_bytes = Some(turbine_core::config::ByteSize(64 << 10));
        c.scheduler.max_running_requests = 32;
        c.model.max_seq_len = Some(256);
    }

    /// [`flood_scenario`] on a tp 2 `local` group (the leader copies both ranks' shards): the
    /// control of `static_tiers_flood_then_resume`.
    fn flood_local() -> Flood {
        let dir = TempDir::new("engine-tp-local-flood");
        let spec = write_tiny_llama(dir.path(), 11);
        let kv = flood_kv(&dir);
        let (leader, worker) = (
            prepared_with(&spec, 0, flood_edit),
            prepared_with(&spec, 1, flood_edit),
        );
        let reg = MetricsRegistry::new();
        let group = TpGroupStart {
            workers: vec![worker],
            library: turbine_distributed::collective::HostBackend
                .load(None)
                .expect("host library"),
            init_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            route_max_bytes: None,
            depth: 2,
            metrics: CollectiveMetrics::register(&reg),
            clock: Arc::new(turbine_core::clock::SystemClock::new()),
            remote: None,
            experts: None,
        };
        let loaded = load_group(
            &leader,
            group,
            0,
            &ModelMetrics::register(&reg),
            &ReliabilityMetrics::register(&reg),
            &|_| {},
        )
        .expect("local group");
        let clock: Arc<dyn Clock> = Arc::new(turbine_core::clock::SystemClock::new());
        let (engine, tx, shared, ereg) = engine_over_kv(
            &leader,
            loaded,
            &kv,
            |metrics| {
                crate::kv_orchestrator::open_l2(
                    &kv,
                    &crate::kv_orchestrator::tp_kv_format(leader.pool.layout, 2),
                    &leader.identity,
                    clock,
                    metrics,
                )
                .expect("L2 opens")
            },
            |_| {},
        );
        let engine = std::thread::spawn(move || engine.run());
        let out = flood_scenario("local", &tx, &shared, &ereg);
        drop(tx);
        assert_eq!(engine.join().expect("engine thread"), Ok(()));
        out
    }

    /// [`flood_scenario`] on a tp 2 `static` group, each rank with its own L2.
    fn flood_static() -> Flood {
        use super::super::tp_tiers::{self, WorkerTierStart};

        let dir = TempDir::new("engine-tp-static-flood");
        let spec = write_tiny_llama(dir.path(), 11);
        let kv = flood_kv(&dir);
        let (mut kv0, mut kv1) = (kv.clone(), kv);
        tp_tiers::static_rank_kv(&mut kv0, 0, 2);
        tp_tiers::static_rank_kv(&mut kv1, 1, 2);
        let rank1 = prepared_with(&spec, 1, flood_edit);
        let clock: Arc<dyn Clock> = Arc::new(turbine_core::clock::SystemClock::new());
        let worker_l2 = tp_tiers::open_rank_l2(
            &kv1,
            rank1.pool.layout,
            2,
            &rank1.identity,
            Arc::clone(&clock),
            turbine_kv::KvMetrics::register(&MetricsRegistry::new()),
        )
        .expect("the worker's L2 opens");
        let StaticRun {
            leader,
            loaded,
            worker,
            ..
        } = start_static(
            &spec,
            flood_edit,
            MIRROR_CHECK_STEPS,
            std::convert::identity,
            WorkerTierStart {
                kv: kv1,
                l2: worker_l2,
            },
        );
        let (engine, tx, shared, reg) = engine_over_kv(
            &leader,
            loaded,
            &kv0,
            |metrics| {
                tp_tiers::open_rank_l2(
                    &kv0,
                    leader.pool.layout,
                    2,
                    &leader.identity,
                    Arc::clone(&clock),
                    metrics,
                )
                .expect("the leader's L2 opens")
            },
            |_| {},
        );
        let engine = std::thread::spawn(move || engine.run());
        let out = flood_scenario("static", &tx, &shared, &reg);
        drop(tx);
        assert_eq!(engine.join().expect("engine thread"), Ok(()));
        assert_eq!(worker.join().expect("worker thread"), Ok(()));
        out
    }

    /// Run 8 of the Task 30 lab on the host (cpu backend, tp 2, a 32-block L0 per rank, L2):
    /// the `static` group, each rank with its own L2, behaves as the `local` group whose leader
    /// copies both shards — every later turn reuses its session's previous one, and after a
    /// flood and two pool-wide prompts each resumed session reuses its prefix, both modes
    /// promote the survivors from L2, and the resumed sessions serve the same greedy tokens;
    /// no copy is left in flight once idle. Breaks if static mode loses the sessions' blocks,
    /// never completes a promotion, reuses less than local mode, serves different tokens, or
    /// leaks a copy (a lost acknowledgement would hold its bytes in flight). The resumed
    /// cached-token counts are floored, not compared: on the cpu backend the planner's
    /// retrieve-vs-recompute choice flips between runs (a copy and a prefill both take about
    /// one engine turn), and since the GREEN admission headroom (P3 S-9 amendment 2026-10-02)
    /// queues a burst against live utilization the flips reach the resume — local resumed
    /// [32, 32, 32, 48] while static resumed [32, 48, 48, 48] on the same workload. What pins
    /// static ≡ local through that schedule noise is the shared system prefix coming back in
    /// both modes (32 tokens), a promotion happening in both, and the identical greedy answers:
    /// a retrieved and a recomputed block hold identical bytes, so the resumed sessions must
    /// serve the same outputs.
    #[test]
    fn static_tiers_flood_then_resume() {
        let local = flood_local();
        let stat = flood_static();
        assert!(
            stat.later.iter().all(|&c| c > 0),
            "later turns reuse: {stat:?}"
        );
        assert_eq!(stat.later, local.later, "static reuses as local does");
        // The resumed cached-token vector is schedule-dependent (see the doc comment): the two
        // shared system blocks are the floor every run must clear.
        let floor = 32;
        assert!(
            stat.resumed.iter().all(|&c| c >= floor),
            "static resumes with the shared prefix: {stat:?}"
        );
        assert!(
            local.resumed.iter().all(|&c| c >= floor),
            "local resumes with the shared prefix: {local:?}"
        );
        assert!(
            stat.promoted > 0.0 && local.promoted > 0.0,
            "both modes promote from L2"
        );
        assert_eq!(
            stat.answers, local.answers,
            "static serves local's tokens: the same greedy answers"
        );
        // The retrieve / recompute choices are printed, not compared: on the cpu backend a copy
        // and a block's prefill both take about one engine turn (~2 ms), so either mode's
        // choices flip between runs. `tp_tiers::tests::copy_time_is_the_ranks_own_not_the_ack_delay`
        // pins what static mode feeds its estimates.
        eprintln!(
            "plans (retrieve, recompute): local {:?}, static {:?}",
            local.plans, stat.plans
        );
    }

    /// A request's greedy tokens and `usage.cached_tokens` (no logprobs), read to its end; the
    /// third field is filled by the caller.
    fn tokens_and_cached_plain(rx: &mut Events) -> (Vec<u32>, u32, Duration) {
        use turbine_core::request::GenerationEvent;
        let mut tokens = Vec::new();
        while let Some(e) = rx.blocking_recv() {
            match e {
                GenerationEvent::Token { token_id, .. } => tokens.push(token_id),
                GenerationEvent::Finished { usage, .. } => {
                    return (tokens, usage.expect("usage").cached_tokens, Duration::ZERO);
                }
                GenerationEvent::Error { code, message } => panic!("{code:?}: {message}"),
                _ => {}
            }
        }
        panic!("stream closed without an end");
    }
}
