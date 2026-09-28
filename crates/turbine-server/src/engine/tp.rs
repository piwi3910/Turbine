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
//! A worker's sticky device error is returned as that error, so the process still exits 3. The
//! communicator is not re-created: the replica stays failed until the process restarts (the
//! circuit's probes fail against the aborted communicator).
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
//! executes plans until the leader shuts it down or is lost (exit 1). There is no per-step
//! acknowledgement across processes, so KV tiers (which copy every rank's shard from the
//! leader) are off in static mode.

use std::net::SocketAddr;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use turbine_api::backend::NotReadyReason;
use turbine_core::clock::Clock;
use turbine_core::types::{BlockId, KvLayout, ModelShape};
use turbine_distributed::collective::{
    Collective, CollectiveError, CollectiveInit, CollectiveLibrary, CollectiveMetrics, ReduceOp,
};
use turbine_distributed::rank::{
    ExecError, HelloExpect, RankError, RankMessage, RankRuntime, StepExecutor, StepOutput,
    StepPlan, StepSeq,
};
use turbine_distributed::transport::Transport;
use turbine_kv::BlockPool;
use turbine_model::executor::{
    BatchInput, ExecutorLimits, ForwardTimings, Logits, ModelExecutor, RowReduce, SeqSlice,
};
use turbine_model::tp::{self, ShardSpec, TpContext};
use turbine_model::{ModelError, ModelMetrics};
use turbine_reliability::budget::DeviceBudget;
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_reliability::metrics::ReliabilityMetrics;
use turbine_reliability::reserve::EmergencyReserve;
use turbine_tensor::{DType, DeviceBuffer, DeviceMemory};

use crate::kv_orchestrator::{BlockAddresses, KvShard};
use crate::model::{self, LoadedModel, PreparedModel, StartupError};

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
}

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
        }
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

    /// Hands `plan` to the workers, runs `own` on rank 0 and waits until every worker is idle.
    fn run<T>(
        &mut self,
        plan: StepPlan,
        own: impl FnOnce(&mut dyn ModelExecutor) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        self.step += 1;
        if let Err(e) = self.runtime.step(plan) {
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
        self.run(plan, |leader| leader.forward(batch))
    }

    fn last_timings(&self) -> ForwardTimings {
        self.leader.last_timings()
    }

    fn reduces_logits(&self) -> bool {
        self.leader.reduces_logits()
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
        };
        self.run(plan, |leader| leader.copy_blocks(kv, src, dst))
    }
}

impl Drop for TpExecutor {
    fn drop(&mut self) {
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
    /// Its ledger, reservations and emergency reserve, held while the rank runs.
    _keep: Box<dyn Send>,
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
}

/// What the engine thread needs to load a tensor-parallel group besides rank 0's prepared model.
pub(crate) struct TpGroupStart {
    /// Ranks 1.. in rank order, each prepared on its device (`local` mode).
    pub workers: Vec<PreparedModel>,
    /// The loaded collective backend (`parallel.collective_backend`).
    pub library: Arc<dyn CollectiveLibrary>,
    pub init_timeout: Duration,
    pub op_timeout: Duration,
    /// `parallel.plan_queue_depth`.
    pub depth: usize,
    pub metrics: CollectiveMetrics,
    pub clock: Arc<dyn Clock>,
    /// `static` rank mode (P5 S-5): the worker ranks are other processes that join this leader
    /// over the rank transport; `workers` is empty.
    pub remote: Option<StaticLeader>,
}

/// The leader's side of a `static` group: where it listens and what every joining rank must
/// match (its `Hello`).
pub(crate) struct StaticLeader {
    pub transport: &'static dyn Transport,
    pub listen: SocketAddr,
    pub expect: HelloExpect,
    pub world: u32,
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
    pub metrics: CollectiveMetrics,
    pub clock: Arc<dyn Clock>,
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
    let collective = Arc::clone(g.library)
        .open(CollectiveInit {
            rank: s.rank as usize,
            world: s.world as usize,
            unique_id: g.unique_id,
            init_timeout: g.init_timeout,
            op_timeout: g.op_timeout,
            clock: Arc::clone(g.clock),
            metrics: Some(g.metrics.clone()),
        })
        .map_err(|e| {
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
        let executor = tp::build_executor(
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
            TpContext {
                rank: s.rank,
                world: s.world,
                collective: Arc::clone(&collective),
                stream: mem.compute_stream(),
            },
        )
        .map_err(|e| rank_error("executor", &e))?;
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
        workers.push(Box::new(WorkerRank::new(
            i as u32 + 1,
            rank.executor,
            rank.pool,
            rank.collective,
            Arc::clone(&faults),
            Box::new((rank.held, rank.reserve, rank.ledger)),
        )));
    }
    let runtime = match remote {
        Some(runtime) => runtime,
        None => RankRuntime::local(workers, group.depth),
    };
    let mut executor = TpExecutor::new(rank0.executor, runtime, rank0.collective, faults);
    let mut pool = rank0.pool;
    phase(NotReadyReason::LoadingModel);
    model::warm_up(&mut executor, &mut pool, warmup_token)?;
    let load_seconds = started.elapsed().as_secs_f64();
    metrics.record_load(
        load_seconds,
        leader.arch.weight_format.0.name(),
        rank0.weight_bytes,
    );
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
        group: ledgers,
    })
}

/// A `static`-mode worker rank process (P5 S-5): joins its leader (`rank_missing` until the
/// leader welcomes it), opens the communicator with the leader's unique id, loads its shard,
/// agrees on the pool with the group, calls `ready` and then executes every step plan until the
/// leader shuts it down (`Ok`). A lost leader aborts the communicator and returns `Err` (the
/// process exits 1 and releases its device memory), as does a failed load or step.
pub(crate) fn run_static_worker(
    prepared: &PreparedModel,
    start: StaticWorker,
    reliability: &ReliabilityMetrics,
    phase: &(dyn Fn(NotReadyReason) + Sync),
    ready: impl FnOnce(),
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
        clock: &start.clock,
        metrics: &start.metrics,
        reliability,
        agreement: Agree::Collective,
        phase,
    };
    let rank = load_rank(prepared, &load).map_err(|e| e.to_string())?;
    let collective = Arc::clone(&rank.collective);
    let mut worker = WorkerRank::new(
        s.rank,
        rank.executor,
        rank.pool,
        rank.collective,
        Arc::new(Mutex::new(None)),
        Box::new((rank.held, rank.reserve, rank.ledger)),
    );
    tracing::info!(
        event = "tp_static_worker_ready",
        rank = s.rank,
        "worker rank loaded; executing step plans"
    );
    ready();
    link.run(&mut worker, collective.as_ref())
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
    /// cpu backend on host device `rank`, a 16 MiB pool.
    fn prepared(spec: &TinySpec, rank: u32) -> PreparedModel {
        let mut config = turbine_core::config::Config::default();
        config.model.path = spec.dir.clone();
        config.execution.backend = turbine_core::config::ModuleName::new("cpu").unwrap();
        config.execution.device = DeviceId(rank);
        config.kv.gpu.max_bytes = Some(turbine_core::config::ByteSize(16 << 20));
        config.reliability.emergency_vram_reserve = turbine_core::config::ByteSize(1 << 20);
        let inventory = turbine_device::DeviceInventory {
            devices: Vec::new(),
            backends: Vec::new(),
        };
        model::prepare_rank(
            &config,
            &inventory,
            &MetricsRegistry::new(),
            Some(ShardSpec { rank, world: 2 }),
        )
        .expect("prepare")
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

        let (leader, worker) = (prepared(&spec, 0), prepared(&spec, 1));
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
        let reg = MetricsRegistry::new();
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
            metrics: metrics.clone(),
            clock: Arc::clone(&clock),
        };
        let worker_reliability = reliability.clone();
        let worker_thread = std::thread::spawn(move || {
            let phase = |_| {};
            run_static_worker(&worker, worker_start, &worker_reliability, &phase, || {})
        });
        let group = TpGroupStart {
            workers: Vec::new(),
            library,
            init_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            depth: 2,
            metrics,
            clock,
            remote: Some(StaticLeader {
                transport,
                listen,
                expect,
                world: 2,
            }),
        };
        let model_metrics = ModelMetrics::register(&reg);
        let loaded = load_group(&leader, group, 0, &model_metrics, &reliability, &|_| {})
            .expect("static group");
        assert!(loaded.shards.is_empty(), "no tier shards across processes");
        let LoadedModel {
            mut executor, pool, ..
        } = loaded;
        let got = greedy_with_fork(executor.as_mut(), &pool, &prompt, 5);
        assert_eq!(got, want);
        drop(executor);
        let stopped = worker_thread.join().expect("worker thread");
        assert_eq!(
            stopped,
            Ok(()),
            "the leader's shutdown stops the worker cleanly"
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
}
