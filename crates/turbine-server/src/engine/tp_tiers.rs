//! KV tiers in `static` rank mode (P5 Task 30, decision "P5: KV tiers in static rank mode" B).
//!
//! In `local` mode the leader copies every rank's shard of a block itself, on each worker's
//! context ([`crate::kv_orchestrator::KvShard`]). In `static` mode the other ranks are other
//! processes, so each process keeps its own tiers: its own pinned L1 (its share of
//! `kv.cpu.max_bytes`, `/ tensor_parallel_size`, pinned through its own context) and its own L2
//! under `kv.nvme.path/rank-<r>` (its share of `kv.nvme.max_bytes`), each holding that rank's
//! shard of a block — the same per-rank split as `local` mode, whose L1 is one pinned tier per
//! rank. The leader's KV orchestrator keeps the only directory and policy and drives the
//! workers over the rank link:
//!
//! - every copy the leader's transfer engine starts on its own tiers ([`StaticBackend`]) is
//!   also queued as a [`TierCopy::Copy`] of the same key and logical block, and every eviction
//!   from its L1/L2 ([`MirroredTier`]) as a [`TierCopy::Evict`]; the queue goes out as one
//!   bounded [`RankMessage::TierCopy`] batch per engine turn (after the transfer pump and at the
//!   end of the turn), on the links the step plans use, so a worker applies it after every
//!   step before it — a copy out of a block reads what those steps wrote;
//! - each worker applies the batch to its own pool and tiers on its link's tier thread
//!   ([`WorkerTiers`], [`turbine_distributed::rank::WorkerLink::run_with_tiers`]), overlapping its
//!   steps, and acknowledges each copy once done;
//! - the leader completes a copy only when its own copy and every worker's have finished: the
//!   source block of a demotion stays unallocatable and the target of a promotion unused until
//!   then, exactly as with `local` mode's copy streams. If any rank failed, the copy fails for
//!   all (the ranks that made a copy drop it again) and the hierarchy recomputes;
//! - the transfer estimates learn the copy's time as its slowest rank's own: the leader's copy
//!   as its poll saw it (as in `local` mode) or a worker's as the worker timed it
//!   ([`TierAck::took_ns`]), not the extra engine turn the acknowledgement took to arrive, which
//!   would read every static copy as slower and tilt admission towards recomputing;
//! - the wait is bounded by `parallel.collective.op_timeout`: a worker that does not answer
//!   within it fails the group (`RankError`), as a lost worker (a closed link) or a failed step
//!   (`StepFailed`, P5 Task 28) already does, so the leader's next step fails like any failed
//!   rank's (circuit `collective_failed`, then the group's re-creation while probing). A copy in
//!   flight while the group fails fails for every rank — once each rank's part is over, since
//!   the workers' tier threads keep answering while they wait for `Reinit` (without waiting
//!   when a link closed: that worker is gone) — and no copy starts until the group is
//!   re-created.
//!
//! A tier the leader or any worker lacks (no copy stream for L1, `kv.nvme.enabled: false`) is off
//! for the whole group with a WARN (`kv_tier_unavailable_on_rank`), as `local` mode needs a copy
//! stream on every rank for L1. The leader's hierarchy counts in its own rank's shard bytes (its
//! tiers hold rank 0's shard), and the KV namespace carries the tensor-parallel size
//! ([`crate::kv_orchestrator::tp_kv_format`]) as in `local` mode.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use turbine_core::clock::Clock;
use turbine_core::config::{ByteSize, KvConfig};
use turbine_core::types::{KvLayout, MemoryKind, ModelIdentity, PressureState};
use turbine_distributed::rank::{
    RankError, RankRuntime, TierAck, TierCopy, TierLevel, TierLink, TierPath, TierWorker,
    WorkerLink,
};
use turbine_kv::identity::{KvKey, namespace_key};
use turbine_kv::tier::{
    KvTier, L1Config, L1PinnedTier, L2Config, L2NvmeTier, ShardedL1Tier, TierBlockMut,
    TierBlockRef, TierError, TierId, TierSlot,
};
use turbine_kv::transfer::{
    TransferBackend, TransferCodec, TransferPath, TransferPurpose, TransferRequest, TransferTicket,
};
use turbine_kv::{BlockPool, KvMetrics};

use crate::kv_orchestrator::{
    BlockAddresses, CopyDevice, CopyStreamBackend, IoPoolBackend, KvShard, L1_SLAB_BYTES,
    tp_kv_format,
};
use crate::model::{PreparedModel, StartupError};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `static` rank `rank` of `tp`: this process's share of the tiers (module comment) — L1 and
/// L2 caps divided by `tp`, L2 under `kv.nvme.path/rank-<rank>` (ranks on one host never share
/// slab files, which each wipes at startup).
pub(crate) fn static_rank_kv(kv: &mut KvConfig, rank: u32, tp: u32) {
    let tp = u64::from(tp.max(1));
    kv.cpu.max_bytes = ByteSize(kv.cpu.max_bytes.0 / tp);
    kv.nvme.max_bytes = ByteSize(kv.nvme.max_bytes.0 / tp);
    kv.nvme.path = kv.nvme.path.join(format!("rank-{rank}"));
    if kv.cpu.enabled && kv.cpu.max_bytes.0 < L1_SLAB_BYTES {
        tracing::warn!(
            event = "kv_l1_share_below_slab",
            tier = "l1",
            rank,
            max_bytes = kv.cpu.max_bytes.0,
            slab_bytes = L1_SLAB_BYTES,
            "kv.cpu.max_bytes / tensor_parallel_size is below one L1 slab per rank; L1 holds \
             nothing (raise kv.cpu.max_bytes to at least tensor_parallel_size GiB)"
        );
    }
}

/// Opens this `static` rank's L2 before the listener binds (exit 1 naming the path, as
/// [`crate::kv_orchestrator::open_l2`]): slots of one rank shard of a block (`layout` is this
/// rank's pool), in the namespace of the group's `tp`-shard format. `None` when
/// `kv.nvme.enabled` is false.
pub(crate) fn open_rank_l2(
    cfg: &KvConfig,
    layout: KvLayout,
    tp: u32,
    identity: &ModelIdentity,
    clock: Arc<dyn Clock>,
    metrics: KvMetrics,
) -> Result<Option<Arc<L2NvmeTier>>, StartupError> {
    if !cfg.nvme.enabled {
        return Ok(None);
    }
    let l2 = L2NvmeTier::open(
        L2Config {
            path: cfg.nvme.path.clone(),
            max_bytes: cfg.nvme.max_bytes.0,
            slab_bytes: cfg.nvme.slab_bytes.0,
            max_queue_depth: cfg.nvme.max_queue_depth,
            block_bytes: layout.block_bytes(),
            namespace: namespace_key(identity, &tp_kv_format(layout, tp), ""),
        },
        clock,
        metrics,
    )
    .map_err(|e| StartupError::new(format!("kv.nvme.path {}: {e}", cfg.nvme.path.display())))?;
    tracing::info!(
        event = "kv_l2_opened",
        path = %cfg.nvme.path.display(),
        max_bytes = cfg.nvme.max_bytes.0,
        direct_io = l2.uses_direct_io(),
        shard_bytes = layout.block_bytes(),
        "L2 NVMe tier ready (this static rank's shards)"
    );
    Ok(Some(Arc::new(l2)))
}

fn wire_path(p: TransferPath) -> TierPath {
    match p {
        TransferPath::L0ToL1 => TierPath::L0ToL1,
        TransferPath::L1ToL0 => TierPath::L1ToL0,
        TransferPath::L1ToL2 => TierPath::L1ToL2,
        TransferPath::L2ToL1 => TierPath::L2ToL1,
        TransferPath::L0ToL2 => TierPath::L0ToL2,
        TransferPath::L2ToL0 => TierPath::L2ToL0,
    }
}

fn kv_path(p: TierPath) -> TransferPath {
    match p {
        TierPath::L0ToL1 => TransferPath::L0ToL1,
        TierPath::L1ToL0 => TransferPath::L1ToL0,
        TierPath::L1ToL2 => TransferPath::L1ToL2,
        TierPath::L2ToL1 => TransferPath::L2ToL1,
        TierPath::L0ToL2 => TransferPath::L0ToL2,
        TierPath::L2ToL0 => TransferPath::L2ToL0,
    }
}

/// The L0 block a copy reads (out of L0) or writes (into L0); 0 for L1 ↔ L2.
fn l0_block(req: &TransferRequest) -> u64 {
    match (req.path.from(), req.path.to()) {
        (TierId::L0, _) => req.src_slot,
        (_, TierId::L0) => req.dst_slot,
        _ => 0,
    }
}

/// A block key that no prompt produces in practice, for the startup slab check.
const PREFLIGHT_KEY: KvKey = KvKey([0xC5; 16]);

/// What a `static` worker process needs for its tiers (its share of the `kv` section and the L2
/// it opened before the listener bound), handed to [`start_worker_tiers`].
pub(crate) struct WorkerTierStart {
    pub kv: KvConfig,
    pub l2: Option<Arc<L2NvmeTier>>,
}

impl WorkerTierStart {
    /// No tiers (tests of the rank protocol without KV tiers).
    #[cfg(test)]
    pub(crate) fn off() -> WorkerTierStart {
        let mut kv = KvConfig::default();
        kv.cpu.enabled = false;
        kv.nvme.enabled = false;
        WorkerTierStart { kv, l2: None }
    }
}

/// A `static` worker's tiers once its pool is allocated: builds [`WorkerTiers`] (exit 1, as on
/// the leader, when `kv.cpu.enabled` and the first pinned slab cannot be allocated), tells the
/// leader which tiers it has (`TierReady`) and returns them for the link's tier thread; `None`
/// when neither tier is configured.
pub(crate) fn start_worker_tiers(
    prepared: &PreparedModel,
    start: WorkerTierStart,
    pool: &BlockPool,
    link: &mut WorkerLink,
) -> Result<Option<Box<dyn TierWorker>>, String> {
    let rank = link.rank();
    if !start.kv.cpu.enabled && start.l2.is_none() {
        link.tiers_ready(false, false)
            .map_err(|e| format!("rank {rank}: reporting its tiers: {e}"))?;
        return Ok(None);
    }
    let tiers = WorkerTiers::new(
        &start.kv,
        prepared.provider.opened.memory_kind,
        super::copy_device(prepared),
        BlockAddresses::of(pool),
        start.l2,
        Arc::new(turbine_core::clock::SystemClock::new()),
    )
    .map_err(|e| format!("rank {rank}: {e}"))?;
    let (l1, l2) = (tiers.l1_enabled(), tiers.l2_enabled());
    tracing::info!(
        event = "kv_rank_tiers_ready",
        rank,
        l1,
        l2,
        "this static rank keeps its own KV tier shards, driven by the leader"
    );
    link.tiers_ready(l1, l2)
        .map_err(|e| format!("rank {rank}: reporting its tiers: {e}"))?;
    Ok(Some(Box::new(tiers)))
}

/// A `static` worker rank's tiers (module comment): its pool's copy backend (its own copy
/// stream or synchronous copies, its own staging buffers and I/O threads), its own L1 and L2.
/// The pool whose block addresses it holds must outlive it; the link's tier thread finishes
/// every copy before the pool is dropped.
pub(crate) struct WorkerTiers {
    backend: CopyStreamBackend,
    l1: Option<Arc<ShardedL1Tier>>,
    l2: Option<Arc<dyn KvTier>>,
    /// Copies in flight with their start.
    inflight: Vec<(TransferTicket, Instant)>,
    done: Vec<TierAck>,
    shard_bytes: u64,
}

impl WorkerTiers {
    /// L1 needs a copy stream on a dedicated-memory device (as on the leader); its first
    /// pinned slab is allocated now, and an allocation failure is an error when `kv.cpu` is
    /// enabled.
    pub(crate) fn new(
        kv: &KvConfig,
        memory_kind: MemoryKind,
        device: CopyDevice,
        addresses: BlockAddresses,
        l2: Option<Arc<L2NvmeTier>>,
        clock: Arc<dyn Clock>,
    ) -> Result<WorkerTiers, String> {
        let shard_bytes = addresses.block_bytes();
        let l1 = match &device {
            CopyDevice::Stream { pinned, .. }
                if kv.cpu.enabled && memory_kind != MemoryKind::Unified =>
            {
                let tier = Arc::new(L1PinnedTier::new(
                    L1Config {
                        enabled: true,
                        max_bytes: kv.cpu.max_bytes.0,
                        slab_bytes: L1_SLAB_BYTES,
                        block_bytes: shard_bytes,
                        memory_kind,
                    },
                    Arc::clone(pinned),
                    clock,
                ));
                Some(Arc::new(ShardedL1Tier::new(vec![tier], shard_bytes)))
            }
            _ => {
                if kv.cpu.enabled {
                    tracing::warn!(
                        event = "kv_l1_disabled_no_pinned_memory",
                        tier = "l1",
                        "no copy stream or pinned-memory API on this rank; L1 is off for the group"
                    );
                }
                None
            }
        };
        let l1 = l1.filter(|t| t.enabled());
        if let Some(t) = &l1 {
            // The first slab now, as the leader's calibration takes it.
            t.reserve(PREFLIGHT_KEY).map_err(|e| {
                format!("kv.cpu.enabled: the first pinned L1 slab cannot be allocated: {e}")
            })?;
            t.abort_reservation(&PREFLIGHT_KEY);
        }
        let l2 = l2.filter(|t| t.enabled()).map(|t| t as Arc<dyn KvTier>);
        let io_capacity = (kv.transfer.max_inflight_bytes.0 / shard_bytes.max(1)) as usize + 1;
        let backend = CopyStreamBackend::new(
            vec![KvShard { device, addresses }],
            l1.clone(),
            l2.clone(),
            IoPoolBackend::new(kv.nvme.io_threads.max(1) as usize, io_capacity),
            shard_bytes as usize,
        );
        Ok(WorkerTiers {
            backend,
            l1,
            l2,
            inflight: Vec::new(),
            done: Vec::new(),
            shard_bytes,
        })
    }

    pub(crate) fn l1_enabled(&self) -> bool {
        self.l1.is_some()
    }

    pub(crate) fn l2_enabled(&self) -> bool {
        self.l2.is_some()
    }
}

impl TierWorker for WorkerTiers {
    fn submit(&mut self, copies: Vec<TierCopy>) {
        for c in copies {
            match c {
                TierCopy::Copy {
                    id,
                    path,
                    key,
                    block,
                } => {
                    let t = TransferTicket {
                        id,
                        req: TransferRequest {
                            path: kv_path(path),
                            key: KvKey(key),
                            bytes: self.shard_bytes,
                            owner: None,
                            purpose: TransferPurpose::Demote,
                            src_slot: u64::from(block.0),
                            dst_slot: u64::from(block.0),
                            codec: TransferCodec::l0(self.shard_bytes),
                        },
                    };
                    match self.backend.start(&t) {
                        Ok(()) => self.inflight.push((t, Instant::now())),
                        Err(e) => self.done.push(TierAck {
                            id,
                            error: Some(e.to_string()),
                            took_ns: 0,
                        }),
                    }
                }
                TierCopy::Evict { tier, key } => {
                    let tier = match tier {
                        TierLevel::L1 => self.l1.clone().map(|t| t as Arc<dyn KvTier>),
                        TierLevel::L2 => self.l2.clone(),
                    };
                    if let Some(t) = tier {
                        let _ = t.evict(&KvKey(key));
                    }
                }
            }
        }
    }

    fn poll(&mut self) -> Vec<TierAck> {
        let mut running = Vec::with_capacity(self.inflight.len());
        for (t, started) in std::mem::take(&mut self.inflight) {
            match self.backend.poll(&t) {
                Ok(None) => running.push((t, started)),
                Ok(Some(_)) => self.done.push(TierAck {
                    id: t.id,
                    error: None,
                    took_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                }),
                Err(e) => self.done.push(TierAck {
                    id: t.id,
                    error: Some(e.to_string()),
                    took_ns: 0,
                }),
            }
        }
        self.inflight = running;
        std::mem::take(&mut self.done)
    }

    fn busy(&self) -> bool {
        !self.inflight.is_empty()
    }
}

/// The leader's `static` tier driver, when this group runs in `static` mode: waits for every
/// worker's `TierReady` (up to `init_timeout`) and bounds each copy's acknowledgements by
/// `op_timeout`. `None` in `local` mode.
pub(crate) fn leader_driver(
    runtime: &RankRuntime,
    init_timeout: Duration,
    op_timeout: Duration,
) -> Result<Option<TierDriver>, StartupError> {
    let Some(link) = runtime.tier_link() else {
        return Ok(None);
    };
    let ready = runtime
        .tier_ready(init_timeout)
        .map_err(|e| StartupError::new(format!("static ranks' KV tiers: {e}")))?;
    Ok(Some(TierDriver::new(link, &ready, op_timeout)))
}

/// One copy in flight on the leader: its own copy's outcome once done, every worker's, and why
/// it fails for the group whatever they say.
struct Pending {
    queued: Instant,
    local: Option<Result<TierSlot, TierError>>,
    /// When the leader saw its own copy complete, from the start.
    local_took: Option<Duration>,
    acks: HashMap<u32, Option<String>>,
    /// Each worker's own copy time (`TierAck::took_ns`).
    worker_took: Duration,
    /// The copy never reached the workers (their link closed): nothing to wait for.
    unsent: bool,
    /// The group failed while the copy was in flight (a `StepFailed`, a lost or silent worker).
    broken: Option<String>,
}

/// The leader's end of the `static` tiers (module comment): the outgoing operations, the
/// copies in flight and the workers' acknowledgements.
pub(crate) struct TierDriver {
    link: TierLink,
    timeout: Duration,
    workers_l1: bool,
    workers_l2: bool,
    outbox: Arc<Mutex<Vec<TierCopy>>>,
    pending: HashMap<u64, Pending>,
    /// A link to the workers closed: no operation reaches them any more.
    closed: Option<RankError>,
    local_l1: Option<Arc<ShardedL1Tier>>,
    local_l2: Option<Arc<dyn KvTier>>,
    /// Copy times of the copies completed by the last `poll`s, taken by `took`.
    took: HashMap<u64, Duration>,
}

impl TierDriver {
    /// Over `link`, with every worker's `(rank, l1, l2)` from `TierReady`.
    pub(crate) fn new(link: TierLink, ready: &[(u32, bool, bool)], timeout: Duration) -> Self {
        let has = |tier: &str, pick: fn(&(u32, bool, bool)) -> bool| {
            let missing: Vec<u32> = ready.iter().filter(|r| !pick(r)).map(|r| r.0).collect();
            if !missing.is_empty() {
                tracing::warn!(
                    event = "kv_tier_unavailable_on_rank",
                    tier,
                    ranks = ?missing,
                    "a static worker rank lacks this KV tier; it is off for the whole group"
                );
            }
            missing.is_empty() && ready.len() == link.ranks().len()
        };
        TierDriver {
            workers_l1: has("l1", |r| r.1),
            workers_l2: has("l2", |r| r.2),
            link,
            timeout,
            outbox: Arc::new(Mutex::new(Vec::new())),
            pending: HashMap::new(),
            closed: None,
            local_l1: None,
            local_l2: None,
            took: HashMap::new(),
        }
    }

    /// Ranks of the group, the leader included.
    pub(crate) fn world(&self) -> u32 {
        self.link.ranks().len() as u32 + 1
    }

    /// Every worker has `tier` (L1 or L2).
    pub(crate) fn workers_have(&self, tier: TierId) -> bool {
        match tier {
            TierId::L1 => self.workers_l1,
            TierId::L2 => self.workers_l2,
            _ => true,
        }
    }

    /// The leader's own tiers, which a failed copy's undo drops the key from.
    pub(crate) fn bind(&mut self, l1: Option<Arc<ShardedL1Tier>>, l2: Option<Arc<dyn KvTier>>) {
        self.local_l1 = l1;
        self.local_l2 = l2;
    }

    /// `tier` as the hierarchy sees it: every eviction also goes to the workers.
    pub(crate) fn mirror(&self, tier: Arc<dyn KvTier>) -> Arc<dyn KvTier> {
        let level = match tier.id() {
            TierId::L1 => TierLevel::L1,
            _ => TierLevel::L2,
        };
        Arc::new(MirroredTier {
            local: tier,
            level,
            outbox: Arc::clone(&self.outbox),
        })
    }

    /// The transfer backend of one pump: the leader's `local` copies plus the workers'.
    pub(crate) fn backend<'a>(&'a mut self, local: &'a mut CopyStreamBackend) -> StaticBackend<'a> {
        StaticBackend {
            local,
            driver: self,
        }
    }

    /// Sends the queued operations to every worker (one bounded batch per frame). Copies that
    /// cannot be sent (a closed link) fail without waiting for the workers.
    pub(crate) fn flush(&mut self) {
        let ops = std::mem::take(&mut *lock(&self.outbox));
        if ops.is_empty() {
            return;
        }
        if self.closed.is_none()
            && let Err(e) = self.link.send(&ops)
        {
            tracing::error!(event = "kv_tier_link_failed", error = %e, "static ranks' tier copies cannot be sent");
            self.closed = Some(e);
        }
        if self.closed.is_some() {
            for op in &ops {
                if let TierCopy::Copy { id, .. } = op
                    && let Some(p) = self.pending.get_mut(id)
                {
                    p.unsent = true;
                }
            }
        }
    }

    /// Takes the acknowledgements that arrived; while the group is failed, every copy in flight
    /// fails with it.
    fn collect(&mut self) {
        for (rank, acks) in self.link.acks() {
            for a in acks {
                if let Some(p) = self.pending.get_mut(&a.id) {
                    p.worker_took = p.worker_took.max(Duration::from_nanos(a.took_ns));
                    p.acks.insert(rank, a.error);
                }
            }
        }
        if let Some(e) = self.link.failure().or_else(|| self.closed.clone()) {
            for p in self.pending.values_mut() {
                p.broken.get_or_insert_with(|| format!("static ranks: {e}"));
            }
        }
    }

    /// The first worker rank that has not answered `p`.
    fn missing(&self, p: &Pending) -> Option<u32> {
        self.link
            .ranks()
            .iter()
            .copied()
            .find(|r| !p.acks.contains_key(r))
    }

    /// No answer will come: a link is closed (the worker is gone).
    fn gone(&self) -> bool {
        matches!(
            self.link.failure().or_else(|| self.closed.clone()),
            Some(RankError::Closed { .. })
        )
    }

    fn queue(&self, op: TierCopy) {
        lock(&self.outbox).push(op);
    }

    /// Drops `key` from the leader's `tier` (L1/L2) after a failed copy.
    fn drop_local(&self, tier: TierId, key: &KvKey) {
        let t = match tier {
            TierId::L1 => self.local_l1.clone().map(|t| t as Arc<dyn KvTier>),
            TierId::L2 => self.local_l2.clone(),
            _ => None,
        };
        if let Some(t) = t {
            let _ = t.evict(key);
        }
    }

    /// The outcome of a copy every rank finished (or whose missing answers will not come).
    fn finish(&mut self, t: &TransferTicket, p: Pending) -> Result<Option<TierSlot>, TierError> {
        let missing = self.missing(&p);
        let local = p.local.unwrap_or(Err(TierError::Missing));
        let worker = p
            .acks
            .iter()
            .filter_map(|(rank, e)| e.as_ref().map(|e| format!("rank {rank}: {e}")))
            .min();
        let worker_failed = worker.is_some();
        let group = p.broken.or_else(|| {
            missing.map(|rank| {
                if p.unsent {
                    format!("rank {rank}: the copy could not be sent")
                } else {
                    format!("rank {rank}: no answer")
                }
            })
        });
        let error = match (&local, worker.or(group)) {
            (Ok(slot), None) => {
                // The copy took as long as its slowest rank's part: the leader's own as its
                // poll saw it (as in `local` mode) or a worker's as the worker timed it — not
                // the extra engine turn its acknowledgement took to arrive.
                let own = p.local_took.unwrap_or_default();
                self.took.insert(t.id, own.max(p.worker_took));
                return Ok(Some(*slot));
            }
            (_, Some(e)) => TierError::Io(e),
            (Err(e), None) => e.clone(),
        };
        // Every rank drops what it made of the copy, so the tiers stay symmetric.
        let (from, to) = (t.req.path.from(), t.req.path.to());
        if to != TierId::L0 {
            if local.is_ok() {
                self.drop_local(to, &t.req.key);
            }
            self.queue(TierCopy::Evict {
                tier: if to == TierId::L1 {
                    TierLevel::L1
                } else {
                    TierLevel::L2
                },
                key: t.req.key.0,
            });
        }
        // A worker's copy-stream error counts toward L1's degraded window, as a shard's does in
        // `local` mode.
        if worker_failed
            && local.is_ok()
            && (from == TierId::L1 || to == TierId::L1)
            && (from == TierId::L0 || to == TierId::L0)
            && let Some(l1) = &self.local_l1
        {
            l1.record_copy_error(0);
        }
        tracing::debug!(event = "kv_static_copy_failed", path = t.req.path.as_str(), key = %t.req.key, error = %error);
        Err(error)
    }
}

/// The leader's tier as the hierarchy holds it in `static` mode: the leader's own tier, whose
/// evictions are also queued for every worker.
struct MirroredTier {
    local: Arc<dyn KvTier>,
    level: TierLevel,
    outbox: Arc<Mutex<Vec<TierCopy>>>,
}

impl KvTier for MirroredTier {
    fn id(&self) -> TierId {
        self.local.id()
    }
    fn enabled(&self) -> bool {
        self.local.enabled()
    }
    fn capacity_bytes(&self) -> u64 {
        self.local.capacity_bytes()
    }
    fn used_bytes(&self) -> u64 {
        self.local.used_bytes()
    }
    fn pressure(&self) -> PressureState {
        self.local.pressure()
    }
    fn est_latency(&self) -> Duration {
        self.local.est_latency()
    }
    fn est_bandwidth(&self) -> Option<f64> {
        self.local.est_bandwidth()
    }
    fn contains(&self, key: &KvKey) -> bool {
        self.local.contains(key)
    }
    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        self.local.put(key, src)
    }
    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError> {
        self.local.get(key, dst)
    }
    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        lock(&self.outbox).push(TierCopy::Evict {
            tier: self.level,
            key: key.0,
        });
        self.local.evict(key)
    }
    fn degraded(&self) -> bool {
        self.local.degraded()
    }
}

/// The transfer backend of one pump in `static` mode (module comment): starts each copy on the
/// leader's own tiers and queues it for the workers; a copy completes once every rank's did.
pub(crate) struct StaticBackend<'a> {
    local: &'a mut CopyStreamBackend,
    driver: &'a mut TierDriver,
}

impl TransferBackend for StaticBackend<'_> {
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
        // A failed group copies nothing until it is re-created (P5 Task 28).
        if let Some(e) = self
            .driver
            .link
            .failure()
            .or_else(|| self.driver.closed.clone())
        {
            return Err(TierError::Io(format!("static ranks: {e}")));
        }
        // A copy the leader cannot start is never sent: it fails for every rank at once.
        self.local.start(t)?;
        self.driver.queue(TierCopy::Copy {
            id: t.id,
            path: wire_path(t.req.path),
            key: t.req.key.0,
            block: turbine_core::types::BlockId(l0_block(&t.req) as u32),
        });
        self.driver.pending.insert(
            t.id,
            Pending {
                queued: Instant::now(),
                local: None,
                local_took: None,
                acks: HashMap::new(),
                worker_took: Duration::ZERO,
                unsent: false,
                broken: None,
            },
        );
        Ok(())
    }

    /// Completes a copy once the leader's own copy is done and every worker answered — also
    /// when the group failed meanwhile (the workers' tier threads still answer, and a copy into
    /// or out of a block must be over on every rank before the block is reused), unless a worker
    /// is gone or the copy never reached it. A worker silent beyond the timeout fails the group.
    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
        self.driver.collect();
        let timeout = self.driver.timeout;
        let gone = self.driver.gone();
        let Some(p) = self.driver.pending.get(&t.id) else {
            return Err(TierError::Missing);
        };
        let missing = self.driver.missing(p);
        let timed_out = missing.is_some() && !p.unsent && !gone && p.queued.elapsed() > timeout;
        let wait = missing.is_some() && !p.unsent && !gone && !timed_out;
        if timed_out && let Some(rank) = missing {
            let detail = format!(
                "no KV tier copy acknowledgement within {timeout:?} \
                 (parallel.collective.op_timeout)"
            );
            tracing::error!(
                event = "kv_tier_ack_timeout",
                rank,
                timeout_s = timeout.as_secs_f64(),
                "a static worker rank did not acknowledge a KV tier copy; failing the group"
            );
            self.driver.link.fail(RankError::Executor {
                rank,
                detail: detail.clone(),
            });
            self.driver.collect();
        }
        let p = self.driver.pending.get_mut(&t.id).expect("present above");
        if p.local.is_none() {
            match self.local.poll(t) {
                Ok(None) => {}
                Ok(Some(slot)) => {
                    p.local = Some(Ok(slot));
                    p.local_took = Some(p.queued.elapsed());
                }
                Err(e) => p.local = Some(Err(e)),
            }
        }
        // The leader's own copy must be done before its buffers or blocks are given back.
        if wait || p.local.is_none() {
            return Ok(None);
        }
        let p = self.driver.pending.remove(&t.id).expect("present above");
        self.driver.finish(t, p)
    }

    fn took(&mut self, t: &TransferTicket) -> Option<Duration> {
        self.driver.took.remove(&t.id)
    }
}

#[cfg(test)]
mod tests {
    use std::net::{SocketAddr, TcpListener};

    use turbine_core::clock::SystemClock;
    use turbine_core::types::{BlockId, DType, DeviceId, ModelFingerprint, Vendor};
    use turbine_distributed::rank::{
        ExecError, HelloExpect, PROTOCOL_VERSION, RankMessage, StepExecutor, StepOutput, StepPlan,
    };
    use turbine_kv::BlockPoolConfig;
    use turbine_kv::tier::MemTier;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 16,
            dtype: DType::BF16,
            block_tokens: 16,
        }
    }

    struct NoSteps;
    impl StepExecutor for NoSteps {
        fn execute(&mut self, _plan: &StepPlan) -> Result<StepOutput, ExecError> {
            Ok(StepOutput {
                logits: None,
                rows: 0,
                vocab: 0,
            })
        }
    }

    /// Fails every step (not a sticky error): the worker reports `StepFailed` and waits.
    struct FailSteps;
    impl StepExecutor for FailSteps {
        fn execute(&mut self, plan: &StepPlan) -> Result<StepOutput, ExecError> {
            Err(ExecError::Executor(format!("step {} failed", plan.step)))
        }
    }

    /// Answers every copy with success, but only once `release` is set.
    struct Gated {
        release: Arc<std::sync::atomic::AtomicBool>,
        held: Vec<u64>,
    }
    impl TierWorker for Gated {
        fn submit(&mut self, copies: Vec<TierCopy>) {
            self.held.extend(copies.iter().filter_map(|c| match c {
                TierCopy::Copy { id, .. } => Some(*id),
                TierCopy::Evict { .. } => None,
            }));
        }
        fn poll(&mut self) -> Vec<TierAck> {
            if !self.release.load(std::sync::atomic::Ordering::Acquire) {
                return Vec::new();
            }
            self.held
                .drain(..)
                .map(|id| TierAck {
                    id,
                    error: None,
                    took_ns: 0,
                })
                .collect()
        }
        fn busy(&self) -> bool {
            !self.held.is_empty()
        }
    }

    /// Takes every copy and never answers one.
    struct Silent;
    impl TierWorker for Silent {
        fn submit(&mut self, _copies: Vec<TierCopy>) {}
        fn poll(&mut self) -> Vec<TierAck> {
            Vec::new()
        }
        fn busy(&self) -> bool {
            false
        }
    }

    /// A `static` leader of one worker running `tiers` on its own thread (or leaving the group
    /// at once, `None`), and the leader's own copy backend over a host pool with an in-memory
    /// L2; the worker thread's `run_with_tiers` result comes back on its handle.
    struct Group {
        runtime: RankRuntime,
        driver: TierDriver,
        local: CopyStreamBackend,
        l2: Arc<dyn KvTier>,
        worker: std::thread::JoinHandle<Result<(), RankError>>,
        _pool: BlockPool,
    }

    fn group(
        tiers: Option<Box<dyn TierWorker>>,
        mut exec: Box<dyn StepExecutor>,
        timeout: Duration,
    ) -> Group {
        let transport = turbine_distributed::transport::select("tcp").expect("tcp");
        let addr: SocketAddr = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let expect = HelloExpect {
            model_fingerprint: ModelFingerprint([2; 32]),
            config_fingerprint: [3; 32],
            device_vendor: Vendor::Amd,
            device_arch: "test".into(),
        };
        let hello = RankMessage::Hello {
            protocol: PROTOCOL_VERSION,
            rank: 1,
            world_size: 2,
            model_fingerprint: expect.model_fingerprint,
            config_fingerprint: expect.config_fingerprint,
            device_vendor: expect.device_vendor,
            device_arch: expect.device_arch.clone(),
        };
        let worker = std::thread::spawn(move || {
            let mut link =
                RankRuntime::static_worker(transport, addr, hello, Duration::from_secs(10))?;
            link.tiers_ready(false, true)?;
            match tiers {
                Some(t) => link.run_with_tiers(exec.as_mut(), Some(t)),
                None => Ok(()),
            }
        });
        let runtime = RankRuntime::static_leader(
            transport,
            addr,
            expect,
            2,
            Duration::from_secs(10),
            [0; 128],
            2,
        )
        .expect("joined");
        let driver = leader_driver(&runtime, Duration::from_secs(10), timeout)
            .expect("tier ready")
            .expect("static");
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 24);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: layout(),
                num_blocks: 4,
            },
            Arc::clone(&mem),
        )
        .unwrap();
        let l2: Arc<dyn KvTier> = Arc::new(MemTier::new(TierId::L2, 1 << 20, clock));
        let local = CopyStreamBackend::new(
            vec![KvShard {
                device: CopyDevice::Sync { mem },
                addresses: BlockAddresses::of(&pool),
            }],
            None,
            Some(Arc::clone(&l2)),
            IoPoolBackend::new(1, 4),
            layout().block_bytes() as usize,
        );
        let mut driver = driver;
        driver.bind(None, Some(Arc::clone(&l2)));
        Group {
            runtime,
            driver,
            local,
            l2,
            worker,
            _pool: pool,
        }
    }

    /// Starts one L0 -> L2 copy of block 1 and polls it to its end; returns the outcome and
    /// how long it took.
    fn copy_down(g: &mut Group, key: KvKey) -> (Result<TierSlot, TierError>, Duration) {
        let t = TransferTicket {
            id: 1,
            req: TransferRequest {
                path: TransferPath::L0ToL2,
                key,
                bytes: layout().block_bytes(),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot: 1,
                dst_slot: 0,
                codec: TransferCodec::l0(layout().block_bytes()),
            },
        };
        let started = Instant::now();
        g.driver
            .backend(&mut g.local)
            .start(&t)
            .expect("the leader's own copy starts");
        g.driver.flush();
        loop {
            match g.driver.backend(&mut g.local).poll(&t) {
                Ok(None) => {}
                Ok(Some(slot)) => return (Ok(slot), started.elapsed()),
                Err(e) => return (Err(e), started.elapsed()),
            }
            assert!(started.elapsed() < Duration::from_secs(20), "never ended");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// P5 Task 30: the leader's wait for a worker's tier acknowledgement is bounded. A worker
    /// that takes a copy and never answers fails it after the timeout, for every rank (the
    /// leader drops its own copy again), and fails the group, so the next step plan is refused
    /// like any failed rank's and no new copy starts. Breaks if the leader waits forever,
    /// completes a copy the worker never confirmed, or leaves its half of it behind.
    #[test]
    fn silent_worker_fails_the_copy_and_the_group() {
        let timeout = Duration::from_millis(300);
        let mut g = group(Some(Box::new(Silent)), Box::new(NoSteps), timeout);
        let key = KvKey([9; 16]);
        let (result, took) = copy_down(&mut g, key);
        let err = result.expect_err("no acknowledgement came");
        assert!(err.to_string().contains("rank 1"), "{err}");
        assert!(
            took >= timeout && took < Duration::from_secs(10),
            "{took:?}"
        );
        assert!(!g.l2.contains(&key), "the leader dropped its own half");
        let plan = StepPlan {
            step: 1,
            sequences: Vec::new(),
            copies: vec![(BlockId(0), BlockId(1))],
            ledger: Vec::new(),
        };
        assert!(g.runtime.step(plan).is_err(), "the group failed");
        let t = TransferTicket {
            id: 2,
            req: TransferRequest {
                path: TransferPath::L0ToL2,
                key: KvKey([8; 16]),
                bytes: layout().block_bytes(),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot: 2,
                dst_slot: 0,
                codec: TransferCodec::l0(layout().block_bytes()),
            },
        };
        assert!(
            g.driver.backend(&mut g.local).start(&t).is_err(),
            "no copy starts on a failed group"
        );
        g.runtime.shutdown("test done");
        assert!(g.worker.join().unwrap().is_ok());
    }

    /// A worker that is gone (its link closed) fails the group, and no copy starts: the
    /// hierarchy recomputes at once instead of waiting out the timeout. Breaks if a lost
    /// worker's copies are started or waited for.
    #[test]
    fn lost_worker_refuses_copies() {
        let mut g = group(None, Box::new(NoSteps), Duration::from_secs(60));
        assert!(g.worker.join().unwrap().is_ok(), "the worker left");
        let started = Instant::now();
        while g.driver.link.failure().is_none() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the loss was never seen"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let t = TransferTicket {
            id: 3,
            req: TransferRequest {
                path: TransferPath::L0ToL2,
                key: KvKey([7; 16]),
                bytes: layout().block_bytes(),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot: 1,
                dst_slot: 0,
                codec: TransferCodec::l0(layout().block_bytes()),
            },
        };
        let err = g
            .driver
            .backend(&mut g.local)
            .start(&t)
            .expect_err("no copy starts once the worker is gone");
        assert!(err.to_string().contains("rank 1"), "{err}");
        g.runtime.shutdown("test done");
    }

    /// P5 Task 28 x Task 30: a step that fails on the worker while one of its copies is in flight
    /// (`StepFailed`: the rank stays up and its tier thread keeps answering) fails that copy
    /// for the group — once the worker's part of it is over, not before — and the group is
    /// failed. Breaks if the copy completes as if nothing happened, or is given up while the
    /// worker may still be writing it.
    #[test]
    fn step_failed_during_a_copy_fails_it() {
        let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let gated = Gated {
            release: Arc::clone(&release),
            held: Vec::new(),
        };
        let mut g = group(
            Some(Box::new(gated)),
            Box::new(FailSteps),
            Duration::from_secs(60),
        );
        let t = TransferTicket {
            id: 4,
            req: TransferRequest {
                path: TransferPath::L0ToL2,
                key: KvKey([6; 16]),
                bytes: layout().block_bytes(),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot: 1,
                dst_slot: 0,
                codec: TransferCodec::l0(layout().block_bytes()),
            },
        };
        g.driver.backend(&mut g.local).start(&t).expect("starts");
        g.driver.flush();
        let plan = StepPlan {
            step: 1,
            sequences: Vec::new(),
            copies: vec![(BlockId(0), BlockId(1))],
            ledger: Vec::new(),
        };
        g.runtime.step(plan).expect("the plan is handed out");
        let started = Instant::now();
        while g.driver.link.failure().is_none() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "StepFailed never seen"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        for _ in 0..20 {
            assert_eq!(
                g.driver.backend(&mut g.local).poll(&t),
                Ok(None),
                "the worker's part of the copy is not over yet"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        release.store(true, std::sync::atomic::Ordering::Release);
        let err = loop {
            match g.driver.backend(&mut g.local).poll(&t) {
                Ok(None) => std::thread::sleep(Duration::from_millis(2)),
                Ok(Some(_)) => panic!("a copy of a failed group completed"),
                Err(e) => break e,
            }
            assert!(started.elapsed() < Duration::from_secs(10), "never ended");
        };
        assert!(err.to_string().contains("static ranks"), "{err}");
        assert!(
            !g.l2.contains(&KvKey([6; 16])),
            "the leader dropped its half"
        );
        g.runtime.shutdown("test done");
        assert!(g.worker.join().unwrap().is_ok());
    }

    /// Answers every copy only `delay` after it was submitted, reporting `took` as its own copy
    /// time: a worker whose acknowledgement arrives late for reasons other than its copy.
    struct Late {
        delay: Duration,
        took: Duration,
        held: Vec<(u64, Instant)>,
    }
    impl TierWorker for Late {
        fn submit(&mut self, copies: Vec<TierCopy>) {
            let now = Instant::now();
            self.held.extend(copies.iter().filter_map(|c| match c {
                TierCopy::Copy { id, .. } => Some((*id, now)),
                TierCopy::Evict { .. } => None,
            }));
        }
        fn poll(&mut self) -> Vec<TierAck> {
            let (due, held): (Vec<_>, Vec<_>) = std::mem::take(&mut self.held)
                .into_iter()
                .partition(|(_, at)| at.elapsed() >= self.delay);
            self.held = held;
            due.into_iter()
                .map(|(id, _)| TierAck {
                    id,
                    error: None,
                    took_ns: self.took.as_nanos() as u64,
                })
                .collect()
        }
        fn busy(&self) -> bool {
            !self.held.is_empty()
        }
    }

    /// P5 Task 30: the time a static copy is recorded as taking (what the transfer estimates
    /// learn from) is its slowest rank's own copy time — the leader's copy as its poll saw it,
    /// or a worker's as the worker timed it — not the time the acknowledgement took to reach
    /// the leader. Breaks if a late acknowledgement is read as a slow copy (static mode would
    /// then plan recomputes local mode plans as retrievals).
    #[test]
    fn copy_time_is_the_ranks_own_not_the_ack_delay() {
        let delay = Duration::from_millis(300);
        let reported = Duration::from_millis(1);
        let late = Late {
            delay,
            took: reported,
            held: Vec::new(),
        };
        let mut g = group(
            Some(Box::new(late)),
            Box::new(NoSteps),
            Duration::from_secs(10),
        );
        let (result, waited) = copy_down(&mut g, KvKey([3; 16]));
        result.expect("the copy completes");
        assert!(waited >= delay, "the ack came after {waited:?}");
        let t = TransferTicket {
            id: 1,
            req: TransferRequest {
                path: TransferPath::L0ToL2,
                key: KvKey([3; 16]),
                bytes: layout().block_bytes(),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot: 1,
                dst_slot: 0,
                codec: TransferCodec::l0(layout().block_bytes()),
            },
        };
        let took = g.driver.backend(&mut g.local).took(&t).expect("measured");
        assert!(took >= reported, "at least the worker's own copy: {took:?}");
        assert!(
            took < delay / 3,
            "not the acknowledgement's delay: {took:?}"
        );
        assert!(
            g.driver.backend(&mut g.local).took(&t).is_none(),
            "taken once"
        );
        g.runtime.shutdown("test done");
        assert!(g.worker.join().unwrap().is_ok());
    }
}
