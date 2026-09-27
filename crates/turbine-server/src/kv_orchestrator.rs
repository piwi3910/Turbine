//! The KV orchestrator (P4 §Data: "the directory, tiers and session table are owned by the KV
//! orchestrator task in `turbine-server`"; contract §16.4). [`KvOrchestrator`] owns the
//! [`KvHierarchy`] and the [`CopyStreamBackend`] that moves block bytes, and runs on the engine
//! thread next to the scheduler and the L0 [`BlockPool`], driven synchronously at iteration
//! boundaries in the order of the scheduler's `KvSimDriver` (P4 Task 11): transfer
//! completions → attach retries → reclaim-order refresh → `plan` → [`KvOrchestrator::after_plan`]
//! → execution → [`KvOrchestrator::commit`] → `complete` → [`KvOrchestrator::request_done`] →
//! [`KvOrchestrator::end_turn`] (reclaim requests, session TTLs). No lock is taken on that path.
//! The HTTP side reaches it only through [`KvHandle`]'s bounded command channel
//! (`POST /turbine/v1/kv/prefetch`); the KV document is published with the engine's documents.
//!
//! Tiers (P4 S-5, S-11): L1 pinned host memory exists only with a copy stream (kernel ABI v2.3 +
//! v2.5, `ShimContext::has_copy_engine`) on a discrete-memory device; without a pinned-memory API
//! L1 is disabled with a WARN. L2 is
//! opened before the listener binds ([`open_l2`]: path created, old slab files deleted,
//! writability checked, exit 1 naming the path) and is used with or without a GPU: without a
//! copy stream (the cpu backend) L0 ↔ L2 copies go through synchronous `DeviceMemory` copies.
//!
//! Calibration (P4 S-6): at startup one 64 MiB copy (in whole blocks, through the production
//! copy paths) per enabled path seeds the transfer estimates and the L2 slow-tier baseline; it
//! is logged as `kv_calibration`. A failed calibration keeps `TransferPath::fallback` with a
//! WARN.
//!
//! Phase 3 wiring: the L0 pressure state the planner and prefetch see is the pressure
//! controller's (`before_plan` takes it from the engine's snapshot), the controller's reclaim
//! step drives the hierarchy through [`KvOrchestrator::reclaimer`] (registered as its
//! `KvReclaimer`), and the L2 queue depth and p99 latency reach its `storage_queue_depth` and
//! `storage_latency` signals through [`L2StorageProbe`] on the telemetry sampler's fast tick.

use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use tokio::sync::{mpsc, oneshot};
use turbine_core::clock::Clock;
use turbine_core::config::KvConfig;
use turbine_core::request::SessionHints;
use turbine_core::telemetry::{StorageProbe, StorageSample};
use turbine_core::types::{
    BlockId, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState, Priority, RequestId,
};
use turbine_kv::document::HitWindow;
use turbine_kv::hierarchy::{
    AttachOutcome, AttachRequest, HierarchyConfig, KvHierarchy, KvReclaimHandle, PrefetchAccepted,
    PrefetchError, PrefetchTarget, PrefixAttach,
};
use turbine_kv::identity::{KvFormat, KvKey, namespace_key};
use turbine_kv::metrics::EvictReason;
use turbine_kv::planner::PathCost;
use turbine_kv::tier::{
    KvTier, L1Config, L1PinnedTier, L2Config, L2NvmeTier, TierBlockMut, TierBlockRef, TierError,
    TierId, TierSlot,
};
use turbine_kv::transfer::{TransferBackend, TransferPath, TransferTicket};
use turbine_kv::{BlockPool, KvDocument, KvMetrics};
use turbine_tensor::{
    CopyEngine, CopyTarget, CopyTicket, DeviceMemory, DevicePtr, PinnedBuffer, PinnedMemory,
};

use crate::model::StartupError;

/// Bytes per L1 pinned slab (P4 S-5: grown lazily in 1 GiB slabs).
pub const L1_SLAB_BYTES: u64 = 1 << 30;
/// Bytes each calibration copy moves per path (P4 S-6).
pub const CALIBRATION_BYTES: u64 = 64 << 20;
/// Physical L0 fill (referenced plus cached blocks over the pool) above which cached blocks are
/// demoted by capacity, down to this fill: Phase 3's `kv_utilization` YELLOW threshold
/// (provisional decision "Phase 4: L0 capacity demotion").
pub const CAPACITY_DEMOTE_AT: f64 = 0.70;
/// Below this share of free L0 blocks the pool's reclaim order is refreshed before planning.
const REFRESH_FREE_SHARE: u32 = 10;
/// EWMA weight of one prefill-rate sample.
const PREFILL_ALPHA: f64 = 0.2;
/// Prefill iterations smaller than this do not update the prefill rate (launch overhead).
const PREFILL_MIN_TOKENS: u32 = 64;

/// The device side of block copies.
#[derive(Clone)]
pub enum CopyDevice {
    /// Kernel ABI v3: the device's copy stream and page-locked host memory.
    Stream {
        engine: Arc<dyn CopyEngine>,
        pinned: Arc<dyn PinnedMemory>,
    },
    /// No copy stream (the cpu backend): synchronous `DeviceMemory` copies on the engine thread.
    Sync { mem: Arc<dyn DeviceMemory> },
}

/// Opens L2 before the listener binds (P4 §Configuration `kv.nvme.path`: created if absent,
/// must be writable, else exit 1; §Data: old `turbine-kv-*.slab` files deleted). `None` when
/// `kv.nvme.enabled` is false.
pub fn open_l2(
    cfg: &KvConfig,
    format: &KvFormat,
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
            block_bytes: format.layout.block_bytes(),
            namespace: namespace_key(identity, format, ""),
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
        "L2 NVMe tier ready"
    );
    Ok(Some(Arc::new(l2)))
}

/// The L2 tier's readings for the pressure controller's storage signals (P4 S-11, contract
/// §8.3): queue depth over `kv.nvme.max_queue_depth` and p99 latency over the calibration p99.
/// Read on the telemetry sampler's fast tick; the tier's own lock guards the figures.
pub struct L2StorageProbe {
    pub l2: Arc<L2NvmeTier>,
    pub max_queue_depth: u32,
}

impl StorageProbe for L2StorageProbe {
    fn storage(&self) -> Option<StorageSample> {
        Some(StorageSample {
            queue_depth: self.l2.queue_depth(),
            max_queue_depth: self.max_queue_depth,
            p99_latency_s: self.l2.p99_latency(),
            calibration_latency_s: self.l2.calibration_p99().unwrap_or(0.0),
        })
    }
}

/// The KV format of `layout` (KV is BF16 only before Phase 8a).
pub fn kv_format(layout: KvLayout) -> KvFormat {
    KvFormat {
        dtype: KvDtype::Bf16,
        layout,
    }
}

/// What a prefetch names, owned so it can cross the command channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefetchTargetOwned {
    Session(String),
    Tokens {
        prompt: Vec<u32>,
        cache_salt: String,
    },
}

/// Why a prefetch was not queued.
#[derive(Debug, PartialEq, Eq)]
pub enum PrefetchRefused {
    Kv(PrefetchError),
    /// The engine thread has stopped.
    EngineGone,
}

/// Commands from the HTTP side, drained every iteration.
pub enum KvCommand {
    Prefetch {
        target: PrefetchTargetOwned,
        reply: oneshot::Sender<Result<PrefetchAccepted, PrefetchError>>,
    },
}

/// The Tokio side of the orchestrator: a bounded command channel of capacity
/// `kv.prefetch.max_queue`.
#[derive(Clone)]
pub struct KvHandle {
    tx: mpsc::Sender<KvCommand>,
}

impl KvHandle {
    /// `POST /turbine/v1/kv/prefetch`. A full command channel is `QueueFull` at once: a
    /// prefetch never waits (P4 Failure modes, "prefetch overload").
    pub async fn prefetch(
        &self,
        target: PrefetchTargetOwned,
    ) -> Result<PrefetchAccepted, PrefetchRefused> {
        let (reply, answer) = oneshot::channel();
        match self.tx.try_send(KvCommand::Prefetch { target, reply }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(PrefetchRefused::Kv(PrefetchError::QueueFull));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(PrefetchRefused::EngineGone),
        }
        match answer.await {
            Ok(r) => r.map_err(PrefetchRefused::Kv),
            Err(_) => Err(PrefetchRefused::EngineGone),
        }
    }
}

/// Startup inputs of [`KvOrchestrator::start`].
pub struct KvStart<'a> {
    pub cfg: &'a KvConfig,
    pub memory_kind: MemoryKind,
    pub identity: ModelIdentity,
    pub device: CopyDevice,
    pub l2: Option<Arc<L2NvmeTier>>,
    pub clock: Arc<dyn Clock>,
    pub metrics: KvMetrics,
}

/// Owns the hierarchy and its copy backend on the engine thread (module comment).
pub struct KvOrchestrator {
    h: KvHierarchy,
    backend: CopyStreamBackend,
    clock: Arc<dyn Clock>,
    commands: mpsc::Receiver<KvCommand>,
    hits: HitWindow,
    prefill_tps: Option<f64>,
}

impl KvOrchestrator {
    /// Builds the tiers (L1 only with a copy stream on a dedicated-memory device), the
    /// hierarchy over `pool` and the copy backend, then calibrates every enabled path.
    /// Exit 1 (`StartupError`) when `kv.cpu.enabled` and the first pinned slab cannot be
    /// allocated.
    pub fn start(
        s: KvStart<'_>,
        pool: &mut BlockPool,
    ) -> Result<(KvOrchestrator, KvHandle), StartupError> {
        let layout = pool.layout();
        let format = kv_format(layout);
        let block_bytes = layout.block_bytes();
        let l1 = match &s.device {
            CopyDevice::Stream { pinned, .. } if s.cfg.cpu.enabled => {
                Some(Arc::new(L1PinnedTier::new(
                    L1Config {
                        enabled: true,
                        max_bytes: s.cfg.cpu.max_bytes.0,
                        slab_bytes: L1_SLAB_BYTES,
                        block_bytes,
                        memory_kind: s.memory_kind,
                    },
                    Arc::clone(pinned),
                    Arc::clone(&s.clock),
                )))
            }
            CopyDevice::Sync { .. } if s.cfg.cpu.enabled => {
                tracing::warn!(
                    event = "kv_l1_disabled_no_pinned_memory",
                    tier = "l1",
                    "no pinned-memory API on this backend; L1 disabled"
                );
                None
            }
            _ => None,
        };
        let l1 = l1.filter(|t| t.enabled());
        let l2_dyn = s.l2.clone().map(|t| t as Arc<dyn KvTier>);
        // `kv.policy` was checked against the registry before any port was bound
        // (`Config::validate_modules`); `select` logs `module_selected`.
        turbine_kv::policy::registry()
            .select(s.cfg.policy.as_str(), "kv.policy")
            .map_err(|e| StartupError::new(e.to_string()))?;
        let h = KvHierarchy::new(
            HierarchyConfig::from_config(s.cfg, block_bytes, s.memory_kind)
                .map_err(|e| StartupError::new(e.to_string()))?,
            s.identity,
            format,
            pool.total_blocks(),
            l1.clone().map(|t| t as Arc<dyn KvTier>),
            l2_dyn.clone(),
            Arc::clone(&s.clock),
            s.metrics,
        );
        let io_threads = s.cfg.nvme.io_threads.max(1) as usize;
        // Every I/O job belongs to an in-flight ticket, so this bound is never reached; the
        // L2 tier bounds concurrent I/O itself (`kv.nvme.max_queue_depth`).
        let io_capacity = (s.cfg.transfer.max_inflight_bytes.0 / block_bytes.max(1)) as usize + 1;
        let backend = CopyStreamBackend::new(
            s.device,
            BlockAddresses::of(pool),
            l1.clone(),
            l2_dyn,
            IoPoolBackend::new(io_threads, io_capacity),
            block_bytes as usize,
        );
        let (tx, commands) = mpsc::channel(s.cfg.prefetch.max_queue.max(1) as usize);
        let mut o = KvOrchestrator {
            h,
            backend,
            clock: s.clock,
            commands,
            hits: HitWindow::default(),
            prefill_tps: None,
        };
        o.calibrate(
            pool,
            s.cfg.cpu.enabled && l1.is_some(),
            s.l2.as_deref(),
            s.cfg.nvme.io_threads,
        )?;
        tracing::info!(
            event = "kv_hierarchy_ready",
            l1 = o.h.l1_enabled(),
            l2 = o.h.l2_enabled(),
            policy = s.cfg.policy.as_str(),
            prefix_sharing = s.cfg.prefix_sharing,
            block_tokens = layout.block_tokens,
            "KV hierarchy ready"
        );
        Ok((o, KvHandle { tx }))
    }

    /// The lock-free reclaim requests the Phase 3 controller issues (`KvReclaimer`).
    pub fn reclaimer(&self) -> Arc<KvReclaimHandle> {
        self.h.reclaimer()
    }

    /// No copy is queued or in flight.
    pub fn transfers_idle(&self) -> bool {
        self.h.transfer().is_idle()
    }

    /// Admission-time prefix match of a request (P4 S-3).
    pub fn attach(
        &mut self,
        pool: &mut BlockPool,
        request: RequestId,
        prompt: &[u32],
        cache_salt: &str,
        session: Option<&SessionHints>,
        priority: Priority,
    ) -> AttachOutcome {
        let outcome = self.h.attach_prefix(
            pool,
            &AttachRequest {
                request,
                prompt,
                cache_salt,
                session,
                priority,
            },
        );
        if let AttachOutcome::Ready(a) = &outcome {
            self.record_hit(prompt.len(), a);
        }
        outcome
    }

    /// Transfer completions: requests whose promotions all landed, with their prefixes.
    pub fn poll(&mut self, pool: &mut BlockPool) -> Vec<(RequestId, PrefixAttach)> {
        self.h.poll(pool, &mut self.backend)
    }

    /// Counts a request's prompt and cached tokens in the 300 s hit-rate window.
    pub fn record_hit(&mut self, prompt_tokens: usize, attach: &PrefixAttach) {
        let now = self.clock.now_mono().as_secs();
        self.hits
            .record(now, prompt_tokens as u64, u64::from(attach.cached_tokens));
    }

    /// Serves the queued prefetch commands.
    pub fn serve_commands(&mut self, pool: &mut BlockPool) {
        while let Ok(cmd) = self.commands.try_recv() {
            match cmd {
                KvCommand::Prefetch { target, reply } => {
                    let result = match &target {
                        PrefetchTargetOwned::Session(id) => {
                            self.h.prefetch(pool, PrefetchTarget::Session(id))
                        }
                        PrefetchTargetOwned::Tokens { prompt, cache_salt } => self
                            .h
                            .prefetch(pool, PrefetchTarget::Tokens { prompt, cache_salt }),
                    };
                    let _ = reply.send(result);
                }
            }
        }
    }

    /// Before `Scheduler::plan`: refreshes the reclaim order when free L0 blocks run short,
    /// hands the planner the pressure controller's state (`l0_pressure` at RED and above,
    /// prefetch only at GREEN/YELLOW) and keeps L0 headroom by capacity (P4 S-8 "an allocation
    /// needing blocks"): while cached blocks fill the pool past [`CAPACITY_DEMOTE_AT`], the
    /// lowest-valued unreferenced ones are copied down (reason `capacity`), so an allocation
    /// reclaims blocks that already have a lower-tier copy instead of dropping the valuable
    /// ones. The controller's own reclaim (reservation pressure) arrives through
    /// [`KvOrchestrator::reclaimer`].
    pub fn before_plan(&mut self, pool: &mut BlockPool, state: PressureState) {
        if pool.free_blocks() < pool.total_blocks().div_ceil(REFRESH_FREE_SHARE) {
            self.h.refresh_reclaim_order(pool);
        }
        self.h.set_l0_state(state);
        let total = f64::from(pool.total_blocks().max(1));
        if pool.cached_unreferenced() > 0
            && f64::from(pool.used_blocks()) / total >= CAPACITY_DEMOTE_AT
        {
            self.h
                .demote_to(pool, CAPACITY_DEMOTE_AT, EvictReason::Capacity);
        }
    }

    /// After `Scheduler::plan`: the directory forgets L0 blocks the plan's allocations
    /// reclaimed.
    pub fn after_plan(&mut self, pool: &mut BlockPool) {
        self.h.after_plan(pool);
    }

    /// The KV the iteration wrote: `tokens` are the tokens of `blocks` so far.
    pub fn commit(
        &mut self,
        pool: &mut BlockPool,
        request: RequestId,
        blocks: &[BlockId],
        tokens: &[u32],
    ) {
        self.h.commit_progress(pool, request, blocks, tokens);
    }

    /// The request finished (`cancelled`: dropped before completing).
    pub fn request_done(&mut self, pool: &mut BlockPool, request: RequestId, cancelled: bool) {
        self.h.request_done(pool, request, cancelled);
    }

    /// End of an iteration: pending reclaim requests, session TTLs and predicted-resume
    /// prefetches.
    pub fn end_turn(&mut self, pool: &mut BlockPool) {
        self.h.apply_reclaim(pool);
        self.h.tick(pool);
    }

    /// One executed iteration prefilled `tokens` tokens in `seconds`: folds the rate into the
    /// prefill EWMA the planner's recompute cost uses.
    pub fn record_prefill(&mut self, tokens: u32, seconds: f64) {
        if tokens < PREFILL_MIN_TOKENS || seconds <= 0.0 {
            return;
        }
        let sample = f64::from(tokens) / seconds;
        let tps = match self.prefill_tps {
            Some(prev) => prev + PREFILL_ALPHA * (sample - prev),
            None => sample,
        };
        self.prefill_tps = Some(tps);
        self.h.set_prefill_tps(tps);
    }

    /// `GET /turbine/v1/kv` (P4 §Data).
    pub fn document(&self, pool: &BlockPool) -> KvDocument {
        let now = self.clock.now_mono().as_secs();
        self.h.document(pool, self.hits.totals(now))
    }

    /// Seeds every enabled path's estimate from one 64 MiB copy through the production copy
    /// paths (module comment).
    fn calibrate(
        &mut self,
        pool: &mut BlockPool,
        l1_required: bool,
        l2: Option<&L2NvmeTier>,
        io_threads: u32,
    ) -> Result<(), StartupError> {
        let bb = pool.layout().block_bytes();
        let blocks = CALIBRATION_BYTES.div_ceil(bb.max(1)).max(1);
        let mut measured: Vec<(TransferPath, PathCost)> = Vec::new();
        if self.h.l1_enabled() {
            match self.calibrate_l1(pool, blocks) {
                Ok(costs) => measured.extend(costs),
                Err(CalibrationError::FirstSlab(e)) if l1_required => {
                    return Err(StartupError::new(format!(
                        "kv.cpu.enabled: the first pinned L1 slab cannot be allocated: {e}"
                    )));
                }
                Err(e) => tracing::warn!(
                    event = "kv_calibration_failed",
                    path = "l0_l1",
                    error = %e,
                    "L0 <-> L1 calibration failed; using the conservative estimates"
                ),
            }
        }
        if let Some(l2) = l2 {
            match calibrate_l2(l2, bb, blocks) {
                Ok((write, read, p99)) => {
                    // The baseline of the `storage_latency` signal and of the tier's own "slow"
                    // rule: one lone I/O's p99 service time, times the I/Os that share the
                    // device in operation (`kv.nvme.io_threads`), since each then gets that
                    // share of its bandwidth. A lone-I/O baseline read a demotion burst on a
                    // healthy disk as 20x slow and drove the pressure state to RED.
                    l2.set_calibration(p99 * f64::from(io_threads.max(1)), read.bandwidth_bps);
                    measured.extend([
                        (TransferPath::L1ToL2, write),
                        (TransferPath::L0ToL2, write),
                        (TransferPath::L2ToL1, read),
                        (TransferPath::L2ToL0, read),
                    ]);
                }
                Err(e) => tracing::warn!(
                    event = "kv_calibration_failed",
                    path = "l2",
                    error = %e,
                    "L2 calibration failed; using the conservative estimates"
                ),
            }
        }
        for (path, cost) in &measured {
            self.h.transfer_mut().seed(*path, *cost);
        }
        let bandwidth = |p: TransferPath| {
            measured
                .iter()
                .find(|(q, _)| *q == p)
                .map(|(_, c)| c.bandwidth_bps)
        };
        tracing::info!(
            event = "kv_calibration",
            bytes = blocks * bb,
            l0_to_l1_bytes_per_second = bandwidth(TransferPath::L0ToL1),
            l1_to_l0_bytes_per_second = bandwidth(TransferPath::L1ToL0),
            to_l2_bytes_per_second = bandwidth(TransferPath::L1ToL2),
            from_l2_bytes_per_second = bandwidth(TransferPath::L2ToL1),
            "KV transfer calibration"
        );
        Ok(())
    }

    /// L0 → L1 and back through the copy stream, `blocks` blocks each way (at most what L0
    /// has free).
    fn calibrate_l1(
        &mut self,
        pool: &mut BlockPool,
        blocks: u64,
    ) -> Result<[(TransferPath, PathCost); 2], CalibrationError> {
        let CopyDevice::Stream { engine, .. } = &self.backend.device else {
            return Err(CalibrationError::Other("no copy stream".into()));
        };
        let engine = Arc::clone(engine);
        let l1 = self
            .backend
            .l1
            .clone()
            .ok_or_else(|| CalibrationError::Other("L1 disabled".into()))?;
        let n = blocks.min(u64::from(pool.free_blocks())).max(1) as u32;
        let ids = pool
            .allocate(n)
            .map_err(|e| CalibrationError::Other(format!("L0 blocks: {e}")))?;
        let keys: Vec<KvKey> = (0..n).map(calibration_key).collect();
        let result = (|| {
            let mut slots = Vec::with_capacity(ids.len());
            for key in &keys {
                match l1.reserve(*key) {
                    Ok(slot) => slots.push(slot),
                    Err(e) if l1.slab_count() == 0 => {
                        return Err(CalibrationError::FirstSlab(e.to_string()));
                    }
                    Err(e) => return Err(CalibrationError::Other(e.to_string())),
                }
            }
            let bb = self.backend.block_bytes as u64;
            let timed = |to_l1: bool| -> Result<PathCost, CalibrationError> {
                let started = Instant::now();
                let mut tickets = Vec::new();
                for (b, &(buffer_id, offset)) in ids.iter().zip(&slots) {
                    let mut acc = 0usize;
                    for &(ptr, len) in self.backend.addresses.segments(*b) {
                        let pinned = CopyTarget::Pinned {
                            buffer_id,
                            offset: offset + acc,
                        };
                        let (dst, src) = if to_l1 {
                            (pinned, CopyTarget::Device(ptr))
                        } else {
                            (CopyTarget::Device(ptr), pinned)
                        };
                        tickets.push(
                            engine
                                .copy_async(dst, src, len)
                                .map_err(|e| CalibrationError::Other(e.to_string()))?,
                        );
                        acc += len;
                    }
                }
                for t in &tickets {
                    engine
                        .wait(t)
                        .map_err(|e| CalibrationError::Other(e.to_string()))?;
                }
                Ok(cost_of(started.elapsed(), u64::from(n) * bb, n))
            };
            let down = timed(true)?;
            let up = timed(false)?;
            Ok([(TransferPath::L0ToL1, down), (TransferPath::L1ToL0, up)])
        })();
        for key in &keys {
            l1.abort_reservation(key);
        }
        pool.release(&ids);
        if let Ok([_, (_, up)]) = &result {
            l1.set_estimates(Duration::from_secs_f64(up.latency_s), up.bandwidth_bps);
        }
        result
    }
}

#[derive(Debug)]
enum CalibrationError {
    /// L1 could not allocate its first pinned slab.
    FirstSlab(String),
    Other(String),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::FirstSlab(e) => write!(f, "first pinned slab: {e}"),
            CalibrationError::Other(e) => f.write_str(e),
        }
    }
}

/// A key no prompt produces in practice (a BLAKE3 prefix of all `0xCA` bytes but the index).
fn calibration_key(i: u32) -> KvKey {
    let mut k = [0xCA; 16];
    k[12..].copy_from_slice(&i.to_le_bytes());
    KvKey(k)
}

/// The path cost of `blocks` block copies of `bytes` in total that took `took`: bandwidth over
/// the whole copy, latency the per-block share left beyond it (floor 1 µs).
fn cost_of(took: Duration, bytes: u64, blocks: u32) -> PathCost {
    let secs = took.as_secs_f64().max(1e-9);
    PathCost {
        latency_s: (secs / f64::from(blocks.max(1)) * 0.05).max(1e-6),
        bandwidth_bps: bytes as f64 / secs,
    }
}

/// Writes then reads `blocks` blocks through L2, one at a time; returns (write cost, read cost,
/// p99 of the per-block write and read latencies in seconds). The blocks are evicted afterwards.
fn calibrate_l2(
    l2: &L2NvmeTier,
    block_bytes: u64,
    blocks: u64,
) -> Result<(PathCost, PathCost, f64), TierError> {
    let n = blocks.min((l2.capacity_bytes() / block_bytes.max(1)).max(1)) as u32;
    let mut buf = vec![0u8; block_bytes as usize];
    let keys: Vec<KvKey> = (0..n).map(calibration_key).collect();
    let result = (|| {
        let mut ops = Vec::with_capacity(2 * keys.len());
        let started = Instant::now();
        for key in &keys {
            let one = Instant::now();
            l2.put(*key, TierBlockRef::Host(&buf))?;
            ops.push(one.elapsed().as_secs_f64());
        }
        let write = cost_of(started.elapsed(), u64::from(n) * block_bytes, n);
        let started = Instant::now();
        for key in &keys {
            let one = Instant::now();
            l2.get(key, TierBlockMut::Host(&mut buf))?;
            ops.push(one.elapsed().as_secs_f64());
        }
        let read = cost_of(started.elapsed(), u64::from(n) * block_bytes, n);
        ops.sort_by(f64::total_cmp);
        let p99 = ops[((ops.len() * 99).div_ceil(100)).saturating_sub(1)];
        Ok((write, read, p99))
    })();
    for key in &keys {
        let _ = l2.evict(key);
    }
    result
}

/// The device segments of every L0 block, computed once from the pool (its storage never
/// moves): the copy backend has no pool access while a copy runs.
pub struct BlockAddresses {
    blocks: Vec<SmallVec<[(DevicePtr, usize); 32]>>,
}

impl BlockAddresses {
    pub fn of(pool: &BlockPool) -> BlockAddresses {
        BlockAddresses {
            blocks: (0..pool.total_blocks())
                .map(|b| pool.block_segments(BlockId(b)))
                .collect(),
        }
    }

    fn segments(&self, b: BlockId) -> &[(DevicePtr, usize)] {
        self.blocks.get(b.0 as usize).map_or(&[], |s| s.as_slice())
    }
}

/// A host staging buffer of one block: page-locked with a copy stream, heap memory otherwise.
enum HostBuf {
    Pinned(PinnedBuffer),
    Heap(Vec<u8>),
}

impl HostBuf {
    fn with<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        match self {
            HostBuf::Pinned(b) => b.with_bytes(f),
            HostBuf::Heap(v) => f(v),
        }
    }

    fn with_mut<R>(&mut self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        match self {
            HostBuf::Pinned(b) => b.with_bytes_mut(f),
            HostBuf::Heap(v) => f(v),
        }
    }
}

/// One file-I/O job of the pool.
enum IoOp {
    /// L1 ↔ L2: read the block from one tier and store it in the other.
    Move {
        from: Arc<dyn KvTier>,
        to: Arc<dyn KvTier>,
        key: KvKey,
        bytes: usize,
    },
    /// The staged block of an L0 → L2 copy.
    Write {
        to: Arc<dyn KvTier>,
        key: KvKey,
        buf: HostBuf,
    },
    /// The first half of an L2 → L0 copy.
    Read {
        from: Arc<dyn KvTier>,
        key: KvKey,
        buf: HostBuf,
    },
}

struct IoDone {
    ticket: u64,
    result: Result<TierSlot, TierError>,
    buf: Option<HostBuf>,
}

fn run_io(op: IoOp) -> (Result<TierSlot, TierError>, Option<HostBuf>) {
    match op {
        IoOp::Move {
            from,
            to,
            key,
            bytes,
        } => {
            let mut v = vec![0u8; bytes];
            let r = from
                .get(&key, TierBlockMut::Host(&mut v))
                .and_then(|()| to.put(key, TierBlockRef::Host(&v)));
            (r, None)
        }
        IoOp::Write { to, key, buf } => {
            let r = buf.with(|b| to.put(key, TierBlockRef::Host(b)));
            (r, Some(buf))
        }
        IoOp::Read { from, key, mut buf } => {
            let r = buf
                .with_mut(|b| from.get(&key, TierBlockMut::Host(b)))
                .map(|()| TierSlot(0));
            (r, Some(buf))
        }
    }
}

/// L1 ↔ L2 copies and the file half of L0 ↔ L2 copies on `kv.nvme.io_threads` threads (P4
/// S-6). Jobs wait in a bounded channel; results come back on another and are collected on the
/// engine thread by `poll`.
pub struct IoPoolBackend {
    jobs: Option<std_mpsc::SyncSender<(u64, IoOp)>>,
    results: std_mpsc::Receiver<IoDone>,
    done: HashMap<u64, IoDone>,
    threads: Vec<JoinHandle<()>>,
    l1: Option<Arc<dyn KvTier>>,
    l2: Option<Arc<dyn KvTier>>,
    block_bytes: usize,
}

impl IoPoolBackend {
    pub fn new(threads: usize, capacity: usize) -> IoPoolBackend {
        let (jobs, rx) = std_mpsc::sync_channel::<(u64, IoOp)>(capacity.max(1));
        let (results_tx, results) = std_mpsc::channel();
        let rx = Arc::new(Mutex::new(rx));
        let threads = (0..threads.max(1))
            .filter_map(|i| {
                let rx = Arc::clone(&rx);
                let tx = results_tx.clone();
                std::thread::Builder::new()
                    .name(format!("turbine-kv-io-{i}"))
                    .spawn(move || {
                        loop {
                            let job = rx
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .recv();
                            let Ok((ticket, op)) = job else { return };
                            let (result, buf) = run_io(op);
                            if tx
                                .send(IoDone {
                                    ticket,
                                    result,
                                    buf,
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                    })
                    .map_err(|e| tracing::error!(error = %e, "cannot start a KV I/O thread"))
                    .ok()
            })
            .collect();
        IoPoolBackend {
            jobs: Some(jobs),
            results,
            done: HashMap::new(),
            threads,
            l1: None,
            l2: None,
            block_bytes: 0,
        }
    }

    fn submit(&mut self, ticket: u64, op: IoOp) -> Result<(), TierError> {
        let jobs = self.jobs.as_ref().ok_or(TierError::Degraded)?;
        jobs.try_send((ticket, op)).map_err(|e| match e {
            std_mpsc::TrySendError::Full(_) => TierError::Full,
            std_mpsc::TrySendError::Disconnected(_) => TierError::Io("KV I/O pool stopped".into()),
        })
    }

    fn collect(&mut self) {
        while let Ok(d) = self.results.try_recv() {
            self.done.insert(d.ticket, d);
        }
    }

    fn take(&mut self, ticket: u64) -> Option<IoDone> {
        self.collect();
        self.done.remove(&ticket)
    }

    fn tier(&self, t: TierId) -> Option<Arc<dyn KvTier>> {
        match t {
            TierId::L1 => self.l1.clone(),
            TierId::L2 => self.l2.clone(),
            _ => None,
        }
    }
}

impl TransferBackend for IoPoolBackend {
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
        let (from, to) = (t.req.path.from(), t.req.path.to());
        let (Some(from), Some(to)) = (self.tier(from), self.tier(to)) else {
            return Err(TierError::Missing);
        };
        self.submit(
            t.id,
            IoOp::Move {
                from,
                to,
                key: t.req.key,
                bytes: self.block_bytes,
            },
        )
    }

    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
        match self.take(t.id) {
            None => Ok(None),
            Some(d) => d.result.map(Some),
        }
    }
}

impl Drop for IoPoolBackend {
    fn drop(&mut self) {
        // Closing the job channel ends every thread after its current job.
        self.jobs = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// What finishes a ticket once its stream copies completed.
enum AfterCopies {
    /// L0 → L1: the reserved slot becomes visible.
    CommitL1(KvKey),
    /// → L0: done; the staging buffer (if any) goes back to the pool.
    IntoL0 { block: u64, buf: Option<HostBuf> },
    /// L0 → L2: the staged block is written by the I/O pool.
    WriteL2 { key: KvKey, buf: HostBuf },
}

enum IoStage {
    /// The I/O result completes the ticket.
    Final,
    /// L2 → L0: the block read into the staging buffer is copied into L0 block `block`.
    ThenIntoL0 { block: u64 },
}

enum Job {
    Copies {
        copies: Vec<CopyTicket>,
        then: AfterCopies,
    },
    Io(IoStage),
    Done(Result<TierSlot, TierError>),
}

/// Moves block bytes for the hierarchy's transfer engine (P4 S-6): L0 ↔ L1 on the device's
/// copy stream straight into reserved L1 slots, L0 ↔ L2 through a one-block staging buffer
/// (pinned with a copy stream, heap memory with synchronous copies) plus the I/O pool, L1 ↔ L2
/// on the [`IoPoolBackend`]. Copy errors abort the L1 reservation and count toward L1's
/// degraded window; the hierarchy then recomputes.
pub struct CopyStreamBackend {
    device: CopyDevice,
    addresses: BlockAddresses,
    l1: Option<Arc<L1PinnedTier>>,
    l2: Option<Arc<dyn KvTier>>,
    io: IoPoolBackend,
    block_bytes: usize,
    staging: Vec<HostBuf>,
    jobs: HashMap<u64, Job>,
}

impl CopyStreamBackend {
    pub fn new(
        device: CopyDevice,
        addresses: BlockAddresses,
        l1: Option<Arc<L1PinnedTier>>,
        l2: Option<Arc<dyn KvTier>>,
        mut io: IoPoolBackend,
        block_bytes: usize,
    ) -> CopyStreamBackend {
        io.l1 = l1.clone().map(|t| t as Arc<dyn KvTier>);
        io.l2 = l2.clone();
        io.block_bytes = block_bytes;
        CopyStreamBackend {
            device,
            addresses,
            l1,
            l2,
            io,
            block_bytes,
            staging: Vec::new(),
            jobs: HashMap::new(),
        }
    }

    fn staging_buf(&mut self) -> Result<HostBuf, TierError> {
        if let Some(b) = self.staging.pop() {
            return Ok(b);
        }
        match &self.device {
            CopyDevice::Stream { pinned, .. } => pinned
                .alloc_pinned(self.block_bytes)
                .map(HostBuf::Pinned)
                .map_err(|e| TierError::Io(format!("staging buffer: {e}"))),
            CopyDevice::Sync { .. } => Ok(HostBuf::Heap(vec![0; self.block_bytes])),
        }
    }

    fn release_staging(&mut self, buf: HostBuf) {
        self.staging.push(buf);
    }

    /// Enqueues the per-layer copies of L0 block `block` into (`to_device` false) or out of
    /// pinned buffer `buffer_id` starting at byte `offset`.
    fn stream_copies(
        &self,
        engine: &Arc<dyn CopyEngine>,
        block: u64,
        buffer_id: u64,
        offset: usize,
        to_device: bool,
    ) -> Result<Vec<CopyTicket>, TierError> {
        let mut tickets = Vec::new();
        let mut acc = 0usize;
        for &(ptr, len) in self.addresses.segments(BlockId(block as u32)) {
            let pinned = CopyTarget::Pinned {
                buffer_id,
                offset: offset + acc,
            };
            let (dst, src) = if to_device {
                (CopyTarget::Device(ptr), pinned)
            } else {
                (pinned, CopyTarget::Device(ptr))
            };
            match engine.copy_async(dst, src, len) {
                Ok(t) => tickets.push(t),
                Err(e) => {
                    // Nothing may still write the destination when the caller releases it.
                    for t in &tickets {
                        let _ = engine.wait(t);
                    }
                    return Err(TierError::Io(format!("copy stream: {e}")));
                }
            }
            acc += len;
        }
        Ok(tickets)
    }

    /// Synchronous copy of L0 block `block` into (`to_device` false) or from `buf`.
    fn sync_copy(
        &self,
        mem: &Arc<dyn DeviceMemory>,
        block: u64,
        buf: &mut HostBuf,
        to_device: bool,
    ) -> Result<(), TierError> {
        let segments = self.addresses.segments(BlockId(block as u32));
        buf.with_mut(|bytes| {
            let mut acc = 0usize;
            for &(ptr, len) in segments {
                let part = &mut bytes[acc..acc + len];
                let r = if to_device {
                    mem.copy_h2d(ptr, part)
                } else {
                    mem.copy_d2h(part, ptr)
                };
                r.map_err(|e| TierError::Io(format!("device copy: {e}")))?;
                acc += len;
            }
            Ok(())
        })
    }

    fn start_job(&mut self, t: &TransferTicket) -> Result<Job, TierError> {
        let req = &t.req;
        match req.path {
            TransferPath::L1ToL2 | TransferPath::L2ToL1 => {
                self.io.start(t)?;
                Ok(Job::Io(IoStage::Final))
            }
            TransferPath::L0ToL1 => {
                let CopyDevice::Stream { engine, .. } = self.device.clone() else {
                    return Err(TierError::Io("L1 needs a copy stream".into()));
                };
                let l1 = self.l1.clone().ok_or(TierError::Missing)?;
                let (buffer_id, offset) = l1.reserve(req.key)?;
                match self.stream_copies(&engine, req.src_slot, buffer_id, offset, false) {
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::CommitL1(req.key),
                    }),
                    Err(e) => {
                        l1.abort_reservation(&req.key);
                        l1.record_copy_error();
                        Err(e)
                    }
                }
            }
            TransferPath::L1ToL0 => {
                let CopyDevice::Stream { engine, .. } = self.device.clone() else {
                    return Err(TierError::Io("L1 needs a copy stream".into()));
                };
                let l1 = self.l1.clone().ok_or(TierError::Missing)?;
                let (buffer_id, offset) = l1.locate(&req.key).ok_or(TierError::Missing)?;
                match self.stream_copies(&engine, req.dst_slot, buffer_id, offset, true) {
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::IntoL0 {
                            block: req.dst_slot,
                            buf: None,
                        },
                    }),
                    Err(e) => {
                        l1.record_copy_error();
                        Err(e)
                    }
                }
            }
            TransferPath::L0ToL2 => {
                let l2 = self.l2.clone().ok_or(TierError::Missing)?;
                let mut buf = self.staging_buf()?;
                match self.device.clone() {
                    CopyDevice::Stream { engine, .. } => {
                        let HostBuf::Pinned(p) = &buf else {
                            unreachable!("a copy stream stages in pinned memory")
                        };
                        let id = p.id();
                        match self.stream_copies(&engine, req.src_slot, id, 0, false) {
                            Ok(copies) => Ok(Job::Copies {
                                copies,
                                then: AfterCopies::WriteL2 { key: req.key, buf },
                            }),
                            Err(e) => {
                                self.release_staging(buf);
                                Err(e)
                            }
                        }
                    }
                    CopyDevice::Sync { mem } => {
                        if let Err(e) = self.sync_copy(&mem, req.src_slot, &mut buf, false) {
                            self.release_staging(buf);
                            return Err(e);
                        }
                        self.io.submit(
                            t.id,
                            IoOp::Write {
                                to: l2,
                                key: req.key,
                                buf,
                            },
                        )?;
                        Ok(Job::Io(IoStage::Final))
                    }
                }
            }
            TransferPath::L2ToL0 => {
                let l2 = self.l2.clone().ok_or(TierError::Missing)?;
                let buf = self.staging_buf()?;
                self.io.submit(
                    t.id,
                    IoOp::Read {
                        from: l2,
                        key: req.key,
                        buf,
                    },
                )?;
                Ok(Job::Io(IoStage::ThenIntoL0 {
                    block: req.dst_slot,
                }))
            }
        }
    }

    fn finish_copies(&mut self, t: &TransferTicket, then: AfterCopies) -> Result<Job, TierError> {
        match then {
            AfterCopies::CommitL1(key) => {
                let l1 = self.l1.clone().ok_or(TierError::Missing)?;
                Ok(Job::Done(Ok(l1.commit(&key))))
            }
            AfterCopies::IntoL0 { block, buf } => {
                if let Some(buf) = buf {
                    self.release_staging(buf);
                }
                Ok(Job::Done(Ok(TierSlot(block))))
            }
            AfterCopies::WriteL2 { key, buf } => {
                let l2 = self.l2.clone().ok_or(TierError::Missing)?;
                self.io.submit(t.id, IoOp::Write { to: l2, key, buf })?;
                Ok(Job::Io(IoStage::Final))
            }
        }
    }

    fn finish_io(&mut self, stage: IoStage, done: IoDone) -> Result<Job, TierError> {
        let IoDone { result, buf, .. } = done;
        match stage {
            IoStage::Final => {
                if let Some(buf) = buf {
                    self.release_staging(buf);
                }
                Ok(Job::Done(result))
            }
            IoStage::ThenIntoL0 { block } => {
                let mut buf = buf.ok_or(TierError::Missing)?;
                if let Err(e) = result {
                    self.release_staging(buf);
                    return Err(e);
                }
                match self.device.clone() {
                    CopyDevice::Stream { engine, .. } => {
                        let HostBuf::Pinned(p) = &buf else {
                            unreachable!("a copy stream stages in pinned memory")
                        };
                        let id = p.id();
                        match self.stream_copies(&engine, block, id, 0, true) {
                            Ok(copies) => Ok(Job::Copies {
                                copies,
                                then: AfterCopies::IntoL0 {
                                    block,
                                    buf: Some(buf),
                                },
                            }),
                            Err(e) => {
                                self.release_staging(buf);
                                Err(e)
                            }
                        }
                    }
                    CopyDevice::Sync { mem } => {
                        let r = self.sync_copy(&mem, block, &mut buf, true);
                        self.release_staging(buf);
                        r.map(|()| Job::Done(Ok(TierSlot(block))))
                    }
                }
            }
        }
    }

    /// Advances ticket `t` as far as it goes now.
    fn advance(&mut self, t: &TransferTicket, job: Job) -> Result<Job, TierError> {
        match job {
            Job::Done(r) => Ok(Job::Done(r)),
            Job::Io(stage) => match self.io.take(t.id) {
                None => Ok(Job::Io(stage)),
                Some(done) => self.finish_io(stage, done),
            },
            Job::Copies { copies, then } => {
                let CopyDevice::Stream { engine, .. } = self.device.clone() else {
                    unreachable!("stream copies need a copy stream")
                };
                let mut all = true;
                for c in &copies {
                    match engine.poll(c) {
                        Ok(true) => {}
                        Ok(false) => all = false,
                        Err(e) => {
                            for c in &copies {
                                let _ = engine.wait(c);
                            }
                            self.copy_failed(t, then);
                            return Err(TierError::Io(format!("copy stream: {e}")));
                        }
                    }
                }
                if all {
                    self.finish_copies(t, then)
                } else {
                    Ok(Job::Copies { copies, then })
                }
            }
        }
    }

    fn copy_failed(&mut self, t: &TransferTicket, then: AfterCopies) {
        match then {
            AfterCopies::CommitL1(key) => {
                if let Some(l1) = &self.l1 {
                    l1.abort_reservation(&key);
                    l1.record_copy_error();
                }
            }
            AfterCopies::IntoL0 { buf, .. } => {
                if t.req.path == TransferPath::L1ToL0
                    && let Some(l1) = &self.l1
                {
                    l1.record_copy_error();
                }
                if let Some(buf) = buf {
                    self.release_staging(buf);
                }
            }
            AfterCopies::WriteL2 { buf, .. } => self.release_staging(buf),
        }
    }
}

impl TransferBackend for CopyStreamBackend {
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
        let job = self.start_job(t)?;
        self.jobs.insert(t.id, job);
        Ok(())
    }

    fn poll(&mut self, t: &TransferTicket) -> Result<Option<TierSlot>, TierError> {
        let job = self.jobs.remove(&t.id).ok_or(TierError::Missing)?;
        match self.advance(t, job)? {
            Job::Done(r) => r.map(Some),
            pending => {
                self.jobs.insert(t.id, pending);
                Ok(None)
            }
        }
    }
}
