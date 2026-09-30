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
//! Tensor parallelism (Phase 5 §Data, decision "P5 T17" B): with tp ranks the orchestrator
//! runs on the leader over the leader's pool and every worker rank's [`KvShard`]. A logical
//! block is the tp rank shards in rank order: L1 is one pinned tier per rank
//! ([`ShardedL1Tier`], `kv.cpu.max_bytes / tp` each), L2 stores the concatenation, every shard
//! is copied on its own rank's copy stream, and the KV namespace carries the shard count
//! ([`tp_kv_format`]) so no other tp size ever reads the blobs. tp = 1 is unchanged. In
//! `static` rank mode ([`KvStart::remote`], P5 Task 30) the workers are other processes: the
//! leader's tiers hold its own shard and every copy and eviction is mirrored on the workers'
//! own tiers over the rank link (`crate::engine::tp_tiers`).
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
use turbine_core::telemetry::{StorageProbe, StorageSample};
use turbine_core::types::{
    BlockId, DType, KvDtype, KvLayout, MemoryKind, ModelIdentity, PressureState, RequestId,
};
use turbine_kv::codec::{CodecParams, KvCodec};
use turbine_kv::document::HitWindow;
use turbine_kv::hierarchy::{
    AttachOutcome, AttachRequest, HierarchyConfig, KvHierarchy, KvReclaimHandle, PrefetchAccepted,
    PrefetchError, PrefetchTarget, PrefixAttach,
};
use turbine_kv::identity::{KvFormat, KvKey, KvScaleHashes, namespace_key};
use turbine_kv::metrics::EvictReason;
use turbine_kv::planner::PathCost;
use turbine_kv::tier::{
    KvTier, L0_FORMAT, L1Config, L1PinnedTier, L2Config, L2NvmeTier, ShardSlots, ShardedL1Tier,
    TierBlockMut, TierBlockRef, TierError, TierId, TierSlot,
};
use turbine_kv::transfer::{
    TransferBackend, TransferCodec, TransferPath, TransferPurpose, TransferTicket,
};
use turbine_kv::{BlockPool, KvDocument, KvMetrics};
use turbine_model::kv_scales::KvCache;
use turbine_tensor::{
    CopyEngine, CopyTarget, CopyTicket, DeviceMemory, DevicePtr, PinnedBuffer, PinnedMemory,
};

use crate::engine::tp_tiers::TierDriver;
use crate::model::StartupError;

/// Bytes per L1 pinned slab (P4 S-5: grown lazily in 1 GiB slabs).
pub const L1_SLAB_BYTES: u64 = 1 << 30;
/// Bytes each calibration copy moves per path (P4 S-6).
pub const CALIBRATION_BYTES: u64 = 64 << 20;
/// Physical L0 fill (referenced plus cached blocks over the pool) above which cached blocks are
/// demoted by capacity, down to this fill: Phase 3's `kv_utilization` YELLOW threshold
/// (provisional decision "Phase 4: L0 capacity demotion").
pub const CAPACITY_DEMOTE_AT: f64 = 0.70;
/// Capacity demotion and the reclaim-order refresh run at most this often (engine-thread
/// work per turn stays bounded; the pool changes little within it).
const HOUSEKEEPING_INTERVAL: Duration = Duration::from_millis(50);
/// Below this share of free L0 blocks the pool's reclaim order is refreshed before planning.
const REFRESH_FREE_SHARE: u32 = 10;
/// EWMA weight of one prefill-rate sample.
const PREFILL_ALPHA: f64 = 0.2;
/// Prefill iterations smaller than this do not update the prefill rate (launch overhead).
const PREFILL_MIN_TOKENS: u32 = 64;

/// Refuses (exit 1) a lower-tier format the host path cannot store (P6b S-1): one the codec
/// cannot encode for this layout, or any format other than `l0` when a block is split into
/// rank or stage shards (the host codec works on one shard's layout).
fn check_tier_formats(
    cfg: &KvConfig,
    format: &KvFormat,
    sharded: bool,
) -> Result<(), StartupError> {
    for (on, key, name) in [
        (cfg.cpu.enabled, "kv.cpu.format", cfg.cpu.format.as_str()),
        (cfg.nvme.enabled, "kv.nvme.format", cfg.nvme.format.as_str()),
    ] {
        if !on || name == L0_FORMAT {
            continue;
        }
        let codec = turbine_kv::codec::registry()
            .get(name)
            .ok_or_else(|| StartupError::new(format!("{key}: no kv_format codec `{name}`")))?;
        codec
            .supports(&format.layout)
            .map_err(|e| StartupError::new(format!("{key} {name}: {e}")))?;
        if sharded {
            return Err(StartupError::new(format!(
                "{key} {name}: lower-tier formats other than l0 need one KV shard per block \
                 (no tensor or pipeline parallelism) until the GPU transcode"
            )));
        }
    }
    Ok(())
}

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
            block_bytes: format.block_bytes(),
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

/// The KV format of `layout` on one device: BF16 pages, or FP8 e4m3 pages with per-layer
/// scales (`kv.dtype: fp8_e4m3`, Phase 6a S-13; the scales join in [`KvOrchestrator::start`]
/// from [`KvStart::kv_scales`]). Lower tiers store the L0 page bytes as they are (S-14).
///
/// TurboQuant pages (P6b S-5) take their own namespace (`KvDtype::Tq4` / `Tq2`), so a block
/// cached under one L0 format is never attached under another.
pub fn kv_format(layout: KvLayout) -> KvFormat {
    let dtype = match layout.dtype {
        DType::F8E4M3 => KvDtype::Fp8E4m3PerTensorScale,
        DType::Tq4 => KvDtype::Tq4,
        DType::Tq2 => KvDtype::Tq2,
        _ => KvDtype::Bf16,
    };
    KvFormat::single(dtype, layout)
}

/// `format` scoped by the model's FP8 KV scales (Phase 6a S-16): required for FP8 pages,
/// refused for BF16 ones.
pub fn with_scales(
    format: KvFormat,
    scales: Option<KvScaleHashes>,
) -> Result<KvFormat, StartupError> {
    match (format.dtype, scales) {
        (KvDtype::Fp8E4m3PerTensorScale, Some(_))
        | (KvDtype::Bf16 | KvDtype::Tq4 | KvDtype::Tq2, None) => Ok(KvFormat { scales, ..format }),
        (dtype, _) => Err(StartupError::new(format!(
            "a {} KV pool {} per-layer KV scales",
            dtype.as_str(),
            if scales.is_some() {
                "takes no"
            } else {
                "needs the model's"
            }
        ))),
    }
}

/// The namespace hashes of a model's FP8 KV scales ([`KvStart::kv_scales`]); `None` for BF16.
pub fn scale_hashes(cache: &KvCache) -> Option<KvScaleHashes> {
    cache
        .is_fp8()
        .then(|| KvScaleHashes::of(&cache.k_scales, &cache.v_scales))
}

/// The KV format of a tensor-parallel group of `tp` ranks whose pools each have `layout` (one
/// rank's KV heads): a tier copy of a block is the `tp` rank shards concatenated. `tp` = 1 is
/// [`kv_format`].
pub fn tp_kv_format(layout: KvLayout, tp: u32) -> KvFormat {
    KvFormat {
        shards: tp.max(1),
        ..kv_format(layout)
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

/// One tensor-parallel worker rank's end of the tier copies (Phase 5, decision "P5 T17" B): its
/// copy device (its own kernel-library context: pinned buffers are only copy targets on the
/// context that allocated them) and its pool's block addresses. The pool lives on the rank's
/// own thread and must have the leader pool's block count and layout; it must outlive the
/// orchestrator, and the rank must not touch a block while the leader's pool has it free or
/// cached-unreferenced (tier copies read and write it then).
pub struct KvShard {
    pub device: CopyDevice,
    pub addresses: BlockAddresses,
}

/// Startup inputs of [`KvOrchestrator::start`].
pub struct KvStart<'a> {
    pub cfg: &'a KvConfig,
    pub memory_kind: MemoryKind,
    pub identity: ModelIdentity,
    /// Rank 0's copy device: the device of the pool `start` takes.
    pub device: CopyDevice,
    /// Tensor parallelism: ranks 1..tp in rank order; empty without it (tp = 1). A logical
    /// block is then `tp` × the pool's block bytes, and `l2` must have been opened with
    /// [`tp_kv_format`] of the same tp.
    pub shards: Vec<KvShard>,
    pub l2: Option<Arc<L2NvmeTier>>,
    pub clock: Arc<dyn Clock>,
    pub metrics: KvMetrics,
    /// `static` rank mode (P5 Task 30): the worker processes' tiers, driven over the rank link;
    /// `shards` is then empty and `l2` holds this rank's shards only
    /// (`crate::engine::tp_tiers::open_rank_l2`).
    pub remote: Option<TierDriver>,
    /// The FP8 KV scales' hashes ([`scale_hashes`] of the model's `kv_cache`), which scope the
    /// KV namespace (Phase 6a S-16); required with an FP8 pool, `None` with BF16.
    pub kv_scales: Option<KvScaleHashes>,
}

/// Owns the hierarchy and its copy backend on the engine thread (module comment).
pub struct KvOrchestrator {
    h: KvHierarchy,
    backend: CopyStreamBackend,
    clock: Arc<dyn Clock>,
    commands: mpsc::Receiver<KvCommand>,
    hits: HitWindow,
    prefill_tps: Option<f64>,
    /// When capacity demotion and the reclaim-order refresh last ran.
    last_housekeeping: Option<Duration>,
    /// `static` rank mode: every copy also runs on the workers (module comment).
    remote: Option<TierDriver>,
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
        KvOrchestrator::start_with(s, pool, false)
    }

    /// [`KvOrchestrator::start`] for a pipeline (P5 S-10): `pool` is the last stage's (the
    /// engine thread's) and `s.shards` are the stages before it, in stage order, each with the
    /// pool's block count and its own layers' block size. A logical block is every stage's
    /// shard in stage order — the model's layers in order, byte for byte one device's block —
    /// so its format is the whole model's ([`kv_format`] of every layer); L1 is one pinned tier
    /// per stage, each sized to its share of `kv.cpu.max_bytes` by its shard's bytes.
    pub fn start_pipeline(
        s: KvStart<'_>,
        pool: &mut BlockPool,
    ) -> Result<(KvOrchestrator, KvHandle), StartupError> {
        KvOrchestrator::start_with(s, pool, true)
    }

    /// The rank (or stage) shards of a logical block, in order, and its format.
    fn shards_of(
        device: CopyDevice,
        others: Vec<KvShard>,
        pool: &BlockPool,
        pipeline: bool,
    ) -> Result<(Vec<KvShard>, KvFormat), StartupError> {
        let layout = pool.layout();
        let shard_bytes = layout.block_bytes();
        let own = KvShard {
            device,
            addresses: BlockAddresses::of(pool),
        };
        if pipeline {
            let per_layer = (shard_bytes / u64::from(layout.num_layers.max(1))).max(1);
            let mut layers = layout.num_layers;
            let mut shards = Vec::with_capacity(1 + others.len());
            for (stage, shard) in others.into_iter().enumerate() {
                let (blocks, bytes) = (shard.addresses.blocks(), shard.addresses.block_bytes());
                if blocks != pool.total_blocks() || bytes == 0 || bytes % per_layer != 0 {
                    return Err(StartupError::new(format!(
                        "KV stage {stage}: its pool has {blocks} blocks of {bytes} bytes; the \
                         last stage's has {} blocks of {per_layer} bytes per layer, and every \
                         stage needs the same block count",
                        pool.total_blocks()
                    )));
                }
                layers += (bytes / per_layer) as u32;
                shards.push(shard);
            }
            shards.push(own);
            let format = kv_format(KvLayout {
                num_layers: layers,
                ..layout
            });
            return Ok((shards, format));
        }
        let mut shards = Vec::with_capacity(1 + others.len());
        shards.push(own);
        for (i, shard) in others.into_iter().enumerate() {
            let (blocks, bytes) = (shard.addresses.blocks(), shard.addresses.block_bytes());
            if blocks != pool.total_blocks() || bytes != shard_bytes {
                return Err(StartupError::new(format!(
                    "KV rank {}: its pool has {blocks} blocks of {bytes} bytes, rank 0's {} of \
                     {shard_bytes}; every tensor-parallel rank needs the same pool",
                    i + 1,
                    pool.total_blocks()
                )));
            }
            shards.push(shard);
        }
        let world = shards.len() as u32;
        Ok((shards, tp_kv_format(layout, world)))
    }

    fn start_with(
        s: KvStart<'_>,
        pool: &mut BlockPool,
        pipeline: bool,
    ) -> Result<(KvOrchestrator, KvHandle), StartupError> {
        let layout = pool.layout();
        let (shards, format) = KvOrchestrator::shards_of(s.device, s.shards, pool, pipeline)?;
        let world = shards.len() as u32;
        let sizes: Vec<u64> = shards.iter().map(|sh| sh.addresses.block_bytes()).collect();
        // `static` rank mode: this process holds rank 0's shard only, in the group's format.
        let mut remote = s.remote;
        let format = match &remote {
            Some(r) => tp_kv_format(layout, r.world()),
            None => format,
        };
        let format = with_scales(format, s.kv_scales)?;
        check_tier_formats(s.cfg, &format, world > 1 || remote.is_some())?;
        let host_codec = HostCodec::of(&s.identity, &format);
        if world > 1
            && s.cfg.cpu.enabled
            && s.cfg.cpu.max_bytes.0 / u64::from(world) < L1_SLAB_BYTES
        {
            // Each rank pins its share in whole slabs: a share below one slab holds none.
            tracing::warn!(
                event = "kv_l1_share_below_slab",
                tier = "l1",
                ranks = world,
                max_bytes = s.cfg.cpu.max_bytes.0,
                slab_bytes = L1_SLAB_BYTES,
                "kv.cpu.max_bytes / tensor_parallel_size is below one L1 slab per rank; L1 holds \
                 nothing (raise kv.cpu.max_bytes to at least tensor_parallel_size GiB)"
            );
        }
        // One logical block: every rank's (or stage's) shard of it this process copies, in
        // order (`format.block_bytes()` but in `static` mode, whose other ranks copy their own).
        let block_bytes: u64 = sizes.iter().sum();
        let l1 = if !s.cfg.cpu.enabled {
            None
        } else if let Some(pinned) = shards
            .iter()
            .map(|sh| match &sh.device {
                CopyDevice::Stream { pinned, .. } => Some(Arc::clone(pinned)),
                CopyDevice::Sync { .. } => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            // Each rank (stage) pins its share of `kv.cpu.max_bytes` through its own context,
            // in proportion to its shard of a block (even shares under tensor parallelism).
            let total: u64 = sizes.iter().sum::<u64>().max(1);
            let tiers = pinned
                .into_iter()
                .zip(&sizes)
                .map(|(p, &bytes)| {
                    Arc::new(L1PinnedTier::new(
                        L1Config {
                            enabled: true,
                            max_bytes: (u128::from(s.cfg.cpu.max_bytes.0) * u128::from(bytes)
                                / u128::from(total)) as u64,
                            slab_bytes: L1_SLAB_BYTES,
                            block_bytes: bytes,
                            memory_kind: s.memory_kind,
                        },
                        p,
                        Arc::clone(&s.clock),
                    ))
                })
                .collect();
            Some(Arc::new(ShardedL1Tier::with_sizes(tiers, sizes.clone())))
        } else {
            tracing::warn!(
                event = "kv_l1_disabled_no_pinned_memory",
                tier = "l1",
                "no pinned-memory API on this backend; L1 disabled"
            );
            None
        };
        let l1 = l1.filter(|t| t.enabled());
        // `static` mode: a tier any worker lacks is off for the group.
        let has = |tier| remote.as_ref().is_none_or(|r| r.workers_have(tier));
        let l1 = l1.filter(|_| has(TierId::L1));
        let l2 = s.l2.filter(|_| has(TierId::L2));
        let l2_dyn = l2.clone().map(|t| t as Arc<dyn KvTier>);
        let (l1_seen, l2_seen) = match &mut remote {
            None => (l1.clone().map(|t| t as Arc<dyn KvTier>), l2_dyn.clone()),
            Some(r) => {
                r.bind(l1.clone(), l2_dyn.clone());
                (
                    l1.clone().map(|t| r.mirror(t as Arc<dyn KvTier>)),
                    l2_dyn.clone().map(|t| r.mirror(t)),
                )
            }
        };
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
            l1_seen,
            l2_seen,
            Arc::clone(&s.clock),
            s.metrics,
        );
        let io_threads = s.cfg.nvme.io_threads.max(1) as usize;
        // Every I/O job belongs to an in-flight ticket, so this bound is never reached; the
        // L2 tier bounds concurrent I/O itself (`kv.nvme.max_queue_depth`).
        let io_capacity = (s.cfg.transfer.max_inflight_bytes.0 / block_bytes.max(1)) as usize + 1;
        let mut backend = CopyStreamBackend::new(
            shards,
            l1.clone(),
            l2_dyn,
            IoPoolBackend::new(io_threads, io_capacity),
            block_bytes as usize,
        );
        backend.set_host_codec(host_codec);
        let (tx, commands) = mpsc::channel(s.cfg.prefetch.max_queue.max(1) as usize);
        let mut o = KvOrchestrator {
            h,
            backend,
            clock: s.clock,
            commands,
            hits: HitWindow::default(),
            prefill_tps: None,
            last_housekeeping: None,
            remote,
        };
        o.calibrate(
            pool,
            s.cfg.cpu.enabled && l1.is_some(),
            l2.as_deref(),
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
        if world > 1 {
            tracing::info!(
                event = "kv_tier_shards",
                shards = world,
                shard_bytes = ?sizes,
                block_bytes,
                pipeline,
                "KV tier copies are stored per tensor-parallel rank (or pipeline stage) shard"
            );
        }
        Ok((o, KvHandle { tx }))
    }

    /// `reliability.pressure.deescalate_dwell`: the compression ladder's step-up dwell (P6b S-6).
    pub fn set_ladder_dwell(&mut self, dwell: Duration) {
        self.h.set_ladder_dwell(dwell);
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
    pub fn attach(&mut self, pool: &mut BlockPool, req: &AttachRequest<'_>) -> AttachOutcome {
        let outcome = self.h.attach_prefix(pool, req);
        if let AttachOutcome::Ready(a) = &outcome {
            self.record_hit(req.prompt.len(), a);
        }
        outcome
    }

    /// Transfer completions: requests whose promotions all landed, with their prefixes.
    pub fn poll(&mut self, pool: &mut BlockPool) -> Vec<(RequestId, PrefixAttach)> {
        match &mut self.remote {
            None => self.h.poll(pool, &mut self.backend),
            Some(r) => {
                let ready = self.h.poll(pool, &mut r.backend(&mut self.backend));
                // The copies this pump started, before the turn's step plan.
                r.flush();
                ready
            }
        }
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
    ///
    /// Both scans run at most every [`HOUSEKEEPING_INTERVAL`], and capacity demotion copies only
    /// blocks with reuse evidence, at most `CAPACITY_BATCH` per run.
    pub fn before_plan(&mut self, pool: &mut BlockPool, state: PressureState) {
        self.h.set_l0_state(state);
        let now = self.clock.now_mono();
        if self
            .last_housekeeping
            .is_some_and(|t| now.saturating_sub(t) < HOUSEKEEPING_INTERVAL)
        {
            return;
        }
        self.last_housekeeping = Some(now);
        if pool.free_blocks() < pool.total_blocks().div_ceil(REFRESH_FREE_SHARE) {
            self.h.refresh_reclaim_order(pool);
        }
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
        if let Some(r) = &mut self.remote {
            r.flush();
        }
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
        // Logical blocks (every rank shard): what the hierarchy moves and counts.
        let bb = self.backend.block_bytes as u64;
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

    /// L0 → L1 and back through the copy streams (every rank shard on its own), `blocks`
    /// blocks each way (at most what L0 has free).
    fn calibrate_l1(
        &mut self,
        pool: &mut BlockPool,
        blocks: u64,
    ) -> Result<[(TransferPath, PathCost); 2], CalibrationError> {
        if !self.backend.all_streams() {
            return Err(CalibrationError::Other("no copy stream".into()));
        }
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
            let backend = &self.backend;
            let timed = |to_l1: bool| -> Result<PathCost, CalibrationError> {
                let started = Instant::now();
                let mut tickets = Copies::new();
                for (b, slots) in ids.iter().zip(&slots) {
                    match backend.l1_copies(u64::from(b.0), slots, !to_l1) {
                        Ok(t) => tickets.extend(t),
                        Err((_, e)) => {
                            backend.wait_all(&tickets);
                            return Err(CalibrationError::Other(e.to_string()));
                        }
                    }
                }
                for (s, t) in &tickets {
                    let engine = backend.engine(*s).expect("every shard has a copy stream");
                    if let Err(e) = engine.wait(t) {
                        backend.wait_all(&tickets);
                        return Err(CalibrationError::Other(e.to_string()));
                    }
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
/// moves): the copy backend has no pool access while a copy runs. Under tensor parallelism a
/// worker rank computes its own with [`BlockAddresses::of`] on its pool and hands them to the
/// leader ([`KvShard`]); they stay valid while that pool lives.
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

    /// Blocks of the pool.
    pub fn blocks(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// Bytes of one block (the first block's segments; every block has the same layout).
    pub fn block_bytes(&self) -> u64 {
        self.blocks
            .first()
            .map_or(0, |s| s.iter().map(|&(_, len)| len as u64).sum())
    }

    fn segments(&self, b: BlockId) -> &[(DevicePtr, usize)] {
        self.blocks.get(b.0 as usize).map_or(&[], |s| s.as_slice())
    }
}

/// A host staging buffer of one rank shard of a block: page-locked with a copy stream, heap
/// memory otherwise.
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

    fn len(&self) -> usize {
        match self {
            HostBuf::Pinned(b) => b.len(),
            HostBuf::Heap(v) => v.len(),
        }
    }
}

/// One staging buffer per rank shard, in rank order.
type Staged = SmallVec<[HostBuf; 2]>;

/// One file-I/O job of the pool.
enum IoOp {
    /// L1 ↔ L2: read the block from one tier and store it in the other.
    Move {
        from: Arc<dyn KvTier>,
        to: Arc<dyn KvTier>,
        key: KvKey,
        codec: HostTranscode,
    },
    /// The staged shards of an L0 → L2 copy, stored as one block (the shards concatenated),
    /// encoded in the tier's format.
    Write {
        to: Arc<dyn KvTier>,
        key: KvKey,
        bufs: Staged,
        codec: HostTranscode,
    },
    /// The first half of an L2 → L0 copy: the block decoded to the L0 format and split into
    /// the shards' staging buffers.
    Read {
        from: Arc<dyn KvTier>,
        key: KvKey,
        bufs: Staged,
        codec: HostTranscode,
    },
}

/// The L0 layout and codec parameters the host path encodes and decodes tier copies with (P6b
/// S-1: the reference `encode_cpu` / `decode_cpu` on the I/O threads, until the v2.11 GPU
/// transcode). Only for one logical block of one shard: startup refuses a lower-tier format
/// other than `l0` under tensor or pipeline parallelism.
#[derive(Clone, Debug)]
pub struct HostCodec {
    pub layout: KvLayout,
    pub params: CodecParams,
}

impl HostCodec {
    /// The layout of `format` (one shard) with the tier codecs' rotation seed: the first 8
    /// bytes of the unsalted namespace key, little-endian, as L2's slab header records it.
    /// FP8 L0 scales are not needed: the host path moves FP8 pages to `fp8_e4m3` unchanged and
    /// the TurboQuant codecs decode in the L0 page's own scaled domain.
    pub fn of(identity: &ModelIdentity, format: &KvFormat) -> HostCodec {
        let ns = namespace_key(identity, format, "");
        let mut seed = [0u8; 8];
        seed.copy_from_slice(&ns.0[..8]);
        HostCodec {
            layout: format.layout,
            params: CodecParams {
                seed: u64::from_le_bytes(seed),
                ..CodecParams::default()
            },
        }
    }
}

/// One copy's formats with the host codec that converts between them.
#[derive(Clone)]
struct HostTranscode {
    codec: TransferCodec,
    host: Option<Arc<HostCodec>>,
}

impl HostTranscode {
    fn codec(name: &str) -> Result<&'static dyn KvCodec, TierError> {
        turbine_kv::codec::registry()
            .get(name)
            .ok_or_else(|| TierError::Io(format!("no kv_format codec `{name}`")))
    }

    fn host(&self) -> Result<&HostCodec, TierError> {
        self.host.as_deref().ok_or_else(|| {
            TierError::Io(format!(
                "no host codec for a {} → {} copy",
                self.codec.from, self.codec.to
            ))
        })
    }

    /// Encodes an L0-format block into the destination format (borrowed when that is `l0`).
    fn encode<'a>(&self, l0: &'a [u8]) -> Result<std::borrow::Cow<'a, [u8]>, TierError> {
        if self.codec.to == L0_FORMAT {
            return Ok(std::borrow::Cow::Borrowed(l0));
        }
        let host = self.host()?;
        let c = Self::codec(self.codec.to)?;
        let mut out = vec![0u8; c.bytes_per_block(&host.layout) as usize];
        c.encode_cpu(l0, &host.layout, &mut out, &host.params)
            .map_err(|e| TierError::Io(e.to_string()))?;
        Ok(std::borrow::Cow::Owned(out))
    }

    /// Decodes a source copy into an L0-format block (`out`, the L0 block's bytes).
    fn decode(&self, src: &[u8], out: &mut [u8]) -> Result<(), TierError> {
        if self.codec.from == L0_FORMAT {
            if src.len() != out.len() {
                return Err(TierError::Io(format!(
                    "block is {} bytes, buffer {}",
                    src.len(),
                    out.len()
                )));
            }
            out.copy_from_slice(src);
            return Ok(());
        }
        let host = self.host()?;
        Self::codec(self.codec.from)?
            .decode_cpu(src, &host.layout, out, &host.params)
            .map_err(|e| TierError::Io(e.to_string()))
    }

    /// Stores `l0` (an L0-format block) under `key` in `to`, in the destination format.
    fn store(&self, to: &dyn KvTier, key: KvKey, l0: &[u8]) -> Result<TierSlot, TierError> {
        let bytes = self.encode(l0)?;
        to.put_as(
            key,
            self.codec.to,
            bytes.len() as u64,
            TierBlockRef::Host(&bytes),
        )
    }

    /// Reads the source copy of `key` from `from` (`codec.from_bytes`).
    fn load(&self, from: &dyn KvTier, key: &KvKey) -> Result<Vec<u8>, TierError> {
        let mut v = vec![0u8; self.codec.from_bytes as usize];
        from.get(key, TierBlockMut::Host(&mut v))?;
        Ok(v)
    }
}

struct IoDone {
    ticket: u64,
    result: Result<TierSlot, TierError>,
    bufs: Staged,
}

fn run_io(op: IoOp) -> (Result<TierSlot, TierError>, Staged) {
    match op {
        IoOp::Move {
            from,
            to,
            key,
            codec,
        } => {
            let r = codec.load(from.as_ref(), &key).and_then(|v| {
                if codec.codec.is_identity() {
                    to.put_as(key, codec.codec.to, v.len() as u64, TierBlockRef::Host(&v))
                } else {
                    let host = codec.host()?;
                    let mut l0 = vec![0u8; host.layout.block_bytes() as usize];
                    codec.decode(&v, &mut l0)?;
                    codec.store(to.as_ref(), key, &l0)
                }
            });
            (r, Staged::new())
        }
        IoOp::Write {
            to,
            key,
            bufs,
            codec,
        } => {
            let r = match bufs.as_slice() {
                [one] => one.with(|b| codec.store(to.as_ref(), key, b)),
                shards => {
                    let mut v = Vec::with_capacity(shards.iter().map(HostBuf::len).sum());
                    for b in shards {
                        b.with(|b| v.extend_from_slice(b));
                    }
                    codec.store(to.as_ref(), key, &v)
                }
            };
            (r, bufs)
        }
        IoOp::Read {
            from,
            key,
            mut bufs,
            codec,
        } => {
            let r = match bufs.as_mut_slice() {
                [one] if codec.codec.from == L0_FORMAT => {
                    one.with_mut(|b| from.get(&key, TierBlockMut::Host(b)))
                }
                [one] => codec
                    .load(from.as_ref(), &key)
                    .and_then(|v| one.with_mut(|b| codec.decode(&v, b))),
                shards => {
                    let mut v = vec![0u8; shards.iter().map(HostBuf::len).sum()];
                    let read = if codec.codec.from == L0_FORMAT {
                        from.get(&key, TierBlockMut::Host(&mut v))
                    } else {
                        codec
                            .load(from.as_ref(), &key)
                            .and_then(|src| codec.decode(&src, &mut v))
                    };
                    read.map(|()| {
                        let mut at = 0usize;
                        for b in shards.iter_mut() {
                            b.with_mut(|b| {
                                b.copy_from_slice(&v[at..at + b.len()]);
                                at += b.len();
                            });
                        }
                    })
                }
            };
            (r.map(|()| TierSlot(0)), bufs)
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
    /// Encodes and decodes tier copies not stored at the L0 format (P6b S-1).
    host_codec: Option<Arc<HostCodec>>,
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
                            let (result, bufs) = run_io(op);
                            if tx
                                .send(IoDone {
                                    ticket,
                                    result,
                                    bufs,
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
            host_codec: None,
        }
    }

    fn transcode(&self, codec: TransferCodec) -> HostTranscode {
        HostTranscode {
            codec,
            host: self.host_codec.clone(),
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
    /// An L1 ↔ L2 copy, or a ladder rewrite (P6b S-6: the copy re-encoded by the host codec
    /// and stored back in its own tier) of an L1 or L2 copy.
    fn start(&mut self, t: &TransferTicket) -> Result<(), TierError> {
        let from = t.req.path.from();
        let to = if t.req.purpose == TransferPurpose::Compress {
            from
        } else {
            t.req.path.to()
        };
        let (Some(from), Some(to)) = (self.tier(from), self.tier(to)) else {
            return Err(TierError::Missing);
        };
        self.submit(
            t.id,
            IoOp::Move {
                from,
                to,
                key: t.req.key,
                codec: self.transcode(t.req.codec),
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

/// Stream copies in flight, each on the copy engine of rank shard `.0`.
type Copies = Vec<(usize, CopyTicket)>;

/// What finishes a ticket once its stream copies completed.
enum AfterCopies {
    /// L0 → L1: the reserved slots become visible.
    CommitL1(KvKey),
    /// → L0: done; the staging buffers (if any) go back to their shards.
    IntoL0 { block: u64, bufs: Staged },
    /// L0 → L2: the staged shards are written by the I/O pool.
    WriteL2 { key: KvKey, bufs: Staged },
}

enum IoStage {
    /// The I/O result completes the ticket.
    Final,
    /// L2 → L0: the shards read into the staging buffers are copied into L0 block `block`.
    ThenIntoL0 { block: u64 },
}

enum Job {
    Copies { copies: Copies, then: AfterCopies },
    Io(IoStage),
    Done(Result<TierSlot, TierError>),
}

/// One rank's end of the copy backend: its device, its pool's block addresses and its staging
/// buffers (pinned through its own context with a copy stream).
struct Shard {
    device: CopyDevice,
    addresses: BlockAddresses,
    /// Bytes of this shard of a block.
    bytes: usize,
    staging: Vec<HostBuf>,
}

/// Moves block bytes for the hierarchy's transfer engine (P4 S-6): L0 ↔ L1 on the device's
/// copy stream straight into reserved L1 slots, L0 ↔ L2 through a one-block staging buffer
/// (pinned with a copy stream, heap memory with synchronous copies) plus the I/O pool, L1 ↔ L2
/// on the [`IoPoolBackend`]. Copy errors abort the L1 reservation and count toward L1's
/// degraded window; the hierarchy then recomputes.
///
/// Under tensor parallelism (Phase 5, decision "P5 T17" B) a logical block is every rank's
/// shard of it, in rank order: each shard is copied on its own rank's copy stream (or with its
/// own `DeviceMemory`), into its own L1 tier or staging buffer, and a ticket completes only when
/// every shard's copies have; a copy error on any shard aborts every shard's L1 reservation.
/// L2 stores the shards concatenated as one block.
pub struct CopyStreamBackend {
    shards: Vec<Shard>,
    l1: Option<Arc<ShardedL1Tier>>,
    l2: Option<Arc<dyn KvTier>>,
    io: IoPoolBackend,
    /// Bytes of one logical block (every shard).
    block_bytes: usize,
    jobs: HashMap<u64, Job>,
}

impl CopyStreamBackend {
    /// `shards` in rank (or stage) order; `block_bytes` is one logical block, the shards'
    /// bytes together. Each shard's part is its pool's block size (the shards of a pipeline's
    /// stages differ, P5 S-10); a shard without blocks takes an even share.
    pub fn new(
        shards: Vec<KvShard>,
        l1: Option<Arc<ShardedL1Tier>>,
        l2: Option<Arc<dyn KvTier>>,
        mut io: IoPoolBackend,
        block_bytes: usize,
    ) -> CopyStreamBackend {
        io.l1 = l1.clone().map(|t| t as Arc<dyn KvTier>);
        io.l2 = l2.clone();
        let even = block_bytes / shards.len().max(1);
        CopyStreamBackend {
            shards: shards
                .into_iter()
                .map(|s| Shard {
                    bytes: match s.addresses.block_bytes() {
                        0 => even,
                        b => b as usize,
                    },
                    device: s.device,
                    addresses: s.addresses,
                    staging: Vec::new(),
                })
                .collect(),
            l1,
            l2,
            io,
            block_bytes,
            jobs: HashMap::new(),
        }
    }

    /// The host codec that encodes and decodes lower-tier copies not stored at the L0 format
    /// (P6b S-1); without one such copies fail (and the hierarchy recomputes).
    pub fn set_host_codec(&mut self, codec: HostCodec) {
        self.io.host_codec = Some(Arc::new(codec));
    }

    /// Every shard has a copy stream (L1 needs one on every rank).
    fn all_streams(&self) -> bool {
        self.shards
            .iter()
            .all(|s| matches!(s.device, CopyDevice::Stream { .. }))
    }

    fn engine(&self, shard: usize) -> Option<&Arc<dyn CopyEngine>> {
        match &self.shards.get(shard)?.device {
            CopyDevice::Stream { engine, .. } => Some(engine),
            CopyDevice::Sync { .. } => None,
        }
    }

    /// Waits for every copy of `copies` (nothing may still write a buffer that is released).
    fn wait_all(&self, copies: &Copies) {
        for (s, c) in copies {
            if let Some(engine) = self.engine(*s) {
                let _ = engine.wait(c);
            }
        }
    }

    /// One staging buffer per shard, from its free list or newly allocated on its device.
    fn staging_bufs(&mut self) -> Result<Staged, TierError> {
        let mut bufs = Staged::new();
        for i in 0..self.shards.len() {
            let shard = &mut self.shards[i];
            let buf = match shard.staging.pop() {
                Some(b) => Ok(b),
                None => match &shard.device {
                    CopyDevice::Stream { pinned, .. } => pinned
                        .alloc_pinned(shard.bytes)
                        .map(HostBuf::Pinned)
                        .map_err(|e| TierError::Io(format!("staging buffer: {e}"))),
                    CopyDevice::Sync { .. } => Ok(HostBuf::Heap(vec![0; shard.bytes])),
                },
            };
            match buf {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    self.release_staging(bufs);
                    return Err(e);
                }
            }
        }
        Ok(bufs)
    }

    /// Returns staging buffers (in rank order, as [`staging_bufs`](Self::staging_bufs) gave
    /// them) to their shards.
    fn release_staging(&mut self, bufs: Staged) {
        for (shard, buf) in self.shards.iter_mut().zip(bufs) {
            shard.staging.push(buf);
        }
    }

    /// Enqueues the per-layer copies of shard `shard` of L0 block `block` into (`to_device`
    /// false) or out of pinned buffer `buffer_id` starting at byte `offset`.
    fn stream_copies(
        &self,
        shard: usize,
        engine: &Arc<dyn CopyEngine>,
        block: u64,
        buffer_id: u64,
        offset: usize,
        to_device: bool,
    ) -> Result<Vec<CopyTicket>, TierError> {
        let mut tickets = Vec::new();
        let mut acc = 0usize;
        for &(ptr, len) in self.shards[shard].addresses.segments(BlockId(block as u32)) {
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

    /// Synchronous copy of shard `shard` of L0 block `block` into (`to_device` false) or from
    /// `buf`.
    fn sync_copy(
        &self,
        shard: usize,
        mem: &Arc<dyn DeviceMemory>,
        block: u64,
        buf: &mut HostBuf,
        to_device: bool,
    ) -> Result<(), TierError> {
        let segments = self.shards[shard].addresses.segments(BlockId(block as u32));
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

    /// Every shard of L0 block `block` into (`to_device` false) or out of the L1 slots `slots`
    /// (one per shard) on the shards' copy streams. On an error (with the failing shard) no
    /// copy is left in flight.
    fn l1_copies(
        &self,
        block: u64,
        slots: &ShardSlots,
        to_device: bool,
    ) -> Result<Copies, (usize, TierError)> {
        let mut copies = Copies::new();
        for (s, &(buffer_id, offset)) in slots.iter().enumerate() {
            let Some(engine) = self.engine(s) else {
                self.wait_all(&copies);
                return Err((s, TierError::Io("L1 needs a copy stream".into())));
            };
            match self.stream_copies(s, engine, block, buffer_id, offset, to_device) {
                Ok(t) => copies.extend(t.into_iter().map(|t| (s, t))),
                Err(e) => {
                    self.wait_all(&copies);
                    return Err((s, e));
                }
            }
        }
        Ok(copies)
    }

    /// Every shard of L0 block `block` into (`to_device` false) or out of its staging buffer
    /// in `bufs`: copy-stream shards enqueue their copies (returned), synchronous shards copy
    /// before this returns. On an error no copy is left in flight.
    fn staged_copies(
        &self,
        block: u64,
        bufs: &mut Staged,
        to_device: bool,
    ) -> Result<Copies, TierError> {
        let mut copies = Copies::new();
        for (s, buf) in bufs.iter_mut().enumerate() {
            let r = match &self.shards[s].device {
                CopyDevice::Stream { engine, .. } => {
                    let HostBuf::Pinned(p) = buf else {
                        unreachable!("a copy stream stages in pinned memory")
                    };
                    self.stream_copies(s, engine, block, p.id(), 0, to_device)
                        .map(|t| copies.extend(t.into_iter().map(|t| (s, t))))
                }
                CopyDevice::Sync { mem } => self.sync_copy(s, mem, block, buf, to_device),
            };
            if let Err(e) = r {
                self.wait_all(&copies);
                return Err(e);
            }
        }
        Ok(copies)
    }

    fn start_job(&mut self, t: &TransferTicket) -> Result<Job, TierError> {
        let req = &t.req;
        if req.purpose == TransferPurpose::Compress {
            // A ladder rewrite runs on the I/O pool with the host codec (the v2.11 device
            // transcode through staging replaces it with P6b Task 5); the ladder is refused at
            // startup until then.
            self.io.start(t)?;
            return Ok(Job::Io(IoStage::Final));
        }
        match req.path {
            TransferPath::L1ToL2 | TransferPath::L2ToL1 => {
                self.io.start(t)?;
                Ok(Job::Io(IoStage::Final))
            }
            TransferPath::L0ToL1 | TransferPath::L1ToL0 if !req.codec.is_identity() => {
                // Copy-stream L1 slots hold what the device pages hold: another format needs the
                // v2.11 GPU transcode (P6b Task 5), which startup requires first.
                Err(TierError::Io(format!(
                    "the L1 copy stream moves L0-format blocks only, not {} → {}",
                    req.codec.from, req.codec.to
                )))
            }
            TransferPath::L0ToL1 => {
                if !self.all_streams() {
                    return Err(TierError::Io("L1 needs a copy stream".into()));
                }
                let l1 = self.l1.clone().ok_or(TierError::Missing)?;
                let slots = l1.reserve(req.key)?;
                match self.l1_copies(req.src_slot, &slots, false) {
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::CommitL1(req.key),
                    }),
                    Err((shard, e)) => {
                        l1.abort_reservation(&req.key);
                        l1.record_copy_error(shard);
                        Err(e)
                    }
                }
            }
            TransferPath::L1ToL0 => {
                if !self.all_streams() {
                    return Err(TierError::Io("L1 needs a copy stream".into()));
                }
                let l1 = self.l1.clone().ok_or(TierError::Missing)?;
                let slots = l1.locate(&req.key).ok_or(TierError::Missing)?;
                match self.l1_copies(req.dst_slot, &slots, true) {
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::IntoL0 {
                            block: req.dst_slot,
                            bufs: Staged::new(),
                        },
                    }),
                    Err((shard, e)) => {
                        l1.record_copy_error(shard);
                        Err(e)
                    }
                }
            }
            TransferPath::L0ToL2 => {
                if self.l2.is_none() {
                    return Err(TierError::Missing);
                }
                let mut bufs = self.staging_bufs()?;
                match self.staged_copies(req.src_slot, &mut bufs, false) {
                    // Synchronous copies only: the block is staged already.
                    Ok(copies) if copies.is_empty() => {
                        self.finish_copies(t, AfterCopies::WriteL2 { key: req.key, bufs })
                    }
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::WriteL2 { key: req.key, bufs },
                    }),
                    Err(e) => {
                        self.release_staging(bufs);
                        Err(e)
                    }
                }
            }
            TransferPath::L2ToL0 => {
                let l2 = self.l2.clone().ok_or(TierError::Missing)?;
                let bufs = self.staging_bufs()?;
                let codec = self.io.transcode(req.codec);
                self.io.submit(
                    t.id,
                    IoOp::Read {
                        from: l2,
                        key: req.key,
                        bufs,
                        codec,
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
            AfterCopies::IntoL0 { block, bufs } => {
                self.release_staging(bufs);
                Ok(Job::Done(Ok(TierSlot(block))))
            }
            AfterCopies::WriteL2 { key, bufs } => {
                let l2 = self.l2.clone().ok_or(TierError::Missing)?;
                let codec = self.io.transcode(t.req.codec);
                self.io.submit(
                    t.id,
                    IoOp::Write {
                        to: l2,
                        key,
                        bufs,
                        codec,
                    },
                )?;
                Ok(Job::Io(IoStage::Final))
            }
        }
    }

    fn finish_io(&mut self, stage: IoStage, done: IoDone) -> Result<Job, TierError> {
        let IoDone { result, bufs, .. } = done;
        match stage {
            IoStage::Final => {
                self.release_staging(bufs);
                Ok(Job::Done(result))
            }
            IoStage::ThenIntoL0 { block } => {
                let mut bufs = bufs;
                if bufs.is_empty() {
                    return Err(TierError::Missing);
                }
                if let Err(e) = result {
                    self.release_staging(bufs);
                    return Err(e);
                }
                match self.staged_copies(block, &mut bufs, true) {
                    Ok(copies) if copies.is_empty() => {
                        self.release_staging(bufs);
                        Ok(Job::Done(Ok(TierSlot(block))))
                    }
                    Ok(copies) => Ok(Job::Copies {
                        copies,
                        then: AfterCopies::IntoL0 { block, bufs },
                    }),
                    Err(e) => {
                        self.release_staging(bufs);
                        Err(e)
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
                let mut all = true;
                for (s, c) in &copies {
                    let engine = self.engine(*s).expect("stream copies need a copy stream");
                    match engine.poll(c) {
                        Ok(true) => {}
                        Ok(false) => all = false,
                        Err(e) => {
                            self.wait_all(&copies);
                            self.copy_failed(t, then, *s);
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

    /// A copy of rank `shard` failed: every shard's reservation is aborted and the staging
    /// buffers go back.
    fn copy_failed(&mut self, t: &TransferTicket, then: AfterCopies, shard: usize) {
        match then {
            AfterCopies::CommitL1(key) => {
                if let Some(l1) = &self.l1 {
                    l1.abort_reservation(&key);
                    l1.record_copy_error(shard);
                }
            }
            AfterCopies::IntoL0 { bufs, .. } => {
                if t.req.path == TransferPath::L1ToL0
                    && let Some(l1) = &self.l1
                {
                    l1.record_copy_error(shard);
                }
                self.release_staging(bufs);
            }
            AfterCopies::WriteL2 { bufs, .. } => self.release_staging(bufs),
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use turbine_core::clock::SystemClock;
    use turbine_core::config::{ByteSize, ModuleName};
    use turbine_core::types::{DType, DeviceId};
    use turbine_kv::BlockPoolConfig;
    use turbine_kv::transfer::{TransferCodec, TransferPurpose, TransferRequest};
    use turbine_model::testing::TempDir;
    use turbine_observability::MetricsRegistry;
    use turbine_tensor::host::HostMemory;
    use turbine_tensor::{MemoryError, PinnedOwner};

    use super::*;

    const BLOCKS: u32 = 16;

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 16,
            dtype: DType::BF16,
            block_tokens: 16,
        }
    }

    fn identity() -> ModelIdentity {
        ModelIdentity::from_bytes(b"shard config", b"shard index")
    }

    type Rank = (Arc<dyn DeviceMemory>, BlockPool);

    /// One rank: its device memory and its pool.
    fn rank(device: u32) -> Rank {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(device), 1 << 30);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: layout(),
                num_blocks: BLOCKS,
            },
            Arc::clone(&mem),
        )
        .unwrap();
        (mem, pool)
    }

    /// The `kv` section with L2 under `dir` (and L1 when `l1`).
    fn kv_config(dir: &TempDir, l1: bool) -> KvConfig {
        let mut kv = KvConfig {
            block_tokens: layout().block_tokens,
            ..KvConfig::default()
        };
        kv.cpu.enabled = l1;
        // Each of the two ranks pins one slab (zero-filled lazily by the allocator).
        kv.cpu.max_bytes = ByteSize(2 * L1_SLAB_BYTES);
        kv.nvme.enabled = true;
        kv.nvme.path = dir.path().join("kv");
        kv.nvme.max_bytes = ByteSize(16 << 20);
        kv.nvme.slab_bytes = ByteSize(1 << 20);
        kv
    }

    /// The bytes rank `rank` writes into block `b` (distinct per rank and block).
    fn pattern(rank: u8, b: u32, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(13) ^ rank.wrapping_mul(101) ^ (b as u8))
            .collect()
    }

    fn write_block(r: &Rank, b: u32, bytes: &[u8]) {
        let mut at = 0;
        for (ptr, len) in r.1.block_segments(BlockId(b)) {
            r.0.copy_h2d(ptr, &bytes[at..at + len]).unwrap();
            at += len;
        }
    }

    fn read_block(r: &Rank, b: u32) -> Vec<u8> {
        let mut out = vec![0u8; r.1.layout().block_bytes() as usize];
        let mut at = 0;
        for (ptr, len) in r.1.block_segments(BlockId(b)) {
            r.0.copy_d2h(&mut out[at..at + len], ptr).unwrap();
            at += len;
        }
        out
    }

    /// Runs one copy through the backend to completion.
    fn run(
        o: &mut KvOrchestrator,
        id: u64,
        path: TransferPath,
        key: KvKey,
        src_slot: u64,
        dst_slot: u64,
    ) -> Result<TierSlot, TierError> {
        let codec = TransferCodec::l0(o.backend.block_bytes as u64);
        run_as(o, id, path, key, (src_slot, dst_slot), codec)
    }

    /// [`run`] with the copy's formats.
    fn run_as(
        o: &mut KvOrchestrator,
        id: u64,
        path: TransferPath,
        key: KvKey,
        (src_slot, dst_slot): (u64, u64),
        codec: TransferCodec,
    ) -> Result<TierSlot, TierError> {
        let t = TransferTicket {
            id,
            req: TransferRequest {
                path,
                key,
                bytes: codec.from_bytes.min(codec.to_bytes),
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot,
                dst_slot,
                codec,
            },
        };
        o.backend.start(&t)?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(slot) = o.backend.poll(&t)? {
                return Ok(slot);
            }
            assert!(Instant::now() < deadline, "copy {path:?} did not complete");
            std::thread::yield_now();
        }
    }

    struct Started {
        o: KvOrchestrator,
        l2: Arc<L2NvmeTier>,
        ranks: Vec<Rank>,
    }

    /// An orchestrator over two ranks: rank 0's pool is the leader's, rank 1 is a shard.
    fn start_two(
        dir: &TempDir,
        l1: bool,
        device: impl Fn(&Arc<dyn DeviceMemory>, u64) -> CopyDevice,
    ) -> Started {
        let kv = kv_config(dir, l1);
        let reg = MetricsRegistry::new();
        let metrics = KvMetrics::register(&reg);
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let l2 = open_l2(
            &kv,
            &tp_kv_format(layout(), 2),
            &identity(),
            Arc::clone(&clock),
            metrics.clone(),
        )
        .unwrap()
        .expect("L2 is enabled");
        let (mem0, mut pool0) = rank(0);
        let (mem1, pool1) = rank(1);
        let (o, _handle) = KvOrchestrator::start(
            KvStart {
                cfg: &kv,
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: device(&mem0, 0),
                shards: vec![KvShard {
                    device: device(&mem1, 1),
                    addresses: BlockAddresses::of(&pool1),
                }],
                l2: Some(Arc::clone(&l2)),
                clock,
                metrics,
                remote: None,
                kv_scales: None,
            },
            &mut pool0,
        )
        .expect("the sharded KV hierarchy starts");
        Started {
            o,
            l2,
            ranks: vec![(mem0, pool0), (mem1, pool1)],
        }
    }

    fn fill(ranks: &[Rank], b: u32) -> Vec<Vec<u8>> {
        ranks
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let bytes = pattern(i as u8, b, r.1.layout().block_bytes() as usize);
                write_block(r, b, &bytes);
                bytes
            })
            .collect()
    }

    fn wipe(ranks: &[Rank]) {
        for r in ranks {
            let zeros = vec![0u8; r.1.layout().block_bytes() as usize];
            for b in 0..BLOCKS {
                write_block(r, b, &zeros);
            }
        }
    }

    fn assert_block(ranks: &[Rank], b: u32, want: &[Vec<u8>]) {
        for (i, r) in ranks.iter().enumerate() {
            assert_eq!(
                read_block(r, b),
                want[i],
                "rank {i}'s shard came back into its own pool"
            );
        }
    }

    /// Phase 6a S-13 / S-16: an F8E4M3 pool's format is FP8 at half the block bytes, scoped by
    /// the model's scales; FP8 without scales or BF16 with scales is refused. Breaks if an FP8
    /// pool shares a namespace with BF16 or with other scales.
    #[test]
    fn fp8_pool_format_carries_the_scales() {
        let bf16 = kv_format(layout());
        assert_eq!(bf16.dtype, KvDtype::Bf16);
        let fp8_layout = KvLayout {
            dtype: DType::F8E4M3,
            ..layout()
        };
        let fp8 = kv_format(fp8_layout);
        assert_eq!(fp8.dtype, KvDtype::Fp8E4m3PerTensorScale);
        assert_eq!(2 * fp8.block_bytes(), bf16.block_bytes());
        let cache = KvCache::fp8_e4m3(vec![1.0, 0.5], vec![1.0, 1.0]);
        let hashes = scale_hashes(&cache).expect("FP8 has scales");
        assert_eq!(scale_hashes(&KvCache::bf16()), None);
        let scoped = with_scales(fp8, Some(hashes)).unwrap();
        let other = with_scales(
            tp_kv_format(fp8_layout, 1),
            scale_hashes(&KvCache::fp8_e4m3(vec![1.0, 1.0], vec![1.0, 1.0])),
        )
        .unwrap();
        let ns = |f: &KvFormat| namespace_key(&identity(), f, "");
        assert_ne!(ns(&scoped), ns(&other), "other scales, other namespace");
        assert_ne!(ns(&scoped), ns(&bf16));
        assert_eq!(with_scales(bf16, None).unwrap(), bf16);
        assert!(with_scales(fp8, None).is_err());
        assert!(with_scales(bf16, Some(hashes)).is_err());
    }

    #[test]
    fn tp_namespace_differs_from_tp1() {
        let one = kv_format(layout());
        assert_eq!(tp_kv_format(layout(), 1), one);
        assert_eq!(one.shards, 1);
        let two = tp_kv_format(layout(), 2);
        assert_ne!(
            namespace_key(&identity(), &two, ""),
            namespace_key(&identity(), &one, ""),
            "a tp = 2 blob is never read by a tp = 1 process"
        );
        assert_eq!(two.block_bytes(), 2 * layout().block_bytes());
    }

    /// Sync mode (the cpu backend): every rank copies its shard with its own `DeviceMemory`;
    /// L2 stores the shards concatenated and gives each back to its own pool.
    #[test]
    fn two_shards_round_trip_through_l2_sync() {
        let dir = TempDir::new("turbine-kv-shards-sync");
        let Started { mut o, l2, ranks } = start_two(&dir, false, |mem, _| CopyDevice::Sync {
            mem: Arc::clone(mem),
        });
        assert!(!o.h.l1_enabled());
        assert_eq!(
            o.backend.block_bytes as u64,
            2 * layout().block_bytes(),
            "a logical block is both shards"
        );
        let key = KvKey([3; 16]);
        let want = fill(&ranks, 5);
        run(&mut o, 1, TransferPath::L0ToL2, key, 5, 0).unwrap();
        let mut blob = vec![0u8; o.backend.block_bytes];
        l2.get(&key, TierBlockMut::Host(&mut blob)).unwrap();
        assert_eq!(blob, [want[0].as_slice(), want[1].as_slice()].concat());

        wipe(&ranks);
        run(&mut o, 2, TransferPath::L2ToL0, key, 0, 9).unwrap();
        assert_block(&ranks, 9, &want);
        let zeros = vec![0u8; want[0].len()];
        assert_block(&ranks, 5, &[zeros.clone(), zeros]);
    }

    /// P6b S-1, host path: with `kv.nvme.format: fp8_e4m3` an L0 → L2 copy stores the block
    /// encoded by the codec's `encode_cpu` (the slab's slots hold the smaller bytes) and the
    /// L2 → L0 copy decodes it, byte for byte the codec's own round trip.
    #[test]
    fn sync_l2_stores_the_tier_format() {
        let dir = TempDir::new("turbine-kv-tier-format-sync");
        let mut kv = kv_config(&dir, false);
        kv.nvme.format = ModuleName::new("fp8_e4m3").unwrap();
        let reg = MetricsRegistry::new();
        let metrics = KvMetrics::register(&reg);
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let format = kv_format(layout());
        let l2 = open_l2(
            &kv,
            &format,
            &identity(),
            Arc::clone(&clock),
            metrics.clone(),
        )
        .unwrap()
        .expect("L2 is enabled");
        let (mem, mut pool) = rank(0);
        let (mut o, _handle) = KvOrchestrator::start(
            KvStart {
                cfg: &kv,
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: CopyDevice::Sync {
                    mem: Arc::clone(&mem),
                },
                shards: Vec::new(),
                l2: Some(Arc::clone(&l2)),
                clock,
                metrics,
                remote: None,
                kv_scales: None,
            },
            &mut pool,
        )
        .expect("the KV hierarchy starts");
        let r = (mem, pool);
        let bb = layout().block_bytes() as usize;
        let block: Vec<u8> = (0..bb / 2)
            .flat_map(|i| {
                let x = ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32);
                turbine_kv::codec::f32_to_bf16(x).to_le_bytes()
            })
            .collect();
        write_block(&r, 4, &block);
        let fp8 = turbine_kv::codec::registry().get("fp8_e4m3").unwrap();
        let small = fp8.bytes_per_block(&layout());
        assert!(small < bb as u64);
        let down = TransferCodec {
            from: L0_FORMAT,
            from_bytes: bb as u64,
            to: "fp8_e4m3",
            to_bytes: small,
        };
        let key = KvKey([5; 16]);
        run_as(&mut o, 1, TransferPath::L0ToL2, key, (4, 0), down).unwrap();
        assert_eq!(l2.used_bytes(), small, "L2 holds the encoded block");
        let mut stored = vec![0u8; small as usize];
        l2.get(&key, TierBlockMut::Host(&mut stored)).unwrap();
        let params = HostCodec::of(&identity(), &format).params;
        let mut want_enc = vec![0u8; small as usize];
        fp8.encode_cpu(&block, &layout(), &mut want_enc, &params)
            .unwrap();
        assert_eq!(stored, want_enc, "encode_cpu on the I/O thread");

        let up = TransferCodec {
            from: "fp8_e4m3",
            from_bytes: small,
            to: L0_FORMAT,
            to_bytes: bb as u64,
        };
        run_as(&mut o, 2, TransferPath::L2ToL0, key, (0, 9), up).unwrap();
        let mut want = vec![0u8; bb];
        fp8.decode_cpu(&want_enc, &layout(), &mut want, &params)
            .unwrap();
        assert_eq!(read_block(&r, 9), want, "decode_cpu into the L0 block");
        assert_ne!(want, block, "fp8_e4m3 from BF16 is lossy");

        // A copy stream moves L0-format blocks only (the GPU transcode is not there yet), and
        // a sharded block refuses a lower-tier format other than l0 at startup.
        assert!(check_tier_formats(&kv, &format, false).is_ok());
        let err = check_tier_formats(&kv, &format, true).unwrap_err();
        assert!(err.to_string().contains("kv.nvme.format fp8_e4m3"), "{err}");
    }

    /// A kernel-library context stand-in: copies are done when enqueued, and a pinned buffer
    /// is only a copy target on the context that allocated it (buffer ids never overlap
    /// between contexts, so a cross-rank copy fails).
    struct FakeCtx {
        mem: Arc<dyn DeviceMemory>,
        inner: Arc<FakeInner>,
        copies: AtomicU64,
    }

    struct FakeInner {
        bufs: Mutex<HashMap<u64, Box<[u8]>>>,
        next: AtomicU64,
    }

    impl FakeCtx {
        fn new(mem: &Arc<dyn DeviceMemory>, rank: u64) -> Arc<FakeCtx> {
            Arc::new(FakeCtx {
                mem: Arc::clone(mem),
                inner: Arc::new(FakeInner {
                    bufs: Mutex::new(HashMap::new()),
                    next: AtomicU64::new(1 + rank * 1_000_000),
                }),
                copies: AtomicU64::new(0),
            })
        }
    }

    impl PinnedOwner for FakeInner {
        fn with_bytes_dyn(&self, id: u64, f: &mut dyn FnMut(&mut [u8])) {
            f(self.bufs.lock().unwrap().get_mut(&id).expect("live buffer"));
        }

        fn free_pinned(&self, id: u64) {
            self.bufs.lock().unwrap().remove(&id);
        }
    }

    impl PinnedMemory for FakeCtx {
        fn alloc_pinned(&self, bytes: usize) -> Result<PinnedBuffer, MemoryError> {
            let id = self.inner.next.fetch_add(1, Ordering::Relaxed);
            self.inner
                .bufs
                .lock()
                .unwrap()
                .insert(id, vec![0u8; bytes].into_boxed_slice());
            Ok(PinnedBuffer::new(id, bytes, Arc::clone(&self.inner) as _))
        }
    }

    impl CopyEngine for FakeCtx {
        fn copy_async(
            &self,
            dst: CopyTarget,
            src: CopyTarget,
            bytes: usize,
        ) -> Result<CopyTicket, MemoryError> {
            let mut bufs = self.inner.bufs.lock().unwrap();
            let foreign =
                |id: u64| MemoryError::InvalidArgument(format!("pinned buffer {id} is foreign"));
            match (dst, src) {
                (CopyTarget::Pinned { buffer_id, offset }, CopyTarget::Device(ptr)) => {
                    let buf = bufs.get_mut(&buffer_id).ok_or_else(|| foreign(buffer_id))?;
                    self.mem.copy_d2h(&mut buf[offset..offset + bytes], ptr)?;
                }
                (CopyTarget::Device(ptr), CopyTarget::Pinned { buffer_id, offset }) => {
                    let buf = bufs.get(&buffer_id).ok_or_else(|| foreign(buffer_id))?;
                    self.mem.copy_h2d(ptr, &buf[offset..offset + bytes])?;
                }
                _ => return Err(MemoryError::InvalidArgument("unsupported copy".into())),
            }
            let id = self.copies.fetch_add(1, Ordering::Relaxed);
            Ok(CopyTicket { id, bytes })
        }

        fn poll(&self, _t: &CopyTicket) -> Result<bool, MemoryError> {
            Ok(true)
        }

        fn wait(&self, _t: &CopyTicket) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    fn stream(mem: &Arc<dyn DeviceMemory>, rank: u64) -> CopyDevice {
        let ctx = FakeCtx::new(mem, rank);
        CopyDevice::Stream {
            engine: Arc::clone(&ctx) as _,
            pinned: ctx as _,
        }
    }

    /// Copy streams: each rank's shard goes through its own copy engine into its own L1 tier
    /// (pinned through its own context) and back; L1 -> L2 stores the concatenation, and
    /// L0 <-> L2 stages each shard in its own pinned buffer.
    #[test]
    fn two_shards_round_trip_through_l1_and_l2_streams() {
        let dir = TempDir::new("turbine-kv-shards-stream");
        let Started { mut o, l2, ranks } = start_two(&dir, true, stream);
        assert!(o.h.l1_enabled(), "L1 calibrated through both ranks");
        let l1 = o.backend.l1.clone().expect("L1");
        assert_eq!(l1.shards().len(), 2);
        let shard_bytes = layout().block_bytes() as usize;

        // L0 -> L1 -> L0.
        let key = KvKey([4; 16]);
        let want = fill(&ranks, 2);
        run(&mut o, 1, TransferPath::L0ToL1, key, 2, 0).unwrap();
        for (i, shard) in l1.shards().iter().enumerate() {
            let mut out = vec![0u8; shard_bytes];
            shard.get(&key, TierBlockMut::Host(&mut out)).unwrap();
            assert_eq!(out, want[i], "rank {i}'s L1 tier holds its shard");
        }
        wipe(&ranks);
        run(&mut o, 2, TransferPath::L1ToL0, key, 0, 7).unwrap();
        assert_block(&ranks, 7, &want);

        // L1 -> L2 (the concatenation) -> L0.
        run(&mut o, 3, TransferPath::L1ToL2, key, 0, 0).unwrap();
        let mut blob = vec![0u8; 2 * shard_bytes];
        l2.get(&key, TierBlockMut::Host(&mut blob)).unwrap();
        assert_eq!(blob, [want[0].as_slice(), want[1].as_slice()].concat());
        wipe(&ranks);
        run(&mut o, 4, TransferPath::L2ToL0, key, 0, 11).unwrap();
        assert_block(&ranks, 11, &want);

        // L0 -> L2 through per-rank pinned staging, and back.
        let key2 = KvKey([5; 16]);
        let want2 = fill(&ranks, 3);
        run(&mut o, 5, TransferPath::L0ToL2, key2, 3, 0).unwrap();
        wipe(&ranks);
        run(&mut o, 6, TransferPath::L2ToL0, key2, 0, 12).unwrap();
        assert_block(&ranks, 12, &want2);
        assert!(
            o.backend.shards.iter().all(|s| s.staging.len() == 1),
            "each rank's staging buffer went back to that rank"
        );
    }

    /// A pool of `layers` layers of the test layout on host device `device`.
    fn stage_rank(device: u32, layers: u32) -> Rank {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(device), 1 << 30);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: KvLayout {
                    num_layers: layers,
                    ..layout()
                },
                num_blocks: BLOCKS,
            },
            Arc::clone(&mem),
        )
        .unwrap();
        (mem, pool)
    }

    /// P5 S-10 (plan Task 24): a pipeline's stages hold unequal layer counts — stage 0 three
    /// layers, the last stage (the engine's pool) one — so their shards of a block differ in
    /// size. The logical block is the stages' shards in stage order (the whole model's layers:
    /// the format is one device's of 4 layers), each copied on its own stage's copy stream into
    /// its own L1 tier (sized in proportion) and back; L2 stores the concatenation and gives
    /// each shard back to its own stage; a stage pool with another block count is refused.
    /// Breaks if a shard is cut at an even split or lands in the wrong stage.
    #[test]
    fn pipeline_stages_round_trip_through_l1_and_l2() {
        let dir = TempDir::new("turbine-kv-stages");
        let mut kv = kv_config(&dir, true);
        kv.cpu.max_bytes = ByteSize(8 * L1_SLAB_BYTES);
        let whole = KvLayout {
            num_layers: 4,
            ..layout()
        };
        let reg = MetricsRegistry::new();
        let metrics = KvMetrics::register(&reg);
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let l2 = open_l2(
            &kv,
            &kv_format(whole),
            &identity(),
            Arc::clone(&clock),
            metrics.clone(),
        )
        .unwrap()
        .expect("L2 is enabled");
        let (mem0, pool0) = stage_rank(0, 3);
        let (mem1, mut pool1) = stage_rank(1, 1);
        let (mut o, _handle) = KvOrchestrator::start_pipeline(
            KvStart {
                cfg: &kv,
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: stream(&mem1, 1),
                shards: vec![KvShard {
                    device: stream(&mem0, 0),
                    addresses: BlockAddresses::of(&pool0),
                }],
                l2: Some(Arc::clone(&l2)),
                clock,
                metrics,
                remote: None,
                kv_scales: None,
            },
            &mut pool1,
        )
        .expect("the pipeline's KV hierarchy starts");
        assert_eq!(o.backend.block_bytes as u64, whole.block_bytes());
        assert!(o.h.l1_enabled(), "L1 calibrated through both stages");
        let l1 = o.backend.l1.clone().expect("L1");
        // Each stage's L1 share follows its shard's bytes: both hold about as many blocks.
        let (c0, c1) = (
            l1.shards()[0].capacity_bytes() as f64,
            l1.shards()[1].capacity_bytes() as f64,
        );
        assert!((c0 / (3.0 * c1) - 1.0).abs() < 0.01, "{c0} vs {c1}");
        let ranks = vec![(mem0, pool0), (mem1, pool1)];

        let key = KvKey([6; 16]);
        let want = fill(&ranks, 2);
        assert_eq!(want[0].len(), 3 * want[1].len());
        run(&mut o, 1, TransferPath::L0ToL1, key, 2, 0).unwrap();
        wipe(&ranks);
        run(&mut o, 2, TransferPath::L1ToL0, key, 0, 7).unwrap();
        assert_block(&ranks, 7, &want);
        run(&mut o, 3, TransferPath::L1ToL2, key, 0, 0).unwrap();
        let mut blob = vec![0u8; o.backend.block_bytes];
        l2.get(&key, TierBlockMut::Host(&mut blob)).unwrap();
        assert_eq!(blob, [want[0].as_slice(), want[1].as_slice()].concat());
        wipe(&ranks);
        run(&mut o, 4, TransferPath::L2ToL0, key, 0, 11).unwrap();
        assert_block(&ranks, 11, &want);
        let key2 = KvKey([7; 16]);
        let want2 = fill(&ranks, 3);
        run(&mut o, 5, TransferPath::L0ToL2, key2, 3, 0).unwrap();
        wipe(&ranks);
        run(&mut o, 6, TransferPath::L2ToL0, key2, 0, 12).unwrap();
        assert_block(&ranks, 12, &want2);

        // A stage pool with another block count is refused.
        let (mem2, mut last) = stage_rank(2, 1);
        let (mem3, _) = stage_rank(3, 3);
        let small = BlockPool::new(
            BlockPoolConfig {
                layout: KvLayout {
                    num_layers: 3,
                    ..layout()
                },
                num_blocks: BLOCKS / 2,
            },
            Arc::clone(&mem3),
        )
        .unwrap();
        let err = KvOrchestrator::start_pipeline(
            KvStart {
                cfg: &kv_config(&dir, false),
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: CopyDevice::Sync { mem: mem2 },
                shards: vec![KvShard {
                    device: CopyDevice::Sync { mem: mem3 },
                    addresses: BlockAddresses::of(&small),
                }],
                l2: None,
                clock: Arc::new(SystemClock::new()),
                metrics: KvMetrics::register(&MetricsRegistry::new()),
                remote: None,
                kv_scales: None,
            },
            &mut last,
        )
        .err()
        .expect("a stage pool with fewer blocks is refused");
        assert!(err.to_string().contains("KV stage 0"), "{err}");
    }

    /// A worker pool that differs from the leader's is refused at startup.
    #[test]
    fn mismatched_shard_pool_is_refused() {
        let dir = TempDir::new("turbine-kv-shards-mismatch");
        let kv = kv_config(&dir, false);
        let (mem0, mut pool0) = rank(0);
        let mem1: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(1), 1 << 30);
        let small = BlockPool::new(
            BlockPoolConfig {
                layout: layout(),
                num_blocks: BLOCKS / 2,
            },
            Arc::clone(&mem1),
        )
        .unwrap();
        let reg = MetricsRegistry::new();
        let err = KvOrchestrator::start(
            KvStart {
                cfg: &kv,
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: CopyDevice::Sync { mem: mem0 },
                shards: vec![KvShard {
                    device: CopyDevice::Sync { mem: mem1 },
                    addresses: BlockAddresses::of(&small),
                }],
                l2: None,
                clock: Arc::new(SystemClock::new()),
                metrics: KvMetrics::register(&reg),
                remote: None,
                kv_scales: None,
            },
            &mut pool0,
        )
        .err()
        .expect("a smaller worker pool is refused");
        assert!(err.to_string().contains("KV rank 1"), "{err}");
    }

    /// Runs one copy of a `static` group to completion: the leader's own copy and every
    /// worker's, over the rank link.
    fn run_static(
        o: &mut KvOrchestrator,
        id: u64,
        path: TransferPath,
        key: KvKey,
        src_slot: u64,
        dst_slot: u64,
    ) -> Result<TierSlot, TierError> {
        let t = TransferTicket {
            id,
            req: TransferRequest {
                path,
                key,
                bytes: o.backend.block_bytes as u64,
                owner: None,
                purpose: TransferPurpose::Demote,
                src_slot,
                dst_slot,
                codec: TransferCodec::l0(o.backend.block_bytes as u64),
            },
        };
        let driver = o.remote.as_mut().expect("a static group");
        driver.backend(&mut o.backend).start(&t)?;
        driver.flush();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(slot) = driver.backend(&mut o.backend).poll(&t)? {
                driver.flush();
                return Ok(slot);
            }
            assert!(Instant::now() < deadline, "copy {path:?} did not complete");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Executes nothing (the tier test's worker runs no steps).
    struct NoSteps;
    impl turbine_distributed::rank::StepExecutor for NoSteps {
        fn execute(
            &mut self,
            _plan: &turbine_distributed::rank::StepPlan,
        ) -> Result<turbine_distributed::rank::StepOutput, turbine_distributed::rank::ExecError>
        {
            Ok(turbine_distributed::rank::StepOutput {
                logits: None,
                rows: 0,
                vocab: 0,
            })
        }
    }

    /// P5 Task 30 (decision "P5: KV tiers in static rank mode" B): a tp 2 `static` group whose
    /// two ranks run on separate threads, linked only by the loopback `tcp` rank transport —
    /// each with its own pool, its own copy engine and pinned L1 (its own context) and its own
    /// L2 directory (`rank-<r>`). The leader's orchestrator drives every copy: L0 -> L1 -> L0,
    /// L1 -> L2 -> L0 and L0 -> L2 -> L0 bring each rank's shard back into its own pool bit-exact,
    /// and each rank's L2 holds only its own shard. An eviction through the tier the hierarchy
    /// sees reaches the worker: a later promotion the leader can still serve fails on the worker
    /// (and so for the group). Breaks if a copy is not mirrored on the worker, completes before
    /// the worker's copy did, crosses the ranks' bytes, or an eviction stays on the leader.
    #[test]
    fn static_workers_round_trip_through_l1_and_l2() {
        use turbine_distributed::rank::{HelloExpect, PROTOCOL_VERSION, RankMessage, RankRuntime};

        use crate::engine::tp_tiers::{self, WorkerTiers};

        let dir = TempDir::new("turbine-kv-static-tiers");
        let kv = kv_config(&dir, true);
        let (mut kv0, mut kv1) = (kv.clone(), kv);
        tp_tiers::static_rank_kv(&mut kv0, 0, 2);
        tp_tiers::static_rank_kv(&mut kv1, 1, 2);
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let transport = turbine_distributed::transport::select("tcp").expect("tcp");
        let addr = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let expect = HelloExpect {
            model_fingerprint: identity().fingerprint(),
            config_fingerprint: [1; 32],
            device_vendor: turbine_core::types::Vendor::Amd,
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
        let (mem0, mut pool0) = rank(0);
        let (mem1, pool1) = rank(1);
        // The worker rank: its own tiers over its own pool, on its own thread.
        let (device1, addresses1) = (stream(&mem1, 1), BlockAddresses::of(&pool1));
        let worker_clock = Arc::clone(&clock);
        let worker = std::thread::spawn(move || {
            let mut link =
                RankRuntime::static_worker(transport, addr, hello, Duration::from_secs(10))
                    .expect("welcome");
            let l2 = tp_tiers::open_rank_l2(
                &kv1,
                layout(),
                2,
                &identity(),
                Arc::clone(&worker_clock),
                KvMetrics::register(&MetricsRegistry::new()),
            )
            .expect("the worker's L2 opens");
            let tiers = WorkerTiers::new(
                &kv1,
                MemoryKind::Dedicated,
                device1,
                addresses1,
                l2,
                worker_clock,
            )
            .expect("the worker's tiers");
            link.tiers_ready(tiers.l1_enabled(), tiers.l2_enabled())
                .expect("tiers ready");
            link.run_with_tiers(&mut NoSteps, Some(Box::new(tiers)))
        });
        let mut rt = RankRuntime::static_leader(
            transport,
            addr,
            expect,
            2,
            Duration::from_secs(10),
            [0; 128],
            2,
        )
        .expect("the worker joins");
        let driver = tp_tiers::leader_driver(&rt, Duration::from_secs(10), Duration::from_secs(10))
            .expect("tier ready")
            .expect("a static group drives tiers");
        let reg = MetricsRegistry::new();
        let metrics = KvMetrics::register(&reg);
        let l2 = tp_tiers::open_rank_l2(
            &kv0,
            layout(),
            2,
            &identity(),
            Arc::clone(&clock),
            metrics.clone(),
        )
        .unwrap()
        .expect("L2 is enabled");
        let (mut o, _handle) = KvOrchestrator::start(
            KvStart {
                cfg: &kv0,
                memory_kind: MemoryKind::Dedicated,
                identity: identity(),
                device: stream(&mem0, 0),
                shards: Vec::new(),
                l2: Some(Arc::clone(&l2)),
                clock,
                metrics,
                remote: Some(driver),
                kv_scales: None,
            },
            &mut pool0,
        )
        .expect("the static leader's KV hierarchy starts");
        assert!(o.h.l1_enabled() && o.h.l2_enabled(), "both tiers on");
        let shard_bytes = layout().block_bytes() as usize;
        assert_eq!(
            o.backend.block_bytes, shard_bytes,
            "the leader copies its own shard"
        );
        let ranks = vec![(mem0, pool0), (mem1, pool1)];

        // L0 -> L1 -> L0.
        let key = KvKey([4; 16]);
        let want = fill(&ranks, 2);
        run_static(&mut o, 1, TransferPath::L0ToL1, key, 2, 0).unwrap();
        wipe(&ranks);
        run_static(&mut o, 2, TransferPath::L1ToL0, key, 0, 7).unwrap();
        assert_block(&ranks, 7, &want);

        // L1 -> L2 (each rank its own shard) -> L0.
        run_static(&mut o, 3, TransferPath::L1ToL2, key, 0, 0).unwrap();
        let mut blob = vec![0u8; shard_bytes];
        l2.get(&key, TierBlockMut::Host(&mut blob)).unwrap();
        assert_eq!(blob, want[0], "the leader's L2 holds its own shard only");
        wipe(&ranks);
        run_static(&mut o, 4, TransferPath::L2ToL0, key, 0, 11).unwrap();
        assert_block(&ranks, 11, &want);

        // L0 -> L2 through each rank's own staging, and back.
        let key2 = KvKey([5; 16]);
        let want2 = fill(&ranks, 3);
        run_static(&mut o, 5, TransferPath::L0ToL2, key2, 3, 0).unwrap();
        wipe(&ranks);
        run_static(&mut o, 6, TransferPath::L2ToL0, key2, 0, 12).unwrap();
        assert_block(&ranks, 12, &want2);

        // An eviction through the tier the hierarchy holds reaches the worker: with the block
        // put back on the leader alone, its promotion fails on the worker, so for the group.
        let seen = o
            .remote
            .as_ref()
            .unwrap()
            .mirror(Arc::clone(&l2) as Arc<dyn KvTier>);
        seen.evict(&key2).unwrap();
        l2.put(key2, TierBlockRef::Host(&want2[0])).unwrap();
        let err = run_static(&mut o, 7, TransferPath::L2ToL0, key2, 0, 13)
            .expect_err("the worker no longer holds the block");
        assert!(err.to_string().contains("rank 1"), "{err}");

        drop(o);
        rt.shutdown("test done");
        assert_eq!(
            worker.join().expect("worker thread"),
            Ok(()),
            "the leader's shutdown ends the worker cleanly"
        );
    }
}
