//! `KvHierarchy`: the synchronous composition of identity, directory, tiers, policy, planner,
//! transfers and sessions over the L0 `BlockPool` (P4 S-3, S-8, S-12). It runs on the engine
//! thread next to the scheduler (the `turbine-server` KV orchestrator owns it) and in the
//! scheduler's deterministic `kv_sim` tests; it uses only the other modules' public APIs
//! (TS §21 rule 6) and reads time only through the injected clock.
//!
//! Ownership: the pool owns L0 references. A request holds one reference per attached block;
//! a demotion copy out of L0 holds one reference on its source block until the copy completes,
//! so allocation can never reclaim a block while it is being read.

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use smallvec::SmallVec;
use turbine_core::clock::Clock;
use turbine_core::config::{KvConfig, KvPrefetchConfig, KvSessionConfig, LossyReuse};
use turbine_core::registry::UnknownModule;
use turbine_core::request::SessionHints;
use turbine_core::types::{
    BlockId, KvLayout, MemoryKind, ModelFingerprint, ModelIdentity, PressureState, Priority,
    RequestId,
};

use crate::codec;
use crate::directory::{
    CostEstimate, KvBlock, KvDirectory, KvPriority, Lineage, PrefixMatch, SessionId, TokenRange,
    format_is_lossy,
};
use crate::document::{
    FormatUsage, HitRate, KvDocument, KvSummary, KvTierDocument, Prefetch, Sessions, TierState,
    Transfers,
};
use crate::identity::{
    Blake3Hasher, KeyHasher, KvFormat, KvKey, NamespaceCache, lossy_key, prefix_keys,
};
use crate::metrics::{EvictReason, KvMetrics, LADDER_EVICT, LadderReason, PrefetchOutcome};
use crate::planner::{KvPlan, PlanInputs, PlanReason, plan_cost, plan_prefix, reuse_cap};
use crate::policy::{
    BlockScoreInputs, EvictAction, KvBlockSummary, LadderContext, LadderLimits, SelectedPolicy,
    make_policy, recompute_seconds,
};
use crate::pool::BlockPool;
use crate::session::{PrefetchTracker, SessionAction, SessionTable};
use crate::tier::{KvLocation, KvTier, L0_FORMAT, TierError, TierId, demotion_target};
use turbine_reliability::throttle::KvReclaimer;

use crate::transfer::{
    TransferBackend, TransferCodec, TransferEngine, TransferPath, TransferPurpose, TransferRequest,
};

/// The hierarchy's settings: the `kv` section plus the model's block size and memory kind.
#[derive(Clone, Debug)]
pub struct HierarchyConfig {
    pub block_tokens: u32,
    pub block_bytes: u64,
    /// The `kv.policy` registry module with the `kv.policy_weights`.
    pub policy: SelectedPolicy,
    pub demote_min_value: f64,
    pub prefix_sharing: bool,
    pub memory_kind: MemoryKind,
    pub max_inflight_bytes: u64,
    /// Bound of the transfer queue (promotions, demotions and prefetches together).
    pub transfer_queue: usize,
    pub session: KvSessionConfig,
    pub prefetch: KvPrefetchConfig,
    /// `kv_format` codecs L1 and L2 store blocks in (`kv.cpu.format`, `kv.nvme.format`, P6b
    /// S-2); `l0` keeps the L0 bytes unchanged.
    pub l1_format: &'static str,
    pub l2_format: &'static str,
    /// `kv.lossless_tail_blocks`: the last N full blocks of a finished sequence are demoted at
    /// the L0 format whatever the tier's format.
    pub lossless_tail_blocks: u32,
    /// `kv.recent_window_blocks`: while the L0 base format is lossy, the newest N full blocks
    /// of each live sequence hold BF16 pages (S-5); a block that leaves the window is
    /// recompressed in place into the L0 base format. 0 turns the window off.
    pub recent_window_blocks: u32,
    /// The L0 page format's name (`kv.dtype`): `l0` ranks at its rung ([`codec::tier_rung`]),
    /// so a TurboQuant L0's rung order skips the more precise codecs.
    pub l0_dtype: &'static str,
    /// `kv.ladder.l0`: L0 joins the ladder as its top tier (S-7).
    pub ladder_l0: bool,
    /// `kv.lossy_reuse: allow`: requests without `x-turbine-kv-lossy` may reuse lossy blocks
    /// (P6b S-3).
    pub allow_lossy: bool,
    /// `kv.lossy_penalty` resolved for every registered `kv_format` codec (one entry each):
    /// a lossy block's retrieval cost is multiplied by `1 + penalty`.
    pub lossy_penalty: Vec<(&'static str, f64)>,
    /// The compression ladder in L1/L2 (P6b S-6), when `kv.ladder.enabled`.
    pub ladder: Option<LadderConfig>,
}

/// The compression ladder's settings (P6b S-6): `kv.ladder.*` and the pressure controller's
/// `reliability.pressure.deescalate_dwell`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderConfig {
    /// The lossiest rung (`kv.ladder.max_format`, a registered lossy codec).
    pub max_format: &'static str,
    /// `kv.ladder.high_water`.
    pub high_water: f64,
    /// `kv.ladder.low_water`: also the fill a tier must stay below for `dwell` before its rung
    /// steps back up (at GREEN only); an upper tier (L0 included) compresses only above it.
    pub low_water: f64,
    /// How long a tier stays below `low_water` before its rung for new demotions steps back up
    /// one rung, which happens only at GREEN (`reliability.pressure.deescalate_dwell`; the
    /// server sets it).
    pub dwell: Duration,
    /// The L0 page format's name (`kv.dtype`): `l0` ranks at its rung
    /// (`codec::tier_rung`), so a TurboQuant L0's rung order skips the more precise codecs.
    pub l0_dtype: &'static str,
}

impl LadderConfig {
    /// `reliability.pressure.deescalate_dwell`'s default, until the server sets the configured
    /// one ([`KvHierarchy::set_ladder_dwell`]).
    pub const DEFAULT_DWELL: Duration = Duration::from_secs(10);

    fn limits(&self) -> LadderLimits {
        LadderLimits {
            max_format: self.max_format,
            high_water: self.high_water,
            low_water: self.low_water,
            l0_dtype: self.l0_dtype,
        }
    }
}

/// Most ladder rewrites started per ladder tick, and most in flight (P6b S-6; TS §21 rule 8).
pub const LADDER_REWRITES_PER_TICK: usize = 32;

/// Ladder ticks are at least this far apart (P6b S-6).
pub const LADDER_TICK_INTERVAL: Duration = Duration::from_millis(50);

impl HierarchyConfig {
    /// Copies waiting to start, across all paths.
    pub const TRANSFER_QUEUE: usize = 4096;

    /// `UnknownModule` when `kv.policy` names no registered eviction policy or a tier format no
    /// registered `kv_format` codec (the server validates the names before any port is bound,
    /// so only a hand-built config gets here).
    pub fn from_config(
        kv: &KvConfig,
        block_bytes: u64,
        memory_kind: MemoryKind,
    ) -> Result<Self, UnknownModule> {
        Ok(HierarchyConfig {
            block_tokens: kv.block_tokens,
            block_bytes,
            policy: make_policy(kv.policy.as_str(), &kv.policy_weights)?,
            demote_min_value: kv.demote_min_value,
            prefix_sharing: kv.prefix_sharing,
            memory_kind,
            max_inflight_bytes: kv.transfer.max_inflight_bytes.0,
            transfer_queue: Self::TRANSFER_QUEUE,
            session: kv.session.clone(),
            prefetch: kv.prefetch.clone(),
            l1_format: codec_name(kv.cpu.format.as_str())?,
            l2_format: codec_name(kv.nvme.format.as_str())?,
            lossless_tail_blocks: kv.lossless_tail_blocks,
            recent_window_blocks: window_blocks(kv),
            l0_dtype: kv.dtype.as_str(),
            ladder_l0: kv.ladder.l0,
            allow_lossy: kv.lossy_reuse == LossyReuse::Allow,
            lossy_penalty: crate::codec::registry()
                .iter()
                .map(|c| {
                    let p = kv
                        .lossy_penalty_override(c.name())
                        .unwrap_or_else(|| c.default_lossy_penalty());
                    (c.name(), p)
                })
                .collect(),
            ladder: if kv.ladder.enabled {
                Some(LadderConfig {
                    max_format: codec_name(kv.ladder.max_format.as_str())?,
                    high_water: kv.ladder.high_water,
                    low_water: kv.ladder.low_water,
                    dwell: LadderConfig::DEFAULT_DWELL,
                    l0_dtype: kv.dtype.as_str(),
                })
            } else {
                None
            },
        })
    }

    /// The planner penalty of a block served lossy in codec `format`.
    pub fn penalty_of(&self, format: &str) -> f64 {
        self.lossy_penalty
            .iter()
            .find(|(n, _)| *n == format)
            .map_or(0.0, |(_, p)| *p)
    }
}

/// `kv.recent_window_blocks` as the hierarchy holds it: only with a lossy `kv.dtype` (S-5).
fn window_blocks(kv: &KvConfig) -> u32 {
    if kv.dtype.is_lossy() {
        kv.recent_window_blocks
    } else {
        0
    }
}

/// The dir format tag of a fresh L0 block: `l0` for a base-class page, else the page's class
/// name (the recent window's BF16 class, S-5).
fn l0_tag(pool: &BlockPool, block: BlockId) -> &'static str {
    let f = pool.format_of(block);
    if f == pool.base_format() {
        crate::tier::L0_FORMAT
    } else {
        f
    }
}

/// The registered `kv_format` codec's own (`'static`) name.
fn codec_name(name: &str) -> Result<&'static str, UnknownModule> {
    let reg = crate::codec::registry();
    reg.get(name)
        .map(|c| c.name())
        .ok_or_else(|| reg.unknown(name))
}

/// Bytes of one block stored in codec `format`: `block_bytes` (every shard) at the L0 format,
/// else the codec's size of each rank shard's `layout`, times `shards`.
fn format_bytes(format: &str, block_bytes: u64, layout: &KvLayout, shards: u32) -> u64 {
    match crate::codec::registry().get(format) {
        Some(c) if format != crate::tier::L0_FORMAT => {
            c.bytes_per_block(layout) * u64::from(shards.max(1))
        }
        _ => block_bytes,
    }
}

/// One demotion copy in flight.
#[derive(Clone, Copy, Debug)]
struct Demoting {
    from: TierId,
    to: TierId,
    /// Bytes of the destination copy (its format's).
    bytes: u64,
    /// A copy ahead ([`KvHierarchy::copy_ahead`]): the source copy stays when it lands.
    keep_source: bool,
}

/// What one admission sees of its cached prefix (the scheduler's `SchedRequest.cached_prefix`).
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixAttach {
    /// L0 blocks of the reused prefix; the request owns one reference to each.
    pub blocks: SmallVec<[BlockId; 16]>,
    /// Tokens the blocks cover (`blocks × block_tokens`); feeds
    /// `ResourceEstimate.cached_prefix_tokens`.
    pub cached_tokens: u32,
    /// Of `cached_tokens`, those served from lossy blocks (P6b S-3): `usage.prompt_tokens_details
    /// .lossy_cached_tokens`.
    pub lossy_tokens: u32,
    pub plan: KvPlan,
}

#[derive(Debug, PartialEq)]
pub enum AttachOutcome {
    /// The prefix is resident in L0: admit now.
    Ready(PrefixAttach),
    /// Promotions are queued; `poll` returns the request once they all landed.
    Promoting,
    /// Another request is computing the next prefix block; attach again next iteration (the
    /// directory stops reporting it pending after `PENDING_WAIT`).
    WaitForPrefix,
}

pub struct AttachRequest<'a> {
    pub request: RequestId,
    pub prompt: &'a [u32],
    pub cache_salt: &'a str,
    pub session: Option<&'a SessionHints>,
    pub priority: Priority,
    /// `x-turbine-kv-lossy` (P6b S-3): `Some(false)` never reuses a lossy block; `None` takes
    /// `kv.lossy_reuse`.
    pub allow_lossy: Option<bool>,
}

/// Plain counters for diagnostics and the offline simulator.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvStats {
    /// Per full prompt block looked up: `[l0, l1, l2, miss]`.
    pub lookups: [u64; 4],
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub recompute_tokens: u64,
    pub transfer_bytes: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
    pub drops: u64,
    pub transfer_errors: u64,
    /// Ladder rewrites started (P6b S-6).
    pub compressions: u64,
    /// Ladder ticks run (each ≥ [`LADDER_TICK_INTERVAL`] after the previous one).
    pub ladder_ticks: u64,
    /// Ladder rewrites not submitted because their tier is backed off after a `Full` (user
    /// decision "6b Task 16: ladder proof results — four open points", 3 A).
    pub ladder_backoff_skips: u64,
    /// Copy-ahead copies landed: a shared parent copied down while its L0 copy stays
    /// ([`KvHierarchy::copy_ahead`]).
    pub copy_aheads: u64,
    /// Blocks queued requests released for pressure reclaim and held by no other request
    /// ([`KvHierarchy::detach_prefix`]).
    pub queued_prefix_detached: u64,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrefetchError {
    #[error("session not found")]
    SessionNotFound,
    #[error("prefetch queue full")]
    QueueFull,
    #[error("KV pressure too high for a prefetch")]
    PressureTooHigh,
}

/// What `POST /turbine/v1/kv/prefetch` names.
pub enum PrefetchTarget<'a> {
    /// A `prompt_cache_key`; its session's recorded salt applies.
    Session(&'a str),
    Tokens {
        prompt: &'a [u32],
        cache_salt: &'a str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchAccepted {
    pub blocks_queued: u32,
    pub blocks_resident: u32,
}

const NO_TARGET: u64 = u64::MAX;

/// Lock-free reclaim requests from the Phase 3 pressure controller: the
/// `turbine_reliability::throttle::KvReclaimer` the server registers (replacing Phase 3's
/// `L0Reclaimer`). `demote` and `free_unreferenced` store the target L0 utilisation (physical:
/// referenced plus cached blocks over the pool) and return the bytes the request covers from the
/// last published snapshot; the engine thread carries them out in `KvHierarchy::apply_reclaim`
/// at the next iteration boundary.
#[derive(Debug)]
pub struct KvReclaimHandle {
    demote_target: AtomicU64,
    free_target: AtomicU64,
    /// Published by the hierarchy: bytes of cached unreferenced L0 blocks.
    reclaimable_bytes: AtomicU64,
    l0_capacity_bytes: AtomicU64,
    l0_used_bytes: AtomicU64,
}

impl KvReclaimHandle {
    fn new() -> Self {
        KvReclaimHandle {
            demote_target: AtomicU64::new(NO_TARGET),
            free_target: AtomicU64::new(NO_TARGET),
            reclaimable_bytes: AtomicU64::new(0),
            l0_capacity_bytes: AtomicU64::new(0),
            l0_used_bytes: AtomicU64::new(0),
        }
    }

    /// Demote idle blocks until L0 utilisation ≤ `target_utilization`; returns the bytes
    /// requested.
    pub fn demote(&self, target_utilization: f64) -> u64 {
        self.demote_target
            .store(target_utilization.to_bits(), Ordering::SeqCst);
        self.requested(target_utilization)
    }

    /// Free unreferenced cached blocks until L0 utilisation ≤ `target_utilization` (copying
    /// the valuable ones down first); returns the bytes requested.
    pub fn free_unreferenced(&self, target_utilization: f64) -> u64 {
        self.free_target
            .store(target_utilization.to_bits(), Ordering::SeqCst);
        self.requested(target_utilization)
    }

    fn take(slot: &AtomicU64) -> Option<f64> {
        let v = slot.swap(NO_TARGET, Ordering::SeqCst);
        (v != NO_TARGET).then(|| f64::from_bits(v))
    }

    fn requested(&self, target: f64) -> u64 {
        let cap = self.l0_capacity_bytes.load(Ordering::SeqCst) as f64;
        let used = self.l0_used_bytes.load(Ordering::SeqCst) as f64;
        let over = (used - target * cap).max(0.0) as u64;
        over.min(self.reclaimable_bytes.load(Ordering::SeqCst))
    }
}

impl KvReclaimer for KvReclaimHandle {
    fn demote(&self, target_utilization: f64) -> u64 {
        KvReclaimHandle::demote(self, target_utilization)
    }

    fn free_unreferenced(&self, target_utilization: f64) -> u64 {
        KvReclaimHandle::free_unreferenced(self, target_utilization)
    }
}

/// Blocks capacity demotion moves per call (P4: bounds the engine thread's work per turn).
pub const CAPACITY_BATCH: usize = 32;

/// Most L0 blocks copied down at once by capacity and pressure reclaim: each pins its L0 block
/// until the copy completes, and each costs the engine thread one enqueue per layer and one
/// poll per layer and turn, so new copies start only as earlier ones finish.
pub const DEMOTION_INFLIGHT: usize = 32;

/// Copy ahead fills a lower tier only up to this share of its capacity (P6b S-8, decision "6b:
/// shared-prefix eval never demotes the prefix to L1", A). A copied-ahead parent's lower copy
/// is held there by the leaf-first rule while its children stay in L0, so the rest is left for
/// demotions that free L0. Debt: a constant, not measured; revisit with a real shared-prompt
/// workload.
pub const COPY_AHEAD_MAX_FILL: f64 = 0.5;

/// Evidence that a cached block will be read again, the precondition for spending a copy on
/// it: it was attached at least once since it was written (a hit), it belongs to a session
/// (`prompt_cache_key`), or it is a shared prefix (two or more cached children). A block of a
/// one-off request has none, however its cost terms score.
pub fn has_reuse_evidence(b: &KvBlock) -> bool {
    b.access_count > 0 || b.session.is_some() || b.child_count >= 2
}

/// Heap entry of `victims`: by value, then older last access, then the deeper block, then key.
struct Victim {
    value: f64,
    last_access: Duration,
    depth: u32,
    key: KvKey,
}

impl Ord for Victim {
    fn cmp(&self, o: &Self) -> CmpOrdering {
        self.value
            .total_cmp(&o.value)
            .then_with(|| self.last_access.cmp(&o.last_access))
            .then_with(|| o.depth.cmp(&self.depth))
            .then_with(|| self.key.cmp(&o.key))
    }
}

impl PartialOrd for Victim {
    fn partial_cmp(&self, o: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(o))
    }
}

impl PartialEq for Victim {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == CmpOrdering::Equal
    }
}

impl Eq for Victim {}

/// One lower tier's ladder state (P6b S-6).
#[derive(Clone, Copy, Debug)]
struct TierRung {
    tier: TierId,
    /// The tier's configured format (`kv.cpu.format` / `kv.nvme.format`): its top rung.
    base: &'static str,
    /// The rung new demotions into the tier take.
    rung: &'static str,
    /// Since when the tier has been below low water (the step-up dwell).
    below_since: Option<Duration>,
    /// Set after a rewrite into the tier ended `Full`: the tier's `room_epoch` at that
    /// rewrite's submit. No rewrite is submitted into the tier until the epoch changes (user
    /// decision "6b Task 16: ladder proof results — four open points", 3 A).
    backoff: Option<u64>,
}

/// The ladder's state: per-tier rungs and the current tick window.
#[derive(Debug)]
struct Ladder {
    cfg: LadderConfig,
    /// The enabled lower tiers, fastest first.
    tiers: SmallVec<[TierRung; 3]>,
    last_tick: Option<Duration>,
    /// Rewrites started since `last_tick`.
    window_rewrites: usize,
}

/// A ladder rewrite in flight.
#[derive(Clone, Copy, Debug)]
struct Compressing {
    tier: TierId,
    /// Bytes the rewritten copy saves (old format's minus new format's).
    saved: u64,
    /// The tier's `room_epoch` when the rewrite was submitted (the back-off's mark).
    epoch: u64,
}

/// An attach waiting for its promotions; `promotions` are the L0 targets still in flight.
struct Pending {
    attach: PrefixAttach,
    promotions: Vec<BlockId>,
    /// Per attached block: served lossy (recounts `lossy_tokens` when a promotion fails).
    lossy: Vec<bool>,
}

/// Per-request KV state from attach until `request_done`.
struct RequestKv {
    salt: String,
    session: Option<SessionId>,
    priority: KvPriority,
    /// Keys of the full blocks seen so far (prompt, then generated tokens): the hash chain
    /// the request's own blocks are keyed by (a lossy chain after a lossy reused block).
    keys: Vec<KvKey>,
    /// Blocks already keyed in the directory (or attached from it).
    committed: usize,
    /// Lineage of the blocks this request computes (P6b S-3).
    lineage: Lineage,
    /// The directory entries of the attached blocks (a lossy copy's promoted entry for a
    /// lossy-promoted block); the first computed block's parent is the last of them.
    used: Vec<KvKey>,
    /// With a lossy lineage: the first lossy attached block and the prompt's exact keys, to
    /// fall back to when a failed promotion cuts the prefix before it.
    lossy_from: Option<(usize, Vec<KvKey>)>,
    /// `Some(cached tokens)` after a queued request released its prefix
    /// ([`KvHierarchy::detach_prefix`]) until it attaches again: that attach is not a new
    /// prompt (its tokens are not counted twice).
    detached_tokens: Option<u32>,
}

impl RequestKv {
    /// The directory entry of full block `i`: the attached entry, else its key.
    fn entry(&self, i: usize) -> KvKey {
        self.used.get(i).copied().unwrap_or(self.keys[i])
    }

    /// The prefix was cut to its first `n` attached blocks.
    fn truncate_attached(&mut self, n: usize) {
        self.used.truncate(n);
        if self.lossy_from.as_ref().is_some_and(|(s, _)| *s >= n)
            && let Some((_, exact)) = self.lossy_from.take()
        {
            self.keys = exact;
            self.lineage = Lineage::Exact;
        }
    }
}

pub struct KvHierarchy {
    cfg: HierarchyConfig,
    clock: Arc<dyn Clock>,
    metrics: KvMetrics,
    stats: KvStats,
    model: ModelFingerprint,
    namespaces: NamespaceCache,
    dir: KvDirectory,
    l1: Option<Arc<dyn KvTier>>,
    l2: Option<Arc<dyn KvTier>>,
    policy: SelectedPolicy,
    transfer: TransferEngine,
    sessions: SessionTable,
    prefetch: PrefetchTracker,
    /// Prefetch tickets in flight → their L0 target.
    prefetch_inflight: HashMap<u64, BlockId>,
    /// Promotion tickets in flight → (owner, L0 target).
    promotions_by_ticket: HashMap<u64, (RequestId, BlockId)>,
    /// Keys with a demotion copy in flight.
    demoting: HashMap<KvKey, Demoting>,
    /// Promotion and prefetch tickets copying a lossy copy of an exact block into L0 → the
    /// lossy key and codec its L0 copy is filed under (P6b S-3). Bounded by the transfers.
    lossy_targets: HashMap<u64, (KvKey, &'static str)>,
    /// The `lossy_key` seed: the tier codecs' rotation seed (the unsalted namespace's).
    lossy_seed: u64,
    /// The L0 layout of one rank shard and the shard count (codec sizes, P6b S-1).
    layout: KvLayout,
    shards: u32,
    /// Keys within the last `lossless_tail_blocks` full blocks of the latest finished sequence
    /// that holds them (P6b S-2): demoted at the L0 format. Entries leave with their block.
    tail: HashSet<KvKey>,
    pending: HashMap<RequestId, Pending>,
    requests: HashMap<RequestId, RequestKv>,
    /// L0 block → the key the directory holds for it.
    l0_keys: HashMap<BlockId, KvKey>,
    l0_state: PressureState,
    prefill_tps: f64,
    reclaim: Arc<KvReclaimHandle>,
    ready: Vec<(RequestId, PrefixAttach)>,
    /// The compression ladder (P6b S-6), when enabled.
    ladder: Option<Ladder>,
    /// Keys with a ladder rewrite in flight.
    compressing: HashMap<KvKey, Compressing>,
    /// L0 blocks the pool holds (`KvHierarchy::new`'s `l0_blocks`): the L0 tier's capacity
    /// divisor (`tier_fill(TierId::L0)`).
    l0_blocks: u32,
    /// The pool's `room_epoch` at the last time the ladder saw the pool: the L0 back-off's
    /// mark, refreshed wherever the ladder reads the pool.
    l0_room_epoch: u64,
    /// Bytes the last pressure reclaim wanted to demote into L1 and L2 (the ladder's demand).
    ladder_demand: [u64; 2],
    /// Blocks the last pressure reclaim at YELLOW or ORANGE wanted and found no unreferenced
    /// block for: what queued requests' prefixes may cover ([`Self::take_queued_prefix_demand`]).
    queued_prefix_demand: usize,
}

impl KvHierarchy {
    /// Prefill rate assumed until the engine measures one (`set_prefill_tps`).
    pub const DEFAULT_PREFILL_TPS: f64 = 5_000.0;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: HierarchyConfig,
        model: ModelIdentity,
        format: KvFormat,
        l0_blocks: u32,
        l1: Option<Arc<dyn KvTier>>,
        l2: Option<Arc<dyn KvTier>>,
        clock: Arc<dyn Clock>,
        metrics: KvMetrics,
    ) -> Self {
        // Prefix keys are cut at `cfg.block_tokens` and the pool pages at the layout's: the two
        // must be the same block size, or keys would name the wrong token ranges.
        assert_eq!(
            cfg.block_tokens, format.layout.block_tokens,
            "HierarchyConfig.block_tokens must equal the KV layout's block_tokens"
        );
        // L1 exists only on discrete-VRAM devices (S-5): on unified memory L0 demotes to L2.
        let l1 = l1.filter(|t| t.enabled() && cfg.memory_kind != MemoryKind::Unified);
        let l2 = l2.filter(|t| t.enabled());
        // A tier holds the most blocks at its own format (lossless-tail blocks are larger), or
        // at the ladder's lossiest rung.
        let smallest = |f: &'static str| {
            cfg.ladder
                .map_or(f, |l| crate::codec::lossier(f, l.max_format))
        };
        let blocks_of = |t: &Option<Arc<dyn KvTier>>, f: &'static str| {
            let f = smallest(f);
            t.as_ref().map_or(0, |t| {
                let bytes = format_bytes(f, cfg.block_bytes, &format.layout, format.shards);
                usize::try_from(t.capacity_bytes() / bytes.max(1)).unwrap_or(usize::MAX)
            })
        };
        let max_entries = (l0_blocks as usize)
            .saturating_add(blocks_of(&l1, cfg.l1_format))
            .saturating_add(blocks_of(&l2, cfg.l2_format));
        let mut namespaces = NamespaceCache::new(model, format);
        let lossy_seed = namespaces.get("").seed();
        // The ladder's tiers, fastest first: L0 when `kv.ladder.l0`, then L1, then L2. L0's
        // base rung is `l0` (the L0 bytes unchanged); its capacity is the pool's.
        let mut tiers: SmallVec<[TierRung; 3]> = SmallVec::new();
        if cfg.ladder_l0 {
            tiers.push(TierRung {
                tier: TierId::L0,
                base: L0_FORMAT,
                rung: L0_FORMAT,
                below_since: None,
                backoff: None,
            });
        }
        for (tier, t, base) in [
            (TierId::L1, &l1, cfg.l1_format),
            (TierId::L2, &l2, cfg.l2_format),
        ] {
            if t.is_some() {
                tiers.push(TierRung {
                    tier,
                    base,
                    rung: base,
                    below_since: None,
                    backoff: None,
                });
            }
        }
        let ladder = cfg.ladder.map(|lc| Ladder {
            cfg: lc,
            tiers,
            last_tick: None,
            window_rewrites: 0,
        });
        if let Some(l) = &ladder {
            for t in &l.tiers {
                metrics.set_ladder_rung(t.tier, t.rung);
            }
        }
        KvHierarchy {
            lossy_targets: HashMap::new(),
            lossy_seed,
            ladder,
            compressing: HashMap::new(),
            l0_blocks,
            l0_room_epoch: 0,
            ladder_demand: [0; 2],
            queued_prefix_demand: 0,
            layout: format.layout,
            shards: format.shards.max(1),
            tail: HashSet::new(),
            policy: cfg.policy,
            transfer: TransferEngine::new(
                cfg.max_inflight_bytes,
                cfg.transfer_queue,
                clock.clone(),
            ),
            sessions: SessionTable::new(cfg.session.clone(), &cfg.prefetch),
            namespaces,
            dir: KvDirectory::new(max_entries),
            model: model.fingerprint(),
            cfg,
            clock,
            metrics,
            stats: KvStats::default(),
            l1,
            l2,
            prefetch: PrefetchTracker::default(),
            prefetch_inflight: HashMap::new(),
            promotions_by_ticket: HashMap::new(),
            demoting: HashMap::new(),
            pending: HashMap::new(),
            requests: HashMap::new(),
            l0_keys: HashMap::new(),
            l0_state: PressureState::Green,
            prefill_tps: Self::DEFAULT_PREFILL_TPS,
            reclaim: Arc::new(KvReclaimHandle::new()),
            ready: Vec::new(),
        }
    }

    /// The handle the pressure controller calls from its own task.
    pub fn reclaimer(&self) -> Arc<KvReclaimHandle> {
        self.reclaim.clone()
    }

    pub fn stats(&self) -> &KvStats {
        &self.stats
    }

    pub fn directory(&self) -> &KvDirectory {
        &self.dir
    }

    /// The directory entry holding L0 block `block`, if it is keyed.
    pub fn l0_entry(&self, block: BlockId) -> Option<&KvBlock> {
        self.l0_keys.get(&block).and_then(|k| self.dir.get(k))
    }

    pub fn transfer(&self) -> &TransferEngine {
        &self.transfer
    }

    /// L0 blocks being copied down: each stays referenced (unallocatable) until its copy
    /// completes.
    pub fn l0_demotions_in_flight(&self) -> usize {
        self.demoting
            .values()
            .filter(|d| d.from == TierId::L0)
            .count()
    }

    /// L0 blocks being copied down that leave L0 when the copy lands (copies ahead keep theirs).
    fn l0_leaving(&self) -> usize {
        self.demoting
            .values()
            .filter(|d| d.from == TierId::L0 && !d.keep_source)
            .count()
    }

    /// Startup calibration seeds path estimates here.
    pub fn transfer_mut(&mut self) -> &mut TransferEngine {
        &mut self.transfer
    }

    pub fn sessions(&self) -> &SessionTable {
        &self.sessions
    }

    pub fn prefetch_stats(&self) -> &PrefetchTracker {
        &self.prefetch
    }

    /// L0 pressure as the Phase 3 controller last reported it.
    pub fn set_l0_state(&mut self, s: PressureState) {
        self.l0_state = s;
    }

    pub fn set_prefill_tps(&mut self, tps: f64) {
        self.prefill_tps = tps.max(1.0);
    }

    pub fn l1_enabled(&self) -> bool {
        self.l1.is_some()
    }

    pub fn l2_enabled(&self) -> bool {
        self.l2.is_some()
    }

    fn tier(&self, t: TierId) -> Option<&Arc<dyn KvTier>> {
        match t {
            TierId::L1 => self.l1.as_ref(),
            TierId::L2 => self.l2.as_ref(),
            _ => None,
        }
    }

    fn now(&self) -> Duration {
        self.clock.now_mono()
    }

    /// A copy of `key` into L0 (promotion or prefetch), or a ladder rewrite of one of its
    /// copies, is queued or in flight: an attach waits for it.
    fn incoming(&self, key: &KvKey) -> bool {
        self.transfer.is_busy_with(key) && !self.demoting.contains_key(key)
    }

    fn lookup_slot(tier: TierId) -> usize {
        match tier {
            TierId::L0 => 0,
            TierId::L1 => 1,
            _ => 2,
        }
    }

    /// The `kv_format` codec `tier` stores blocks in (`l0` for L0 itself).
    fn tier_format(&self, tier: TierId) -> &'static str {
        match tier {
            TierId::L1 => self.cfg.l1_format,
            TierId::L2 => self.cfg.l2_format,
            _ => L0_FORMAT,
        }
    }

    fn format_bytes(&self, format: &str) -> u64 {
        format_bytes(format, self.cfg.block_bytes, &self.layout, self.shards)
    }

    /// Whether `tier` can store one more copy in `format` (its ladder rung) beyond the copies of
    /// that size already in flight into it (P6b S-6). After a rung change the tier's slabs all
    /// hold the old slot size until one empties: a copy at the new rung would end `Full` there
    /// while the tier has free slots of its own format, and the L0 block it should free stays
    /// (the Task 16 multi-turn regression: L0 → L1 demotions halved, ORANGE five times as long).
    fn rung_has_slot(&self, tier: TierId, format: &'static str) -> bool {
        let Some(t) = self.tier(tier) else {
            return false;
        };
        let bytes = self.format_bytes(format);
        let inflight = self
            .demoting
            .values()
            .filter(|d| d.to == tier && d.bytes == bytes)
            .count() as u64;
        t.free_slots(format, bytes) > inflight
    }

    /// The codec new demotions into `tier` take: the ladder's rung, else the tier's format.
    fn rung(&self, tier: TierId) -> &'static str {
        self.ladder
            .as_ref()
            .and_then(|l| l.tiers.iter().find(|t| t.tier == tier))
            .map_or_else(|| self.tier_format(tier), |t| t.rung)
    }

    /// The formats of a copy of `key` from its copy in `from` to `to` (P6b S-1, S-2, S-6): into
    /// L0 the L0 format (a promotion decodes); into a lower tier that tier's ladder rung (its
    /// configured format without the ladder) while the tier has a slot of the rung's size free
    /// ([`KvHierarchy::rung_has_slot`]), else the tier's own format; the L0 format for a
    /// lossless-tail block; never more precise than the source copy.
    fn copy_codec(&self, key: &KvKey, from: TierId, to: TierId) -> Option<TransferCodec> {
        let src = self.dir.get(key)?.location(from)?.format;
        let dst = if to == TierId::L0 {
            L0_FORMAT
        } else if self.tail.contains(key) {
            // A tail block keeps the format it holds (the L0 base format, or its recent-window
            // BF16 class): `lossier` would rank the unregistered `bf16` most precise.
            if src == "bf16" {
                src
            } else {
                crate::codec::lossier(src, L0_FORMAT)
            }
        } else {
            let rung = crate::codec::lossier(src, self.rung(to));
            let own = crate::codec::lossier(src, self.tier_format(to));
            if rung != own && !self.rung_has_slot(to, rung) {
                own
            } else {
                rung
            }
        };
        Some(TransferCodec {
            from: src,
            from_bytes: self.format_bytes(src),
            to: dst,
            to_bytes: self.format_bytes(dst),
        })
    }

    /// A copy request of `key` along `path` with its formats; the bytes it counts in flight
    /// are the smaller copy's (the lower tier's, which crosses the host link).
    #[allow(clippy::too_many_arguments)]
    fn copy_request(
        &self,
        path: TransferPath,
        key: KvKey,
        owner: Option<RequestId>,
        purpose: TransferPurpose,
        src_slot: u64,
        dst_slot: u64,
    ) -> Option<TransferRequest> {
        let codec = self.copy_codec(&key, path.from(), path.to())?;
        Some(TransferRequest {
            path,
            key,
            bytes: codec.from_bytes.min(codec.to_bytes),
            owner,
            purpose,
            src_slot,
            dst_slot,
            codec,
        })
    }

    /// Blocks of `tier`'s format a demotion of `key` from `from` into `tier` takes (more than
    /// one for a lossless-tail block stored at the larger L0 format).
    /// The bytes a demotion of `key` from `from` into `tier` stores: the codec's destination
    /// size ([`KvHierarchy::copy_codec`], the tier's format when the rung has no slot), the
    /// currency [`KvHierarchy::make_room`] counts in.
    fn demotion_bytes(&self, key: &KvKey, from: TierId, tier: TierId) -> u64 {
        self.copy_codec(key, from, tier)
            .map_or(1, |c| c.to_bytes.max(1))
    }

    /// Admission-time prefix match (S-3, S-9): the planner's cutoff, L0 blocks attached by
    /// reference, lower-tier blocks promoted into freshly allocated L0 blocks.
    pub fn attach_prefix(
        &mut self,
        pool: &mut BlockPool,
        req: &AttachRequest<'_>,
    ) -> AttachOutcome {
        let now = self.now();
        let bt = self.cfg.block_tokens;
        if !self.requests.contains_key(&req.request) {
            let session = req.session.map(|h| SessionId {
                key: h.session_id.clone(),
                salt: req.cache_salt.to_string(),
            });
            if let (Some(id), Some(h)) = (&session, req.session) {
                self.sessions.begin(id.clone(), h, now);
            }
            self.requests.insert(
                req.request,
                RequestKv {
                    salt: req.cache_salt.to_string(),
                    session,
                    priority: KvPriority::from_class(req.priority.class()),
                    keys: Vec::new(),
                    committed: 0,
                    lineage: Lineage::Exact,
                    used: Vec::new(),
                    lossy_from: None,
                    detached_tokens: None,
                },
            );
        }
        let detached = self
            .requests
            .get(&req.request)
            .and_then(|r| r.detached_tokens);
        let allow_lossy = req.allow_lossy.unwrap_or(self.cfg.allow_lossy);
        let hasher = Blake3Hasher(self.namespaces.get(req.cache_salt));
        let m = if self.cfg.prefix_sharing {
            self.dir
                .lookup(&hasher, req.prompt, bt, now, allow_lossy, self.lossy_seed)
        } else {
            PrefixMatch {
                keys: prefix_keys(&hasher, req.prompt, bt),
                ..PrefixMatch::default()
            }
        };
        // Wait while another request computes the next block, or while a matched block is
        // already on its way into L0 (a second promotion would duplicate its L0 copy).
        if m.pending.is_some()
            || m.blocks
                .iter()
                .any(|b| b.tier != TierId::L0 && self.incoming(&b.key))
        {
            return AttachOutcome::WaitForPrefix;
        }
        self.metrics.record_lookup(&m);
        for b in &m.blocks {
            self.stats.lookups[Self::lookup_slot(b.tier)] += 1;
        }
        self.stats.lookups[3] += (m.keys.len() - m.blocks.len()) as u64;

        let tiers: Vec<TierId> = m.blocks.iter().map(|b| b.tier).collect();
        let penalties: Vec<Option<f64>> = m
            .blocks
            .iter()
            .map(|b| b.lossy.map(|f| self.cfg.penalty_of(f)))
            .collect();
        let copy_bytes: Vec<u64> = m
            .blocks
            .iter()
            .map(|b| self.format_bytes(b.location.format))
            .collect();
        let inputs = PlanInputs {
            matched: &tiers,
            prompt_tokens: req.prompt.len() as u32,
            block_tokens: bt,
            block_bytes: self.cfg.block_bytes,
            prefill_tps: self.prefill_tps,
            l1_to_l0: self
                .l1
                .as_ref()
                .map(|_| self.transfer.estimate(TransferPath::L1ToL0)),
            l2_to_l0: self
                .l2
                .as_ref()
                .map(|_| self.transfer.estimate(TransferPath::L2ToL0)),
            l0_state: self.l0_state,
            l1_degraded: self.l1.as_ref().is_some_and(|t| t.degraded()),
            l2_degraded: self.l2.as_ref().is_some_and(|t| t.degraded()),
            copy_bytes: &copy_bytes,
            lossy_penalty: &penalties,
            allow_lossy,
        };
        let mut plan = plan_prefix(&inputs);
        // The planner's inputs and costs, for the `kv_plan` DEBUG event: the cost of no reuse,
        // of the chosen cutoff and of reusing every block the plan may reuse.
        let reusable = reuse_cap(inputs.prompt_tokens, bt).min(tiers.len());
        let plan_costs = (
            plan_cost(&inputs, 0),
            plan_cost(&inputs, plan.cutoff_blocks() as usize),
            plan_cost(&inputs, reusable),
        );
        let l1_est = inputs.l1_to_l0;
        let lower_bytes = copy_bytes
            .iter()
            .zip(&tiers)
            .find(|(_, t)| **t != TierId::L0)
            .map_or(0, |(b, _)| *b);
        let k = plan.cutoff_blocks() as usize;
        // Reference every reused L0 block first so allocating promotion targets can never
        // reclaim one of them.
        for mb in m.blocks[..k].iter().filter(|mb| mb.tier == TierId::L0) {
            pool.incref(BlockId(mb.location.slot as u32));
        }
        let mut blocks: SmallVec<[BlockId; 16]> = SmallVec::new();
        let mut promotions = Vec::new();
        let mut cut = k;
        for (i, mb) in m.blocks[..k].iter().enumerate() {
            if mb.tier == TierId::L0 {
                blocks.push(BlockId(mb.location.slot as u32));
                if self.prefetch.attached(&mb.key) {
                    self.metrics.prefetch_outcome(PrefetchOutcome::Used);
                }
            } else {
                let Ok(ids) = pool.allocate(1) else {
                    cut = i;
                    break;
                };
                let path = TransferPath::between(mb.tier, TierId::L0)
                    .expect("a lower local tier has a path to L0");
                let tr = self
                    .copy_request(
                        path,
                        mb.key,
                        Some(req.request),
                        TransferPurpose::Promote,
                        mb.location.slot,
                        u64::from(ids[0].0),
                    )
                    .expect("a matched block has a copy in its tier");
                match self.transfer.submit(tr) {
                    Ok(t) => {
                        if let Some(target) = self.lossy_target(&mb.key, mb.location) {
                            self.lossy_targets.insert(t.id, target);
                        }
                        self.promotions_by_ticket
                            .insert(t.id, (req.request, ids[0]));
                        promotions.push(ids[0]);
                        blocks.push(ids[0]);
                    }
                    Err(_) => {
                        pool.release(&ids);
                        cut = i;
                        break;
                    }
                }
            }
            self.dir
                .record_hit(&mb.key, now, self.cfg.policy.weights.hit_half_life);
        }
        // Allocation may have reclaimed cached blocks: forget their L0 copies now.
        self.after_plan(pool);
        if cut < k {
            let unused: Vec<BlockId> = m.blocks[cut..k]
                .iter()
                .filter(|mb| mb.tier == TierId::L0)
                .map(|mb| BlockId(mb.location.slot as u32))
                .collect();
            pool.release(&unused);
            plan = truncated_plan(&tiers[..cut], req.prompt.len() as u32, bt, plan.reason);
        }
        // The keys and lineage of the blocks this request computes (P6b S-3): a lossy chain
        // once a block before the cut is served lossy.
        let (keys, lineage) = m.keys_for(cut);
        let keys = keys.to_vec();
        // Blocks this request will compute are pending, so an identical concurrent prefix waits.
        for key in keys.iter().skip(m.blocks.len()) {
            self.dir.register_pending(*key, now);
        }
        let lossy: Vec<bool> = m.blocks[..cut].iter().map(|b| b.lossy.is_some()).collect();
        let used: Vec<KvKey> = m.blocks[..cut]
            .iter()
            .map(|b| {
                self.lossy_target(&b.key, b.location)
                    .map_or(b.key, |(k, _)| k)
            })
            .collect();
        let cached_tokens = blocks.len() as u32 * bt;
        let lossy_tokens = lossy.iter().filter(|l| **l).count() as u32 * bt;
        if detached.is_none() {
            self.metrics
                .lossy_cached_tokens
                .inc_by(u64::from(lossy_tokens));
        }
        if m.denied {
            self.metrics.lossy_denied.inc();
        }
        if lossy_tokens > 0 || m.denied {
            let mut formats: Vec<&str> = m.blocks[..cut].iter().filter_map(|b| b.lossy).collect();
            formats.dedup();
            tracing::debug!(
                event = "kv_lossy_reuse",
                request_id = ?req.request,
                lossy_tokens,
                formats = ?formats,
                denied = m.denied,
            );
        }
        match detached {
            None => {
                self.metrics.plan(plan.reason, plan.recompute_tokens);
                self.metrics.prompt_tokens.inc_by(req.prompt.len() as u64);
                self.metrics
                    .prefix_cached_tokens
                    .inc_by(u64::from(cached_tokens));
                self.stats.prompt_tokens += req.prompt.len() as u64;
                self.stats.cached_tokens += u64::from(cached_tokens);
                self.stats.recompute_tokens += u64::from(plan.recompute_tokens);
            }
            // The attach after a queued request released its prefix: its prompt and cached
            // tokens were counted by the first attach; only what it now recomputes of them is new.
            Some(before) => {
                let extra = before.saturating_sub(cached_tokens);
                self.metrics.plan(plan.reason, extra);
                self.stats.recompute_tokens += u64::from(extra);
            }
        }
        tracing::debug!(
            event = "kv_plan",
            request_id = ?req.request,
            matched_l0 = tiers.iter().filter(|t| **t == TierId::L0).count(),
            matched_l1 = tiers.iter().filter(|t| **t == TierId::L1).count(),
            matched_l2 = tiers.iter().filter(|t| **t == TierId::L2).count(),
            matched_lossy = penalties.iter().filter(|p| p.is_some()).count(),
            cutoff = blocks.len(),
            recompute_tokens = plan.recompute_tokens,
            reason = plan.reason.as_str(),
            prefill_tps = self.prefill_tps,
            l1_latency_s = l1_est.map_or(0.0, |e| e.latency_s),
            l1_bandwidth_bps = l1_est.map_or(0.0, |e| e.bandwidth_bps),
            lower_copy_bytes = lower_bytes,
            cost_recompute_s = plan_costs.0,
            cost_chosen_s = plan_costs.1,
            cost_reuse_all_s = plan_costs.2,
        );
        if let Some(r) = self.requests.get_mut(&req.request) {
            r.committed = blocks.len();
            r.lossy_from = match (lineage, m.lossy_from) {
                (Lineage::Lossy { .. }, Some((s, _))) => Some((s, m.keys.clone())),
                _ => None,
            };
            r.keys = keys;
            r.lineage = lineage;
            r.used = used;
            r.detached_tokens = None;
        }
        let attach = PrefixAttach {
            blocks,
            cached_tokens,
            lossy_tokens,
            plan,
        };
        if promotions.is_empty() {
            AttachOutcome::Ready(attach)
        } else {
            self.pending.insert(
                req.request,
                Pending {
                    attach,
                    promotions,
                    lossy,
                },
            );
            AttachOutcome::Promoting
        }
    }

    /// A request waiting in the admission queue gives its attached prefix back for pressure
    /// reclaim (decision "6b: queued-prefix demotion — granularity and scope", 1 A, 2 A): its
    /// references to `attach.blocks` are released, so the blocks it alone held turn cached and
    /// unreferenced and the next reclaim demotes them (or frees them at once when copy ahead
    /// already put a lower copy down). The blocks it registered as pending are cleared, or its
    /// own next attach would wait on them for `PENDING_WAIT`; that attach goes through the
    /// planner again. `alone` (blocks no other request held) is counted in
    /// `turbine_kv_queued_prefix_detached_blocks_total`; DEBUG event `kv_queued_prefix_detach`
    /// with reason `queued_prefix`.
    pub fn detach_prefix(
        &mut self,
        pool: &mut BlockPool,
        request: RequestId,
        attach: &PrefixAttach,
        alone: usize,
    ) {
        pool.release(&attach.blocks);
        if let Some(r) = self.requests.get_mut(&request) {
            for key in r.keys.iter().skip(r.committed) {
                self.dir.clear_pending(key);
            }
            r.keys.clear();
            r.used.clear();
            r.committed = 0;
            r.lineage = Lineage::Exact;
            r.lossy_from = None;
            r.detached_tokens = Some(attach.cached_tokens);
        }
        self.metrics
            .queued_prefix_detached_blocks
            .inc_by(alone as u64);
        self.stats.queued_prefix_detached += alone as u64;
        tracing::debug!(
            event = "kv_queued_prefix_detach",
            reason = "queued_prefix",
            request_id = ?request,
            blocks = attach.blocks.len(),
            alone,
            cached_tokens = attach.cached_tokens,
        );
    }

    /// L0 blocks the pending attach of `request` holds (reused ones and promotion targets), 0
    /// when none is pending.
    pub fn pending_blocks(&self, request: RequestId) -> usize {
        self.pending
            .get(&request)
            .map_or(0, |p| p.attach.blocks.len())
    }

    /// The blocks the last pressure reclaim wanted at YELLOW or ORANGE beyond every
    /// unreferenced L0 block (0 at GREEN, RED and SURVIVAL); taken once.
    pub fn take_queued_prefix_demand(&mut self) -> usize {
        std::mem::take(&mut self.queued_prefix_demand)
    }

    /// The lossy key and codec a copy of `key` at `loc` promoted into L0 is filed under: a
    /// lossy-format copy of an exact block (P6b S-3); `None` otherwise (its L0 copy joins the
    /// block's own entry).
    fn lossy_target(&self, key: &KvKey, loc: KvLocation) -> Option<(KvKey, &'static str)> {
        if loc.tier == TierId::L0 {
            return None;
        }
        let b = self.dir.get(key)?;
        (b.lineage == Lineage::Exact && format_is_lossy(loc.format, &b.format.layout))
            .then(|| (lossy_key(*key, loc.format, self.lossy_seed), loc.format))
    }

    /// The entry a lossy copy of exact block `key` promoted into L0 is filed under: `lk`,
    /// created (lossy lineage, the block's parent) when absent. `None` when `key` is gone or
    /// the directory is full.
    fn lossy_entry(&mut self, key: &KvKey, lk: KvKey, format: &'static str) -> Option<KvKey> {
        if self.dir.get(&lk).is_some() {
            return Some(lk);
        }
        let src = self.dir.get(key)?;
        let entry = KvBlock {
            key: lk,
            locations: SmallVec::new(),
            access_count: 0,
            ref_count: 0,
            child_count: 0,
            lineage: Lineage::Lossy { format },
            ..src.clone()
        };
        match self.dir.insert(entry) {
            Ok(()) => Some(lk),
            Err(e) => {
                tracing::debug!(event = "kv_commit_skipped", key = %lk, error = %e);
                None
            }
        }
    }

    /// Keys the full blocks of `tokens` (prompt, then generated tokens) that `table` holds and
    /// marks them cached in L0. Call after every iteration that advanced the sequence.
    pub fn commit_progress(
        &mut self,
        pool: &mut BlockPool,
        request: RequestId,
        table: &[BlockId],
        tokens: &[u32],
    ) {
        if !self.cfg.prefix_sharing {
            return;
        }
        let bt = self.cfg.block_tokens as usize;
        let now = self.now();
        let Some(r) = self.requests.get_mut(&request) else {
            return;
        };
        let full = (tokens.len() / bt).min(table.len());
        if r.keys.len() < full {
            let hasher = Blake3Hasher(self.namespaces.get(&r.salt));
            while r.keys.len() < full {
                let i = r.keys.len();
                let key = hasher.key(r.keys.last(), &tokens[i * bt..(i + 1) * bt]);
                r.keys.push(key);
            }
        }
        let start = r.committed;
        if start >= full {
            return;
        }
        let (session, priority, lineage) = (r.session.clone(), r.priority, r.lineage);
        let keys: Vec<KvKey> = r.keys[..full].to_vec();
        // The first computed block hangs off the last attached entry (a lossy copy's entry
        // after a lossy-promoted block); later ones off their predecessor.
        let attached = (r.used.len(), r.used.last().copied());
        r.committed = full;
        for i in start..full {
            let (key, block) = (keys[i], table[i]);
            self.dir.clear_pending(&key);
            if let Some(b) = self.dir.get(&key) {
                // An exact block whose every copy is lossy, recomputed by a request that did
                // not reuse them: its exact L0 copy is published alongside (P6b S-3).
                let publish = lineage == Lineage::Exact
                    && b.lineage == Lineage::Exact
                    && b.location(TierId::L0).is_none()
                    && !b.locations.is_empty()
                    && b.locations
                        .iter()
                        .all(|l| format_is_lossy(l.format, &b.format.layout))
                    && !self.incoming(&key);
                if publish {
                    self.dir.add_location(
                        &key,
                        KvLocation {
                            tier: TierId::L0,
                            slot: u64::from(block.0),
                            format: l0_tag(pool, block),
                        },
                    );
                    pool.set_keyed(block);
                    self.l0_keys.insert(block, key);
                }
                continue;
            }
            let parent = match i.checked_sub(1) {
                Some(p) if i == attached.0 => attached.1.or(Some(keys[p])),
                Some(p) => Some(keys[p]),
                None => None,
            };
            let kv = KvBlock {
                key,
                model: self.model,
                token_range: TokenRange {
                    start: (i * bt) as u32,
                    end: ((i + 1) * bt) as u32,
                },
                format: self.namespaces.format(),
                size_bytes: self.cfg.block_bytes,
                locations: SmallVec::from_slice(&[KvLocation {
                    tier: TierId::L0,
                    slot: u64::from(block.0),
                    format: l0_tag(pool, block),
                }]),
                access_count: 0,
                last_access: now,
                ref_count: pool.refcount(block),
                priority,
                recompute_cost: CostEstimate {
                    seconds: recompute_seconds(bt as u32, ((i + 1) * bt) as u32, self.prefill_tps),
                },
                decayed_hits: 0.0,
                decayed_at: now,
                session: session.clone(),
                parent,
                child_count: 0,
                tokens: tokens[i * bt..(i + 1) * bt].into(),
                lineage,
            };
            match self.dir.insert(kv) {
                Ok(()) => {
                    pool.set_keyed(block);
                    self.l0_keys.insert(block, key);
                }
                Err(e) => {
                    tracing::debug!(event = "kv_commit_skipped", key = %key, error = %e);
                    break;
                }
            }
        }
    }

    /// The request finished, failed or was cancelled; the scheduler already released the
    /// references of an admitted request. A cancelled request's queued promotions are dropped
    /// and its in-flight ones land as cached unreferenced blocks, never in the sequence.
    pub fn request_done(&mut self, pool: &mut BlockPool, request: RequestId, cancelled: bool) {
        let now = self.now();
        if cancelled {
            for t in self.transfer.cancel_owner(request) {
                self.lossy_targets.remove(&t.id);
                if let Some((_, block)) = self.promotions_by_ticket.remove(&t.id) {
                    pool.release(&[block]);
                }
            }
            if let Some(p) = self.pending.remove(&request) {
                // Reused L0 blocks are released now; promotion targets are released above
                // (queued) or on completion (in flight).
                let targets: HashSet<BlockId> = p.promotions.iter().copied().collect();
                let reused: Vec<BlockId> = p
                    .attach
                    .blocks
                    .iter()
                    .copied()
                    .filter(|b| !targets.contains(b))
                    .collect();
                pool.release(&reused);
            }
        }
        if let Some(r) = self.requests.remove(&request) {
            for key in r.keys.iter().skip(r.committed) {
                self.dir.clear_pending(key);
            }
            // The sequence's last full blocks form its lossless tail (P6b S-2); its earlier
            // blocks are no longer the tail of the latest sequence holding them.
            // Directory entries, so a lossy-promoted attached block names its lossy entry.
            let committed: Vec<KvKey> = (0..r.committed).map(|i| r.entry(i)).collect();
            let cut = committed
                .len()
                .saturating_sub(self.cfg.lossless_tail_blocks as usize);
            for key in &committed[..cut] {
                self.tail.remove(key);
            }
            for key in &committed[cut..] {
                if self.dir.get(key).is_some() {
                    self.tail.insert(*key);
                }
            }
            if let Some(id) = r.session {
                let keys = committed;
                if self.sessions.finish(&id, keys, now) {
                    tracing::debug!(event = "kv_session_ended", session = %id.key);
                }
            }
        }
        self.metrics.sessions.set(self.sessions.len() as i64);
    }

    /// Advances transfers and applies their completions; returns the requests whose
    /// promotions all landed (or were cut short by a failed copy).
    pub fn poll(
        &mut self,
        pool: &mut BlockPool,
        backend: &mut dyn TransferBackend,
    ) -> Vec<(RequestId, PrefixAttach)> {
        for c in self.transfer.pump(backend) {
            let t = &c.ticket;
            let path = t.req.path;
            match c.result {
                Ok((took, slot)) => {
                    if t.req.purpose != TransferPurpose::Compress {
                        self.stats.transfer_bytes += t.req.bytes;
                        self.metrics.transfer(
                            path,
                            t.req.bytes,
                            took.as_secs_f64(),
                            self.transfer.estimate(path).bandwidth_bps,
                        );
                    }
                    self.on_copy_done(pool, t.id, &t.req, slot.0, c.owner_cancelled);
                }
                Err(e) => {
                    self.stats.transfer_errors += 1;
                    tracing::debug!(
                        event = "kv_transfer_failed",
                        path = path.as_str(),
                        key = %t.req.key,
                        error = %e
                    );
                    self.on_copy_failed(pool, t.id, &t.req, &e);
                }
            }
        }
        self.publish(pool);
        std::mem::take(&mut self.ready)
    }

    fn on_copy_done(
        &mut self,
        pool: &mut BlockPool,
        ticket: u64,
        req: &TransferRequest,
        slot: u64,
        owner_cancelled: bool,
    ) {
        let (from, to) = (req.path.from(), req.path.to());
        match req.purpose {
            TransferPurpose::Demote => {
                let keep = self
                    .demoting
                    .remove(&req.key)
                    .is_some_and(|d| d.keep_source);
                if from == TierId::L0 {
                    pool.release(&[BlockId(req.src_slot as u32)]);
                }
                if self.dir.get(&req.key).is_none() {
                    // Dropped while the copy ran: the new copy has no owner.
                    if let Some(t) = self.tier(to) {
                        let _ = t.evict(&req.key);
                    }
                    return;
                }
                self.dir.add_location(
                    &req.key,
                    KvLocation {
                        tier: to,
                        slot,
                        format: req.codec.to,
                    },
                );
                if keep {
                    self.metrics.copy_ahead(to);
                    self.stats.copy_aheads += 1;
                } else {
                    self.metrics.demotion(from, to);
                    self.stats.demotions += 1;
                }
                // Stored at the tier's ladder rung, lossier than its own format would make it.
                let unladdered = crate::codec::lossier(req.codec.from, self.tier_format(to));
                let rung = crate::codec::lossier(req.codec.from, self.rung(to));
                if self.ladder.is_some() && req.codec.to != unladdered {
                    self.metrics.ladder_action(
                        to,
                        unladdered,
                        req.codec.to,
                        LadderReason::NewDemotion,
                    );
                } else if self.ladder.is_some()
                    && rung != unladdered
                    && !self.tail.contains(&req.key)
                {
                    // Stored at the tier's own format: no slot of the rung's size was free.
                    self.metrics
                        .ladder_action(to, rung, req.codec.to, LadderReason::RungNoSlot);
                }
                if !keep {
                    self.remove_copy(pool, &req.key, from, EvictReason::Pressure);
                }
            }
            TransferPurpose::Compress if from == TierId::L0 => {
                self.compressing.remove(&req.key);
                let dst = BlockId(req.dst_slot as u32);
                if self
                    .dir
                    .get(&req.key)
                    .and_then(|b| b.location(TierId::L0))
                    .is_none()
                {
                    // The block left L0 while it was rewritten: the page is free again.
                    pool.release(&[dst]);
                    return;
                }
                // The page was allocated at refcount 1 (the rewrite's); as a cached copy it
                // is held by the keyed flag at refcount 0.
                pool.set_keyed(dst);
                pool.release(&[dst]);
                self.l0_keys.insert(dst, req.key);
                self.dir.add_location(
                    &req.key,
                    KvLocation {
                        tier: TierId::L0,
                        slot: req.dst_slot,
                        format: req.codec.to,
                    },
                );
                let src = BlockId(req.src_slot as u32);
                if pool.is_keyed(src) {
                    pool.evict_cached(src);
                } else {
                    pool.release(&[src]);
                }
                self.metrics.eviction(TierId::L0, EvictReason::Compressed);
            }
            TransferPurpose::Compress => {
                self.compressing.remove(&req.key);
                let tier = from;
                if self
                    .dir
                    .get(&req.key)
                    .and_then(|b| b.location(tier))
                    .is_none()
                {
                    // The copy left the tier (or the block was dropped) while it was rewritten.
                    if let Some(t) = self.tier(tier) {
                        let _ = t.evict(&req.key);
                    }
                    return;
                }
                self.dir.add_location(
                    &req.key,
                    KvLocation {
                        tier,
                        slot,
                        format: req.codec.to,
                    },
                );
                self.metrics.eviction(tier, EvictReason::Compressed);
            }
            TransferPurpose::Promote | TransferPurpose::Prefetch => {
                let block = BlockId(req.dst_slot as u32);
                // A lossy copy of an exact block lands under its lossy key (P6b S-3), never as
                // an exact L0 copy of the block.
                let key = match self.lossy_targets.remove(&ticket) {
                    Some((lk, format)) => self.lossy_entry(&req.key, lk, format),
                    None => self.dir.get(&req.key).map(|_| req.key),
                };
                let filed = key;
                if let Some(key) = key
                    && self
                        .dir
                        .get(&key)
                        .is_some_and(|b| b.location(TierId::L0).is_none())
                {
                    pool.set_keyed(block);
                    self.l0_keys.insert(block, key);
                    self.dir.add_location(
                        &key,
                        KvLocation {
                            tier: TierId::L0,
                            slot: u64::from(block.0),
                            format: L0_FORMAT,
                        },
                    );
                }
                self.metrics.promotion(from, TierId::L0);
                self.stats.promotions += 1;
                if req.purpose == TransferPurpose::Prefetch {
                    self.prefetch_inflight.remove(&ticket);
                    // Tracked under the entry the copy is filed in, the key a request attaches.
                    self.prefetch.issued(filed.unwrap_or(req.key));
                    pool.release(&[block]);
                    return;
                }
                self.promotions_by_ticket.remove(&ticket);
                let owner = req.owner.expect("promotions have an owner");
                if owner_cancelled {
                    pool.release(&[block]);
                    return;
                }
                if let Some(p) = self.pending.get_mut(&owner) {
                    p.promotions.retain(|b| *b != block);
                    if p.promotions.is_empty() {
                        let p = self.pending.remove(&owner).expect("present above");
                        self.ready.push((owner, p.attach));
                    }
                }
            }
        }
    }

    fn on_copy_failed(
        &mut self,
        pool: &mut BlockPool,
        ticket: u64,
        req: &TransferRequest,
        error: &TierError,
    ) {
        self.lossy_targets.remove(&ticket);
        for t in [req.path.from(), req.path.to()] {
            if let Some(tier) = self.tier(t)
                && tier.degraded()
            {
                self.metrics.set_degraded(t, true);
                tracing::warn!(
                    event = "kv_tier_degraded",
                    tier = t.as_str(),
                    "KV tier degraded after repeated copy errors"
                );
            }
        }
        match req.purpose {
            TransferPurpose::Demote => {
                self.demoting.remove(&req.key);
                if req.path.from() == TierId::L0 {
                    pool.release(&[BlockId(req.src_slot as u32)]);
                }
            }
            TransferPurpose::Compress => {
                // The copy keeps its old format unless the failed rewrite lost it.
                let compressing = self.compressing.remove(&req.key);
                let tier = req.path.from();
                if matches!(error, TierError::Full)
                    && let Some(c) = compressing
                {
                    self.ladder_back_off(tier, c.epoch, req.codec);
                }
                if tier == TierId::L0 {
                    // The destination page was allocated at submit; the source page keeps the
                    // block's bytes (a rewrite never loses an L0 copy).
                    pool.release(&[BlockId(req.dst_slot as u32)]);
                    return;
                }
                let lost = self.tier(tier).is_some_and(|t| !t.contains(&req.key));
                if lost
                    && self
                        .dir
                        .get(&req.key)
                        .is_some_and(|b| b.location(tier).is_some())
                {
                    self.remove_copy(pool, &req.key, tier, EvictReason::TierDegraded);
                }
            }
            TransferPurpose::Prefetch => {
                self.prefetch_inflight.remove(&ticket);
                pool.release(&[BlockId(req.dst_slot as u32)]);
                self.prefetch.reject(1);
                self.metrics.prefetch_outcome(PrefetchOutcome::Rejected);
            }
            TransferPurpose::Promote => {
                // The copy is discarded; the request recomputes from the failed block on and
                // is never failed by it.
                self.promotions_by_ticket.remove(&ticket);
                let owner = req.owner.expect("promotions have an owner");
                let failed = BlockId(req.dst_slot as u32);
                let Some(mut p) = self.pending.remove(&owner) else {
                    pool.release(&[failed]);
                    return;
                };
                for t in self.transfer.cancel_owner(owner) {
                    self.promotions_by_ticket.remove(&t.id);
                    self.lossy_targets.remove(&t.id);
                }
                // In-flight targets are released when their copies complete (owner cancelled).
                let still_inflight: HashSet<BlockId> = self
                    .promotions_by_ticket
                    .values()
                    .filter(|(r, _)| *r == owner)
                    .map(|(_, b)| *b)
                    .collect();
                let pos = p
                    .attach
                    .blocks
                    .iter()
                    .position(|b| *b == failed)
                    .unwrap_or(p.attach.blocks.len());
                let dropped: Vec<BlockId> = p
                    .attach
                    .blocks
                    .drain(pos..)
                    .filter(|b| !still_inflight.contains(b))
                    .collect();
                pool.release(&dropped);
                let bt = self.cfg.block_tokens;
                let prompt = p.attach.cached_tokens + p.attach.plan.recompute_tokens;
                p.attach.cached_tokens = p.attach.blocks.len() as u32 * bt;
                p.lossy.truncate(p.attach.blocks.len());
                p.attach.lossy_tokens = p.lossy.iter().filter(|l| **l).count() as u32 * bt;
                p.attach.plan = KvPlan {
                    reuse_l0: p.attach.plan.reuse_l0.min(p.attach.blocks.len() as u32),
                    promote: Vec::new(),
                    recompute_tokens: prompt - p.attach.cached_tokens,
                    reason: PlanReason::TierDegraded,
                };
                self.metrics
                    .plan(PlanReason::TierDegraded, p.attach.plan.recompute_tokens);
                if let Some(r) = self.requests.get_mut(&owner) {
                    r.committed = p.attach.blocks.len();
                    r.truncate_attached(p.attach.blocks.len());
                }
                self.ready.push((owner, p.attach));
            }
        }
    }

    /// Removes the copy of `key` in `tier`; forgets the block once no copy remains.
    fn remove_copy(
        &mut self,
        pool: &mut BlockPool,
        key: &KvKey,
        tier: TierId,
        reason: EvictReason,
    ) {
        if tier == TierId::L0 {
            let Some(loc) = self.dir.get(key).and_then(|b| b.location(TierId::L0)) else {
                return;
            };
            let b = BlockId(loc.slot as u32);
            if pool.refcount(b) > 0 {
                // Attached again while the copy ran: the L0 copy stays.
                return;
            }
            if !pool.evict_cached(b) {
                self.dir.stale_location(key, TierId::L0);
                return;
            }
            self.l0_keys.remove(&b);
            if let Some(e) = self.dir.get_mut(key) {
                e.ref_count = 0;
            }
            if self.prefetch.evicted(key) {
                self.metrics.prefetch_outcome(PrefetchOutcome::Wasted);
            }
        } else if let Some(t) = self.tier(tier) {
            let _ = t.evict(key);
        }
        self.metrics.eviction(tier, reason);
        self.stats.evictions += 1;
        if self.dir.remove_location(key, tier) == Some(0) {
            self.metrics.drop_block(reason);
            self.stats.drops += 1;
            self.forget(key);
        }
    }

    /// Removes location-less entries bottom-up (a parent goes once its last child is gone).
    fn forget(&mut self, key: &KvKey) {
        let mut next = Some(*key);
        while let Some(k) = next.take() {
            let Some(b) = self.dir.get(&k) else { break };
            if !b.locations.is_empty() || b.child_count > 0 {
                break;
            }
            next = b.parent;
            self.dir.remove(&k);
            self.tail.remove(&k);
        }
    }

    fn score_one(
        &self,
        b: &KvBlock,
        tier: TierId,
        pool: &BlockPool,
        now: Duration,
    ) -> BlockScoreInputs {
        let (capacity, pressure) = match tier {
            TierId::L0 => (
                u64::from(pool.total_blocks()) * self.cfg.block_bytes,
                self.l0_state,
            ),
            t => self.tier(t).map_or((1, PressureState::Green), |x| {
                (x.capacity_bytes(), x.pressure())
            }),
        };
        let depth = b.token_range.end;
        let recompute = recompute_seconds(
            b.token_range.end - b.token_range.start,
            depth,
            self.prefill_tps,
        );
        // Bringing the block back costs a copy from where it would go, or a recompute when it
        // would be dropped. The copy is priced at the bytes it would be stored at there (path
        // estimates are rates per encoded byte, like the planner's `copy_bytes`), and the
        // memory the copy holds is its encoded size in `tier` (user decision 2026-09-30, B).
        // For eviction order a lossless last block is priced like its history, at the rung of
        // `to`; it is still demoted (and its memory counted) at the L0 format (user decision
        // 2026-10-02, 1 A: at its own L0 size it outranked every history block, so histories
        // drained first).
        let retrieval = match demotion_target(tier, self.l1.is_some(), self.l2.is_some()) {
            Some(to) => {
                let path = TransferPath::between(to, TierId::L0).expect("lower tier to L0");
                let bytes = self
                    .copy_codec(&b.key, tier, to)
                    .map_or(self.cfg.block_bytes, |c| {
                        self.format_bytes(crate::codec::lossier(c.from, self.rung(to)))
                    });
                self.transfer.estimate(path).block_seconds(bytes)
            }
            None => recompute,
        };
        let mut block = KvBlockSummary::of(b);
        if let Some(loc) = b.location(tier) {
            block.size_bytes = self.format_bytes(loc.format);
        }
        BlockScoreInputs {
            block,
            tier,
            tier_capacity: capacity,
            tier_pressure: pressure,
            prefill_tps: self.prefill_tps,
            retrieval: CostEstimate { seconds: retrieval },
            session_hot: b
                .session
                .as_ref()
                .is_some_and(|s| self.sessions.is_hot(s, now)),
            depth_tokens: depth,
        }
    }

    fn eligible(
        &self,
        b: &KvBlock,
        tier: TierId,
        pool: &BlockPool,
        departing: &HashSet<KvKey>,
    ) -> bool {
        b.location(tier).is_some()
            && !self.demoting.contains_key(&b.key)
            && !self.transfer.is_busy_with(&b.key)
            && (tier != TierId::L0
                || b.location(TierId::L0)
                    .is_some_and(|l| pool.refcount(BlockId(l.slot as u32)) == 0))
            && self.dir.evictable(b, tier, departing)
    }

    /// Leaf-first victims of `tier`, lowest value first, at most `limit`. Choosing a block
    /// makes its parent eligible, so a cold chain drains in one pass.
    fn victims(&self, pool: &BlockPool, tier: TierId, limit: usize) -> Vec<(KvKey, f64)> {
        self.victims_where(pool, tier, limit, |_| true)
    }

    /// [`KvHierarchy::victims`] among the blocks `keep` accepts. L0 candidates come from the L0
    /// index (the blocks resident there), not a scan of the whole directory.
    fn victims_where(
        &self,
        pool: &BlockPool,
        tier: TierId,
        limit: usize,
        keep: impl Fn(&KvBlock) -> bool,
    ) -> Vec<(KvKey, f64)> {
        let now = self.now();
        let mut departing: HashSet<KvKey> = self.demoting.keys().copied().collect();
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<Victim>>, b: &KvBlock| {
            let inputs = self.score_one(b, tier, pool, now);
            heap.push(Reverse(Victim {
                value: self.policy.score(&inputs, now),
                last_access: b.last_access,
                depth: b.token_range.end,
                key: b.key,
            }));
        };
        if tier == TierId::L0 {
            for key in self.l0_keys.values() {
                if let Some(b) = self.dir.get(key)
                    && keep(b)
                    && self.eligible(b, tier, pool, &departing)
                {
                    push(&mut heap, b);
                }
            }
        } else {
            for b in self.dir.candidates(tier, &departing) {
                if keep(b) && self.eligible(b, tier, pool, &departing) {
                    push(&mut heap, b);
                }
            }
        }
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(Reverse(v)) = heap.pop() else { break };
            out.push((v.key, v.value));
            departing.insert(v.key);
            let parent = self.dir.get(&v.key).and_then(|b| b.parent);
            if let Some(pb) = parent.and_then(|p| self.dir.get(&p))
                && !departing.contains(&pb.key)
                && keep(pb)
                && self.eligible(pb, tier, pool, &departing)
            {
                push(&mut heap, pb);
            }
        }
        out
    }

    /// Mirrors L0 reference counts (owned by the pool) into `KvBlock.ref_count`.
    fn sync_l0_refs(&mut self, pool: &BlockPool) {
        for (block, key) in &self.l0_keys {
            if let Some(b) = self.dir.get_mut(key) {
                b.ref_count = pool.refcount(*block);
            }
        }
    }

    /// Publishes the cost-aware, leaf-first order in which `BlockPool::allocate` reclaims
    /// cached L0 blocks. Call when free L0 blocks fall below 10 % of the pool.
    pub fn refresh_reclaim_order(&mut self, pool: &mut BlockPool) {
        self.sync_l0_refs(pool);
        let ids = self
            .victims(pool, TierId::L0, usize::MAX)
            .iter()
            .filter_map(|(k, _)| {
                self.dir
                    .get(k)
                    .and_then(|b| b.location(TierId::L0))
                    .map(|l| BlockId(l.slot as u32))
            })
            .collect();
        pool.set_reclaim_order(ids);
    }

    /// After the scheduler allocated blocks: forget the L0 copies allocation reclaimed.
    pub fn after_plan(&mut self, pool: &mut BlockPool) {
        for b in pool.take_reclaimed() {
            let Some(key) = self.l0_keys.remove(&b) else {
                continue;
            };
            if let Some(e) = self.dir.get_mut(&key) {
                e.ref_count = 0;
            }
            if self.prefetch.evicted(&key) {
                self.metrics.prefetch_outcome(PrefetchOutcome::Wasted);
            }
            self.metrics.eviction(TierId::L0, EvictReason::Capacity);
            self.stats.evictions += 1;
            if self.dir.remove_location(&key, TierId::L0) == Some(0) {
                self.metrics.drop_block(EvictReason::Capacity);
                self.stats.drops += 1;
                self.forget(&key);
            }
        }
    }

    /// Carries out the reclaim requests the pressure controller stored in the handle (free
    /// first: dropping is cheap and may already meet the demote target); returns the bytes
    /// scheduled or freed. See [`KvHierarchy::pressure_reclaim`].
    pub fn apply_reclaim(&mut self, pool: &mut BlockPool) -> u64 {
        self.ladder_demand = [0; 2];
        self.queued_prefix_demand = 0;
        let mut bytes = 0;
        if let Some(t) = KvReclaimHandle::take(&self.reclaim.free_target) {
            bytes += self.pressure_reclaim(pool, t, true);
        }
        if let Some(t) = KvReclaimHandle::take(&self.reclaim.demote_target) {
            bytes += self.pressure_reclaim(pool, t, false);
        }
        bytes
    }

    /// Room for more L0 demotions under [`DEMOTION_INFLIGHT`].
    fn demotion_budget(&self) -> usize {
        DEMOTION_INFLIGHT.saturating_sub(self.l0_demotions_in_flight())
    }

    /// The Phase 3 controller's reclaim toward L0 utilisation `target` (its `demote` when
    /// `free` is false, `free_unreferenced` when true), lowest-value unreferenced blocks first
    /// (provisional decision "Phase 4: pressure reclaim copies only blocks with reuse evidence,
    /// bounded in flight"). The controller asks again on every tick while the state lasts, so
    /// the work per call is bounded:
    ///
    /// - a block with reuse evidence ([`has_reuse_evidence`]) scoring at least
    ///   `kv.demote_min_value` is copied down, while fewer than [`DEMOTION_INFLIGHT`] L0 copies
    ///   are in flight (each pins its L0 block until done, so the copy stream paces the rest);
    /// - any other block is dropped by `free`, exactly as without a lower tier, and left cached
    ///   by `demote` (an allocation takes it at no cost; below-`demote_min_value` blocks are
    ///   dropped by both, as before).
    ///
    /// Without a lower tier this is [`KvHierarchy::demote_to`]: every victim is dropped.
    pub fn pressure_reclaim(&mut self, pool: &mut BlockPool, target: f64, free: bool) -> u64 {
        let Some(to) = self.l0_demotion_target() else {
            return self.demote_to(pool, target, EvictReason::Pressure);
        };
        let total = f64::from(pool.total_blocks());
        let leaving = self.l0_leaving() as f64;
        let need = (f64::from(pool.used_blocks()) - leaving - target * total)
            .ceil()
            .max(0.0) as usize;
        if need == 0 {
            return 0;
        }
        self.sync_l0_refs(pool);
        let victims = self.victims(pool, TierId::L0, need);
        // What no unreferenced block covers: at YELLOW and ORANGE the prefixes queued requests
        // hold may (decision "6b: after the held-prefix ledger fix", 1 B); RED and SURVIVAL keep
        // their own rules.
        if matches!(self.l0_state, PressureState::Yellow | PressureState::Orange) {
            self.queued_prefix_demand = self
                .queued_prefix_demand
                .max(need.saturating_sub(victims.len()));
        }
        // The ladder's demand: the copies this reclaim wants in the lower tiers and cannot
        // start this tick (recorded below).
        let mut wanted = 0;
        if self.ladder.is_some() {
            wanted = victims
                .iter()
                .filter(|(k, v)| {
                    *v >= self.cfg.demote_min_value
                        && self
                            .dir
                            .get(k)
                            .is_some_and(|b| has_reuse_evidence(b) && b.location(to).is_none())
                })
                .count();
        }
        let mut budget = self.demotion_budget();
        let mut room = 0;
        if budget > 0 {
            let copies = victims
                .iter()
                .filter(|(k, v)| {
                    *v >= self.cfg.demote_min_value
                        && self.dir.get(k).is_some_and(has_reuse_evidence)
                })
                .count();
            room = self.make_room(pool, to, copies.min(budget));
        }
        // Victims are leaf-first on the assumption that each chosen child departs: a block
        // whose child stays in L0 is kept.
        let mut departing: HashSet<KvKey> = self.demoting.keys().copied().collect();
        let bb = self.cfg.block_bytes;
        let mut bytes = 0;
        for (key, value) in victims {
            let Some(b) = self.dir.get(&key) else {
                continue;
            };
            if !self.dir.evictable(b, TierId::L0, &departing) {
                continue;
            }
            let evidence = has_reuse_evidence(b);
            let copied = b.location(to).is_some();
            if value < self.cfg.demote_min_value || (!evidence && free) {
                let why = if value < self.cfg.demote_min_value {
                    EvictReason::BelowMinValue
                } else {
                    EvictReason::Pressure
                };
                self.drop_everywhere(pool, &key, why);
                departing.insert(key);
                bytes += bb;
            } else if !evidence {
                // `demote` leaves a one-off block cached in L0.
            } else if copied {
                // Already copied down: free the L0 copy now.
                self.metrics.demotion(TierId::L0, to);
                self.stats.demotions += 1;
                self.remove_copy(pool, &key, TierId::L0, EvictReason::Pressure);
                departing.insert(key);
                bytes += bb;
            } else if budget > 0
                && room >= self.demotion_bytes(&key, TierId::L0, to)
                && self.submit_demotion(pool, key, TierId::L0, to)
            {
                room -= self.demotion_bytes(&key, TierId::L0, to);
                budget -= 1;
                departing.insert(key);
                bytes += bb;
                wanted = wanted.saturating_sub(1);
            }
        }
        if wanted > 0 {
            self.record_demand(to, wanted as u64);
        }
        self.copy_ahead(pool, to, budget.min(CAPACITY_BATCH));
        bytes
    }

    /// Bytes of the demotion copies in flight into `tier`.
    fn inflight_into(&self, tier: TierId) -> u64 {
        self.demoting
            .values()
            .filter(|d| d.to == tier)
            .map(|d| d.bytes)
            .sum()
    }

    /// Demotes (copy first, free on completion) the lowest-value unreferenced L0 blocks until
    /// L0 utilisation would be ≤ `target`. Blocks scoring below `kv.demote_min_value`, or with
    /// no lower tier, are dropped. Returns the bytes scheduled or freed.
    ///
    /// Under `capacity` (the orchestrator's continuous headroom keeping) only blocks with reuse
    /// evidence ([`has_reuse_evidence`]) are worth a copy (provisional decision "Phase 4:
    /// capacity demotion only for blocks with reuse evidence"): the others stay cached in L0
    /// until an allocation reclaims them, at no cost, and at most [`CAPACITY_BATCH`] blocks move
    /// per call, fewer when [`DEMOTION_INFLIGHT`] copies are already running. Under any other
    /// reason every unreferenced block is taken in value order, as the eviction policy ranks
    /// them (the offline simulator's reclaim); the Phase 3 controller's requests go through
    /// [`KvHierarchy::pressure_reclaim`].
    pub fn demote_to(&mut self, pool: &mut BlockPool, target: f64, reason: EvictReason) -> u64 {
        let total = f64::from(pool.total_blocks());
        let leaving = self.l0_leaving() as f64;
        let need = (f64::from(pool.used_blocks()) - leaving - target * total)
            .ceil()
            .max(0.0) as usize;
        if need == 0 {
            return 0;
        }
        self.sync_l0_refs(pool);
        let capacity = reason == EvictReason::Capacity;
        let victims = if capacity {
            let batch = need.min(CAPACITY_BATCH).min(self.demotion_budget());
            if batch == 0 {
                return 0;
            }
            self.victims_where(pool, TierId::L0, batch, has_reuse_evidence)
        } else {
            self.victims(pool, TierId::L0, need)
        };
        let to = self.l0_demotion_target();
        let mut room = to.map_or(0, |t| self.make_room(pool, t, victims.len()));
        let bb = self.cfg.block_bytes;
        let mut bytes = 0;
        let mut started = 0;
        for (key, value) in victims {
            let Some(to) = to.filter(|_| value >= self.cfg.demote_min_value) else {
                let why = if value < self.cfg.demote_min_value {
                    EvictReason::BelowMinValue
                } else {
                    EvictReason::NoRoom
                };
                self.drop_everywhere(pool, &key, why);
                bytes += bb;
                continue;
            };
            if self.dir.get(&key).is_some_and(|b| b.location(to).is_some()) {
                // Already copied down: free the L0 copy now.
                self.metrics.demotion(TierId::L0, to);
                self.stats.demotions += 1;
                self.remove_copy(pool, &key, TierId::L0, reason);
                bytes += bb;
                continue;
            }
            let units = self.demotion_bytes(&key, TierId::L0, to);
            if room == 0 {
                break;
            }
            if room >= units && self.submit_demotion(pool, key, TierId::L0, to) {
                room -= units;
                bytes += bb;
                started += 1;
            }
        }
        if capacity && let Some(to) = to {
            self.copy_ahead(pool, to, CAPACITY_BATCH.saturating_sub(started));
        }
        bytes
    }

    /// Where L0 victims go: L1, else L2 while storage accepts demotions, else nowhere.
    fn l0_demotion_target(&self) -> Option<TierId> {
        demotion_target(
            TierId::L0,
            self.l1.is_some(),
            self.l2.is_some() && self.storage_accepts_demotions(),
        )
    }

    /// Demotes the given unreferenced L0 copies (session TTL), leaves first; a block whose
    /// child stays in L0 is kept, and nothing is dropped when no tier has room.
    fn demote_keys(&mut self, pool: &mut BlockPool, keys: Vec<KvKey>, reason: EvictReason) {
        let Some(to) = self.l0_demotion_target() else {
            return;
        };
        self.sync_l0_refs(pool);
        let movable: Vec<KvKey> = keys
            .into_iter()
            .filter(|k| {
                !self.demoting.contains_key(k)
                    && !self.transfer.is_busy_with(k)
                    && self
                        .dir
                        .get(k)
                        .and_then(|b| b.location(TierId::L0))
                        .is_some_and(|l| pool.refcount(BlockId(l.slot as u32)) == 0)
            })
            .collect();
        let mut room = self.make_room(pool, to, movable.len());
        let mut departing: HashSet<KvKey> = self.demoting.keys().copied().collect();
        for k in movable.into_iter().rev() {
            if !self
                .dir
                .get(&k)
                .is_some_and(|b| self.dir.evictable(b, TierId::L0, &departing))
            {
                continue;
            }
            departing.insert(k);
            if self.dir.get(&k).is_some_and(|b| b.location(to).is_some()) {
                self.metrics.demotion(TierId::L0, to);
                self.stats.demotions += 1;
                self.remove_copy(pool, &k, TierId::L0, reason);
            } else {
                let units = self.demotion_bytes(&k, TierId::L0, to);
                if room >= units && self.submit_demotion(pool, k, TierId::L0, to) {
                    room -= units;
                }
            }
        }
    }

    /// Demotions to L2 pause while storage is degraded or at ORANGE pressure or above.
    fn storage_accepts_demotions(&self) -> bool {
        self.l2
            .as_ref()
            .is_some_and(|t| !t.degraded() && t.pressure() < PressureState::Orange)
    }

    fn drop_everywhere(&mut self, pool: &mut BlockPool, key: &KvKey, why: EvictReason) {
        let tiers: Vec<TierId> = self
            .dir
            .get(key)
            .map(|b| b.locations.iter().map(|l| l.tier).collect())
            .unwrap_or_default();
        for t in tiers {
            self.remove_copy(pool, key, t, why);
        }
    }

    /// Queues the copy `from` → `to`; the source copy is removed when it completes. An L0
    /// source block is referenced for the copy's lifetime.
    fn submit_demotion(
        &mut self,
        pool: &mut BlockPool,
        key: KvKey,
        from: TierId,
        to: TierId,
    ) -> bool {
        self.submit_copy(pool, key, from, to, false)
    }

    /// [`KvHierarchy::submit_demotion`]; with `keep_source` the source copy stays (a copy
    /// ahead).
    fn submit_copy(
        &mut self,
        pool: &mut BlockPool,
        key: KvKey,
        from: TierId,
        to: TierId,
        keep_source: bool,
    ) -> bool {
        let Some(src_slot) = self
            .dir
            .get(&key)
            .and_then(|b| b.location(from))
            .map(|l| l.slot)
        else {
            return false;
        };
        let path = TransferPath::between(from, to).expect("a demotion path exists");
        let Some(req) = self.copy_request(path, key, None, TransferPurpose::Demote, src_slot, 0)
        else {
            return false;
        };
        let bytes = req.codec.to_bytes;
        if self.transfer.submit(req).is_err() {
            return false;
        }
        if from == TierId::L0 {
            pool.incref(BlockId(src_slot as u32));
        }
        self.demoting.insert(
            key,
            Demoting {
                from,
                to,
                bytes,
                keep_source,
            },
        );
        true
    }

    /// Copy ahead (P6b S-8; decision "6b: shared-prefix eval never demotes the prefix to L1",
    /// A): while the pressure controller is GREEN or YELLOW, when the reclaim would demote, the
    /// blocks of shared prefixes that the leaf-first rule holds in L0 — unreferenced blocks with
    /// two or more children (or an ancestor of one), reuse evidence, a child resident in L0 and
    /// the L0 copy as their only one — are copied into `to` while their L0 copy stays, lowest
    /// value first (the policy's score; the likeliest to leave next), at most `limit` per call
    /// within [`DEMOTION_INFLIGHT`] and only into `to`'s free room up to [`COPY_AHEAD_MAX_FILL`]
    /// of its capacity (nothing is evicted for them). A linear re-used chain (a session's
    /// history) is left to the leaf-first demotion. The L0 copy then leaves by the normal rules
    /// (once its children have left, or under higher pressure) at no copy cost: every reclaim
    /// path frees a copy that already has a lower one. Into a lossy tier the copy is one more
    /// location of the exact entry (S-3). From ORANGE on nothing is copied ahead: freeing L0
    /// comes first, and the copies in flight it needs are the same [`DEMOTION_INFLIGHT`].
    /// Returns the copies started.
    fn copy_ahead(&mut self, pool: &mut BlockPool, to: TierId, limit: usize) -> usize {
        if self.l0_state >= PressureState::Orange {
            return 0;
        }
        let limit = limit.min(self.demotion_budget());
        let Some(tier) = self.tier(to).cloned() else {
            return 0;
        };
        if limit == 0 {
            return 0;
        }
        let ceiling = (tier.capacity_bytes() as f64 * COPY_AHEAD_MAX_FILL) as u64;
        let mut used = tier.used_bytes() + self.inflight_into(to);
        if used >= ceiling {
            return 0;
        }
        // Shared prefixes: L0 blocks with two or more children, and their ancestors.
        let mut shared: HashSet<KvKey> = HashSet::new();
        for key in self.l0_keys.values() {
            let mut next = self
                .dir
                .get(key)
                .filter(|b| b.child_count >= 2)
                .map(|b| b.key);
            while let Some(k) = next.take() {
                if !shared.insert(k) {
                    break;
                }
                next = self.dir.get(&k).and_then(|b| b.parent);
            }
        }
        let now = self.now();
        let mut order: Vec<Victim> = self
            .l0_keys
            .iter()
            .filter_map(|(block, key)| {
                let b = self.dir.get(key)?;
                let qualifies = pool.refcount(*block) == 0
                    && b.locations.len() == 1
                    && shared.contains(key)
                    && has_reuse_evidence(b)
                    && !self.demoting.contains_key(key)
                    && !self.transfer.is_busy_with(key)
                    && self.dir.has_child_in(key, TierId::L0);
                qualifies.then(|| Victim {
                    value: self
                        .policy
                        .score(&self.score_one(b, TierId::L0, pool, now), now),
                    last_access: b.last_access,
                    depth: b.token_range.end,
                    key: *key,
                })
            })
            .filter(|v| v.value >= self.cfg.demote_min_value)
            .collect();
        order.sort();
        let mut started = 0;
        for v in order {
            if started == limit {
                break;
            }
            let bytes = self
                .copy_codec(&v.key, TierId::L0, to)
                .map_or(self.cfg.block_bytes, |c| c.to_bytes);
            if used + bytes > ceiling {
                break;
            }
            if self.submit_copy(pool, v.key, TierId::L0, to, true) {
                used += bytes;
                started += 1;
            }
        }
        if started > 0 {
            tracing::debug!(
                event = "kv_copy_ahead",
                reason = "copy_ahead",
                to = to.as_str(),
                blocks = started,
                "shared-prefix KV blocks copied down while their L0 copies stay"
            );
        }
        started
    }

    /// Makes room for up to `n` more blocks of `tier`'s format in `tier` and returns how many
    /// fit now (a lossless-tail block takes [`KvHierarchy::demotion_bytes`] of them). L1
    /// victims without an L2 copy move to L2 while storage accepts them (room appears when that
    /// copy completes); other victims are evicted now.
    /// Room in bytes for more demotions into `tier`: the free bytes, plus what spilling or
    /// evicting up to `n` victims frees (a victim is only touched while the tier is short of
    /// `n` demotions at the rung's size). Counts the bytes copies actually take
    /// ([`KvHierarchy::copy_codec`], the tier's format when the rung has no slot), so the
    /// callers' `demotion_bytes` accounting stays exact.
    fn make_room(&mut self, pool: &mut BlockPool, tier: TierId, n: usize) -> u64 {
        let Some(t) = self.tier(tier).cloned() else {
            return 0;
        };
        let unit = self.format_bytes(self.rung(tier)).max(1);
        let room = t
            .capacity_bytes()
            .saturating_sub(t.used_bytes() + self.inflight_into(tier));
        if room >= n as u64 * unit {
            return room;
        }
        // Copies already leaving `tier` (spills in flight) free their room when they complete:
        // count it, or every call before they land would spill further victims.
        let leaving: u64 = self
            .demoting
            .iter()
            .filter(|(_, d)| d.from == tier)
            .filter_map(|(k, _)| self.dir.get(k)?.location(tier))
            .map(|l| self.format_bytes(l.format))
            .sum();
        let pending = (leaving / unit) as usize;
        let spill = tier == TierId::L1 && self.storage_accepts_demotions();
        let mut room = room;
        for (victim, _) in self.victims(
            pool,
            tier,
            n.saturating_sub((room / unit) as usize + pending),
        ) {
            let has_l2 = self
                .dir
                .get(&victim)
                .is_some_and(|b| b.location(TierId::L2).is_some());
            let cost = self.demotion_bytes(&victim, TierId::L1, TierId::L2);
            let want = cost.div_ceil(self.format_bytes(self.rung(TierId::L2)).max(1)) as usize;
            if spill
                && !has_l2
                && self.make_room(pool, TierId::L2, want) >= cost
                && self.submit_demotion(pool, victim, TierId::L1, TierId::L2)
            {
                continue;
            }
            // A removed copy frees its stored bytes: the victim's real size, not a rung-unit
            // count. Read before the removal.
            let freed = self
                .dir
                .get(&victim)
                .and_then(|b| b.location(tier))
                .map_or(1, |l| self.format_bytes(l.format).max(1));
            match self.ladder_victim(pool, &victim, tier) {
                Some(false) => continue,
                Some(true) => self.remove_copy(pool, &victim, tier, EvictReason::LadderFloor),
                None => self.remove_copy(pool, &victim, tier, EvictReason::Capacity),
            }
            room += freed;
        }
        room
    }

    /// Adds `blocks` demotions into `to` to the ladder's demand; what `to` (L1) has no free room
    /// for spills on into L2.
    fn record_demand(&mut self, to: TierId, blocks: u64) {
        let bytes = blocks * self.format_bytes(self.rung(to));
        match to {
            TierId::L1 => {
                self.ladder_demand[0] += bytes;
                if let (Some(l1), true) = (self.l1.as_ref(), self.l2.is_some()) {
                    let free = l1
                        .capacity_bytes()
                        .saturating_sub(l1.used_bytes() + self.inflight_into(TierId::L1));
                    let over = bytes.saturating_sub(free);
                    let unit = self.format_bytes(self.rung(TierId::L1)).max(1);
                    self.ladder_demand[1] +=
                        over.div_ceil(unit) * self.format_bytes(self.rung(TierId::L2));
                }
            }
            TierId::L2 => self.ladder_demand[1] += bytes,
            _ => {}
        }
    }

    /// `tier`'s fill once the copies in flight into it land and its rewrites complete.
    fn tier_fill(&self, tier: TierId) -> f64 {
        if tier == TierId::L0 {
            let cap = (u64::from(self.l0_blocks) * self.cfg.block_bytes).max(1);
            return self.l0_used_bytes() as f64 / cap as f64;
        }
        let Some(t) = self.tier(tier) else {
            return 0.0;
        };
        let saved: u64 = self
            .compressing
            .values()
            .filter(|c| c.tier == tier)
            .map(|c| c.saved)
            .sum();
        let used = (t.used_bytes() + self.inflight_into(tier)).saturating_sub(saved);
        used as f64 / t.capacity_bytes().max(1) as f64
    }

    /// Bytes of the L0 copies the directory holds, by format (`tier_fill(TierId::L0)`).
    fn l0_used_bytes(&self) -> u64 {
        self.dir
            .iter()
            .filter_map(|b| b.location(TierId::L0))
            .map(|l| self.format_bytes(l.format))
            .sum()
    }

    /// The dtype-aware ladder rank of `format` ([`codec::tier_rung`]).
    fn rung_rank(&self, format: &str) -> usize {
        let dtype = self
            .ladder
            .as_ref()
            .map_or(self.cfg.l0_dtype, |l| l.cfg.l0_dtype);
        codec::tier_rung(format, dtype).unwrap_or(0)
    }

    /// The newest `kv.recent_window_blocks` full blocks of every live request (the recent
    /// window, S-5): the ladder's L0 sweep never rewrites them, and a block that leaves the
    /// window of its request is recompressed into the L0 base format.
    fn window_keys(&self) -> HashSet<KvKey> {
        let n = self.cfg.recent_window_blocks as usize;
        let mut out = HashSet::new();
        if n == 0 {
            return out;
        }
        for r in self.requests.values() {
            for k in &r.keys[r.keys.len().saturating_sub(n)..] {
                out.insert(*k);
            }
        }
        out
    }

    /// The recent window's exits (S-5): every L0 block that still holds its BF16 window-class
    /// pages, that sits outside every live sequence's window and the lossless tail, and that
    /// nothing references is recompressed in place into the L0 base format through the
    /// ladder's L0 rewrite path (purpose `Compress`, path `L0ToL0`, reason `recent_window`),
    /// within the ladder's rewrite budget. A block a sequence still references keeps its BF16
    /// pages until it is released (never a live block): the window costs a live sequence at
    /// most `recent_window_blocks` extra base-format blocks.
    /// The ladder's facts for a copy in `format` of `tier` (`None` with the ladder off).
    fn ladder_context(
        &self,
        tier: TierId,
        format: &'static str,
        must_leave: bool,
        pressure: PressureState,
    ) -> Option<LadderContext> {
        let l = self.ladder.as_ref()?;
        let dtype = l.cfg.l0_dtype;
        let below = match tier {
            TierId::L0 => self
                .l1
                .as_ref()
                .map(|_| TierId::L1)
                .or_else(|| self.l2.as_ref().map(|_| TierId::L2)),
            TierId::L1 if self.l2.is_some() => Some(TierId::L2),
            _ => None,
        };
        let cap = self.tier(tier).map_or(1, |t| t.capacity_bytes()).max(1);
        let demand = match tier {
            TierId::L1 => self.ladder_demand[0],
            TierId::L2 => self.ladder_demand[1],
            _ => 0,
        };
        Some(LadderContext {
            tier,
            fill: self.tier_fill(tier),
            demand: demand as f64 / cap as f64,
            rung: self.rung(tier),
            pressure,
            format,
            must_leave,
            demote_to: below.map(|t| (t, self.rung(t))),
            // A tier compresses to a rung only once every enabled tier below has reached it:
            // for L0 that is the most precise rung among L1 and L2 (dtype-aware ranks, S-2).
            lower_rung: if tier == TierId::L0 {
                [
                    self.l1.as_ref().map(|_| self.rung(TierId::L1)),
                    self.l2.as_ref().map(|_| self.rung(TierId::L2)),
                ]
                .into_iter()
                .flatten()
                .map(|r| codec::tier_rung(r, dtype).unwrap_or(0))
                .min()
                .and_then(|m| {
                    codec::registry()
                        .iter()
                        .find(|c| codec::tier_rung(c.name(), dtype) == Some(m))
                        .map(|c| c.name())
                })
            } else {
                below.map(|t| self.rung(t))
            },
            ladder: Some(l.cfg.limits()),
        })
    }

    /// Room for another rewrite in this tick and in flight (the ladder's rewrites and, with
    /// a lossy L0 base format, the recent window's conversions — S-5, S-7).
    fn ladder_budget(&self) -> bool {
        let window_ok = self
            .ladder
            .as_ref()
            .is_none_or(|l| l.window_rewrites < LADDER_REWRITES_PER_TICK);
        window_ok && self.compressing.len() < LADDER_REWRITES_PER_TICK
    }

    /// The back-off's `room_epoch` of `tier`: the pool's snapshot for L0, the tier's for L1/L2.
    fn room_epoch_of(&self, tier: TierId) -> u64 {
        if tier == TierId::L0 {
            self.l0_room_epoch
        } else {
            self.tier(tier).map_or(0, |x| x.room_epoch())
        }
    }

    /// Whether rewrites into `tier` wait for room (a rewrite ended `Full` and the tier's
    /// `room_epoch` has not changed since).
    fn ladder_backed_off(&self, tier: TierId) -> bool {
        self.ladder
            .as_ref()
            .and_then(|l| l.tiers.iter().find(|t| t.tier == tier))
            .is_some_and(|t| t.backoff.is_some())
    }

    /// Backs off rewrites into `tier` after one submitted at `room_epoch` `epoch` found no room
    /// (`codec` its formats): counted once per back-off as
    /// `turbine_kv_ladder_actions_total{reason="no_room_backoff"}`.
    fn ladder_back_off(&mut self, tier: TierId, epoch: u64, codec: TransferCodec) {
        let Some(t) = self
            .ladder
            .as_mut()
            .and_then(|l| l.tiers.iter_mut().find(|t| t.tier == tier))
        else {
            return;
        };
        if t.backoff.is_some() {
            return;
        }
        t.backoff = Some(epoch);
        self.metrics
            .ladder_action(tier, codec.from, codec.to, LadderReason::NoRoomBackoff);
        tracing::debug!(
            event = "kv_ladder",
            tier = tier.as_str(),
            from = codec.from,
            to = codec.to,
            reason = LadderReason::NoRoomBackoff.as_str(),
            "no room for ladder rewrites; backing off until a slab frees"
        );
    }

    /// Ends the back-off of every tier whose `room_epoch` changed since it started.
    fn ladder_resume(&mut self) {
        let epochs: SmallVec<[(TierId, u64); 2]> = self
            .ladder
            .as_ref()
            .map(|l| {
                l.tiers
                    .iter()
                    .filter(|t| t.backoff.is_some())
                    .map(|t| (t.tier, self.room_epoch_of(t.tier)))
                    .collect()
            })
            .unwrap_or_default();
        for (tier, now) in epochs {
            if let Some(t) = self
                .ladder
                .as_mut()
                .and_then(|l| l.tiers.iter_mut().find(|t| t.tier == tier))
                && t.backoff.is_some_and(|e| e != now)
            {
                t.backoff = None;
                tracing::debug!(
                    event = "kv_ladder",
                    tier = tier.as_str(),
                    "room freed; ladder rewrites resume"
                );
            }
        }
    }

    /// Sets `tier`'s rung for new demotions and reports the change (`kv_ladder`, the rung gauge
    /// and, for a step up, `rung_step_up`).
    fn set_rung(&mut self, tier: TierId, to: &'static str, reason: LadderReason, now: Duration) {
        let fill = self.tier_fill(tier);
        let pressure = self.l0_state;
        let Some(t) = self
            .ladder
            .as_mut()
            .and_then(|l| l.tiers.iter_mut().find(|t| t.tier == tier))
        else {
            return;
        };
        let from = t.rung;
        if from == to {
            return;
        }
        t.rung = to;
        // A move down restarts the step-up dwell; a step up starts the next one now.
        t.below_since = (reason == LadderReason::RungStepUp).then_some(now);
        self.metrics.set_ladder_rung(tier, to);
        if reason == LadderReason::RungStepUp {
            self.metrics.ladder_action(tier, from, to, reason);
        }
        tracing::info!(
            event = "kv_ladder",
            tier = tier.as_str(),
            from,
            to,
            fill,
            pressure = pressure.as_str(),
            reason = reason.as_str(),
            "KV compression ladder rung changed"
        );
    }

    /// Starts the ladder rewrite of `key`'s copy in `tier` into codec `to`; moves the tier's rung
    /// down to `to` when it is lossier. False when the copy or the transfer queue is not there.
    fn submit_compress(
        &mut self,
        pool: &mut BlockPool,
        key: KvKey,
        tier: TierId,
        to: &'static str,
        reason: LadderReason,
    ) -> bool {
        let Some(loc) = self.dir.get(&key).and_then(|b| b.location(tier)) else {
            return false;
        };
        if tier != TierId::L0 && self.tier(tier).is_none_or(|t| t.degraded()) {
            return false;
        }
        // An L0 rewrite fills a page of the smaller class, allocated now so a class that
        // cannot grow is the back-off's trigger (`no_room`, S-7); the base page is freed at
        // completion. A lower tier's rewrite stores back into its own slot.
        let dst = if tier == TierId::L0 {
            match pool.allocate_in(to, 1) {
                Ok(mut b) => b.pop(),
                Err(e) => {
                    tracing::debug!(
                        event = "kv_ladder",
                        tier = tier.as_str(),
                        to,
                        error = %e,
                        "no room in the destination page class; backing off"
                    );
                    self.ladder_back_off(
                        tier,
                        self.l0_room_epoch,
                        TransferCodec {
                            from: loc.format,
                            from_bytes: self.format_bytes(loc.format),
                            to,
                            to_bytes: self.format_bytes(to),
                        },
                    );
                    return false;
                }
            }
        } else {
            None
        };
        let path = if tier == TierId::L0 {
            TransferPath::L0ToL0
        } else {
            TransferPath::between(tier, TierId::L0).expect("a lower tier has a path to L0")
        };
        let codec = TransferCodec {
            from: loc.format,
            from_bytes: self.format_bytes(loc.format),
            to,
            to_bytes: self.format_bytes(to),
        };
        let req = TransferRequest {
            path,
            key,
            bytes: codec.from_bytes.min(codec.to_bytes),
            owner: None,
            purpose: TransferPurpose::Compress,
            src_slot: loc.slot,
            dst_slot: dst.map_or(loc.slot, |b| u64::from(b.0)),
            codec,
        };
        if self.transfer.submit(req).is_err() {
            // The queue is full: the destination page goes back (a lower tier's rewrite has
            // no page of its own).
            if let Some(b) = dst {
                pool.release(&[b]);
            }
            return false;
        }
        self.compressing.insert(
            key,
            Compressing {
                tier,
                saved: codec.from_bytes.saturating_sub(codec.to_bytes),
                epoch: self.room_epoch_of(tier),
            },
        );
        if let Some(l) = self.ladder.as_mut() {
            l.window_rewrites += 1;
        }
        self.stats.compressions += 1;
        self.metrics.ladder_action(tier, loc.format, to, reason);
        let rung = self.rung(tier);
        let dtype = self.ladder.as_ref().map_or("bf16", |l| l.cfg.l0_dtype);
        if codec::tier_rung(to, dtype) > codec::tier_rung(rung, dtype) {
            let now = self.now();
            self.set_rung(tier, to, reason, now);
        }
        true
    }

    /// The ladder's say on a copy `make_room` must remove from `tier` (P6b S-6, `would_drop`):
    /// `Some(false)` when it is being rewritten one rung down instead (no room yet), `Some(true)`
    /// when it was evicted at the floor (`ladder_floor`), `None` to evict it as before.
    fn ladder_victim(&mut self, pool: &mut BlockPool, key: &KvKey, tier: TierId) -> Option<bool> {
        if self.l0_state == PressureState::Green {
            return None;
        }
        let b = self.dir.get(key)?;
        if b.ref_count > 0 {
            return None;
        }
        let format = b.location(tier)?.format;
        let ctx = self.ladder_context(tier, format, true, self.l0_state)?;
        let inputs = self.score_one(b, tier, pool, self.now());
        match self.policy.policy.action(&inputs, &ctx) {
            EvictAction::Compress { .. } if self.ladder_backed_off(tier) => {
                self.stats.ladder_backoff_skips += 1;
                None
            }
            EvictAction::Compress { to } => (self.ladder_budget()
                && self.submit_compress(pool, *key, tier, to, LadderReason::WouldDrop))
            .then_some(false),
            EvictAction::Drop => {
                self.metrics
                    .ladder_action(tier, format, LADDER_EVICT, LadderReason::FloorEvict);
                Some(true)
            }
            _ => None,
        }
    }

    /// One ladder tick (P6b S-6), at most every [`LADDER_TICK_INTERVAL`]: at GREEN, a tier that
    /// has stayed below low water for the dwell steps its rung for new demotions back up one
    /// rung (never above its configured format; the dwell runs whatever the pressure, but no
    /// rung steps up while the controller is not GREEN); then, while the pressure controller
    /// is not GREEN, the eviction policy decides on the copies of each enabled lower tier,
    /// lowest tier first — oldest, least-reusable and most precise first — and at most
    /// [`LADDER_REWRITES_PER_TICK`] rewrites start (fewer while earlier ones are in flight). A
    /// tier's sweep stops at the first copy the policy keeps. Copies of blocks a running
    /// request references, and copies being copied, are never rewritten. `tick` calls it.
    pub fn ladder_tick(&mut self, pool: &mut BlockPool, pressure: PressureState, now: Duration) {
        let Some(l) = self.ladder.as_mut() else {
            return;
        };
        if l.last_tick
            .is_some_and(|t| now.saturating_sub(t) < LADDER_TICK_INTERVAL)
        {
            return;
        }
        l.last_tick = Some(now);
        l.window_rewrites = 0;
        self.l0_room_epoch = pool.room_epoch();
        let (low, dwell) = (l.cfg.low_water, l.cfg.dwell);
        let dtype = l.cfg.l0_dtype;
        let tiers: SmallVec<[TierRung; 3]> = l.tiers.clone();
        self.stats.ladder_ticks += 1;
        // Hysteresis: one rung up once the controller is GREEN and the tier has been below low
        // water for the dwell (user decision 2026-09-30, option A: never while not GREEN, so a
        // sustained pressure does not step a floor tier up and compress it again every dwell).
        for t in &tiers {
            let below = self.tier_fill(t.tier) < low;
            let Some(tr) = self
                .ladder
                .as_mut()
                .and_then(|l| l.tiers.iter_mut().find(|x| x.tier == t.tier))
            else {
                continue;
            };
            if !below {
                tr.below_since = None;
                continue;
            }
            let since = *tr.below_since.get_or_insert(now);
            if pressure == PressureState::Green
                && now.saturating_sub(since) >= dwell
                && tr.rung != tr.base
            {
                let up = codec::prev_rung_in(tr.rung, dtype)
                    .filter(|u| codec::tier_rung(u, dtype) >= codec::tier_rung(tr.base, dtype))
                    .unwrap_or(tr.base);
                self.set_rung(t.tier, up, LadderReason::RungStepUp, now);
            }
        }
        if pressure == PressureState::Green {
            return;
        }
        self.sync_l0_refs(pool);
        self.ladder_resume();
        for t in tiers.iter().rev() {
            if !self.ladder_budget() {
                break;
            }
            self.ladder_sweep(pool, t.tier, pressure, now);
        }
    }

    /// The ladder's sweep of one tier ([`KvHierarchy::ladder_tick`]).
    fn ladder_sweep(
        &mut self,
        pool: &mut BlockPool,
        tier: TierId,
        pressure: PressureState,
        now: Duration,
    ) {
        let mut order: Vec<(usize, Victim)> = self
            .dir
            .iter()
            .filter(|b| {
                b.ref_count == 0
                    && !self.demoting.contains_key(&b.key)
                    && !self.transfer.is_busy_with(&b.key)
            })
            .filter_map(|b| {
                let loc = b.location(tier)?;
                // S-7: an L0 sweep never rewrites a block of the lossless tail or of a live
                // sequence's recent window.
                if tier == TierId::L0
                    && (self.tail.contains(&b.key) || self.window_keys().contains(&b.key))
                {
                    return None;
                }
                let inputs = self.score_one(b, tier, pool, now);
                Some((
                    self.rung_rank(loc.format),
                    Victim {
                        value: self.policy.score(&inputs, now),
                        last_access: b.last_access,
                        depth: b.token_range.end,
                        key: b.key,
                    },
                ))
            })
            .collect();
        order.sort_by(|(ra, a), (rb, b)| ra.cmp(rb).then_with(|| a.cmp(b)));
        let mut departing: HashSet<KvKey> = self.demoting.keys().copied().collect();
        for (_, v) in order {
            if !self.ladder_budget() {
                break;
            }
            let Some(b) = self.dir.get(&v.key) else {
                continue;
            };
            let Some(format) = b.location(tier).map(|l| l.format) else {
                continue;
            };
            let Some(ctx) = self.ladder_context(tier, format, false, pressure) else {
                return;
            };
            let inputs = self.score_one(b, tier, pool, now);
            match self.policy.policy.action(&inputs, &ctx) {
                // No room for the new format: the copy stays (floor evictions still run).
                EvictAction::Compress { .. } if self.ladder_backed_off(tier) => {
                    self.stats.ladder_backoff_skips += 1;
                }
                EvictAction::Compress { to } => {
                    if !self.submit_compress(pool, v.key, tier, to, LadderReason::FillHighWater) {
                        return;
                    }
                }
                EvictAction::Drop => {
                    if !self.dir.evictable(b, tier, &departing) {
                        continue;
                    }
                    self.metrics.ladder_action(
                        tier,
                        format,
                        LADDER_EVICT,
                        LadderReason::FloorEvict,
                    );
                    self.remove_copy(pool, &v.key, tier, EvictReason::LadderFloor);
                    departing.insert(v.key);
                }
                _ => return,
            }
        }
    }

    /// The rung new demotions into `tier` take (`None` when the ladder is off or the tier is
    /// not enabled).
    pub fn ladder_rung(&self, tier: TierId) -> Option<&'static str> {
        self.ladder
            .as_ref()?
            .tiers
            .iter()
            .find(|t| t.tier == tier)
            .map(|t| t.rung)
    }

    /// The keys with a ladder rewrite in flight and their tier.
    pub fn ladder_in_flight(&self) -> impl Iterator<Item = (&KvKey, TierId)> {
        self.compressing.iter().map(|(k, c)| (k, c.tier))
    }

    /// `reliability.pressure.deescalate_dwell`: how long a tier stays below low water before
    /// its ladder rung steps back up.
    pub fn set_ladder_dwell(&mut self, dwell: Duration) {
        if let Some(l) = self.ladder.as_mut() {
            l.cfg.dwell = dwell;
        }
    }

    pub fn window_maintain(&mut self, pool: &mut BlockPool) {
        let target = self.cfg.l0_dtype;
        if target == crate::tier::L0_FORMAT
            || target == "bf16"
            || self.cfg.recent_window_blocks == 0
        {
            return;
        }
        if self.ladder_backed_off(TierId::L0) {
            return;
        }
        self.sync_l0_refs(pool);
        self.l0_room_epoch = pool.room_epoch();
        let windows = self.window_keys();
        let mut candidates: Vec<KvKey> = self
            .dir
            .iter()
            .filter(|b| {
                b.ref_count == 0
                    && !self.tail.contains(&b.key)
                    && !self.demoting.contains_key(&b.key)
                    && !self.compressing.contains_key(&b.key)
                    && !self.transfer.is_busy_with(&b.key)
            })
            .filter_map(|b| {
                b.location(TierId::L0)
                    .filter(|l| l.format == "bf16")
                    .filter(|_| !windows.contains(&b.key))
                    .map(|_| b.key)
            })
            .collect();
        candidates.sort();
        candidates.dedup();
        for k in candidates {
            if !self.ladder_budget() {
                break;
            }
            self.submit_compress(pool, k, TierId::L0, target, LadderReason::RecentWindow);
        }
    }

    /// Session TTLs, expiry and predicted-resume prefetch    /// Session TTLs, expiry and predicted-resume prefetch; call once per iteration.
    pub fn tick(&mut self, pool: &mut BlockPool) {
        self.sync_l0_refs(pool);
        self.window_maintain(pool);
        let now = self.now();
        for action in self.sessions.sweep(now, self.l0_state) {
            match action {
                SessionAction::DemoteFromL0(keys) => {
                    self.demote_keys(pool, keys, EvictReason::SessionExpired);
                }
                SessionAction::DemoteFromL1(keys) => {
                    if !self.storage_accepts_demotions() {
                        continue;
                    }
                    for k in keys {
                        let movable = self.dir.get(&k).is_some_and(|b| {
                            b.location(TierId::L1).is_some()
                                && b.location(TierId::L2).is_none()
                                && b.location(TierId::L0).is_none()
                        });
                        let cost = self.demotion_bytes(&k, TierId::L1, TierId::L2);
                        let want = cost.div_ceil(self.format_bytes(self.rung(TierId::L2)).max(1));
                        if movable
                            && !self.demoting.contains_key(&k)
                            && !self.transfer.is_busy_with(&k)
                            && self.make_room(pool, TierId::L2, want as usize) >= cost
                        {
                            self.submit_demotion(pool, k, TierId::L1, TierId::L2);
                        }
                    }
                }
                SessionAction::Prefetch(_, keys) => {
                    // A predicted resume is best effort: a full queue only counts rejections.
                    let _ = self.prefetch_keys(pool, &keys);
                }
                SessionAction::Dropped(id) => {
                    tracing::debug!(event = "kv_session_expired", session = %id.key);
                }
            }
        }
        self.metrics.sessions.set(self.sessions.len() as i64);
        let state = self.l0_state;
        self.ladder_tick(pool, state, now);
    }

    /// `POST /turbine/v1/kv/prefetch`: promote the cached blocks of a session or a prompt.
    pub fn prefetch(
        &mut self,
        pool: &mut BlockPool,
        target: PrefetchTarget<'_>,
    ) -> Result<PrefetchAccepted, PrefetchError> {
        if self.l0_state >= PressureState::Orange {
            return Err(PrefetchError::PressureTooHigh);
        }
        let keys = match target {
            PrefetchTarget::Session(id) => self
                .sessions
                .find_by_key(id)
                .ok_or(PrefetchError::SessionNotFound)?
                .blocks
                .clone(),
            PrefetchTarget::Tokens { prompt, cache_salt } => {
                let hasher = Blake3Hasher(self.namespaces.get(cache_salt));
                prefix_keys(&hasher, prompt, self.cfg.block_tokens)
            }
        };
        self.prefetch_keys(pool, &keys)
    }

    fn prefetch_keys(
        &mut self,
        pool: &mut BlockPool,
        keys: &[KvKey],
    ) -> Result<PrefetchAccepted, PrefetchError> {
        let mut acc = PrefetchAccepted {
            blocks_queued: 0,
            blocks_resident: 0,
        };
        for key in keys {
            let Some(b) = self.dir.get(key) else { break };
            if b.location(TierId::L0).is_some() {
                acc.blocks_resident += 1;
                continue;
            }
            // A lossy copy promoted earlier sits in L0 under its lossy key (P6b S-3).
            if let Some((lk, _)) = b.fastest().and_then(|t| {
                let loc = b.location(t)?;
                self.lossy_target(key, loc)
            }) && self
                .dir
                .get(&lk)
                .is_some_and(|l| l.location(TierId::L0).is_some())
            {
                acc.blocks_resident += 1;
                continue;
            }
            if self.transfer.is_busy_with(key) {
                continue;
            }
            if self.prefetch_inflight.len() >= self.cfg.prefetch.max_queue as usize {
                self.prefetch.reject(1);
                self.metrics.prefetch_outcome(PrefetchOutcome::Rejected);
                if acc.blocks_queued == 0 {
                    return Err(PrefetchError::QueueFull);
                }
                break;
            }
            let Some(from) = b.fastest() else { break };
            let loc = b.location(from).expect("the fastest tier has a copy");
            let (src_slot, target) = (loc.slot, self.lossy_target(key, loc));
            let Ok(ids) = pool.allocate(1) else { break };
            let req = self
                .copy_request(
                    TransferPath::between(from, TierId::L0).expect("lower tier to L0"),
                    *key,
                    None,
                    TransferPurpose::Prefetch,
                    src_slot,
                    u64::from(ids[0].0),
                )
                .expect("the fastest tier has a copy");
            match self.transfer.submit(req) {
                Ok(t) => {
                    if let Some(target) = target {
                        self.lossy_targets.insert(t.id, target);
                    }
                    self.prefetch_inflight.insert(t.id, ids[0]);
                    acc.blocks_queued += 1;
                }
                Err(_) => {
                    pool.release(&ids);
                    break;
                }
            }
        }
        self.after_plan(pool);
        tracing::info!(
            event = "kv_prefetch",
            queued = acc.blocks_queued,
            resident = acc.blocks_resident
        );
        Ok(acc)
    }

    /// Publishes tier gauges and the reclaim handle's snapshot.
    fn publish(&self, pool: &BlockPool) {
        let bb = self.cfg.block_bytes;
        self.metrics.tier_usage(
            TierId::L0,
            u64::from(pool.total_blocks()) * bb,
            u64::from(pool.used_blocks()) * bb,
            u64::from(pool.used_blocks()),
            u64::from(pool.free_blocks()),
        );
        for t in [TierId::L1, TierId::L2] {
            match self.tier(t) {
                Some(x) => {
                    let used = x.used_bytes() / bb.max(1);
                    let cap = x.capacity_bytes() / bb.max(1);
                    self.metrics.tier_usage(
                        t,
                        x.capacity_bytes(),
                        x.used_bytes(),
                        used,
                        cap.saturating_sub(used),
                    );
                }
                None => self.metrics.tier_usage(t, 0, 0, 0, 0),
            }
        }
        self.reclaim
            .l0_capacity_bytes
            .store(u64::from(pool.total_blocks()) * bb, Ordering::SeqCst);
        self.reclaim
            .l0_used_bytes
            .store(u64::from(pool.used_blocks()) * bb, Ordering::SeqCst);
        self.reclaim
            .reclaimable_bytes
            .store(u64::from(pool.cached_unreferenced()) * bb, Ordering::SeqCst);
    }

    /// Body of `GET /turbine/v1/kv` (P4 §Data plus the P2 per-tier fields; CONFLICT C-18
    /// `transfers`). `hit_window` is (prompt tokens, cached tokens) over the last 300 s.
    pub fn document(&self, pool: &BlockPool, hit_window: (u64, u64)) -> KvDocument {
        let bb = self.cfg.block_bytes;
        let dtype = self.namespaces.format().layout.dtype.as_str();
        let blocks = |bytes: u64| u32::try_from(bytes / bb.max(1)).unwrap_or(u32::MAX);
        // Copies per tier and codec (the codec's encoded size per copy).
        let mut formats: HashMap<TierId, BTreeMap<&'static str, FormatUsage>> = HashMap::new();
        for b in self.dir.iter() {
            for l in &b.locations {
                let u = formats
                    .entry(l.tier)
                    .or_default()
                    .entry(l.format)
                    .or_default();
                u.blocks += 1;
                u.bytes += self.format_bytes(l.format);
            }
        }
        let mut tiers = vec![KvTierDocument {
            tier: TierId::L0.as_str(),
            dtype,
            block_tokens: self.cfg.block_tokens,
            block_bytes: bb,
            blocks_total: pool.total_blocks(),
            blocks_used: pool.used_blocks(),
            blocks_free: pool.free_blocks(),
            state: Some(TierState {
                enabled: true,
                capacity_bytes: u64::from(pool.total_blocks()) * bb,
                used_bytes: u64::from(pool.used_blocks()) * bb,
                blocks: u64::from(pool.used_blocks()),
                referenced_blocks: u64::from(pool.referenced_blocks()),
                pressure: self.l0_state,
                degraded: false,
                est_latency_seconds: 0.0,
                est_bandwidth_bytes_per_second: None,
            }),
            formats: formats.remove(&TierId::L0).unwrap_or_default(),
        }];
        for id in [TierId::L1, TierId::L2] {
            let t = self.tier(id);
            let (cap, used) = t.map_or((0, 0), |t| (t.capacity_bytes(), t.used_bytes()));
            tiers.push(KvTierDocument {
                tier: id.as_str(),
                dtype,
                block_tokens: self.cfg.block_tokens,
                block_bytes: bb,
                blocks_total: blocks(cap),
                blocks_used: blocks(used),
                blocks_free: blocks(cap.saturating_sub(used)),
                state: Some(TierState {
                    enabled: t.is_some(),
                    capacity_bytes: cap,
                    used_bytes: used,
                    blocks: u64::from(blocks(used)),
                    referenced_blocks: 0,
                    pressure: t.map_or(PressureState::Green, |t| t.pressure()),
                    degraded: t.is_some_and(|t| t.degraded()),
                    est_latency_seconds: t.map_or(0.0, |t| t.est_latency().as_secs_f64()),
                    est_bandwidth_bytes_per_second: t.and_then(|t| t.est_bandwidth()),
                }),
                formats: formats.remove(&id).unwrap_or_default(),
            });
        }
        KvDocument {
            tiers,
            summary: Some(KvSummary {
                policy: self.policy.name(),
                prefix_sharing: self.cfg.prefix_sharing,
                block_tokens: self.cfg.block_tokens,
                block_bytes: bb,
                unified_memory: self.cfg.memory_kind == MemoryKind::Unified,
                hit_rate: HitRate {
                    window_seconds: crate::document::HitWindow::WINDOW_SECONDS,
                    prompt_tokens: hit_window.0,
                    cached_tokens: hit_window.1,
                },
                sessions: Sessions {
                    active: self.sessions.len() as u64,
                    max: u64::from(self.sessions.max()),
                },
                prefetch: Prefetch {
                    queued: self.prefetch_inflight.len() as u64,
                    used: self.prefetch.used,
                    wasted: self.prefetch.wasted,
                    cancelled: self.prefetch.cancelled,
                },
                transfers: Transfers {
                    inflight_bytes: self.transfer.inflight_bytes(),
                    max_inflight_bytes: self.transfer.max_inflight_bytes(),
                },
            }),
        }
    }
}

/// The plan of an attach cut short after the blocks of `used` (a failed allocation or a full
/// transfer queue): reuse and promotion counts of those blocks, the rest recomputed.
fn truncated_plan(
    used: &[TierId],
    prompt_tokens: u32,
    block_tokens: u32,
    reason: PlanReason,
) -> KvPlan {
    let mut promote: Vec<(TierId, u32)> = Vec::new();
    for t in used.iter().filter(|t| **t != TierId::L0) {
        match promote.iter_mut().find(|(x, _)| x == t) {
            Some(e) => e.1 += 1,
            None => promote.push((*t, 1)),
        }
    }
    KvPlan {
        reuse_l0: used.iter().filter(|t| **t == TierId::L0).count() as u32,
        promote,
        recompute_tokens: prompt_tokens - used.len() as u32 * block_tokens,
        reason,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use turbine_core::clock::FakeClock;
    use turbine_core::types::DeviceId;
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::directory::tests::fmt16;
    use crate::pool::BlockPoolConfig;
    use crate::tier::MemTier;
    use crate::transfer::SimTransferBackend;

    pub(crate) struct Rig {
        pub h: KvHierarchy,
        pub pool: BlockPool,
        pub backend: SimTransferBackend,
        pub clock: FakeClock,
    }

    /// A hierarchy over an `l0_blocks` pool and `MemTier` L1/L2 of the given sizes (0 = none).
    pub(crate) fn rig(
        l0_blocks: u32,
        l1: Option<Arc<MemTier>>,
        l2: Option<Arc<MemTier>>,
        clock: FakeClock,
    ) -> Rig {
        rig_with("cost_aware", l0_blocks, l1, l2, clock)
    }

    /// `rig` under the eviction policy registered as `policy`.
    pub(crate) fn rig_with(
        policy: &str,
        l0_blocks: u32,
        l1: Option<Arc<MemTier>>,
        l2: Option<Arc<MemTier>>,
        clock: FakeClock,
    ) -> Rig {
        rig_kv(policy, l0_blocks, l1, l2, clock, |_| {})
    }

    /// `rig_with` over the `kv` section `edit` adjusts.
    pub(crate) fn rig_kv(
        policy: &str,
        l0_blocks: u32,
        l1: Option<Arc<MemTier>>,
        l2: Option<Arc<MemTier>>,
        clock: FakeClock,
        edit: impl FnOnce(&mut KvConfig),
    ) -> Rig {
        let l1 = l1.map(|t| t as Arc<dyn KvTier>);
        let l2 = l2.map(|t| t as Arc<dyn KvTier>);
        rig_tiers(policy, l0_blocks, l1, l2, clock, edit)
    }

    /// `rig_kv` over any L1/L2 tiers (the slab tiers, for slot-size behaviour).
    pub(crate) fn rig_tiers(
        policy: &str,
        l0_blocks: u32,
        l1: Option<Arc<dyn KvTier>>,
        l2: Option<Arc<dyn KvTier>>,
        clock: FakeClock,
        edit: impl FnOnce(&mut KvConfig),
    ) -> Rig {
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let fmt = fmt16();
        let bb = fmt.layout.block_bytes();
        // The rig pages at 16 tokens (fmt16), not the 128-token default.
        let mut kv = KvConfig {
            block_tokens: fmt.layout.block_tokens,
            policy: turbine_core::config::ModuleName::new(policy).expect("a module name"),
            ..KvConfig::default()
        };
        edit(&mut kv);
        let cfg = HierarchyConfig::from_config(&kv, bb, MemoryKind::Dedicated)
            .expect("a registered eviction policy");
        let model = ModelIdentity {
            config_hash: [1; 32],
            weights_index_hash: [2; 32],
            rope_hash: [0; 32],
        };
        let h = KvHierarchy::new(
            cfg,
            model,
            fmt,
            l0_blocks,
            l1.clone(),
            l2.clone(),
            arc.clone(),
            KvMetrics::unregistered(),
        );
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 24);
        let pool = BlockPool::new(
            BlockPoolConfig {
                layout: fmt.layout,
                num_blocks: l0_blocks,
            },
            mem,
        )
        .expect("the pool fits the host memory");
        let backend = SimTransferBackend::new(arc, l1, l2, bb as usize);
        Rig {
            h,
            pool,
            backend,
            clock,
        }
    }

    pub(crate) fn attach(r: &mut Rig, id: RequestId, prompt: &[u32]) -> AttachOutcome {
        let req = AttachRequest {
            request: id,
            prompt,
            cache_salt: "",
            session: None,
            priority: Priority(0),
            allow_lossy: None,
        };
        r.h.attach_prefix(&mut r.pool, &req)
    }

    /// Prefill of an attached request: allocate the rest of the prompt, commit, finish.
    pub(crate) fn prefill_and_finish(
        r: &mut Rig,
        id: RequestId,
        prompt: &[u32],
        attach: &PrefixAttach,
    ) {
        let bt = r.pool.layout().block_tokens as usize;
        let need = prompt.len().div_ceil(bt) - attach.blocks.len();
        let mut table: Vec<BlockId> = attach.blocks.to_vec();
        table.extend(r.pool.allocate(need as u32).expect("room for the prompt"));
        r.h.commit_progress(&mut r.pool, id, &table, prompt);
        r.pool.release(&table);
        r.h.request_done(&mut r.pool, id, false);
    }

    fn run(r: &mut Rig, prompt: &[u32]) -> PrefixAttach {
        let id = RequestId::new_v4();
        let a = match attach(r, id, prompt) {
            AttachOutcome::Ready(a) => a,
            other => panic!("expected Ready, got {other:?}"),
        };
        prefill_and_finish(r, id, prompt, &a);
        a
    }

    #[test]
    fn demote_promote_round_trip() {
        round_trip("cost_aware");
    }

    /// The mechanism (leaf-first chain, copy then free, promotion before reuse) holds whatever
    /// the registered eviction policy values: part of the `eviction_policy` conformance
    /// (`docs/extending/eviction-policy.md`). Breaks when a policy's scores make the hierarchy
    /// skip, repeat or free a block early.
    #[test]
    fn round_trip_under_every_policy() {
        for policy in crate::policy::registry().names() {
            round_trip(policy);
        }
    }

    fn round_trip(policy: &str) {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig_with(policy, 8, Some(l1.clone()), None, clock);
        let prompt: Vec<u32> = (0..66).collect();

        let a = run(&mut r, &prompt);
        assert_eq!(a.cached_tokens, 0);
        assert_eq!(a.plan.reason, PlanReason::NoMatch);
        assert_eq!(r.pool.cached_unreferenced(), 4, "4 full blocks stay cached");
        let b = run(&mut r, &prompt);
        assert_eq!((b.cached_tokens, b.plan.reason), (64, PlanReason::AllL0));

        // One call schedules the whole leaf-first chain; nothing is freed before its copy.
        assert_eq!(
            r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure),
            4 * bb
        );
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(
            r.pool.used_blocks(),
            4,
            "not freed before the copies complete"
        );
        assert_eq!(r.h.stats().demotions, 0);
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(r.pool.used_blocks(), 0);
        assert_eq!(r.h.stats().demotions, 4);
        assert_eq!(l1.len(), 4);

        let id = RequestId::new_v4();
        assert_eq!(attach(&mut r, id, &prompt), AttachOutcome::Promoting);
        assert!(r.h.poll(&mut r.pool, &mut r.backend).is_empty());
        r.clock.advance(Duration::from_millis(10));
        let ready = r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, id);
        assert_eq!(ready[0].1.cached_tokens, 64);
        assert_eq!(ready[0].1.plan.reason, PlanReason::RetrieveCheaper);
        assert_eq!(ready[0].1.plan.promote, vec![(TierId::L1, 4)]);
        assert_eq!(r.h.stats().promotions, 4);
        // The request owns exactly the four promoted blocks.
        assert_eq!(r.pool.referenced_blocks(), 4);
        r.pool.release(&ready[0].1.blocks);
        r.h.request_done(&mut r.pool, id, false);
        assert_eq!(r.pool.referenced_blocks(), 0);

        let doc = serde_json::to_value(r.h.document(&r.pool, (130, 64))).unwrap();
        assert_eq!(doc["policy"], policy);
        assert_eq!(doc["tiers"][0]["tier"], "l0");
        assert_eq!(doc["tiers"][0]["blocks_total"], 8);
        assert_eq!(doc["tiers"][0]["referenced_blocks"], 0);
        assert_eq!(doc["tiers"][1]["tier"], "l1");
        assert_eq!(doc["tiers"][1]["blocks"], 4);
        assert_eq!(doc["tiers"][2]["enabled"], false);
        assert_eq!(doc["hit_rate"]["cached_tokens"], 64);
        assert_eq!(doc["transfers"]["inflight_bytes"], 0);
        assert_eq!(doc["unified_memory"], false);
    }

    /// P6b S-6 (user decision "6b Task 16: ladder proof results — four open points", 3 A): once
    /// a ladder rewrite into a tier ends `Full`, the ladder submits no rewrite into that tier —
    /// not every tick — until the tier's `room_epoch` changes (a slab freed), counts the back-off
    /// once (`no_room_backoff`) and every skipped rewrite, and then resumes. Breaks if the
    /// sweep resubmits while no room freed, or never resumes after it did.
    #[test]
    fn ladder_backs_off_a_tier_without_room_until_a_slab_frees() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        // L1 is the lowest tier and full: the four demoted blocks and one block of its own.
        let l1 = Arc::new(MemTier::new(TierId::L1, 5 * bb, arc));
        let mut r = rig_kv("cost_aware", 8, Some(l1.clone()), None, clock, |kv| {
            kv.ladder.enabled = true;
            kv.ladder.l0 = false;
        });
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        run(&mut r, &prompt);
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        r.h.poll(&mut r.pool, &mut r.backend);
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(l1.len(), 4);
        let other = KvKey([99; 16]);
        l1.put(
            other,
            crate::tier::TierBlockRef::Host(&vec![0u8; bb as usize]),
        )
        .unwrap();
        l1.inject_full(u32::MAX);
        let tick = |r: &mut Rig| {
            r.clock.advance(LADDER_TICK_INTERVAL);
            let now = r.clock.now_mono();
            r.h.ladder_tick(&mut r.pool, PressureState::Orange, now);
            // Every rewrite completes before the next tick.
            for _ in 0..2 {
                r.clock.advance(Duration::from_secs(1));
                r.h.poll(&mut r.pool, &mut r.backend);
            }
            assert_eq!(r.h.ladder_in_flight().count(), 0);
        };
        tick(&mut r);
        let first = r.h.stats().compressions;
        assert_eq!(first, 4, "every copy is rewritten one rung down");
        assert_eq!(l1.len(), 5, "each failed rewrite keeps its copy");
        for _ in 0..5 {
            tick(&mut r);
        }
        assert_eq!(
            r.h.stats().compressions,
            first,
            "no resubmission without room"
        );
        assert_eq!(r.h.stats().ladder_backoff_skips, 5 * 4);
        let backoffs =
            r.h.metrics
                .ladder_actions
                .get_or_create(&[
                    ("tier", "l1"),
                    ("from", crate::tier::L0_FORMAT),
                    ("to", "fp8_e4m3"),
                    ("reason", LadderReason::NoRoomBackoff.as_str()),
                ])
                .get();
        assert_eq!(backoffs, 1, "one back-off, however many rewrites failed");
        // A copy leaves the tier: room may have freed, the ladder tries again.
        l1.evict(&other).unwrap();
        tick(&mut r);
        assert_eq!(r.h.stats().compressions, first + 4, "rewrites resume");
        tick(&mut r);
        assert_eq!(
            r.h.stats().compressions,
            first + 4,
            "and back off again on the next Full"
        );
    }

    /// The Task 16 multi-turn regression (perf log "Compression ladder in L1/L2 on the server",
    /// ladder on: L0 → L1 demotions halved, ORANGE five times as long): once L1's rung stepped
    /// to `fp8_e4m3`, every new demotion into it needed an `fp8_e4m3` slot, but both slabs held
    /// `l0` slots and never emptied, so each copy ended `Full` and its L0 block stayed although
    /// the tier had free `l0` slots. A new demotion takes the tier's rung only while the tier
    /// has a slot of that size; else it is stored at the tier's own format (never lossier than
    /// the rung). Breaks if a demotion into a tier with free slots fails for want of a slot size.
    #[test]
    fn a_demotion_without_a_slot_of_the_rung_stores_at_the_tier_format() {
        use crate::tier::{L1Config, L1PinnedTier, TierBlockRef};
        use turbine_tensor::host::HostPinned;
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(L1PinnedTier::new(
            L1Config {
                enabled: true,
                max_bytes: 8 * bb,
                slab_bytes: 4 * bb,
                block_bytes: bb,
                memory_kind: MemoryKind::Dedicated,
            },
            Arc::new(HostPinned::new(u64::MAX)),
            arc,
        ));
        // Both slabs hold `l0` slots and neither is empty: five other copies, three slots free.
        for i in 0..5u8 {
            l1.put(
                KvKey([200 + i; 16]),
                TierBlockRef::Host(&vec![0u8; bb as usize]),
            )
            .unwrap();
        }
        let tier: Arc<dyn KvTier> = l1.clone();
        let mut r = rig_tiers("cost_aware", 8, Some(tier), None, clock, |kv| {
            kv.ladder.enabled = true;
            kv.ladder.l0 = false;
        });
        let now = r.clock.now_mono();
        r.h.set_rung(TierId::L1, "fp8_e4m3", LadderReason::FillHighWater, now);
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        run(&mut r, &prompt);
        let cached = r.pool.used_blocks();
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert_eq!(r.h.stats().transfer_errors, 0, "no demotion ends Full");
        assert_eq!(
            l1.used_bytes(),
            8 * bb,
            "the three free l0 slots took demotions"
        );
        assert_eq!(r.h.stats().demotions, 3);
        assert!(r.pool.used_blocks() < cached, "L0 drained");
    }

    /// [`KvTier::free_slots`] on the L1 slab tier counts what [`take_slot`](L1PinnedTier::take_slot)
    /// would find: free slots of the asked size, an empty slab of another size at the slab limit
    /// (it reformats), and the slabs it may still allocate (none under host RED). Breaks if the
    /// count under- or over-shoots and a rung copy is refused or lands `Full`.
    #[test]
    fn l1_free_slots_counts_what_take_slot_finds() {
        use crate::tier::{L1Config, L1PinnedTier, TierBlockRef};
        use turbine_tensor::host::HostPinned;
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let fp8 = crate::codec::registry()
            .get("fp8_e4m3")
            .expect("fp8 registered")
            .bytes_per_block(&fmt16().layout);
        let l1 = Arc::new(L1PinnedTier::new(
            L1Config {
                enabled: true,
                max_bytes: 2 * 4 * bb,
                slab_bytes: 4 * bb,
                block_bytes: bb,
                memory_kind: MemoryKind::Dedicated,
            },
            Arc::new(HostPinned::new(u64::MAX)),
            arc.clone(),
        ));
        // Fresh: both slabs may still be allocated at the asked size.
        assert_eq!(l1.free_slots("l0", bb), 8);
        assert_eq!(l1.free_slots("fp8_e4m3", fp8), 2 * (4 * bb / fp8));
        // Five `l0` copies fill the first slab and spill one into the second: both non-empty,
        // so an `fp8` copy has no slot anywhere.
        for i in 0..5u8 {
            l1.put(
                KvKey([7 + i; 16]),
                TierBlockRef::Host(&vec![0u8; bb as usize]),
            )
            .unwrap();
        }
        assert_eq!(l1.free_slots("l0", bb), 3);
        assert_eq!(l1.free_slots("fp8_e4m3", fp8), 0);
        // At the slab limit an empty other-size slab is counted at the asked size.
        for i in 0..5u8 {
            l1.evict(&KvKey([7 + i; 16])).unwrap();
        }
        assert_eq!(l1.free_slots("fp8_e4m3", fp8), 2 * (4 * bb / fp8));
        // A size above the slot limit fits nowhere; under host RED the emptied slabs are
        // released and no new slab is counted.
        assert_eq!(l1.free_slots("l0", bb + 1), 0);
        l1.set_host_pressure(PressureState::Red);
        assert_eq!(l1.free_slots("fp8_e4m3", fp8), 0);
    }

    /// The rung is taken only while it has a free slot beyond the copies in flight into the tier
    /// at its size: with two `fp8` slots free and three demotions queued, the third stores at the
    /// tier's format instead of landing `Full` on a slot the first two will take. Breaks if the
    /// in-flight copies are not subtracted.
    #[test]
    fn a_rung_slot_counts_the_copies_in_flight() {
        use crate::tier::{L1Config, L1PinnedTier, TierBlockRef};
        use turbine_tensor::host::HostPinned;
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let fp8 = crate::codec::registry()
            .get("fp8_e4m3")
            .expect("fp8 registered")
            .bytes_per_block(&fmt16().layout);
        let l1 = Arc::new(L1PinnedTier::new(
            L1Config {
                enabled: true,
                max_bytes: 2 * 4 * bb,
                slab_bytes: 4 * bb,
                block_bytes: bb,
                memory_kind: MemoryKind::Dedicated,
            },
            Arc::new(HostPinned::new(u64::MAX)),
            arc,
        ));
        // Slab A holds `fp8` slots with two free; slab B holds `l0` slots with two free.
        for i in 0..5u8 {
            l1.put_as(
                KvKey([30 + i; 16]),
                "fp8_e4m3",
                fp8,
                TierBlockRef::Host(&vec![0u8; fp8 as usize]),
            )
            .unwrap();
        }
        for i in 0..2u8 {
            l1.put(
                KvKey([40 + i; 16]),
                TierBlockRef::Host(&vec![0u8; bb as usize]),
            )
            .unwrap();
        }
        let tier: Arc<dyn KvTier> = l1.clone();
        let mut r = rig_tiers("cost_aware", 8, Some(tier), None, clock, |kv| {
            kv.ladder.enabled = true;
            kv.ladder.l0 = false;
        });
        let now = r.clock.now_mono();
        r.h.set_rung(TierId::L1, "fp8_e4m3", LadderReason::FillHighWater, now);
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert_eq!(
            r.h.stats().transfer_errors,
            0,
            "the third rung copy fell back"
        );
        assert_eq!(r.h.stats().demotions, 4);
        assert_eq!(l1.free_slots("l0", bb), 0, "the fallback took the l0 slot");
    }

    /// The reclaim handle answers from the published snapshot and the engine thread applies it.
    #[test]
    fn reclaim_handle_requests_are_applied_at_the_boundary() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(8, Some(l1), None, clock);
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        // Sent again: a hit, the reuse evidence pressure reclaim copies a block for.
        run(&mut r, &prompt);
        let handle = r.h.reclaimer();
        assert_eq!(handle.demote(0.0), 0, "nothing published yet");
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(handle.demote(0.25), 2 * bb, "4 of 8 used, target 2");
        assert_eq!(r.h.apply_reclaim(&mut r.pool), 2 * bb);
        assert_eq!(
            r.h.apply_reclaim(&mut r.pool),
            0,
            "a request is applied once"
        );
        // Background copies are paced by the time between pumps (one at a time here, where
        // the first pumps share an instant).
        for _ in 0..3 {
            r.h.poll(&mut r.pool, &mut r.backend);
            r.clock.advance(Duration::from_millis(10));
        }
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(r.pool.used_blocks(), 2);
        assert_eq!(handle.free_unreferenced(0.0), 2 * bb);
    }

    /// The overload soak regression (ITL p99 393 ms with L1 against 205 ms without): at ORANGE
    /// and RED the Phase 3 controller asks for every unreferenced block on each tick, and every
    /// finished one-off request's blocks were copied to L1, each pinning its L0 block until
    /// done. Catches a copy spent on a block without reuse evidence under controller reclaim,
    /// a one-off block `free_unreferenced` fails to drop, a re-used block it fails to copy
    /// down, or more than `DEMOTION_INFLIGHT` copies started at once.
    #[test]
    fn pressure_reclaim_drops_one_off_blocks_and_bounds_copies() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 64 * bb, arc));
        let mut r = rig(48, Some(l1.clone()), None, clock);
        let handle = r.h.reclaimer();
        let settle = |r: &mut Rig| {
            for _ in 0..4 {
                r.clock.advance(Duration::from_millis(10));
                r.h.poll(&mut r.pool, &mut r.backend);
            }
        };

        // A one-off prompt (4 blocks) and a re-used one (36 blocks, sent twice).
        let one_off: Vec<u32> = (0..66).collect();
        run(&mut r, &one_off);
        let reused: Vec<u32> = (1000..1578).collect();
        run(&mut r, &reused);
        assert_eq!(run(&mut r, &reused).cached_tokens, 576);
        assert_eq!(r.pool.cached_unreferenced(), 40);

        // `demote` (YELLOW): the re-used blocks go down, at most DEMOTION_INFLIGHT at once; the
        // one-off blocks stay cached in L0.
        handle.demote(0.0);
        r.h.apply_reclaim(&mut r.pool);
        assert_eq!(r.h.l0_demotions_in_flight(), DEMOTION_INFLIGHT);
        handle.demote(0.0);
        r.h.apply_reclaim(&mut r.pool);
        assert_eq!(
            r.h.l0_demotions_in_flight(),
            DEMOTION_INFLIGHT,
            "no new copy before earlier ones finish"
        );
        settle(&mut r);
        handle.demote(0.0);
        r.h.apply_reclaim(&mut r.pool);
        settle(&mut r);
        assert_eq!(l1.len(), 36, "every re-used block copied down");
        assert_eq!(r.pool.cached_unreferenced(), 4, "the one-off blocks stay");

        // `free_unreferenced` (ORANGE and above): the one-off blocks are dropped now, no copy.
        let drops = r.h.stats().drops;
        handle.free_unreferenced(0.0);
        r.h.apply_reclaim(&mut r.pool);
        assert_eq!(r.h.l0_demotions_in_flight(), 0);
        assert_eq!(r.pool.used_blocks(), 0, "freed at once");
        assert_eq!(r.h.stats().drops, drops + 4);
        assert_eq!(l1.len(), 36);
    }

    /// User decision 2026-09-30 (B): the eviction score values a copy at its encoded size — its
    /// memory term at the copy's bytes in its tier's format, its retrieval at the bytes it would
    /// be stored at in the tier below (path estimates are rates per encoded byte). An L0 copy
    /// keeps the L0 block size. Breaks if a lossy-tier copy is scored at its L0 size.
    #[test]
    fn score_prices_a_copy_at_its_encoded_size() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc.clone()));
        let l2 = Arc::new(MemTier::new(TierId::L2, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), Some(l2), clock);
        r.h.cfg.l1_format = "fp8_e4m3";
        // (TurboQuant's per-block overhead makes it larger than L0 on this tiny layout.)
        r.h.cfg.l2_format = "fp8_e4m3";
        let fp8 = r.h.format_bytes("fp8_e4m3");
        assert!(fp8 < bb);

        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        let now = r.h.now();
        // Not the sequence's lossless tail, which demotes at the L0 format.
        let b =
            r.h.dir
                .iter()
                .find(|b| !r.h.tail.contains(&b.key))
                .expect("a cached non-tail block")
                .clone();
        let l0 = r.h.score_one(&b, TierId::L0, &r.pool, now);
        assert_eq!(
            l0.block.size_bytes, bb,
            "an L0 copy is scored at the L0 size"
        );
        let l1_rate = r.h.transfer.estimate(TransferPath::L1ToL0);
        assert_eq!(l0.retrieval.seconds, l1_rate.block_seconds(fp8));

        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..3 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert!(!l1.is_empty(), "blocks demoted to L1");
        let b =
            r.h.dir
                .iter()
                .find(|b| {
                    b.location(TierId::L1)
                        .is_some_and(|l| l.format == "fp8_e4m3")
                })
                .expect("an fp8 L1 copy")
                .clone();
        let now = r.h.now();
        let s = r.h.score_one(&b, TierId::L1, &r.pool, now);
        assert_eq!(s.block.size_bytes, fp8, "memory at the copy's encoded size");
        let l2_rate = r.h.transfer.estimate(TransferPath::L2ToL0);
        assert_eq!(
            s.retrieval.seconds,
            l2_rate.block_seconds(fp8),
            "retrieval at the size it would be stored at in L2"
        );
    }

    /// User decision 2026-10-02 (1 A): a sequence's lossless last block is scored like its
    /// history for eviction order — its retrieval at the bytes its history would be stored at in
    /// the tier below — while its demotion and its memory keep its real L0-format size. Before,
    /// its retrieval was priced at the L0 size (several times its history's), so the leaf
    /// outranked every history block and other sessions' histories drained first. Breaks if the
    /// last block's retrieval is priced at its own L0-format bytes, or if it demotes lossy.
    #[test]
    fn last_block_is_scored_like_its_history() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc.clone()));
        let l2 = Arc::new(MemTier::new(TierId::L2, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), Some(l2), clock);
        r.h.cfg.l1_format = "fp8_e4m3";
        r.h.cfg.l2_format = "fp8_e4m3";
        let fp8 = r.h.format_bytes("fp8_e4m3");
        assert!(fp8 < bb);

        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        let now = r.h.now();
        let tail =
            r.h.dir
                .iter()
                .find(|b| r.h.tail.contains(&b.key))
                .expect("a lossless last block")
                .clone();
        let history =
            r.h.dir
                .iter()
                .find(|b| !r.h.tail.contains(&b.key))
                .expect("a history block")
                .clone();
        let t = r.h.score_one(&tail, TierId::L0, &r.pool, now);
        let h = r.h.score_one(&history, TierId::L0, &r.pool, now);
        let l1_rate = r.h.transfer.estimate(TransferPath::L1ToL0);
        assert_eq!(h.retrieval.seconds, l1_rate.block_seconds(fp8));
        assert_eq!(
            t.retrieval.seconds, h.retrieval.seconds,
            "the last block's retrieval is priced like its history's"
        );
        assert_eq!(t.block.size_bytes, bb, "its memory stays the L0 size");
        let codec = r.h.copy_codec(&tail.key, TierId::L0, TierId::L1).unwrap();
        assert_eq!(
            (codec.to, codec.to_bytes),
            (L0_FORMAT, bb),
            "it still demotes at the L0 format"
        );

        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..3 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        let tail =
            r.h.dir
                .iter()
                .find(|b| {
                    r.h.tail.contains(&b.key)
                        && b.location(TierId::L1)
                            .is_some_and(|l| l.format == L0_FORMAT)
                })
                .expect("a lossless last block in L1")
                .clone();
        let now = r.h.now();
        let s = r.h.score_one(&tail, TierId::L1, &r.pool, now);
        assert_eq!(s.block.size_bytes, bb, "L1 memory at its real encoded size");
        let l2_rate = r.h.transfer.estimate(TransferPath::L2ToL0);
        assert_eq!(
            s.retrieval.seconds,
            l2_rate.block_seconds(fp8),
            "retrieval at the size its history would be stored at in L2"
        );
    }

    /// User decision 2026-09-30 (A), a Phase 4 fix: `make_room` counts L1 → L2 spills already in
    /// flight as room on the way. Before, every call made while those copies were still moving
    /// spilled further victims, so under RED a full L1 emptied itself in a few steps. Breaks if a
    /// second call before the spills land sends more victims to L2.
    #[test]
    fn make_room_counts_spills_in_flight() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 4 * bb, arc.clone()));
        let l2 = Arc::new(MemTier::new(TierId::L2, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), Some(l2), clock);
        run(&mut r, &(0..66).collect::<Vec<u32>>());
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert_eq!(l1.len(), 4, "L1 full");

        let spilling =
            |h: &KvHierarchy| h.demoting.values().filter(|d| d.from == TierId::L1).count();
        assert_eq!(r.h.make_room(&mut r.pool, TierId::L1, 2), 0);
        assert_eq!(spilling(&r.h), 2, "two victims leave for L2");
        r.h.make_room(&mut r.pool, TierId::L1, 2);
        assert_eq!(
            spilling(&r.h),
            2,
            "the spills in flight are the room: none more"
        );
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert_eq!(l1.len(), 2, "L1 gave up the room asked for, no more");
    }

    /// The Phase 4 bench regression (8 % Llama throughput on a no-reuse workload): capacity
    /// demotion copied every finished one-off request's blocks to L1. Catches a copy spent on a
    /// block without reuse evidence (no hit, no session, not a shared prefix), or a session
    /// block / re-used block that capacity demotion fails to copy down.
    #[test]
    fn capacity_demotion_needs_reuse_evidence() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), None, clock);

        // A one-off prompt: four cached blocks, none worth a copy.
        let one_off: Vec<u32> = (0..66).collect();
        run(&mut r, &one_off);
        assert_eq!(r.pool.cached_unreferenced(), 4);
        assert_eq!(r.h.demote_to(&mut r.pool, 0.0, EvictReason::Capacity), 0);
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(l1.len(), 0, "no copy of a block without reuse evidence");
        assert_eq!(r.pool.cached_unreferenced(), 4, "left cached in L0");

        // A session's blocks carry evidence from the first turn: they are copied down.
        let hints = SessionHints {
            session_id: "s1".into(),
            resume_within_secs: None,
            end: false,
        };
        let session_prompt: Vec<u32> = (1000..1066).collect();
        let id = RequestId::new_v4();
        let req = AttachRequest {
            request: id,
            prompt: &session_prompt,
            cache_salt: "",
            session: Some(&hints),
            priority: Priority(0),
            allow_lossy: None,
        };
        let a = match r.h.attach_prefix(&mut r.pool, &req) {
            AttachOutcome::Ready(a) => a,
            other => panic!("expected Ready, got {other:?}"),
        };
        prefill_and_finish(&mut r, id, &session_prompt, &a);
        assert_eq!(r.pool.cached_unreferenced(), 8);
        assert_eq!(
            r.h.demote_to(&mut r.pool, 0.0, EvictReason::Capacity),
            4 * bb,
            "only the session's four blocks move"
        );
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(l1.len(), 4);
        assert_eq!(r.pool.cached_unreferenced(), 4, "the one-off blocks stay");

        // A hit is evidence too: the one-off prompt sent again makes its blocks worth a copy.
        let again = run(&mut r, &one_off);
        assert_eq!(again.cached_tokens, 64);
        assert_eq!(
            r.h.demote_to(&mut r.pool, 0.0, EvictReason::Capacity),
            4 * bb
        );

        // `demote_to` under `Pressure` (the offline simulator's reclaim) is not gated: one-off
        // blocks leave L0 too. The controller's requests go through `pressure_reclaim`.
        let other: Vec<u32> = (2000..2033).collect();
        run(&mut r, &other);
        let before = l1.len();
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..3 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert_eq!(l1.len(), before + 6, "the re-used four and the one-off two");
        assert_eq!(
            r.pool.used_blocks(),
            0,
            "pressure empties L0 of unreferenced blocks"
        );
    }

    /// P6b Task 6: after a prefetch of a prompt whose lower-tier copies are lossy (fp8 L2), the
    /// promoted L0 copies sit under their `lossy_key` entries and the next lookup reuses them
    /// from L0 (the fastest lossy copy), not the exact entries' lossy L2 locations (which the
    /// planner then priced against recompute). No exact entry gains an L0 location (S-3). Breaks
    /// if the lookup prefers the exact entry's lossy L2 copy over the promoted L0 one.
    #[test]
    fn lookup_prefers_the_promoted_lossy_l0_copy() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l2 = Arc::new(MemTier::new(TierId::L2, 16 * bb, arc));
        let mut r = rig(16, None, Some(l2.clone()), clock);
        r.h.cfg.l2_format = "fp8_e4m3";
        r.h.cfg.allow_lossy = true;
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        assert!(l2.len() >= 2, "blocks sit in L2");

        let accepted =
            r.h.prefetch(
                &mut r.pool,
                PrefetchTarget::Tokens {
                    prompt: &prompt,
                    cache_salt: "",
                },
            )
            .expect("prefetch accepted");
        assert!(accepted.blocks_queued >= 2, "{accepted:?}");
        for _ in 0..6 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        let hasher = Blake3Hasher(r.h.namespaces.get(""));
        for k in prefix_keys(&hasher, &prompt, 16) {
            let exact = r.h.directory().get(&k).expect("the exact entry");
            // (The sequence's tail stays lossless, and now every demoted non-tail block sits at
            // fp8; the exact entries keep their L0 locations only where no L0 copy exists.)
            let has_fp8 = exact.locations.iter().any(|l| l.format == "fp8_e4m3");
            assert!(
                !has_fp8 || exact.location(TierId::L0).is_none(),
                "a lossy L0 copy is never filed under the exact entry: {:?}",
                exact.locations
            );
        }

        let id = RequestId::new_v4();
        let a = match attach(&mut r, id, &prompt) {
            AttachOutcome::Ready(a) => a,
            other => panic!("the promoted blocks are in L0: Ready, got {other:?}"),
        };
        assert_eq!(a.cached_tokens, 64, "all four blocks reused");
        assert_eq!(
            a.lossy_tokens, 48,
            "the three promoted lossy blocks are served lossy"
        );
        assert!(a.plan.promote.is_empty(), "no copy: {:?}", a.plan);
        assert_eq!(a.plan.reason, PlanReason::AllL0);
        // Every prefetched block is counted used once attached (the lossy ones by their lossy
        // key), and a second prefetch finds them all resident instead of promoting again.
        assert_eq!(r.h.prefetch_stats().used, accepted.blocks_queued as u64);
        r.pool.release(&a.blocks);
        r.h.request_done(&mut r.pool, id, false);
        let again =
            r.h.prefetch(
                &mut r.pool,
                PrefetchTarget::Tokens {
                    prompt: &prompt,
                    cache_salt: "",
                },
            )
            .expect("prefetch accepted");
        assert_eq!(again.blocks_queued, 0, "{again:?}");
        assert_eq!(again.blocks_resident, 4, "{again:?}");
    }

    /// P6b S-2 (`GET /turbine/v1/kv`): each tier lists its copies per codec, with the bytes at
    /// that codec's encoded block size, so a lossy tier's capacity in its own blocks can be read
    /// (`blocks_total` counts L0-format blocks). Breaks if the lists drop a copy, count it in
    /// the wrong tier or codec, or price an fp8 copy at the L0 block size.
    #[test]
    fn document_lists_copies_per_codec() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l2 = Arc::new(MemTier::new(TierId::L2, 16 * bb, arc));
        let mut r = rig(16, None, Some(l2), clock);
        r.h.cfg.l2_format = "fp8_e4m3";
        let fp8 = r.h.format_bytes("fp8_e4m3");
        assert!(fp8 < bb);
        run(&mut r, &(0..66).collect::<Vec<u32>>());
        r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
        let doc = r.h.document(&r.pool, (0, 0));
        let tier = |name: &str| doc.tiers.iter().find(|t| t.tier == name).expect("tier");
        let l2 = &tier("l2").formats;
        // The three non-tail full blocks demote at the rung (fp8); before `make_room` counted
        // bytes, the root stayed in L0 because the rung-unit room ran out one victim early.
        assert_eq!(l2["fp8_e4m3"].blocks, 3, "{l2:?}");
        assert_eq!(l2["fp8_e4m3"].bytes, 3 * fp8);
        // The lossless tail is demoted at the L0 format.
        assert_eq!(l2[L0_FORMAT].blocks, 1, "{l2:?}");
        assert_eq!(l2[L0_FORMAT].bytes, bb);
        let l0 = &tier("l0").formats;
        assert!(
            tier("l0")
                .formats
                .get(L0_FORMAT)
                .is_none_or(|u| u.blocks == 0),
            "{l0:?}"
        );
        assert!(tier("l1").formats.is_empty());
    }

    /// Attach (waiting for promotions), prefill and finish `prompt`; the attach.
    fn run_through_promotions(r: &mut Rig, prompt: &[u32]) -> PrefixAttach {
        let id = RequestId::new_v4();
        let a = match attach(r, id, prompt) {
            AttachOutcome::Ready(a) => a,
            AttachOutcome::Promoting => (0..20)
                .find_map(|_| {
                    r.clock.advance(Duration::from_millis(10));
                    r.h.poll(&mut r.pool, &mut r.backend)
                        .into_iter()
                        .find(|(rid, _)| *rid == id)
                        .map(|(_, a)| a)
                })
                .expect("the promotions land"),
            other => panic!("expected an attach, got {other:?}"),
        };
        prefill_and_finish(r, id, prompt, &a);
        a
    }

    /// The OLMoE multi-turn shortfall of lower-tier tq4 (decision "6b: lower-tier tq4 after the
    /// promotion fix"): one session's turns, with its history leaving L0 leaf-first between
    /// turns, so the first lossy block of each turn's prefix moves *earlier*. Turn 2 reuses
    /// blocks 0-1 exact and 2-3 lossy, so its new blocks 4-5 are keyed by the lossy chain that
    /// starts at block 2. Turn 3 finds block 0 lossy, so its lookup's chain starts at block 0
    /// and never names blocks 4-5: they were computed over a lossy prefix of the same tokens
    /// and are still cached, but turn 3 recomputes them. Breaks if a lookup can only reach a
    /// lossy-lineage block through a chain that starts at the request's own first lossy block.
    #[test]
    fn lossy_lineage_blocks_survive_an_earlier_switch() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 32 * bb, arc));
        let mut r = rig(32, Some(l1), None, clock);
        r.h.cfg.l1_format = "fp8_e4m3";
        r.h.cfg.allow_lossy = true;
        r.h.cfg.lossless_tail_blocks = 0;
        let turn = |blocks: u32| -> Vec<u32> { (0..blocks * 16).chain([9000, 9001]).collect() };
        let hasher = Blake3Hasher(r.h.namespaces.get(""));
        let keys = prefix_keys(&hasher, &turn(8), 16);

        // Turn 1: blocks 0-3; then its last two blocks leave L0 (leaf-first) for the lossy L1.
        assert_eq!(run(&mut r, &turn(4)).cached_tokens, 0);
        r.h.demote_keys(&mut r.pool, vec![keys[2], keys[3]], EvictReason::Pressure);
        settle(&mut r);
        for k in &keys[2..4] {
            let b = r.h.directory().get(k).expect("the exact entry");
            assert!(b.location(TierId::L0).is_none(), "{:?}", b.locations);
        }

        // Turn 2: blocks 0-1 exact, 2-3 lossy (promoted), 4-5 computed over the lossy prefix.
        let t2 = run_through_promotions(&mut r, &turn(6));
        assert_eq!((t2.cached_tokens, t2.lossy_tokens), (64, 32), "{t2:?}");

        // Everything leaves L0 for the lossy L1, block 0 included.
        for _ in 0..4 {
            r.h.demote_to(&mut r.pool, 0.0, EvictReason::Pressure);
            settle(&mut r);
        }
        assert_eq!(r.pool.cached_unreferenced(), 0, "L0 holds nothing cached");

        // Turn 3 reuses all six cached blocks, lossy from block 0 on.
        let t3 = run_through_promotions(&mut r, &turn(8));
        assert_eq!(
            (t3.cached_tokens, t3.lossy_tokens),
            (96, 96),
            "blocks 4-5 (computed over a lossy prefix by turn 2) are reused: {t3:?}"
        );
    }

    /// The Task 4 kv-sim regression (MultiTurn lru 251 → 199 s, cost_aware/lru 0.78 → 0.93):
    /// the P6b S-3 publish rule ("an exact entry whose every copy is lossy") also fired for an
    /// entry with *no* copy — a parent kept only by its children — so with L0-format tiers a
    /// recomputed parent was re-keyed where the Phase 4 hierarchy leaves it unkeyed. Catches
    /// `commit_progress` adding a copy to an existing entry that holds no lossy copy.
    #[test]
    fn commit_leaves_a_copyless_parent_unkeyed() {
        let clock = FakeClock::new(Duration::ZERO);
        let mut r = rig(8, None, None, clock);
        let prompt: Vec<u32> = (0..66).collect();
        run(&mut r, &prompt);
        let hasher = Blake3Hasher(r.h.namespaces.get(""));
        let keys = prefix_keys(&hasher, &prompt, 16);
        // Block 0 loses its only copy; its entry stays for its child.
        r.h.remove_copy(&mut r.pool, &keys[0], TierId::L0, EvictReason::Capacity);
        let parent = r.h.directory().get(&keys[0]).expect("kept for its child");
        assert!(parent.locations.is_empty());

        let again = run(&mut r, &prompt);
        assert_eq!(again.cached_tokens, 0, "the prefix misses at block 0");
        let parent =
            r.h.directory()
                .get(&keys[0])
                .expect("still kept for its child");
        assert!(
            parent.locations.is_empty(),
            "the recomputed block was filed under a copyless exact entry: {:?}",
            parent.locations
        );
        assert_eq!(
            r.h.l0_keys.len(),
            3,
            "only the first run's blocks 1-3 are keyed"
        );
    }

    /// The shared-prefix eval shape (decision "6b: shared-prefix eval never demotes the prefix
    /// to L1", A): a two-block prefix `P` shared by two finished items whose own blocks have no
    /// reuse evidence, plus one-off fillers, fill L0 past the capacity threshold. Leaf-first, `P`
    /// cannot leave L0 while the items' blocks stay, so capacity demotion has no victim.
    fn shared_prefix_rig(l1_format: &'static str, lower: bool) -> (Rig, Option<Arc<MemTier>>) {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = lower.then(|| Arc::new(MemTier::new(TierId::L1, 16 * bb, arc)));
        let mut r = rig(16, l1.clone(), None, clock);
        r.h.cfg.l1_format = l1_format;
        r.h.cfg.allow_lossy = true;
        // Two items behind the prefix: the second's attach is the prefix's hit.
        run(&mut r, &item(100));
        assert_eq!(run(&mut r, &item(200)).cached_tokens, 32);
        // Two one-off fillers: 4 + 8 = 12 of 16 L0 blocks, past the 0.7 threshold.
        run(&mut r, &(5000..5066).collect::<Vec<u32>>());
        run(&mut r, &(6000..6066).collect::<Vec<u32>>());
        assert_eq!(r.pool.cached_unreferenced(), 12);
        (r, l1)
    }

    /// The shared prefix (tokens 0..32) followed by one item block and two more tokens.
    fn item(start: u32) -> Vec<u32> {
        (0..32).chain(start..start + 18).collect()
    }

    fn settle(r: &mut Rig) {
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            r.h.poll(&mut r.pool, &mut r.backend);
        }
    }

    fn prefix_keys_of(r: &mut Rig) -> Vec<KvKey> {
        let hasher = Blake3Hasher(r.h.namespaces.get(""));
        prefix_keys(&hasher, &(0..32).collect::<Vec<u32>>(), 16)
    }

    /// Copy ahead (decision A): under capacity pressure a shared parent held in L0 by its
    /// children is copied down while its L0 copy stays (`copy_ahead`), only once; the L0 copy
    /// later leaves by the normal rules (here the controller's `free_unreferenced`, which drops
    /// the evidence-free children first) without a new transfer, and the next sharer promotes
    /// the prefix from L1. Breaks if the copy frees the L0 copy, if a shared parent gets no
    /// lower copy, if a second pass copies it again, if the later L0 drop copies it again, or if
    /// it copies ahead from ORANGE on.
    #[test]
    fn copy_ahead_keeps_a_shared_prefix_in_l0() {
        let (mut r, l1) = shared_prefix_rig(L0_FORMAT, true);
        let l1 = l1.expect("an L1");
        let keys = prefix_keys_of(&mut r);
        // ORANGE: freeing comes first, nothing is copied ahead.
        r.h.set_l0_state(PressureState::Orange);
        r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
        settle(&mut r);
        assert_eq!(r.h.stats().copy_aheads, 0, "no copy ahead at ORANGE");
        assert!(l1.is_empty());
        r.h.set_l0_state(PressureState::Green);
        assert_eq!(
            r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity),
            0,
            "a copy ahead frees no L0 block"
        );
        settle(&mut r);
        assert_eq!(l1.len(), 2, "both prefix blocks copied down");
        assert_eq!(r.h.stats().copy_aheads, 2);
        assert_eq!(r.h.stats().demotions, 0, "nothing left L0");
        assert_eq!(r.h.metrics.copy_ahead_value(TierId::L1), 2);
        for k in &keys {
            let b = r.h.directory().get(k).expect("the prefix entry");
            assert!(b.location(TierId::L0).is_some(), "the L0 copy stays");
            assert_eq!(b.location(TierId::L1).map(|l| l.format), Some(L0_FORMAT));
        }
        assert_eq!(r.pool.used_blocks(), 12);
        assert_eq!(r.pool.cached_unreferenced(), 12, "no copy pins a block");

        // A second pass has nothing left to copy ahead.
        let moved = r.h.stats().transfer_bytes;
        r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
        settle(&mut r);
        assert_eq!(r.h.stats().transfer_bytes, moved, "copied once");

        // Higher pressure: the children are dropped, then the prefix's L0 copies are freed at
        // once (their L1 copies exist), with no new copy.
        let handle = r.h.reclaimer();
        handle.free_unreferenced(0.0);
        r.h.apply_reclaim(&mut r.pool);
        assert_eq!(r.h.l0_demotions_in_flight(), 0, "no demotion copy started");
        settle(&mut r);
        assert_eq!(r.h.stats().transfer_bytes, moved, "the L0 drop is free");
        assert_eq!(r.pool.used_blocks(), 0);
        for k in &keys {
            let b = r.h.directory().get(k).expect("kept in L1");
            assert_eq!(b.fastest(), Some(TierId::L1));
        }

        // The next sharer promotes the prefix from L1.
        let id = RequestId::new_v4();
        assert_eq!(attach(&mut r, id, &item(300)), AttachOutcome::Promoting);
        let mut ready = Vec::new();
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            ready.extend(r.h.poll(&mut r.pool, &mut r.backend));
        }
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].1.cached_tokens, 32);
        assert_eq!(ready[0].1.plan.promote, vec![(TierId::L1, 2)]);
    }

    /// Copy ahead into a lossy tier (P6b S-3, decision "6b Task 4" item 2): the fp8 copy of an
    /// exact block is one more location on the exact entry, so while the L0 copy exists the
    /// prefix is served exact; once that L0 copy is gone it is served lossy. Breaks if the copy
    /// ahead is filed under a lossy key, stored at the L0 format, or served lossy while the
    /// exact L0 copy exists.
    #[test]
    fn copy_ahead_into_a_lossy_tier_keeps_the_exact_entry() {
        let (mut r, l1) = shared_prefix_rig("fp8_e4m3", true);
        let l1 = l1.expect("an L1");
        let keys = prefix_keys_of(&mut r);
        r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
        settle(&mut r);
        assert_eq!(l1.len(), 2);
        for k in &keys {
            let b = r.h.directory().get(k).expect("the exact entry");
            assert_eq!(b.lineage, Lineage::Exact);
            assert!(b.location(TierId::L0).is_some());
            assert_eq!(b.location(TierId::L1).map(|l| l.format), Some("fp8_e4m3"));
            for c in crate::codec::registry().iter() {
                let lk = lossy_key(*k, c.name(), r.h.lossy_seed);
                assert!(r.h.directory().get(&lk).is_none(), "no lossy entry");
            }
        }
        // Exact while the L0 copy exists.
        let id = RequestId::new_v4();
        let a = match attach(&mut r, id, &item(400)) {
            AttachOutcome::Ready(a) => a,
            other => panic!("the prefix is in L0: Ready, got {other:?}"),
        };
        assert_eq!((a.cached_tokens, a.lossy_tokens), (32, 0));
        prefill_and_finish(&mut r, id, &item(400), &a);

        // Lossy once the L0 copy is gone.
        r.h.reclaimer().free_unreferenced(0.0);
        r.h.apply_reclaim(&mut r.pool);
        settle(&mut r);
        assert_eq!(r.pool.used_blocks(), 0);
        let id = RequestId::new_v4();
        assert_eq!(attach(&mut r, id, &item(500)), AttachOutcome::Promoting);
        let mut ready = Vec::new();
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            ready.extend(r.h.poll(&mut r.pool, &mut r.backend));
        }
        assert_eq!(ready.len(), 1);
        assert_eq!(
            (ready[0].1.cached_tokens, ready[0].1.lossy_tokens),
            (32, 32)
        );
    }

    /// No lower tier, no copy ahead: capacity demotion leaves the shared prefix exactly as
    /// Phase 4 did. Breaks if a copy is attempted without a tier to hold it.
    #[test]
    fn copy_ahead_needs_a_lower_tier() {
        let (mut r, _) = shared_prefix_rig(L0_FORMAT, false);
        assert_eq!(r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity), 0);
        settle(&mut r);
        assert_eq!(r.h.stats().copy_aheads, 0);
        assert_eq!(r.h.stats().transfer_bytes, 0);
        assert_eq!(r.pool.cached_unreferenced(), 12);
    }

    /// The eval's path at GREEN (P6b S-8): after the copy ahead, one-off fillers keep arriving;
    /// allocation reclaims cached L0 blocks in the published leaf-first order, so the shared
    /// prefix's evidence-free children go first, and then capacity demotion frees the prefix's
    /// L0 copies at no copy cost. Breaks if the prefix never leaves L0 once its children are
    /// gone, or if it is copied again when it does.
    #[test]
    fn copied_ahead_prefix_leaves_l0_once_its_children_are_reclaimed() {
        let (mut r, _) = shared_prefix_rig(L0_FORMAT, true);
        let keys = prefix_keys_of(&mut r);
        r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
        settle(&mut r);
        assert_eq!(r.h.stats().copy_aheads, 2);
        let moved = r.h.stats().transfer_bytes;
        for i in 0..6u32 {
            r.h.refresh_reclaim_order(&mut r.pool);
            let filler: Vec<u32> = (10_000 + i * 100..10_066 + i * 100).collect();
            let id = RequestId::new_v4();
            let AttachOutcome::Ready(a) = attach(&mut r, id, &filler) else {
                panic!("a filler misses");
            };
            let table: Vec<BlockId> = r.pool.allocate(5).expect("reclaimable").to_vec();
            r.h.after_plan(&mut r.pool);
            assert!(a.blocks.is_empty());
            r.h.commit_progress(&mut r.pool, id, &table, &filler);
            r.pool.release(&table);
            r.h.request_done(&mut r.pool, id, false);
            r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
            settle(&mut r);
        }
        for k in &keys {
            let b = r.h.directory().get(k).expect("kept in L1");
            assert!(b.location(TierId::L0).is_none(), "the prefix left L0");
            assert!(b.location(TierId::L1).is_some());
        }
        assert_eq!(r.h.stats().transfer_bytes, moved, "no second copy");
    }

    /// Only shared prefixes are copied ahead: a re-used chain whose blocks each have one child (a
    /// session's history, a prompt sent twice) waits for the leaf-first demotion, so its copies
    /// do not take lower-tier room the leaf-first rule would hold while its leaf stays in L0.
    /// Breaks if a block without a shared descendant is copied ahead.
    #[test]
    fn copy_ahead_skips_unshared_chains() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), None, clock);
        // A 66-token prompt sent twice: four blocks with a hit each, one child each.
        let chain: Vec<u32> = (0..66).collect();
        run(&mut r, &chain);
        assert_eq!(run(&mut r, &chain).cached_tokens, 64);
        run(&mut r, &(5000..5066).collect::<Vec<u32>>());
        run(&mut r, &(6000..6066).collect::<Vec<u32>>());
        assert_eq!(r.pool.cached_unreferenced(), 12);
        // The chain's leaf is a capacity victim (copied, then freed); its parents are not
        // copied ahead.
        r.h.demote_to(&mut r.pool, 0.7, EvictReason::Capacity);
        settle(&mut r);
        assert_eq!(r.h.stats().copy_aheads, 0);
        assert_eq!(l1.len(), r.h.stats().demotions as usize);
    }

    /// Pressure reclaim's shortfall becomes the queued-prefix demand only at YELLOW and ORANGE
    /// (decision "6b: after the held-prefix ledger fix", 1 B): blocks a request holds are no
    /// victims, so `need` minus the victims found is what queued requests' prefixes may cover.
    /// Breaks if the demand is recorded at GREEN, RED or SURVIVAL, ignores the victims found,
    /// or is not reset once taken.
    #[test]
    fn queued_prefix_demand_only_at_yellow_and_orange() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(16, Some(l1), None, clock);
        let held: Vec<u32> = (0..66).collect();
        run(&mut r, &held);
        // A queued request holds the 4 blocks; a one-off prompt leaves 4 unreferenced ones.
        let id = RequestId::new_v4();
        let AttachOutcome::Ready(a) = attach(&mut r, id, &held) else {
            panic!("the prompt is cached in L0");
        };
        assert_eq!(a.blocks.len(), 4);
        run(&mut r, &(7000..7066).collect::<Vec<u32>>());
        assert_eq!(r.pool.cached_unreferenced(), 4);
        let handle = r.h.reclaimer();
        for (state, want) in [
            (PressureState::Green, 0),
            (PressureState::Yellow, 4),
            (PressureState::Orange, 4),
            (PressureState::Red, 0),
            (PressureState::Survival, 0),
        ] {
            r.h.set_l0_state(state);
            // `demote` leaves one-off blocks cached: the victims stay and the demand is the same
            // on every call.
            handle.demote(0.0);
            r.h.apply_reclaim(&mut r.pool);
            assert_eq!(r.h.take_queued_prefix_demand(), want, "{state:?}");
            assert_eq!(r.h.take_queued_prefix_demand(), 0, "taken once");
        }
        r.pool.release(&a.blocks);
        r.h.request_done(&mut r.pool, id, false);
    }

    /// A queued request that released its prefix (`detach_prefix`) attaches again through the
    /// planner: from the lower tier its blocks were demoted to, without waiting on the blocks it
    /// had registered as pending itself, and without counting its prompt twice. Breaks if the
    /// detach keeps its pending keys (the re-attach waits `PENDING_WAIT`), keeps its references,
    /// or counts the re-attach as a new prompt.
    #[test]
    fn detached_prefix_attaches_again_through_the_planner() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(16, Some(l1.clone()), None, clock);
        let base: Vec<u32> = (0..66).collect();
        run(&mut r, &base);
        // The queued prompt extends the cached one: 4 blocks attached, 2 pending (its own).
        let prompt: Vec<u32> = (0..100).collect();
        let id = RequestId::new_v4();
        let AttachOutcome::Ready(a) = attach(&mut r, id, &prompt) else {
            panic!("the first 4 blocks are cached in L0");
        };
        assert_eq!(a.cached_tokens, 64);
        assert_eq!(
            attach(&mut r, RequestId::new_v4(), &prompt),
            AttachOutcome::WaitForPrefix,
            "its uncomputed blocks are pending"
        );
        let prompt_tokens = r.h.stats().prompt_tokens;
        r.h.detach_prefix(&mut r.pool, id, &a, 4);
        assert_eq!(r.pool.referenced_blocks(), 0, "the references are released");
        assert_eq!(r.pool.cached_unreferenced(), 4);
        assert_eq!(r.h.stats().queued_prefix_detached, 4);
        // The normal reclaim demotes them (they have the first attach's hit).
        r.h.set_l0_state(PressureState::Yellow);
        r.h.reclaimer().demote(0.0);
        r.h.apply_reclaim(&mut r.pool);
        settle(&mut r);
        assert_eq!(l1.len(), 4, "the released prefix went down to L1");
        assert_eq!(r.pool.used_blocks(), 0, "and left L0");
        let outcome = attach(&mut r, id, &prompt);
        assert_eq!(outcome, AttachOutcome::Promoting, "back from L1, no wait");
        assert_eq!(
            r.h.pending_blocks(id),
            4,
            "the promotion targets are held now"
        );
        let mut ready = Vec::new();
        for _ in 0..4 {
            r.clock.advance(Duration::from_millis(10));
            ready.extend(r.h.poll(&mut r.pool, &mut r.backend));
        }
        assert_eq!(ready.len(), 1);
        assert_eq!((ready[0].0, ready[0].1.cached_tokens), (id, 64));
        assert_eq!(
            r.h.stats().prompt_tokens,
            prompt_tokens,
            "the re-attach is not a new prompt"
        );
        r.pool.release(&ready[0].1.blocks);
        r.h.request_done(&mut r.pool, id, false);
    }
}
