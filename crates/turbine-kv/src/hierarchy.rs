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
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use smallvec::SmallVec;
use turbine_core::clock::Clock;
use turbine_core::config::{KvConfig, KvPrefetchConfig, KvSessionConfig};
use turbine_core::registry::UnknownModule;
use turbine_core::request::SessionHints;
use turbine_core::types::{
    BlockId, MemoryKind, ModelFingerprint, ModelIdentity, PressureState, Priority, RequestId,
};

use crate::directory::{
    CostEstimate, KvBlock, KvDirectory, KvPriority, PrefixMatch, SessionId, TokenRange,
};
use crate::document::{
    HitRate, KvDocument, KvSummary, KvTierDocument, Prefetch, Sessions, TierState, Transfers,
};
use crate::identity::{Blake3Hasher, KeyHasher, KvFormat, KvKey, NamespaceCache, prefix_keys};
use crate::metrics::{EvictReason, KvMetrics, PrefetchOutcome};
use crate::planner::{KvPlan, PlanInputs, PlanReason, plan_prefix};
use crate::policy::{
    BlockScoreInputs, KvBlockSummary, SelectedPolicy, make_policy, recompute_seconds,
};
use crate::pool::BlockPool;
use crate::session::{PrefetchTracker, SessionAction, SessionTable};
use crate::tier::{KvLocation, KvTier, TierId, demotion_target};
use turbine_reliability::throttle::KvReclaimer;

use crate::transfer::{
    TransferBackend, TransferEngine, TransferPath, TransferPurpose, TransferRequest,
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
}

impl HierarchyConfig {
    /// Copies waiting to start, across all paths.
    pub const TRANSFER_QUEUE: usize = 4096;

    /// `UnknownModule` when `kv.policy` names no registered eviction policy (the server
    /// validates the name before any port is bound, so only a hand-built config gets here).
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
        })
    }
}

/// What one admission sees of its cached prefix (the scheduler's `SchedRequest.cached_prefix`).
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixAttach {
    /// L0 blocks of the reused prefix; the request owns one reference to each.
    pub blocks: SmallVec<[BlockId; 16]>,
    /// Tokens the blocks cover (`blocks × block_tokens`); feeds
    /// `ResourceEstimate.cached_prefix_tokens`.
    pub cached_tokens: u32,
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

/// An attach waiting for its promotions; `promotions` are the L0 targets still in flight.
struct Pending {
    attach: PrefixAttach,
    promotions: Vec<BlockId>,
}

/// Per-request KV state from attach until `request_done`.
struct RequestKv {
    salt: String,
    session: Option<SessionId>,
    priority: KvPriority,
    /// Keys of the full blocks seen so far (prompt, then generated tokens).
    keys: Vec<KvKey>,
    /// Blocks already keyed in the directory (or attached from it).
    committed: usize,
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
    /// Keys with a demotion copy in flight: (from, to).
    demoting: HashMap<KvKey, (TierId, TierId)>,
    pending: HashMap<RequestId, Pending>,
    requests: HashMap<RequestId, RequestKv>,
    /// L0 block → the key the directory holds for it.
    l0_keys: HashMap<BlockId, KvKey>,
    l0_state: PressureState,
    prefill_tps: f64,
    reclaim: Arc<KvReclaimHandle>,
    ready: Vec<(RequestId, PrefixAttach)>,
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
        let blocks_of = |t: &Option<Arc<dyn KvTier>>| {
            t.as_ref().map_or(0, |t| {
                usize::try_from(t.capacity_bytes() / cfg.block_bytes.max(1)).unwrap_or(usize::MAX)
            })
        };
        let max_entries = (l0_blocks as usize)
            .saturating_add(blocks_of(&l1))
            .saturating_add(blocks_of(&l2));
        KvHierarchy {
            policy: cfg.policy,
            transfer: TransferEngine::new(
                cfg.max_inflight_bytes,
                cfg.transfer_queue,
                clock.clone(),
            ),
            sessions: SessionTable::new(cfg.session.clone(), &cfg.prefetch),
            namespaces: NamespaceCache::new(model, format),
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

    pub fn transfer(&self) -> &TransferEngine {
        &self.transfer
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

    /// A copy of `key` into L0 (promotion or prefetch) is queued or in flight.
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
                },
            );
        }
        let hasher = Blake3Hasher(self.namespaces.get(req.cache_salt));
        let m = if self.cfg.prefix_sharing {
            self.dir.lookup(&hasher, req.prompt, bt, now)
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
        if let Some(r) = self.requests.get_mut(&req.request) {
            r.keys = m.keys.clone();
        }
        self.metrics.record_lookup(&m);
        for b in &m.blocks {
            self.stats.lookups[Self::lookup_slot(b.tier)] += 1;
        }
        self.stats.lookups[3] += (m.keys.len() - m.blocks.len()) as u64;

        let tiers: Vec<TierId> = m.blocks.iter().map(|b| b.tier).collect();
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
        };
        let mut plan = plan_prefix(&inputs);
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
                let tr = TransferRequest {
                    path,
                    key: mb.key,
                    bytes: self.cfg.block_bytes,
                    owner: Some(req.request),
                    purpose: TransferPurpose::Promote,
                    src_slot: mb.location.slot,
                    dst_slot: u64::from(ids[0].0),
                };
                match self.transfer.submit(tr) {
                    Ok(t) => {
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
        // Blocks this request will compute are pending, so an identical concurrent prefix waits.
        for key in m.keys.iter().skip(m.blocks.len()) {
            self.dir.register_pending(*key, now);
        }
        let cached_tokens = blocks.len() as u32 * bt;
        self.metrics.plan(plan.reason, plan.recompute_tokens);
        self.metrics.prompt_tokens.inc_by(req.prompt.len() as u64);
        self.metrics
            .prefix_cached_tokens
            .inc_by(u64::from(cached_tokens));
        self.stats.prompt_tokens += req.prompt.len() as u64;
        self.stats.cached_tokens += u64::from(cached_tokens);
        self.stats.recompute_tokens += u64::from(plan.recompute_tokens);
        tracing::debug!(
            event = "kv_plan",
            request_id = ?req.request,
            matched_l0 = tiers.iter().filter(|t| **t == TierId::L0).count(),
            matched_l1 = tiers.iter().filter(|t| **t == TierId::L1).count(),
            matched_l2 = tiers.iter().filter(|t| **t == TierId::L2).count(),
            cutoff = blocks.len(),
            recompute_tokens = plan.recompute_tokens,
            reason = plan.reason.as_str(),
        );
        if let Some(r) = self.requests.get_mut(&req.request) {
            r.committed = blocks.len();
        }
        let attach = PrefixAttach {
            blocks,
            cached_tokens,
            plan,
        };
        if promotions.is_empty() {
            AttachOutcome::Ready(attach)
        } else {
            self.pending
                .insert(req.request, Pending { attach, promotions });
            AttachOutcome::Promoting
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
        let (session, priority) = (r.session.clone(), r.priority);
        let keys: Vec<KvKey> = r.keys[..full].to_vec();
        r.committed = full;
        for i in start..full {
            let (key, block) = (keys[i], table[i]);
            self.dir.clear_pending(&key);
            if self.dir.get(&key).is_some() {
                continue;
            }
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
                parent: i.checked_sub(1).map(|p| keys[p]),
                child_count: 0,
                tokens: tokens[i * bt..(i + 1) * bt].into(),
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
            if let Some(id) = r.session {
                let keys = r.keys[..r.committed].to_vec();
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
                    self.stats.transfer_bytes += t.req.bytes;
                    self.metrics.transfer(
                        path,
                        t.req.bytes,
                        took.as_secs_f64(),
                        self.transfer.estimate(path).bandwidth_bps,
                    );
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
                    self.on_copy_failed(pool, t.id, &t.req);
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
                self.demoting.remove(&req.key);
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
                self.dir
                    .add_location(&req.key, KvLocation { tier: to, slot });
                self.metrics.demotion(from, to);
                self.stats.demotions += 1;
                self.remove_copy(pool, &req.key, from, EvictReason::Pressure);
            }
            TransferPurpose::Promote | TransferPurpose::Prefetch => {
                let block = BlockId(req.dst_slot as u32);
                if self.dir.get(&req.key).is_some() {
                    pool.set_keyed(block);
                    self.l0_keys.insert(block, req.key);
                    self.dir.add_location(
                        &req.key,
                        KvLocation {
                            tier: TierId::L0,
                            slot: u64::from(block.0),
                        },
                    );
                }
                self.metrics.promotion(from, TierId::L0);
                self.stats.promotions += 1;
                if req.purpose == TransferPurpose::Prefetch {
                    self.prefetch_inflight.remove(&ticket);
                    self.prefetch.issued(req.key);
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

    fn on_copy_failed(&mut self, pool: &mut BlockPool, ticket: u64, req: &TransferRequest) {
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
        // would be dropped.
        let retrieval = match demotion_target(tier, self.l1.is_some(), self.l2.is_some()) {
            Some(to) => {
                let path = TransferPath::between(to, TierId::L0).expect("lower tier to L0");
                self.transfer
                    .estimate(path)
                    .block_seconds(self.cfg.block_bytes)
            }
            None => recompute,
        };
        BlockScoreInputs {
            block: KvBlockSummary::of(b),
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

    /// Carries out the reclaim requests the pressure controller stored in the handle; returns
    /// the bytes scheduled or freed.
    pub fn apply_reclaim(&mut self, pool: &mut BlockPool) -> u64 {
        let mut bytes = 0;
        if let Some(t) = KvReclaimHandle::take(&self.reclaim.demote_target) {
            bytes += self.demote_to(pool, t, EvictReason::Pressure);
        }
        if let Some(t) = KvReclaimHandle::take(&self.reclaim.free_target) {
            bytes += self.demote_to(pool, t, EvictReason::Pressure);
        }
        bytes
    }

    fn inflight_into(&self, tier: TierId) -> usize {
        self.demoting.values().filter(|(_, to)| *to == tier).count()
    }

    /// Demotes (copy first, free on completion) the lowest-value unreferenced L0 blocks until
    /// L0 utilisation would be ≤ `target`. Blocks scoring below `kv.demote_min_value`, or with
    /// no lower tier, are dropped. Returns the bytes scheduled or freed.
    ///
    /// Under `capacity` (the orchestrator's continuous headroom keeping) only blocks with reuse
    /// evidence ([`has_reuse_evidence`]) are worth a copy (provisional decision "Phase 4:
    /// capacity demotion only for blocks with reuse evidence"): the others stay cached in L0
    /// until an allocation reclaims them, at no cost, and at most [`CAPACITY_BATCH`] blocks move
    /// per call. Pressure reclaim (the Phase 3 controller) takes every unreferenced block in
    /// value order, as the eviction policy ranks them.
    pub fn demote_to(&mut self, pool: &mut BlockPool, target: f64, reason: EvictReason) -> u64 {
        let total = f64::from(pool.total_blocks());
        let leaving = self
            .demoting
            .values()
            .filter(|(from, _)| *from == TierId::L0)
            .count() as f64;
        let need = (f64::from(pool.used_blocks()) - leaving - target * total)
            .ceil()
            .max(0.0) as usize;
        if need == 0 {
            return 0;
        }
        self.sync_l0_refs(pool);
        let capacity = reason == EvictReason::Capacity;
        let victims = if capacity {
            self.victims_where(
                pool,
                TierId::L0,
                need.min(CAPACITY_BATCH),
                has_reuse_evidence,
            )
        } else {
            self.victims(pool, TierId::L0, need)
        };
        let to = self.l0_demotion_target();
        let mut room = to.map_or(0, |t| self.make_room(pool, t, victims.len()));
        let bb = self.cfg.block_bytes;
        let mut bytes = 0;
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
            if room == 0 {
                break;
            }
            if self.submit_demotion(pool, key, TierId::L0, to) {
                room -= 1;
                bytes += bb;
            }
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
            } else if room > 0 && self.submit_demotion(pool, k, TierId::L0, to) {
                room -= 1;
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
        let Some(src_slot) = self
            .dir
            .get(&key)
            .and_then(|b| b.location(from))
            .map(|l| l.slot)
        else {
            return false;
        };
        let path = TransferPath::between(from, to).expect("a demotion path exists");
        let req = TransferRequest {
            path,
            key,
            bytes: self.cfg.block_bytes,
            owner: None,
            purpose: TransferPurpose::Demote,
            src_slot,
            dst_slot: 0,
        };
        if self.transfer.submit(req).is_err() {
            return false;
        }
        if from == TierId::L0 {
            pool.incref(BlockId(src_slot as u32));
        }
        self.demoting.insert(key, (from, to));
        true
    }

    /// Makes room for up to `n` more blocks in `tier` and returns how many fit now. L1 victims
    /// without an L2 copy move to L2 while storage accepts them (room appears when that copy
    /// completes); other victims are evicted now.
    fn make_room(&mut self, pool: &mut BlockPool, tier: TierId, n: usize) -> usize {
        let Some(t) = self.tier(tier).cloned() else {
            return 0;
        };
        let bb = self.cfg.block_bytes.max(1);
        let free = ((t.capacity_bytes() / bb) as usize)
            .saturating_sub((t.used_bytes() / bb) as usize + self.inflight_into(tier));
        if free >= n {
            return n;
        }
        let spill = tier == TierId::L1 && self.storage_accepts_demotions();
        let mut room = free;
        for (victim, _) in self.victims(pool, tier, n - free) {
            let has_l2 = self
                .dir
                .get(&victim)
                .is_some_and(|b| b.location(TierId::L2).is_some());
            if spill
                && !has_l2
                && self.make_room(pool, TierId::L2, 1) > 0
                && self.submit_demotion(pool, victim, TierId::L1, TierId::L2)
            {
                continue;
            }
            self.remove_copy(pool, &victim, tier, EvictReason::Capacity);
            room += 1;
        }
        room
    }

    /// Session TTLs, expiry and predicted-resume prefetch; call once per iteration.
    pub fn tick(&mut self, pool: &mut BlockPool) {
        self.sync_l0_refs(pool);
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
                        if movable
                            && !self.demoting.contains_key(&k)
                            && !self.transfer.is_busy_with(&k)
                            && self.make_room(pool, TierId::L2, 1) > 0
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
            let src_slot = b.location(from).expect("the fastest tier has a copy").slot;
            let Ok(ids) = pool.allocate(1) else { break };
            let req = TransferRequest {
                path: TransferPath::between(from, TierId::L0).expect("lower tier to L0"),
                key: *key,
                bytes: self.cfg.block_bytes,
                owner: None,
                purpose: TransferPurpose::Prefetch,
                src_slot,
                dst_slot: u64::from(ids[0].0),
            };
            match self.transfer.submit(req) {
                Ok(t) => {
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
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let fmt = fmt16();
        let bb = fmt.layout.block_bytes();
        let l1 = l1.map(|t| t as Arc<dyn KvTier>);
        let l2 = l2.map(|t| t as Arc<dyn KvTier>);
        // The rig pages at 16 tokens (fmt16), not the 128-token default.
        let kv = KvConfig {
            block_tokens: fmt.layout.block_tokens,
            policy: turbine_core::config::ModuleName::new(policy).expect("a module name"),
            ..KvConfig::default()
        };
        let cfg = HierarchyConfig::from_config(&kv, bb, MemoryKind::Dedicated)
            .expect("a registered eviction policy");
        let model = ModelIdentity {
            config_hash: [1; 32],
            weights_index_hash: [2; 32],
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
        let prompt: Vec<u32> = (0..65).collect();

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

    /// The reclaim handle answers from the published snapshot and the engine thread applies it.
    #[test]
    fn reclaim_handle_requests_are_applied_at_the_boundary() {
        let clock = FakeClock::new(Duration::ZERO);
        let arc: Arc<dyn Clock> = Arc::new(clock.clone());
        let bb = fmt16().layout.block_bytes();
        let l1 = Arc::new(MemTier::new(TierId::L1, 16 * bb, arc));
        let mut r = rig(8, Some(l1), None, clock);
        let prompt: Vec<u32> = (0..65).collect();
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
        r.h.poll(&mut r.pool, &mut r.backend);
        r.clock.advance(Duration::from_millis(10));
        r.h.poll(&mut r.pool, &mut r.backend);
        assert_eq!(r.pool.used_blocks(), 2);
        assert_eq!(handle.free_unreferenced(0.0), 2 * bb);
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
        let one_off: Vec<u32> = (0..65).collect();
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
        let session_prompt: Vec<u32> = (1000..1065).collect();
        let id = RequestId::new_v4();
        let req = AttachRequest {
            request: id,
            prompt: &session_prompt,
            cache_salt: "",
            session: Some(&hints),
            priority: Priority(0),
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

        // Pressure reclaim (the Phase 3 controller) is not gated: one-off blocks leave L0 too.
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
}
