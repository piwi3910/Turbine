//! Local KV directory (P4 S-2): the only owner of block metadata; tiers own bytes.
//!
//! Lookup walks a prompt's full blocks through the key chain and stops at the first block that is
//! missing or whose stored tokens differ from the prompt (a hash collision is a miss, never
//! another sequence's KV). Eviction eligibility is leaf-first: a copy may leave a tier only while
//! no child is cached in the same or a faster tier.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use smallvec::SmallVec;
use turbine_core::types::{BlockId, ModelFingerprint, PriorityClass};

use crate::identity::{KeyHasher, KvFormat, KvKey};
use crate::metrics::PartialOutcome;
use crate::tier::{KvLocation, TierId};

/// Monotonic time from the engine clock.
pub type Timestamp = Duration;

/// Token positions `[start, end)` a block covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenRange {
    pub start: u32,
    pub end: u32,
}

/// Eviction weight of a block's most recent user (P4 §Cost-aware value terms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvPriority(pub f32);

impl KvPriority {
    /// CONFLICT C-10: High 2.0, Normal 1.0, Low 0.5.
    pub fn from_class(c: PriorityClass) -> Self {
        match c {
            PriorityClass::High => KvPriority(2.0),
            PriorityClass::Low => KvPriority(0.5),
            // `PriorityClass` is non-exhaustive; a future class weighs like Normal.
            _ => KvPriority(1.0),
        }
    }
}

/// A cost in seconds.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CostEstimate {
    pub seconds: f64,
}

/// A session: the `prompt_cache_key` value plus the cache salt of the session's first request.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId {
    pub key: String,
    pub salt: String,
}

/// Whether a block's KV was computed over an exact prefix (P6b S-3). A block computed over a
/// prefix with any lossy block — or a lossy copy promoted into L0 — is `Lossy`: its key chains
/// from a [`crate::identity::lossy_key`], so it never aliases the exact block of the same tokens,
/// and a request that opted out of lossy reuse is never attached to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Lineage {
    #[default]
    Exact,
    /// `format`: the codec of the first lossy block of the prefix it was computed over (or of
    /// the promoted copy).
    Lossy { format: &'static str },
}

impl Lineage {
    pub fn is_lossy(self) -> bool {
        matches!(self, Lineage::Lossy { .. })
    }
}

/// True when a copy stored in codec `format` may differ from the L0 bytes of `l0` (an
/// unregistered name counts as lossy).
pub fn format_is_lossy(format: &str, l0: &turbine_core::types::KvLayout) -> bool {
    format != crate::tier::L0_FORMAT
        && crate::codec::registry()
            .get(format)
            .is_none_or(|c| c.lossy(l0))
}

/// TS §8 `KvBlock` plus the Phase 4 fields (P4 S-2).
#[derive(Clone, Debug)]
pub struct KvBlock {
    pub key: KvKey,
    /// TS `ModelId` (CONFLICT C-17).
    pub model: ModelFingerprint,
    pub token_range: TokenRange,
    pub format: KvFormat,
    pub size_bytes: u64,
    /// One entry per tier holding a copy.
    pub locations: SmallVec<[KvLocation; 3]>,
    pub access_count: u64,
    pub last_access: Timestamp,
    /// Sequences holding the L0 copy (mirrors the L0 pool's refcount).
    pub ref_count: u32,
    pub priority: KvPriority,
    pub recompute_cost: CostEstimate,
    /// Hit count decayed to `decayed_at` with half-life `kv.policy_weights.hit_half_life`.
    pub decayed_hits: f64,
    pub decayed_at: Timestamp,
    pub session: Option<SessionId>,
    pub parent: Option<KvKey>,
    /// Children currently in the directory.
    pub child_count: u32,
    /// The block's token ids, compared on every lookup.
    pub tokens: Box<[u32]>,
    /// P6b S-3: exact, or computed over (or promoted from) a lossy block.
    pub lineage: Lineage,
    /// P7 S-12 (Task 32): the per-sequence state a hybrid family snapshotted at this block's
    /// boundary, when it carries one. A partial entry never carries one (`insert_partial`
    /// clears it); an attention-only family's blocks never do.
    pub snapshot: Option<crate::state::SnapshotRef>,
}

impl KvBlock {
    /// True when the block holds fewer than `block_tokens` tokens: an L0-only partial entry
    /// (P5p S-4), never demoted, prefetched or shared by reference.
    pub fn is_partial(&self, block_tokens: u32) -> bool {
        (self.tokens.len() as u32) < block_tokens.max(1)
    }

    pub fn location(&self, tier: TierId) -> Option<KvLocation> {
        self.locations.iter().copied().find(|l| l.tier == tier)
    }

    /// Fastest tier holding a copy.
    pub fn fastest(&self) -> Option<TierId> {
        self.locations.iter().map(|l| l.tier).min()
    }
}

/// One matched block of a prefix lookup, at its fastest tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchedBlock {
    /// The directory entry used (the block's key, a lossy key, or a lossy-lineage key).
    pub key: KvKey,
    pub tier: TierId,
    pub location: KvLocation,
    /// `Some(codec)` when the block is served lossy: a copy in a lossy codec, or an entry of
    /// lossy lineage (P6b S-3). Never `Some` in a lookup that denied lossy reuse.
    pub lossy: Option<&'static str>,
}

/// A token-granular match beyond the matched full blocks (P5p S-4): the first `tokens` tokens
/// of the L0 block `block` (a partial entry, or a full block whose stored tokens share the
/// prompt's trailing tokens) are reusable through a tail copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialMatch {
    /// The directory entry the match was found in.
    pub key: KvKey,
    /// The block's L0 id (the copy the tail copy reads).
    pub block: BlockId,
    /// Matched tokens, at least 1.
    pub tokens: u32,
}

/// Result of a longest-prefix lookup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Consecutive matched full blocks from the start of the prompt.
    pub blocks: Vec<MatchedBlock>,
    /// Keys found whose stored tokens differ from the prompt (collision or corruption).
    pub mismatches: u32,
    /// Keys of every full block of the prompt, matched or not.
    pub keys: Vec<KvKey>,
    /// The first unmatched block, when another request registered it as being computed.
    pub pending: Option<KvKey>,
    /// A lookup that denied lossy reuse stopped at a block only a lossy copy held (P6b S-3).
    pub denied: bool,
    /// P6b S-3: the first matched block served lossy and its codec. From it on, blocks
    /// computed over the matched prefix are keyed by `lossy_chain`.
    pub lossy_from: Option<(usize, &'static str)>,
    /// With `lossy_from = Some((s, f))`: the keys of every full block, equal to `keys` before
    /// `s`, then `lossy_key(keys[s], f, seed)` chained on; empty otherwise.
    pub lossy_chain: Vec<KvKey>,
    /// P5p S-4: the token-granular match beyond the matched full blocks, when one was found.
    pub partial: Option<PartialMatch>,
    /// P5p S-4: how the token-granular comparison ended, when the prompt has tokens beyond
    /// its matched full blocks (or ends exactly on them and the tail may be split).
    pub partial_outcome: Option<PartialOutcome>,
}

impl PrefixMatch {
    /// The keys a request that reuses the first `cut` matched blocks commits its blocks under,
    /// with their lineage: the exact keys unless a block before `cut` is served lossy.
    pub fn keys_for(&self, cut: usize) -> (&[KvKey], Lineage) {
        match self.lossy_from {
            Some((s, format)) if s < cut => (&self.lossy_chain, Lineage::Lossy { format }),
            _ => (&self.keys, Lineage::Exact),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DirectoryError {
    #[error("KV directory full ({0} entries)")]
    Full(usize),
    #[error("parent block {0} is not in the KV directory")]
    MissingParent(KvKey),
}

/// How long a request waits on a block another request is computing before computing its own.
pub const PENDING_WAIT: Duration = Duration::from_secs(2);

/// Children of the last matched key a lookup's token-granular comparison (P5p S-4) compares,
/// most recently used first.
pub const PARTIAL_SCAN_LIMIT: usize = 64;

/// Partial entries kept per parent key; a further insert evicts the least recently used.
pub const PARTIAL_PER_PARENT: usize = 8;

/// True when an L0 copy in the page class `tag` over `layout` holds the sequence's exact
/// bytes (P5p S-4, the 6b interaction): the base class of a lossless-with-matched-scales L0
/// (`bf16`, `fp8_e4m3`) or the BF16 recent-window class; a TurboQuant base class and every
/// ladder rung class are lossy rung classes.
fn exact_page_class(tag: &str, layout: &turbine_core::types::KvLayout) -> bool {
    match tag {
        crate::tier::L0_FORMAT => !matches!(
            layout.dtype,
            turbine_core::types::DType::Tq4 | turbine_core::types::DType::Tq2
        ),
        "bf16" => true,
        _ => false,
    }
}

/// Length of the longest common prefix of two token slices.
fn common_prefix(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).take_while(|(x, y)| x == y).count() as u32
}

/// Block key → metadata, with a children index for leaf-first eviction.
pub struct KvDirectory {
    blocks: HashMap<KvKey, KvBlock>,
    children: HashMap<KvKey, SmallVec<[KvKey; 4]>>,
    pending: HashMap<KvKey, Timestamp>,
    max_entries: usize,
    /// Partial entries evicted by [`KvDirectory::insert_partial`]'s bounds; the caller frees
    /// their L0 slots.
    evicted: Vec<KvBlock>,
}

impl KvDirectory {
    /// `max_entries` is the sum of the enabled tiers' capacities in blocks (P4 Constraints).
    pub fn new(max_entries: usize) -> Self {
        KvDirectory {
            blocks: HashMap::new(),
            children: HashMap::new(),
            pending: HashMap::new(),
            max_entries,
            evicted: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn get(&self, key: &KvKey) -> Option<&KvBlock> {
        self.blocks.get(key)
    }

    pub fn get_mut(&mut self, key: &KvKey) -> Option<&mut KvBlock> {
        self.blocks.get_mut(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = &KvBlock> {
        self.blocks.values()
    }

    /// Inserts a block (an already present key is left unchanged). Its parent must be present.
    pub fn insert(&mut self, block: KvBlock) -> Result<(), DirectoryError> {
        if self.blocks.contains_key(&block.key) {
            return Ok(());
        }
        if self.blocks.len() >= self.max_entries {
            return Err(DirectoryError::Full(self.max_entries));
        }
        if let Some(p) = block.parent {
            let parent = self
                .blocks
                .get_mut(&p)
                .ok_or(DirectoryError::MissingParent(p))?;
            parent.child_count += 1;
            self.children.entry(p).or_default().push(block.key);
        }
        self.pending.remove(&block.key);
        self.blocks.insert(block.key, block);
        Ok(())
    }

    /// Inserts an L0-only partial entry (P5p S-4): a block holding fewer than
    /// `kv.block_tokens` tokens whose KV is written, keyed like a full block and chained from
    /// its parent. Beyond [`PARTIAL_PER_PARENT`] entries under one parent, and when the
    /// directory is full, the least recently used partial entry is evicted first (its L0 slot
    /// is handed to the caller through [`KvDirectory::evicted_partials`]).
    pub fn insert_partial(&mut self, mut block: KvBlock) -> Result<(), DirectoryError> {
        debug_assert!(
            block.location(TierId::L0).is_some(),
            "a partial entry is L0-only"
        );
        // A partial entry carries no snapshot (P7 S-12): state is restored only at full-block
        // boundaries.
        block.snapshot = None;
        let block_tokens = block.format.layout.block_tokens;
        let parent = block.parent;
        // Room: drop the LRU partial entry of this parent (any parent when the entry has no
        // parent or the directory is at capacity).
        if self.partial_children(parent, block_tokens).count() >= PARTIAL_PER_PARENT
            || self.blocks.len() >= self.max_entries
        {
            let victim = self
                .blocks
                .values()
                .filter(|b| b.parent == parent && b.is_partial(block_tokens) && b.child_count == 0)
                .min_by_key(|b| b.last_access)
                .map(|b| b.key);
            if let Some(k) = victim
                && let Some(b) = self.remove(&k)
            {
                self.evicted.push(b);
            }
        }
        self.insert(block)
    }

    /// Partial entries [`KvDirectory::insert_partial`] evicted; the caller frees their L0
    /// slots.
    pub fn evicted_partials(&mut self) -> Vec<KvBlock> {
        std::mem::take(&mut self.evicted)
    }

    /// Makes room for a full block: evicts the least recently used leaf partial entry of any
    /// parent when the directory is at capacity, handing its L0 slot to the caller through
    /// [`KvDirectory::evicted_partials`] (P5p S-5: a shareable, tierable full block is worth
    /// more than an opportunistic partial entry). False when there is no leaf partial entry.
    pub fn evict_one_partial(&mut self, block_tokens: u32) -> bool {
        if self.blocks.len() < self.max_entries {
            return false;
        }
        let victim = self
            .blocks
            .values()
            .filter(|b| b.is_partial(block_tokens) && b.child_count == 0)
            .min_by_key(|b| b.last_access)
            .map(|b| b.key);
        match victim {
            Some(k) => {
                if let Some(b) = self.remove(&k) {
                    self.evicted.push(b);
                }
                true
            }
            None => false,
        }
    }

    /// How many partial entries the directory holds.
    pub fn partial_count(&self, block_tokens: u32) -> usize {
        self.blocks
            .values()
            .filter(|b| b.is_partial(block_tokens))
            .count()
    }

    /// The partial-entry children of `parent` (the children index holds keyed parents only;
    /// a root-level entry is found by scan).
    fn partial_children(
        &self,
        parent: Option<KvKey>,
        block_tokens: u32,
    ) -> impl Iterator<Item = &KvBlock> {
        let kids: SmallVec<[KvKey; 4]> = parent
            .and_then(|p| self.children.get(&p).cloned())
            .unwrap_or_default();
        let source: Vec<&KvBlock> = if parent.is_none() {
            self.blocks
                .values()
                .filter(|b| b.parent.is_none())
                .collect()
        } else {
            kids.iter().filter_map(|k| self.blocks.get(k)).collect()
        };
        source
            .into_iter()
            .filter(move |b| b.is_partial(block_tokens))
    }

    /// Removes a block that has no children left and returns it; `None` if absent or a parent.
    pub fn remove(&mut self, key: &KvKey) -> Option<KvBlock> {
        if self.blocks.get(key)?.child_count > 0 {
            return None;
        }
        let b = self.blocks.remove(key)?;
        self.children.remove(key);
        if let Some(p) = b.parent {
            if let Some(parent) = self.blocks.get_mut(&p) {
                parent.child_count -= 1;
            }
            if let Some(kids) = self.children.get_mut(&p) {
                kids.retain(|k| k != key);
            }
        }
        Some(b)
    }

    /// Records a copy in `loc.tier`, replacing any previous copy in that tier.
    pub fn add_location(&mut self, key: &KvKey, loc: KvLocation) {
        if let Some(b) = self.blocks.get_mut(key) {
            b.locations.retain(|l| l.tier != loc.tier);
            b.locations.push(loc);
        }
    }

    /// Records (`Some`) or clears (`None`) the snapshot a hybrid block carries (P7 S-12). A
    /// full block's `SnapshotRef` is compared on lookups and reused with it; a partial entry's
    /// is always cleared by [`KvDirectory::insert_partial`].
    pub fn set_snapshot(&mut self, key: &KvKey, snapshot: Option<crate::state::SnapshotRef>) {
        if let Some(b) = self.blocks.get_mut(key) {
            b.snapshot = snapshot;
        }
    }

    /// Forgets the copy in `tier`; returns the number of copies left (`None` for an unknown key).
    pub fn remove_location(&mut self, key: &KvKey, tier: TierId) -> Option<usize> {
        let b = self.blocks.get_mut(key)?;
        b.locations.retain(|l| l.tier != tier);
        Some(b.locations.len())
    }

    /// Longest cached prefix of `tokens` across all tiers, each block at its fastest tier.
    /// Matched blocks get `last_access = now`; a stored-token mismatch is logged and is a miss.
    ///
    /// Lossy reuse (P6b S-3): the walk follows the exact chain, taking each block's fastest
    /// exact copy. Where none exists and `allow_lossy`, it continues on the block's fastest
    /// lossy copy — a lossy-format copy of the block, or its promoted copy under
    /// `lossy_key(key, codec, seed)` — and from then on also on the blocks computed over that
    /// lossy prefix (keyed by [`PrefixMatch::lossy_chain`], or by another request's chain that
    /// started at a later block: reached as lossy-lineage children of the previous block's
    /// matching entries). Without `allow_lossy` it stops at
    /// the first block only a lossy copy holds (`denied`).
    pub fn lookup(
        &mut self,
        hasher: &dyn KeyHasher,
        tokens: &[u32],
        block_tokens: u32,
        now: Timestamp,
        allow_lossy: bool,
        seed: u64,
    ) -> PrefixMatch {
        let mut m = PrefixMatch::default();
        let mut matching = true;
        // The previous block's entries whose tokens matched: their lossy-lineage children are
        // candidates too (see below).
        let mut prev: SmallVec<[KvKey; 8]> = SmallVec::new();
        for (i, chunk) in tokens.chunks_exact(block_tokens as usize).enumerate() {
            let key = hasher.key(m.keys.last(), chunk);
            m.keys.push(key);
            let chain = m
                .lossy_from
                .map(|_| hasher.key(m.lossy_chain.last(), chunk));
            if let Some(c) = chain {
                m.lossy_chain.push(c);
            }
            if !matching {
                continue;
            }
            // Candidates: the block's own entry, its promoted lossy copies, and (on a lossy
            // prefix) the block computed over that prefix.
            let mut cands: SmallVec<[KvKey; 8]> = SmallVec::new();
            cands.push(key);
            for c in crate::codec::registry().iter() {
                let lk = crate::identity::lossy_key(key, c.name(), seed);
                if self.blocks.contains_key(&lk) {
                    cands.push(lk);
                }
            }
            cands.extend(chain);
            // A block computed over a lossy prefix is keyed by the chain of the request that
            // computed it, which starts at *that* request's first lossy block; once an earlier
            // block has turned lossy too (history leaves L0 leaf-first), this lookup's own
            // chain starts earlier and never names it. Its directory parent is the entry its
            // predecessor was served from, so it is reached through the children of the
            // previous block's matching entries.
            for p in &prev {
                for c in self.children.get(p).into_iter().flatten() {
                    if !cands.contains(c)
                        && self.blocks.get(c).is_some_and(|b| b.lineage.is_lossy())
                    {
                        cands.push(*c);
                    }
                }
            }
            let mut here: SmallVec<[KvKey; 8]> = SmallVec::new();
            // (entry, location, lossy codec) of the best exact and the best lossy copy.
            let mut exact: Option<(KvKey, KvLocation)> = None;
            let mut lossy: Option<(KvKey, KvLocation, &'static str)> = None;
            let mut lossy_seen = false;
            let mut mismatch = false;
            for k in &cands {
                let Some(b) = self.blocks.get(k) else {
                    continue;
                };
                if *b.tokens != *chunk {
                    mismatch = true;
                    continue;
                }
                here.push(*k);
                for loc in &b.locations {
                    let codec = match b.lineage {
                        Lineage::Lossy { format } => Some(format),
                        Lineage::Exact if format_is_lossy(loc.format, &b.format.layout) => {
                            Some(loc.format)
                        }
                        Lineage::Exact => None,
                    };
                    match codec {
                        None => {
                            if exact.is_none_or(|(_, e)| loc.tier < e.tier) {
                                exact = Some((*k, *loc));
                            }
                        }
                        Some(f) => {
                            lossy_seen = true;
                            if allow_lossy && lossy.is_none_or(|(_, l, _)| loc.tier < l.tier) {
                                lossy = Some((*k, *loc, f));
                            }
                        }
                    }
                }
            }
            let chosen = match (exact, lossy) {
                (Some((k, loc)), _) => Some((k, loc, None)),
                (None, Some((k, loc, f))) => Some((k, loc, Some(f))),
                (None, None) => None,
            };
            prev = here;
            match chosen {
                Some((k, location, codec)) => {
                    if let Some(b) = self.blocks.get_mut(&k) {
                        b.last_access = now;
                    }
                    // After the switch an exact copy is still its block's exact KV: only
                    // lossy copies and lossy-lineage blocks count as served lossy.
                    let served_lossy = codec;
                    if m.lossy_from.is_none()
                        && let Some(f) = served_lossy
                    {
                        m.lossy_from = Some((i, f));
                        m.lossy_chain = m.keys[..i].to_vec();
                        m.lossy_chain.push(crate::identity::lossy_key(key, f, seed));
                    }
                    m.blocks.push(MatchedBlock {
                        key: k,
                        tier: location.tier,
                        location,
                        lossy: served_lossy,
                    });
                }
                None => {
                    if mismatch {
                        m.mismatches += 1;
                        tracing::warn!(
                            event = "kv_key_mismatch",
                            key = %key,
                            "cached block tokens differ from the prompt; treated as a miss"
                        );
                    } else if lossy_seen && !allow_lossy {
                        m.denied = true;
                    } else if !self.blocks.contains_key(&chain.unwrap_or(key)) {
                        let pending = chain.unwrap_or(key);
                        let computing = self
                            .pending
                            .get(&pending)
                            .is_some_and(|since| now.saturating_sub(*since) < PENDING_WAIT);
                        if computing {
                            m.pending = Some(pending);
                        }
                    }
                    matching = false;
                }
            }
        }
        self.compare_tail(&mut m, hasher, tokens, block_tokens, allow_lossy, seed);
        m
    }

    /// The token-granular comparison of the prompt beyond its matched full blocks (P5p S-4):
    /// the remaining prompt tokens against the stored tokens of the L0-resident children of
    /// the last matched key (the root parent when none matched) — full blocks and partial
    /// entries, most recently used first, at most [`PARTIAL_SCAN_LIMIT`] of them. The longest
    /// common prefix (at least one token) fills `partial`; only children resident in an exact
    /// page class ([`exact_page_class`]) and of exact lineage are sources.
    fn compare_tail(
        &self,
        m: &mut PrefixMatch,
        hasher: &dyn KeyHasher,
        tokens: &[u32],
        block_tokens: u32,
        _allow_lossy: bool,
        _seed: u64,
    ) {
        let base = m.blocks.len() as u32 * block_tokens;
        let remaining = &tokens[base as usize..];
        if remaining.is_empty() {
            return;
        }
        let parent = m.blocks.last().map(|b| b.key);
        let mut kids: SmallVec<[KvKey; PARTIAL_SCAN_LIMIT]> = match parent {
            Some(p) => self
                .children
                .get(&p)
                .into_iter()
                .flatten()
                .copied()
                .collect(),
            None => {
                // The root scan compares raw token ids, so it must not cross a cache salt's
                // namespace: a candidate passes only when its tokens hash to its own key under
                // THIS lookup's namespace (a full-block walk is key-driven and safe by
                // construction).
                self.blocks
                    .values()
                    .filter(|b| b.parent.is_none() && hasher.key(None, &b.tokens) == b.key)
                    .map(|b| b.key)
                    .collect()
            }
        };
        if kids.len() > PARTIAL_SCAN_LIMIT {
            kids.sort_by(|a, b| {
                let at = self.blocks.get(a).map(|b| b.last_access);
                let bt = self.blocks.get(b).map(|b| b.last_access);
                bt.cmp(&at)
            });
            kids.truncate(PARTIAL_SCAN_LIMIT);
        }
        let mut best: Option<(KvKey, KvLocation, u32)> = None;
        let mut lossy = false;
        let mut not_resident = false;
        for k in kids {
            let Some(b) = self.blocks.get(&k) else {
                continue;
            };
            let lcp = common_prefix(&b.tokens, remaining);
            if lcp == 0 {
                continue;
            }
            if b.lineage.is_lossy() {
                lossy = true;
                continue;
            }
            let Some(loc) = b.location(TierId::L0) else {
                not_resident = true;
                continue;
            };

            if !exact_page_class(loc.format, &b.format.layout) {
                lossy = true;
                continue;
            }
            if best.as_ref().is_none_or(|(_, _, l)| lcp > *l) {
                best = Some((k, loc, lcp));
            }
        }
        m.partial = best.map(|(key, loc, tokens)| PartialMatch {
            key,
            block: BlockId(loc.slot as u32),
            tokens,
        });
        m.partial_outcome = Some(if m.partial.is_some() {
            PartialOutcome::Reused
        } else if lossy {
            PartialOutcome::LossySource
        } else if not_resident {
            PartialOutcome::NotResident
        } else {
            PartialOutcome::NoCandidate
        });
    }

    /// Registers a block a running prefill is computing (the first registration's time counts).
    pub fn register_pending(&mut self, key: KvKey, now: Timestamp) {
        self.pending.entry(key).or_insert(now);
    }

    pub fn clear_pending(&mut self, key: &KvKey) {
        self.pending.remove(key);
    }

    /// One reuse of a block: access count, decayed hits (+1 after decay), last access.
    pub fn record_hit(&mut self, key: &KvKey, now: Timestamp, half_life: Duration) {
        if let Some(b) = self.blocks.get_mut(key) {
            b.decayed_hits = decay(b.decayed_hits, b.decayed_at, now, half_life) + 1.0;
            b.decayed_at = now;
            b.access_count += 1;
            b.last_access = now;
        }
    }

    /// Blocks whose copy in `tier` may leave it now (see [`KvDirectory::evictable`]).
    pub fn candidates(&self, tier: TierId, departing: &HashSet<KvKey>) -> Vec<&KvBlock> {
        self.blocks
            .values()
            .filter(|b| self.evictable(b, tier, departing))
            .collect()
    }

    /// The copy of `b` in `tier` exists, is not referenced by a running sequence (L0), and no
    /// child outside `departing` (copies already chosen to leave) is cached in `tier` or faster.
    pub fn evictable(&self, b: &KvBlock, tier: TierId, departing: &HashSet<KvKey>) -> bool {
        b.location(tier).is_some()
            && !(tier == TierId::L0 && b.ref_count > 0)
            && self.children.get(&b.key).is_none_or(|kids| {
                kids.iter().all(|k| {
                    departing.contains(k)
                        || self
                            .blocks
                            .get(k)
                            .is_none_or(|c| c.locations.iter().all(|l| l.tier > tier))
                })
            })
    }

    /// True when a child of `key` has a copy in `tier` (a parent the leaf-first rule holds
    /// there while that child stays).
    pub fn has_child_in(&self, key: &KvKey, tier: TierId) -> bool {
        self.children.get(key).is_some_and(|kids| {
            kids.iter().any(|k| {
                self.blocks
                    .get(k)
                    .is_some_and(|c| c.location(tier).is_some())
            })
        })
    }

    /// A location found pointing at a freed slot: logged, removed; debug builds assert.
    pub fn stale_location(&mut self, key: &KvKey, tier: TierId) {
        tracing::error!(
            event = "kv_directory_inconsistent",
            key = %key,
            tier = tier.as_str(),
            "KV directory location pointed at a freed slot"
        );
        self.remove_location(key, tier);
        debug_assert!(
            false,
            "KV directory location {key} in {} pointed at a freed slot",
            tier.as_str()
        );
    }
}

/// `hits` recorded at `at`, decayed to `now` with the given half-life (0 half-life → 0).
pub fn decay(hits: f64, at: Timestamp, now: Timestamp, half_life: Duration) -> f64 {
    if half_life.is_zero() {
        return 0.0;
    }
    let dt = now.saturating_sub(at).as_secs_f64();
    hits * 0.5f64.powf(dt / half_life.as_secs_f64())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::identity::{Blake3Hasher, NamespaceKey, ROOT_PARENT, prefix_keys};
    use crate::metrics::KvMetrics;
    use crate::test_log;
    use turbine_core::types::{DType, KvDtype, KvLayout, Priority};

    /// A small 16-token-block format (2 layers × 2 heads × 16 dims, BF16).
    pub(crate) fn fmt16() -> KvFormat {
        KvFormat {
            dtype: KvDtype::Bf16,
            layout: KvLayout {
                num_layers: 2,
                num_kv_heads: 2,
                head_dim: 16,
                dtype: DType::BF16,
                block_tokens: 16,
            },
            shards: 1,
            scales: None,
        }
    }

    /// The same small format with 128-token blocks (the production `kv.block_tokens`).
    pub(crate) fn fmt128() -> KvFormat {
        KvFormat {
            layout: KvLayout {
                block_tokens: 128,
                ..fmt16().layout
            },
            ..fmt16()
        }
    }

    pub(crate) fn l0(slot: u64) -> KvLocation {
        KvLocation {
            tier: TierId::L0,
            slot,
            format: crate::tier::L0_FORMAT,
        }
    }

    /// Block `index` of a prompt, unreferenced, with the given copies.
    pub(crate) fn block(
        key: KvKey,
        parent: Option<KvKey>,
        index: u32,
        tokens: &[u32],
        locs: &[KvLocation],
    ) -> KvBlock {
        KvBlock {
            key,
            model: ModelFingerprint([0; 32]),
            token_range: TokenRange {
                start: index * 16,
                end: index * 16 + 16,
            },
            format: fmt16(),
            size_bytes: fmt16().layout.block_bytes(),
            locations: locs.iter().copied().collect(),
            access_count: 0,
            last_access: Duration::ZERO,
            ref_count: 0,
            priority: KvPriority(1.0),
            recompute_cost: CostEstimate::default(),
            decayed_hits: 0.0,
            decayed_at: Duration::ZERO,
            session: None,
            parent,
            child_count: 0,
            tokens: tokens.into(),
            lineage: Lineage::Exact,
            snapshot: None,
        }
    }

    /// Maps every block to the same key, forcing a collision.
    struct Colliding;

    impl KeyHasher for Colliding {
        fn key(&self, _parent: Option<&KvKey>, _tokens: &[u32]) -> KvKey {
            KvKey([0xab; 16])
        }
    }

    #[test]
    fn collision_is_a_miss() {
        let a: Vec<u32> = (0..16).collect();
        let b: Vec<u32> = (100..116).collect();
        let mut dir = KvDirectory::new(16);
        dir.insert(block(Colliding.key(None, &a), None, 0, &a, &[l0(3)]))
            .unwrap();
        let hit = dir.lookup(&Colliding, &a, 16, Duration::ZERO, true, 0);
        assert_eq!(hit.blocks.len(), 1, "the same tokens hit");
        assert_eq!(hit.blocks[0].location, l0(3));

        let metrics = KvMetrics::unregistered();
        let (m, logs) =
            test_log::capture(|| dir.lookup(&Colliding, &b, 16, Duration::ZERO, true, 0));
        assert!(
            m.blocks.is_empty(),
            "a colliding key must never return another sequence's KV"
        );
        assert_eq!(m.mismatches, 1);
        metrics.record_lookup(&m);
        assert_eq!(
            metrics.lookups_value(None),
            1,
            "counted as lookups{{result=\"miss\"}}"
        );
        assert!(
            logs.contains("kv_key_mismatch"),
            "mismatch is logged: {logs}"
        );
        assert!(logs.contains("WARN"), "at WARN: {logs}");
    }

    #[test]
    fn longest_prefix_across_tiers() {
        let tokens: Vec<u32> = (0..16 * 9).collect();
        let h = Blake3Hasher(NamespaceKey([1; 32]));
        let keys = prefix_keys(&h, &tokens, 16);
        let mut dir = KvDirectory::new(64);
        for i in 0..7usize {
            let tier = match i {
                0..=3 => TierId::L0,
                4 | 5 => TierId::L1,
                _ => TierId::L2,
            };
            let parent = (i > 0).then(|| keys[i - 1]);
            let loc = KvLocation {
                tier,
                slot: i as u64,
                format: crate::tier::L0_FORMAT,
            };
            dir.insert(block(
                keys[i],
                parent,
                i as u32,
                &tokens[i * 16..i * 16 + 16],
                &[loc],
            ))
            .unwrap();
        }
        // Block 8 is cached while block 7 is missing: the lookup must stop at the gap.
        dir.insert(block(keys[8], None, 8, &tokens[128..144], &[l0(8)]))
            .unwrap();
        let m = dir.lookup(&h, &tokens, 16, Duration::from_secs(1), true, 0);
        let tiers: Vec<TierId> = m.blocks.iter().map(|b| b.tier).collect();
        use TierId::{L0, L1, L2};
        assert_eq!(tiers, [L0, L0, L0, L0, L1, L1, L2]);
        assert_eq!(m.keys, keys, "keys of every full block");
        assert_eq!(
            dir.get(&keys[0]).unwrap().last_access,
            Duration::from_secs(1)
        );

        // A block with copies in L0 and L2 is reported in L0, the fastest tier.
        dir.add_location(&keys[6], l0(60));
        let m = dir.lookup(&h, &tokens, 16, Duration::from_secs(2), true, 0);
        assert_eq!((m.blocks[6].tier, m.blocks[6].location), (L0, l0(60)));

        // The metrics count one lookup per full block: its fastest tier, or a miss.
        let metrics = KvMetrics::unregistered();
        metrics.record_lookup(&m);
        assert_eq!(
            [Some(L0), Some(L1), Some(L2), None].map(|t| metrics.lookups_value(t)),
            [5, 2, 0, 2]
        );
    }

    #[test]
    fn root_partial_never_crosses_a_cache_salt() {
        // The root-parent scan of the token-granular comparison compares raw token ids, so it
        // must not hand another cache salt's (namespace's) root entry over as a match — the
        // full-block walk is key-driven and safe by construction. Breaks if a salted lookup
        // matches another namespace's root partial entry (the leak `cache_salt_isolates`
        // caught when Task 7 started consuming the matches).
        let other = Blake3Hasher(NamespaceKey([7; 32]));
        let tokens: Vec<u32> = (0..20).collect();
        let mut dir = KvDirectory::new(16);
        dir.insert_partial(block(other.key(None, &tokens), None, 0, &tokens, &[l0(0)]))
            .unwrap();
        let mine = Blake3Hasher(NamespaceKey([9; 32]));
        let m = dir.lookup(&mine, &tokens, 16, Duration::from_secs(1), true, 0);
        assert_eq!(m.partial, None, "another salt's entry is not a match");
        assert_eq!(
            m.partial_outcome,
            Some(PartialOutcome::NoCandidate),
            "the scan sees nothing of the other namespace"
        );
        // A same-prompt root entry under this namespace still matches with 20.
        dir.insert_partial(block(mine.key(None, &tokens), None, 0, &tokens, &[l0(1)]))
            .unwrap();
        let m = dir.lookup(&mine, &tokens, 16, Duration::from_secs(2), true, 0);
        assert_eq!(m.partial.map(|p| p.tokens), Some(20));
        assert_eq!(m.partial_outcome, Some(PartialOutcome::Reused));
    }

    #[test]
    fn pending_blocks_and_leaf_first_eligibility() {
        let tokens: Vec<u32> = (0..48).collect();
        let h = Blake3Hasher(NamespaceKey([2; 32]));
        let keys = prefix_keys(&h, &tokens, 16);
        let mut dir = KvDirectory::new(2);
        dir.insert(block(keys[0], None, 0, &tokens[..16], &[l0(0)]))
            .unwrap();
        assert_eq!(
            dir.insert(block(keys[2], Some(keys[1]), 2, &tokens[32..], &[l0(2)])),
            Err(DirectoryError::MissingParent(keys[1]))
        );
        // A block another request is computing is reported for up to PENDING_WAIT.
        dir.register_pending(keys[1], Duration::from_secs(10));
        let m = dir.lookup(&h, &tokens, 16, Duration::from_secs(11), true, 0);
        assert_eq!((m.blocks.len(), m.pending), (1, Some(keys[1])));
        let m = dir.lookup(&h, &tokens, 16, Duration::from_secs(12), true, 0);
        assert_eq!(m.pending, None, "after 2 s the request computes its own");

        dir.insert(block(keys[1], Some(keys[0]), 1, &tokens[16..32], &[l0(1)]))
            .unwrap();
        assert_eq!(dir.get(&keys[0]).unwrap().child_count, 1);
        assert_eq!(
            dir.insert(block(keys[2], Some(keys[1]), 2, &tokens[32..], &[l0(2)])),
            Err(DirectoryError::Full(2)),
            "bounded by max_entries"
        );

        let none = HashSet::new();
        let evictable: Vec<KvKey> = dir
            .candidates(TierId::L0, &none)
            .iter()
            .map(|b| b.key)
            .collect();
        assert_eq!(evictable, [keys[1]], "only the leaf may leave L0");
        let departing: HashSet<KvKey> = [keys[1]].into();
        assert!(
            dir.evictable(dir.get(&keys[0]).unwrap(), TierId::L0, &departing),
            "a parent whose child is already leaving becomes eligible"
        );
        dir.get_mut(&keys[1]).unwrap().ref_count = 1;
        assert!(dir.candidates(TierId::L0, &none).is_empty(), "referenced");
        assert!(
            dir.remove(&keys[0]).is_none(),
            "a parent with children stays"
        );

        // Hits decay with the half-life.
        let half = Duration::from_secs(60);
        dir.record_hit(&keys[0], Duration::from_secs(0), half);
        dir.record_hit(&keys[0], Duration::from_secs(60), half);
        let b = dir.get(&keys[0]).unwrap();
        assert_eq!((b.access_count, b.decayed_hits), (2, 1.5));
        assert_eq!(
            decay(1.5, b.decayed_at, Duration::from_secs(180), half),
            0.375
        );

        // Removing the copy of a stale location keeps the metadata consistent.
        assert_eq!(dir.remove_location(&keys[1], TierId::L0), Some(0));
        assert_eq!(dir.get(&keys[1]).unwrap().fastest(), None);
        dir.get_mut(&keys[1]).unwrap().ref_count = 0;
        assert!(dir.remove(&keys[1]).is_some());
        assert_eq!(dir.get(&keys[0]).unwrap().child_count, 0);
        assert_eq!(dir.len(), 1);

        // Eviction weight of a request priority (CONFLICT C-10).
        let weights = [-3, 0, 5].map(|p| KvPriority::from_class(Priority(p).class()));
        assert_eq!(weights, [KvPriority(2.0), KvPriority(1.0), KvPriority(0.5)]);
    }

    /// P5p S-4 (`token_granular_match`): after the full-block walk, `lookup` compares the
    /// prompt's remaining tokens with the L0-resident children of the last matched key —
    /// full blocks and partial entries — and returns the longest common prefix; collisions
    /// never match, an L1/L2-only candidate is `not_resident`, a lossy-class or
    /// lossy-lineage child is `lossy_source` with no partial, the scan is bounded, and the
    /// ninth partial entry under one parent evicts the least recently used one. Breaks if a
    /// partial match returns another sequence's tokens, scans unbounded, or grows unbounded.
    #[test]
    fn token_granular_match() {
        use turbine_core::types::BlockId;
        let h = Blake3Hasher(NamespaceKey([5; 32]));

        /// A partial entry (a block holding fewer tokens than its 128-token format) at `slot`.
        fn partial_entry(h: &Blake3Hasher, parent: KvKey, tail: &[u32], slot: u64) -> KvBlock {
            let mut b = block(
                h.key(Some(&parent), tail),
                Some(parent),
                0,
                tail,
                &[l0(slot)],
            );
            b.format = fmt128();
            let end = tail.len() as u32;
            b.token_range = TokenRange { start: 0, end };
            b
        }

        /// A full-block directory with the cached 128-token blocks of `tokens`, block
        /// `l1_index` at L1 only.
        fn chain(h: &Blake3Hasher, tokens: &[u32], l1_index: Option<usize>) -> KvDirectory {
            let keys = prefix_keys(h, tokens, 128);
            let mut dir = KvDirectory::new(256);
            for i in 0..keys.len() {
                let parent = (i > 0).then(|| keys[i - 1]);
                let loc = if Some(i) == l1_index {
                    KvLocation {
                        tier: TierId::L1,
                        slot: i as u64,
                        format: crate::tier::L0_FORMAT,
                    }
                } else {
                    l0(i as u64)
                };
                dir.insert(block(
                    keys[i],
                    parent,
                    i as u32,
                    &tokens[i * 128..i * 128 + 128],
                    &[loc],
                ))
                .unwrap();
            }
            dir
        }

        // A cached chain of three 128-token blocks plus a 40-token partial entry under the
        // third block's key; a prompt sharing the chain and 25 of the 40 tokens matches it
        // with 25.
        let cached: Vec<u32> = (0..384).collect();
        let keys = prefix_keys(&h, &cached, 128);
        let mut dir = chain(&h, &cached, None);
        let tail40: Vec<u32> = (384..424).collect();
        let pkey = h.key(Some(&keys[2]), &tail40);
        let mut pe = partial_entry(&h, keys[2], &tail40, 9);
        pe.token_range = TokenRange {
            start: 384,
            end: 424,
        };
        dir.insert_partial(pe).unwrap();
        let prompt: Vec<u32> = (0..409).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(1), true, 0);
        let p = m.partial.expect("the partial entry matches");
        assert_eq!(
            p.tokens, 25,
            "the longest common prefix, not the entry's 40"
        );
        assert_eq!(p.key, pkey);
        assert_eq!(p.block, BlockId(9));
        assert_eq!(m.partial_outcome, Some(PartialOutcome::Reused));
        assert_eq!(m.blocks.len(), 3, "the full blocks before it");

        // A prompt diverging at token 70 of a cached full block matches that block with 70
        // tokens (its stored tokens differ from the prompt's, so no full block matches).
        let cached: Vec<u32> = (0..640).collect();
        let keys = prefix_keys(&h, &cached, 128);
        let mut diverged = cached.clone();
        diverged[70] += 1000;
        let m =
            chain(&h, &cached, None).lookup(&h, &diverged, 128, Duration::from_secs(2), true, 0);
        let p = m.partial.expect("the diverging block matches");
        assert_eq!(p.tokens, 70);
        assert_eq!(p.key, keys[0]);
        assert_eq!(m.blocks.len(), 0, "block 0 diverges inside the first chunk");

        // A prompt shorter than the cached sequence: its trailing chunk is a prefix of the
        // next cached full block, cached only in L1 — never a partial source, reported
        // `not_resident`.
        let short: Vec<u32> = (0..520).collect();
        let m =
            chain(&h, &cached, Some(4)).lookup(&h, &short, 128, Duration::from_secs(3), true, 0);
        assert_eq!(
            m.blocks.len(),
            4,
            "the full blocks, block 4's key unreached"
        );
        assert_eq!(m.partial, None, "an L1 copy is never a partial source");
        assert_eq!(m.partial_outcome, Some(PartialOutcome::NotResident));

        // A colliding hasher never matches another sequence's tokens.
        let colliding_tokens = vec![7u32; 128];
        let mut dir = KvDirectory::new(16);
        dir.insert(block(
            Colliding.key(None, &colliding_tokens),
            None,
            0,
            &colliding_tokens,
            &[l0(0)],
        ))
        .unwrap();
        let other: Vec<u32> = (1000..2128).collect();
        let m = dir.lookup(&Colliding, &other, 128, Duration::from_secs(4), true, 0);
        assert_eq!((m.blocks.len(), m.partial), (0, None));
        assert_eq!(m.mismatches, 1, "the colliding full block is still a miss");

        // A prompt shorter than one block matches a partial entry under the root parent.
        let root_tail: Vec<u32> = (100..140).collect();
        let mut root_entry = partial_entry(&h, ROOT_PARENT, &root_tail, 20);
        root_entry.parent = None;
        let mut dir = KvDirectory::new(16);
        dir.insert_partial(root_entry).unwrap();
        let prompt: Vec<u32> = (100..110).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(5), true, 0);
        let p = m
            .partial
            .expect("a partial entry under the root parent matches");
        assert_eq!(p.tokens, 10);

        // At most PARTIAL_SCAN_LIMIT children are compared, most recently used first: a
        // matching child ranked 71st is not scanned; once most recently used it is found.
        let base: Vec<u32> = (0..128).collect();
        let key0 = h.key(None, &base);
        let mut dir = KvDirectory::new(256);
        dir.insert(block(key0, None, 0, &base, &[l0(0)])).unwrap();
        let kid = |tokens: &[u32], j: u64| {
            let mut b = block(
                h.key(Some(&key0), tokens),
                Some(key0),
                j as u32,
                tokens,
                &[l0(100 + j)],
            );
            b.format = fmt128();
            b
        };
        for j in 0..70u64 {
            let tokens = vec![10_000 + j as u32; 128];
            dir.insert(kid(&tokens, j)).unwrap();
        }
        let matching: Vec<u32> = (128..256).collect();
        dir.insert(kid(&matching, 70)).unwrap();
        let touch_all = |dir: &mut KvDirectory, except: &KvKey, at: Timestamp| {
            for b in dir.blocks.values_mut() {
                if b.key != *except {
                    b.last_access = at;
                }
            }
        };
        touch_all(
            &mut dir,
            &h.key(Some(&key0), &matching),
            Duration::from_secs(1),
        );
        let prompt: Vec<u32> = (0..144).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(2), true, 0);
        assert_eq!(
            m.partial, None,
            "the matching child is ranked 71st by last_access"
        );
        assert_eq!(
            m.partial_outcome,
            Some(PartialOutcome::NoCandidate),
            "the scan compared 64 children, none matching"
        );
        dir.get_mut(&h.key(Some(&key0), &matching))
            .unwrap()
            .last_access = Duration::from_secs(3);
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(4), true, 0);
        assert_eq!(
            m.partial.map(|p| p.tokens),
            Some(16),
            "most recently used, it is scanned"
        );

        // The ninth partial entry under one parent evicts the least recently used one; the
        // caller frees the evicted block's L0 slot.
        let cached: Vec<u32> = (0..128).collect();
        let key0 = h.key(None, &cached);
        let mut dir = KvDirectory::new(64);
        dir.insert(block(key0, None, 0, &cached, &[l0(0)])).unwrap();
        for j in 0..9u32 {
            let tail: Vec<u32> = (500 + j * 40..500 + j * 40 + 40).collect();
            dir.insert_partial(partial_entry(&h, key0, &tail, 10 + j as u64))
                .unwrap();
            dir.get_mut(&h.key(Some(&key0), &tail)).unwrap().last_access =
                Duration::from_secs(j as u64);
        }
        assert_eq!(dir.len(), 9, "the full block plus 8 partial entries");
        let first = h.key(Some(&key0), &(500..540).collect::<Vec<u32>>());
        assert!(dir.get(&first).is_none(), "the least recently used went");
        let evicted = dir.evicted_partials();
        assert_eq!(evicted.len(), 1, " the caller frees the evicted L0 slot");
        assert_eq!(evicted[0].key, first);
        let second = h.key(Some(&key0), &(540..580).collect::<Vec<u32>>());
        assert!(dir.get(&second).is_some(), "the second-oldest stays");
        let prompt: Vec<u32> = (0..128).chain(540..570).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(10), true, 0);
        assert_eq!(m.partial.map(|p| p.tokens), Some(30));

        // A TurboQuant base-class entry (the `l0` tag over TurboQuant pages) is a lossy rung
        // class: `lossy_source`, no partial. The BF16 window class over the same pages is
        // exact.
        let tail: Vec<u32> = (16..56).collect();
        let mut dir = KvDirectory::new(64);
        dir.insert(block(key0, None, 0, &cached, &[l0(0)])).unwrap();
        let tq_fmt = crate::identity::KvFormat {
            dtype: KvDtype::Tq4,
            layout: KvLayout {
                dtype: DType::Tq4,
                ..fmt128().layout
            },
            shards: 1,
            scales: None,
        };
        let mut tq_entry = partial_entry(&h, key0, &tail, 30);
        tq_entry.format = tq_fmt;
        dir.insert_partial(tq_entry).unwrap();
        let prompt: Vec<u32> = (0..128).chain(16..30).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(11), true, 0);
        assert_eq!(
            m.partial, None,
            "TurboQuant base pages are never a partial source"
        );
        assert_eq!(m.partial_outcome, Some(PartialOutcome::LossySource));

        let win_tail: Vec<u32> = (500..540).collect();
        let mut win_entry = partial_entry(&h, key0, &win_tail, 31);
        win_entry.locations[0].format = "bf16";
        dir.insert_partial(win_entry).unwrap();
        let prompt: Vec<u32> = (0..128).chain(500..520).collect();
        let m = dir.lookup(&h, &prompt, 128, Duration::from_secs(12), true, 0);
        assert_eq!(
            m.partial.map(|p| p.tokens),
            Some(20),
            "the BF16 window class is an exact source"
        );
        assert_eq!(m.partial_outcome, Some(PartialOutcome::Reused));
    }
}
