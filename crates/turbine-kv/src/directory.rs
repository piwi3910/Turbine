//! Local KV directory (P4 S-2): the only owner of block metadata; tiers own bytes.
//!
//! Lookup walks a prompt's full blocks through the key chain and stops at the first block that is
//! missing or whose stored tokens differ from the prompt (a hash collision is a miss, never
//! another sequence's KV). Eviction eligibility is leaf-first: a copy may leave a tier only while
//! no child is cached in the same or a faster tier.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use smallvec::SmallVec;
use turbine_core::types::{ModelFingerprint, PriorityClass};

use crate::identity::{KeyHasher, KvFormat, KvKey};
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
}

impl KvBlock {
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

/// Block key → metadata, with a children index for leaf-first eviction.
pub struct KvDirectory {
    blocks: HashMap<KvKey, KvBlock>,
    children: HashMap<KvKey, SmallVec<[KvKey; 4]>>,
    pending: HashMap<KvKey, Timestamp>,
    max_entries: usize,
}

impl KvDirectory {
    /// `max_entries` is the sum of the enabled tiers' capacities in blocks (P4 Constraints).
    pub fn new(max_entries: usize) -> Self {
        KvDirectory {
            blocks: HashMap::new(),
            children: HashMap::new(),
            pending: HashMap::new(),
            max_entries,
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
    /// lossy prefix (keyed by [`PrefixMatch::lossy_chain`]). Without `allow_lossy` it stops at
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
            let mut cands: SmallVec<[KvKey; 5]> = SmallVec::new();
            cands.push(key);
            for c in crate::codec::registry().iter() {
                let lk = crate::identity::lossy_key(key, c.name(), seed);
                if self.blocks.contains_key(&lk) {
                    cands.push(lk);
                }
            }
            cands.extend(chain);
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
        m
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
    use crate::identity::{Blake3Hasher, NamespaceKey, prefix_keys};
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
}
