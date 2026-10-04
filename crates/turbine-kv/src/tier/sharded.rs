//! L1 under tensor parallelism (Phase 5 §Data, decision "P5 T17", answer B): one
//! [`L1PinnedTier`] per rank, each growing its slabs through that rank's kernel-library context
//! (a pinned buffer is only a copy target on the context that allocated it), seen by the
//! hierarchy as one tier of logical blocks, the rank shards concatenated in rank order.
//!
//! A block is stored only when every shard holds it: `contains` needs all of them, a failed
//! `put` on one shard evicts the shards already written, and the copy-stream protocol
//! ([`ShardedL1Tier::reserve`] / [`commit`](ShardedL1Tier::commit) /
//! [`abort_reservation`](ShardedL1Tier::abort_reservation)) reserves, commits or aborts the key in
//! every shard. With one shard every call is the shard's own, so tp = 1 behaves as Phase 4.

use std::sync::Arc;
use std::time::Duration;

use smallvec::SmallVec;
use turbine_core::types::PressureState;

use super::{KvTier, L1PinnedTier, TierBlockMut, TierBlockRef, TierError, TierId, TierSlot};
use crate::identity::KvKey;

/// Pinned buffer id and byte offset of one shard's slot, in rank order.
pub type ShardSlots = SmallVec<[(u64, usize); 8]>;

pub struct ShardedL1Tier {
    shards: Vec<Arc<L1PinnedTier>>,
    /// Bytes of each shard of a block (each shard tier's block size), in rank order.
    shard_bytes: Vec<usize>,
}

impl ShardedL1Tier {
    /// `shards` in rank order, each storing `shard_bytes`-byte blocks. Panics without a shard.
    pub fn new(shards: Vec<Arc<L1PinnedTier>>, shard_bytes: u64) -> Self {
        let sizes = vec![shard_bytes; shards.len()];
        ShardedL1Tier::with_sizes(shards, sizes)
    }

    /// `shards` in order, shard `i` storing `sizes[i]`-byte blocks: a pipeline's stages hold
    /// different layer counts (Phase 5 S-10), so their shards of a block differ in size. Panics
    /// without a shard or with a size per shard missing.
    pub fn with_sizes(shards: Vec<Arc<L1PinnedTier>>, sizes: Vec<u64>) -> Self {
        assert!(!shards.is_empty(), "a sharded L1 needs at least one shard");
        assert_eq!(shards.len(), sizes.len(), "one block size per shard");
        ShardedL1Tier {
            shards,
            shard_bytes: sizes.into_iter().map(|b| b as usize).collect(),
        }
    }

    /// `bytes` cut into the shards' parts, in shard order.
    fn parts<'a>(&self, mut bytes: &'a [u8]) -> Vec<&'a [u8]> {
        self.shard_bytes
            .iter()
            .map(|&n| {
                let (part, rest) = bytes.split_at(n);
                bytes = rest;
                part
            })
            .collect()
    }

    /// [`Self::parts`] of a mutable block.
    fn parts_mut<'a>(&self, mut bytes: &'a mut [u8]) -> Vec<&'a mut [u8]> {
        self.shard_bytes
            .iter()
            .map(|&n| {
                let (part, rest) = std::mem::take(&mut bytes).split_at_mut(n);
                bytes = rest;
                part
            })
            .collect()
    }

    /// The rank tiers, in rank order.
    pub fn shards(&self) -> &[Arc<L1PinnedTier>] {
        &self.shards
    }

    /// Live slabs of the emptiest shard (0 while any shard has none).
    pub fn slab_count(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.slab_count())
            .min()
            .unwrap_or(0)
    }

    /// Phase 3 host-memory pressure, on every shard.
    pub fn set_host_pressure(&self, p: PressureState) {
        for s in &self.shards {
            s.set_host_pressure(p);
        }
    }

    /// Calibrated L1 → L0 estimates of one logical block, on every shard.
    pub fn set_estimates(&self, latency: Duration, bandwidth: f64) {
        for s in &self.shards {
            s.set_estimates(latency, bandwidth);
        }
    }

    /// Every shard's slot of a stored block; `None` unless all shards hold it.
    pub fn locate(&self, key: &KvKey) -> Option<ShardSlots> {
        self.shards.iter().map(|s| s.locate(key)).collect()
    }

    /// Rank 0's slot of a stored block with its bytes ([`L1PinnedTier::locate_len`]), when the
    /// tier has one shard.
    pub fn locate_len(&self, key: &KvKey) -> Option<(u64, usize, usize)> {
        match self.shards.as_slice() {
            [one] => one.locate_len(key),
            _ => None,
        }
    }

    /// Reserves `key` in every shard, or in none: a shard that cannot reserve aborts the
    /// reservations already taken.
    pub fn reserve(&self, key: KvKey) -> Result<ShardSlots, TierError> {
        let mut slots = ShardSlots::new();
        for (i, s) in self.shards.iter().enumerate() {
            match s.reserve(key) {
                Ok(slot) => slots.push(slot),
                Err(e) => {
                    for t in &self.shards[..i] {
                        t.abort_reservation(&key);
                    }
                    return Err(e);
                }
            }
        }
        Ok(slots)
    }

    /// Makes the reserved block visible in every shard; the slot is rank 0's.
    pub fn commit(&self, key: &KvKey) -> TierSlot {
        let mut first = None;
        for s in &self.shards {
            let slot = s.commit(key);
            first.get_or_insert(slot);
        }
        first.expect("a sharded L1 has a shard")
    }

    /// Frees the reservation of `key` in every shard; false when no shard had one.
    pub fn abort_reservation(&self, key: &KvKey) -> bool {
        self.shards
            .iter()
            .fold(false, |any, s| s.abort_reservation(key) | any)
    }

    /// One copy error on rank `shard`'s copy stream; true when it degrades that shard (and with
    /// it the tier).
    pub fn record_copy_error(&self, shard: usize) -> bool {
        self.shards
            .get(shard)
            .is_some_and(|s| s.record_copy_error())
    }

    fn check_len(&self, len: usize) -> Result<(), TierError> {
        let want: usize = self.shard_bytes.iter().sum();
        if len == want {
            Ok(())
        } else {
            Err(TierError::Io(format!(
                "block is {len} bytes, sharded L1 slot {want}"
            )))
        }
    }
}

impl KvTier for ShardedL1Tier {
    fn id(&self) -> TierId {
        TierId::L1
    }

    fn enabled(&self) -> bool {
        self.shards.iter().all(|s| s.enabled())
    }

    fn capacity_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.capacity_bytes()).sum()
    }

    fn used_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.used_bytes()).sum()
    }

    fn pressure(&self) -> PressureState {
        self.shards
            .iter()
            .map(|s| s.pressure())
            .max()
            .unwrap_or(PressureState::Green)
    }

    fn est_latency(&self) -> Duration {
        self.shards
            .iter()
            .map(|s| s.est_latency())
            .max()
            .unwrap_or_default()
    }

    fn est_bandwidth(&self) -> Option<f64> {
        self.shards
            .iter()
            .filter_map(|s| s.est_bandwidth())
            .min_by(f64::total_cmp)
    }

    fn contains(&self, key: &KvKey) -> bool {
        self.shards.iter().all(|s| s.contains(key))
    }

    fn put(&self, key: KvKey, src: TierBlockRef<'_>) -> Result<TierSlot, TierError> {
        if let [one] = self.shards.as_slice() {
            return one.put(key, src);
        }
        let TierBlockRef::Host(bytes) = src;
        self.check_len(bytes.len())?;
        let mut first = None;
        for (i, (s, part)) in self.shards.iter().zip(self.parts(bytes)).enumerate() {
            match s.put(key, TierBlockRef::Host(part)) {
                Ok(slot) => {
                    first.get_or_insert(slot);
                }
                Err(e) => {
                    for t in &self.shards[..i] {
                        let _ = t.evict(&key);
                    }
                    return Err(e);
                }
            }
        }
        first.ok_or(TierError::Missing)
    }

    fn get(&self, key: &KvKey, dst: TierBlockMut<'_>) -> Result<(), TierError> {
        if let [one] = self.shards.as_slice() {
            return one.get(key, dst);
        }
        let TierBlockMut::Host(out) = dst;
        if !self.contains(key) {
            return Err(TierError::Missing);
        }
        self.check_len(out.len())?;
        for (s, part) in self.shards.iter().zip(self.parts_mut(out)) {
            s.get(key, TierBlockMut::Host(part))?;
        }
        Ok(())
    }

    /// Evicts `key` from every shard; `Missing` only when no shard held it.
    fn evict(&self, key: &KvKey) -> Result<(), TierError> {
        let mut found = false;
        let mut failed = None;
        for s in &self.shards {
            match s.evict(key) {
                Ok(()) => found = true,
                Err(TierError::Missing) => {}
                Err(e) => {
                    failed.get_or_insert(e);
                }
            }
        }
        match (failed, found) {
            (Some(e), _) => Err(e),
            (None, true) => Ok(()),
            (None, false) => Err(TierError::Missing),
        }
    }

    fn degraded(&self) -> bool {
        self.shards.iter().any(|s| s.degraded())
    }

    /// Changes when any shard's does.
    fn room_epoch(&self) -> u64 {
        self.shards
            .iter()
            .fold(0u64, |a, s| a.wrapping_add(s.room_epoch()))
    }

    /// A block is stored only when every shard takes its part: the fewest any shard has room
    /// for, each shard asked for its share of `bytes`.
    fn free_slots(&self, format: &'static str, bytes: u64) -> u64 {
        if let [one] = self.shards.as_slice() {
            return one.free_slots(format, bytes);
        }
        let total: u64 = self.shard_bytes.iter().map(|&b| b as u64).sum();
        self.shards
            .iter()
            .zip(&self.shard_bytes)
            .map(|(s, &b)| s.free_slots(format, (bytes * b as u64).div_ceil(total.max(1))))
            .min()
            .unwrap_or(0)
    }
}
