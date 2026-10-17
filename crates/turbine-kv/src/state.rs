//! Hybrid per-sequence state (P7 S-12, Task 32): the slot pool a hybrid family's live state
//! lives in, the snapshot pool that keeps block-boundary copies of it, and the copy
//! descriptions the scheduler and the engine exchange. Nothing here touches device memory —
//! the pools are index bookkeeping, the bytes live in the L0 state buffer the executor owns
//! (`StateLayout::bytes_per_sequence` per slot and per snapshot entry). The scheduler allocates
//! and releases slots with the sequence; the engine allocates snapshot entries with the blocks
//! they belong to and hands the executor their device locations.

use std::collections::HashMap;

use turbine_core::types::{BlockId, StateLayout};

/// A hybrid-state pool error. Exhaustion is a bounded-queue refusal (spec §21 rule 4): the
/// caller waits for a free slot rather than growing the pool.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvError {
    /// Every live-state slot is in use.
    #[error("hybrid state slots exhausted")]
    StateSlotsExhausted,
    /// Every snapshot-pool entry is in use.
    #[error("hybrid state snapshots exhausted")]
    SnapshotsExhausted,
}

/// One per-sequence live-state slot; its bytes start at `slot.0 × bytes_per_sequence()` in the
/// executor's state buffer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct StateSlot(pub u32);

/// The state a hybrid full block carries: its logical KV block and the snapshot-pool entry
/// holding its bytes. A block without a snapshot carries `None` ([`crate::directory::KvBlock`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SnapshotRef {
    pub block: BlockId,
    pub entry: u32,
}

/// Where a state copy reads or writes: a live slot, or a block's snapshot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateLoc {
    Slot(StateSlot),
    Snapshot(SnapshotRef),
}

/// One state copy `src -> dst`, executed before a step, after its KV tail copies. A fresh
/// sequence's slot is zeroed by the executor instead (nothing to copy); a prefix hit copies
/// `Snapshot -> Slot`; an `n > 1` fork copies `Slot -> Slot`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StateCopy {
    pub src: StateLoc,
    pub dst: StateLoc,
}

/// The per-sequence live-state slots (P7 S-12): `slots` slots of `layout.bytes_per_sequence()`
/// bytes each, handed out whole and released whole. A pure free list — which sequence owns a
/// slot is the scheduler's bookkeeping.
#[derive(Clone, Debug)]
pub struct StateSlotPool {
    layout: StateLayout,
    used: Vec<bool>,
    free: Vec<u32>,
}

impl StateSlotPool {
    /// A pool of `slots` slots sized by `layout`.
    pub fn new(layout: StateLayout, slots: u32) -> StateSlotPool {
        StateSlotPool {
            layout,
            used: vec![false; slots as usize],
            free: (0..slots).rev().collect(),
        }
    }

    /// Takes a free slot (`KvError::StateSlotsExhausted` when none is).
    pub fn alloc(&mut self) -> Result<StateSlot, KvError> {
        let slot = self.free.pop().ok_or(KvError::StateSlotsExhausted)?;
        self.used[slot as usize] = true;
        Ok(StateSlot(slot))
    }

    /// Returns `slot`; releasing a free slot is a no-op.
    pub fn release(&mut self, slot: StateSlot) {
        if let Some(used) = self.used.get_mut(slot.0 as usize)
            && *used
        {
            *used = false;
            self.free.push(slot.0);
        }
    }

    /// Free slots.
    pub fn free(&self) -> u32 {
        self.free.len() as u32
    }

    /// Slots in all (free plus in use).
    pub fn capacity(&self) -> u32 {
        self.used.len() as u32
    }

    /// True while `slot` is allocated.
    pub fn is_used(&self, slot: StateSlot) -> bool {
        self.used.get(slot.0 as usize).copied().unwrap_or(false)
    }

    pub fn layout(&self) -> StateLayout {
        self.layout
    }

    /// Bytes the pool's slots hold (the budget's state-slot term).
    pub fn bytes(&self) -> u64 {
        u64::from(self.capacity()).saturating_mul(self.layout.bytes_per_sequence())
    }
}

/// The snapshot entries (P7 S-12): each holds the state after one full block of a sequence, keyed
/// by that block's [`BlockId`]. Snapshots live in their own bounded pool — the budget charges
/// `entries × layout.bytes_per_sequence()` — not in KV pages; Task 33 makes them travel with
/// their block through L1/L2.
#[derive(Clone, Debug)]
pub struct SnapshotPool {
    layout: StateLayout,
    entries: u32,
    /// Block -> entry, for every stored snapshot.
    by_block: HashMap<BlockId, u32>,
    /// Entry -> owning block, `None` when free.
    owner: Vec<Option<BlockId>>,
    free: Vec<u32>,
}

impl SnapshotPool {
    /// A pool of `entries` snapshot entries sized by `layout`.
    pub fn new(layout: StateLayout, entries: u32) -> SnapshotPool {
        SnapshotPool {
            layout,
            entries,
            by_block: HashMap::new(),
            owner: vec![None; entries as usize],
            free: (0..entries).rev().collect(),
        }
    }

    /// The entry for `block`, taking a free one when it has none. Storing twice for one block
    /// returns the same entry (its bytes are overwritten in place).
    pub fn alloc(&mut self, block: BlockId) -> Result<SnapshotRef, KvError> {
        if let Some(&entry) = self.by_block.get(&block) {
            return Ok(SnapshotRef { block, entry });
        }
        let entry = self.free.pop().ok_or(KvError::SnapshotsExhausted)?;
        self.owner[entry as usize] = Some(block);
        self.by_block.insert(block, entry);
        Ok(SnapshotRef { block, entry })
    }

    /// The entry holding `block`'s snapshot, if any.
    pub fn entry_of(&self, block: BlockId) -> Option<u32> {
        self.by_block.get(&block).copied()
    }

    /// Drops `block`'s snapshot, returning whether one was held.
    pub fn release(&mut self, block: BlockId) -> bool {
        let Some(entry) = self.by_block.remove(&block) else {
            return false;
        };
        self.owner[entry as usize] = None;
        self.free.push(entry);
        true
    }

    /// Free entries.
    pub fn free(&self) -> u32 {
        self.free.len() as u32
    }

    /// Entries in all (free plus in use).
    pub fn capacity(&self) -> u32 {
        self.entries
    }

    pub fn layout(&self) -> StateLayout {
        self.layout
    }

    /// Bytes of one entry.
    pub fn entry_bytes(&self) -> u64 {
        self.layout.bytes_per_sequence()
    }

    /// Bytes the pool holds (the budget's snapshot term).
    pub fn bytes(&self) -> u64 {
        u64::from(self.entries).saturating_mul(self.layout.bytes_per_sequence())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbine_core::types::StateDtype;

    fn layout() -> StateLayout {
        StateLayout {
            linear_layers: 4,
            recurrent_bytes_per_layer: 1024,
            conv_bytes_per_layer: 512,
            dtype: Some(StateDtype::F32),
        }
    }

    /// A pool of 4 hands out 4 slots, refuses the 5th with `KvError::StateSlotsExhausted`, and
    /// releasing restores the free count; a double release is a no-op. Breaks if the free list
    /// leaks a slot or hands out an allocated one.
    #[test]
    fn slots_alloc_release_and_exhaust() {
        let mut pool = StateSlotPool::new(layout(), 4);
        assert_eq!((pool.capacity(), pool.free()), (4, 4));
        let slots: Vec<StateSlot> = (0..4).map(|_| pool.alloc().expect("alloc")).collect();
        assert_eq!(pool.free(), 0);
        assert!(slots.iter().all(|s| pool.is_used(*s)));
        assert_eq!(
            pool.alloc().expect_err("exhausted"),
            KvError::StateSlotsExhausted
        );
        pool.release(slots[1]);
        assert_eq!(pool.free(), 1);
        assert!(!pool.is_used(slots[1]));
        // Releasing a slot that is already free is a no-op (never two free entries for one).
        pool.release(slots[1]);
        assert_eq!(pool.free(), 1);
        let reused = pool.alloc().expect("reuse");
        assert_eq!(reused, slots[1], "the released slot comes back");
        assert!(pool.is_used(reused));
        assert_eq!(pool.free(), 0);
        pool.release(reused);
        assert_eq!(pool.free(), 1);
        assert_eq!(pool.bytes(), 4 * 4 * (1024 + 512));
    }

    /// Snapshot entries are keyed by block, handed out free-list wise, refused on exhaustion and
    /// freed on release. Breaks if two blocks share an entry or a release leaks it.
    #[test]
    fn snapshots_alloc_by_block_and_release() {
        let mut pool = SnapshotPool::new(layout(), 2);
        assert_eq!(
            (pool.capacity(), pool.free(), pool.entry_bytes()),
            (2, 2, 4 * (1024 + 512))
        );
        let a = pool.alloc(BlockId(7)).expect("a");
        let b = pool.alloc(BlockId(9)).expect("b");
        assert_ne!(a.entry, b.entry);
        assert_eq!(pool.alloc(BlockId(7)).expect("again").entry, a.entry);
        assert_eq!(pool.entry_of(BlockId(7)), Some(a.entry));
        assert_eq!(
            pool.alloc(BlockId(11)).expect_err("exhausted"),
            KvError::SnapshotsExhausted
        );
        assert!(pool.release(BlockId(7)));
        assert!(!pool.release(BlockId(7)), "already released");
        assert_eq!(pool.entry_of(BlockId(7)), None);
        assert_eq!(pool.free(), 1);
        assert_eq!(pool.alloc(BlockId(11)).expect("reuse").entry, a.entry);
        assert_eq!(pool.bytes(), 2 * 4 * (1024 + 512));
    }
}
