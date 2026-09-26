//! The L0 (GPU) block pool: one preallocated device buffer, a free list and per-block
//! reference counts. After construction the pool never calls the device allocator, so running
//! out of blocks is `PoolError::Exhausted`, never an allocation failure.

use std::sync::Arc;

use smallvec::SmallVec;
use turbine_core::types::{BlockId, DeviceId, KvLayout};
use turbine_reliability::budget::PoolKind;
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView, MemoryError};

use crate::table::BlockTable;

/// Shape of the pool: per-token layout and block count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPoolConfig {
    pub layout: KvLayout,
    /// From `kv.gpu.max_bytes / layout.block_bytes()` (Phase 2).
    pub num_blocks: u32,
}

impl BlockPoolConfig {
    /// As many whole blocks as fit in `bytes` (`kv.gpu.max_bytes`). A zero-byte layout (the
    /// simulator's) yields 0 blocks; the simulator sets `num_blocks` directly.
    pub fn for_bytes(layout: KvLayout, bytes: u64) -> BlockPoolConfig {
        let num_blocks = bytes
            .checked_div(layout.block_bytes())
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX));
        BlockPoolConfig { layout, num_blocks }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("KV block pool exhausted: requested {requested} blocks, {available} free")]
    Exhausted { requested: u32, available: u32 },
    #[error("cannot allocate the KV block pool: {0}")]
    Memory(#[from] MemoryError),
    #[error(
        "KV block pool of {num_blocks} blocks × {block_bytes} bytes does not fit the address space"
    )]
    TooLarge { num_blocks: u32, block_bytes: u64 },
    /// The allocation alone is larger than the whole KV reservation it is paid from: the
    /// request's worst-case estimate was wrong (P3 S-3).
    #[error(
        "KV reservation short: {needed} bytes needed, {uncommitted} of the reservation uncommitted"
    )]
    ReservationShort { needed: u64, uncommitted: u64 },
}

/// L0 block accounting over one device allocation of `num_blocks × block_bytes`.
pub struct BlockPool {
    layout: KvLayout,
    storage: DeviceBuffer,
    /// Free block ids; `allocate` pops from the end, so ids are handed out in ascending order
    /// from a fresh pool.
    free: Vec<BlockId>,
    /// Reference count per block; 0 exactly when the block is on the free list.
    refcounts: Vec<u32>,
    /// The reservation ledger whose `kv` pool pays for the blocks (P3), and its device.
    ledger: Option<(Arc<Ledger>, DeviceId)>,
}

impl std::fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockPool")
            .field("layout", &self.layout)
            .field("total_blocks", &self.total_blocks())
            .field("free_blocks", &self.free_blocks())
            .finish_non_exhaustive()
    }
}

impl BlockPool {
    /// Allocates the whole pool once from `mem`.
    pub fn new(cfg: BlockPoolConfig, mem: Arc<dyn DeviceMemory>) -> Result<BlockPool, PoolError> {
        let block_bytes = cfg.layout.block_bytes();
        let too_large = PoolError::TooLarge {
            num_blocks: cfg.num_blocks,
            block_bytes,
        };
        let bytes = u64::from(cfg.num_blocks)
            .checked_mul(block_bytes)
            .and_then(|b| usize::try_from(b).ok())
            .ok_or(too_large)?;
        let storage = DeviceBuffer::alloc(&mem, bytes)?;
        Ok(BlockPool {
            layout: cfg.layout,
            storage,
            free: (0..cfg.num_blocks).rev().map(BlockId).collect(),
            refcounts: vec![0; cfg.num_blocks as usize],
            ledger: None,
        })
    }

    /// Link the pool to `ledger`'s `kv` pool on `device` (P3 S-3): blocks are then paid from
    /// per-request reservations through `allocate_reserved`. Panics when the pool's bytes
    /// exceed the ledger's `kv` capacity — the budget must cover every block.
    pub fn with_ledger(mut self, ledger: Arc<Ledger>, device: DeviceId) -> BlockPool {
        let bytes = u64::from(self.total_blocks()) * self.layout.block_bytes();
        let capacity = ledger.usage(device, PoolKind::Kv).capacity;
        assert!(
            bytes <= capacity,
            "KV block pool of {bytes} bytes exceeds the ledger's kv pool of {capacity} bytes"
        );
        self.ledger = Some((ledger, device));
        self
    }

    /// The ledger and device the pool is linked to.
    pub fn ledger(&self) -> Option<&(Arc<Ledger>, DeviceId)> {
        self.ledger.as_ref()
    }

    /// `allocate`, paid from `reservation`: commits `n × block_bytes` of it, saturating at its
    /// total, so blocks re-allocated after a recompute are not paid twice. Releasing blocks
    /// never touches the ledger — dropping the request's reservation does. `ReservationShort`
    /// when the allocation alone exceeds the whole reservation (nothing is allocated).
    pub fn allocate_reserved(
        &mut self,
        n: u32,
        reservation: &mut Reservation,
    ) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        debug_assert_eq!(
            reservation.pool(),
            PoolKind::Kv,
            "KV blocks paid from a non-kv pool"
        );
        let needed = u64::from(n) * self.layout.block_bytes();
        if needed > reservation.bytes() {
            return Err(PoolError::ReservationShort {
                needed,
                uncommitted: reservation.bytes() - reservation.committed(),
            });
        }
        let blocks = self.allocate(n)?;
        reservation.commit_bytes(needed);
        Ok(blocks)
    }

    pub fn layout(&self) -> KvLayout {
        self.layout
    }

    pub fn total_blocks(&self) -> u32 {
        self.refcounts.len() as u32
    }

    pub fn free_blocks(&self) -> u32 {
        self.free.len() as u32
    }

    pub fn used_blocks(&self) -> u32 {
        self.total_blocks() - self.free_blocks()
    }

    /// `n` fresh blocks with refcount 1, or `Exhausted` with nothing allocated.
    pub fn allocate(&mut self, n: u32) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        let available = self.free_blocks();
        if n > available {
            return Err(PoolError::Exhausted {
                requested: n,
                available,
            });
        }
        let mut out = SmallVec::with_capacity(n as usize);
        for _ in 0..n {
            let b = self.free.pop().expect("checked above: enough free blocks");
            self.refcounts[b.0 as usize] = 1;
            out.push(b);
        }
        Ok(out)
    }

    /// One more holder of an allocated block. Panics on a free block (a caller bug: the block
    /// could be handed out again).
    pub fn incref(&mut self, b: BlockId) {
        let rc = &mut self.refcounts[b.0 as usize];
        assert!(*rc > 0, "incref of free KV block {b:?}");
        *rc += 1;
    }

    /// Drops one reference to each block; a block whose count reaches 0 returns to the free
    /// list. Panics on a block that is already free (a double release).
    pub fn release(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            let rc = &mut self.refcounts[b.0 as usize];
            assert!(*rc > 0, "release of free KV block {b:?}");
            *rc -= 1;
            if *rc == 0 {
                self.free.push(b);
            }
        }
    }

    /// A table for a forked sequence (`n > 1`): full blocks are shared by reference count;
    /// a partial tail block gets a fresh block, returned as `(src, dst)` for the engine to copy
    /// with `copy_blocks`. All-or-nothing: `Exhausted` leaves the pool unchanged.
    pub fn fork(
        &mut self,
        table: &BlockTable,
    ) -> Result<(BlockTable, Option<(BlockId, BlockId)>), PoolError> {
        let block_tokens = self.layout.block_tokens.max(1);
        let partial_tail = !table.tokens.is_multiple_of(block_tokens) && !table.blocks.is_empty();
        let tail = if partial_tail {
            Some(self.allocate(1)?[0])
        } else {
            None
        };
        let shared = table.blocks.len() - usize::from(partial_tail);
        let mut blocks: SmallVec<[BlockId; 16]> = SmallVec::with_capacity(table.blocks.len());
        for &b in &table.blocks[..shared] {
            self.incref(b);
            blocks.push(b);
        }
        let copy = tail.map(|dst| {
            blocks.push(dst);
            (*table.blocks.last().expect("partial tail exists"), dst)
        });
        Ok((
            BlockTable {
                blocks,
                tokens: table.tokens,
            },
            copy,
        ))
    }

    /// The device storage and layout for the executor.
    pub fn view(&self) -> KvPoolView<'_> {
        let num_blocks = self.total_blocks();
        let per_layer_block = u64::from(self.layout.num_kv_heads)
            * u64::from(self.layout.head_dim)
            * u64::from(self.layout.block_tokens)
            * 2
            * self.layout.dtype.size_bytes() as u64;
        KvPoolView {
            storage: &self.storage,
            layout: self.layout,
            num_blocks,
            layer_stride_bytes: u64::from(num_blocks) * per_layer_block,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use proptest::prelude::*;
    use turbine_core::types::{BlockId, DType, DeviceId, KvLayout};
    use turbine_tensor::DeviceMemory;
    use turbine_tensor::host::HostMemory;

    use super::*;
    use crate::table::{BlockTable, blocks_for_tokens};

    const BLOCK_TOKENS: u32 = 16;

    /// A small real layout (2 layers, 2 KV heads, head_dim 4, BF16): 512 bytes per block.
    fn small_layout() -> KvLayout {
        layout_of(BLOCK_TOKENS)
    }

    /// [`small_layout`] with `block_tokens` tokens per block.
    fn layout_of(block_tokens: u32) -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            dtype: DType::BF16,
            block_tokens,
        }
    }

    fn pool(num_blocks: u32) -> BlockPool {
        pool_of(num_blocks, BLOCK_TOKENS)
    }

    fn pool_of(num_blocks: u32, block_tokens: u32) -> BlockPool {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 20);
        BlockPool::new(
            BlockPoolConfig {
                layout: layout_of(block_tokens),
                num_blocks,
            },
            mem,
        )
        .expect("pool fits the host memory")
    }

    #[derive(Clone, Debug)]
    enum Op {
        /// Start a new sequence with this many tokens.
        Allocate(u32),
        /// Append tokens to the table at this index (modulo the live tables).
        Append(usize, u32),
        /// Release the table at this index.
        Free(usize),
        /// Fork the table at this index.
        Fork(usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (1u32..=120).prop_map(Op::Allocate),
            (any::<usize>(), 1u32..=40).prop_map(|(i, n)| Op::Append(i, n)),
            any::<usize>().prop_map(Op::Free),
            any::<usize>().prop_map(Op::Fork),
        ]
    }

    fn free_set(p: &BlockPool) -> BTreeSet<BlockId> {
        p.free.iter().copied().collect()
    }

    /// Every invariant of the pool against the tables that hold its blocks.
    fn check(p: &BlockPool, tables: &[BlockTable]) {
        let bt = p.layout().block_tokens;
        assert_eq!(p.used_blocks() + p.free_blocks(), p.total_blocks());
        let free = free_set(p);
        assert_eq!(
            free.len(),
            p.free.len(),
            "a block is on the free list twice"
        );
        let mut held = vec![0usize; p.total_blocks() as usize];
        for t in tables {
            for b in &t.blocks {
                held[b.0 as usize] += 1;
            }
        }
        for b in 0..p.total_blocks() {
            let id = BlockId(b);
            let holders = held[b as usize];
            assert_eq!(
                p.refcounts[b as usize] as usize, holders,
                "refcount of {id:?} differs from the tables holding it"
            );
            assert_eq!(
                free.contains(&id),
                holders == 0,
                "{id:?} free/held mismatch"
            );
        }
        for t in tables {
            let distinct: BTreeSet<_> = t.blocks.iter().collect();
            assert_eq!(
                distinct.len(),
                t.blocks.len(),
                "a table holds a block twice"
            );
            assert_eq!(t.blocks.len() as u32, blocks_for_tokens(t.tokens, bt));
        }
    }

    proptest! {
        #[test]
        fn blocks_conserved(
            ops in proptest::collection::vec(op(), 1..200),
            bt in prop_oneof![Just(BLOCK_TOKENS), Just(128u32)],
        ) {
            let mut p = pool_of(64, bt);
            let mut tables: Vec<BlockTable> = Vec::new();
            check(&p, &tables);
            for op in ops {
                match op {
                    Op::Allocate(tokens) => {
                        let need = blocks_for_tokens(tokens, bt);
                        let before = p.free_blocks();
                        match p.allocate(need) {
                            Ok(blocks) => {
                                prop_assert_eq!(blocks.len() as u32, need);
                                tables.push(BlockTable { blocks: blocks.into_iter().collect(), tokens });
                            }
                            Err(PoolError::Exhausted { requested, available }) => {
                                prop_assert_eq!(requested, need);
                                prop_assert_eq!(available, before);
                                prop_assert!(need > before);
                                prop_assert_eq!(p.free_blocks(), before, "failed allocate changed the pool");
                            }
                            Err(e) => prop_assert!(false, "unexpected {e}"),
                        }
                    }
                    Op::Append(i, extra) if !tables.is_empty() => {
                        let i = i % tables.len();
                        let need = tables[i].blocks_needed(extra, bt);
                        if let Ok(blocks) = p.allocate(need) {
                            tables[i].blocks.extend(blocks);
                            tables[i].tokens += extra;
                        } else {
                            prop_assert!(need > p.free_blocks());
                        }
                    }
                    Op::Free(i) if !tables.is_empty() => {
                        let t = tables.swap_remove(i % tables.len());
                        // Exactly the blocks no other table still holds return to the free list.
                        let expected: BTreeSet<BlockId> = t
                            .blocks
                            .iter()
                            .copied()
                            .filter(|b| !tables.iter().any(|o| o.blocks.contains(b)))
                            .collect();
                        let before = free_set(&p);
                        p.release(&t.blocks);
                        let after = free_set(&p);
                        let returned: BTreeSet<BlockId> = after.difference(&before).copied().collect();
                        prop_assert_eq!(returned, expected);
                        prop_assert!(before.is_subset(&after));
                    }
                    Op::Fork(i) if !tables.is_empty() => {
                        let i = i % tables.len();
                        let src = tables[i].clone();
                        let before = p.free_blocks();
                        match p.fork(&src) {
                            Ok((child, copy)) => {
                                prop_assert_eq!(child.tokens, src.tokens);
                                prop_assert_eq!(child.blocks.len(), src.blocks.len());
                                let partial = !src.tokens.is_multiple_of(bt);
                                prop_assert_eq!(copy.is_some(), partial);
                                if let Some((from, to)) = copy {
                                    prop_assert_eq!(Some(&from), src.blocks.last());
                                    prop_assert_eq!(Some(&to), child.blocks.last());
                                    prop_assert_ne!(from, to);
                                }
                                let shared = src.blocks.len() - usize::from(partial);
                                prop_assert_eq!(&child.blocks[..shared], &src.blocks[..shared]);
                                tables.push(child);
                            }
                            Err(PoolError::Exhausted { .. }) => {
                                prop_assert_eq!(before, 0);
                                prop_assert_eq!(p.free_blocks(), 0);
                            }
                            Err(e) => prop_assert!(false, "unexpected {e}"),
                        }
                    }
                    _ => {}
                }
                check(&p, &tables);
            }
            for t in tables.drain(..) {
                p.release(&t.blocks);
            }
            prop_assert_eq!(p.used_blocks(), 0);
        }
    }

    #[test]
    fn storage_is_one_preallocated_buffer() {
        let p = pool(10);
        let view = p.view();
        assert_eq!(view.storage.len() as u64, 10 * small_layout().block_bytes());
        assert_eq!(view.num_blocks, 10);
        // Per layer: [num_blocks, 2, block_tokens, kv_heads, head_dim] BF16.
        assert_eq!(view.layer_stride_bytes, 10 * 2 * 16 * 2 * 4 * 2);
        assert_eq!(view.layout, small_layout());

        // Too large for the backing memory: a memory error at construction, never later.
        let tiny: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1024);
        let err = BlockPool::new(
            BlockPoolConfig {
                layout: small_layout(),
                num_blocks: 10,
            },
            tiny,
        )
        .unwrap_err();
        assert!(matches!(err, PoolError::Memory(_)), "{err}");
    }

    #[test]
    fn allocation_is_all_or_nothing_and_ordered() {
        let mut p = pool(4);
        assert_eq!(p.allocate(2).unwrap().as_slice(), &[BlockId(0), BlockId(1)]);
        let err = p.allocate(3).unwrap_err();
        assert!(matches!(
            err,
            PoolError::Exhausted {
                requested: 3,
                available: 2
            }
        ));
        assert_eq!(p.free_blocks(), 2);
        assert!(p.allocate(0).unwrap().is_empty());
        // incref keeps a block alive through one release.
        p.incref(BlockId(0));
        p.release(&[BlockId(0)]);
        assert_eq!(p.used_blocks(), 2);
        p.release(&[BlockId(0), BlockId(1)]);
        assert_eq!(p.used_blocks(), 0);
    }
}
