//! The L0 (GPU) block pool: one preallocated device buffer, a free list and per-block
//! reference counts. After construction the pool never calls the device allocator, so running
//! out of blocks is `PoolError::Exhausted`, never an allocation failure.
//!
//! Phase 4 adds cached blocks: a block the KV directory has keyed (`set_keyed`) stays allocated
//! when its reference count reaches 0, so a later request can attach it again. `allocate`
//! reclaims such cached unreferenced blocks, lowest value first in the order the hierarchy
//! published (`set_reclaim_order`), when the free list is short, and reports them through
//! `take_reclaimed` so the directory forgets their L0 copies.
//!
//! Phase 6b (S-5): the pool is carved into page classes, one per KV format, from the one
//! allocation. The base class (the L0 dtype, `kv.dtype`) owns every page at construction;
//! [`BlockPool::with_page_classes`] splits the base pages into slabs of `slab_blocks` pages, and
//! [`BlockPool::allocate_in`] turns a wholly free base slab into a slab of smaller pages of
//! another format on demand. A slab returns to the base class as soon as its last page is
//! freed. A page's format is fixed for its life ([`BlockPool::format_of`]), and its bytes are
//! exactly its class's page ([`BlockPool::block_segments`]), so a block's bytes always match
//! its tag. The bytes of every class sum to the one allocation ([`BlockPool::class_usage`]).

use std::collections::VecDeque;
use std::sync::Arc;

use smallvec::SmallVec;
use turbine_core::types::{BlockId, DType, DeviceId, KvLayout};
use turbine_reliability::budget::PoolKind;
use turbine_reliability::ledger::{Ledger, Reservation};
use turbine_tensor::{
    DeviceBuffer, DeviceMemory, DevicePtr, KvPageClasses, KvPoolView, MemoryError,
};

use crate::table::BlockTable;

/// The `TURBINE_KVFMT_*` codes (`turbine-kernels`' ABI v2.11 block-format codes, mirrored so
/// `turbine-kv` stays free of the kernel crate).
const KV_FMT_BF16: u8 = 0;
const KV_FMT_FP8_E4M3: u8 = 1;
const KV_FMT_TQ4: u8 = 2;
const KV_FMT_TQ2: u8 = 3;

/// A class of L0 pages of one KV format (P6b S-5): `page_bytes` is one page over every layer
/// (a base-class page is `layout.block_bytes()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageClass {
    pub format: &'static str,
    pub page_bytes: u64,
}

/// What one page class holds right now ([`BlockPool::class_usage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageClassUsage {
    pub class: PageClass,
    /// Pages the class holds (its slabs' pages; the base class: the pages not carved away).
    pub pages: u32,
    /// Pages allocated (referenced or cached).
    pub used_pages: u32,
    /// Bytes of the one allocation the class owns (a non-base class: whole slabs, its pages
    /// plus the slack a slab does not divide into).
    pub bytes: u64,
}

/// The `TURBINE_KVFMT_*` code of a page-class name under an L0 `dtype`: `l0` names the base
/// class, the others their format's code (`bf16` is code 0 even on a TurboQuant base pool —
/// the recent window's BF16 class, S-5 — while `l0` follows `dtype`).
fn class_code(format: &str, dtype: DType) -> u8 {
    let base = match dtype {
        DType::F8E4M3 => 1,
        DType::Tq4 => 2,
        DType::Tq2 => 3,
        _ => 0,
    };
    match format {
        "bf16" => KV_FMT_BF16,
        "fp8_e4m3" => KV_FMT_FP8_E4M3,
        "tq4" => KV_FMT_TQ4,
        "tq2" => KV_FMT_TQ2,
        _ => base,
    }
}

/// The name of the base page class of an L0 dtype (the `kv.dtype` spelling).
pub fn base_format(dtype: DType) -> &'static str {
    match dtype {
        DType::F8E4M3 => "fp8_e4m3",
        other => other.as_str(),
    }
}

/// A non-base page class and its free pages.
#[derive(Debug)]
struct ClassState {
    class: PageClass,
    /// Bytes of one page in one layer's region.
    per_layer: u64,
    /// Pages one slab holds.
    pages_per_slab: u32,
    /// Free pages of the class's slabs; `allocate_in` pops from the end.
    free: Vec<BlockId>,
    /// Keyed pages at reference count 0.
    cached: u32,
}

impl ClassState {
    /// The `TURBINE_KVFMT_*` code of the class's pages.
    fn code(&self, dtype: DType) -> u8 {
        class_code(self.class.format, dtype)
    }
}

/// The base pages carved into slabs, and the non-base classes grown into them.
#[derive(Debug)]
struct Slabs {
    /// Base pages per slab.
    slab_blocks: u32,
    /// Largest page count of a slab over the classes: the ids of slab `s`'s pages are
    /// `base_blocks + s·stride ..`.
    stride: u32,
    /// Owner of each slab: `None` = the base class, `Some(i)` = `classes[i]`.
    owner: Vec<Option<usize>>,
    /// Allocated (referenced or cached) pages per slab of a non-base class.
    live: Vec<u32>,
    classes: Vec<ClassState>,
}

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
    /// `allocate_in` of a format with no page class.
    #[error("no KV page class for format {0}")]
    UnknownFormat(String),
    /// `with_page_classes` refused a class.
    #[error("invalid KV page class: {0}")]
    InvalidClass(String),
}

/// L0 block accounting over one device allocation of `num_blocks × block_bytes`.
pub struct BlockPool {
    layout: KvLayout,
    storage: DeviceBuffer,
    /// Free block ids; `allocate` pops from the end, so ids are handed out in ascending order
    /// from a fresh pool.
    free: Vec<BlockId>,
    /// Reference count per block; 0 exactly when the block is free or cached.
    refcounts: Vec<u32>,
    /// The reservation ledger whose `kv` pool pays for the blocks (P3), and its device.
    ledger: Option<(Arc<Ledger>, DeviceId)>,
    /// Phase 4: the directory holds an L0 location for this block, so it stays cached at
    /// reference count 0 instead of returning to the free list.
    keyed: Vec<bool>,
    /// Keyed blocks at reference count 0.
    cached: u32,
    /// Reclaim order hint (lowest value first); stale entries are skipped.
    reclaim_order: VecDeque<BlockId>,
    /// Cached blocks `allocate` reclaimed since the last `take_reclaimed`.
    reclaimed: Vec<BlockId>,
    /// Base-class pages of the allocation (`num_blocks`); ids `0..base_blocks`.
    base_blocks: u32,
    /// Whole id space: base pages plus every slab's class pages (with classes).
    total_ids: u32,
    /// The executor's view of the classes ([`turbine_tensor::KvPageClass`]); empty without
    /// classes.
    page_classes: Vec<turbine_tensor::KvPageClass>,
    /// The `TURBINE_KVFMT_*` code of every id (`total_ids`); empty without classes.
    class_codes: Vec<u8>,
    /// Page classes other than the base (P6b L0 ladder, S-7): bumped on every page free, the
    /// back-off's `room_epoch` for L0 (a class grows from freed base pages, so any free is
    /// new room).
    room_epoch: u64,
    /// Page classes other than the base (P6b S-5); `None` = the base class only.
    slabs: Option<Slabs>,
}

impl BlockPool {
    /// Monotonic room marker: bumped every time a page returns to a class's free pages or the
    /// base free list (the L0 ladder back-off's resume mark, S-7; cf.
    /// `KvTier::room_epoch`).
    pub fn room_epoch(&self) -> u64 {
        self.room_epoch
    }
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
            keyed: vec![false; cfg.num_blocks as usize],
            cached: 0,
            reclaim_order: VecDeque::new(),
            reclaimed: Vec::new(),
            base_blocks: cfg.num_blocks,
            total_ids: cfg.num_blocks,
            page_classes: Vec::new(),
            class_codes: Vec::new(),
            room_epoch: 0,
            slabs: None,
        })
    }

    /// Adds page classes of other formats (P6b S-5), grown on demand in slabs of
    /// `slab_blocks` base pages. Only on a pool with nothing allocated; a class's `page_bytes`
    /// must split evenly over the layers and fit a slab at least once.
    pub fn with_page_classes(
        mut self,
        classes: &[PageClass],
        slab_blocks: u32,
    ) -> Result<BlockPool, PoolError> {
        let bad = |m: String| Err(PoolError::InvalidClass(m));
        if self.used_blocks() != 0 || self.slabs.is_some() {
            return bad("page classes are set once, on an empty pool".into());
        }
        if slab_blocks == 0 {
            return bad("slab_blocks must be positive".into());
        }
        let layers = u64::from(self.layout.num_layers.max(1));
        let slab_layer_bytes = u64::from(slab_blocks) * self.base_layer_bytes();
        let mut states: Vec<ClassState> = Vec::with_capacity(classes.len());
        for c in classes {
            if c.format == self.base_format() || states.iter().any(|s| s.class.format == c.format) {
                return bad(format!("{} is listed twice or is the base class", c.format));
            }
            if c.page_bytes == 0 || !c.page_bytes.is_multiple_of(layers) {
                return bad(format!(
                    "{}: page_bytes {} does not split over {layers} layers",
                    c.format, c.page_bytes
                ));
            }
            let per_layer = c.page_bytes / layers;
            let pages = slab_layer_bytes / per_layer;
            if pages == 0 {
                return bad(format!(
                    "{}: a page of {per_layer} bytes per layer does not fit a slab of {slab_layer_bytes}",
                    c.format
                ));
            }
            states.push(ClassState {
                class: *c,
                per_layer,
                pages_per_slab: u32::try_from(pages).unwrap_or(u32::MAX),
                free: Vec::new(),
                cached: 0,
            });
        }
        let num_slabs = self.base_blocks / slab_blocks;
        let stride = states.iter().map(|s| s.pages_per_slab).max().unwrap_or(0);
        let ids = u64::from(self.base_blocks) + u64::from(num_slabs) * u64::from(stride);
        if ids > u64::from(u32::MAX) {
            return bad(format!("{ids} page ids do not fit a BlockId"));
        }
        self.refcounts.resize(ids as usize, 0);
        self.keyed.resize(ids as usize, false);
        self.total_ids = ids as u32;
        self.page_classes = states
            .iter()
            .map(|c| turbine_tensor::KvPageClass {
                fmt: class_code(c.class.format, self.layout.dtype),
                per_layer_bytes: c.per_layer,
            })
            .collect();
        self.class_codes = vec![self.base_code(); ids as usize];
        self.slabs = Some(Slabs {
            slab_blocks,
            stride,
            owner: vec![None; num_slabs as usize],
            live: vec![0; num_slabs as usize],
            classes: states,
        });
        Ok(self)
    }

    /// The format of the base class (the L0 dtype).
    pub fn base_format(&self) -> &'static str {
        base_format(self.layout.dtype)
    }

    /// Bytes of one base page in one layer's region.
    fn base_layer_bytes(&self) -> u64 {
        self.layout.block_bytes() / u64::from(self.layout.num_layers.max(1))
    }

    /// `(slab, class, index in the slab)` of a non-base page id, `None` for a base page.
    fn class_page(&self, b: BlockId) -> Option<(usize, usize, u32)> {
        let slabs = self.slabs.as_ref()?;
        let rel = b.0.checked_sub(self.base_blocks)?;
        let slab = (rel / slabs.stride) as usize;
        let class = slabs.owner[slab].expect("a class page lives in a slab its class owns");
        Some((slab, class, rel % slabs.stride))
    }

    /// Whether the pool grows page classes besides the base (`kv.ladder.l0` or the recent
    /// window): a block's format tag can then differ from the pool base.
    pub fn has_page_classes(&self) -> bool {
        self.slabs.is_some()
    }

    /// The v2.11 `block_formats` byte of block `b` (`TURBINE_KVFMT_*`): its page class, the
    /// base dtype's code for a base page.
    pub fn format_code(&self, b: BlockId) -> u8 {
        class_code(self.format_of(b), self.layout.dtype)
    }

    /// The format of block `b`: the base format, or the class of the slab it was carved from.
    pub fn format_of(&self, b: BlockId) -> &'static str {
        match (self.class_page(b), &self.slabs) {
            (Some((_, c, _)), Some(s)) => s.classes[c].class.format,
            _ => self.base_format(),
        }
    }

    /// Every page class — the base first — with its pages, allocated pages and bytes. The
    /// bytes always sum to the pool's one allocation.
    pub fn class_usage(&self) -> Vec<PageClassUsage> {
        let carved = self.slabs.as_ref().map_or(0, |s| {
            s.owner.iter().filter(|o| o.is_some()).count() as u32 * s.slab_blocks
        });
        let base_pages = self.base_blocks - carved;
        let mut out = vec![PageClassUsage {
            class: PageClass {
                format: self.base_format(),
                page_bytes: self.layout.block_bytes(),
            },
            pages: base_pages,
            used_pages: base_pages - self.free.len() as u32,
            bytes: u64::from(base_pages) * self.layout.block_bytes(),
        }];
        if let Some(s) = &self.slabs {
            for (i, c) in s.classes.iter().enumerate() {
                let slabs = s.owner.iter().filter(|o| **o == Some(i)).count() as u32;
                let pages = slabs * c.pages_per_slab;
                out.push(PageClassUsage {
                    class: c.class,
                    pages,
                    used_pages: pages - c.free.len() as u32,
                    bytes: u64::from(slabs * s.slab_blocks) * self.layout.block_bytes(),
                });
            }
        }
        out
    }

    /// Slabs whose base pages are all free, highest first (base allocation hands out the
    /// lowest ids first, so the high slabs empty first).
    fn convertible_slabs(&self) -> Vec<usize> {
        let Some(s) = &self.slabs else {
            return Vec::new();
        };
        (0..s.owner.len())
            .rev()
            .filter(|&i| {
                s.owner[i].is_none()
                    && (i as u32 * s.slab_blocks..(i as u32 + 1) * s.slab_blocks)
                        .all(|b| self.refcounts[b as usize] == 0 && !self.keyed[b as usize])
            })
            .collect()
    }

    /// `n` fresh pages of format `format` with refcount 1, or `Exhausted` with nothing
    /// allocated. The base format is [`BlockPool::allocate`]; another class takes free pages
    /// of its slabs, then carves wholly free base slabs. Cached pages of other classes are not
    /// reclaimed here (the ladder evicts them explicitly, S-7).
    pub fn allocate_in(
        &mut self,
        format: &str,
        n: u32,
    ) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        if format == self.base_format() {
            return self.allocate(n);
        }
        let Some(ci) = self
            .slabs
            .as_ref()
            .and_then(|s| s.classes.iter().position(|c| c.class.format == format))
        else {
            return Err(PoolError::UnknownFormat(format.to_string()));
        };
        let convertible = self.convertible_slabs();
        let base_blocks = self.base_blocks;
        let s = self.slabs.as_mut().expect("class found above");
        let pps = s.classes[ci].pages_per_slab;
        let free = s.classes[ci].free.len() as u32;
        let available = free.saturating_add(pps.saturating_mul(convertible.len() as u32));
        if n > available {
            return Err(PoolError::Exhausted {
                requested: n,
                available,
            });
        }
        let mut carve = convertible.into_iter();
        while (s.classes[ci].free.len() as u32) < n {
            let slab = carve
                .next()
                .expect("counted above: enough convertible slabs");
            s.owner[slab] = Some(ci);
            let first = slab as u32 * s.slab_blocks;
            let carved = first..first + s.slab_blocks;
            self.free.retain(|b| !carved.contains(&b.0));
            let start = base_blocks + slab as u32 * s.stride;
            s.classes[ci]
                .free
                .extend((start..start + pps).rev().map(BlockId));
            let code = s.classes[ci].code(self.layout.dtype);
            for id in start..start + pps {
                self.class_codes[id as usize] = code;
            }
        }
        let mut out = SmallVec::with_capacity(n as usize);
        for _ in 0..n {
            let b = s.classes[ci]
                .free
                .pop()
                .expect("checked above: enough free pages");
            self.refcounts[b.0 as usize] = 1;
            s.live[((b.0 - base_blocks) / s.stride) as usize] += 1;
            out.push(b);
        }
        Ok(out)
    }

    /// Returns block `b` (reference count 0, not cached) to its class's free pages; a slab
    /// whose last page this was goes back to the base class.
    fn free_page(&mut self, b: BlockId) {
        let base_code = class_code(self.base_format(), self.layout.dtype);
        self.room_epoch += 1;
        let Some((slab, ci, _)) = self.class_page(b) else {
            self.free.push(b);
            return;
        };
        let base_blocks = self.base_blocks;
        let s = self.slabs.as_mut().expect("class page");
        s.classes[ci].free.push(b);
        s.live[slab] -= 1;
        if s.live[slab] == 0 {
            let start = base_blocks + slab as u32 * s.stride;
            let ids = start..start + s.classes[ci].pages_per_slab;
            s.classes[ci].free.retain(|p| !ids.contains(&p.0));
            s.owner[slab] = None;
            let first = slab as u32 * s.slab_blocks;
            self.free
                .extend((first..first + s.slab_blocks).rev().map(BlockId));
            for id in first..first + s.slab_blocks {
                self.class_codes[id as usize] = base_code;
            }
        }
    }

    /// One more (`up`) or one fewer cached block in `b`'s class.
    fn add_cached(&mut self, b: BlockId, up: bool) {
        let class = self.class_page(b).map(|(_, c, _)| c);
        let n = match (class, self.slabs.as_mut()) {
            (Some(c), Some(s)) => &mut s.classes[c].cached,
            _ => &mut self.cached,
        };
        if up {
            *n += 1;
        } else {
            *n -= 1;
        }
    }

    /// Link the pool to `ledger`'s `kv` pool on `device` (P3 S-3): blocks are then paid from
    /// per-request reservations through `allocate_reserved`. Panics when the pool's bytes
    /// exceed the ledger's `kv` capacity — the budget must cover every block.
    pub fn with_ledger(mut self, ledger: Arc<Ledger>, device: DeviceId) -> BlockPool {
        let bytes = self.storage.len() as u64;
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

    /// `n` pages of page class `format` paid from `reservation` like
    /// [`BlockPool::allocate_reserved`] (P6b S-5, the recent window's BF16 class): the
    /// commitment is the base page size, the reservation's worst case, whatever the class.
    pub fn allocate_in_reserved(
        &mut self,
        format: &str,
        n: u32,
        reservation: &mut Reservation,
    ) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        let needed = u64::from(n) * self.layout.block_bytes();
        if needed > reservation.bytes() {
            return Err(PoolError::ReservationShort {
                needed,
                uncommitted: reservation.bytes() - reservation.committed(),
            });
        }
        let blocks = self.allocate_in(format, n)?;
        reservation.commit_bytes(needed);
        Ok(blocks)
    }

    pub fn layout(&self) -> KvLayout {
        self.layout
    }

    /// Pages of every class the pool holds now (the base pool size with one class).
    pub fn total_blocks(&self) -> u32 {
        self.class_usage().iter().map(|u| u.pages).sum()
    }

    /// Free pages of every class.
    pub fn free_blocks(&self) -> u32 {
        let class_free: usize = self
            .slabs
            .as_ref()
            .map_or(0, |s| s.classes.iter().map(|c| c.free.len()).sum());
        (self.free.len() + class_free) as u32
    }

    pub fn used_blocks(&self) -> u32 {
        self.total_blocks() - self.free_blocks()
    }

    /// `n` fresh blocks with refcount 1, or `Exhausted` with nothing allocated. When the free
    /// list is short, cached unreferenced blocks are reclaimed first (Phase 4).
    pub fn allocate(&mut self, n: u32) -> Result<SmallVec<[BlockId; 8]>, PoolError> {
        let available = self.available_blocks();
        if n > available {
            return Err(PoolError::Exhausted {
                requested: n,
                available,
            });
        }
        while (self.free.len() as u32) < n {
            let victim = self
                .next_reclaimable()
                .expect("available_blocks counted a cached unreferenced block");
            self.keyed[victim.0 as usize] = false;
            self.cached -= 1;
            self.reclaimed.push(victim);
            self.free.push(victim);
        }
        let mut out = SmallVec::with_capacity(n as usize);
        for _ in 0..n {
            let b = self.free.pop().expect("checked above: enough free blocks");
            self.refcounts[b.0 as usize] = 1;
            out.push(b);
        }
        Ok(out)
    }

    fn is_cached(&self, b: BlockId) -> bool {
        self.keyed[b.0 as usize] && self.refcounts[b.0 as usize] == 0
    }

    /// The next cached unreferenced base block: first from the published order, then by id.
    /// Pages of other classes are never reclaimed into the base class.
    fn next_reclaimable(&mut self) -> Option<BlockId> {
        while let Some(b) = self.reclaim_order.pop_front() {
            if b.0 < self.base_blocks && self.is_cached(b) {
                return Some(b);
            }
        }
        (0..self.base_blocks)
            .map(BlockId)
            .find(|&b| self.is_cached(b))
    }

    /// One more holder of an allocated block (referenced or cached). Panics on a free block (a
    /// caller bug: the block could be handed out again).
    pub fn incref(&mut self, b: BlockId) {
        let cached = self.is_cached(b);
        let rc = &mut self.refcounts[b.0 as usize];
        assert!(*rc > 0 || cached, "incref of free KV block {b:?}");
        *rc += 1;
        if cached {
            self.add_cached(b, false);
        }
    }

    /// Drops one reference to each block; a block whose count reaches 0 returns to the free
    /// list, or stays cached when keyed. Panics on an unreferenced block (a double release).
    pub fn release(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            let rc = &mut self.refcounts[b.0 as usize];
            assert!(*rc > 0, "release of unreferenced KV block {b:?}");
            *rc -= 1;
            if *rc == 0 {
                if self.keyed[b.0 as usize] {
                    self.add_cached(b, true);
                } else {
                    self.free_page(b);
                }
            }
        }
    }

    /// Holders of `b` (0 for a free or cached block).
    pub fn refcount(&self, b: BlockId) -> u32 {
        self.refcounts[b.0 as usize]
    }

    pub fn is_keyed(&self, b: BlockId) -> bool {
        self.keyed[b.0 as usize]
    }

    /// Marks a referenced block as cached content: the directory now holds an L0 location for
    /// it, so reference count 0 keeps it. Panics on an unreferenced block.
    pub fn set_keyed(&mut self, b: BlockId) {
        assert!(
            self.refcounts[b.0 as usize] > 0 || self.keyed[b.0 as usize],
            "set_keyed of unreferenced KV block {b:?}"
        );
        self.keyed[b.0 as usize] = true;
    }

    /// Frees a cached unreferenced block; false when it is referenced, free or not keyed.
    pub fn evict_cached(&mut self, b: BlockId) -> bool {
        if !self.is_cached(b) {
            return false;
        }
        self.keyed[b.0 as usize] = false;
        self.add_cached(b, false);
        self.free_page(b);
        true
    }

    /// Keyed blocks nobody references, of every class (the base ones are reclaimable by
    /// `allocate`).
    pub fn cached_unreferenced(&self) -> u32 {
        self.cached
            + self
                .slabs
                .as_ref()
                .map_or(0, |s| s.classes.iter().map(|c| c.cached).sum::<u32>())
    }

    /// Blocks with at least one holder: every allocated page less every cached unreferenced
    /// one, the base class and the page classes alike (a classed copy the ladder's L0 rung or
    /// the recent window left cached is not a holder; the P6b-exit soak's `kv_idle` failure
    /// read this through `EngineLoop::sync_kv_held`).
    pub fn referenced_blocks(&self) -> u32 {
        self.used_blocks() - self.cached_unreferenced()
    }

    /// Free plus cached unreferenced base blocks: what `allocate` can hand out.
    pub fn available_blocks(&self) -> u32 {
        self.free.len() as u32 + self.cached
    }

    /// The order in which `allocate` reclaims cached blocks, lowest value first; blocks that are
    /// no longer cached when their turn comes are skipped.
    pub fn set_reclaim_order(&mut self, order: Vec<BlockId>) {
        self.reclaim_order = order.into();
    }

    /// Cached blocks `allocate` reclaimed since the last call, oldest first.
    pub fn take_reclaimed(&mut self) -> Vec<BlockId> {
        std::mem::take(&mut self.reclaimed)
    }

    /// A table for a forked sequence (`n > 1`): full blocks are shared by reference count;
    /// a partial tail block gets a fresh block of the tail's own page class (a byte copy of a
    /// class page into a base page would hold the wrong bytes), returned as `(src, dst)` for
    /// the engine to copy with `copy_blocks`. All-or-nothing: `Exhausted` leaves the pool
    /// unchanged.
    pub fn fork(
        &mut self,
        table: &BlockTable,
    ) -> Result<(BlockTable, Option<(BlockId, BlockId)>), PoolError> {
        let block_tokens = self.layout.block_tokens.max(1);
        let partial_tail = !table.tokens.is_multiple_of(block_tokens) && !table.blocks.is_empty();
        let tail = if partial_tail {
            let src = *table.blocks.last().expect("partial tail exists");
            let tail = if self.class_page(src).is_some() {
                let fmt = self.format_of(src);
                self.allocate_in(fmt, 1)?[0]
            } else {
                self.allocate(1)?[0]
            };
            Some(tail)
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

    /// The device bytes of block `b`: one `(address, length)` segment per layer, the block's K
    /// and V of that layer (the [`KvPoolView`] layout). A block copy moves exactly these. A
    /// page of another class lies inside its slab's range of each layer's region, as long as
    /// that class's page per layer.
    pub fn block_segments(&self, b: BlockId) -> SmallVec<[(DevicePtr, usize); 32]> {
        let view = self.view();
        let base_layer = self.base_layer_bytes();
        let (start, per_layer) = match (self.class_page(b), &self.slabs) {
            (Some((slab, ci, i)), Some(s)) => {
                let slab_start = slab as u64 * u64::from(s.slab_blocks) * base_layer;
                let c = &s.classes[ci];
                (slab_start + u64::from(i) * c.per_layer, c.per_layer)
            }
            _ => (u64::from(b.0) * base_layer, base_layer),
        };
        let base = self.storage.ptr();
        (0..u64::from(self.layout.num_layers))
            .map(|l| {
                let off = l * view.layer_stride_bytes + start;
                (base.offset(off), per_layer as usize)
            })
            .collect()
    }

    /// The device storage and layout for the executor: the base class's addressing plus, with
    /// page classes, the per-class resolution of the ids above the base ([`KvPageClasses`]).
    /// `num_blocks` is the whole id space.
    pub fn view(&self) -> KvPoolView<'_> {
        let per_layer_block = self.layout.layer_block_bytes();
        let layer_stride_bytes = u64::from(self.base_blocks) * per_layer_block;
        let classes = self.slabs.as_ref().map(|s| KvPageClasses {
            num_blocks: self.total_ids,
            base_blocks: self.base_blocks,
            base_page_bytes: per_layer_block,
            slab_stride: s.stride,
            slab_base_blocks: s.slab_blocks,
            page_classes: &self.page_classes,
            class_codes: &self.class_codes,
        });
        KvPoolView {
            storage: &self.storage,
            layout: self.layout,
            num_blocks: self.total_ids,
            layer_stride_bytes,
            classes,
        }
    }

    /// The `TURBINE_KVFMT_*` code of the base class (`kv.dtype`).
    pub fn base_code(&self) -> u8 {
        class_code(self.base_format(), self.layout.dtype)
    }

    /// The page bytes of block `b` in one layer's region (`view().classes` resolution, or the
    /// base page when the pool is flat).
    pub fn page_per_layer(&self, b: BlockId) -> u64 {
        match self.class_page(b) {
            Some((_, ci, _)) => self.slabs.as_ref().expect("class page").classes[ci].per_layer,
            None => self.layout.layer_block_bytes(),
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

    /// One segment per layer, each the block's K and V bytes of that layer at the offset the
    /// executor's `KvPoolView` reads (P4 Task 15: the copy stream moves exactly these bytes).
    /// Breaks if a segment is missed, overlaps another block, or is sized wrongly.
    #[test]
    fn block_segments_cover_the_block_per_layer() {
        let p = pool(4);
        let view = p.view();
        let base = view.storage.ptr().addr();
        let per_layer = small_layout().block_bytes() as usize / 2;
        for b in 0..4u32 {
            let segs = p.block_segments(BlockId(b));
            assert_eq!(segs.len(), 2, "one segment per layer");
            for (l, (ptr, len)) in segs.iter().enumerate() {
                assert_eq!(*len, per_layer);
                let expect =
                    base + l as u64 * view.layer_stride_bytes + u64::from(b) * per_layer as u64;
                assert_eq!(ptr.addr(), expect, "block {b} layer {l}");
            }
            let total: usize = segs.iter().map(|(_, n)| n).sum();
            assert_eq!(total as u64, small_layout().block_bytes());
        }
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

    /// P6b exit (the ladder soak's `kv_idle` failure, 2026-10-04): a cached unreferenced page
    /// of a non-base class is not referenced. `referenced_blocks` feeds the ledger's `kv` held
    /// bytes (`sync_kv_held`) and the pressure document's kv pool, so a phantom referenced
    /// count keeps `kv_utilization` up and the soak's idle check false forever once the
    /// ladder's L0 rung or the recent window leaves classed copies cached at idle. Breaks if
    /// `referenced_blocks` subtracts only the base class's cached count.
    #[test]
    fn classed_cached_pages_are_not_referenced() {
        let layout = KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        let classes = [PageClass {
            format: "tq4",
            page_bytes: 2 * 2 * 16 * 144,
        }];
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 23);
        let mut p = BlockPool::new(
            BlockPoolConfig {
                layout,
                num_blocks: 16,
            },
            mem,
        )
        .unwrap()
        .with_page_classes(&classes, 4)
        .unwrap();
        // A live base block and a live classed page are both referenced.
        let base = p.allocate(2).unwrap();
        let classed = p.allocate_in("tq4", 1).unwrap();
        assert_eq!(p.referenced_blocks(), 3);
        // The classed page finishes and stays cached: keyed, then released.
        p.set_keyed(classed[0]);
        p.release(&classed);
        assert_eq!(p.cached_unreferenced(), 1);
        assert_eq!(
            p.referenced_blocks(),
            2,
            "the cached classed page is not referenced"
        );
        // And so is a cached base page; the other base block stays held.
        p.set_keyed(base[0]);
        p.release(&[base[0]]);
        assert_eq!(p.cached_unreferenced(), 2);
        assert_eq!(p.referenced_blocks(), 1, "only the held block");
        p.release(&[base[1]]);
        assert_eq!(p.referenced_blocks(), 0, "idle: nothing referenced");
    }

    /// P6b S-5: page classes carved from the one allocation. A realistic layout (2 layers,
    /// 2 KV heads, head_dim 128, 16 tokens, BF16: 32 KiB pages) with `tq4` (9216-byte pages,
    /// 14 per 4-page slab) and `tq2` (5120 bytes, 25 per slab). Breaks if a class leaks a slab,
    /// the byte accounting drifts from the allocation, a page's bytes disagree with its tag or
    /// overlap another live page, or a base allocation reclaims a page of another class.
    #[test]
    fn page_classes() {
        let layout = KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 128,
            dtype: DType::BF16,
            block_tokens: 16,
        };
        let classes = [
            PageClass {
                format: "tq4",
                page_bytes: 2 * 2 * 16 * 144,
            },
            PageClass {
                format: "tq2",
                page_bytes: 2 * 2 * 16 * 80,
            },
        ];
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 23);
        let cfg = BlockPoolConfig {
            layout,
            num_blocks: 64,
        };
        let mut p = BlockPool::new(cfg, mem.clone())
            .unwrap()
            .with_page_classes(&classes, 4)
            .unwrap();
        let total_bytes = 64 * layout.block_bytes();
        let class_of = |p: &BlockPool, f: &str| {
            p.class_usage()
                .into_iter()
                .find(|u| u.class.format == f)
                .unwrap()
        };
        // Every invariant against the live blocks.
        let check = |p: &BlockPool, live: &[BlockId]| {
            let usage = p.class_usage();
            assert_eq!(usage[0].class.format, "bf16");
            assert_eq!(
                usage.iter().map(|u| u.bytes).sum::<u64>(),
                total_bytes,
                "class bytes must sum to the one allocation: {usage:?}"
            );
            for u in &usage {
                assert!(u64::from(u.pages) * u.class.page_bytes <= u.bytes, "{u:?}");
            }
            assert_eq!(
                usage.iter().map(|u| u.used_pages).sum::<u32>(),
                p.used_blocks()
            );
            assert_eq!(p.used_blocks() + p.free_blocks(), p.total_blocks());
            let start = p.view().storage.ptr().addr();
            let end = start + total_bytes;
            let mut spans = Vec::new();
            for &b in live {
                let fmt = p.format_of(b);
                let page = usage.iter().find(|u| u.class.format == fmt).unwrap();
                let segs = p.block_segments(b);
                let bytes: usize = segs.iter().map(|(_, n)| n).sum();
                assert_eq!(bytes as u64, page.class.page_bytes, "{b:?} tagged {fmt}");
                for (ptr, n) in segs {
                    assert!(ptr.addr() >= start && ptr.addr() + n as u64 <= end);
                    spans.push((ptr.addr(), ptr.addr() + n as u64, b));
                }
            }
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "{:?} overlaps {:?}", w[0].2, w[1].2);
            }
        };
        check(&p, &[]);
        assert_eq!(p.total_blocks(), 64);
        assert_eq!(class_of(&p, "tq4").pages, 0);

        // tq4 grows one slab (4 base pages → 14 pages), then a second.
        let a = p.allocate_in("tq4", 3).unwrap();
        assert!(a.iter().all(|b| p.format_of(*b) == "tq4"));
        assert_eq!(class_of(&p, "tq4").pages, 14);
        assert_eq!(class_of(&p, "bf16").pages, 60);
        assert_eq!(
            p.available_blocks(),
            60,
            "base allocation sees only base pages"
        );
        let b = p.allocate_in("tq4", 12).unwrap();
        assert_eq!(class_of(&p, "tq4").pages, 28);
        let c = p.allocate_in("tq2", 30).unwrap();
        assert_eq!(class_of(&p, "tq2").pages, 50);
        let base = p.allocate_in("bf16", 5).unwrap();
        assert!(base.iter().all(|b| p.format_of(*b) == "bf16"));
        let live: Vec<BlockId> = a.iter().chain(&b).chain(&c).chain(&base).copied().collect();
        check(&p, &live);
        assert!(matches!(
            p.allocate_in("tq3", 1),
            Err(PoolError::UnknownFormat(_))
        ));

        // A keyed tq4 page stays cached at refcount 0 and keeps its slab; a base allocation
        // never reclaims it, even when it heads the published order.
        p.set_keyed(a[0]);
        p.release(&a);
        p.release(&b);
        assert_eq!(
            class_of(&p, "tq4").pages,
            14,
            "one slab kept by the cached page"
        );
        assert_eq!(class_of(&p, "tq4").used_pages, 1);
        assert_eq!(p.cached_unreferenced(), 1);
        p.set_reclaim_order(vec![a[0]]);
        let rest = p.available_blocks();
        let more = p.allocate(rest).unwrap();
        assert!(p.take_reclaimed().is_empty());
        assert_eq!(p.format_of(a[0]), "tq4");
        let one = p.allocate_in("tq2", 1);
        assert!(one.is_ok(), "tq2 still has free pages in its slabs");
        p.release(&one.unwrap());
        // No free base slab left: a tq4 page beyond its slab is exhausted, nothing changes.
        let before = p.class_usage();
        let err = p.allocate_in("tq4", 14).unwrap_err();
        assert!(matches!(
            err,
            PoolError::Exhausted {
                requested: 14,
                available: 13
            }
        ));
        assert_eq!(p.class_usage(), before);
        let live: Vec<BlockId> = c.iter().chain(&base).chain(&more).copied().collect();
        check(&p, &live);

        // Freeing returns empty slabs to the base class; the cached page's slab goes back once
        // it is evicted.
        p.release(&c);
        assert_eq!(class_of(&p, "tq2").pages, 0);
        assert!(p.evict_cached(a[0]));
        assert_eq!(class_of(&p, "tq4").pages, 0);
        p.release(&base);
        p.release(&more);
        check(&p, &[]);
        assert_eq!((p.total_blocks(), p.free_blocks()), (64, 64));
        assert_eq!(p.cached_unreferenced(), 0);
        // The pool is whole again: every base page allocates.
        let all = p.allocate(64).unwrap();
        assert_eq!(all.len(), 64);
        p.release(&all);

        // Invalid classes are refused.
        let bad = |classes: &[PageClass], slab| {
            BlockPool::new(cfg, mem.clone())
                .unwrap()
                .with_page_classes(classes, slab)
                .unwrap_err()
        };
        let odd = PageClass {
            format: "tq4",
            page_bytes: 9217,
        };
        assert!(matches!(bad(&[odd], 4), PoolError::InvalidClass(_)));
        let base_again = PageClass {
            format: "bf16",
            page_bytes: 1024,
        };
        assert!(matches!(bad(&[base_again], 4), PoolError::InvalidClass(_)));
        assert!(matches!(bad(&classes, 0), PoolError::InvalidClass(_)));
    }

    /// Phase 4: a keyed block stays cached at reference count 0; `allocate` reclaims cached
    /// blocks in the published order (then by id) and reports them through `take_reclaimed`.
    #[test]
    fn keyed_blocks_stay_cached_and_reclaim_in_order() {
        let mut p = pool(4);
        let held = p.allocate(3).unwrap();
        p.set_keyed(BlockId(0));
        p.set_keyed(BlockId(1));
        assert!(p.is_keyed(BlockId(0)) && !p.is_keyed(BlockId(2)));
        p.release(&held);
        assert_eq!(p.free_blocks(), 2, "the unkeyed block is free again");
        assert_eq!(p.cached_unreferenced(), 2);
        assert_eq!(p.available_blocks(), 4);
        assert_eq!((p.used_blocks(), p.referenced_blocks()), (2, 0));
        assert_eq!(p.refcount(BlockId(0)), 0);

        // A cached block can be attached again by reference.
        p.incref(BlockId(0));
        assert_eq!((p.refcount(BlockId(0)), p.cached_unreferenced()), (1, 1));
        assert_eq!(p.referenced_blocks(), 1);
        p.release(&[BlockId(0)]);
        assert_eq!(
            p.cached_unreferenced(),
            2,
            "released back to cached, not free"
        );

        // Not enough free blocks: reclaim in the published order.
        p.set_reclaim_order(vec![BlockId(1), BlockId(0)]);
        let three = p.allocate(3).unwrap();
        assert_eq!(p.take_reclaimed(), vec![BlockId(1)]);
        assert!(p.take_reclaimed().is_empty());
        assert!(three.contains(&BlockId(1)));
        assert!(!p.is_keyed(BlockId(1)), "a reclaimed block is fresh");
        assert_eq!(p.refcount(BlockId(1)), 1);
        let err = p.allocate(2).unwrap_err();
        assert!(matches!(
            err,
            PoolError::Exhausted {
                requested: 2,
                available: 1
            }
        ));
        assert_eq!(p.allocate(1).unwrap().as_slice(), &[BlockId(0)]);
        assert_eq!(p.take_reclaimed(), vec![BlockId(0)]);
        assert_eq!(p.available_blocks(), 0);
        p.release(&three);
        p.release(&[BlockId(0)]);

        // Without a published order the lowest cached id goes first; evict_cached frees only
        // unreferenced cached blocks.
        let all = p.allocate(4).unwrap();
        for &b in &all {
            p.set_keyed(b);
        }
        let rest: Vec<BlockId> = all.iter().copied().filter(|b| *b != BlockId(3)).collect();
        p.release(&rest);
        assert!(!p.evict_cached(BlockId(3)), "still referenced");
        assert!(p.evict_cached(BlockId(2)));
        assert!(!p.evict_cached(BlockId(2)), "already free");
        assert_eq!((p.free_blocks(), p.cached_unreferenced()), (1, 2));
        // The reclaimed block joins the free list last, so it is handed out first.
        assert_eq!(p.allocate(2).unwrap().as_slice(), &[BlockId(0), BlockId(2)]);
        assert_eq!(p.take_reclaimed(), vec![BlockId(0)]);
    }
}
