//! Borrowed view of the L0 KV block pool, handed to the executor. Lives here (not in
//! `turbine-kv`) because `turbine-model` may not depend on `turbine-kv`.

use turbine_core::types::KvLayout;

use crate::buffer::DeviceBuffer;

/// One page class of the pool besides the base (P6b S-5/S-7): the `TURBINE_KVFMT_*` code its
/// pages hold and the bytes of one page in one layer's region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPageClass {
    pub fmt: u8,
    pub per_layer_bytes: u64,
}

/// Per-class page addressing (P6b S-5/S-7, the recent window's BF16 class and the ladder's L0
/// rungs): the pool constants a consumer needs to resolve any block id to its page bytes.
///
/// Block ids below `base_blocks` are base pages: layer `l`'s copy of block `b` starts at
/// `l × layer_stride_bytes + b × base_page_bytes`. Ids at and above `base_blocks` are class
/// pages: with `rel = b − base_blocks`, `slab = rel / slab_stride` and `i = rel % slab_stride`,
/// layer `l`'s copy starts at `l × layer_stride_bytes + slab × slab_base_blocks ×
/// base_page_bytes + i × page_classes[c].per_layer_bytes`, where `c` is the `page_classes`
/// entry whose [`KvPageClass::fmt`] equals the block's format code. Every id's page lies
/// inside its layer's `layer_stride_bytes` region, and the pool constants never change, so a
/// decode graph can bake them.
#[derive(Clone, Copy, Debug)]
pub struct KvPageClasses<'a> {
    /// The whole id space: base pages plus every slab's class pages.
    pub num_blocks: u32,
    /// Ids below this are flat base pages.
    pub base_blocks: u32,
    /// One base page's bytes in one layer's region.
    pub base_page_bytes: u64,
    /// Class page ids one slab holds (the largest class page count of a slab).
    pub slab_stride: u32,
    /// Base pages one slab was carved from.
    pub slab_base_blocks: u32,
    /// The pool's classes besides the base; `fmt` codes are unique.
    pub page_classes: &'a [KvPageClass],
    /// The `TURBINE_KVFMT_*` code of every block id of the pool (len = `num_blocks`): the base
    /// code for a base page, its class's code for a class page. Maintained by the pool as
    /// slabs are carved and returned.
    pub class_codes: &'a [u8],
}

impl<'a> KvPageClasses<'a> {
    /// The class entry of format code `fmt`.
    pub fn class_of(&self, fmt: u8) -> Option<KvPageClass> {
        self.page_classes.iter().copied().find(|c| c.fmt == fmt)
    }

    /// The code of the base class (every id below `base_blocks`).
    pub fn base_code(&self) -> u8 {
        self.class_codes
            .first()
            .copied()
            .unwrap_or(KV_FMT_BASE_FALLBACK)
    }

    /// The byte offset of block `b`'s page within one layer's region, by the block's format
    /// code `fmt`; `None` when the id or the format is not one of the pool's.
    pub fn page_offset(&self, b: u32, fmt: u8) -> Option<u64> {
        if b >= self.num_blocks {
            return None;
        }
        if b < self.base_blocks {
            return Some(u64::from(b) * self.base_page_bytes);
        }
        let per = self.class_of(fmt)?.per_layer_bytes;
        let rel = u64::from(b - self.base_blocks);
        let (slab, i) = (
            rel / u64::from(self.slab_stride),
            rel % u64::from(self.slab_stride),
        );
        Some(slab * u64::from(self.slab_base_blocks) * self.base_page_bytes + i * per)
    }

    /// Bytes of block `b`'s page in one layer's region, by its format code `fmt`.
    pub fn page_bytes(&self, fmt: u8) -> Option<u64> {
        if fmt == self.base_code() {
            return Some(self.base_page_bytes);
        }
        self.class_of(fmt).map(|c| c.per_layer_bytes)
    }
}

/// The base format code when a pool reports no class codes at all (never in practice: the
/// codes slice is as long as the id space).
const KV_FMT_BASE_FALLBACK: u8 = 0;

/// The pool's single device allocation plus its layout. Layer `l` occupies
/// `[num_blocks, 2, block_tokens, kv_heads, head_dim]` elements of `layout.dtype` starting at
/// byte `l × layer_stride_bytes`; block `b` of layer `l` starts at
/// `l × layer_stride_bytes + b × layout.block_bytes() / layout.num_layers`.
///
/// `num_blocks` is the whole id space the pool can hand out: with page classes
/// ([`Self::classes`]) a block's page follows [`KvPageClasses`] instead of the flat formula,
/// and `layer_stride_bytes` stays one base class's region (base blocks × one base page).
#[derive(Clone, Copy, Debug)]
pub struct KvPoolView<'a> {
    pub storage: &'a DeviceBuffer,
    pub layout: KvLayout,
    pub num_blocks: u32,
    pub layer_stride_bytes: u64,
    /// Page classes besides the base; `None` = every id is a flat base page.
    pub classes: Option<KvPageClasses<'a>>,
}

impl<'a> KvPoolView<'a> {
    /// A flat pool: every id `0..num_blocks` is a base page.
    pub fn flat(storage: &'a DeviceBuffer, layout: KvLayout, num_blocks: u32) -> KvPoolView<'a> {
        KvPoolView {
            storage,
            layout,
            num_blocks,
            layer_stride_bytes: u64::from(num_blocks) * layout.layer_block_bytes(),
            classes: None,
        }
    }
}
