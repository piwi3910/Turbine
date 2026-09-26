//! Borrowed view of the L0 KV block pool, handed to the executor. Lives here (not in
//! `turbine-kv`) because `turbine-model` may not depend on `turbine-kv`.

use turbine_core::types::KvLayout;

use crate::buffer::DeviceBuffer;

/// The pool's single device allocation plus its layout. Layer `l` occupies
/// `[num_blocks, 2, block_tokens, kv_heads, head_dim]` elements of `layout.dtype` starting at
/// byte `l × layer_stride_bytes`; block `b` of layer `l` starts at
/// `l × layer_stride_bytes + b × layout.block_bytes() / layout.num_layers`.
#[derive(Clone, Copy, Debug)]
pub struct KvPoolView<'a> {
    pub storage: &'a DeviceBuffer,
    pub layout: KvLayout,
    pub num_blocks: u32,
    pub layer_stride_bytes: u64,
}
