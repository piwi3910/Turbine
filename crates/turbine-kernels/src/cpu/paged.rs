//! Paged-KV reference ops: ragged-batch attention over one layer of the block pool
//! (`attention_prefill_paged`, `attention_decode_paged`) and the block fork (`copy_blocks`).
//!
//! The pool is read and written block by block (never whole), so the cost of a call is
//! proportional to the blocks its sequences own, not to the pool size.
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use turbine_tensor::tensor::contiguous_strides;

use super::{
    CpuReference, FloatCodec, expect_rank, expect_shape, invalid, load, load_i32, math, store,
};
use crate::KernelError;
use crate::ops::{KvCopyConfig, KvCopyContext, KvCopyKernel, PagedAttentionContext};

/// One sequence of the ragged batch.
struct Seq {
    /// First row of the sequence in `q`/`k_new`/`v_new`/`out`.
    row: usize,
    q_len: usize,
    kv_len: usize,
    /// The blocks holding tokens `0..kv_len`, in token order.
    blocks: Vec<usize>,
}

fn to_usize(name: &str, v: i32) -> Result<usize, KernelError> {
    usize::try_from(v).map_err(|_| invalid(format!("{name} = {v} is negative")))
}

/// Validates the batch description (`q_indptr`, `kv_lens`, `block_table`) against the tensor
/// shapes and the `max_*` bounds, and resolves every sequence's blocks.
fn sequences(
    ctx: &PagedAttentionContext<'_>,
    total_q: usize,
    num_blocks: usize,
    block_tokens: usize,
) -> Result<Vec<Seq>, KernelError> {
    expect_rank("kv_lens", &ctx.kv_lens, 1)?;
    let num_seqs = ctx.kv_lens.shape[0];
    let max_blocks = ctx.max_blocks_per_seq as usize;
    expect_shape("q_indptr", &ctx.q_indptr, &[num_seqs + 1])?;
    expect_shape("block_table", &ctx.block_table, &[num_seqs, max_blocks])?;
    let indptr = load_i32(&ctx.q_indptr)?;
    let kv_lens = load_i32(&ctx.kv_lens)?;
    let table = load_i32(&ctx.block_table)?;
    if indptr[0] != 0 || to_usize("q_indptr", indptr[num_seqs])? != total_q {
        return Err(invalid(format!(
            "q_indptr must run from 0 to total_q {total_q}, got {indptr:?}"
        )));
    }
    let mut seqs = Vec::with_capacity(num_seqs);
    for s in 0..num_seqs {
        let row = to_usize("q_indptr", indptr[s])?;
        let end = to_usize("q_indptr", indptr[s + 1])?;
        let kv_len = to_usize("kv_lens", kv_lens[s])?;
        let Some(q_len) = end.checked_sub(row) else {
            return Err(invalid(format!("q_indptr decreases at sequence {s}")));
        };
        if q_len > kv_len || q_len > ctx.max_q_len as usize || kv_len > ctx.max_kv_len as usize {
            return Err(invalid(format!(
                "sequence {s}: q_len {q_len}, kv_len {kv_len} exceed max_q_len {} / max_kv_len {} or q_len > kv_len",
                ctx.max_q_len, ctx.max_kv_len
            )));
        }
        let needed = kv_len.div_ceil(block_tokens);
        if needed > max_blocks {
            return Err(invalid(format!(
                "sequence {s}: kv_len {kv_len} needs {needed} blocks of {block_tokens} tokens but max_blocks_per_seq is {max_blocks}"
            )));
        }
        let blocks = table[s * max_blocks..s * max_blocks + needed]
            .iter()
            .map(|&b| match usize::try_from(b) {
                Ok(b) if b < num_blocks => Ok(b),
                _ => Err(invalid(format!(
                    "sequence {s}: block id {b} is outside the pool of {num_blocks} blocks"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        seqs.push(Seq {
            row,
            q_len,
            kv_len,
            blocks,
        });
    }
    Ok(seqs)
}

/// Appends `k_new`/`v_new` into their page slots, then runs causal GQA attention per sequence
/// over its gathered K/V history with the same math as the contiguous op.
pub(super) fn attention(ctx: &PagedAttentionContext<'_>) -> Result<(), KernelError> {
    let hq = ctx.cfg.num_q_heads as usize;
    let hkv = ctx.cfg.num_kv_heads as usize;
    let d = ctx.cfg.head_dim as usize;
    let block_tokens = ctx.cfg.block_tokens.unwrap_or(0) as usize;
    expect_rank("q", &ctx.q, 3)?;
    let total_q = ctx.q.shape[0];
    expect_shape("q", &ctx.q, &[total_q, hq, d])?;
    expect_shape("out", &ctx.out, &[total_q, hq, d])?;
    expect_shape("k_new", &ctx.k_new, &[total_q, hkv, d])?;
    expect_shape("v_new", &ctx.v_new, &[total_q, hkv, d])?;

    let pool = &ctx.kv_layer;
    expect_rank("kv_layer", pool, 5)?;
    let num_blocks = pool.shape[0];
    expect_shape("kv_layer", pool, &[num_blocks, 2, block_tokens, hkv, d])?;
    if pool.strides != contiguous_strides(&pool.shape) {
        return Err(invalid(format!(
            "kv_layer must be dense, has strides {:?}",
            pool.strides.as_slice()
        )));
    }
    let codec = FloatCodec::require(pool.dtype)?;
    let es = pool.dtype.size_bytes();
    let token_elems = hkv * d;
    let block_bytes = 2 * block_tokens * token_elems * es;
    if pool.slice.len() < num_blocks * block_bytes {
        return Err(invalid(format!(
            "kv_layer slice holds {} bytes, {num_blocks} blocks need {}",
            pool.slice.len(),
            num_blocks * block_bytes
        )));
    }
    let seqs = sequences(ctx, total_q, num_blocks, block_tokens)?;

    let q = load(&ctx.q)?;
    let k_new = load(&ctx.k_new)?;
    let v_new = load(&ctx.v_new)?;
    let block_slice = |b: usize| pool.slice.sub(b * block_bytes, block_bytes);

    // Append: every touched block is read once, patched and written back once.
    let mut touched: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for seq in &seqs {
        let first_new = seq.kv_len - seq.q_len;
        for i in 0..seq.q_len {
            let pos = first_new + i;
            let block = seq.blocks[pos / block_tokens];
            let slot = pos % block_tokens;
            let bytes = match touched.entry(block) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(block_slice(block).read_bytes()?),
            };
            let src = (seq.row + i) * token_elems;
            for (half, values) in [(0, &k_new), (1, &v_new)] {
                let base = (half * block_tokens + slot) * token_elems;
                for (j, &v) in values[src..src + token_elems].iter().enumerate() {
                    codec.encode(v, &mut bytes[(base + j) * es..]);
                }
            }
        }
    }
    for (&block, bytes) in &touched {
        block_slice(block).write_bytes(bytes)?;
    }

    // Attend: gather each sequence's K/V from its blocks (after the append).
    let mut out = vec![0f32; total_q * hq * d];
    for seq in &seqs {
        let mut k = Vec::with_capacity(seq.kv_len * token_elems);
        let mut v = Vec::with_capacity(seq.kv_len * token_elems);
        for (n, &block) in seq.blocks.iter().enumerate() {
            let tokens = (seq.kv_len - n * block_tokens).min(block_tokens);
            let bytes = match touched.get(&block) {
                Some(bytes) => bytes.clone(),
                None => block_slice(block).read_bytes()?,
            };
            for (half, dst) in [(0, &mut k), (1, &mut v)] {
                let base = half * block_tokens * token_elems;
                dst.extend(
                    (base..base + tokens * token_elems).map(|e| codec.decode(&bytes[e * es..])),
                );
            }
        }
        let shape = math::AttnShape {
            q_len: seq.q_len,
            q_start: seq.kv_len - seq.q_len,
            hq,
            hkv,
            d,
            causal: ctx.cfg.causal,
        };
        let rows = seq.row * hq * d..(seq.row + seq.q_len) * hq * d;
        let o = math::attention(&q[rows.clone()], &k, &v, &shape, ctx.scale);
        out[rows].copy_from_slice(&o);
    }
    store(&ctx.out, &out)
}

fn byte_offset(name: &str, v: u64) -> Result<usize, KernelError> {
    usize::try_from(v).map_err(|_| invalid(format!("{name} = {v} does not fit usize")))
}

impl KvCopyKernel for CpuReference {
    fn supports(&self, cfg: &KvCopyConfig) -> bool {
        cfg.num_layers > 0 && cfg.block_bytes > 0
    }

    fn implementation(&self, _cfg: &KvCopyConfig) -> String {
        "cpu_copy_blocks".into()
    }

    fn execute(&self, ctx: &mut KvCopyContext<'_>) -> Result<(), KernelError> {
        let block_bytes = byte_offset("block_bytes", ctx.block_bytes)?;
        let layer_stride = byte_offset("layer_stride_bytes", ctx.layer_stride_bytes)?;
        let layers = ctx.num_layers as usize;
        if block_bytes == 0 || layer_stride < block_bytes {
            return Err(invalid(format!(
                "block_bytes {block_bytes} must be positive and at most layer_stride_bytes {layer_stride}"
            )));
        }
        let blocks_per_layer = layer_stride / block_bytes;
        if layers > 0
            && (layers - 1) * layer_stride + blocks_per_layer * block_bytes > ctx.pool.len()
        {
            return Err(invalid(format!(
                "{layers} layers of {layer_stride} bytes exceed the pool of {} bytes",
                ctx.pool.len()
            )));
        }
        for &(src, dst) in ctx.pairs {
            for b in [src.0, dst.0] {
                if b as usize >= blocks_per_layer {
                    return Err(invalid(format!(
                        "block id {b} is outside the {blocks_per_layer} blocks of a layer"
                    )));
                }
            }
        }
        for layer in 0..layers {
            for &(src, dst) in ctx.pairs {
                let at = |b: u32| layer * layer_stride + b as usize * block_bytes;
                let bytes = ctx.pool.sub(at(src.0), block_bytes).read_bytes()?;
                ctx.pool.sub(at(dst.0), block_bytes).write_bytes(&bytes)?;
            }
        }
        Ok(())
    }
}
