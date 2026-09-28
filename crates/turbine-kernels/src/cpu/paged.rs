//! Paged-KV reference attention: ragged-batch attention over one layer of the block pool
//! (`attention_prefill_paged`, `attention_decode_paged`); `AttentionKernel::execute_paged`
//! (attention.rs) runs it. The block fork (`copy_blocks`) is in kv_copy.rs.
//!
//! The pool is read and written block by block (never whole), so the cost of a call is
//! proportional to the blocks its sequences own, not to the pool size.
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use turbine_tensor::tensor::contiguous_strides;

use super::{FloatCodec, expect_rank, expect_shape, invalid, load, load_i32, math, store};
use crate::KernelError;
use crate::ops::PagedAttentionContext;

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
        let dt = ctx.cfg.dtype;
        let o = math::attention(&q[rows.clone()], &k, &v, &shape, ctx.scale, |p| {
            super::round_to(dt, p)
        });
        out[rows].copy_from_slice(&o);
    }
    store(&ctx.out, &out)
}

#[cfg(test)]
mod tests {
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    #[test]
    fn paged_attention_equals_contiguous_and_moe_route_ties() {
        let mem = HostMemory::new(DeviceId(0), 1 << 22) as Arc<dyn DeviceMemory>;
        let cpu = cpu_reference_provider();
        let (hq, hkv, d, bt) = (4usize, 2usize, 8usize, 16usize);
        let dtype = DType::BF16;
        let q_lens = [5usize, 1, 3];
        let kv_lens = [20usize, 9, 3];
        // Shuffled block table over an 8-block pool; unused entries are -1 and must not be read.
        let table = [5, 2, 7, -1, 0, -1];
        let num_blocks = 8;
        let paged_cfg = AttentionConfig {
            kind: AttentionKind::PrefillPaged,
            num_q_heads: hq as u32,
            num_kv_heads: hkv as u32,
            head_dim: d as u32,
            dtype,
            block_tokens: Some(bt as u32),
            causal: true,
        };
        let attn = cpu.attention().expect("attention family");
        assert!(attn.supports(&paged_cfg));
        assert_eq!(
            attn.implementation(&paged_cfg),
            "cpu_attention_paged_f32acc"
        );

        // Every sequence's full K/V history and its new queries.
        let full_k: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(10 + s as u64, kv_lens[s] * hkv * d))
            .collect();
        let full_v: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(20 + s as u64, kv_lens[s] * hkv * d))
            .collect();
        let qs: Vec<Vec<f32>> = (0..3)
            .map(|s| seeded(30 + s as u64, q_lens[s] * hq * d))
            .collect();

        // The pool of this layer holds the tokens before this step (kv_len − q_len of them).
        let pool = Tensor::empty(&mem, &[num_blocks, 2, bt, hkv, d], dtype).expect("pool");
        let mut pool_values = vec![0f32; num_blocks * 2 * bt * hkv * d];
        let token = hkv * d;
        for s in 0..3 {
            for p in 0..kv_lens[s] - q_lens[s] {
                let block = table[s * 2 + p / bt] as usize;
                let k_at = ((block * 2) * bt + p % bt) * token;
                let v_at = ((block * 2 + 1) * bt + p % bt) * token;
                pool_values[k_at..k_at + token]
                    .copy_from_slice(&full_k[s][p * token..(p + 1) * token]);
                pool_values[v_at..v_at + token]
                    .copy_from_slice(&full_v[s][p * token..(p + 1) * token]);
            }
        }
        store(&pool.view(), &pool_values).expect("fill pool");

        let total_q: usize = q_lens.iter().sum();
        let mut q_all = Vec::new();
        let (mut k_new, mut v_new) = (Vec::new(), Vec::new());
        for s in 0..3 {
            q_all.extend_from_slice(&qs[s]);
            let first_new = (kv_lens[s] - q_lens[s]) * token;
            k_new.extend_from_slice(&full_k[s][first_new..]);
            v_new.extend_from_slice(&full_v[s][first_new..]);
        }
        let q = tensor(&mem, &[total_q, hq, d], dtype, &q_all);
        let kn = tensor(&mem, &[total_q, hkv, d], dtype, &k_new);
        let vn = tensor(&mem, &[total_q, hkv, d], dtype, &v_new);
        let out = Tensor::empty(&mem, &[total_q, hq, d], dtype).expect("out");
        let block_table = i32_tensor_2d(&mem, 3, &table);
        let q_indptr = i32_tensor(&mem, &[0, 5, 6, 9]);
        let kv_lens_t = i32_tensor(&mem, &[20, 9, 3]);
        attn.execute_paged(&mut PagedAttentionContext {
            cfg: paged_cfg,
            q: q.view(),
            k_new: kn.view(),
            v_new: vn.view(),
            out: out.view(),
            kv_layer: pool.view(),
            block_table: block_table.view(),
            q_indptr: q_indptr.view(),
            kv_lens: kv_lens_t.view(),
            max_q_len: 5,
            max_kv_len: 20,
            max_blocks_per_seq: 2,
            scale: 1.0 / (d as f32).sqrt(),
        })
        .expect("paged attention");
        let paged_out = load(&out.view()).expect("load");

        // Per sequence, the contiguous op over the same history gives the same rows.
        let contiguous_cfg = AttentionConfig {
            kind: AttentionKind::Prefill,
            block_tokens: None,
            ..paged_cfg
        };
        let mut row = 0;
        for s in 0..3 {
            let k = tensor(&mem, &[kv_lens[s], hkv, d], dtype, &full_k[s]);
            let v = tensor(&mem, &[kv_lens[s], hkv, d], dtype, &full_v[s]);
            let qv = tensor(&mem, &[q_lens[s], hq, d], dtype, &qs[s]);
            let o = Tensor::empty(&mem, &[q_lens[s], hq, d], dtype).expect("out");
            attn.execute(&mut AttentionContext {
                cfg: contiguous_cfg,
                q: qv.view(),
                k_cache: k.view(),
                v_cache: v.view(),
                out: o.view(),
                q_start: (kv_lens[s] - q_lens[s]) as u32,
                scale: 1.0 / (d as f32).sqrt(),
            })
            .expect("contiguous attention");
            let want = load(&o.view()).expect("load");
            let n = q_lens[s] * hq * d;
            assert_eq!(&paged_out[row..row + n], want.as_slice(), "sequence {s}");
            row += n;
        }
        // The new K/V rows were appended into their page slots: sequence 0, token 17 → block 2,
        // slot 1; sequence 2, token 0 → block 0, slot 0.
        let pool_after = load(&pool.view()).expect("pool");
        let k_slot = |block: usize, slot: usize| {
            let at = (block * 2 * bt + slot) * token;
            pool_after[at..at + token].to_vec()
        };
        let v_slot = |block: usize, slot: usize| {
            let at = ((block * 2 + 1) * bt + slot) * token;
            pool_after[at..at + token].to_vec()
        };
        let rounded = |v: &[f32]| v.iter().map(|&x| round_to(dtype, x)).collect::<Vec<_>>();
        assert_eq!(k_slot(2, 1), rounded(&full_k[0][17 * token..18 * token]));
        assert_eq!(v_slot(2, 1), rounded(&full_v[0][17 * token..18 * token]));
        assert_eq!(k_slot(0, 0), rounded(&full_k[2][..token]));

        // moe_route: logits [1, 1, 0, 0] top-2 → experts 0 then 1 with unrenormalised softmax
        // weights; [0, 0, 0, 2] → expert 3, then of the tied rest the one torch.topk keeps (2,
        // not the lowest id 0).
        let route_cfg = MoeRouteConfig {
            num_experts: 4,
            top_k: 2,
            renormalize: false,
            bf16_logits: false,
        };
        let moe = cpu.moe().expect("moe family");
        assert!(moe.supports_route(&route_cfg));
        assert_eq!(moe.implementation_route(&route_cfg), "cpu_moe_route");
        let logits = tensor(
            &mem,
            &[2, 4],
            DType::F32,
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        );
        let topk_ids = Tensor::empty(&mem, &[2, 2], DType::I32).expect("ids");
        let topk_weights = Tensor::empty(&mem, &[2, 2], DType::F32).expect("weights");
        let sorted_rows = Tensor::empty(&mem, &[4], DType::I32).expect("rows");
        let expert_offsets = Tensor::empty(&mem, &[5], DType::I32).expect("offsets");
        moe.route(&mut MoeRouteContext {
            cfg: route_cfg,
            router_logits: logits.view(),
            topk_ids: topk_ids.view(),
            topk_weights: topk_weights.view(),
            sorted_rows: sorted_rows.view(),
            expert_offsets: expert_offsets.view(),
        })
        .expect("route");
        assert_eq!(load_i32(&topk_ids.view()).expect("ids"), [0, 1, 3, 2]);
        let e = std::f32::consts::E;
        let (e2, sum1) = (e * e, 2.0 * e + 2.0);
        assert_close(
            &load(&topk_weights.view()).expect("weights"),
            &[e / sum1, e / sum1, e2 / (e2 + 3.0), 1.0 / (e2 + 3.0)],
            1e-6,
        );
        // Rows token·top_k + slot grouped by expert: e0 {0}, e1 {1}, e2 {3}, e3 {2}.
        assert_eq!(load_i32(&sorted_rows.view()).expect("rows"), [0, 1, 3, 2]);
        assert_eq!(
            load_i32(&expert_offsets.view()).expect("offsets"),
            [0, 1, 2, 3, 4]
        );

        // BF16 router logits: [1.001, 1, 1, 1.002] top-1 is expert 3 in F32, but all four
        // round to 1.0 in BF16 and torch.topk keeps expert 2 of the four-way tie (weight 1/4).
        let logits = tensor(&mem, &[1, 4], DType::F32, &[1.001, 1.0, 1.0, 1.002]);
        let ids = Tensor::empty(&mem, &[1, 1], DType::I32).expect("ids");
        let weights = Tensor::empty(&mem, &[1, 1], DType::F32).expect("weights");
        let rows = Tensor::empty(&mem, &[1], DType::I32).expect("rows");
        for (bf16_logits, want_id) in [(false, 3), (true, 2)] {
            let cfg = MoeRouteConfig {
                num_experts: 4,
                top_k: 1,
                renormalize: false,
                bf16_logits,
            };
            moe.route(&mut MoeRouteContext {
                cfg,
                router_logits: logits.view(),
                topk_ids: ids.view(),
                topk_weights: weights.view(),
                sorted_rows: rows.view(),
                expert_offsets: expert_offsets.view(),
            })
            .expect("route");
            assert_eq!(load_i32(&ids.view()).expect("ids"), [want_id], "{cfg}");
            if bf16_logits {
                assert_eq!(load(&weights.view()).expect("weights"), [0.25]);
            }
        }

        // copy_blocks duplicates block 3 into block 7 on every layer and nothing else.
        let (layers, blocks, block_bytes) = (3usize, 8usize, 24usize);
        let buf = DeviceBuffer::alloc(&mem, layers * blocks * block_bytes).expect("pool");
        let before: Vec<u8> = (0..buf.len()).map(|i| (i % 251) as u8).collect();
        buf.whole().write_bytes(&before).expect("fill");
        let copy_cfg = KvCopyConfig {
            num_layers: layers as u32,
            block_bytes: block_bytes as u64,
        };
        let kv_copy = cpu.kv_copy().expect("kv_copy family");
        assert!(kv_copy.supports(&copy_cfg));
        assert_eq!(kv_copy.implementation(&copy_cfg), "cpu_copy_blocks");
        kv_copy
            .execute(&mut KvCopyContext {
                pool: buf.whole(),
                layer_stride_bytes: (blocks * block_bytes) as u64,
                block_bytes: block_bytes as u64,
                num_layers: layers as u32,
                pairs: &[(BlockId(3), BlockId(7))],
            })
            .expect("copy_blocks");
        let after = buf.whole().read_bytes().expect("read");
        let mut want = before.clone();
        for l in 0..layers {
            let src = (l * blocks + 3) * block_bytes;
            let dst = (l * blocks + 7) * block_bytes;
            want.copy_within(src..src + block_bytes, dst);
        }
        assert_eq!(after, want);
        assert_ne!(after, before);
    }

    /// A layer stride whose extent overflows `usize` is refused, not a panic or a wrapped bound
    /// (Scout 93a929a9 / 8df17681).
    #[test]
    fn copy_blocks_refuses_an_overflowing_layer_stride() {
        let mem = HostMemory::new(DeviceId(0), 1 << 16) as Arc<dyn DeviceMemory>;
        let cpu = cpu_reference_provider();
        let buf = DeviceBuffer::alloc(&mem, 4096).expect("pool");
        let err = cpu
            .kv_copy()
            .expect("kv_copy family")
            .execute(&mut KvCopyContext {
                pool: buf.whole(),
                layer_stride_bytes: u64::MAX / 2,
                block_bytes: 16,
                num_layers: 3,
                pairs: &[(BlockId(0), BlockId(1))],
            })
            .expect_err("an overflowing extent is refused");
        assert!(err.to_string().contains("exceed"), "{err}");
    }
}
