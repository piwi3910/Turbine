//! Paged-KV reference attention: ragged-batch attention over one layer of the block pool
//! (`attention_prefill_paged`, `attention_decode_paged`); `AttentionKernel::execute_paged`
//! (attention.rs) runs it. The block fork (`copy_blocks`) is in kv_copy.rs.
//!
//! The pool is read and written block by block (never whole), so the cost of a call is
//! proportional to the blocks its sequences own, not to the pool size.
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use turbine_core::types::DType;
use turbine_tensor::tensor::contiguous_strides;

use super::quant::{fp8_e4m3_round, fp8_e4m3_value};
use super::tq_attention::{self, Formulation, MixedPagedLayer, TQ_DIM};
use super::{
    FloatCodec, expect_rank, expect_shape, invalid, is_float, load, load_i32, math, store,
};
use crate::KernelError;
use crate::ops::{KV_FMT_BF16, KV_FMT_FP8_E4M3, PagedAttentionContext, TqParams, kv_format_code};

/// How the pool pages hold K and V: a float type, or FP8 e4m3 with per-half scales (Phase 6a
/// S-13: `e4m3(x / scale)` written, `e4m3 · scale` read in F32 and rounded to the activation
/// dtype, so attention runs on BF16 K/V exactly as over BF16 pages holding those values).
#[derive(Clone, Copy)]
enum Pages {
    Float(FloatCodec),
    Fp8 { k_scale: f32, v_scale: f32 },
}

impl Pages {
    fn of(ctx: &PagedAttentionContext<'_>) -> Result<Pages, KernelError> {
        let pool = ctx.kv_layer.dtype;
        if ctx.cfg.dtype != DType::F8E4M3 {
            if pool != ctx.cfg.dtype {
                return Err(invalid(format!(
                    "kv_layer is {} but the config names {}",
                    pool.as_str(),
                    ctx.cfg.dtype.as_str()
                )));
            }
            return Ok(Pages::Float(FloatCodec::require(pool)?));
        }
        if pool != DType::F8E4M3 {
            return Err(invalid(format!(
                "FP8 paged attention needs an f8e4m3 kv_layer, got {}",
                pool.as_str()
            )));
        }
        for (name, s) in [("k_scale", ctx.k_scale), ("v_scale", ctx.v_scale)] {
            if !(s.is_finite() && s > 0.0) {
                return Err(invalid(format!(
                    "{name} must be finite and positive, got {s}"
                )));
            }
        }
        Ok(Pages::Fp8 {
            k_scale: ctx.k_scale,
            v_scale: ctx.v_scale,
        })
    }

    /// Writes `v` of half `half` (0 = K, 1 = V) into `out`.
    fn encode(self, half: usize, v: f32, out: &mut [u8]) {
        match self {
            Pages::Float(codec) => codec.encode(v, out),
            Pages::Fp8 { k_scale, v_scale } => {
                let scale = if half == 0 { k_scale } else { v_scale };
                out[0] = fp8_e4m3_round(v / scale);
            }
        }
    }

    /// Reads an element of half `half` from `b`.
    fn decode(self, half: usize, b: &[u8]) -> f32 {
        match self {
            Pages::Float(codec) => codec.decode(b),
            Pages::Fp8 { k_scale, v_scale } => {
                let scale = if half == 0 { k_scale } else { v_scale };
                fp8_e4m3_value(b[0]) * scale
            }
        }
    }
}

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
    let base = kv_format_code(ctx.cfg.dtype);
    let formats = load_formats(ctx)?;
    if ctx.cfg.dtype.tq_record_bytes().is_some()
        || formats.iter().flatten().any(|f| Some(*f) != base)
    {
        return mixed(ctx, formats);
    }
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
    let pages = Pages::of(ctx)?;
    // The activation dtype: the config's, or Q's under FP8 pages.
    let act = if ctx.cfg.dtype == DType::F8E4M3 {
        ctx.q.dtype
    } else {
        ctx.cfg.dtype
    };
    if !is_float(act) {
        return Err(invalid(format!(
            "q must be bf16/f16/f32, is {}",
            act.as_str()
        )));
    }
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
                    pages.encode(half, v, &mut bytes[(base + j) * es..]);
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
            // A block the append touched is read from its patched copy, without cloning it.
            let bytes: std::borrow::Cow<'_, [u8]> = match touched.get(&block) {
                Some(bytes) => bytes.as_slice().into(),
                None => block_slice(block).read_bytes()?.into(),
            };
            for (half, dst) in [(0, &mut k), (1, &mut v)] {
                let base = half * block_tokens * token_elems;
                dst.extend(
                    (base..base + tokens * token_elems)
                        .map(|e| super::round_to(act, pages.decode(half, &bytes[e * es..]))),
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
        let o = math::attention(&q[rows.clone()], &k, &v, &shape, ctx.scale, |p| {
            super::round_to(act, p)
        });
        out[rows].copy_from_slice(&o);
    }
    store(&ctx.out, &out)
}

/// The block formats of `ctx` (`None`: every block in `cfg.dtype`): a dense U8
/// `[num_seqs, max_blocks_per_seq]` view.
fn load_formats(ctx: &PagedAttentionContext<'_>) -> Result<Option<Vec<u8>>, KernelError> {
    let Some(v) = &ctx.block_formats else {
        return Ok(None);
    };
    let num_seqs = ctx.kv_lens.shape.first().copied().unwrap_or(0);
    let want = [num_seqs, ctx.max_blocks_per_seq as usize];
    if v.dtype != DType::U8 || v.shape.as_slice() != want {
        return Err(invalid(format!(
            "block_formats must be u8 {want:?}, is {} {:?}",
            v.dtype.as_str(),
            v.shape.as_slice()
        )));
    }
    let offsets = super::element_offsets(v)?;
    let bytes = v.slice.read_bytes()?;
    Ok(Some(offsets.iter().map(|&o| bytes[o]).collect()))
}

/// Mixed-format pages (P6b S-5): TurboQuant L0 pages, or a block table whose format codes
/// differ from `cfg.dtype`'s. Each new row is encoded into its block by the block's format
/// (TurboQuant records through the caller's codec, `ctx.tq`), then attention reads every block
/// by its format through [`tq_attention`] (the rotated-domain formulation), rows of a prefill
/// chunk included. Block `b`'s bytes are the first page bytes of its format in slot `b` of
/// `kv_layer` (a dense view whose rows are the slots).
fn mixed(ctx: &PagedAttentionContext<'_>, formats: Option<Vec<u8>>) -> Result<(), KernelError> {
    let hq = ctx.cfg.num_q_heads as usize;
    let hkv = ctx.cfg.num_kv_heads as usize;
    let d = ctx.cfg.head_dim as usize;
    let bt = ctx.cfg.block_tokens.unwrap_or(0) as usize;
    let Some(base) = kv_format_code(ctx.cfg.dtype) else {
        return Err(invalid(format!(
            "{} is not a KV page dtype",
            ctx.cfg.dtype.as_str()
        )));
    };
    expect_rank("q", &ctx.q, 3)?;
    let total_q = ctx.q.shape[0];
    expect_shape("q", &ctx.q, &[total_q, hq, d])?;
    expect_shape("out", &ctx.out, &[total_q, hq, d])?;
    expect_shape("k_new", &ctx.k_new, &[total_q, hkv, d])?;
    expect_shape("v_new", &ctx.v_new, &[total_q, hkv, d])?;
    let act = ctx.q.dtype;
    if !is_float(act) {
        return Err(invalid(format!(
            "q must be bf16/f16/f32, is {}",
            act.as_str()
        )));
    }
    let pool = &ctx.kv_layer;
    let Some(&num_blocks) = pool.shape.first() else {
        return Err(invalid("kv_layer must have a block dimension".into()));
    };
    if pool.strides != contiguous_strides(&pool.shape) {
        return Err(invalid(format!(
            "kv_layer must be dense, has strides {:?}",
            pool.strides.as_slice()
        )));
    }
    let slot = pool.shape[1..].iter().product::<usize>() * pool.dtype.size_bytes();
    if pool.slice.len() < num_blocks * slot {
        return Err(invalid(format!(
            "kv_layer slice holds {} bytes, {num_blocks} slots of {slot} need more",
            pool.slice.len()
        )));
    }
    let seqs = sequences(ctx, total_q, num_blocks, bt)?;
    let max_blocks = ctx.max_blocks_per_seq as usize;
    let entries = seqs.len() * max_blocks;
    let formats: Vec<u8> = formats.unwrap_or_else(|| vec![base; entries]);
    let page_len = |fmt: u8| -> Result<usize, KernelError> {
        match tq_attention::page_bytes_of(fmt, bt, hkv, d) {
            Some(n) if n <= slot => Ok(n),
            Some(n) => Err(invalid(format!(
                "a format {fmt} page of {n} bytes does not fit a kv_layer slot of {slot}"
            ))),
            None => Err(invalid(format!("unknown KV block format {fmt}"))),
        }
    };
    // Every block the batch reads, with its one format.
    let mut block_fmt: BTreeMap<usize, u8> = BTreeMap::new();
    for (s, seq) in seqs.iter().enumerate() {
        for (n, &b) in seq.blocks.iter().enumerate() {
            let fmt = formats[s * max_blocks + n];
            page_len(fmt)?;
            if *block_fmt.entry(b).or_insert(fmt) != fmt {
                return Err(invalid(format!("block {b} is tagged with two formats")));
            }
        }
    }
    let empty = TqParams {
        heads: Vec::new(),
        codebooks: Default::default(),
    };
    let tq = ctx.tq.as_ref();
    if block_fmt.values().any(|f| tq_attention::is_turboquant(*f)) {
        let Some(t) = tq else {
            return Err(invalid(
                "TurboQuant blocks need the layer's TurboQuant tables and codec".into(),
            ));
        };
        if d != TQ_DIM || t.params.heads.len() != hkv {
            return Err(invalid(format!(
                "TurboQuant blocks need head_dim {TQ_DIM} and tables for {hkv} KV heads (head_dim {d}, {} tables)",
                t.params.heads.len()
            )));
        }
    }
    let (k_scale, v_scale) = (ctx.k_scale, ctx.v_scale);
    if block_fmt.values().any(|f| *f == KV_FMT_FP8_E4M3)
        && !(k_scale.is_finite() && k_scale > 0.0 && v_scale.is_finite() && v_scale > 0.0)
    {
        return Err(invalid(format!(
            "FP8 blocks need finite positive scales, got {k_scale} / {v_scale}"
        )));
    }

    let q = load(&ctx.q)?;
    let k_new = load(&ctx.k_new)?;
    let v_new = load(&ctx.v_new)?;
    let slot_slice = |b: usize| pool.slice.sub(b * slot, slot);
    let mut bytes: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for &b in block_fmt.keys() {
        bytes.insert(b, slot_slice(b).read_bytes()?);
    }
    // Append: each new row, per KV head, encoded by its block's format.
    let mut touched = std::collections::BTreeSet::new();
    let token = hkv * d;
    for seq in &seqs {
        let first_new = seq.kv_len - seq.q_len;
        for i in 0..seq.q_len {
            let pos = first_new + i;
            let b = seq.blocks[pos / bt];
            let t = pos % bt;
            let fmt = block_fmt[&b];
            let page = bytes.get_mut(&b).expect("read above");
            touched.insert(b);
            let src = (seq.row + i) * token;
            for g in 0..hkv {
                let (k, v) = (
                    &k_new[src + g * d..src + (g + 1) * d],
                    &v_new[src + g * d..src + (g + 1) * d],
                );
                match (fmt, tq) {
                    (KV_FMT_BF16 | KV_FMT_FP8_E4M3, _) => {
                        for (half, x) in [(0, k), (1, v)] {
                            let base = ((half * bt + t) * hkv + g) * d;
                            for (j, &val) in x.iter().enumerate() {
                                if fmt == KV_FMT_BF16 {
                                    FloatCodec::Bf16.encode(val, &mut page[(base + j) * 2..]);
                                } else {
                                    let scale = if half == 0 { k_scale } else { v_scale };
                                    page[base + j] = fp8_e4m3_round(val / scale);
                                }
                            }
                        }
                    }
                    (_, Some(tq)) => {
                        let rec = page_len(fmt)? / (hkv * bt);
                        let at = (g * bt + t) * rec;
                        (tq.encode)(fmt, k, v, &tq.params.heads[g], &mut page[at..at + rec]);
                    }
                    (_, None) => unreachable!("checked above: TurboQuant blocks have tables"),
                }
            }
        }
    }
    for b in &touched {
        slot_slice(*b).write_bytes(&bytes[b])?;
    }

    // Attend over every block by its format.
    let mut pages: Vec<&[u8]> = vec![&[]; num_blocks];
    for (&b, page) in &bytes {
        pages[b] = &page[..page_len(block_fmt[&b])?];
    }
    let mut q_indptr = vec![0usize];
    q_indptr.extend(seqs.iter().map(|s| s.row + s.q_len));
    let kv_lens: Vec<usize> = seqs.iter().map(|s| s.kv_len).collect();
    let table = load_i32(&ctx.block_table)?;
    let block_table: Vec<u32> = table.iter().map(|&b| b.max(0) as u32).collect();
    let layer = MixedPagedLayer {
        num_q_heads: hq,
        num_kv_heads: hkv,
        head_dim: d,
        block_tokens: bt,
        pages: &pages,
        block_table: &block_table,
        block_formats: &formats,
        max_blocks_per_seq: max_blocks,
        q_indptr: &q_indptr,
        kv_lens: &kv_lens,
        k_scale,
        v_scale,
        scale: ctx.scale,
        causal: ctx.cfg.causal,
    };
    let params = tq.map_or(&empty, |t| t.params);
    let out = tq_attention::prefill(&q, &layer, params, Formulation::Rotated)?;
    store(&ctx.out, &out)
}

#[cfg(test)]
mod tests {
    use crate::cpu::quant::{fp8_e4m3_round, fp8_e4m3_value};
    use crate::cpu::test_util::*;
    use crate::cpu::*;

    /// One 20-token prefill of one sequence into blocks [3, 1] of a 4-block pool of `pool_dtype`
    /// pages (`cfg_dtype` in the config), K and V from `k` / `v`: the output and the pool bytes.
    fn prefill(
        cfg_dtype: DType,
        pool_dtype: DType,
        k: &[f32],
        v: &[f32],
        scales: (f32, f32),
    ) -> Result<(Vec<f32>, Vec<u8>), KernelError> {
        let mem = HostMemory::new(DeviceId(0), 1 << 22) as Arc<dyn DeviceMemory>;
        let (hq, hkv, d, bt, t) = (4usize, 2usize, 8usize, 16usize, 20usize);
        let cfg = AttentionConfig {
            kind: AttentionKind::PrefillPaged,
            num_q_heads: hq as u32,
            num_kv_heads: hkv as u32,
            head_dim: d as u32,
            dtype: cfg_dtype,
            block_tokens: Some(bt as u32),
            causal: true,
        };
        let pool = Tensor::empty(&mem, &[4, 2, bt, hkv, d], pool_dtype).expect("pool");
        let q = tensor(&mem, &[t, hq, d], DType::BF16, &seeded(41, t * hq * d));
        let kn = tensor(&mem, &[t, hkv, d], DType::BF16, k);
        let vn = tensor(&mem, &[t, hkv, d], DType::BF16, v);
        let out = Tensor::empty(&mem, &[t, hq, d], DType::BF16).expect("out");
        let table = i32_tensor_2d(&mem, 1, &[3, 1]);
        let q_indptr = i32_tensor(&mem, &[0, t as i32]);
        let kv_lens = i32_tensor(&mem, &[t as i32]);
        let provider = cpu_reference_provider();
        let attn = provider.attention().expect("attention family");
        attn.execute_paged(&mut PagedAttentionContext {
            cfg,
            q: q.view(),
            k_new: kn.view(),
            v_new: vn.view(),
            out: out.view(),
            kv_layer: pool.view(),
            block_table: table.view(),
            q_indptr: q_indptr.view(),
            kv_lens: kv_lens.view(),
            max_q_len: t as u32,
            max_kv_len: t as u32,
            max_blocks_per_seq: 2,
            scale: 1.0 / (d as f32).sqrt(),
            k_scale: scales.0,
            v_scale: scales.1,
            block_formats: None,
            tq: None,
        })?;
        Ok((
            load(&out.view()).expect("out"),
            pool.view().slice.read_bytes().expect("pool"),
        ))
    }

    /// FP8 pages (Phase 6a S-13): the append writes `e4m3(x / scale)` bytes and attention reads
    /// `e4m3 · scale`, so the output equals BF16-page attention over K and V quantize-dequantized
    /// with the same scales (power-of-two scales keep the dequantized values exact in BF16).
    /// Breaks if K and V swap scales, a scale is applied twice or not at all, or a page byte is
    /// not the e4m3 of the scaled value.
    #[test]
    fn fp8_pages_hold_scaled_e4m3_and_attend_like_dequantized_kv() {
        let (t, hkv, d) = (20usize, 2usize, 8usize);
        let bf = |x: &[f32]| {
            x.iter()
                .map(|&v| round_to(DType::BF16, v))
                .collect::<Vec<_>>()
        };
        // Magnitudes up to ~12: e4m3 rounding is visible at these scales, nothing saturates.
        let k = bf(&seeded(42, t * hkv * d)
            .iter()
            .map(|x| x * 3.0)
            .collect::<Vec<_>>());
        let v = bf(&seeded(43, t * hkv * d)
            .iter()
            .map(|x| x * 4.0)
            .collect::<Vec<_>>());
        let (ks, vs) = (0.0625f32, 0.125f32);
        let qdq = |x: &[f32], s: f32| {
            x.iter()
                .map(|&e| round_to(DType::BF16, fp8_e4m3_value(fp8_e4m3_round(e / s)) * s))
                .collect::<Vec<_>>()
        };
        let (kq, vq) = (qdq(&k, ks), qdq(&v, vs));
        assert_ne!(kq, k, "the scales must make e4m3 rounding visible");

        let (fp8_out, pages) =
            prefill(DType::F8E4M3, DType::F8E4M3, &k, &v, (ks, vs)).expect("fp8");
        let (want, _) = prefill(DType::BF16, DType::BF16, &kq, &vq, (1.0, 1.0)).expect("bf16");
        assert_eq!(fp8_out, want);
        assert_eq!(
            AttentionKernel::implementation(
                &CpuReference,
                &AttentionConfig {
                    kind: AttentionKind::DecodePaged,
                    num_q_heads: 4,
                    num_kv_heads: 2,
                    head_dim: 8,
                    dtype: DType::F8E4M3,
                    block_tokens: Some(16),
                    causal: true,
                }
            ),
            "cpu_attention_paged_fp8kv_f32acc"
        );

        // Token 0 is block 3 slot 0; token 17 is block 1 slot 1. One byte per element.
        let token = hkv * d;
        let block = 2 * 16 * token;
        let page = |b: usize, half: usize, slot: usize| {
            let at = b * block + (half * 16 + slot) * token;
            pages[at..at + token].to_vec()
        };
        let enc = |x: &[f32], s: f32| x.iter().map(|&e| fp8_e4m3_round(e / s)).collect::<Vec<_>>();
        assert_eq!(page(3, 0, 0), enc(&k[..token], ks));
        assert_eq!(page(3, 1, 0), enc(&v[..token], vs));
        assert_eq!(page(1, 0, 1), enc(&k[17 * token..18 * token], ks));
        assert_eq!(page(1, 1, 1), enc(&v[17 * token..18 * token], vs));

        // Power-of-two scales only move the e4m3 range; others change every rounding, so K and V
        // with swapped scales give another output.
        let (odd_k, odd_v) = (0.07f32, 0.11f32);
        let (odd, _) = prefill(DType::F8E4M3, DType::F8E4M3, &k, &v, (odd_k, odd_v)).expect("fp8");
        let (swapped, _) =
            prefill(DType::F8E4M3, DType::F8E4M3, &k, &v, (odd_v, odd_k)).expect("fp8");
        assert_ne!(swapped, odd, "K and V scales are not interchangeable");
        // Any scale: the dequantized element is rounded to BF16, as BF16 pages would hold it.
        let (want_odd, _) = prefill(
            DType::BF16,
            DType::BF16,
            &qdq(&k, odd_k),
            &qdq(&v, odd_v),
            (1.0, 1.0),
        )
        .expect("bf16");
        assert_eq!(odd, want_odd);
        for bad in [0.0, f32::NAN, -1.0] {
            let err = prefill(DType::F8E4M3, DType::F8E4M3, &k, &v, (bad, vs)).unwrap_err();
            assert!(err.to_string().contains("k_scale"), "{err}");
        }
        let err = prefill(DType::F8E4M3, DType::BF16, &k, &v, (ks, vs)).unwrap_err();
        assert!(err.to_string().contains("f8e4m3 kv_layer"), "{err}");
    }

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
            k_scale: 1.0,
            v_scale: 1.0,
            block_formats: None,
            tq: None,
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

    /// P6b S-5: a prefill over a block table mixing BF16, TurboQuant `tq4` and FP8 blocks
    /// appends every new row by its block's tag (BF16 bytes, the caller codec's TurboQuant
    /// record at `(head · block_tokens + token) · record_bytes`, e4m3 of the scaled value) and
    /// attends over each block by its tag: the output matches the decode-then-attend reference
    /// over independently built pages. Breaks if the append ignores the tag, misplaces a
    /// record, or a block is read in another format.
    #[test]
    fn mixed_formats_append_and_read_by_tag() {
        use turbine_kv::codec::turboquant::codebook::codebook;
        use turbine_kv::codec::turboquant::hadamard::{SignKind, rademacher};
        use turbine_kv::codec::turboquant::{Tq4Codec, encode_record};

        use crate::cpu::tq_attention::{self, Formulation, MixedPagedLayer};

        const SEED: u64 = 0x0123_4567_89ab_cdef;
        fn encode(fmt: u8, k: &[f32], v: &[f32], h: &TqHeadTables, record: &mut [u8]) {
            assert_eq!(fmt, KV_FMT_TQ4);
            encode_record(Tq4Codec::WIDTHS, k, v, &h.k_signs, &h.v_signs, record);
        }
        let mem = HostMemory::new(DeviceId(0), 1 << 24) as Arc<dyn DeviceMemory>;
        let (hq, hkv, d, bt, t) = (4usize, 2usize, 128usize, 16usize, 40usize);
        let (k_scale, v_scale) = (0.02f32, 0.015f32);
        let params = TqParams {
            heads: (0..hkv as u32)
                .map(|h| TqHeadTables {
                    k_signs: rademacher(SEED, 0, h, SignKind::K, d),
                    v_signs: rademacher(SEED, 0, h, SignKind::V, d),
                })
                .collect(),
            codebooks: [codebook(1), codebook(2), codebook(3), codebook(4)].map(<[f32]>::to_vec),
        };
        let cfg = AttentionConfig {
            kind: AttentionKind::PrefillPaged,
            num_q_heads: hq as u32,
            num_kv_heads: hkv as u32,
            head_dim: d as u32,
            dtype: DType::BF16,
            block_tokens: Some(bt as u32),
            causal: true,
        };
        // Slots of BF16 pages (the largest format); positions 0..16 in block 2 (tq4),
        // 16..32 in block 0 (bf16), 32..40 in block 1 (fp8).
        let pool = Tensor::empty(&mem, &[3, 2, bt, hkv, d], DType::BF16).expect("pool");
        let table = [2u32, 0, 1];
        let formats = [KV_FMT_TQ4, KV_FMT_BF16, KV_FMT_FP8_E4M3];
        let bf = |x: &[f32]| -> Vec<f32> {
            x.iter()
                .map(|v| half::bf16::from_f32(*v).to_f32())
                .collect()
        };
        let qv = bf(&seeded(51, t * hq * d));
        let kv = bf(&seeded(52, t * hkv * d));
        let vv = bf(&seeded(53, t * hkv * d));
        let q = tensor(&mem, &[t, hq, d], DType::BF16, &qv);
        let kn = tensor(&mem, &[t, hkv, d], DType::BF16, &kv);
        let vn = tensor(&mem, &[t, hkv, d], DType::BF16, &vv);
        let out = Tensor::empty(&mem, &[t, hq, d], DType::F32).expect("out");
        let table_t = i32_tensor_2d(&mem, 1, &table.map(|b| b as i32));
        let q_indptr = i32_tensor(&mem, &[0, t as i32]);
        let kv_lens = i32_tensor(&mem, &[t as i32]);
        let formats_t = Tensor::empty(&mem, &[1, 3], DType::U8).expect("formats");
        formats_t
            .view()
            .slice
            .write_bytes(&formats)
            .expect("formats");
        cpu_reference_provider()
            .attention()
            .expect("attention family")
            .execute_paged(&mut PagedAttentionContext {
                cfg,
                q: q.view(),
                k_new: kn.view(),
                v_new: vn.view(),
                out: out.view(),
                kv_layer: pool.view(),
                block_table: table_t.view(),
                q_indptr: q_indptr.view(),
                kv_lens: kv_lens.view(),
                max_q_len: t as u32,
                max_kv_len: t as u32,
                max_blocks_per_seq: 3,
                scale: 1.0 / (d as f32).sqrt(),
                k_scale,
                v_scale,
                block_formats: Some(formats_t.view()),
                tq: Some(TqPaged {
                    params: &params,
                    encode,
                    seed: 0,
                    device: None,
                }),
            })
            .expect("mixed paged attention");
        let got = load(&out.view()).expect("out");
        let bytes = pool.view().slice.read_bytes().expect("pool");
        let slot = 2 * bt * hkv * d * 2;

        // The pages each format's append must have written, built here from the rows.
        let row = |x: &[f32], pos: usize, g: usize| x[(pos * hkv + g) * d..][..d].to_vec();
        let rec = Tq4Codec::WIDTHS.record_bytes();
        let mut want: Vec<Vec<u8>> = Vec::new();
        for (n, &fmt) in formats.iter().enumerate() {
            let len = tq_attention::page_bytes_of(fmt, bt, hkv, d).expect("format");
            let mut page = vec![0u8; len];
            for tok in 0..bt.min(t - n * bt) {
                let pos = n * bt + tok;
                for g in 0..hkv {
                    let (k, v) = (row(&kv, pos, g), row(&vv, pos, g));
                    match fmt {
                        KV_FMT_TQ4 => {
                            let at = (g * bt + tok) * rec;
                            encode(fmt, &k, &v, &params.heads[g], &mut page[at..at + rec]);
                        }
                        _ => {
                            for (half, x) in [(0, &k), (1, &v)] {
                                let base = ((half * bt + tok) * hkv + g) * d;
                                for (j, val) in x.iter().enumerate() {
                                    if fmt == KV_FMT_BF16 {
                                        page[(base + j) * 2..(base + j) * 2 + 2].copy_from_slice(
                                            &half::bf16::from_f32(*val).to_le_bytes(),
                                        );
                                    } else {
                                        let s = if half == 0 { k_scale } else { v_scale };
                                        page[base + j] = fp8_e4m3_round(val / s);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let b = table[n] as usize;
            let written = &bytes[b * slot..b * slot + len];
            if fmt == KV_FMT_FP8_E4M3 {
                // Only the first 8 tokens exist: compare their bytes in both halves.
                for half in 0..2 {
                    let at = half * bt * hkv * d;
                    let n = 8 * hkv * d;
                    assert_eq!(&written[at..at + n], &page[at..at + n], "fp8 half {half}");
                }
            } else {
                assert_eq!(written, &page[..], "format {fmt} page");
            }
            want.push(page);
        }
        let mut pages: Vec<&[u8]> = vec![&[]; 3];
        for (n, &b) in table.iter().enumerate() {
            pages[b as usize] = &want[n];
        }
        let layer = MixedPagedLayer {
            num_q_heads: hq,
            num_kv_heads: hkv,
            head_dim: d,
            block_tokens: bt,
            pages: &pages,
            block_table: &table,
            block_formats: &formats,
            max_blocks_per_seq: 3,
            q_indptr: &[0, t],
            kv_lens: &[t],
            k_scale,
            v_scale,
            scale: 1.0 / (d as f32).sqrt(),
            causal: true,
        };
        let reference = tq_attention::prefill(&qv, &layer, &params, Formulation::DecodeThenAttend)
            .expect("reference");
        assert_close(&got, &reference, 1e-4);
    }
}
