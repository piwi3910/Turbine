//! Reference TurboQuant attention over mixed-format block tables (P6b S-5), the oracle every
//! GPU mixed-format paged attention is tested against.
//!
//! One layer of a ragged batch, the CPU paged attention's shapes (`q` `[total_q, hq, d]`,
//! `q_indptr`, `kv_lens`, a `[num_seqs, max_blocks]` block table), plus one format byte per
//! block-table entry ([`FMT_BF16`], [`FMT_FP8_E4M3`], [`FMT_TQ4`], [`FMT_TQ2`]) and the
//! TurboQuant tables of the layer ([`TqParams`]). K/V are read from the pages only — the
//! paged append has written every row, a prefill chunk's own rows included — so prefill and
//! decode see the same KV.
//!
//! Page bytes of one layer of one block, by format:
//! - `bf16`: `[2, block_tokens, kv_heads, head_dim]` BF16, K before V (the L0 page layout);
//! - `fp8_e4m3`: the same elements as OCP e4m3 bytes, read as `bf16(e4m3 × scale)` with the
//!   layer's K / V scale (the 6a FP8 page contract, as the GPU kernels read them);
//! - `tq4` / `tq2`: `kv_heads × block_tokens` records of the `turbine-kv` TurboQuant codec
//!   (record `head·block_tokens + token`; K codes, K norm, QJL signs, residual norm, V codes,
//!   V norm, padded to 16 bytes; `tq4` 144 bytes, `tq2` 80).
//!
//! Two formulations, equal within 1e-5 (`rotated_equals_decoded`):
//! - [`Formulation::DecodeThenAttend`]: every TurboQuant record decoded to F32 exactly as the
//!   codec's `decode_record` does, then exact attention (F64 accumulation) over the values;
//! - [`Formulation::Rotated`]: per query row and KV head, `q` is rotated once
//!   (`Rq = H·(s_k ⊙ q)/√d`) and projected once more (`S·Rq`) for the QJL residual; a TurboQuant
//!   key scores `norm/√d · ⟨Rq, c[codes]⟩ + √(π/2)/d · ‖r‖ · ⟨S·Rq, z⟩`; V is accumulated as
//!   `norm/√d · c[codes]` in the rotated domain and rotated back once per output row
//!   (`s_v ⊙ H·acc/√d`), added to the plain-domain accumulation of BF16 / FP8 blocks.
//!
//! The TurboQuant math here is a copy of `turbine-kv`'s codec decode (this crate does not
//! depend on `turbine-kv`; the rotation signs, the QJL projection and the codebooks arrive as
//! data in [`TqParams`], which the caller builds from the codec, as the GPU receives them).
//! `matches_decoded_attention` holds the copy to the codec's `decode_record`.

use half::bf16;

use super::invalid;
use super::quant::fp8_e4m3_value;
use crate::KernelError;

pub use crate::ops::{
    KV_FMT_BF16 as FMT_BF16, KV_FMT_FP8_E4M3 as FMT_FP8_E4M3, KV_FMT_TQ2 as FMT_TQ2,
    KV_FMT_TQ4 as FMT_TQ4, TqHeadTables, TqParams,
};

/// The head dimension TurboQuant pages are defined for.
pub const TQ_DIM: usize = 128;

/// Which of the two equivalent computations to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formulation {
    /// Decode every TurboQuant record to F32, then exact attention.
    DecodeThenAttend,
    /// Score and accumulate TurboQuant blocks in the rotated domain.
    Rotated,
}

/// One layer of a ragged batch over mixed-format pages.
#[derive(Clone, Copy, Debug)]
pub struct MixedPagedLayer<'a> {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_tokens: usize,
    /// This layer's bytes of every pool block, indexed by block id, in the block's format.
    pub pages: &'a [&'a [u8]],
    /// `[num_seqs, max_blocks_per_seq]` block ids.
    pub block_table: &'a [u32],
    /// One format code per block-table entry.
    pub block_formats: &'a [u8],
    pub max_blocks_per_seq: usize,
    /// `num_seqs + 1` row offsets into `q` / the output.
    pub q_indptr: &'a [usize],
    /// KV length per sequence, the sequence's new rows included.
    pub kv_lens: &'a [usize],
    /// The layer's K and V scales of FP8 pages.
    pub k_scale: f32,
    pub v_scale: f32,
    /// Softmax scale of the scores.
    pub scale: f32,
    pub causal: bool,
}

/// The widths of a TurboQuant format: (K MSE bits, V bits).
fn tq_widths(fmt: u8) -> Option<(usize, usize)> {
    match fmt {
        FMT_TQ4 => Some((3, 4)),
        FMT_TQ2 => Some((1, 2)),
        _ => None,
    }
}

/// Byte offsets of a record's fields (the codec's layout).
#[derive(Clone, Copy)]
struct Record {
    k_bits: usize,
    v_bits: usize,
    k_norm: usize,
    qjl: usize,
    r_norm: usize,
    v_codes: usize,
    v_norm: usize,
    bytes: usize,
}

impl Record {
    fn of(k_bits: usize, v_bits: usize) -> Record {
        let k_norm = TQ_DIM * k_bits / 8;
        let qjl = k_norm + 2;
        let r_norm = qjl + TQ_DIM / 8;
        let v_codes = r_norm + 2;
        let v_norm = v_codes + TQ_DIM * v_bits / 8;
        Record {
            k_bits,
            v_bits,
            k_norm,
            qjl,
            r_norm,
            v_codes,
            v_norm,
            bytes: (v_norm + 2).div_ceil(16) * 16,
        }
    }
}

fn get_bf16(b: &[u8], at: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32()
}

/// Code `i` of `bits` bits, packed least-significant bit first.
fn code(packed: &[u8], bits: usize, i: usize) -> usize {
    (0..bits).fold(0, |c, j| {
        let bit = i * bits + j;
        c | (usize::from(packed[bit / 8] >> (bit % 8) & 1) << j)
    })
}

/// QJL sign `i` as ±1.
fn sign(bits: &[u8], i: usize) -> f32 {
    if bits[i / 8] >> (i % 8) & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

fn sqrt_d() -> f32 {
    (TQ_DIM as f64).sqrt() as f32
}

/// In-place unnormalised fast Walsh–Hadamard transform (Sylvester order).
fn fwht<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(x: &mut [T]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// `s ⊙ H·y/√d` in F32, the codec's inverse rotation.
fn unrotate_f32(y: &[f32], signs: &[f32]) -> Vec<f32> {
    let mut x = y.to_vec();
    fwht(&mut x);
    let k = (1.0 / (x.len() as f64).sqrt()) as f32;
    x.iter_mut().zip(signs).for_each(|(v, s)| *v *= k * s);
    x
}

/// K and V of one record in F32, the codec's `decode_record` step for step.
fn decode_record(r: &[u8], rec: Record, head: &TqHeadTables, cbs: &[Vec<f32>; 4]) -> [Vec<f32>; 2] {
    let (kcb, vcb) = (&cbs[rec.k_bits - 1], &cbs[rec.v_bits - 1]);
    // K: ŷ = c[code]·n/√d plus the QJL residual r̂ = √(π/2)/d · ‖r‖ · Sᵀ·z, then unrotated.
    let kk = get_bf16(r, rec.k_norm) / sqrt_d();
    let mut y: Vec<f32> = (0..TQ_DIM)
        .map(|i| kcb[code(&r[..rec.k_norm], rec.k_bits, i)] * kk)
        .collect();
    let mut res = vec![0f32; TQ_DIM];
    for (i, row) in head.qjl.chunks_exact(TQ_DIM).enumerate() {
        let z = sign(&r[rec.qjl..rec.r_norm], i);
        res.iter_mut().zip(row).for_each(|(o, a)| *o += z * a);
    }
    let rk = ((std::f64::consts::PI / 2.0).sqrt() / TQ_DIM as f64) as f32 * get_bf16(r, rec.r_norm);
    y.iter_mut().zip(&res).for_each(|(a, b)| *a += b * rk);
    let k = unrotate_f32(&y, &head.k_signs);
    // V: ŷ = c[code]·n/√d, unrotated.
    let vk = get_bf16(r, rec.v_norm) / sqrt_d();
    let yv: Vec<f32> = (0..TQ_DIM)
        .map(|i| vcb[code(&r[rec.v_codes..rec.v_norm], rec.v_bits, i)] * vk)
        .collect();
    [k, unrotate_f32(&yv, &head.v_signs)]
}

/// One sequence of the batch, validated.
struct Seq {
    row: usize,
    q_len: usize,
    kv_len: usize,
    /// (block id, format) per block, in token order.
    blocks: Vec<(usize, u8)>,
}

/// Bytes of one layer of one page of format `fmt` (`None`: an unknown format code).
pub fn page_bytes_of(
    fmt: u8,
    block_tokens: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<usize> {
    let elems = 2 * block_tokens * kv_heads * head_dim;
    match fmt {
        FMT_BF16 => Some(2 * elems),
        FMT_FP8_E4M3 => Some(elems),
        _ => tq_widths(fmt).map(|(k, v)| kv_heads * block_tokens * Record::of(k, v).bytes),
    }
}

/// True for a TurboQuant format code.
pub fn is_turboquant(fmt: u8) -> bool {
    tq_widths(fmt).is_some()
}

/// Per-layer page bytes of format `fmt`.
fn page_bytes(fmt: u8, l: &MixedPagedLayer<'_>) -> Result<usize, KernelError> {
    page_bytes_of(fmt, l.block_tokens, l.num_kv_heads, l.head_dim)
        .ok_or_else(|| invalid(format!("unknown KV block format {fmt}")))
}

fn sequences(
    l: &MixedPagedLayer<'_>,
    total_q: usize,
    tq: &TqParams,
) -> Result<Vec<Seq>, KernelError> {
    let num_seqs = l.kv_lens.len();
    let mb = l.max_blocks_per_seq;
    if l.q_indptr.len() != num_seqs + 1
        || l.block_table.len() != num_seqs * mb
        || l.block_formats.len() != l.block_table.len()
    {
        return Err(invalid(format!(
            "{num_seqs} sequences need q_indptr of {} and block_table / block_formats of {} entries, got {} / {} / {}",
            num_seqs + 1,
            num_seqs * mb,
            l.q_indptr.len(),
            l.block_table.len(),
            l.block_formats.len()
        )));
    }
    if l.q_indptr[0] != 0 || l.q_indptr[num_seqs] != total_q {
        return Err(invalid(format!(
            "q_indptr must run from 0 to total_q {total_q}"
        )));
    }
    let mut out = Vec::with_capacity(num_seqs);
    for s in 0..num_seqs {
        let (row, end, kv_len) = (l.q_indptr[s], l.q_indptr[s + 1], l.kv_lens[s]);
        let Some(q_len) = end.checked_sub(row).filter(|&n| n <= kv_len) else {
            return Err(invalid(format!(
                "sequence {s}: q rows {row}..{end} exceed kv_len {kv_len}"
            )));
        };
        let needed = kv_len.div_ceil(l.block_tokens);
        if needed > mb {
            return Err(invalid(format!(
                "sequence {s}: kv_len {kv_len} needs {needed} blocks, max_blocks_per_seq is {mb}"
            )));
        }
        let mut blocks = Vec::with_capacity(needed);
        for e in s * mb..s * mb + needed {
            let (b, fmt) = (l.block_table[e] as usize, l.block_formats[e]);
            let Some(page) = l.pages.get(b) else {
                return Err(invalid(format!(
                    "sequence {s}: block id {b} is outside the {} pages",
                    l.pages.len()
                )));
            };
            let want = page_bytes(fmt, l)?;
            if page.len() != want {
                return Err(invalid(format!(
                    "block {b} tagged format {fmt} holds {} bytes, the format needs {want}",
                    page.len()
                )));
            }
            if tq_widths(fmt).is_some()
                && (l.head_dim != TQ_DIM || tq.heads.len() != l.num_kv_heads)
            {
                return Err(invalid(format!(
                    "TurboQuant block {b} needs head_dim {TQ_DIM} and tables for {} KV heads (head_dim {}, {} tables)",
                    l.num_kv_heads,
                    l.head_dim,
                    tq.heads.len()
                )));
            }
            blocks.push((b, fmt));
        }
        out.push(Seq {
            row,
            q_len,
            kv_len,
            blocks,
        });
    }
    Ok(out)
}

/// A key or value of the sequence: plain values, or a TurboQuant record.
enum Entry<'p> {
    Plain([Vec<f32>; 2]),
    Tq(&'p [u8], Record),
}

/// Token `t`'s K/V of KV head `g` in every block of `seq`, in token order.
fn gather<'p>(l: &MixedPagedLayer<'p>, seq: &Seq, g: usize) -> Vec<Entry<'p>> {
    let (bt, hkv, d) = (l.block_tokens, l.num_kv_heads, l.head_dim);
    (0..seq.kv_len)
        .map(|pos| {
            let (b, fmt) = seq.blocks[pos / bt];
            let page: &'p [u8] = l.pages[b];
            let t = pos % bt;
            if let Some((kb, vb)) = tq_widths(fmt) {
                let rec = Record::of(kb, vb);
                return Entry::Tq(&page[(g * bt + t) * rec.bytes..][..rec.bytes], rec);
            }
            let half = |h: usize| -> Vec<f32> {
                let base = ((h * bt + t) * hkv + g) * d;
                (base..base + d)
                    .map(|e| match fmt {
                        FMT_BF16 => get_bf16(page, 2 * e),
                        _ => fp8_read(page[e], if h == 0 { l.k_scale } else { l.v_scale }),
                    })
                    .collect()
            };
            Entry::Plain([half(0), half(1)])
        })
        .collect()
}

/// An FP8 page element as attention reads it: `bf16(e4m3 × scale)`, the 6a contract every GPU
/// paged attention follows (`phase-6a-quantization` S-13).
fn fp8_read(byte: u8, scale: f32) -> f32 {
    bf16::from_f32(fp8_e4m3_value(byte) * scale).to_f32()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// `H·(s ⊙ x)/√d` in F64.
fn rotate_f64(x: &[f32], signs: &[f32]) -> Vec<f64> {
    let mut y: Vec<f64> = x
        .iter()
        .zip(signs)
        .map(|(v, s)| f64::from(*v) * f64::from(*s))
        .collect();
    fwht(&mut y);
    let k = 1.0 / (y.len() as f64).sqrt();
    y.iter_mut().for_each(|v| *v *= k);
    y
}

/// Attention of one query row `q` (one head, KV head `g`) over `entries[..visible]`.
fn attend_row(
    q: &[f32],
    entries: &[Entry<'_>],
    visible: usize,
    head: Option<&TqHeadTables>,
    tq: &TqParams,
    how: Formulation,
    scale: f32,
) -> Vec<f32> {
    let d = q.len();
    let rot = |h: &TqHeadTables| {
        let rq = rotate_f64(q, &h.k_signs);
        let sq: Vec<f64> = h
            .qjl
            .chunks_exact(TQ_DIM)
            .map(|row| row.iter().zip(&rq).map(|(a, b)| f64::from(*a) * b).sum())
            .collect();
        (rq, sq)
    };
    let rotated = match (how, head) {
        (Formulation::Rotated, Some(h)) => Some(rot(h)),
        _ => None,
    };
    let decoded: Vec<Option<[Vec<f32>; 2]>> = entries[..visible]
        .iter()
        .map(|e| match (e, how, head) {
            (Entry::Tq(r, rec), Formulation::DecodeThenAttend, Some(h)) => {
                Some(decode_record(r, *rec, h, &tq.codebooks))
            }
            _ => None,
        })
        .collect();
    let values = |i: usize| -> Option<&[Vec<f32>; 2]> {
        match &entries[i] {
            Entry::Plain(kv) => Some(kv),
            Entry::Tq(..) => decoded[i].as_ref(),
        }
    };
    // Scores.
    let scores: Vec<f64> = (0..visible)
        .map(|i| {
            let s = match (values(i), &entries[i], &rotated) {
                (Some(kv), _, _) => dot(q, &kv[0]),
                (None, Entry::Tq(r, rec), Some((rq, sq))) => {
                    let cb = &tq.codebooks[rec.k_bits - 1];
                    let mse: f64 = (0..TQ_DIM)
                        .map(|j| rq[j] * f64::from(cb[code(&r[..rec.k_norm], rec.k_bits, j)]))
                        .sum();
                    let qjl: f64 = (0..TQ_DIM)
                        .map(|j| sq[j] * f64::from(sign(&r[rec.qjl..rec.r_norm], j)))
                        .sum();
                    f64::from(get_bf16(r, rec.k_norm)) / (TQ_DIM as f64).sqrt() * mse
                        + (std::f64::consts::PI / 2.0).sqrt() / TQ_DIM as f64
                            * f64::from(get_bf16(r, rec.r_norm))
                            * qjl
                }
                _ => unreachable!("a TurboQuant entry is decoded or rotated"),
            };
            s * f64::from(scale)
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    // Values: plain-domain and rotated-domain accumulators.
    let mut plain = vec![0f64; d];
    let mut racc = vec![0f64; TQ_DIM];
    let mut any_rot = false;
    for (i, w) in weights.iter().enumerate() {
        let p = w / total;
        match (values(i), &entries[i]) {
            (Some(kv), _) => plain
                .iter_mut()
                .zip(&kv[1])
                .for_each(|(a, v)| *a += p * f64::from(*v)),
            (None, Entry::Tq(r, rec)) => {
                any_rot = true;
                let cb = &tq.codebooks[rec.v_bits - 1];
                let k = f64::from(get_bf16(r, rec.v_norm)) / (TQ_DIM as f64).sqrt();
                for (j, a) in racc.iter_mut().enumerate() {
                    let c = cb[code(&r[rec.v_codes..rec.v_norm], rec.v_bits, j)];
                    *a += p * k * f64::from(c);
                }
            }
            _ => unreachable!("a TurboQuant entry is decoded or rotated"),
        }
    }
    if any_rot {
        let h = head.expect("rotated entries have tables");
        fwht(&mut racc);
        let k = 1.0 / (TQ_DIM as f64).sqrt();
        for ((a, r), s) in plain.iter_mut().zip(&racc).zip(&h.v_signs) {
            *a += r * k * f64::from(*s);
        }
    }
    plain.iter().map(|v| *v as f32).collect()
}

fn run(
    q: &[f32],
    l: &MixedPagedLayer<'_>,
    tq: &TqParams,
    how: Formulation,
    decode_only: bool,
) -> Result<Vec<f32>, KernelError> {
    let (hq, hkv, d) = (l.num_q_heads, l.num_kv_heads, l.head_dim);
    if hkv == 0
        || !hq.is_multiple_of(hkv)
        || l.block_tokens == 0
        || d == 0
        || !q.len().is_multiple_of(hq * d)
    {
        return Err(invalid(format!(
            "bad shapes: hq {hq}, hkv {hkv}, head_dim {d}, block_tokens {}, q of {} values",
            l.block_tokens,
            q.len()
        )));
    }
    let total_q = q.len() / (hq * d);
    let seqs = sequences(l, total_q, tq)?;
    if decode_only && let Some(s) = seqs.iter().position(|s| s.q_len != 1) {
        return Err(invalid(format!("decode: sequence {s} has q_len != 1")));
    }
    let mut out = vec![0f32; q.len()];
    for seq in &seqs {
        let q_start = seq.kv_len - seq.q_len;
        for g in 0..hkv {
            let entries = gather(l, seq, g);
            let head = tq.heads.get(g);
            for i in 0..seq.q_len {
                let visible = if l.causal {
                    q_start + i + 1
                } else {
                    seq.kv_len
                };
                for qh in g * (hq / hkv)..(g + 1) * (hq / hkv) {
                    let at = ((seq.row + i) * hq + qh) * d;
                    let o = attend_row(&q[at..at + d], &entries, visible, head, tq, how, l.scale);
                    out[at..at + d].copy_from_slice(&o);
                }
            }
        }
    }
    Ok(out)
}

/// Causal (per `causal`) attention of every sequence's `q_len` new rows over its whole KV,
/// read from the mixed-format pages. `q` is `[total_q, hq, d]` F32; returns the same shape.
pub fn prefill(
    q: &[f32],
    layer: &MixedPagedLayer<'_>,
    tq: &TqParams,
    how: Formulation,
) -> Result<Vec<f32>, KernelError> {
    run(q, layer, tq, how, false)
}

/// [`prefill`] with one new row per sequence (`InvalidArgument` otherwise).
pub fn decode(
    q: &[f32],
    layer: &MixedPagedLayer<'_>,
    tq: &TqParams,
    how: Formulation,
) -> Result<Vec<f32>, KernelError> {
    run(q, layer, tq, how, true)
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use turbine_core::types::{DType, KvLayout};
    use turbine_kv::codec::turboquant::codebook::codebook;
    use turbine_kv::codec::turboquant::hadamard::{SignKind, rademacher};
    use turbine_kv::codec::turboquant::{Tq2Codec, Tq4Codec, decode_record, qjl};
    use turbine_kv::codec::{CodecParams, KvCodec};

    use super::*;
    use crate::cpu::quant::fp8_e4m3_round;

    const SEED: u64 = 0x5eed_0123_4567_89ab;
    /// The layer attended (of a 2-layer codec layout), so the layer offset of records and signs
    /// is exercised.
    const LAYER: usize = 1;
    const HQ: usize = 4;
    const HKV: usize = 2;
    const BT: usize = 16;
    const K_SCALE: f32 = 0.02;
    const V_SCALE: f32 = 0.015;

    fn gaussian(seed: u64, n: usize) -> Vec<f32> {
        // SplitMix64 uniforms, Box–Muller.
        let mut s = seed;
        let mut u = move || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (a, b) = (u(), u());
                ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
            })
            .collect()
    }

    fn tq_tables() -> TqParams {
        TqParams {
            heads: (0..HKV)
                .map(|h| TqHeadTables {
                    k_signs: rademacher(SEED, LAYER as u32, h as u32, SignKind::K, TQ_DIM),
                    v_signs: rademacher(SEED, LAYER as u32, h as u32, SignKind::V, TQ_DIM),
                    qjl: qjl::projection(SEED, LAYER as u32, h as u32, TQ_DIM),
                })
                .collect(),
            codebooks: [codebook(1), codebook(2), codebook(3), codebook(4)].map(<[f32]>::to_vec),
        }
    }

    fn fmt_name(fmt: u8) -> &'static str {
        ["bf16", "fp8_e4m3", "tq4", "tq2"][fmt as usize]
    }

    /// One pool block: the true K/V it was written from (`[2][BT][HKV][d]` F32, BF16-rounded),
    /// its page bytes for `LAYER` in `fmt`, and the exact values attention must see.
    struct Block {
        fmt: u8,
        page: Vec<u8>,
        /// `[2][BT][HKV][d]` values the reference reads (the format's decode).
        seen: Vec<f32>,
    }

    fn block(fmt: u8, seed: u64) -> Block {
        let d = TQ_DIM;
        let n = 2 * BT * HKV * d;
        let x: Vec<f32> = gaussian(seed, n)
            .iter()
            .map(|v| bf16::from_f32(*v).to_f32())
            .collect();
        let at = |h: usize, t: usize, g: usize| ((h * BT + t) * HKV + g) * d;
        match fmt {
            FMT_BF16 => Block {
                fmt,
                page: x
                    .iter()
                    .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                    .collect(),
                seen: x,
            },
            FMT_FP8_E4M3 => {
                let mut page = vec![0u8; n];
                let mut seen = vec![0f32; n];
                for h in 0..2 {
                    let s = if h == 0 { K_SCALE } else { V_SCALE };
                    for e in h * n / 2..(h + 1) * n / 2 {
                        page[e] = fp8_e4m3_round(x[e] / s);
                        seen[e] = bf16::from_f32(fp8_e4m3_value(page[e]) * s).to_f32();
                    }
                }
                Block { fmt, page, seen }
            }
            _ => {
                // A 2-layer BF16 L0 block: layer LAYER holds `x`, layer 0 other values.
                let l0 = KvLayout {
                    num_layers: 2,
                    num_kv_heads: HKV as u32,
                    head_dim: d as u32,
                    dtype: DType::BF16,
                    block_tokens: BT as u32,
                };
                let other = gaussian(seed ^ 0xff, n);
                let src: Vec<u8> = other
                    .iter()
                    .chain(&x)
                    .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                    .collect();
                let codec: &dyn KvCodec = if fmt == FMT_TQ4 { &Tq4Codec } else { &Tq2Codec };
                let widths = if fmt == FMT_TQ4 {
                    Tq4Codec::WIDTHS
                } else {
                    Tq2Codec::WIDTHS
                };
                let mut dst = vec![0u8; codec.bytes_per_block(&l0) as usize];
                let params = CodecParams {
                    seed: SEED,
                    ..CodecParams::default()
                };
                codec
                    .encode_cpu(&src, &l0, &mut dst, &params)
                    .expect("encode");
                let per_layer = dst.len() / 2;
                let page = dst[LAYER * per_layer..][..per_layer].to_vec();
                let rec = widths.record_bytes();
                let mut seen = vec![0f32; n];
                for g in 0..HKV {
                    for t in 0..BT {
                        let r = &page[(g * BT + t) * rec..][..rec];
                        let (k, v) = decode_record(widths, r, SEED, LAYER, g);
                        seen[at(0, t, g)..][..d].copy_from_slice(&k);
                        seen[at(1, t, g)..][..d].copy_from_slice(&v);
                    }
                }
                Block { fmt, page, seen }
            }
        }
    }

    /// Exact F64-accumulated causal GQA attention over the `seen` values of each sequence.
    fn expected(
        q: &[f32],
        blocks: &[Block],
        tables: &[Vec<usize>],
        q_lens: &[usize],
        kv_lens: &[usize],
        scale: f32,
    ) -> Vec<f32> {
        let d = TQ_DIM;
        let mut out = Vec::new();
        let mut row = 0;
        for s in 0..tables.len() {
            let kv = |h: usize, pos: usize, g: usize| -> &[f32] {
                let b = &blocks[tables[s][pos / BT]];
                &b.seen[(((h * BT) + pos % BT) * HKV + g) * d..][..d]
            };
            for i in 0..q_lens[s] {
                let visible = kv_lens[s] - q_lens[s] + i + 1;
                for qh in 0..HQ {
                    let g = qh / (HQ / HKV);
                    let qr = &q[((row + i) * HQ + qh) * d..][..d];
                    let sc: Vec<f64> = (0..visible)
                        .map(|p| dot(qr, kv(0, p, g)) * f64::from(scale))
                        .collect();
                    let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                    let tot: f64 = w.iter().sum();
                    let mut o = vec![0f64; d];
                    for (p, wp) in w.iter().enumerate() {
                        for (a, v) in o.iter_mut().zip(kv(1, p, g)) {
                            *a += wp / tot * f64::from(*v);
                        }
                    }
                    out.extend(o.iter().map(|v| *v as f32));
                }
            }
            row += q_lens[s];
        }
        out
    }

    /// Runs `how` over a batch: `tables[s]` are indices into `blocks` (= block ids), sequence
    /// `s` has `kv_lens[s]` tokens of which the last `q_lens[s]` are its new rows. Returns
    /// (output, expected).
    fn case(
        blocks: &[Block],
        tables: &[Vec<usize>],
        q_lens: &[usize],
        kv_lens: &[usize],
        how: Formulation,
        decode_step: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = TQ_DIM;
        let total_q: usize = q_lens.iter().sum();
        let q = gaussian(99, total_q * HQ * d);
        let mb = tables.iter().map(Vec::len).max().unwrap_or(0);
        let mut table = vec![0u32; tables.len() * mb];
        let mut formats = vec![FMT_BF16; tables.len() * mb];
        for (s, t) in tables.iter().enumerate() {
            for (j, &b) in t.iter().enumerate() {
                table[s * mb + j] = b as u32;
                formats[s * mb + j] = blocks[b].fmt;
            }
        }
        let mut indptr = vec![0usize];
        for n in q_lens {
            indptr.push(indptr.last().unwrap() + n);
        }
        let pages: Vec<&[u8]> = blocks.iter().map(|b| b.page.as_slice()).collect();
        let scale = 1.0 / (d as f32).sqrt();
        let layer = MixedPagedLayer {
            num_q_heads: HQ,
            num_kv_heads: HKV,
            head_dim: d,
            block_tokens: BT,
            pages: &pages,
            block_table: &table,
            block_formats: &formats,
            max_blocks_per_seq: mb,
            q_indptr: &indptr,
            kv_lens,
            k_scale: K_SCALE,
            v_scale: V_SCALE,
            scale,
            causal: true,
        };
        let tq = tq_tables();
        let got = if decode_step {
            decode(&q, &layer, &tq, how)
        } else {
            prefill(&q, &layer, &tq, how)
        }
        .expect("tq attention");
        (got, expected(&q, blocks, tables, q_lens, kv_lens, scale))
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    const TOL: f32 = 1e-5;

    /// Prefill (a 40-token chunk after 0 and a 10-token chunk after 30) and decode over
    /// all-`tq4` and all-`tq2` tables equal exact attention over the codec's F32 decode of the
    /// records. Breaks if the record layout, the codebook, the QJL term or the rotation signs
    /// of the reference differ from the codec.
    #[test]
    fn matches_decoded_attention() {
        for fmt in [FMT_TQ4, FMT_TQ2] {
            let blocks: Vec<Block> = (0..3).map(|i| block(fmt, 7 + i)).collect();
            let t = vec![vec![2, 0, 1]];
            for (q_lens, kv_lens, dec) in [
                (vec![40], vec![40], false),
                (vec![10], vec![40], false),
                (vec![1], vec![37], true),
            ] {
                let (got, want) = case(
                    &blocks,
                    &t,
                    &q_lens,
                    &kv_lens,
                    Formulation::DecodeThenAttend,
                    dec,
                );
                let diff = max_diff(&got, &want);
                assert!(
                    diff <= TOL,
                    "{} q {q_lens:?} kv {kv_lens:?}: max |Δ| {diff}",
                    fmt_name(fmt)
                );
            }
        }
    }

    /// The rotated-domain formulation (q rotated once, V accumulated rotated and rotated back
    /// once) equals decode-then-attend within 1e-5 for prefill and decode over every format.
    /// Breaks if the QJL correction, the norm scaling or the V back-rotation drifts.
    #[test]
    fn rotated_equals_decoded() {
        let blocks: Vec<Block> = [FMT_TQ4, FMT_TQ2, FMT_BF16, FMT_FP8_E4M3, FMT_TQ2, FMT_TQ4]
            .iter()
            .enumerate()
            .map(|(i, f)| block(*f, 100 + i as u64))
            .collect();
        let t = vec![vec![0, 1, 2], vec![3, 4, 5]];
        for (q_lens, kv_lens, dec) in [
            (vec![48, 20], vec![48, 45], false),
            (vec![1, 1], vec![33, 48], true),
        ] {
            let (rot, _) = case(&blocks, &t, &q_lens, &kv_lens, Formulation::Rotated, dec);
            let (dta, _) = case(
                &blocks,
                &t,
                &q_lens,
                &kv_lens,
                Formulation::DecodeThenAttend,
                dec,
            );
            let diff = max_diff(&rot, &dta);
            assert!(diff <= TOL, "q {q_lens:?} kv {kv_lens:?}: max |Δ| {diff}");
        }
    }

    /// FP8 blocks are read as `bf16(e4m3 × scale)` (the 6a contract the GPU kernels follow),
    /// not as the unrounded F32 product: the output equals exact attention over the rounded
    /// values, and exact attention over the unrounded ones is measurably different (so the
    /// case discriminates). Breaks if the reference drops the BF16 rounding of FP8 elements.
    #[test]
    fn fp8_blocks_read_as_bf16() {
        let blocks: Vec<Block> = (0..2).map(|i| block(FMT_FP8_E4M3, 300 + i)).collect();
        let t = vec![vec![0, 1]];
        for (q_lens, kv_lens, dec) in [(vec![1], vec![29], true), (vec![12], vec![32], false)] {
            let (got, want) = case(&blocks, &t, &q_lens, &kv_lens, Formulation::Rotated, dec);
            let diff = max_diff(&got, &want);
            assert!(diff <= TOL, "q {q_lens:?} kv {kv_lens:?}: max |Δ| {diff}");
            let unrounded: Vec<Block> = blocks
                .iter()
                .map(|b| Block {
                    fmt: b.fmt,
                    page: b.page.clone(),
                    seen: b
                        .page
                        .iter()
                        .enumerate()
                        .map(|(e, &x)| {
                            let s = if e < b.page.len() / 2 {
                                K_SCALE
                            } else {
                                V_SCALE
                            };
                            fp8_e4m3_value(x) * s
                        })
                        .collect(),
                })
                .collect();
            let (_, off) = case(&unrounded, &t, &q_lens, &kv_lens, Formulation::Rotated, dec);
            assert!(
                max_diff(&got, &off) > 10.0 * TOL,
                "q {q_lens:?}: the unrounded reading is indistinguishable ({})",
                max_diff(&got, &off)
            );
        }
    }

    /// Block tables mixing `bf16`, `fp8_e4m3`, `tq4` and `tq2` blocks within one sequence
    /// (two sequences, GQA 4:2, a prefill chunk over history and a decode step): both
    /// formulations equal exact attention over each block's own decode. Breaks if any block is
    /// read with a format other than its tag.
    #[test]
    fn mixed_block_table() {
        let fmts = [
            FMT_BF16,
            FMT_TQ4,
            FMT_FP8_E4M3,
            FMT_TQ2,
            FMT_TQ4,
            FMT_BF16,
            FMT_TQ2,
            FMT_FP8_E4M3,
        ];
        let blocks: Vec<Block> = fmts
            .iter()
            .enumerate()
            .map(|(i, f)| block(*f, 200 + i as u64))
            .collect();
        let t = vec![vec![0, 1, 2, 3], vec![7, 6, 5, 4]];
        for how in [Formulation::DecodeThenAttend, Formulation::Rotated] {
            for (q_lens, kv_lens, dec) in [
                (vec![64, 64], vec![64, 64], false),
                (vec![9, 17], vec![55, 60], false),
                (vec![1, 1], vec![49, 64], true),
            ] {
                let (got, want) = case(&blocks, &t, &q_lens, &kv_lens, how, dec);
                let diff = max_diff(&got, &want);
                assert!(
                    diff <= TOL,
                    "{how:?} q {q_lens:?} kv {kv_lens:?}: max |Δ| {diff}"
                );
            }
        }
        // A tag that disagrees with the page's bytes is refused, never read.
        let pages: Vec<&[u8]> = blocks.iter().map(|b| b.page.as_slice()).collect();
        let q = vec![0f32; HQ * TQ_DIM];
        let layer = MixedPagedLayer {
            num_q_heads: HQ,
            num_kv_heads: HKV,
            head_dim: TQ_DIM,
            block_tokens: BT,
            pages: &pages,
            block_table: &[1],
            block_formats: &[FMT_BF16],
            max_blocks_per_seq: 1,
            q_indptr: &[0, 1],
            kv_lens: &[1],
            k_scale: K_SCALE,
            v_scale: V_SCALE,
            scale: 1.0,
            causal: true,
        };
        assert!(decode(&q, &layer, &tq_tables(), Formulation::Rotated).is_err());
    }
}
