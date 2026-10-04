//! TurboQuant KV codecs `tq4` and `tq2` (P6b S-4; Zandieh et al., arXiv 2504.19874), CPU
//! reference.
//!
//! Per token-head vector `x` (head_dim 128), with signs from (namespace seed, layer, head):
//!
//! - rotate: `y = H·(s ⊙ x)/√d` ([`hadamard`]);
//! - norm: `‖x‖` stored in BF16 (`n`);
//! - codes: each `y_i·√d/‖x‖` quantized to the nearest centroid of the `b`-bit Lloyd–Max
//!   codebook ([`codebook`]); decode `ŷ_i = c[code_i]·n/√d`;
//! - decode: `x̂ = s ⊙ (H·ŷ)/√d`, written to the L0 dtype.
//!
//! K and V take the same path (the paper's TurboQuant_mse) with their own rotation signs: on real
//! Llama / OLMoE K the inner-product variant (b − 1 MSE bits plus a 1-bit QJL residual) has 5×
//! the reconstruction error and twice the attention-output error (decision "6b Task 9: TurboQuant
//! K quantizer", user decision A, 2026-10-01). `tq4`: K and V 4 bits; `tq2`: K and V 2 bits;
//! every layer alike.
//!
//! Slot layout (spec Data): records of one token-head vector pair, ordered per layer, per KV
//! head, per token (record `(layer·heads + head)·block_tokens + token`); a record is K codes
//! (bit-packed), K norm (BF16), V codes, V norm (BF16), zero padding to a multiple of 16 bytes,
//! so every record — and its K codes — starts 16-byte aligned (V codes 2-byte aligned). Codes are packed least-significant bit first: code `i` of `b` bits
//! occupies bits `i·b ..` of the field (byte `bit / 8`, bit `bit % 8`). BF16 values are
//! little-endian. `tq4` records are 144 bytes (BF16 L0: 512), `tq2` records 80.

pub mod codebook;
pub mod hadamard;

use turbine_core::registry::Module;
use turbine_core::types::KvLayout;

use self::codebook::{TQ_DIM, codebook, nearest};
use self::hadamard::{SignKind, rademacher, rotate, unrotate};
use super::{
    CodecError, CodecParams, KvCodec, L0Geometry, bf16_to_f32, check_l0_dtype, check_size,
    f32_to_bf16, read_l0, write_l0,
};

/// The widths of one TurboQuant variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TqWidths {
    /// Bits of K.
    pub k_bits: u32,
    /// Bits of V.
    pub v_bits: u32,
}

impl TqWidths {
    fn k_code_bytes(self) -> usize {
        TQ_DIM * self.k_bits as usize / 8
    }
    fn v_code_bytes(self) -> usize {
        TQ_DIM * self.v_bits as usize / 8
    }
    /// Bytes of one record (one K and one V vector), padded to 16.
    pub fn record_bytes(self) -> usize {
        let raw = self.k_code_bytes() + 2 + self.v_code_bytes() + 2;
        raw.div_ceil(16) * 16
    }
}

/// One encoded K or V vector: packed codes and the BF16 norm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqVec {
    pub codes: Vec<u8>,
    pub norm: u16,
}

fn sqrt_d() -> f32 {
    (TQ_DIM as f64).sqrt() as f32
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt() as f32
}

/// `ŷ_i = c[code_i]·n/√d` (rotated domain).
fn mse_values(codes: &[u8], norm: u16, bits: u32) -> Vec<f32> {
    let cb = codebook(bits);
    let k = bf16_to_f32(norm) / sqrt_d();
    codes.iter().map(|c| cb[usize::from(*c)] * k).collect()
}

/// Encodes one K or V vector (`TQ_DIM` values) with `bits` bits per coordinate and the
/// rotation `signs` of its kind: codes of `y·√d/‖x‖` against the Lloyd–Max codebook.
pub fn encode_vec(x: &[f32], bits: u32, signs: &[f32]) -> TqVec {
    let cb = codebook(bits);
    let n = norm(x);
    let y = rotate(x, signs);
    let inv = if n > 0.0 { sqrt_d() / n } else { 0.0 };
    let codes: Vec<u8> = y.iter().map(|v| nearest(cb, v * inv)).collect();
    TqVec {
        codes: pack(&codes, bits),
        norm: f32_to_bf16(n),
    }
}

/// Decodes one K or V vector: `s ⊙ H·ŷ/√d`.
pub fn decode_vec(v: &TqVec, bits: u32, signs: &[f32]) -> Vec<f32> {
    let y = mse_values(&unpack(&v.codes, bits, TQ_DIM), v.norm, bits);
    unrotate(&y, signs)
}

/// Packs `bits`-bit codes least-significant bit first.
pub fn pack(codes: &[u8], bits: u32) -> Vec<u8> {
    let b = bits as usize;
    let mut out = vec![0u8; (codes.len() * b).div_ceil(8)];
    for (i, c) in codes.iter().enumerate() {
        for j in 0..b {
            if c >> j & 1 == 1 {
                let bit = i * b + j;
                out[bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    out
}

/// Unpacks `n` codes of `bits` bits.
pub fn unpack(packed: &[u8], bits: u32, n: usize) -> Vec<u8> {
    let b = bits as usize;
    (0..n)
        .map(|i| {
            (0..b).fold(0u8, |c, j| {
                let bit = i * b + j;
                c | ((packed[bit / 8] >> (bit % 8) & 1) << j)
            })
        })
        .collect()
}

/// The rotation signs of one (layer, head).
struct HeadSigns {
    k: Vec<f32>,
    v: Vec<f32>,
}

impl HeadSigns {
    fn of(seed: u64, layer: usize, head: usize) -> Self {
        let s = |kind| rademacher(seed, layer as u32, head as u32, kind, TQ_DIM);
        HeadSigns {
            k: s(SignKind::K),
            v: s(SignKind::V),
        }
    }
}

/// Byte offsets of the fields of a record.
struct Fields {
    k_codes: usize,
    k_norm: usize,
    v_codes: usize,
    v_norm: usize,
}

impl Fields {
    fn of(w: TqWidths) -> Self {
        let k_norm = w.k_code_bytes();
        let v_codes = k_norm + 2;
        let v_norm = v_codes + w.v_code_bytes();
        Fields {
            k_codes: 0,
            k_norm,
            v_codes,
            v_norm,
        }
    }
}

fn get_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// K and V of one record, in F32 (before the L0 rounding).
fn decode_record_with(w: TqWidths, r: &[u8], s: &HeadSigns) -> (Vec<f32>, Vec<f32>) {
    let f = Fields::of(w);
    let k = TqVec {
        codes: r[f.k_codes..f.k_norm].to_vec(),
        norm: get_u16(r, f.k_norm),
    };
    let v = TqVec {
        codes: r[f.v_codes..f.v_norm].to_vec(),
        norm: get_u16(r, f.v_norm),
    };
    (
        decode_vec(&k, w.k_bits, &s.k),
        decode_vec(&v, w.v_bits, &s.v),
    )
}

/// Decodes one record (`w.record_bytes()` bytes of (layer, head) under `seed`) to its K and V
/// in F32, without the rounding to the L0 dtype that `decode_cpu` applies: the values the
/// reference TurboQuant attention (`turbine_kernels::cpu::tq_attention`) is held to.
pub fn decode_record(
    w: TqWidths,
    record: &[u8],
    seed: u64,
    layer: usize,
    head: usize,
) -> (Vec<f32>, Vec<f32>) {
    decode_record_with(w, record, &HeadSigns::of(seed, layer, head))
}

/// Encodes one token-head K/V pair (`TQ_DIM` values each) into `record` (`w.record_bytes()`
/// bytes, fully written, padding zeroed) with the (layer, head) rotation signs: exactly the record `encode_cpu` writes for these values. The L0 paged append of
/// TurboQuant pages (P6b S-5) encodes new rows through it.
pub fn encode_record(
    w: TqWidths,
    k: &[f32],
    v: &[f32],
    k_signs: &[f32],
    v_signs: &[f32],
    record: &mut [u8],
) {
    let f = Fields::of(w);
    let k = encode_vec(k, w.k_bits, k_signs);
    let v = encode_vec(v, w.v_bits, v_signs);
    let r = &mut record[..w.record_bytes()];
    r.fill(0);
    r[f.k_codes..f.k_norm].copy_from_slice(&k.codes);
    r[f.k_norm..f.v_codes].copy_from_slice(&k.norm.to_le_bytes());
    r[f.v_codes..f.v_norm].copy_from_slice(&v.codes);
    r[f.v_norm..f.v_norm + 2].copy_from_slice(&v.norm.to_le_bytes());
}

/// Shared implementation of `tq4` and `tq2`.
fn tq_supports(name: &'static str, l0: &KvLayout) -> Result<(), CodecError> {
    check_l0_dtype(name, l0)?;
    if l0.head_dim as usize != TQ_DIM {
        return Err(CodecError::UnsupportedHeadDim {
            codec: name,
            head_dim: l0.head_dim,
            needed: TQ_DIM as u32,
        });
    }
    Ok(())
}

fn tq_bytes(w: TqWidths, l0: &KvLayout) -> u64 {
    u64::from(l0.num_layers)
        * u64::from(l0.num_kv_heads)
        * u64::from(l0.block_tokens)
        * w.record_bytes() as u64
}

fn tq_encode(
    name: &'static str,
    w: TqWidths,
    src: &[u8],
    l0: &KvLayout,
    dst: &mut [u8],
    p: &CodecParams,
) -> Result<(), CodecError> {
    tq_supports(name, l0)?;
    check_size(name, "source", src.len(), l0.block_bytes())?;
    check_size(name, "destination", dst.len(), tq_bytes(w, l0))?;
    let g = L0Geometry::of(l0);
    let rec = w.record_bytes();
    let read = |layer, kind, t, h| -> Vec<f32> {
        let base = g.vector_elem(layer, kind, t, h);
        let scale = p.l0_scale(layer, kind);
        (base..base + TQ_DIM)
            .map(|i| read_l0(src, l0.dtype, i, scale))
            .collect()
    };
    dst.fill(0);
    for layer in 0..g.layers {
        for h in 0..g.heads {
            let s = HeadSigns::of(p.seed, layer, h);
            for t in 0..g.tokens {
                let r = &mut dst[((layer * g.heads + h) * g.tokens + t) * rec..][..rec];
                let (k, v) = (read(layer, 0, t, h), read(layer, 1, t, h));
                encode_record(w, &k, &v, &s.k, &s.v, r);
            }
        }
    }
    Ok(())
}

fn tq_decode(
    name: &'static str,
    w: TqWidths,
    src: &[u8],
    l0: &KvLayout,
    dst: &mut [u8],
    p: &CodecParams,
) -> Result<(), CodecError> {
    tq_supports(name, l0)?;
    check_size(name, "source", src.len(), tq_bytes(w, l0))?;
    check_size(name, "destination", dst.len(), l0.block_bytes())?;
    let g = L0Geometry::of(l0);
    let rec = w.record_bytes();
    for layer in 0..g.layers {
        for h in 0..g.heads {
            let s = HeadSigns::of(p.seed, layer, h);
            for t in 0..g.tokens {
                let r = &src[((layer * g.heads + h) * g.tokens + t) * rec..][..rec];
                let (k, v) = decode_record_with(w, r, &s);
                for (kind, x) in [(0, k), (1, v)] {
                    let base = g.vector_elem(layer, kind, t, h);
                    let scale = p.l0_scale(layer, kind);
                    for (i, val) in (base..).zip(x) {
                        write_l0(dst, l0.dtype, i, val, scale);
                    }
                }
            }
        }
    }
    Ok(())
}

macro_rules! tq_codec {
    ($ty:ident, $name:literal, $abi:literal, $k:literal, $v:literal, $bound:literal, $penalty:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug)]
        pub struct $ty;

        impl $ty {
            pub const WIDTHS: TqWidths = TqWidths {
                k_bits: $k,
                v_bits: $v,
            };
        }

        impl Module for $ty {
            fn name(&self) -> &'static str {
                $name
            }
        }

        impl KvCodec for $ty {
            fn lossy(&self, _l0: &KvLayout) -> bool {
                true
            }

            fn abi_code(&self) -> u8 {
                $abi
            }

            /// K and V: D_mse(bits) of the committed codebook with a ≈ 25 % margin over the
            /// largest measured block nmse (`kv_codecs_suite`, Gaussian and outlier blocks over
            /// BF16 and FP8 L0: `tq4` ≤ 0.0098, `tq2` ≤ 0.119; real Llama-3.2-3B / OLMoE K and
            /// V, `tq_loss_on_real_kv`: ≤ 0.0094 / ≤ 0.118): `tq4` 0.0125, `tq2` 0.15.
            fn nmse_bound(&self) -> f64 {
                $bound
            }

            fn default_lossy_penalty(&self) -> f64 {
                $penalty
            }

            fn supports(&self, l0: &KvLayout) -> Result<(), CodecError> {
                tq_supports($name, l0)
            }

            fn bytes_per_block(&self, l0: &KvLayout) -> u64 {
                tq_bytes(Self::WIDTHS, l0)
            }

            fn encode_cpu(
                &self,
                src: &[u8],
                l0: &KvLayout,
                dst: &mut [u8],
                params: &CodecParams,
            ) -> Result<(), CodecError> {
                tq_encode($name, Self::WIDTHS, src, l0, dst, params)
            }

            fn decode_cpu(
                &self,
                src: &[u8],
                l0: &KvLayout,
                dst: &mut [u8],
                params: &CodecParams,
            ) -> Result<(), CodecError> {
                tq_decode($name, Self::WIDTHS, src, l0, dst, params)
            }
        }
    };
}

tq_codec!(
    Tq4Codec,
    "tq4",
    2,
    4,
    4,
    0.0125,
    0.5,
    "`tq4`: K and V 4 bits; 144-byte records."
);
tq_codec!(
    Tq2Codec,
    "tq2",
    3,
    2,
    2,
    0.15,
    1.0,
    "`tq2`: K and V 2 bits; 80-byte records."
);

#[cfg(test)]
mod tests;
