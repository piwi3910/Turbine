//! KV codecs (P6b S-1; contract §24 `kv_format`): how a block's bytes are stored in a lower tier.
//!
//! A codec turns one L0 block (the bytes [`crate::pool::BlockPool::block_segments`] copies: per
//! layer `[2, block_tokens, kv_heads, head_dim]` elements of the L0 dtype, K before V) into the
//! bytes of one tier slot and back. A codec is one file (or directory) in this module plus one
//! entry in [`registry`]; `kv.cpu.format` / `kv.nvme.format` select it by name.
//!
//! This crate stays GPU-free: `encode_cpu` / `decode_cpu` are the reference every GPU transcode
//! (the v2.10 `turbine_kv_transcode` op, identified by [`KvCodec::abi_code`]) is tested against.
//!
//! Registration order is the lossiness order of the compression ladder (P6b S-6): `l0` (the L0
//! bytes unchanged) < `fp8_e4m3` < `tq4` < `tq2`. [`rung_index`] and [`next_rung`] read it; the
//! conformance suite checks that slot sizes never grow along it.

#[cfg(test)]
pub(crate) mod conformance;
mod fp8_e4m3;
mod l0;
pub mod turboquant;

pub use fp8_e4m3::Fp8E4m3Codec;
pub use l0::L0Codec;
pub use turboquant::{Tq2Codec, Tq4Codec};

use turbine_core::registry::{Module, Registry};
use turbine_core::types::{DType, KvLayout};

/// Parameters a codec needs besides the block bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CodecParams {
    /// Rotation seed (TurboQuant): the first 8 bytes of the namespace key, little-endian.
    pub seed: u64,
    /// Per-layer K scales of an FP8 L0 (`kv.dtype: fp8_e4m3`, 6a S-13): an L0 element's value is
    /// its FP8 value × the layer's scale. Empty means 1.0 for every layer; ignored for BF16 L0.
    pub k_scales: Vec<f32>,
    /// Per-layer V scales, as [`CodecParams::k_scales`].
    pub v_scales: Vec<f32>,
}

impl CodecParams {
    fn scale(scales: &[f32], layer: usize) -> f32 {
        scales.get(layer).copied().unwrap_or(1.0)
    }

    /// The scale of layer `layer`'s K (`kind` 0) or V (`kind` 1) in an FP8 L0.
    pub fn l0_scale(&self, layer: usize, kind: usize) -> f32 {
        if kind == 0 {
            Self::scale(&self.k_scales, layer)
        } else {
            Self::scale(&self.v_scales, layer)
        }
    }
}

/// Why a codec refused a block.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CodecError {
    #[error("{codec}: L0 dtype {dtype} is not a KV page dtype (bf16 or f8e4m3)")]
    UnsupportedDtype {
        codec: &'static str,
        dtype: &'static str,
    },
    #[error("{codec}: head_dim {head_dim} is not supported (needs {needed})")]
    UnsupportedHeadDim {
        codec: &'static str,
        head_dim: u32,
        needed: u32,
    },
    #[error("{codec}: {what} buffer holds {got} bytes, the layout needs {expected}")]
    Size {
        codec: &'static str,
        what: &'static str,
        got: usize,
        expected: u64,
    },
}

/// A KV codec (contract §24 `kv_format`). Stateless and deterministic: the same block, layout
/// and parameters always give the same bytes.
pub trait KvCodec: Module {
    /// True when `decode(encode(x))` may differ from `x` for an L0 of this layout.
    fn lossy(&self, l0: &KvLayout) -> bool;
    /// `TURBINE_KVFMT_*` code of the v2.10 transcode op (`l0` 0, `fp8_e4m3` 1, `tq4` 2, `tq2` 3).
    fn abi_code(&self) -> u8;
    /// Documented bound on the normalised reconstruction error of a lossy codec, ‖x − x̂‖² / ‖x‖²
    /// over a block's K and V (0 for a lossless one); the conformance suite holds every codec
    /// to it on seeded Gaussian and outlier-heavy blocks.
    fn nmse_bound(&self) -> f64;
    /// Default planner penalty of a block stored in this codec (`kv.lossy_penalty`, P6b S-3,
    /// Q16): its retrieval cost is multiplied by `1 + penalty`; 0 for a lossless codec.
    fn default_lossy_penalty(&self) -> f64;
    /// Whether the codec can store blocks of this L0 layout.
    fn supports(&self, l0: &KvLayout) -> Result<(), CodecError> {
        check_l0_dtype(self.name(), l0)
    }
    /// Bytes of one encoded block (one tier slot).
    fn bytes_per_block(&self, l0: &KvLayout) -> u64;
    /// Encodes one L0 block (`l0.block_bytes()` bytes) into `dst` (`bytes_per_block` bytes).
    fn encode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        params: &CodecParams,
    ) -> Result<(), CodecError>;
    /// Decodes one slot (`bytes_per_block` bytes) into an L0 block of the L0 dtype.
    fn decode_cpu(
        &self,
        src: &[u8],
        l0: &KvLayout,
        dst: &mut [u8],
        params: &CodecParams,
    ) -> Result<(), CodecError>;
}

static REGISTRY: Registry<dyn KvCodec> = Registry::new(
    "kv_format",
    &[&L0Codec, &Fp8E4m3Codec, &Tq4Codec, &Tq2Codec],
);

/// Every registered KV codec, in lossiness order (`l0` first, the configuration default).
pub fn registry() -> &'static Registry<dyn KvCodec> {
    &REGISTRY
}

/// Position of codec `name` on the compression ladder (0 = `l0`), from the registration order.
pub fn rung_index(name: &str) -> Option<usize> {
    registry().iter().position(|c| c.name() == name)
}

/// Rung of a tier that stores codec `format` below an L0 whose page format is named `l0_dtype`
/// (`kv.dtype`: `bf16`, or the name of the codec its pages are stored in): `l0` (the first
/// codec, the L0 bytes unchanged) takes the L0 format's rung — 0 for BF16 — so `fp8_e4m3` below
/// an FP8 L0 ranks as `l0`; every other codec its registration position. `None` for an
/// unregistered `format`. A tier may not rank below the tier above it (P6b S-2).
pub fn tier_rung(format: &str, l0_dtype: &str) -> Option<usize> {
    if rung_index(format)? == 0 {
        Some(rung_index(l0_dtype).unwrap_or(0))
    } else {
        rung_index(format)
    }
}

/// The next lossier registered codec after `name`, if any.
pub fn next_rung(name: &str) -> Option<&'static str> {
    let i = rung_index(name)?;
    registry().iter().nth(i + 1).map(|c| c.name())
}

/// The lossier of two registered codec names (unregistered names count as the most precise).
pub fn lossier(a: &'static str, b: &'static str) -> &'static str {
    if rung_index(b).unwrap_or(0) > rung_index(a).unwrap_or(0) {
        b
    } else {
        a
    }
}

/// `Err` unless the L0 dtype is one a KV page can have (BF16 or FP8 e4m3).
pub(crate) fn check_l0_dtype(codec: &'static str, l0: &KvLayout) -> Result<(), CodecError> {
    match l0.dtype {
        DType::BF16 | DType::F8E4M3 => Ok(()),
        other => Err(CodecError::UnsupportedDtype {
            codec,
            dtype: other.as_str(),
        }),
    }
}

pub(crate) fn check_size(
    codec: &'static str,
    what: &'static str,
    buf: usize,
    expected: u64,
) -> Result<(), CodecError> {
    if buf as u64 == expected {
        Ok(())
    } else {
        Err(CodecError::Size {
            codec,
            what,
            got: buf,
            expected,
        })
    }
}

/// Element offsets of an L0 block: layer `l`, K (`kind` 0) or V (1), token `t`, head `h` start
/// the `head_dim` elements of one token-head vector.
#[derive(Clone, Copy, Debug)]
pub(crate) struct L0Geometry {
    pub layers: usize,
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl L0Geometry {
    pub fn of(l0: &KvLayout) -> Self {
        L0Geometry {
            layers: l0.num_layers as usize,
            tokens: l0.block_tokens as usize,
            heads: l0.num_kv_heads as usize,
            head_dim: l0.head_dim as usize,
        }
    }

    /// Elements of one layer's K (or V).
    pub fn half_layer_elems(&self) -> usize {
        self.tokens * self.heads * self.head_dim
    }

    /// First element of vector (layer, kind, token, head).
    pub fn vector_elem(&self, layer: usize, kind: usize, token: usize, head: usize) -> usize {
        ((layer * 2 + kind) * self.tokens + token) * self.heads * self.head_dim
            + head * self.head_dim
    }
}

/// Reads element `i` of an L0 block as F32 (`scale` multiplies an FP8 value).
pub(crate) fn read_l0(src: &[u8], dtype: DType, i: usize, scale: f32) -> f32 {
    match dtype {
        DType::F8E4M3 => fp8_e4m3_value(src[i]) * scale,
        _ => bf16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]])),
    }
}

/// Writes element `i` of an L0 block from F32 (an FP8 element stores `x / scale`).
pub(crate) fn write_l0(dst: &mut [u8], dtype: DType, i: usize, x: f32, scale: f32) {
    match dtype {
        DType::F8E4M3 => dst[i] = fp8_e4m3_round(x / scale),
        _ => dst[2 * i..2 * i + 2].copy_from_slice(&f32_to_bf16(x).to_le_bytes()),
    }
}

/// BF16 bits of `x`, rounded to nearest with ties to even (NaN stays a quiet NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x40;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// The F32 value of BF16 bits.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Largest finite FP8 e4m3 magnitude.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// OCP e4m3fn bits of `x`: round to nearest, ties to even, saturated to ±448; NaN → `0x7f`.
///
/// A copy of `turbine_kernels::cpu::quant::fp8_e4m3_round` (this crate stays GPU-free and does
/// not depend on `turbine-kernels`; provisional, pending user review — the alternative is to move
/// the rounding into `turbine-core`). `codec::tests::fp8_rounding_matches_kernels_table` holds it
/// to the same table bit for bit.
pub fn fp8_e4m3_round(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a >= FP8_E4M3_MAX {
        return sign | 0x7e;
    }
    // Below the smallest normal (2^-6): subnormals are multiples of 2^-9.
    if a < 2f32.powi(-6) {
        let m = (a * 512.0).round_ties_even() as u8; // exact scaling by a power of two
        return sign | m; // m == 8 is the smallest normal, 0x08
    }
    let bits = a.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x7f_ffff;
    let mut q = mant >> 20;
    let rem = mant & 0xf_ffff;
    let half = 0x8_0000;
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1;
    }
    if q == 8 {
        q = 0;
        exp += 1;
    }
    let biased = (exp + 7) as u8; // a < 448 keeps this ≤ 15 and away from the NaN code
    sign | (biased << 3) | q as u8
}

/// The value of OCP e4m3fn bits (NaN for `0x7f` / `0xff`); a copy of the kernels' reference.
pub fn fp8_e4m3_value(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xf);
    let m = f32::from(b & 7);
    if e == 15 && m == 7.0 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small L0 layout: 2 layers × 2 KV heads × 128 × 16 tokens.
    pub(crate) fn layout(dtype: DType) -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 128,
            dtype,
            block_tokens: 16,
        }
    }

    /// Seeded xorshift64* standard normals (Box–Muller).
    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub fn next_u64(&mut self) -> u64 {
            let mut s = self.0;
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            self.0 = s;
            s.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        pub fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        pub fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// One L0 block of `l0` with seeded values: Gaussian, or with `outliers` a few channels
    /// 20× larger (the K outlier channels of real models) and a few 30× larger tokens.
    pub(crate) fn block(l0: &KvLayout, seed: u64, outliers: bool, p: &CodecParams) -> Vec<u8> {
        let g = L0Geometry::of(l0);
        let mut rng = Rng(seed | 1);
        let mut out = vec![0u8; l0.block_bytes() as usize];
        for layer in 0..g.layers {
            for kind in 0..2 {
                let scale = p.l0_scale(layer, kind);
                for t in 0..g.tokens {
                    for h in 0..g.heads {
                        let base = g.vector_elem(layer, kind, t, h);
                        for d in 0..g.head_dim {
                            let mut x = rng.normal() as f32;
                            if outliers && (d % 37 == 5 || t % 7 == 3 && kind == 1) {
                                x *= if d % 37 == 5 { 20.0 } else { 30.0 };
                            }
                            // FP8 L0: values in the scale's range, as the paged append writes.
                            let x = if l0.dtype == DType::F8E4M3 {
                                x * scale
                            } else {
                                x
                            };
                            write_l0(&mut out, l0.dtype, base + d, x, scale);
                        }
                    }
                }
            }
        }
        out
    }

    /// ‖x − x̂‖² / ‖x‖² between two L0 blocks.
    pub(crate) fn nmse(a: &[u8], b: &[u8], l0: &KvLayout, p: &CodecParams) -> f64 {
        let g = L0Geometry::of(l0);
        let (mut err, mut energy) = (0f64, 0f64);
        for layer in 0..g.layers {
            for kind in 0..2 {
                let scale = p.l0_scale(layer, kind);
                let start = g.vector_elem(layer, kind, 0, 0);
                for i in start..start + g.half_layer_elems() {
                    let x = f64::from(read_l0(a, l0.dtype, i, scale));
                    let y = f64::from(read_l0(b, l0.dtype, i, scale));
                    err += (x - y) * (x - y);
                    energy += x * x;
                }
            }
        }
        err / energy.max(f64::MIN_POSITIVE)
    }

    pub(crate) fn round_trip(
        c: &dyn KvCodec,
        l0: &KvLayout,
        src: &[u8],
        p: &CodecParams,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut enc = vec![0u8; c.bytes_per_block(l0) as usize];
        c.encode_cpu(src, l0, &mut enc, p).unwrap();
        let mut dec = vec![0u8; l0.block_bytes() as usize];
        c.decode_cpu(&enc, l0, &mut dec, p).unwrap();
        (enc, dec)
    }

    /// The `kv_format` registry lists the four codecs in lossiness order; the ladder helpers
    /// read that order. Breaks if a codec is missing, renamed or registered out of order.
    #[test]
    fn registry_lists_codecs() {
        let reg = registry();
        assert_eq!(reg.point(), "kv_format");
        assert_eq!(reg.names(), ["l0", "fp8_e4m3", "tq4", "tq2"]);
        let codes: Vec<u8> = reg.iter().map(|c| c.abi_code()).collect();
        assert_eq!(codes, [0, 1, 2, 3], "TURBINE_KVFMT_* codes");
        assert_eq!(rung_index("l0"), Some(0));
        assert_eq!(rung_index("tq2"), Some(3));
        assert_eq!(rung_index("zstd"), None);
        assert_eq!(next_rung("l0"), Some("fp8_e4m3"));
        assert_eq!(next_rung("fp8_e4m3"), Some("tq4"));
        assert_eq!(next_rung("tq4"), Some("tq2"));
        assert_eq!(next_rung("tq2"), None);
        assert_eq!(lossier("l0", "tq4"), "tq4");
        assert_eq!(lossier("tq2", "fp8_e4m3"), "tq2");
    }

    /// Tier ordering (P6b S-2) is read from the registry: a tier's rung is its codec's
    /// position, except `l0`, which is the L0 page format's own rung (0 for BF16), so `fp8_e4m3`
    /// below an FP8 L0 ranks as `l0`; lossy codecs carry their planner penalty defaults
    /// (`kv.lossy_penalty`, Q16). Breaks if the ordering or the defaults drift.
    #[test]
    fn tier_ordering() {
        assert_eq!(tier_rung("l0", "bf16"), Some(0));
        assert_eq!(tier_rung("fp8_e4m3", "bf16"), Some(1));
        assert_eq!(tier_rung("tq4", "bf16"), Some(2));
        assert_eq!(tier_rung("tq2", "fp8_e4m3"), Some(3));
        assert_eq!(
            tier_rung("l0", "fp8_e4m3"),
            tier_rung("fp8_e4m3", "fp8_e4m3")
        );
        assert_eq!(tier_rung("l0", "tq2"), Some(3));
        assert!(tier_rung("fp8_e4m3", "tq4") < tier_rung("l0", "tq4"));
        assert_eq!(tier_rung("zstd", "bf16"), None);
        let penalties: Vec<f64> = registry()
            .iter()
            .map(|c| c.default_lossy_penalty())
            .collect();
        assert_eq!(penalties, [0.0, 0.1, 0.5, 1.0]);
    }

    /// `l0` stores the L0 bytes unchanged, for BF16 and FP8 pages. Breaks if the identity codec
    /// changes a byte or its slot size differs from the L0 block.
    #[test]
    fn l0_is_identity() {
        let p = CodecParams::default();
        for dtype in [DType::BF16, DType::F8E4M3] {
            let l0 = layout(dtype);
            assert_eq!(L0Codec.bytes_per_block(&l0), l0.block_bytes());
            assert!(!L0Codec.lossy(&l0));
            let src = block(&l0, 7, true, &p);
            let (enc, dec) = round_trip(&L0Codec, &l0, &src, &p);
            assert_eq!(enc, src);
            assert_eq!(dec, src);
        }
    }

    /// From a BF16 L0, `fp8_e4m3` halves the slot (plus one F32 K and V scale per layer) and
    /// reconstructs within its bound, outliers included. Breaks if the scale header is dropped,
    /// the scale ignores outliers (saturation) or the size stops fitting ~2× more blocks.
    #[test]
    fn fp8_from_bf16_within_bound() {
        let p = CodecParams::default();
        let l0 = layout(DType::BF16);
        assert!(Fp8E4m3Codec.lossy(&l0));
        let slot = Fp8E4m3Codec.bytes_per_block(&l0);
        assert_eq!(slot, l0.block_bytes() / 2 + 8 * 2);
        assert!(l0.block_bytes() as f64 / slot as f64 >= 1.9);
        for outliers in [false, true] {
            let src = block(&l0, 11, outliers, &p);
            let (_, dec) = round_trip(&Fp8E4m3Codec, &l0, &src, &p);
            let e = nmse(&src, &dec, &l0, &p);
            assert!(e > 0.0 && e <= Fp8E4m3Codec.nmse_bound(), "nmse {e}");
        }
        // A zero block stays zero (the scale floor keeps it finite).
        let zero = vec![0u8; l0.block_bytes() as usize];
        let (_, dec) = round_trip(&Fp8E4m3Codec, &l0, &zero, &p);
        assert_eq!(dec, zero);
    }

    /// From an FP8 L0 (6a), `fp8_e4m3` keeps the page bytes unchanged — the L0 scales stay the
    /// namespace's. Breaks if an FP8 page is re-quantized.
    #[test]
    fn fp8_from_fp8_is_identity() {
        let p = CodecParams {
            seed: 0,
            k_scales: vec![0.5, 2.0],
            v_scales: vec![0.25, 4.0],
        };
        let l0 = layout(DType::F8E4M3);
        assert!(!Fp8E4m3Codec.lossy(&l0));
        assert_eq!(Fp8E4m3Codec.bytes_per_block(&l0), l0.block_bytes());
        let src = block(&l0, 13, true, &p);
        let (enc, dec) = round_trip(&Fp8E4m3Codec, &l0, &src, &p);
        assert_eq!(enc, src);
        assert_eq!(dec, src);
    }

    /// Wrong buffer sizes and non-page dtypes are refused, never truncated.
    #[test]
    fn sizes_and_dtypes_are_checked() {
        let p = CodecParams::default();
        let l0 = layout(DType::BF16);
        let src = vec![0u8; l0.block_bytes() as usize];
        let mut short = vec![0u8; 3];
        for c in registry().iter() {
            assert!(
                matches!(
                    c.encode_cpu(&src, &l0, &mut short, &p),
                    Err(CodecError::Size { .. })
                ),
                "{}",
                c.name()
            );
            assert!(matches!(
                c.supports(&layout(DType::F32)),
                Err(CodecError::UnsupportedDtype { .. })
            ));
        }
    }

    /// The local FP8 rounding equals `turbine_kernels::cpu::quant::fp8_e4m3_round` on the
    /// kernels' own table (copied verbatim) and every finite code round-trips. Breaks if the
    /// copy drifts from the kernels' reference (ties, subnormals, saturation, NaN).
    #[test]
    fn fp8_rounding_matches_kernels_table() {
        let cases: [(f32, u8); 16] = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (1.0, 0x38),
            (-1.0, 0xb8),
            (448.0, 0x7e),
            (500.0, 0x7e),
            (f32::INFINITY, 0x7e),
            (-1e9, 0xfe),
            (2f32.powi(-9), 0x01),
            (2f32.powi(-10), 0x00),
            (3.0 * 2f32.powi(-10), 0x02),
            (1.0625, 0x38),
            (1.1875, 0x3a),
            (2f32.powi(-6), 0x08),
            (15.5 * 2f32.powi(-10), 0x08),
            (447.0, 0x7e),
        ];
        for (x, code) in cases {
            assert_eq!(fp8_e4m3_round(x), code, "{x}");
        }
        assert_eq!(fp8_e4m3_round(f32::NAN), 0x7f);
        assert!(fp8_e4m3_value(0x7f).is_nan() && fp8_e4m3_value(0xff).is_nan());
        assert_eq!(fp8_e4m3_value(0x7e), 448.0);
        assert_eq!(fp8_e4m3_value(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e4m3_value(0x3a), 1.25);
        for code in 0u8..=255 {
            let v = fp8_e4m3_value(code);
            if v.is_nan() {
                continue;
            }
            assert_eq!(fp8_e4m3_value(fp8_e4m3_round(v)), v, "code {code:#04x}");
        }
    }

    /// BF16 rounding: nearest, ties to even, NaN stays NaN.
    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        // 1 + 2^-8 is a tie between 1.0 and 1 + 2^-7 → 1.0 (even).
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3f80);
        // 1 + 3·2^-8 is a tie between 1 + 2^-7 (odd) and 1 + 2^-6 → the even one.
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3f82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }
}
