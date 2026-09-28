//! CPU reference for the Phase 6a quantized formats (S-5): FP8 e4m3 and MXFP4 rounding, the
//! dequantization of every [`QuantSchemeDesc`] layout, and activation quantize-dequantize for
//! every [`ActQuantDesc`]. Every GPU implementation of the quantized GEMM and of activation
//! quantization is tested against these functions.
//!
//! Rules:
//! - FP8 is OCP `e4m3fn` (bias 7, no infinities, NaN = `0x7f` / `0xff`, largest finite ±448),
//!   rounded to nearest with ties to even and saturated to ±448 (the quantizers of vLLM,
//!   compressed-tensors and Quark clamp before the cast).
//! - MXFP4 elements are OCP E2M1 (`0, 0.5, 1, 1.5, 2, 3, 4, 6`), rounded to nearest with ties to
//!   even and saturated to ±6; the E8M0 shared exponent of a 32-element group follows AMD Quark's
//!   `scale_calculation_mode: even` (amax rounded up to the next power of two when its mantissa
//!   is ≥ 1.75, then `floor(log2) − 2`, clamped to ±127; an all-zero group gets 2^−127), which is
//!   what the Quark W4A4 checkpoints were produced and are meant to be run with.
//! - Dynamic FP8 activation scales are `max(amax / 448, 1 / (448 × 512))` (vLLM's floor keeps a
//!   zero row finite); values are divided by the scale in F32 before rounding.
use crate::quant::{ActQuantDesc, MX_BLOCK, QuantSchemeDesc};

/// Largest finite FP8 e4m3 magnitude.
pub const FP8_E4M3_MAX: f32 = 448.0;
/// Smallest dynamic FP8 scale (vLLM's `1 / (448 × 512)`).
pub const FP8_MIN_SCALE: f32 = 1.0 / (FP8_E4M3_MAX * 512.0);
/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;

const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// OCP e4m3fn bits of `x`: round to nearest, ties to even, saturated to ±448; NaN → `0x7f`.
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

/// The value of OCP e4m3fn bits (NaN for `0x7f` / `0xff`).
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

/// The value of an E2M1 code (bit 3 = sign).
pub fn e2m1_value(code: u8) -> f32 {
    let v = E2M1_VALUES[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// E2M1 code of `x`: nearest value, ties to the even code, saturated to ±6; NaN → 0.
pub fn e2m1_round(x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    let sign: u8 = if x.is_sign_negative() { 8 } else { 0 };
    let a = x.abs().min(E2M1_MAX);
    let mut best = 0u8;
    for c in 1..8u8 {
        let d = (E2M1_VALUES[usize::from(c)] - a).abs();
        let bd = (E2M1_VALUES[usize::from(best)] - a).abs();
        if d < bd || (d == bd && c & 1 == 0) {
            best = c;
        }
    }
    if best == 0 {
        0 // no negative zero code: the sign of a value that rounds to 0 is dropped
    } else {
        sign | best
    }
}

/// `2^(e − 127)`; NaN for 255.
pub fn e8m0_value(e: u8) -> f32 {
    if e == 255 {
        return f32::NAN;
    }
    2f32.powi(i32::from(e) - 127)
}

/// The E8M0 exponent Quark's `even` mode gives a group whose largest magnitude is `amax`.
pub fn mxfp4_scale_even(amax: f32) -> u8 {
    if amax.is_nan() {
        return 255;
    }
    // Round the mantissa up to the next power of two when it is ≥ 1.75 (mbits = 1 for E2M1):
    // add 2^(23 − 1 − 1) to the F32 bits, then keep sign and exponent.
    let rounded = f32::from_bits((amax.to_bits().wrapping_add(1 << 21)) & 0xff80_0000);
    let log2 = if rounded.is_infinite() {
        i32::from(i16::MAX) // rounded past f32::MAX: Quark's log2 is +inf, clamped below
    } else if rounded == 0.0 || !rounded.is_normal() {
        -127 // zero or subnormal: the smallest scale
    } else {
        ((rounded.to_bits() >> 23) & 0xff) as i32 - 127
    };
    let e = (log2 - 2).clamp(-127, 127);
    (e + 127) as u8
}

/// Quantizes one 32-element group to MXFP4: packed codes (low nibble = even element) and the
/// E8M0 exponent (Quark `even` rule).
pub fn mxfp4_quantize_group(x: &[f32; MX_BLOCK]) -> ([u8; MX_BLOCK / 2], u8) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let e = mxfp4_scale_even(amax);
    let scale = e8m0_value(e);
    let mut packed = [0u8; MX_BLOCK / 2];
    for (i, pair) in x.chunks_exact(2).enumerate() {
        let lo = e2m1_round(pair[0] / scale);
        let hi = e2m1_round(pair[1] / scale);
        packed[i] = lo | (hi << 4);
    }
    (packed, e)
}

fn nibble(data: &[u8], idx: usize) -> u8 {
    let b = data[idx / 2];
    if idx.is_multiple_of(2) {
        b & 0xf
    } else {
        b >> 4
    }
}

/// Dequantizes an `n × k` layer of `scheme` to row-major F32. `scales` holds F32 scales for the
/// FP8 and INT4 schemes and the E8M0 exponents as `f32::from(byte)` for MXFP4 (use
/// [`mxfp4_scales_as_f32`]); `zeros` holds INT4 zero points (0..=15) for `Int4GroupZp`.
pub fn dequantize(
    scheme: QuantSchemeDesc,
    data: &[u8],
    scales: &[f32],
    zeros: Option<&[u8]>,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(data.len(), scheme.data_bytes(n, k), "data bytes");
    assert_eq!(scales.len(), scheme.scale_count(n, k), "scale count");
    let mut out = vec![0f32; n * k];
    match scheme {
        QuantSchemeDesc::Fp8Tensor | QuantSchemeDesc::Fp8Channel => {
            for r in 0..n {
                let s = if scheme == QuantSchemeDesc::Fp8Tensor {
                    scales[0]
                } else {
                    scales[r]
                };
                for c in 0..k {
                    out[r * k + c] = fp8_e4m3_value(data[r * k + c]) * s;
                }
            }
        }
        QuantSchemeDesc::Fp8Block { block_n, block_k } => {
            let (bn, bk) = (block_n as usize, block_k as usize);
            let blocks_k = k.div_ceil(bk);
            for r in 0..n {
                for c in 0..k {
                    let s = scales[(r / bn) * blocks_k + c / bk];
                    out[r * k + c] = fp8_e4m3_value(data[r * k + c]) * s;
                }
            }
        }
        QuantSchemeDesc::Int4GroupZp { group } | QuantSchemeDesc::Int4GroupSym { group } => {
            let g = group as usize;
            let groups = k.div_ceil(g);
            let zeros = match scheme {
                QuantSchemeDesc::Int4GroupZp { .. } => {
                    let z = zeros.expect("Int4GroupZp needs zero points");
                    assert_eq!(z.len(), n * groups, "zero-point count");
                    Some(z)
                }
                _ => None,
            };
            for r in 0..n {
                for c in 0..k {
                    let q = f32::from(nibble(data, r * k + c));
                    let gi = r * groups + c / g;
                    let z = zeros.map_or(8.0, |z| f32::from(z[gi]));
                    out[r * k + c] = (q - z) * scales[gi];
                }
            }
        }
        QuantSchemeDesc::Mxfp4 => {
            let groups = k.div_ceil(MX_BLOCK);
            for r in 0..n {
                for c in 0..k {
                    let e = scales[r * groups + c / MX_BLOCK] as u8;
                    out[r * k + c] = e2m1_value(nibble(data, r * k + c)) * e8m0_value(e);
                }
            }
        }
    }
    out
}

/// E8M0 bytes as the `scales` argument of [`dequantize`] for MXFP4.
pub fn mxfp4_scales_as_f32(e8m0: &[u8]) -> Vec<f32> {
    e8m0.iter().map(|&e| f32::from(e)).collect()
}

fn fp8_qdq(x: f32, scale: f32) -> f32 {
    fp8_e4m3_value(fp8_e4m3_round(x / scale)) * scale
}

fn dynamic_fp8_scale(values: &[f32]) -> f32 {
    let amax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    (amax / FP8_E4M3_MAX).max(FP8_MIN_SCALE)
}

/// Quantizes a row-major `rows × cols` activation matrix to FP8 e4m3 bytes for an FP8 `mode`
/// and returns `(bytes, scales)`: the same scales and the same rounding as
/// [`quantize_dequantize_activations`], so `e4m3(byte) × scale` is its result exactly.
pub fn quantize_activations_fp8(
    x: &[f32],
    rows: usize,
    cols: usize,
    mode: ActQuantDesc,
    static_scale: f32,
) -> (Vec<u8>, Vec<f32>) {
    assert!(mode.is_fp8(), "FP8 activation mode");
    assert_eq!(x.len(), rows * cols, "activation shape");
    let mut bytes = vec![0u8; x.len()];
    let mut scales = Vec::with_capacity(mode.scale_count(rows, cols));
    let group = match mode {
        ActQuantDesc::Fp8Group { group } => group as usize,
        _ => cols,
    };
    if mode == ActQuantDesc::Fp8Tensor {
        scales.push(static_scale);
    }
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        for (gi, grp) in row.chunks(group).enumerate() {
            let s = if mode == ActQuantDesc::Fp8Tensor {
                static_scale
            } else {
                let s = dynamic_fp8_scale(grp);
                scales.push(s);
                s
            };
            for (i, v) in grp.iter().enumerate() {
                bytes[r * cols + gi * group + i] = fp8_e4m3_round(v / s);
            }
        }
    }
    (bytes, scales)
}

/// Quantize-dequantizes a row-major `rows × cols` activation matrix in place as `mode`
/// prescribes and returns the scales used (row-major per row / per group; for MXFP4 the group
/// scales as powers of two; empty for [`ActQuantDesc::None`]). `static_scale` is the
/// checkpoint's `input_scale` of [`ActQuantDesc::Fp8Tensor`] (ignored by the other modes).
pub fn quantize_dequantize_activations(
    x: &mut [f32],
    rows: usize,
    cols: usize,
    mode: ActQuantDesc,
    static_scale: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols, "activation shape");
    match mode {
        ActQuantDesc::None => Vec::new(),
        ActQuantDesc::Fp8Tensor => {
            for v in x.iter_mut() {
                *v = fp8_qdq(*v, static_scale);
            }
            vec![static_scale]
        }
        ActQuantDesc::Fp8Token => {
            let mut scales = Vec::with_capacity(rows);
            for row in x.chunks_exact_mut(cols) {
                let s = dynamic_fp8_scale(row);
                for v in row.iter_mut() {
                    *v = fp8_qdq(*v, s);
                }
                scales.push(s);
            }
            scales
        }
        ActQuantDesc::Fp8Group { group } => {
            let mut scales = Vec::new();
            for row in x.chunks_exact_mut(cols) {
                for grp in row.chunks_mut(group as usize) {
                    let s = dynamic_fp8_scale(grp);
                    for v in grp.iter_mut() {
                        *v = fp8_qdq(*v, s);
                    }
                    scales.push(s);
                }
            }
            scales
        }
        ActQuantDesc::Mxfp4Emulated => {
            let mut scales = Vec::new();
            for row in x.chunks_exact_mut(cols) {
                for grp in row.chunks_mut(MX_BLOCK) {
                    let amax = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let s = e8m0_value(mxfp4_scale_even(amax));
                    for v in grp.iter_mut() {
                        *v = e2m1_value(e2m1_round(*v / s)) * s;
                    }
                    scales.push(s);
                }
            }
            scales
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OCP e4m3fn by hand: exact codes, ties to even, subnormals, saturation, NaN, and every
    /// finite code round-trips. Breaks if the rounding is away-from-zero, truncating or
    /// overflows to NaN instead of saturating.
    #[test]
    fn fp8_rounding_table() {
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
            (2f32.powi(-10), 0x00), // tie between 0 and the smallest subnormal → 0 (even)
            (3.0 * 2f32.powi(-10), 0x02), // tie 1.5 → 2
            (1.0625, 0x38),         // tie between 1.0 (m=0) and 1.125 (m=1) → 1.0
            (1.1875, 0x3a),         // tie between 1.125 (m=1) and 1.25 (m=2) → 1.25
            (2f32.powi(-6), 0x08),  // smallest normal
            (15.5 * 2f32.powi(-10), 0x08), // subnormal 7.75 ulp rounds up into the normal range
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
            let back = fp8_e4m3_round(v);
            assert_eq!(fp8_e4m3_value(back), v, "code {code:#04x}");
        }
    }

    /// Every E2M1 code's value, and rounding with ties to even and saturation, as Quark's
    /// kernel documents it (0.25 → 0, 0.75 → 1, 1.25 → 1, 1.75 → 2, 2.5 → 2, 3.5 → 4, 5 → 4).
    #[test]
    fn e2m1_every_code() {
        let values = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (code, v) in values.iter().enumerate() {
            assert_eq!(e2m1_value(code as u8), *v, "code {code}");
            if code != 8 {
                assert_eq!(e2m1_round(*v), code as u8, "{v}");
            }
        }
        for (x, expect) in [
            (0.25, 0.0),
            (0.26, 0.5),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.01, 6.0),
            (100.0, 6.0),
            (-2.5, -2.0),
            (-0.2, 0.0),
        ] {
            assert_eq!(e2m1_value(e2m1_round(x)), expect, "{x}");
        }
    }

    /// Quark `even` exponents: amax mantissa ≥ 1.75 rounds up to the next power of two, then
    /// `floor(log2) − 2`; an all-zero group gets 2^−127; a group scaled by it round-trips its
    /// E2M1 grid exactly. Breaks if the OCP floor rule or a ceil rule is used instead.
    #[test]
    fn mxfp4_group_scale() {
        assert_eq!(mxfp4_scale_even(6.0), 127); // 6 = 1.5 × 2^2 → 2^2 → e = 0
        assert_eq!(mxfp4_scale_even(7.0), 128); // 1.75 × 2^2 rounds to 2^3 → e = 1
        assert_eq!(mxfp4_scale_even(6.99), 127);
        assert_eq!(mxfp4_scale_even(4.0), 127);
        assert_eq!(mxfp4_scale_even(1.0), 125);
        assert_eq!(mxfp4_scale_even(0.0), 0);
        assert_eq!(mxfp4_scale_even(f32::NAN), 255);
        assert_eq!(mxfp4_scale_even(f32::MAX), 254); // clamp at 2^127
        assert_eq!(e8m0_value(127), 1.0);
        assert!(e8m0_value(255).is_nan());

        // A group whose values are exactly scale × E2M1 values round-trips.
        let scale = 2f32.powi(-3);
        let mut x = [0f32; MX_BLOCK];
        for (i, v) in x.iter_mut().enumerate() {
            *v = e2m1_value((i % 16) as u8) * scale;
        }
        let (packed, e) = mxfp4_quantize_group(&x);
        assert_eq!(e8m0_value(e), scale);
        let deq = dequantize(
            QuantSchemeDesc::Mxfp4,
            &packed,
            &mxfp4_scales_as_f32(&[e]),
            None,
            1,
            MX_BLOCK,
        );
        for (a, b) in x.iter().zip(&deq) {
            assert_eq!(a.abs(), b.abs()); // -0.0 codes come back as 0.0
        }
        // An all-zero group.
        let (packed, e) = mxfp4_quantize_group(&[0.0; MX_BLOCK]);
        assert_eq!((packed, e), ([0u8; 16], 0));
    }

    /// INT4 groups: low nibble = even column, zero points per group (AWQ) or 8 (GPTQ sym).
    #[test]
    fn int4_group_dequant() {
        // 2 rows × 8 columns, group 4: row 0 codes 0..7, row 1 codes 15..8.
        let codes: Vec<u8> = (0u8..8).chain((8u8..16).rev()).collect();
        let data: Vec<u8> = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
        let scales = [0.5, 2.0, 1.0, 0.25];
        let zeros = [1u8, 2, 15, 0];
        let zp = dequantize(
            QuantSchemeDesc::Int4GroupZp { group: 4 },
            &data,
            &scales,
            Some(&zeros),
            2,
            8,
        );
        assert_eq!(
            zp,
            [
                -0.5, 0.0, 0.5, 1.0, 4.0, 6.0, 8.0, 10.0, 0.0, -1.0, -2.0, -3.0, 2.75, 2.5, 2.25,
                2.0
            ]
        );
        let sym = dequantize(
            QuantSchemeDesc::Int4GroupSym { group: 4 },
            &data,
            &scales,
            None,
            2,
            8,
        );
        assert_eq!(sym[0], -4.0); // (0 − 8) × 0.5
        assert_eq!(sym[8], 7.0); // (15 − 8) × 1.0
        assert_eq!(sym[15], 0.0); // (8 − 8) × 0.25
    }

    /// FP8 block scales index by (row / block_n, column / block_k), ragged edges included.
    #[test]
    fn fp8_block_dequant() {
        let (n, k) = (3, 5);
        let data = vec![fp8_e4m3_round(1.0); n * k];
        let scheme = QuantSchemeDesc::Fp8Block {
            block_n: 2,
            block_k: 2,
        };
        assert_eq!(scheme.scale_count(n, k), 6);
        let scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = dequantize(scheme, &data, &scales, None, n, k);
        assert_eq!(&out[0..5], &[1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(&out[5..10], &[1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(&out[10..15], &[4.0, 4.0, 5.0, 5.0, 6.0]);
        let ch = dequantize(
            QuantSchemeDesc::Fp8Channel,
            &data,
            &[1.0, 0.5, 0.25],
            None,
            n,
            k,
        );
        assert_eq!(ch[k], 0.5);
        assert_eq!(ch[2 * k + 4], 0.25);
    }

    /// Activation modes: static tensor scale, dynamic per row (amax → 448) with the zero-row
    /// floor, per group of columns, and MXFP4 emulation (idempotent, BF16-exact values).
    #[test]
    fn act_quant_modes() {
        let mut x = vec![1.0, -448.0, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0];
        let s = quantize_dequantize_activations(&mut x, 2, 4, ActQuantDesc::Fp8Token, 1.0);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], FP8_MIN_SCALE);
        assert_eq!(&x[..3], &[1.0, -448.0, 3.0]);
        assert_eq!(x[3], fp8_e4m3_value(fp8_e4m3_round(0.1)));
        assert_eq!(&x[4..], &[0.0; 4]);

        let mut x = vec![2.0, 1000.0];
        let s = quantize_dequantize_activations(&mut x, 1, 2, ActQuantDesc::Fp8Tensor, 2.0);
        assert_eq!(s, [2.0]);
        assert_eq!(x, [2.0, 896.0]); // 1000 / 2 = 500 saturates to 448

        let mut x = vec![448.0, 1.0, 4.0, 2.0];
        let s =
            quantize_dequantize_activations(&mut x, 1, 4, ActQuantDesc::Fp8Group { group: 2 }, 1.0);
        assert_eq!(s, [1.0, 4.0 / 448.0]);
        assert_eq!(x[0], 448.0);

        let mut x: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.37).collect();
        let s = quantize_dequantize_activations(&mut x, 1, 64, ActQuantDesc::Mxfp4Emulated, 1.0);
        assert_eq!(s.len(), 2);
        let once = x.clone();
        quantize_dequantize_activations(&mut x, 1, 64, ActQuantDesc::Mxfp4Emulated, 1.0);
        assert_eq!(x, once, "MXFP4 quantize-dequantize is idempotent");
        for v in &once {
            assert_eq!(half::bf16::from_f32(*v).to_f32(), *v, "{v} is BF16-exact");
        }

        let mut x = vec![1.23, 4.56];
        assert!(quantize_dequantize_activations(&mut x, 1, 2, ActQuantDesc::None, 1.0).is_empty());
        assert_eq!(x, [1.23, 4.56]);
    }
}
