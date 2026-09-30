//! Rotary inverse frequencies, computed on the host in FP64 and stored as FP32, so the Llama-3
//! scaled bands match the transformers formula (`_compute_llama3_parameters`) exactly and the
//! YaRN blend (`_compute_yarn_parameters`, transformers 4.57.1) to within 2 FP32 ulps
//! (transformers evaluates it in FP32 with a 1-ulp `powf`; P6a S-15). YaRN's attention factor
//! is not applied here: the rope op multiplies cos and sin by it before rounding
//! (`ModelArchConfig::rope_attention_factor`, `RopeContext::attn_factor`, kernel ABI v2.10;
//! user decision 2026-09-29, superseding Q19's fold into the softmax scale).
use std::f64::consts::PI;

use crate::config::RopeScaling;

/// transformers' default YaRN attention factor (`get_mscale` in `_compute_yarn_parameters`):
/// `mscale(factor, mscale) / mscale(factor, mscale_all_dim)` when both are given and non-zero,
/// else `mscale(factor, 1)`, with `mscale(s, m) = 0.1 · m · ln(s) + 1` for `s > 1` and `1`
/// otherwise. An explicit `attention_factor` in the config replaces this (the caller's rule).
pub fn yarn_attention_factor(factor: f64, mscale: Option<f64>, mscale_all_dim: Option<f64>) -> f64 {
    let get = |scale: f64, m: f64| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * m * scale.ln() + 1.0
        }
    };
    match (mscale, mscale_all_dim) {
        // Python truthiness: `if mscale and mscale_all_dim` skips zeros too.
        (Some(m), Some(all)) if m != 0.0 && all != 0.0 => get(factor, m) / get(factor, all),
        _ => get(factor, 1.0),
    }
}

/// YaRN's ramp bounds (`find_correction_range`): the dimensions whose wavelength completes
/// `beta_fast` resp. `beta_slow` rotations over the original context, floored / ceiled when
/// `truncate`, clamped to `0..=dim − 1`.
fn yarn_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: f64,
    theta: f64,
    original: f64,
    truncate: bool,
) -> (f64, f64) {
    let correction_dim =
        |rotations: f64| (dim * (original / (rotations * 2.0 * PI)).ln()) / (2.0 * theta.ln());
    let (mut low, mut high) = (correction_dim(beta_fast), correction_dim(beta_slow));
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    (low.max(0.0), high.min(dim - 1.0))
}

/// `inv_freq[i] = theta^(-2i / rotary_dim)` for `i < rotary_dim / 2`, then, with Llama-3
/// scaling, by wavelength `2π / inv_freq`: below `original / high_freq_factor` kept, above
/// `original / low_freq_factor` divided by `factor`, and in between interpolated with
/// `smooth = (original / wavelength − low) / (high − low)`. With YaRN, the blend
/// `interpolation · (1 − e) + extrapolation · e` of `base / factor` and `base`, where the
/// extrapolation weight `e = 1 − clamp((i − low) / (high − low), 0, 1)` ramps between the
/// correction-range bounds ([`yarn_correction_range`]; `high` nudged by 0.001 when equal).
pub fn inv_freq(theta: f64, rotary_dim: u32, scaling: Option<&RopeScaling>) -> Vec<f32> {
    let dim = f64::from(rotary_dim);
    let yarn_range = match scaling {
        Some(&RopeScaling::Yarn {
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            truncate,
            ..
        }) => {
            let (low, high) = yarn_correction_range(
                beta_fast,
                beta_slow,
                dim,
                theta,
                f64::from(original_max_position_embeddings),
                truncate,
            );
            // transformers' `linear_ramp_factor` guard against a zero-width ramp.
            Some((low, if low == high { high + 0.001 } else { high }))
        }
        _ => None,
    };
    (0..rotary_dim / 2)
        .map(|i| {
            let base = 1.0 / theta.powf(f64::from(2 * i) / dim);
            let scaled = match scaling {
                None => base,
                Some(&RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                }) => {
                    let original = f64::from(original_max_position_embeddings);
                    let wavelen = 2.0 * PI / base;
                    if wavelen < original / high_freq_factor {
                        base
                    } else if wavelen > original / low_freq_factor {
                        base / factor
                    } else {
                        let smooth = (original / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * base / factor + smooth * base
                    }
                }
                Some(&RopeScaling::Yarn { factor, .. }) => {
                    let (low, high) = yarn_range.expect("computed for YaRN above");
                    let ramp = ((f64::from(i) - low) / (high - low)).clamp(0.0, 1.0);
                    let extrapolation = 1.0 - ramp;
                    (base / factor) * (1.0 - extrapolation) + base * extrapolation
                }
            };
            scaled as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_bands_follow_transformers() {
        let scaling = RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let plain = inv_freq(500_000.0, 128, None);
        let scaled = inv_freq(500_000.0, 128, Some(&scaling));
        assert_eq!(plain.len(), 64);
        assert_eq!(scaled.len(), 64);
        // Highest frequency (wavelength 2π, below 8192 / 4): unchanged.
        assert_eq!(plain[0], 1.0);
        assert_eq!(scaled[0], plain[0]);
        // Lowest frequency (wavelength above 8192 / 1): divided by the factor.
        let lowest = f64::from(plain[63]);
        assert!(
            (f64::from(scaled[63]) - lowest / 32.0).abs() <= 1e-6 * lowest / 32.0,
            "lowest {} scaled {}",
            plain[63],
            scaled[63]
        );
        // A medium band (wavelength between 2048 and 8192) is interpolated strictly between.
        let medium: Vec<usize> = (0..64)
            .filter(|&i| {
                let wavelen = 2.0 * PI / f64::from(plain[i]);
                wavelen > 2048.0 && wavelen < 8192.0
            })
            .collect();
        assert!(!medium.is_empty(), "no medium band for these parameters");
        for i in medium {
            assert!(
                scaled[i] < plain[i] && scaled[i] > plain[i] / 32.0,
                "band {i}: plain {} scaled {}",
                plain[i],
                scaled[i]
            );
        }
        // Without scaling the formula is theta^(-2i/d).
        assert_eq!(inv_freq(10_000.0, 4, None), [1.0, 0.01]);
    }

    fn yarn(factor: f64, original: u32, beta_fast: f64, beta_slow: f64) -> RopeScaling {
        RopeScaling::Yarn {
            factor,
            original_max_position_embeddings: original,
            beta_fast,
            beta_slow,
            attention_factor: yarn_attention_factor(factor, None, None),
            truncate: true,
        }
    }

    /// The YaRN blend: dimensions up to the ramp's start keep `base` (extrapolation), those
    /// from its end on are divided by the factor (interpolation), those inside lie strictly
    /// between; the ramp bounds are transformers' (18 and 35 for the Llama factor-16 override).
    #[test]
    fn yarn_bands_follow_transformers() {
        let plain = inv_freq(500_000.0, 128, None);
        let scaled = inv_freq(500_000.0, 128, Some(&yarn(16.0, 8192, 32.0, 1.0)));
        let (low, high) = yarn_correction_range(32.0, 1.0, 128.0, 500_000.0, 8192.0, true);
        assert_eq!((low, high), (18.0, 35.0));
        for i in 0..64 {
            let (p, s) = (f64::from(plain[i]), f64::from(scaled[i]));
            if i <= 18 {
                assert_eq!(scaled[i], plain[i], "band {i}");
            } else if i >= 35 {
                assert!((s - p / 16.0).abs() <= 1e-6 * p / 16.0, "band {i}: {p} {s}");
            } else {
                assert!(s < p && s > p / 16.0, "band {i}: {p} {s}");
            }
        }
        // `truncate: false` keeps the fractional bounds.
        let (low, high) = yarn_correction_range(32.0, 1.0, 64.0, 150_000.0, 4096.0, false);
        assert!(low.fract() != 0.0 && high.fract() != 0.0, "{low} {high}");
        // Equal bounds (beta_fast = beta_slow) must not divide by zero.
        let same = inv_freq(10_000.0, 64, Some(&yarn(4.0, 4096, 8.0, 8.0)));
        assert!(same.iter().all(|f| f.is_finite() && *f > 0.0));
        // Factor 1 is the plain table.
        let one = inv_freq(500_000.0, 128, Some(&yarn(1.0, 8192, 32.0, 1.0)));
        assert_eq!(one, plain);
    }

    /// transformers' `get_mscale` rule for the default attention factor.
    #[test]
    fn yarn_attention_factor_rule() {
        assert_eq!(
            yarn_attention_factor(16.0, None, None),
            0.1 * 16f64.ln() + 1.0
        );
        assert_eq!(yarn_attention_factor(1.0, None, None), 1.0);
        assert_eq!(yarn_attention_factor(0.5, Some(2.0), Some(1.0)), 1.0);
        let pair = yarn_attention_factor(40.0, Some(0.707), Some(1.0));
        assert_eq!(
            pair,
            (0.1 * 0.707 * 40f64.ln() + 1.0) / (0.1 * 40f64.ln() + 1.0)
        );
        // Equal mscales cancel; a zero or missing one falls back to the plain rule (Python
        // truthiness).
        assert_eq!(yarn_attention_factor(40.0, Some(0.707), Some(0.707)), 1.0);
        let plain = yarn_attention_factor(40.0, None, None);
        assert_eq!(yarn_attention_factor(40.0, Some(0.0), Some(1.0)), plain);
        assert_eq!(yarn_attention_factor(40.0, Some(1.0), None), plain);
    }

    /// Phase 6a Task 28a (spec S-15): the cpu-reference rope with YaRN's attention factor
    /// equals transformers bitwise on `tests/fixtures/yarn_rope.json`
    /// (`scripts/golden/yarn_params.py --rope-out`, the factor-16 Llama override): at eight
    /// positions up to 60,000, cos·m and sin·m rounded to BF16 (read back by rotating the unit
    /// pairs `(1, 0)`) and the rotated BF16 q (2 heads) and k (1 head) of
    /// `apply_rotary_pos_emb`. The table is transformers' own `inv_freq`, the factor
    /// transformers' default for factor 16 as Turbine resolves it. Breaks if the factor is
    /// folded into the softmax scale (cos/sin unscaled), applied after rounding, or rounded
    /// differently.
    #[test]
    fn yarn_cos_sin_times_factor_match_transformers() {
        use std::sync::Arc;

        use turbine_core::types::{DType, DeviceId};
        use turbine_kernels::ops::{RopeConfig, RopeContext};
        use turbine_tensor::host::HostMemory;
        use turbine_tensor::{DeviceMemory, Tensor};

        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yarn_rope.json");
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("json");
        let floats = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("number") as f32)
                .collect()
        };
        let bits = |v: &serde_json::Value| -> Vec<u16> {
            v.as_array()
                .expect("array")
                .iter()
                .map(|x| u16::try_from(x.as_u64().expect("bits")).expect("16 bits"))
                .collect()
        };
        let rows = |v: &serde_json::Value, f: &dyn Fn(&serde_json::Value) -> Vec<u16>| {
            v.as_array()
                .expect("rows")
                .iter()
                .flat_map(f)
                .collect::<Vec<u16>>()
        };

        // The factor as Turbine resolves it for the override equals transformers' in f32.
        let c = &fx["config"];
        let factor = c["rope_scaling"]["factor"].as_f64().unwrap();
        let m = yarn_attention_factor(factor, None, None) as f32;
        assert_eq!(m, fx["attention_factor"].as_f64().unwrap() as f32);

        let head_dim = c["head_dim"].as_u64().unwrap() as usize;
        let half = head_dim / 2;
        let inv_freq = floats(&fx["inv_freq"]);
        assert_eq!(inv_freq.len(), half);
        let positions: Vec<i32> = fx["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_i64().unwrap() as i32)
            .collect();
        let n = positions.len();
        let (hq, hk) = (
            fx["q_heads"].as_u64().unwrap() as usize,
            fx["k_heads"].as_u64().unwrap() as usize,
        );

        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 24);
        let upload = |shape: &[usize], dtype: DType, bytes: Vec<u8>| {
            let t = Tensor::empty(&mem, shape, dtype).expect("tensor");
            t.view().slice.write_bytes(&bytes).expect("upload");
            t
        };
        let bf16 = |v: &[u16]| v.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>();
        let read_bf16 = |t: &Tensor| -> Vec<u16> {
            t.view()
                .slice
                .read_bytes()
                .expect("read")
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let rope = turbine_kernels::cpu_reference_provider();
        let rope = rope.rope().expect("cpu rope");
        let run = |q: &[u16], k: &[u16], hq: usize, hk: usize| {
            let cfg = RopeConfig {
                num_q_heads: hq as u32,
                num_kv_heads: hk as u32,
                head_dim: head_dim as u32,
                rotary_dim: head_dim as u32,
                dtype: DType::BF16,
            };
            let q = upload(&[n, hq, head_dim], DType::BF16, bf16(q));
            let k = upload(&[n, hk, head_dim], DType::BF16, bf16(k));
            let pos = upload(
                &[n],
                DType::I32,
                positions.iter().flat_map(|p| p.to_le_bytes()).collect(),
            );
            let freq = upload(
                &[half],
                DType::F32,
                inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect(),
            );
            rope.execute(&mut RopeContext {
                cfg,
                q: q.view(),
                k: k.view(),
                positions: pos.view(),
                inv_freq: freq.view(),
                attn_factor: m,
            })
            .expect("rope");
            (read_bf16(&q), read_bf16(&k))
        };

        // cos·m and sin·m: rotating (1, 0) in every pair returns (c, s).
        const ONE: u16 = 0x3f80;
        let unit: Vec<u16> = (0..n)
            .flat_map(|_| (0..head_dim).map(|j| if j < half { ONE } else { 0 }))
            .collect();
        let (cs, _) = run(&unit, &unit, 1, 1);
        let want_cos = rows(&fx["cos_bf16"], &bits);
        let want_sin = rows(&fx["sin_bf16"], &bits);
        for t in 0..n {
            let row = &cs[t * head_dim..][..head_dim];
            assert_eq!(
                &row[..half],
                &want_cos[t * half..][..half],
                "cos·m at position {}",
                positions[t]
            );
            assert_eq!(
                &row[half..],
                &want_sin[t * half..][..half],
                "sin·m at position {}",
                positions[t]
            );
        }
        // The fixture's BF16 cos/sin are its FP32 ones rounded (transformers' `.to(bf16)`).
        let to_bf16 = |v: f32| (turbine_kernels::round_to(DType::BF16, v).to_bits() >> 16) as u16;
        let cos32: Vec<f32> = fx["cos_f32"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(floats)
            .collect();
        assert_eq!(
            cos32.iter().map(|&v| to_bf16(v)).collect::<Vec<_>>(),
            want_cos
        );

        // A full rotation of random BF16 q and k.
        let (q, k) = run(&rows(&fx["q"], &bits), &rows(&fx["k"], &bits), hq, hk);
        assert_eq!(q, rows(&fx["q_rot"], &bits), "rotated q");
        assert_eq!(k, rows(&fx["k_rot"], &bits), "rotated k");
    }
}
