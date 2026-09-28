//! Rotary inverse frequencies, computed on the host in FP64 and stored as FP32, so the Llama-3
//! scaled bands match the transformers formula (`_compute_llama3_parameters`) exactly and the
//! YaRN blend (`_compute_yarn_parameters`, transformers 4.57.1) to within 2 FP32 ulps
//! (transformers evaluates it in FP32 with a 1-ulp `powf`; P6a S-15). YaRN's attention factor
//! is not applied here: it folds into the attention softmax scale
//! (`ModelArchConfig::attention_scale`, user decision 2026-09-28, Q19), so the RoPE ABI and
//! kernels are unchanged.
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
}
