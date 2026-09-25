//! Rotary inverse frequencies, computed on the host in FP64 and stored as FP32, so the Llama-3
//! scaled bands match the transformers formula (`_compute_llama3_parameters`) exactly.
use crate::config::RopeScaling;

/// `inv_freq[i] = theta^(-2i / rotary_dim)` for `i < rotary_dim / 2`, then, with Llama-3
/// scaling, by wavelength `2π / inv_freq`: below `original / high_freq_factor` kept, above
/// `original / low_freq_factor` divided by `factor`, and in between interpolated with
/// `smooth = (original / wavelength − low) / (high − low)`.
pub fn inv_freq(theta: f64, rotary_dim: u32, scaling: Option<&RopeScaling>) -> Vec<f32> {
    let dim = f64::from(rotary_dim);
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
                    let wavelen = 2.0 * std::f64::consts::PI / base;
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
                let wavelen = 2.0 * std::f64::consts::PI / f64::from(plain[i]);
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
}
