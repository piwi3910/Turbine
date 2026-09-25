//! KV exhaustion horizon (P3 S-8): seconds until the committed growth of the running
//! sequences consumes the free KV blocks.

/// Predictor of the `exhaustion_horizon` signal.
#[derive(Clone, Copy, Debug)]
pub struct ExhaustionHorizon;

impl ExhaustionHorizon {
    /// `running[i]` is sequence i's remaining tokens up to its `max_tokens`; each sequence
    /// grows at `decode_tokens_per_s` until it reaches that bound. Solves
    /// Σ min(rᵢ, rate·t) = free tokens over the sorted remaining lengths (piecewise linear).
    /// `+∞` when nothing grows or the whole committed growth fits.
    pub fn predict(
        running: &[u32],
        decode_tokens_per_s: f64,
        free_blocks: u32,
        block_tokens: u32,
    ) -> f64 {
        let free_tokens = f64::from(free_blocks) * f64::from(block_tokens);
        let total: f64 = running.iter().map(|&r| f64::from(r)).sum();
        // NaN and non-positive rates mean nothing is growing.
        let growing_rate = decode_tokens_per_s.is_finite() && decode_tokens_per_s > 0.0;
        if running.is_empty() || !growing_rate || total <= free_tokens {
            return f64::INFINITY;
        }
        let mut remaining: Vec<u32> = running.to_vec();
        remaining.sort_unstable();
        let (mut t, mut consumed) = (0.0, 0.0);
        let mut growing = remaining.len() as f64;
        for r in remaining {
            // Until sequence i stops at t_r, `growing` sequences each add `rate` tokens/s.
            let t_r = f64::from(r) / decode_tokens_per_s;
            let grows = (t_r - t) * decode_tokens_per_s * growing;
            if consumed + grows >= free_tokens {
                return t + (free_tokens - consumed) / (decode_tokens_per_s * growing);
            }
            consumed += grows;
            t = t_r;
            growing -= 1.0;
        }
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_exhaustion() {
        // 20 sequences × 500 remaining tokens at 50 tokens/s against 200 free 16-token blocks = 3,200 tokens.
        let h = ExhaustionHorizon::predict(&[500; 20], 50.0, 200, 16);
        assert!((h - 3.2).abs() <= 0.32, "horizon {h}");
        assert_eq!(
            ExhaustionHorizon::predict(&[], 50.0, 200, 16),
            f64::INFINITY
        );
        // Committed growth that fits the free blocks never exhausts.
        assert_eq!(
            ExhaustionHorizon::predict(&[100; 20], 50.0, 200, 16),
            f64::INFINITY
        );
        // Short sequences finish first and stop growing: 10 × 10 tokens + 10 × 1000 against 3,200 tokens.
        let mixed: Vec<u32> = [10u32; 10].into_iter().chain([1000; 10]).collect();
        let h = ExhaustionHorizon::predict(&mixed, 50.0, 200, 16);
        assert!((h - 6.2).abs() < 1e-9, "horizon {h}");
        // No decode progress means no growth.
        assert_eq!(
            ExhaustionHorizon::predict(&[500; 20], 0.0, 200, 16),
            f64::INFINITY
        );
        assert_eq!(
            ExhaustionHorizon::predict(&[500; 20], f64::NAN, 200, 16),
            f64::INFINITY
        );
        // No free blocks: exhausted now.
        assert_eq!(ExhaustionHorizon::predict(&[500; 20], 50.0, 0, 16), 0.0);
    }
}
