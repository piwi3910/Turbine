//! Host-side token sampling from FP32 logits (P1 S-9): greedy argmax at temperature 0,
//! otherwise temperature → top-k → top-p over a seeded ChaCha8 stream, so an identical `seed`
//! gives identical tokens.
use std::cmp::Ordering;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use turbine_core::request::SamplingParams;

/// One sampled token with its log-probability under the raw (temperature 1, untruncated)
/// distribution and, when requested, the most likely alternatives (highest first, ties by id).
#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub token: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Samples one token per step for one request.
///
/// Order of operations: temperature, then top-k, then top-p over the candidates sorted
/// descending (ties by the lower id), then one 24-bit uniform draw. Temperature 0 is greedy.
#[derive(Clone, Debug)]
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    /// `None` = disabled (request `top_k` −1 or 0).
    top_k: Option<usize>,
    /// Alternatives reported per token (0 = none).
    top_logprobs: usize,
    rng: ChaCha8Rng,
}

/// Index of the largest value; ties go to the lower id and NaN never wins.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_value {
            best = i;
            best_value = v;
        }
    }
    best as u32
}

/// `log(sum(exp(x)))` computed around the maximum (NaN entries ignored).
fn log_sum_exp(logits: &[f32]) -> f32 {
    let max = logits
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = logits
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&v| f64::from(v - max).exp())
        .sum();
    max + sum.ln() as f32
}

/// The log-softmax of `logits`.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let lse = log_sum_exp(logits);
    logits.iter().map(|&v| v - lse).collect()
}

/// Descending by value, ties by the lower id; NaN sorts last.
fn by_value_desc(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)),
    }
}

/// The `n` largest `(id, value)` pairs, highest first, ties by id.
fn top_n(values: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    let n = n.min(pairs.len());
    if n == 0 {
        return Vec::new();
    }
    if n < pairs.len() {
        pairs.select_nth_unstable_by(n - 1, by_value_desc);
        pairs.truncate(n);
    }
    pairs.sort_unstable_by(by_value_desc);
    pairs
}

/// A uniform draw in `[0, 1)` from 24 random bits (exact in f32).
fn uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

impl Sampler {
    /// A sampler for `params`: ChaCha8 seeded from `seed`, else from OS entropy.
    pub fn new(params: &SamplingParams) -> Sampler {
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_os_rng(),
        };
        Sampler {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: usize::try_from(params.top_k).ok().filter(|&k| k > 0),
            top_logprobs: params.logprobs.map_or(0, |n| n as usize),
            rng,
        }
    }

    /// Picks the next token from one vocabulary row. The reported logprobs are the log-softmax
    /// of `logits` as passed in (before temperature). `logits` is mutable so Phase 2 can apply
    /// its token mask in place; Phase 1 leaves it unchanged.
    pub fn sample(&mut self, logits: &mut [f32]) -> SampledToken {
        let lse = log_sum_exp(logits);
        let top_logprobs = if self.top_logprobs > 0 {
            top_n(logits, self.top_logprobs)
                .into_iter()
                .map(|(id, v)| (id, v - lse))
                .collect()
        } else {
            Vec::new()
        };
        let token = if self.temperature <= 0.0 {
            argmax(logits)
        } else {
            self.draw(logits)
        };
        SampledToken {
            token,
            logprob: logits[token as usize] - lse,
            top_logprobs,
        }
    }

    /// Temperature → top-k → top-p, then one draw over the kept candidates.
    fn draw(&mut self, logits: &[f32]) -> u32 {
        let inv_t = 1.0 / self.temperature;
        let vocab = logits.len();
        let k = self.top_k.map_or(vocab, |k| k.min(vocab));
        let truncate = k < vocab || self.top_p < 1.0;
        // Candidates: sorted descending (ties by id) when anything is cut, else in id order.
        // Scaling by 1/T keeps the order, so the top-k cut is taken on the raw logits.
        let mut candidates: Vec<(u32, f32)> = if truncate {
            top_n(logits, k)
        } else {
            logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect()
        };
        for c in &mut candidates {
            c.1 *= inv_t;
        }
        let max = candidates
            .iter()
            .map(|c| c.1)
            .filter(|v| !v.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return argmax(logits);
        }
        let mut weights: Vec<f64> = candidates
            .iter()
            .map(|c| {
                if c.1.is_nan() {
                    0.0
                } else {
                    f64::from(c.1 - max).exp()
                }
            })
            .collect();
        let mut total: f64 = weights.iter().sum();
        if self.top_p < 1.0 {
            // Smallest prefix whose mass reaches top_p (at least one token).
            let target = f64::from(self.top_p) * total;
            let mut cum = 0.0;
            let mut keep = weights.len();
            for (i, w) in weights.iter().enumerate() {
                cum += w;
                if cum >= target {
                    keep = i + 1;
                    break;
                }
            }
            weights.truncate(keep);
            total = weights.iter().sum();
        }
        let u = f64::from(uniform(&mut self.rng)) * total;
        let mut cum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cum += w;
            if u < cum {
                return candidates[i].0;
            }
        }
        // Rounding left u at the total: the last kept candidate with non-zero weight.
        let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
        candidates[last].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f32, top_p: f32, top_k: i32, seed: u64) -> SamplingParams {
        SamplingParams {
            temperature,
            top_p,
            top_k,
            seed: Some(seed),
            logprobs: None,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn argmax_ties_to_lower_id_and_skips_nan() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 0.5, 0.5]), 1);
        assert_eq!(argmax(&[-2.0, -1.0, f32::NAN]), 1);
    }

    #[test]
    fn log_softmax_normalises() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, 1000.0]);
        let total: f64 = lp.iter().map(|&v| f64::from(v).exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "sum {total}");
        assert!(lp[3].abs() < 1e-6);
        assert!((lp[1] - lp[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn greedy_reports_raw_logprobs_and_top_alternatives() {
        let raw = vec![0.5, 2.0, 2.0, -1.0, 1.5];
        let mut p = params(0.0, 0.1, 3, 1);
        p.logprobs = Some(3);
        let mut sampler = Sampler::new(&p);
        let mut logits = raw.clone();
        let s = sampler.sample(&mut logits);
        let lp = log_softmax(&raw);
        assert_eq!(s.token, 1, "greedy ties to the lower id");
        assert!((s.logprob - lp[1]).abs() < 1e-6);
        let ids: Vec<u32> = s.top_logprobs.iter().map(|t| t.0).collect();
        assert_eq!(ids, vec![1, 2, 4]);
        assert!((s.top_logprobs[2].1 - lp[4]).abs() < 1e-6);
        assert_eq!(logits, raw, "Phase 1 leaves the logits unchanged");
    }

    #[test]
    fn logprob_is_untempered() {
        let raw = vec![1.0, 2.0, 3.0];
        let mut sampler = Sampler::new(&params(0.5, 1.0, -1, 3));
        let s = sampler.sample(&mut raw.clone());
        let lp = log_softmax(&raw);
        assert!((s.logprob - lp[s.token as usize]).abs() < 1e-6);
        assert!(s.top_logprobs.is_empty(), "no logprobs requested");
    }

    #[test]
    fn top_k_one_and_tiny_top_p_are_argmax() {
        let raw: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 * 0.1).collect();
        let best = argmax(&raw);
        for seed in 0..20 {
            let mut k1 = Sampler::new(&params(1.5, 1.0, 1, seed));
            assert_eq!(k1.sample(&mut raw.clone()).token, best);
            let mut p0 = Sampler::new(&params(1.5, 1e-6, -1, seed));
            assert_eq!(p0.sample(&mut raw.clone()).token, best);
        }
    }

    #[test]
    fn top_k_and_top_p_restrict_the_support() {
        // probabilities ≈ [0.64, 0.24, 0.09, 0.03]
        let raw = vec![3.0f32, 2.0, 1.0, 0.0];
        let mut k2 = Sampler::new(&params(1.0, 1.0, 2, 11));
        let mut p = Sampler::new(&params(1.0, 0.8, -1, 12));
        let mut seen_k = [0usize; 4];
        let mut seen_p = [0usize; 4];
        for _ in 0..2000 {
            seen_k[k2.sample(&mut raw.clone()).token as usize] += 1;
            seen_p[p.sample(&mut raw.clone()).token as usize] += 1;
        }
        assert_eq!(
            &seen_k[2..],
            &[0, 0],
            "top_k 2 keeps ids 0 and 1: {seen_k:?}"
        );
        assert!(seen_k[1] > 0);
        // 0.64 < 0.8 ≤ 0.64 + 0.24: top_p 0.8 keeps ids 0 and 1.
        assert_eq!(
            &seen_p[2..],
            &[0, 0],
            "top_p 0.8 keeps ids 0 and 1: {seen_p:?}"
        );
        assert!(seen_p[1] > 0);
    }

    #[test]
    fn temperature_sampling_follows_the_distribution() {
        // weights 1:1:2:3
        let raw = vec![0.0f32, 0.0, 2.0f32.ln(), 3.0f32.ln()];
        let mut s = Sampler::new(&params(1.0, 1.0, -1, 5));
        let n = 14_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            counts[s.sample(&mut raw.clone()).token as usize] += 1;
        }
        let expected = [1.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0];
        for (c, e) in counts.iter().zip(expected) {
            let f = *c as f64 / f64::from(n);
            assert!((f - e).abs() < 0.02, "{counts:?}");
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let raw: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let run = |seed: u64| -> Vec<u32> {
            let mut s = Sampler::new(&params(0.8, 0.9, 50, seed));
            (0..32).map(|_| s.sample(&mut raw.clone()).token).collect()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }
}
