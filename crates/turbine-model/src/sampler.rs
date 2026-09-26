//! Host-side token sampling from FP32 logits (P1 S-9, P2 S-10/S-17): logit bias, penalties,
//! `min_tokens` and the constrained-decoding token mask adjust the logits in place, then greedy
//! argmax at temperature 0, otherwise temperature → top-k → top-p over a seeded ChaCha8 stream,
//! so an identical `seed` gives identical tokens.
use std::cmp::Ordering;
use std::collections::HashMap;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use turbine_core::request::SamplingParams;

use crate::structured::TokenMask;

/// One sampled token with its log-probability under the raw (temperature 1, untruncated,
/// unbiased, unmasked) distribution and, when requested, the most likely raw alternatives
/// (highest first, ties by id).
#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub token: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Everything a sampler needs to continue exactly where it stopped (P2 S-6: kept across
/// preemption; P7: the TKV1 sampler segment): the ChaCha8 seed, its word position and the tokens
/// generated so far (the penalty history).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplerState {
    pub seed: [u8; 32],
    pub word_pos: u128,
    pub generated: Vec<u32>,
}

/// Samples one token per step for one request.
///
/// Order of operations on the logits: `logit_bias` → repetition (HF, over prompt and generated
/// tokens) then presence/frequency (OpenAI, over generated tokens) penalties → EOS and stop ids
/// suppressed while fewer than `min_tokens` were generated → the token mask (disallowed → −∞; the
/// mask wins over `logit_bias`) → greedy, or temperature → top-k → top-p over the candidates
/// sorted descending (ties by the lower id) and one 24-bit uniform draw. Reported logprobs are
/// the log-softmax of the logits as passed in, before any of these adjustments.
#[derive(Clone, Debug)]
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    /// `None` = disabled (request `top_k` −1 or 0).
    top_k: Option<usize>,
    /// Alternatives reported per token (0 = none).
    top_logprobs: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    min_tokens: u32,
    /// EOS and stop token ids (sorted, distinct): suppressed until `min_tokens` were generated.
    eos_token_ids: Vec<u32>,
    /// Distinct prompt ids (sorted), kept only when the repetition penalty is active.
    prompt_tokens: Vec<u32>,
    generated: Vec<u32>,
    /// Occurrences of each generated id (the presence/frequency/repetition history).
    counts: HashMap<u32, u32>,
    rng: ChaCha8Rng,
    /// Raw values of the ids this step changed before sampling (reused across steps).
    originals: HashMap<u32, f32>,
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

/// `ids` sorted ascending without duplicates.
fn sorted_distinct(ids: &[u32]) -> Vec<u32> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// A uniform draw in `[0, 1)` from 24 random bits (exact in f32).
fn uniform(rng: &mut ChaCha8Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

impl Sampler {
    /// A sampler for `params`: ChaCha8 seeded from `seed`, else from OS entropy. `prompt_tokens`
    /// feed the repetition penalty; `eos_token_ids` (EOS plus the request's stop token ids) are
    /// suppressed until `min_tokens` tokens were generated.
    pub fn new(params: &SamplingParams, prompt_tokens: &[u32], eos_token_ids: &[u32]) -> Sampler {
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_os_rng(),
        };
        Sampler::with_rng(params, prompt_tokens, eos_token_ids, rng)
    }

    /// A sampler that continues from `state` (other arguments as in [`Sampler::new`]; the
    /// request `seed` is superseded by the saved stream): the next draw is the one the saved
    /// sampler would have made, and the penalty and `min_tokens` history is restored.
    pub fn from_state(
        params: &SamplingParams,
        prompt_tokens: &[u32],
        eos_token_ids: &[u32],
        state: SamplerState,
    ) -> Sampler {
        let mut rng = ChaCha8Rng::from_seed(state.seed);
        rng.set_word_pos(state.word_pos);
        let mut sampler = Sampler::with_rng(params, prompt_tokens, eos_token_ids, rng);
        for token in state.generated {
            sampler.observe(token);
        }
        sampler
    }

    fn with_rng(
        params: &SamplingParams,
        prompt_tokens: &[u32],
        eos_token_ids: &[u32],
        rng: ChaCha8Rng,
    ) -> Sampler {
        let prompt_tokens = if params.repetition_penalty != 1.0 {
            sorted_distinct(prompt_tokens)
        } else {
            Vec::new()
        };
        Sampler {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: usize::try_from(params.top_k).ok().filter(|&k| k > 0),
            top_logprobs: params.logprobs.map_or(0, |n| n as usize),
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            repetition_penalty: params.repetition_penalty,
            logit_bias: params.logit_bias.clone(),
            min_tokens: params.min_tokens,
            eos_token_ids: sorted_distinct(eos_token_ids),
            prompt_tokens,
            generated: Vec::new(),
            counts: HashMap::new(),
            rng,
            originals: HashMap::new(),
        }
    }

    /// The state that [`Sampler::from_state`] resumes from.
    pub fn state(&self) -> SamplerState {
        SamplerState {
            seed: self.rng.get_seed(),
            word_pos: self.rng.get_word_pos(),
            generated: self.generated.clone(),
        }
    }

    /// Records a token the request emitted (after `sample`, once the caller accepted it): it
    /// counts towards `min_tokens` and the penalties from the next step on.
    pub fn observe(&mut self, token: u32) {
        self.generated.push(token);
        *self.counts.entry(token).or_insert(0) += 1;
    }

    /// Picks the next token from one vocabulary row. `logits` is adjusted in place (bias,
    /// penalties, `min_tokens`, `mask`); with none of them active it is left unchanged. The
    /// reported logprobs are the log-softmax of `logits` as passed in (before temperature).
    pub fn sample(&mut self, logits: &mut [f32], mask: Option<&TokenMask>) -> SampledToken {
        let lse = log_sum_exp(logits);
        let top_logprobs = if self.top_logprobs > 0 {
            top_n(logits, self.top_logprobs)
                .into_iter()
                .map(|(id, v)| (id, v - lse))
                .collect()
        } else {
            Vec::new()
        };
        self.adjust(logits, mask);
        let token = if !logits.iter().any(|&v| v > f32::NEG_INFINITY) {
            // `min_tokens` and the mask together forbid every id: the grammar wins, it ends here.
            self.best_allowed_eos(mask)
                .unwrap_or_else(|| argmax(logits))
        } else if self.temperature <= 0.0 {
            argmax(logits)
        } else {
            self.draw(logits)
        };
        let raw = self
            .originals
            .get(&token)
            .copied()
            .unwrap_or(logits[token as usize]);
        SampledToken {
            token,
            logprob: raw - lse,
            top_logprobs,
        }
    }

    /// Applies bias, penalties, `min_tokens` and the mask to `logits`, remembering the raw value
    /// of every id it changes before the mask (an id the mask disallows is never sampled).
    fn adjust(&mut self, logits: &mut [f32], mask: Option<&TokenMask>) {
        self.originals.clear();
        let originals = &mut self.originals;
        let mut touch = |logits: &[f32], id: u32| -> Option<usize> {
            let i = id as usize;
            let v = *logits.get(i)?;
            originals.entry(id).or_insert(v);
            Some(i)
        };
        for &(id, bias) in &self.logit_bias {
            if let Some(i) = touch(logits, id) {
                logits[i] += bias;
            }
        }
        if self.repetition_penalty != 1.0 {
            let p = self.repetition_penalty;
            let prompt = &self.prompt_tokens;
            let seen = prompt.iter().copied().chain(
                self.counts
                    .keys()
                    .copied()
                    .filter(|id| prompt.binary_search(id).is_err()),
            );
            for id in seen {
                if let Some(i) = touch(logits, id) {
                    let v = logits[i];
                    logits[i] = if v > 0.0 { v / p } else { v * p };
                }
            }
        }
        if self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 {
            for (&id, &count) in &self.counts {
                if let Some(i) = touch(logits, id) {
                    logits[i] -= self.frequency_penalty * count as f32 + self.presence_penalty;
                }
            }
        }
        if (self.generated.len() as u64) < u64::from(self.min_tokens) {
            for &id in &self.eos_token_ids {
                if let Some(i) = touch(logits, id) {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(mask) = mask {
            mask.apply(logits);
        }
    }

    /// The mask-allowed EOS or stop id with the highest raw logit (ties by the lower id).
    fn best_allowed_eos(&self, mask: Option<&TokenMask>) -> Option<u32> {
        let mask = mask?;
        self.eos_token_ids
            .iter()
            .filter(|&&id| mask.is_allowed(id))
            .filter_map(|&id| self.originals.get(&id).map(|&v| (id, v)))
            .min_by(by_value_desc)
            .map(|(id, _)| id)
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

    /// A sampler with no prompt history and no EOS ids.
    fn plain(p: &SamplingParams) -> Sampler {
        Sampler::new(p, &[], &[])
    }

    fn greedy() -> SamplingParams {
        params(0.0, 1.0, -1, 1)
    }

    #[test]
    fn logit_bias_moves_greedy_and_keeps_raw_logprob() {
        let raw = vec![1.0f32, 3.0, 2.0, 0.0];
        let mut p = greedy();
        p.logit_bias = vec![(2, 1.5), (99, 5.0)]; // out-of-vocab ids are ignored
        let mut s = plain(&p);
        let mut logits = raw.clone();
        let t = s.sample(&mut logits, None);
        assert_eq!(t.token, 2);
        assert!(
            (t.logprob - log_softmax(&raw)[2]).abs() < 1e-6,
            "raw logprob"
        );
        assert_eq!(logits[2], 3.5, "the bias is applied in place");
    }

    #[test]
    fn presence_and_frequency_penalties_follow_openai() {
        // Id 0 generated twice: logit − 2·frequency − presence.
        let raw = vec![3.0f32, 2.0, 0.0];
        let mut p = greedy();
        p.frequency_penalty = 0.5;
        p.presence_penalty = 0.25;
        let mut s = plain(&p);
        assert_eq!(s.sample(&mut raw.clone(), None).token, 0);
        s.observe(0);
        let mut logits = raw.clone();
        assert_eq!(s.sample(&mut logits, None).token, 0, "3 − 0.75 > 2");
        assert!((logits[0] - 2.25).abs() < 1e-6);
        s.observe(0);
        let mut logits = raw.clone();
        assert_eq!(s.sample(&mut logits, None).token, 1, "3 − 1.25 < 2");
        assert!((logits[0] - 1.75).abs() < 1e-6);
        assert_eq!(logits[1], 2.0, "an unseen id is untouched");
    }

    #[test]
    fn repetition_penalty_follows_hf_over_prompt_and_output() {
        // Positive logits are divided, negative multiplied; prompt ids count as seen.
        let raw = vec![4.0f32, 3.0, -1.0, 2.5];
        let mut p = greedy();
        p.repetition_penalty = 2.0;
        let mut s = Sampler::new(&p, &[0, 2, 0], &[]);
        let mut logits = raw.clone();
        assert_eq!(s.sample(&mut logits, None).token, 1);
        assert_eq!(logits, vec![2.0, 3.0, -2.0, 2.5]);
        s.observe(1);
        let mut logits = raw.clone();
        assert_eq!(s.sample(&mut logits, None).token, 3);
        assert_eq!(logits, vec![2.0, 1.5, -2.0, 2.5]);
    }

    #[test]
    fn min_tokens_suppresses_eos_and_stop_ids() {
        let raw = vec![0.0f32, 1.0, 5.0, 4.0];
        let mut p = greedy();
        p.min_tokens = 2;
        // EOS 2 and stop id 3 are both held back for two tokens.
        let mut s = Sampler::new(&p, &[], &[2, 3]);
        for _ in 0..2 {
            let t = s.sample(&mut raw.clone(), None);
            assert_eq!(t.token, 1);
            s.observe(t.token);
        }
        assert_eq!(s.sample(&mut raw.clone(), None).token, 2);
    }

    #[test]
    fn mask_forbidding_all_but_eos_beats_min_tokens() {
        let raw = vec![0.0f32, 1.0, 5.0, 4.0];
        let mut p = greedy();
        p.min_tokens = 10;
        let mut s = Sampler::new(&p, &[], &[2, 3]);
        let mut mask = TokenMask::new_none(4);
        mask.allow(3);
        let t = s.sample(&mut raw.clone(), Some(&mask));
        assert_eq!(t.token, 3, "the grammar only allows ending");
        assert!((t.logprob - log_softmax(&raw)[3]).abs() < 1e-6);
    }

    #[test]
    fn state_round_trip_continues_the_stream() {
        let raw: Vec<f32> = (0..64).map(|i| (i as f32 * 0.61).cos()).collect();
        let mut p = params(1.0, 0.95, 40, 21);
        p.frequency_penalty = 0.3;
        p.repetition_penalty = 1.2;
        let prompt = [3u32, 5, 8];
        let run = |s: &mut Sampler, n: usize| -> Vec<u32> {
            (0..n)
                .map(|_| {
                    let t = s.sample(&mut raw.clone(), None).token;
                    s.observe(t);
                    t
                })
                .collect()
        };
        let mut whole = Sampler::new(&p, &prompt, &[]);
        let expected = run(&mut whole, 40);

        let mut first = Sampler::new(&p, &prompt, &[]);
        let head = run(&mut first, 17);
        let json = serde_json::to_string(&first.state()).expect("serialise");
        let state: SamplerState = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(state.generated, head);
        let mut resumed = Sampler::from_state(&p, &prompt, &[], state);
        let tail = run(&mut resumed, 23);
        assert_eq!([head, tail].concat(), expected);
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
        let mut sampler = plain(&p);
        let mut logits = raw.clone();
        let s = sampler.sample(&mut logits, None);
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
        let mut sampler = plain(&params(0.5, 1.0, -1, 3));
        let s = sampler.sample(&mut raw.clone(), None);
        let lp = log_softmax(&raw);
        assert!((s.logprob - lp[s.token as usize]).abs() < 1e-6);
        assert!(s.top_logprobs.is_empty(), "no logprobs requested");
    }

    #[test]
    fn top_k_one_and_tiny_top_p_are_argmax() {
        let raw: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 * 0.1).collect();
        let best = argmax(&raw);
        for seed in 0..20 {
            let mut k1 = plain(&params(1.5, 1.0, 1, seed));
            assert_eq!(k1.sample(&mut raw.clone(), None).token, best);
            let mut p0 = plain(&params(1.5, 1e-6, -1, seed));
            assert_eq!(p0.sample(&mut raw.clone(), None).token, best);
        }
    }

    #[test]
    fn top_k_and_top_p_restrict_the_support() {
        // probabilities ≈ [0.64, 0.24, 0.09, 0.03]
        let raw = vec![3.0f32, 2.0, 1.0, 0.0];
        let mut k2 = plain(&params(1.0, 1.0, 2, 11));
        let mut p = plain(&params(1.0, 0.8, -1, 12));
        let mut seen_k = [0usize; 4];
        let mut seen_p = [0usize; 4];
        for _ in 0..2000 {
            seen_k[k2.sample(&mut raw.clone(), None).token as usize] += 1;
            seen_p[p.sample(&mut raw.clone(), None).token as usize] += 1;
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
        let mut s = plain(&params(1.0, 1.0, -1, 5));
        let n = 14_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            counts[s.sample(&mut raw.clone(), None).token as usize] += 1;
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
            let mut s = plain(&params(0.8, 0.9, 50, seed));
            (0..32)
                .map(|_| s.sample(&mut raw.clone(), None).token)
                .collect()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }
}
