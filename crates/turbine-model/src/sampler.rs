//! Host-side token sampling from FP32 logits (P1 S-9, P2 S-10/S-17): logit bias, penalties,
//! `min_tokens` and the constrained-decoding token mask adjust the logits in place, then greedy
//! argmax at temperature 0, otherwise temperature → top-k → top-p over a seeded ChaCha8 stream,
//! so an identical `seed` gives identical tokens (seeded draws keep a fixed f64 arithmetic;
//! unseeded ones use a vectorised f32 `exp`).
use std::cmp::Ordering;
use std::collections::HashMap;

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use turbine_core::request::SamplingParams;

use crate::structured::TokenMask;

/// One sampled token and, when the request asked for logprobs, its log-probability under the
/// raw (temperature 1, untruncated, unbiased, unmasked) distribution and the most likely raw
/// alternatives (highest first, ties by id). Without logprobs `logprob` is `None` and nothing
/// normalises the row (a full-vocabulary pass the draw does not need).
#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub token: u32,
    pub logprob: Option<f32>,
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
    /// The request asked for logprobs (`logprobs` present, even 0).
    logprobs: bool,
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
    /// The request carries a `seed`: draws use the fixed f64 arithmetic of [`Sampler::draw_exact`]
    /// so the tokens never change between releases. Unseeded requests promise no particular
    /// tokens and take the vectorised [`Sampler::draw_fast`].
    exact: bool,
    /// Raw values of the ids this step changed before sampling (reused across steps).
    originals: HashMap<u32, f32>,
    /// Buffers reused across steps, so a step allocates nothing vocabulary-sized.
    scratch: Scratch,
}

/// Vocabulary-sized buffers of one sampler, kept between steps.
#[derive(Clone, Debug, Default)]
struct Scratch {
    /// Sort keys of the top-n selection, see [`desc_key`].
    keys: Vec<u64>,
    /// Truncated candidates `(id, logit / T)`, highest first.
    candidates: Vec<(u32, f32)>,
    /// Unnormalised probabilities of the candidates (of every id when nothing is cut).
    weights: Vec<f64>,
    /// [`Sampler::draw_fast`]: f32 weights of every id (or of the top-k candidates).
    fast: Vec<f32>,
    /// [`Sampler::draw_fast`]: f64 sums of `fast` per [`FAST_BLOCK`] ids.
    blocks: Vec<f64>,
}

/// Ids per partial sum of the fast draw: the draw finds its block from the sums, then scans
/// only that block.
const FAST_BLOCK: usize = 1024;

/// `e^d` for `d ≤ 0` in f32 without a libm call, so a loop over the vocabulary vectorises:
/// Cody–Waite reduction `d = n·ln 2 + r`, `|r| ≤ ln 2 / 2`, a degree-6 Taylor polynomial for
/// `e^r` (relative error below 3e-7) and `2^n` from the exponent bits. `d` below −87 (the
/// weight is under 2e-38), −∞ (a masked id) and NaN give exactly 0.
#[inline(always)]
fn exp_fast(d: f32) -> f32 {
    const ROUND: f32 = 12_582_912.0; // 1.5 · 2^23: adding it rounds to an integer
    const LN2_HI: f32 = 0.693_359_4; // 0.693359375, exact in few bits
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = d.max(-87.0);
    let t = x * std::f32::consts::LOG2_E + ROUND;
    let n = t - ROUND;
    let r = x - n * LN2_HI - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    // t's low mantissa bits hold n + 2^22 + 2^23; n ≥ −126 keeps 2^n a normal float.
    let n_int = t.to_bits() as i32 - ROUND.to_bits() as i32;
    let scale = f32::from_bits(((n_int + 127) << 23) as u32);
    if d >= -87.0 { p * scale } else { 0.0 }
}

/// The largest non-NaN `v · scale` of `values` in eight independent lanes (so it vectorises);
/// `None` when there is none or it is not finite — [`finite_max`] of the scaled values.
fn scaled_max(values: &[f32], scale: f32) -> Option<f32> {
    let mut lanes = [f32::NEG_INFINITY; 8];
    let chunks = values.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            let x = c[i] * scale;
            // NaN compares false and is skipped.
            if x > lanes[i] {
                lanes[i] = x;
            }
        }
    }
    let mut max = f32::NEG_INFINITY;
    for x in lanes.into_iter().chain(rest.iter().map(|&v| v * scale)) {
        if x > max {
            max = x;
        }
    }
    max.is_finite().then_some(max)
}

/// Sum of `w` in f64 with four interleaved accumulators (a fixed order, so repeatable).
fn block_sum(w: &[f32]) -> f64 {
    let mut acc = [0.0f64; 4];
    let chunks = w.chunks_exact(4);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..4 {
            acc[i] += f64::from(c[i]);
        }
    }
    rest.iter()
        .fold((acc[0] + acc[1]) + (acc[2] + acc[3]), |s, &x| {
            s + f64::from(x)
        })
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

/// A `u64` whose ascending order is [`by_value_desc`] on `(id, v)`: the high word is the
/// complement of `v`'s IEEE total-order key (NaN: all ones, so it sorts last), the low word the
/// id. Integer keys sort several times faster than the float comparator over a 128k vocabulary.
fn desc_key(id: u32, v: f32) -> u64 {
    let high = if v.is_nan() {
        u32::MAX
    } else {
        // `f32::total_cmp`'s key: flip the magnitude bits of negatives, then bias the sign.
        let bits = v.to_bits() as i32;
        let total = (bits ^ ((((bits >> 31) as u32) >> 1) as i32)) as u32 ^ 0x8000_0000;
        // Only the all-ones negative NaN has total key 0, so a number never collides with NaN.
        !total
    };
    (u64::from(high) << 32) | u64::from(id)
}

/// The `n` largest `(id, value)` pairs of `values` into `out`, highest first, ties by id
/// (`keys` is scratch).
fn top_n_into(values: &[f32], n: usize, keys: &mut Vec<u64>, out: &mut Vec<(u32, f32)>) {
    out.clear();
    let n = n.min(values.len());
    if n == 0 {
        return;
    }
    keys.clear();
    keys.extend(
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| desc_key(i as u32, v)),
    );
    if n < keys.len() {
        keys.select_nth_unstable(n - 1);
        keys.truncate(n);
    }
    keys.sort_unstable();
    out.extend(keys.iter().map(|&k| {
        let id = k as u32;
        (id, values[id as usize])
    }));
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
            logprobs: params.logprobs.is_some(),
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
            exact: params.seed.is_some(),
            originals: HashMap::new(),
            scratch: Scratch::default(),
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
    /// reported logprobs (only when requested) are the log-softmax of `logits` as passed in
    /// (before temperature).
    pub fn sample(&mut self, logits: &mut [f32], mask: Option<&TokenMask>) -> SampledToken {
        let lse = self.logprobs.then(|| log_sum_exp(logits));
        let top_logprobs = match lse {
            Some(lse) if self.top_logprobs > 0 => {
                let mut top = Vec::with_capacity(self.top_logprobs);
                top_n_into(logits, self.top_logprobs, &mut self.scratch.keys, &mut top);
                for t in &mut top {
                    t.1 -= lse;
                }
                top
            }
            _ => Vec::new(),
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
        let logprob = lse.map(|lse| {
            let raw = self
                .originals
                .get(&token)
                .copied()
                .unwrap_or(logits[token as usize]);
            raw - lse
        });
        SampledToken {
            token,
            logprob,
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

    /// Temperature → top-k → top-p, then one draw over the kept candidates: exact for a
    /// seeded request, fast otherwise.
    fn draw(&mut self, logits: &[f32]) -> u32 {
        if self.exact {
            self.draw_exact(logits)
        } else {
            self.draw_fast(logits)
        }
    }

    /// The seeded draw. Its arithmetic is fixed so seeded streams keep their tokens:
    /// candidates `logit · (1/T)` in f32, weights `exp(c − max)` in f64 (libm) summed in
    /// candidate order, the top-p prefix and the draw `u · total` over that same order.
    fn draw_exact(&mut self, logits: &[f32]) -> u32 {
        let inv_t = 1.0 / self.temperature;
        let vocab = logits.len();
        let k = self.top_k.map_or(vocab, |k| k.min(vocab));
        let Scratch {
            keys,
            candidates,
            weights,
            ..
        } = &mut self.scratch;
        weights.clear();
        if k < vocab || self.top_p < 1.0 {
            // Candidates sorted descending (ties by id). Scaling by 1/T keeps the order, so the
            // top-k cut is taken on the raw logits.
            top_n_into(logits, k, keys, candidates);
            for c in candidates.iter_mut() {
                c.1 *= inv_t;
            }
            let Some(max) = finite_max(candidates.iter().map(|c| c.1)) else {
                return argmax(logits);
            };
            weights.extend(candidates.iter().map(|c| weight(c.1, max)));
        } else {
            // Nothing is cut: every id in id order, so no candidate list is built.
            candidates.clear();
            let Some(max) = finite_max(logits.iter().map(|&v| v * inv_t)) else {
                return argmax(logits);
            };
            weights.extend(logits.iter().map(|&v| weight(v * inv_t, max)));
        }
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
        let id_of = |i: usize| -> u32 {
            if candidates.is_empty() {
                i as u32
            } else {
                candidates[i].0
            }
        };
        let u = f64::from(uniform(&mut self.rng)) * total;
        let mut cum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            cum += w;
            if u < cum {
                return id_of(i);
            }
        }
        // Rounding left u at the total: the last kept candidate with non-zero weight.
        id_of(weights.iter().rposition(|&w| w > 0.0).unwrap_or(0))
    }

    /// The unseeded draw: the same distribution as [`Sampler::draw_exact`] (temperature →
    /// top-k → top-p over candidates in descending logit order, ties by id), with weights from
    /// the vectorised [`exp_fast`] instead of a libm call per id. Without truncation the draw
    /// finds its [`FAST_BLOCK`] from partial sums and scans that block only; top-p without
    /// top-k sorts only the ids above a weight floor that already holds `top_p` of the mass.
    fn draw_fast(&mut self, logits: &[f32]) -> u32 {
        let inv_t = 1.0 / self.temperature;
        let vocab = logits.len();
        let k = self.top_k.map_or(vocab, |k| k.min(vocab));
        let Scratch {
            keys,
            candidates,
            fast,
            blocks,
            ..
        } = &mut self.scratch;
        if k < vocab {
            // Top-k candidates, highest first, then top-p over them in that order.
            top_n_into(logits, k, keys, candidates);
            let Some(max) = finite_max(candidates.iter().map(|c| c.1 * inv_t)) else {
                return argmax(logits);
            };
            fast.clear();
            fast.extend(candidates.iter().map(|c| exp_fast(c.1 * inv_t - max)));
            let total = block_sum(fast);
            let keep = top_p_prefix(fast.iter().copied(), self.top_p, total);
            let u = f64::from(uniform(&mut self.rng)) * keep.1;
            let i = scan(&fast[..keep.0], u);
            return candidates[i].0;
        }
        let Some(max) = scaled_max(logits, inv_t) else {
            return argmax(logits);
        };
        fast.clear();
        fast.resize(vocab, 0.0);
        blocks.clear();
        // Each block's weights are summed while they are still in L1.
        for (ws, vs) in fast.chunks_mut(FAST_BLOCK).zip(logits.chunks(FAST_BLOCK)) {
            for (w, &v) in ws.iter_mut().zip(vs) {
                *w = exp_fast(v * inv_t - max);
            }
            blocks.push(block_sum(ws));
        }
        let total: f64 = blocks.iter().sum();
        if self.top_p < 1.0 {
            let target = f64::from(self.top_p) * total;
            // Ids whose weight reaches a floor, lowered until they hold the target mass (the
            // max has weight 1, so the first floor usually keeps a few hundred ids).
            for floor in [-8.0f32, -16.0, -32.0, f32::NEG_INFINITY] {
                let min_weight = exp_fast(floor);
                keys.clear();
                keys.extend(
                    fast.iter()
                        .enumerate()
                        .filter(|&(_, &w)| w > 0.0 && w >= min_weight)
                        .map(|(i, _)| desc_key(i as u32, logits[i])),
                );
                let mass: f64 = keys
                    .iter()
                    .map(|&key| f64::from(fast[key as u32 as usize]))
                    .sum();
                if mass >= target {
                    break;
                }
            }
            keys.sort_unstable();
            let weight_of = |key: u64| fast[key as u32 as usize];
            let keep = top_p_prefix(keys.iter().map(|&key| weight_of(key)), self.top_p, total);
            let u = f64::from(uniform(&mut self.rng)) * keep.1;
            let mut cum = 0.0;
            for &key in &keys[..keep.0] {
                cum += f64::from(weight_of(key));
                if u < cum {
                    return key as u32;
                }
            }
            return keys[..keep.0]
                .iter()
                .rev()
                .find(|&&key| weight_of(key) > 0.0)
                .map_or_else(|| argmax(logits), |&key| key as u32);
        }
        let u = f64::from(uniform(&mut self.rng)) * total;
        let mut cum = 0.0;
        for (b, &sum) in blocks.iter().enumerate() {
            if u < cum + sum {
                let start = b * FAST_BLOCK;
                let block = &fast[start..(start + FAST_BLOCK).min(vocab)];
                let mut c = cum;
                for (i, &w) in block.iter().enumerate() {
                    c += f64::from(w);
                    if u < c {
                        return (start + i) as u32;
                    }
                }
                // Rounding inside the block: its last id with weight.
                let last = block.iter().rposition(|&w| w > 0.0).unwrap_or(0);
                return (start + last) as u32;
            }
            cum += sum;
        }
        // Rounding left u at the total: the last id with weight.
        fast.iter().rposition(|&w| w > 0.0).unwrap_or(0) as u32
    }
}

/// Length and mass of the smallest prefix of `weights` (in candidate order) whose mass
/// reaches `top_p · total` (at least one candidate); everything when `top_p` ≥ 1.
fn top_p_prefix(weights: impl Iterator<Item = f32>, top_p: f32, total: f64) -> (usize, f64) {
    let target = f64::from(top_p) * total;
    let mut cum = 0.0;
    let mut n = 0;
    for w in weights {
        cum += f64::from(w);
        n += 1;
        if top_p < 1.0 && cum >= target {
            break;
        }
    }
    (n, cum)
}

/// Index of the candidate the draw `u` (in `[0, Σ weights)`) falls on, scanning in order;
/// rounding at the end gives the last candidate with weight.
fn scan(weights: &[f32], u: f64) -> usize {
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += f64::from(w);
        if u < cum {
            return i;
        }
    }
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

/// One row of a batch to sample: the row's sampler, its logits (adjusted in place) and its
/// token mask.
pub struct SampleJob<'a> {
    pub sampler: &'a mut Sampler,
    pub logits: &'a mut [f32],
    pub mask: Option<&'a TokenMask>,
}

/// Most threads [`sample_rows`] uses: a Llama-3 row costs about a millisecond (one f64 `exp`
/// per vocabulary id), so a decode batch of 16–64 rows is done in a few rounds while the
/// HTTP runtime keeps cores of its own.
pub const MAX_SAMPLER_THREADS: usize = 8;

/// Samples every job, in parallel across rows (scoped threads, at most
/// [`MAX_SAMPLER_THREADS`] and the host's parallelism). Each sampler only reads and adjusts its
/// own row, so the tokens are exactly what calling [`Sampler::sample`] on the jobs one after
/// another gives; results are in job order. The caller still `observe`s each accepted token.
pub fn sample_rows(mut jobs: Vec<SampleJob<'_>>) -> Vec<SampledToken> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_SAMPLER_THREADS)
        .min(jobs.len());
    if threads <= 1 {
        return jobs
            .iter_mut()
            .map(|j| j.sampler.sample(j.logits, j.mask))
            .collect();
    }
    let per_thread = jobs.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks_mut(per_thread)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter_mut()
                        .map(|j| j.sampler.sample(j.logits, j.mask))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| match h.join() {
                Ok(tokens) => tokens,
                // A panicking sampler panics the caller, as the serial loop would.
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// The largest non-NaN value; `None` when there is none or it is not finite.
fn finite_max(values: impl Iterator<Item = f32>) -> Option<f32> {
    let max = values
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    max.is_finite().then_some(max)
}

/// Unnormalised probability of the scaled logit `c` under the maximum `max` (NaN weighs 0).
fn weight(c: f32, max: f32) -> f64 {
    if c.is_nan() {
        0.0
    } else {
        f64::from(c - max).exp()
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
        p.logprobs = Some(0);
        let mut s = plain(&p);
        let mut logits = raw.clone();
        let t = s.sample(&mut logits, None);
        assert_eq!(t.token, 2);
        assert!(
            (t.logprob.unwrap() - log_softmax(&raw)[2]).abs() < 1e-6,
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
        p.logprobs = Some(0);
        let mut s = Sampler::new(&p, &[], &[2, 3]);
        let mut mask = TokenMask::new_none(4);
        mask.allow(3);
        let t = s.sample(&mut raw.clone(), Some(&mask));
        assert_eq!(t.token, 3, "the grammar only allows ending");
        assert!((t.logprob.unwrap() - log_softmax(&raw)[3]).abs() < 1e-6);
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

    /// The integer sort keys order exactly like the float comparator, NaN, signed zeros,
    /// infinities, subnormals and ties included.
    #[test]
    fn desc_key_orders_like_by_value_desc() {
        let specials = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 4.0,
            -f32::MIN_POSITIVE / 4.0,
            f32::MAX,
            f32::MIN,
            1.0,
            1.0,
            -1.0,
            2.5,
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut values: Vec<f32> = specials.to_vec();
        values.extend((0..2000).map(|_| (uniform(&mut rng) - 0.5) * 40.0));
        values.extend((0..200).map(|_| f32::from_bits(rng.next_u32())));
        let mut by_cmp: Vec<(u32, f32)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        by_cmp.sort_by(by_value_desc);
        let mut by_key: Vec<u64> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| desc_key(i as u32, v))
            .collect();
        by_key.sort_unstable();
        let ids: Vec<u32> = by_key.iter().map(|&k| k as u32).collect();
        assert_eq!(ids, by_cmp.iter().map(|p| p.0).collect::<Vec<_>>());
        for n in [0, 1, 7, 100, values.len(), values.len() + 5] {
            let want: Vec<u32> = by_cmp.iter().take(n).map(|p| p.0).collect();
            let mut top = Vec::new();
            top_n_into(&values, n, &mut Vec::new(), &mut top);
            let got: Vec<u32> = top.iter().map(|p| p.0).collect();
            assert_eq!(got, want, "top {n}");
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
        let mut sampler = plain(&p);
        let mut logits = raw.clone();
        let s = sampler.sample(&mut logits, None);
        let lp = log_softmax(&raw);
        assert_eq!(s.token, 1, "greedy ties to the lower id");
        assert!((s.logprob.unwrap() - lp[1]).abs() < 1e-6);
        let ids: Vec<u32> = s.top_logprobs.iter().map(|t| t.0).collect();
        assert_eq!(ids, vec![1, 2, 4]);
        assert!((s.top_logprobs[2].1 - lp[4]).abs() < 1e-6);
        assert_eq!(logits, raw, "Phase 1 leaves the logits unchanged");
    }

    #[test]
    fn logprob_is_untempered() {
        let raw = vec![1.0, 2.0, 3.0];
        let mut p = params(0.5, 1.0, -1, 3);
        p.logprobs = Some(0);
        let mut sampler = plain(&p);
        let s = sampler.sample(&mut raw.clone(), None);
        let lp = log_softmax(&raw);
        assert!((s.logprob.unwrap() - lp[s.token as usize]).abs() < 1e-6);
        assert!(s.top_logprobs.is_empty(), "no alternatives requested");
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

    /// The raw log-softmax costs a full pass over the vocabulary: it is computed only when the
    /// request asked for logprobs, and the draw is the same either way.
    #[test]
    fn logprob_only_when_requested() {
        let raw: Vec<f32> = (0..300).map(|i| (i as f32 * 0.13).sin() * 4.0).collect();
        for temperature in [0.0, 1.0] {
            let without = params(temperature, 1.0, -1, 9);
            let mut with = without.clone();
            with.logprobs = Some(0);
            let (mut a, mut b) = (plain(&without), plain(&with));
            for _ in 0..20 {
                let (x, y) = (
                    a.sample(&mut raw.clone(), None),
                    b.sample(&mut raw.clone(), None),
                );
                assert_eq!(x.token, y.token);
                assert_eq!(x.logprob, None, "not requested");
                let lp = y.logprob.expect("requested");
                assert!((lp - log_softmax(&raw)[y.token as usize]).abs() < 1e-6);
                assert!(
                    y.top_logprobs.is_empty(),
                    "logprobs 0 lists no alternatives"
                );
            }
        }
    }

    /// Sampling a batch across threads gives each row exactly what sampling the rows one after
    /// another gives, and adjusts each row in place the same way.
    #[test]
    fn sample_rows_matches_one_by_one() {
        let rows = big_rows(3, 17);
        let vocab = rows[0].len();
        let mut mask = TokenMask::new_none(vocab);
        for id in (0..vocab as u32).step_by(3) {
            mask.allow(id);
        }
        let configs = [
            params(1.0, 1.0, -1, 1),
            params(0.0, 1.0, -1, 2),
            params(0.7, 0.9, 40, 3),
            params(1.2, 0.95, -1, 4),
            SamplingParams {
                logprobs: Some(5),
                frequency_penalty: 0.5,
                ..params(1.0, 1.0, -1, 5)
            },
        ];
        let fresh = || -> Vec<Sampler> {
            configs
                .iter()
                .map(|p| Sampler::new(p, &[1, 2], &[7]))
                .collect()
        };
        let (mut one, mut many) = (fresh(), fresh());
        for step in 0..6 {
            let batch: Vec<f32> = (0..configs.len())
                .flat_map(|i| rows[(i + step) % rows.len()].clone())
                .collect();
            let masks: Vec<Option<&TokenMask>> = (0..configs.len())
                .map(|i| (i == 3).then_some(&mask))
                .collect();
            let mut serial_logits = batch.clone();
            let serial: Vec<SampledToken> = one
                .iter_mut()
                .zip(serial_logits.chunks_exact_mut(vocab))
                .zip(&masks)
                .map(|((s, row), m)| s.sample(row, *m))
                .collect();
            let mut batch_logits = batch;
            let jobs: Vec<SampleJob<'_>> = many
                .iter_mut()
                .zip(batch_logits.chunks_exact_mut(vocab))
                .zip(&masks)
                .map(|((sampler, logits), &mask)| SampleJob {
                    sampler,
                    logits,
                    mask,
                })
                .collect();
            let parallel = sample_rows(jobs);
            assert_eq!(parallel, serial, "step {step}");
            assert!(serial_logits == batch_logits, "rows adjusted alike");
            for ((a, b), t) in one.iter_mut().zip(many.iter_mut()).zip(&serial) {
                a.observe(t.token);
                b.observe(t.token);
            }
        }
    }

    /// A sampler on the unseeded (fast) path with a repeatable stream.
    fn fast(p: &SamplingParams) -> Sampler {
        let mut s = plain(p);
        s.exact = false;
        s
    }

    #[test]
    fn only_seeded_requests_take_the_exact_draw() {
        assert!(plain(&params(1.0, 1.0, -1, 1)).exact);
        let unseeded = SamplingParams {
            seed: None,
            ..params(1.0, 1.0, -1, 1)
        };
        assert!(!Sampler::new(&unseeded, &[], &[]).exact);
    }

    #[test]
    fn scaled_max_matches_finite_max() {
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        for len in [0usize, 1, 7, 8, 9, 63, 1000] {
            for special in [
                None,
                Some(f32::NAN),
                Some(f32::INFINITY),
                Some(f32::NEG_INFINITY),
            ] {
                let mut v: Vec<f32> = (0..len).map(|_| uniform(&mut rng) * 20.0 - 10.0).collect();
                if let (Some(x), true) = (special, len > 0) {
                    v[len / 2] = x;
                }
                for scale in [1.0f32, 1.0 / 0.7] {
                    let want = finite_max(v.iter().map(|&x| x * scale));
                    assert_eq!(scaled_max(&v, scale), want, "len {len} {special:?}");
                }
            }
        }
        assert_eq!(scaled_max(&[f32::NAN; 20], 1.0), None);
    }

    #[test]
    fn exp_fast_is_accurate_and_zero_below_the_floor() {
        let mut d = 0.0f32;
        while d > -87.0 {
            let (got, want) = (f64::from(exp_fast(d)), f64::from(d).exp());
            assert!(((got - want) / want).abs() < 5e-7, "e^{d}: {got} vs {want}");
            d -= 0.0137;
        }
        assert_eq!(exp_fast(0.0), 1.0);
        for zero in [-87.5, -1000.0, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(exp_fast(zero), 0.0, "{zero}");
        }
    }

    /// Unseeded draws follow softmax(logits / T) within sampling noise.
    #[test]
    fn fast_draw_follows_the_distribution() {
        // weights 1:1:2:3 at T = 1, and the same at T = 0.5 on halved logits.
        let raw = vec![0.0f32, 0.0, 2.0f32.ln(), 3.0f32.ln()];
        let half: Vec<f32> = raw.iter().map(|v| v * 0.5).collect();
        for (t, row) in [(1.0, &raw), (0.5, &half)] {
            let mut s = fast(&params(t, 1.0, -1, 5));
            let n = 14_000;
            let mut counts = [0usize; 4];
            for _ in 0..n {
                counts[s.sample(&mut row.clone(), None).token as usize] += 1;
            }
            let expected = [1.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0];
            for (c, e) in counts.iter().zip(expected) {
                let f = *c as f64 / f64::from(n);
                assert!((f - e).abs() < 0.02, "T {t}: {counts:?}");
            }
        }
    }

    #[test]
    fn fast_top_k_and_top_p_restrict_the_support() {
        // probabilities ≈ [0.64, 0.24, 0.09, 0.03]; spread over a large vocabulary so the
        // full-vocabulary top-p path (weight floors) is the one exercised.
        let mut raw = vec![f32::NEG_INFINITY; 5000];
        for (id, v) in [(4000usize, 3.0f32), (17, 2.0), (2500, 1.0), (3, 0.0)] {
            raw[id] = v;
        }
        let mut k2 = fast(&params(1.0, 1.0, 2, 11));
        let mut p8 = fast(&params(1.0, 0.8, -1, 12));
        let mut all = fast(&params(1.0, 1.0, -1, 13));
        let (mut seen_k, mut seen_p, mut seen_all) =
            (HashMap::<u32, usize>::new(), HashMap::new(), HashMap::new());
        for _ in 0..3000 {
            *seen_k
                .entry(k2.sample(&mut raw.clone(), None).token)
                .or_default() += 1;
            *seen_p
                .entry(p8.sample(&mut raw.clone(), None).token)
                .or_default() += 1;
            *seen_all
                .entry(all.sample(&mut raw.clone(), None).token)
                .or_default() += 1;
        }
        let ids = |m: &HashMap<u32, usize>| {
            let mut v: Vec<u32> = m.keys().copied().collect();
            v.sort_unstable();
            v
        };
        assert_eq!(ids(&seen_k), vec![17, 4000], "top_k 2");
        assert_eq!(ids(&seen_p), vec![17, 4000], "top_p 0.8");
        assert_eq!(
            ids(&seen_all),
            vec![3, 17, 2500, 4000],
            "-inf ids are never drawn"
        );
    }

    /// On Llama-sized rows the fast top-p keeps exactly the candidates the exact draw keeps
    /// (the same prefix of the descending order), and masked ids are never drawn.
    #[test]
    fn fast_top_p_keeps_the_exact_prefix_on_big_rows() {
        let rows = big_rows(2, 5);
        let vocab = rows[0].len();
        for (row, top_p, t) in [
            (&rows[0], 0.9f32, 1.0f32),
            (&rows[1], 0.5, 0.7),
            (&rows[0], 0.99, 1.3),
        ] {
            // The exact prefix, computed independently.
            let scaled: Vec<f64> = row.iter().map(|&v| f64::from(v * (1.0 / t))).collect();
            let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]).then(a.cmp(&b)));
            let total: f64 = scaled.iter().map(|c| (c - max).exp()).sum();
            let mut cum = 0.0;
            let mut allowed = std::collections::HashSet::new();
            for &i in &order {
                cum += (scaled[i] - max).exp();
                allowed.insert(i as u32);
                if cum >= f64::from(top_p) * total {
                    break;
                }
            }
            let mut s = fast(&params(t, top_p, -1, 3));
            let mut seen = std::collections::HashSet::new();
            for _ in 0..40 {
                let token = s.sample(&mut row.clone(), None).token;
                assert!(
                    allowed.contains(&token),
                    "{token} outside the top-p {top_p} prefix"
                );
                seen.insert(token);
            }
            assert!(seen.len() > 1, "top_p {top_p} draws vary");
        }
        let mut mask = TokenMask::new_none(vocab);
        for id in (0..vocab as u32).step_by(7) {
            mask.allow(id);
        }
        let mut s = fast(&params(1.0, 1.0, -1, 4));
        for _ in 0..100 {
            let token = s.sample(&mut rows[0].clone(), Some(&mask)).token;
            assert_eq!(token % 7, 0, "masked id {token} drawn");
        }
    }

    /// Llama-3-sized rows: a broad bulk and a few confident candidates, deterministic.
    fn big_rows(n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut u = || uniform(&mut rng);
                let mut row: Vec<f32> = (0..128_256)
                    .map(|_| (u() + u() + u() + u() - 2.0) * 3.0)
                    .collect();
                for k in 0..8 {
                    let id = (u() * 128_256.0) as usize % 128_256;
                    row[id] = 12.0 - k as f32 * 0.5;
                }
                row
            })
            .collect()
    }

    /// Tokens the sampler drew before the host-overhead rework (commit 3e43675): seeded
    /// sampling must keep giving exactly these, whatever the implementation does inside.
    #[test]
    fn seeded_tokens_are_pinned() {
        let rows = big_rows(4, 99);
        let mut small: Vec<f32> = (0..64).map(|i| ((i * 5) % 7) as f32 * 0.5).collect();
        small[3] = f32::NAN;
        small[10] = -0.0;
        small[11] = 0.0;
        let run = |p: &SamplingParams, rows: &[Vec<f32>], steps: usize| -> Vec<u32> {
            let mut s = Sampler::new(p, &[1, 2, 3, 500], &[7]);
            (0..steps)
                .map(|i| {
                    let t = s.sample(&mut rows[i % rows.len()].clone(), None).token;
                    s.observe(t);
                    t
                })
                .collect()
        };
        let mut penalised = params(0.8, 1.0, -1, 5);
        penalised.frequency_penalty = 0.4;
        penalised.repetition_penalty = 1.3;
        penalised.logit_bias = vec![(9, 2.0)];
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, SamplingParams, Vec<Vec<f32>>, usize, Vec<u32>)> = vec![
            (
                "t1",
                params(1.0, 1.0, -1, 1),
                rows.clone(),
                24,
                vec![
                    45042, 26804, 124538, 8354, 18760, 55854, 61148, 9877, 97167, 24856, 121233,
                    70773, 34701, 28032, 101990, 8354, 9053, 99716, 27164, 69803, 5887, 30124,
                    121233, 51127,
                ],
            ),
            (
                "t0.7 p0.9",
                params(0.7, 0.9, -1, 2),
                rows.clone(),
                24,
                vec![
                    18760, 28032, 121233, 19279, 18760, 26804, 121233, 31116, 18760, 5934, 44015,
                    19279, 18760, 28032, 121233, 8354, 18760, 5934, 121233, 8354, 34701, 26804,
                    121233, 53372,
                ],
            ),
            (
                "t1.3 k50",
                params(1.3, 1.0, 50, 3),
                rows.clone(),
                24,
                vec![
                    18760, 28032, 44015, 8354, 83676, 26804, 83007, 8354, 100211, 28032, 121233,
                    86922, 34701, 5934, 3204, 8354, 26046, 5934, 83007, 8354, 77812, 99634, 121233,
                    59009,
                ],
            ),
            (
                "t1 k40 p0.95",
                params(1.0, 0.95, 40, 4),
                rows.clone(),
                24,
                vec![
                    97167, 28032, 61148, 8354, 34701, 83282, 3204, 31116, 18760, 5934, 121233,
                    53372, 18760, 5934, 3204, 19279, 77812, 26804, 121233, 86922, 18760, 83282,
                    3204, 19279,
                ],
            ),
            (
                "penalised",
                penalised,
                rows.clone(),
                24,
                vec![
                    18760, 5934, 61148, 31116, 34701, 83282, 44015, 64337, 97167, 26804, 83007,
                    8354, 2448, 124166, 121233, 53372, 77812, 55219, 20480, 86922, 100211, 101931,
                    28707, 19279,
                ],
            ),
            (
                "small t1",
                params(1.0, 1.0, -1, 6),
                vec![small.clone()],
                64,
                vec![
                    22, 29, 34, 32, 12, 22, 22, 50, 19, 39, 5, 43, 57, 4, 13, 6, 4, 61, 4, 15, 4,
                    23, 16, 54, 25, 57, 41, 4, 25, 25, 55, 12, 43, 50, 1, 8, 25, 57, 50, 32, 22,
                    12, 25, 39, 24, 59, 60, 37, 22, 22, 18, 25, 22, 32, 39, 25, 53, 15, 32, 36, 6,
                    45, 39, 4,
                ],
            ),
            (
                "small p0.8",
                params(1.0, 0.8, -1, 7),
                vec![small.clone()],
                64,
                vec![
                    25, 25, 32, 25, 39, 43, 4, 50, 18, 22, 5, 53, 39, 18, 39, 19, 53, 53, 1, 2, 60,
                    32, 50, 53, 15, 8, 33, 39, 53, 60, 39, 8, 5, 46, 61, 18, 36, 46, 4, 32, 39, 60,
                    39, 12, 53, 4, 18, 18, 1, 39, 15, 1, 32, 33, 57, 18, 25, 25, 15, 61, 25, 43,
                    60, 60,
                ],
            ),
            (
                "small k3",
                params(2.0, 1.0, 3, 8),
                vec![small],
                64,
                vec![
                    4, 4, 4, 25, 25, 25, 4, 18, 4, 18, 25, 25, 18, 25, 4, 25, 18, 18, 18, 25, 4, 4,
                    4, 4, 4, 18, 18, 4, 4, 18, 18, 18, 4, 4, 18, 18, 4, 25, 18, 25, 18, 18, 18, 25,
                    18, 25, 25, 18, 4, 4, 25, 18, 18, 18, 4, 25, 25, 18, 4, 4, 4, 25, 25, 25,
                ],
            ),
        ];
        for (name, p, rows, steps, expected) in cases {
            assert_eq!(run(&p, &rows, steps), expected, "{name}");
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
