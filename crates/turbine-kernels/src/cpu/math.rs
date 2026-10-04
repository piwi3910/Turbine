//! Scalar reference math on logical row-major f32 arrays. Every reduction accumulates in f32,
//! sequentially in index order, so results are deterministic and independent of threading.
//! `round` rounds a value to the activation dtype at the points where the Hugging Face BF16
//! forward materialises a tensor in that dtype.
use std::cmp::Ordering;

/// GEMM shape: `a` is `[m, k]`; `b` is `[n, k]` when `trans_b`, else `[k, n]`; `c` is `[m, n]`.
pub(crate) struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub trans_b: bool,
}

/// `alpha · a·op(b) + beta · c_old`; each dot product accumulates sequentially over `k` in f32.
/// `c_old` is read only when `beta != 0`.
pub(crate) fn gemm(
    a: &[f32],
    b: &[f32],
    c_old: &[f32],
    s: &GemmShape,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut c = vec![0f32; s.m * s.n];
    for i in 0..s.m {
        let a_row = &a[i * s.k..(i + 1) * s.k];
        for j in 0..s.n {
            let mut acc = 0f32;
            if s.trans_b {
                let b_row = &b[j * s.k..(j + 1) * s.k];
                for (x, y) in a_row.iter().zip(b_row) {
                    acc += x * y;
                }
            } else {
                for (kk, x) in a_row.iter().enumerate() {
                    acc += x * b[kk * s.n + j];
                }
            }
            let mut value = alpha * acc;
            if beta != 0.0 {
                value += beta * c_old[i * s.n + j];
            }
            c[i * s.n + j] = value;
        }
    }
    c
}

/// Attention shape: `q`/`out` are `[q_len, hq, d]`, `k`/`v` are `[q_start + q_len, hkv, d]`.
pub(crate) struct AttnShape {
    pub q_len: usize,
    pub q_start: usize,
    pub hq: usize,
    pub hkv: usize,
    pub d: usize,
    pub causal: bool,
}

/// GQA attention in the CK FMHA order (also PyTorch's CPU flash attention): scores and their
/// row max in f32, the unnormalised `exp(s − max)` summed in f32, each of them passed through
/// `round_p` (rounding to the activation dtype, as those kernels feed P to the P·V GEMM in that
/// dtype), P·V accumulated in f32, and the row divided by the f32 sum at the end. Query head `h`
/// reads KV head `h / (hq / hkv)`; query `i` sits at absolute position `q_start + i` and, when
/// causal, attends keys `0..=q_start + i` (otherwise all `q_start + q_len` keys).
pub(crate) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: &AttnShape,
    scale: f32,
    round_p: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let group = s.hq / s.hkv;
    let kv_len = s.q_start + s.q_len;
    let mut out = vec![0f32; s.q_len * s.hq * s.d];
    let mut scores = vec![0f32; kv_len];
    for i in 0..s.q_len {
        let visible = if s.causal { s.q_start + i + 1 } else { kv_len };
        for h in 0..s.hq {
            let kvh = h / group;
            let row = (i * s.hq + h) * s.d;
            let qv = &q[row..row + s.d];
            let mut max = f32::NEG_INFINITY;
            for (j, score) in scores[..visible].iter_mut().enumerate() {
                let kr = (j * s.hkv + kvh) * s.d;
                let mut dot = 0f32;
                for (x, y) in qv.iter().zip(&k[kr..kr + s.d]) {
                    dot += x * y;
                }
                *score = dot * scale;
                max = max.max(*score);
            }
            let mut sum = 0f32;
            for score in &mut scores[..visible] {
                *score = (*score - max).exp();
                sum += *score;
            }
            let o = &mut out[row..row + s.d];
            for (j, score) in scores[..visible].iter().enumerate() {
                let p = round_p(*score);
                let vr = (j * s.hkv + kvh) * s.d;
                for (acc, x) in o.iter_mut().zip(&v[vr..vr + s.d]) {
                    *acc += p * x;
                }
            }
            for acc in o.iter_mut() {
                *acc /= sum;
            }
        }
    }
    out
}

/// HF LlamaRMSNorm on rows of `dim`: `round_out(round_x(x · rsqrt(mean(x²) + eps)) · w)`, the
/// mean of squares accumulated in f32. Exactly [`row_sumsq`] followed by
/// [`rmsnorm_from_sumsq`] with `full_dim = dim`.
pub(crate) fn rmsnorm(
    x: &[f32],
    w: &[f32],
    dim: usize,
    eps: f32,
    round_x: impl Fn(f32) -> f32,
    round_out: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let sumsq = row_sumsq(x, dim);
    rmsnorm_from_sumsq(x, w, dim, &sumsq, dim, eps, round_x, round_out)
}

/// The f32 sum of squares of each row of `dim`, accumulated sequentially over the row.
pub(crate) fn row_sumsq(x: &[f32], dim: usize) -> Vec<f32> {
    x.chunks_exact(dim)
        .map(|row| {
            let mut sum_sq = 0f32;
            for &v in row {
                sum_sq += v * v;
            }
            sum_sq
        })
        .collect()
}

/// The RMSNorm scaling of rows of `dim` with a given sum of squares per row over `full_dim`
/// elements: `round_out(round_x(x · 1 / sqrt(sumsq / full_dim + eps)) · w)` (the ABI v2.6
/// `rmsnorm_sharded`; with `full_dim = dim` and the row's own `row_sumsq`, [`rmsnorm`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnorm_from_sumsq(
    x: &[f32],
    w: &[f32],
    dim: usize,
    sumsq: &[f32],
    full_dim: usize,
    eps: f32,
    round_x: impl Fn(f32) -> f32,
    round_out: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for ((row, o), &sum_sq) in x
        .chunks_exact(dim)
        .zip(out.chunks_exact_mut(dim))
        .zip(sumsq)
    {
        let r = 1.0 / (sum_sq / full_dim as f32 + eps).sqrt();
        for ((o, &xv), &wv) in o.iter_mut().zip(row).zip(w) {
            *o = round_out(round_x(xv * r) * wv);
        }
    }
    out
}

/// HF `rotate_half` RoPE in place on `x` `[tokens, heads, d]` over the first `rotary_dim`
/// elements of each head: `f = pos · inv_freq[i]` in f32, cos/sin multiplied by `attn_factor`
/// (YaRN's `m`, 1.0 = none) in f32 and then rounded to the activation dtype (transformers'
/// `cos() * attention_scaling` then `.to(dtype)`), `x1' = round(round(x1·c) + round(−x2·s))`,
/// `x2' = round(round(x2·c) + round(x1·s))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope(
    x: &mut [f32],
    positions: &[i32],
    inv_freq: &[f32],
    heads: usize,
    d: usize,
    rotary_dim: usize,
    attn_factor: f32,
    round: impl Fn(f32) -> f32,
) {
    let half = rotary_dim / 2;
    for (t, &pos) in positions.iter().enumerate() {
        for (i, &freq) in inv_freq[..half].iter().enumerate() {
            let f = pos as f32 * freq;
            let (c, s) = (round(f.cos() * attn_factor), round(f.sin() * attn_factor));
            for h in 0..heads {
                let base = (t * heads + h) * d;
                let x1 = x[base + i];
                let x2 = x[base + i + half];
                x[base + i] = round(round(x1 * c) + round(-x2 * s));
                x[base + i + half] = round(round(x2 * c) + round(x1 * s));
            }
        }
    }
}

/// `silu(g) = g · σ(g)`, in f32.
pub(crate) fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

// Logits reduction (ABI v2.1 `logits_reduce`). These mirror the host sampler of
// `turbine-model` operation for operation — including its f64 sums, taken sequentially in id
// order — so a row reduced here finishes on the host exactly as the whole row would.

/// Descending by value, ties to the lower id; NaN sorts last.
pub(crate) fn by_value_desc(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)),
    }
}

/// The `n` largest `(id, value)` pairs of `values`, in `by_value_desc` order.
pub(crate) fn top_n(values: &[f32], n: usize) -> Vec<(u32, f32)> {
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

/// Index of the largest value; ties to the lower id, NaN never wins (0 when nothing exceeds −∞).
pub(crate) fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best = i;
            best_value = v;
        }
    }
    best as u32
}

/// `log(Σ exp(v))` around the maximum, NaN ignored: the exponentials of `v − max` summed in f64
/// in id order, `max + ln(sum)` rounded to f32. A row without a finite maximum returns it.
pub(crate) fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = values
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&v| f64::from(v - max).exp())
        .sum();
    max + sum.ln() as f32
}

/// One categorical draw over the whole row at `temperature` with the uniform `u` in `[0, 1)`:
/// the weights `exp(v · (1/T) − max)` (NaN weighs 0) are summed in f64 in id order, and the draw
/// is the first id whose cumulative weight exceeds `u · total` (the last id with a non-zero
/// weight when rounding leaves `u · total` at the total). `temperature` ≤ 0 or a row without a
/// finite scaled maximum gives the argmax.
pub(crate) fn categorical(values: &[f32], temperature: f32, u: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(values);
    }
    let inv_t = 1.0 / temperature;
    let max = values
        .iter()
        .map(|&v| v * inv_t)
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(values);
    }
    let weight = |v: f32| {
        let scaled = v * inv_t;
        if scaled.is_nan() {
            0.0
        } else {
            f64::from(scaled - max).exp()
        }
    };
    let mut total = 0f64;
    let mut last_positive = 0usize;
    for (i, &v) in values.iter().enumerate() {
        let w = weight(v);
        total += w;
        if w > 0.0 {
            last_positive = i;
        }
    }
    let target = f64::from(u) * total;
    let mut cum = 0f64;
    for (i, &v) in values.iter().enumerate() {
        cum += weight(v);
        if target < cum {
            return i as u32;
        }
    }
    last_positive as u32
}

/// The top-p draw over the whole row at `temperature` with nucleus mass `top_p` < 1 and the
/// uniform `u`, in the host sampler's seeded arithmetic: every id in descending order (ties to
/// the lower id, NaN last), weights `exp(v · (1/T) − max)` in f64 (NaN weighs 0) summed in that
/// order, the shortest prefix whose sum reaches `top_p · total` (at least one id; all of them
/// when rounding never reaches it), then the first id of the prefix whose cumulative weight
/// exceeds `u` × the prefix's sum (its last id with a non-zero weight when rounding leaves the
/// target at the sum). `temperature` ≤ 0 or a row without a finite scaled maximum gives the
/// argmax.
pub(crate) fn nucleus(values: &[f32], temperature: f32, top_p: f32, u: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(values);
    }
    let inv_t = 1.0 / temperature;
    let candidates = top_n(values, values.len());
    let max = candidates
        .iter()
        .map(|c| c.1 * inv_t)
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(values);
    }
    let mut weights: Vec<f64> = candidates
        .iter()
        .map(|c| {
            let scaled = c.1 * inv_t;
            if scaled.is_nan() {
                0.0
            } else {
                f64::from(scaled - max).exp()
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let target = f64::from(top_p) * total;
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
    let kept: f64 = weights.iter().sum();
    let target = f64::from(u) * kept;
    let mut cum = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cum += w;
        if target < cum {
            return candidates[i].0;
        }
    }
    candidates[weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_without_transpose_and_with_beta() {
        // a [1,2] · b [2,2] (not transposed) + 0.5 · c_old
        let c = gemm(
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[10.0, 20.0],
            &GemmShape {
                m: 1,
                n: 2,
                k: 2,
                trans_b: false,
            },
            2.0,
            0.5,
        );
        assert_eq!(c, [2.0 * 7.0 + 5.0, 2.0 * 10.0 + 10.0]);
    }

    #[test]
    fn non_causal_attention_sees_every_key() {
        let s = AttnShape {
            q_len: 1,
            q_start: 1,
            hq: 1,
            hkv: 1,
            d: 1,
            causal: false,
        };
        // Equal scores: the output is the mean of the values.
        let out = attention(&[0.0], &[1.0, 2.0], &[4.0, 8.0], &s, 1.0, |p| p);
        assert_eq!(out, [6.0]);
    }

    #[test]
    fn silu_values() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
    }
}
