//! Scalar reference math on logical row-major f32 arrays. Every reduction accumulates in f32,
//! sequentially in index order, so results are deterministic and independent of threading.
//! `round` rounds a value to the activation dtype at the points where the Hugging Face BF16
//! forward materialises a tensor in that dtype.

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
/// mean of squares accumulated in f32.
pub(crate) fn rmsnorm(
    x: &[f32],
    w: &[f32],
    dim: usize,
    eps: f32,
    round_x: impl Fn(f32) -> f32,
    round_out: impl Fn(f32) -> f32,
) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (row, o) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mut sum_sq = 0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let r = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for ((o, &xv), &wv) in o.iter_mut().zip(row).zip(w) {
            *o = round_out(round_x(xv * r) * wv);
        }
    }
    out
}

/// HF `rotate_half` RoPE in place on `x` `[tokens, heads, d]` over the first `rotary_dim`
/// elements of each head: `f = pos · inv_freq[i]` in f32, cos/sin rounded to the activation
/// dtype, `x1' = round(round(x1·c) + round(−x2·s))`, `x2' = round(round(x2·c) + round(x1·s))`.
pub(crate) fn rope(
    x: &mut [f32],
    positions: &[i32],
    inv_freq: &[f32],
    heads: usize,
    d: usize,
    rotary_dim: usize,
    round: impl Fn(f32) -> f32,
) {
    let half = rotary_dim / 2;
    for (t, &pos) in positions.iter().enumerate() {
        for (i, &freq) in inv_freq[..half].iter().enumerate() {
            let f = pos as f32 * freq;
            let (c, s) = (round(f.cos()), round(f.sin()));
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
