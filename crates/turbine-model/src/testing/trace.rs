//! Numerics diagnostics over [`TraceTensor`]s recorded by [`LlamaExecutor::set_trace`]:
//!
//! - [`compare_traces`]: two providers run the same tokens; every recorded op output is compared
//!   element-wise (accumulated divergence, layer by layer).
//! - [`LocalChecker`]: each op of one provider's trace is recomputed on the host from that same
//!   trace's inputs (HF BF16 rounding, sequential f32 accumulation, the cpu-reference numerics),
//!   so the error an op adds on its own is separated from the error it inherits.
//!
//! Differences are reported in absolute terms and in BF16 ulps (`2^(floor(log2 |v|) − 7)`) of
//! the larger of the two magnitudes, so a 1-ulp rounding flip reads as 1.0 at any magnitude.
//!
//! [`LlamaExecutor::set_trace`]: crate::executor::LlamaExecutor::set_trace
use std::io::{Read, Seek, SeekFrom};

use half::bf16;

use crate::SafetensorsIndex;
use crate::config::ModelArchConfig;
use crate::executor::TraceTensor;
use crate::executor::rope::inv_freq;
use crate::safetensors::Dtype;

/// Reads the BF16 tensor `name` of a checkpoint as f32 values (a positioned read of its byte
/// range; no caching). Panics when it is missing, not BF16 or unreadable (a test utility).
pub fn read_bf16_weight(index: &SafetensorsIndex, name: &str) -> Vec<f32> {
    let e = index
        .get(name)
        .unwrap_or_else(|| panic!("checkpoint lacks {name}"));
    assert_eq!(e.dtype, Dtype::BF16, "{name} is not BF16");
    let mut f = std::fs::File::open(&e.file)
        .unwrap_or_else(|err| panic!("open {}: {err}", e.file.display()));
    f.seek(SeekFrom::Start(e.range.start)).expect("seek");
    let mut bytes = vec![0u8; e.byte_len() as usize];
    f.read_exact(&mut bytes).expect("read tensor bytes");
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// One BF16 ulp at `v`'s magnitude (the spacing of BF16 values in `v`'s binade).
pub fn bf16_ulp(v: f32) -> f32 {
    let a = v.abs().max(f32::MIN_POSITIVE);
    2f32.powi(a.log2().floor() as i32 - 7)
}

fn bf(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// Element-wise difference of `got` against `want`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DiffStats {
    pub numel: usize,
    /// Elements whose values differ at all.
    pub differing: usize,
    pub max_abs: f32,
    /// Largest `|got − want|` in BF16 ulps of the larger magnitude of the two (values near zero
    /// that differ read as many ulps: judge those by `max_abs`).
    pub max_ulps: f32,
    /// Largest `|want|`.
    pub max_ref: f32,
}

impl DiffStats {
    /// Panics when the lengths differ (a caller bug).
    pub fn of(want: &[f32], got: &[f32]) -> DiffStats {
        assert_eq!(want.len(), got.len(), "compared tensors differ in length");
        let mut s = DiffStats {
            numel: want.len(),
            ..DiffStats::default()
        };
        for (&w, &g) in want.iter().zip(got) {
            let d = (g - w).abs();
            if d > 0.0 {
                s.differing += 1;
            }
            s.max_abs = s.max_abs.max(d);
            s.max_ulps = s.max_ulps.max(d / bf16_ulp(w.abs().max(g.abs())));
            s.max_ref = s.max_ref.max(w.abs());
        }
        s
    }
}

/// One row of a comparison table.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffRow {
    /// Forward step (0 = prefill).
    pub step: usize,
    pub layer: Option<usize>,
    pub name: &'static str,
    pub stats: DiffStats,
}

/// Compares two traces of the same forward step op by op. Panics when they record different
/// ops or shapes (the executors ran different models or batches).
pub fn compare_traces(step: usize, want: &[TraceTensor], got: &[TraceTensor]) -> Vec<DiffRow> {
    assert_eq!(want.len(), got.len(), "traces record different op counts");
    want.iter()
        .zip(got)
        .map(|(w, g)| {
            assert_eq!(
                (w.layer, w.name, w.shape),
                (g.layer, g.name, g.shape),
                "traces record different ops"
            );
            DiffRow {
                step,
                layer: w.layer,
                name: w.name,
                stats: DiffStats::of(&w.data, &g.data),
            }
        })
        .collect()
}

/// Fixed-width text table of `rows`.
pub fn render(title: &str, rows: &[DiffRow]) -> String {
    let mut out = format!(
        "{title}\n{:>4} {:>5} {:<11} {:>9} {:>9} {:>11} {:>9} {:>9}\n",
        "step", "layer", "op", "numel", "differ%", "max_abs", "max_ulps", "max_ref"
    );
    for r in rows {
        let s = &r.stats;
        let layer = r.layer.map_or_else(|| "-".to_string(), |l| l.to_string());
        out.push_str(&format!(
            "{:>4} {:>5} {:<11} {:>9} {:>8.2}% {:>11.4e} {:>9.2} {:>9.3}\n",
            r.step,
            layer,
            r.name,
            s.numel,
            100.0 * s.differing as f32 / s.numel.max(1) as f32,
            s.max_abs,
            s.max_ulps,
            s.max_ref,
        ));
    }
    out
}

/// Worst row per `(layer, name)` over all steps, in first-seen order.
pub fn worst_per_op(rows: &[DiffRow]) -> Vec<DiffRow> {
    let mut out: Vec<DiffRow> = Vec::new();
    for r in rows {
        match out
            .iter_mut()
            .find(|o| o.layer == r.layer && o.name == r.name)
        {
            Some(o) if r.stats.max_ulps > o.stats.max_ulps => *o = r.clone(),
            Some(_) => {}
            None => out.push(r.clone()),
        }
    }
    out
}

// ------------------------------------------------------------------------- local checks

/// `x [t, k] · wᵀ` with `w [n, k]`, sequential f32 accumulation, unrounded.
fn linear(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    let n = w.len() / k;
    let mut out = Vec::with_capacity(x.len() / k * n);
    for row in x.chunks_exact(k) {
        for wr in w.chunks_exact(k) {
            let mut acc = 0f32;
            for (a, b) in row.iter().zip(wr) {
                acc += a * b;
            }
            out.push(acc);
        }
    }
    out
}

/// How the local attention recomputation treats the softmax probabilities before P·V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnProbs {
    /// f32 probabilities (the cpu-reference provider).
    F32,
    /// Unnormalised `exp(s − max)` rounded to BF16 before P·V, the sum of the rounded values
    /// in f32, normalised after (flash-attention style kernels: CK FMHA, PyTorch CPU flash).
    Bf16Unnormalised,
}

/// Recomputes each op of a Llama trace from the trace's own inputs.
pub struct LocalChecker<'a> {
    cfg: &'a ModelArchConfig,
    /// Weight tensor values by checkpoint name (widened to f32).
    weights: &'a dyn Fn(&str) -> Vec<f32>,
    inv_freq: Vec<f32>,
    /// K (after RoPE) and V rows of every position so far, per layer.
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
}

impl<'a> LocalChecker<'a> {
    pub fn new(cfg: &'a ModelArchConfig, weights: &'a dyn Fn(&str) -> Vec<f32>) -> Self {
        let layers = cfg.num_layers as usize;
        LocalChecker {
            cfg,
            weights,
            inv_freq: inv_freq(cfg.rope_theta, cfg.head_dim, cfg.rope_scaling.as_ref()),
            k_cache: vec![Vec::new(); layers],
            v_cache: vec![Vec::new(); layers],
        }
    }

    /// HF LlamaRMSNorm with BF16 rounding of `x · rsqrt` and of the product with the weight.
    fn rmsnorm(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
        let dim = w.len();
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(dim) {
            let mut sum_sq = 0f32;
            for v in row {
                sum_sq += v * v;
            }
            let r = 1.0 / (sum_sq / dim as f32 + self.cfg.rms_norm_eps).sqrt();
            out.extend(row.iter().zip(w).map(|(v, g)| bf(bf(v * r) * g)));
        }
        out
    }

    fn rope(&self, x: &[f32], heads: usize, p0: usize) -> Vec<f32> {
        let d = self.cfg.head_dim as usize;
        let half = d / 2;
        let mut x = x.to_vec();
        for (t, token) in x.chunks_exact_mut(heads * d).enumerate() {
            for head in token.chunks_exact_mut(d) {
                for i in 0..half {
                    let f = (p0 + t) as f32 * self.inv_freq[i];
                    let (c, s) = (bf(f.cos()), bf(f.sin()));
                    let (x1, x2) = (head[i], head[i + half]);
                    head[i] = bf(bf(x1 * c) + bf(-x2 * s));
                    head[i + half] = bf(bf(x2 * c) + bf(x1 * s));
                }
            }
        }
        x
    }

    /// Causal GQA attention of `q` (`[t, hq·d]`, positions `p0..p0 + t`) over `k`/`v`
    /// (`[p0 + t, hkv·d]`), output rounded to BF16.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], p0: usize, probs: AttnProbs) -> Vec<f32> {
        let d = self.cfg.head_dim as usize;
        let hq = self.cfg.num_attention_heads as usize;
        let hkv = self.cfg.num_kv_heads as usize;
        let t = q.len() / (hq * d);
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0f32; t * hq * d];
        for i in 0..t {
            let visible = p0 + i + 1;
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let qv = &q[(i * hq + h) * d..][..d];
                let mut scores: Vec<f32> = (0..visible)
                    .map(|j| {
                        let kr = &k[(j * hkv + kvh) * d..][..d];
                        let mut dot = 0f32;
                        for (a, b) in qv.iter().zip(kr) {
                            dot += a * b;
                        }
                        dot * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    if probs == AttnProbs::Bf16Unnormalised {
                        *s = bf(*s);
                    }
                    sum += *s;
                }
                let o = &mut out[(i * hq + h) * d..][..d];
                for (j, s) in scores.iter().enumerate() {
                    let vr = &v[(j * hkv + kvh) * d..][..d];
                    match probs {
                        AttnProbs::F32 => {
                            let p = s / sum;
                            for (acc, x) in o.iter_mut().zip(vr) {
                                *acc += p * x;
                            }
                        }
                        AttnProbs::Bf16Unnormalised => {
                            for (acc, x) in o.iter_mut().zip(vr) {
                                *acc += s * x;
                            }
                        }
                    }
                }
                if probs == AttnProbs::Bf16Unnormalised {
                    for acc in o.iter_mut() {
                        *acc /= sum;
                    }
                }
            }
        }
        out.into_iter().map(bf).collect()
    }

    /// Recomputes every op of one forward step (`trace` from a provider that ran positions
    /// `p0..p0 + t`) from the step's own recorded inputs; steps must be fed in order so the
    /// attention sees the K/V rows of earlier steps. The attention row is reported for each
    /// [`AttnProbs`] model (`attn` = f32 probabilities, `attn_p_bf16` = BF16 probabilities).
    pub fn check_step(&mut self, step: usize, p0: usize, trace: &[TraceTensor]) -> Vec<DiffRow> {
        let cfg = self.cfg;
        let hidden = cfg.hidden as usize;
        let q_dim = cfg.num_attention_heads as usize * cfg.head_dim as usize;
        let kv_dim = cfg.num_kv_heads as usize * cfg.head_dim as usize;
        let inter = cfg.intermediate as usize;
        let get = |layer: Option<usize>, name: &str| -> &TraceTensor {
            trace
                .iter()
                .find(|e| e.layer == layer && e.name == name)
                .unwrap_or_else(|| panic!("trace lacks {name} of layer {layer:?}"))
        };
        let mut rows = Vec::new();
        let mut push = |layer: Option<usize>, name: &'static str, want: &[f32], got: &[f32]| {
            rows.push(DiffRow {
                step,
                layer,
                name,
                stats: DiffStats::of(want, got),
            });
        };
        let add = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| bf(x + y)).collect()
        };
        let mut resid_in = get(None, "embed").data.clone();
        for layer in 0..cfg.num_layers as usize {
            let l = Some(layer);
            let w = |s: &str| (self.weights)(&format!("model.layers.{layer}.{s}.weight"));
            let attn_norm = &get(l, "attn_norm").data;
            push(
                l,
                "attn_norm",
                &self.rmsnorm(&resid_in, &w("input_layernorm")),
                attn_norm,
            );
            let bf_linear = |x: &[f32], name: &str, k: usize| -> Vec<f32> {
                linear(x, &w(name), k).into_iter().map(bf).collect()
            };
            let q = &get(l, "q").data;
            let k = &get(l, "k").data;
            let v = &get(l, "v").data;
            push(l, "q", &bf_linear(attn_norm, "self_attn.q_proj", hidden), q);
            push(l, "k", &bf_linear(attn_norm, "self_attn.k_proj", hidden), k);
            push(l, "v", &bf_linear(attn_norm, "self_attn.v_proj", hidden), v);
            let q_rope = &get(l, "q_rope").data;
            let k_rope = &get(l, "k_rope").data;
            push(
                l,
                "q_rope",
                &self.rope(q, cfg.num_attention_heads as usize, p0),
                q_rope,
            );
            push(
                l,
                "k_rope",
                &self.rope(k, cfg.num_kv_heads as usize, p0),
                k_rope,
            );
            // The cache holds positions 0..p0 from earlier steps; this step rewrites from p0.
            self.k_cache[layer].truncate(p0 * kv_dim);
            self.v_cache[layer].truncate(p0 * kv_dim);
            self.k_cache[layer].extend_from_slice(k_rope);
            self.v_cache[layer].extend_from_slice(v);
            let attn = &get(l, "attn").data;
            let (kc, vc) = (&self.k_cache[layer], &self.v_cache[layer]);
            push(
                l,
                "attn",
                &self.attention(q_rope, kc, vc, p0, AttnProbs::F32),
                attn,
            );
            push(
                l,
                "attn_p_bf16",
                &self.attention(q_rope, kc, vc, p0, AttnProbs::Bf16Unnormalised),
                attn,
            );
            let o_proj = &get(l, "o_proj").data;
            push(
                l,
                "o_proj",
                &bf_linear(attn, "self_attn.o_proj", q_dim),
                o_proj,
            );
            let resid_attn = &get(l, "resid_attn").data;
            push(l, "resid_attn", &add(&resid_in, o_proj), resid_attn);
            let mlp_norm = &get(l, "mlp_norm").data;
            push(
                l,
                "mlp_norm",
                &self.rmsnorm(resid_attn, &w("post_attention_layernorm")),
                mlp_norm,
            );
            let gate = &get(l, "gate").data;
            let up = &get(l, "up").data;
            push(
                l,
                "gate",
                &bf_linear(mlp_norm, "mlp.gate_proj", hidden),
                gate,
            );
            push(l, "up", &bf_linear(mlp_norm, "mlp.up_proj", hidden), up);
            let act = &get(l, "act").data;
            let want_act: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| bf(bf(g / (1.0 + (-g).exp())) * u))
                .collect();
            push(l, "act", &want_act, act);
            let down = &get(l, "down").data;
            push(l, "down", &bf_linear(act, "mlp.down_proj", inter), down);
            let resid_mlp = &get(l, "resid_mlp").data;
            push(l, "resid_mlp", &add(resid_attn, down), resid_mlp);
            resid_in = resid_mlp.clone();
        }
        let final_norm = &get(None, "final_norm").data;
        let last = &resid_in[resid_in.len() - hidden..];
        push(
            None,
            "final_norm",
            &self.rmsnorm(last, &(self.weights)("model.norm.weight")),
            final_norm,
        );
        let head = if cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        push(
            None,
            "logits",
            &linear(final_norm, &(self.weights)(head), hidden),
            &get(None, "logits").data,
        );
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_scale_with_the_binade() {
        assert_eq!(bf16_ulp(1.0), 2f32.powi(-7));
        assert_eq!(bf16_ulp(1.99), 2f32.powi(-7));
        assert_eq!(bf16_ulp(-16.0), 0.125);
        let s = DiffStats::of(&[16.0, 1.0, 0.0], &[16.125, 1.0, 0.0]);
        assert_eq!((s.differing, s.max_abs, s.max_ulps), (1, 0.125, 1.0));
        assert_eq!(s.max_ref, 16.0);
    }

    #[test]
    fn worst_per_op_keeps_the_largest_ulps() {
        let row = |step, ulps| DiffRow {
            step,
            layer: Some(0),
            name: "q",
            stats: DiffStats {
                max_ulps: ulps,
                ..DiffStats::default()
            },
        };
        let worst = worst_per_op(&[row(0, 1.0), row(1, 3.0), row(2, 2.0)]);
        assert_eq!(worst, vec![row(1, 3.0)]);
    }
}
