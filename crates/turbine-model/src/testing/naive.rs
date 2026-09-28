//! A naive decoder over a checkpoint directory (Phase 2m S-11, from the Phase 8 run-ahead's
//! `tests/families.rs`): plain f32 loops over the tensors read straight from
//! `model.safetensors` under the checkpoint's own names, recomputing the whole sequence (no KV
//! cache), driven only by [`ModelArchConfig`] fields — never by a family's executor, hooks or
//! weight slots — so a family's executor can be checked against it.
//!
//! Values are rounded to BF16 where the transformers BF16 forward materialises a BF16 tensor
//! (the executor's rounding points): every projection output, RMSNorm (normalised x, then ·
//! weight), RoPE (cos, sin, each product and each sum), the unnormalised attention
//! probabilities fed to P·V, the attention output, silu(gate) and silu·up, and each residual
//! sum. Logits stay f32.
//!
//! - Q/K norm before RoPE: over the full Q and K projections (`qk_norm`, OLMoE) or over each
//!   head of `head_dim` (`qk_norm_per_head`, Qwen3 / Qwen3-MoE).
//! - Mixture of experts (`moe`): router logits as the BF16 output of the router linear, f32
//!   softmax, the top-k set of `torch.topk` ([`turbine_kernels::torch_topk`]), weights
//!   renormalised with `norm_topk_prob`, cast to BF16; each selected expert's SwiGLU output
//!   times its weight rounded to BF16 and added in ascending expert order into a BF16 zero
//!   accumulator. Checkpoint names `mlp.gate` / `mlp.experts.<e>.{gate,up,down}_proj`
//!   (OLMoE, Qwen3-MoE) or `block_sparse_moe.gate` / `block_sparse_moe.experts.<e>.{w1,w3,w2}`
//!   (Mixtral), whichever the checkpoint holds.
//! - Rotary frequencies: `theta^(-2i/d)`, with transformers' Llama-3 scaling bands when
//!   `rope_scaling` says so, written out independently in f64.
//!
//! A test utility: every function panics on I/O or format errors.
use std::collections::HashMap;
use std::path::Path;

use half::bf16;
use turbine_kernels::torch_topk;

use crate::config::{ModelArchConfig, RopeScaling};

fn bf(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// Logits of every position of `tokens` (positions `0..tokens.len()`) of the checkpoint in
/// `dir` described by `cfg`.
pub fn forward(cfg: &ModelArchConfig, dir: &Path, tokens: &[u32]) -> Vec<Vec<f32>> {
    Naive::load(dir, cfg).forward(tokens)
}

/// The naive model. Its switches start from the configuration; tests flip them to show that a
/// feature changes the output (a mutation check).
pub struct Naive {
    cfg: ModelArchConfig,
    w: HashMap<String, Vec<f32>>,
    inv_freq: Vec<f32>,
    /// Q/K norm before RoPE (`qk_norm` or `qk_norm_per_head`).
    pub qk_norm: bool,
    /// Renormalise the selected routing weights (`norm_topk_prob`).
    pub renormalize: bool,
    /// Round the router logits to BF16 (transformers' BF16 router linear).
    pub bf16_router: bool,
    /// Use the embedding as the LM head (`tie_word_embeddings`).
    pub tied: bool,
}

impl Naive {
    /// Reads every tensor of `dir/model.safetensors` (BF16 only).
    pub fn load(dir: &Path, cfg: &ModelArchConfig) -> Naive {
        let bytes = std::fs::read(dir.join("model.safetensors")).expect("read safetensors");
        let st = ::safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let mut w = HashMap::new();
        for (name, view) in st.tensors() {
            assert_eq!(view.dtype(), ::safetensors::Dtype::BF16, "{name}");
            let values = view
                .data()
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            w.insert(name, values);
        }
        Naive {
            cfg: cfg.clone(),
            w,
            inv_freq: inv_freq(cfg),
            qk_norm: cfg.qk_norm || cfg.qk_norm_per_head,
            renormalize: cfg.moe.is_some_and(|m| m.norm_topk_prob),
            bf16_router: true,
            tied: cfg.tie_word_embeddings,
        }
    }

    fn get(&self, name: &str) -> &[f32] {
        self.w.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    /// `x [t, k] · wᵀ` with `w [n, k]`, sequential f32 accumulation.
    fn linear(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len() / k * (w.len() / k));
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

    fn linear_bf(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
        Naive::linear(x, w, k).into_iter().map(bf).collect()
    }

    /// RMSNorm over rows of `w.len()` elements (a hidden row, a whole projection or one head).
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

    /// The Q or K norm of `x` (`[t, heads · head_dim]`) with weight `w`: per head when
    /// `qk_norm_per_head` (`w` is `[head_dim]`), else over the whole projection.
    fn qk_norm(&self, x: &[f32], w: &[f32], heads: usize) -> Vec<f32> {
        let head_dim = self.cfg.head_dim as usize;
        let want = if self.cfg.qk_norm_per_head {
            head_dim
        } else {
            heads * head_dim
        };
        assert_eq!(w.len(), want, "Q/K norm weight length");
        self.rmsnorm(x, w)
    }

    fn rope(&self, x: &mut [f32], heads: usize) {
        let d = self.cfg.head_dim as usize;
        let half = d / 2;
        for (pos, token) in x.chunks_exact_mut(heads * d).enumerate() {
            for head in token.chunks_exact_mut(d) {
                for i in 0..half {
                    let f = pos as f32 * self.inv_freq[i];
                    let (c, s) = (bf(f.cos()), bf(f.sin()));
                    let (x1, x2) = (head[i], head[i + half]);
                    head[i] = bf(bf(x1 * c) + bf(-x2 * s));
                    head[i + half] = bf(bf(x2 * c) + bf(x1 * s));
                }
            }
        }
    }

    /// Causal GQA attention over the whole sequence; query head `h` reads KV head
    /// `h / (heads / kv_heads)`. The f32 sum is of the unrounded exponentials, P·V uses them
    /// rounded to BF16, the row is normalised last (CK FMHA / PyTorch CPU flash order).
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.head_dim as usize;
        let hq = self.cfg.num_attention_heads as usize;
        let hkv = self.cfg.num_kv_heads as usize;
        let scale = self.cfg.attention_scale();
        let mut out = vec![0f32; t * hq * d];
        for i in 0..t {
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let qv = &q[(i * hq + h) * d..][..d];
                let mut scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let kv = &k[(j * hkv + kvh) * d..][..d];
                        let mut dot = 0f32;
                        for (a, b) in qv.iter().zip(kv) {
                            dot += a * b;
                        }
                        dot * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let o = &mut out[(i * hq + h) * d..][..d];
                for (j, s) in scores.iter().enumerate() {
                    let p = bf(*s);
                    let vv = &v[(j * hkv + kvh) * d..][..d];
                    for (acc, x) in o.iter_mut().zip(vv) {
                        *acc += p * x;
                    }
                }
                for acc in o.iter_mut() {
                    *acc /= sum;
                }
            }
        }
        out.into_iter().map(bf).collect()
    }

    /// `down(silu(gate(h)) · up(h))` for rows `h` of `hidden`; `gate`/`up` are
    /// `[inter, hidden]`, `down` is `[hidden, inter]`.
    fn swiglu(&self, h: &[f32], gate: &[f32], up: &[f32], down: &[f32]) -> Vec<f32> {
        let hidden = self.cfg.hidden as usize;
        let inter = gate.len() / hidden;
        let g = Naive::linear_bf(h, gate, hidden);
        let u = Naive::linear_bf(h, up, hidden);
        let act: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| bf(bf(g / (1.0 + (-g).exp())) * u))
            .collect();
        Naive::linear_bf(&act, down, inter)
    }

    /// `(router, expert e's gate, up, down)` checkpoint names of layer prefix `p`, in the
    /// naming the checkpoint uses.
    fn moe_names(&self, p: &str, e: usize) -> [String; 4] {
        let mixtral = format!("{p}.block_sparse_moe.gate.weight");
        if self.w.contains_key(&mixtral) {
            let x = format!("{p}.block_sparse_moe.experts.{e}");
            [
                mixtral,
                format!("{x}.w1.weight"),
                format!("{x}.w3.weight"),
                format!("{x}.w2.weight"),
            ]
        } else {
            let x = format!("{p}.mlp.experts.{e}");
            [
                format!("{p}.mlp.gate.weight"),
                format!("{x}.gate_proj.weight"),
                format!("{x}.up_proj.weight"),
                format!("{x}.down_proj.weight"),
            ]
        }
    }

    /// The sparse MoE block of layer prefix `p` over rows `h`.
    fn moe(&self, h: &[f32], p: &str) -> Vec<f32> {
        let hidden = self.cfg.hidden as usize;
        let moe = self.cfg.moe.expect("a MoE config");
        let top_k = moe.experts_per_token as usize;
        let router = self.get(&self.moe_names(p, 0)[0]);
        let mut out = Vec::with_capacity(h.len());
        for row in h.chunks_exact(hidden) {
            let logits = if self.bf16_router {
                Naive::linear_bf(row, router, hidden)
            } else {
                Naive::linear(row, router, hidden)
            };
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
            let chosen = torch_topk(&probs, top_k);
            let total: f32 = chosen.iter().map(|&e| probs[e]).sum();
            let mut acc = vec![0f32; hidden];
            let mut by_expert = chosen.clone();
            by_expert.sort_unstable();
            for e in by_expert {
                let weight = if self.renormalize {
                    probs[e] / total
                } else {
                    probs[e]
                };
                let [_, gate, up, down] = self.moe_names(p, e);
                let y = self.swiglu(row, self.get(&gate), self.get(&up), self.get(&down));
                for (a, y) in acc.iter_mut().zip(y) {
                    *a = bf(*a + bf(y * bf(weight)));
                }
            }
            out.extend(acc);
        }
        out
    }

    /// Logits of every position of `tokens` (positions `0..tokens.len()`): row `i` depends
    /// only on `tokens[..=i]` (causal attention).
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let (heads, kv_heads) = (c.num_attention_heads as usize, c.num_kv_heads as usize);
        let q_dim = heads * c.head_dim as usize;
        let t = tokens.len();
        let embed = self.get("model.embed_tokens.weight");
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&id| embed[id as usize * hidden..][..hidden].iter().copied())
            .collect();
        for layer in 0..c.num_layers {
            let p = format!("model.layers.{layer}");
            let w = |s: &str| self.get(&format!("{p}.{s}.weight"));
            let h = self.rmsnorm(&x, w("input_layernorm"));
            let mut q = Naive::linear_bf(&h, w("self_attn.q_proj"), hidden);
            let mut k = Naive::linear_bf(&h, w("self_attn.k_proj"), hidden);
            let v = Naive::linear_bf(&h, w("self_attn.v_proj"), hidden);
            if self.qk_norm {
                q = self.qk_norm(&q, w("self_attn.q_norm"), heads);
                k = self.qk_norm(&k, w("self_attn.k_norm"), kv_heads);
            }
            self.rope(&mut q, heads);
            self.rope(&mut k, kv_heads);
            let attn = self.attention(&q, &k, &v, t);
            let o = Naive::linear_bf(&attn, w("self_attn.o_proj"), q_dim);
            x = x.iter().zip(&o).map(|(a, b)| bf(a + b)).collect();
            let h = self.rmsnorm(&x, w("post_attention_layernorm"));
            let down = if c.moe.is_some() {
                self.moe(&h, &p)
            } else {
                self.swiglu(&h, w("mlp.gate_proj"), w("mlp.up_proj"), w("mlp.down_proj"))
            };
            x = x.iter().zip(&down).map(|(a, b)| bf(a + b)).collect();
        }
        let normed = self.rmsnorm(&x, self.get("model.norm.weight"));
        let head = if self.tied {
            embed
        } else {
            self.get("lm_head.weight")
        };
        Naive::linear(&normed, head, hidden)
            .chunks_exact(head.len() / hidden)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// Logits of the last position of `tokens`.
    pub fn last_logits(&self, tokens: &[u32]) -> Vec<f32> {
        self.forward(tokens).pop().expect("at least one token")
    }
}

/// transformers' rotary inverse frequencies (`_compute_llama3_parameters` with Llama-3
/// scaling, the plain `theta^(-2i/d)` without), in f64 then f32.
fn inv_freq(cfg: &ModelArchConfig) -> Vec<f32> {
    let d = f64::from(cfg.head_dim);
    (0..cfg.head_dim / 2)
        .map(|i| {
            let f = cfg.rope_theta.powf(-(2.0 * f64::from(i)) / d);
            let scaled = match cfg.rope_scaling {
                None => f,
                Some(RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                }) => {
                    let old = f64::from(original_max_position_embeddings);
                    let wavelen = 2.0 * std::f64::consts::PI / f;
                    if wavelen < old / high_freq_factor {
                        f
                    } else if wavelen > old / low_freq_factor {
                        f / factor
                    } else {
                        let smooth = (old / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * f / factor + smooth * f
                    }
                }
                // transformers' `_compute_yarn_parameters`: a linear ramp between the
                // correction dimensions of beta_fast and beta_slow rotations blends `f`
                // (below) into `f / factor` (above).
                Some(RopeScaling::Yarn {
                    factor,
                    original_max_position_embeddings,
                    beta_fast,
                    beta_slow,
                    truncate,
                    ..
                }) => {
                    let old = f64::from(original_max_position_embeddings);
                    let corr = |rot: f64| {
                        d * (old / (rot * 2.0 * std::f64::consts::PI)).ln()
                            / (2.0 * cfg.rope_theta.ln())
                    };
                    let (mut lo, mut hi) = (corr(beta_fast), corr(beta_slow));
                    if truncate {
                        (lo, hi) = (lo.floor(), hi.ceil());
                    }
                    let (lo, mut hi) = (lo.max(0.0), hi.min(d - 1.0));
                    if lo == hi {
                        hi += 0.001;
                    }
                    let ramp = ((f64::from(i) - lo) / (hi - lo)).clamp(0.0, 1.0);
                    f / factor * ramp + f * (1.0 - ramp)
                }
            };
            scaled as f32
        })
        .collect()
}
