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
use turbine_kernels::cpu::quant::{e2m1_value, e8m0_value};
use turbine_kernels::torch_topk;

use crate::config::{ModelArchConfig, RopeScaling};
use crate::executor::decoder::unit_offset_applies;
use crate::families::LayerKind;

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
    /// Reads every tensor of `dir/model.safetensors`: BF16 as f32 values, and a gpt-oss
    /// layer's fused MXFP4 expert tensor (`<proj>_blocks` U8 with `<proj>_scales`) dequantized
    /// to the f32 values the kernels decode it to, stored under the base name.
    pub fn load(dir: &Path, cfg: &ModelArchConfig) -> Naive {
        let bytes = std::fs::read(dir.join("model.safetensors")).expect("read safetensors");
        let st = ::safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let mut raw: Vec<(String, ::safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    name,
                    view.dtype(),
                    view.shape().to_vec(),
                    view.data().to_vec(),
                )
            })
            .collect();
        raw.sort_by(|a, b| a.0.cmp(&b.0));
        let mut w = HashMap::new();
        for (name, dtype, shape, data) in &raw {
            match dtype {
                ::safetensors::Dtype::BF16 => {
                    let mut values: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    // transformers' `output · (1 + weight)` RMSNorm form, folded at load as
                    // the executor folds it (the sum is rounded back to the activation dtype).
                    if cfg.unit_offset_norms && unit_offset_applies(name) {
                        for v in &mut values {
                            *v = bf(*v + 1.0);
                        }
                    }
                    w.insert(name.clone(), values);
                }
                ::safetensors::Dtype::U8 => {
                    // `_scales` is consumed with its `_blocks` (a lookup in `raw`).
                    if name.ends_with("_scales") {
                        continue;
                    }
                    let base = name.strip_suffix("_blocks").unwrap_or_else(|| {
                        panic!("{name}: the only U8 tensors are fused expert `_blocks`/`_scales`")
                    });
                    let scales_name = format!("{base}_scales");
                    let at = shape.len();
                    let (se, sn, groups) = (shape[at - 4], shape[at - 3], shape[at - 2]);
                    let (_, _, sshape, sdata) = raw
                        .iter()
                        .find(|(n, ..)| n == &scales_name)
                        .unwrap_or_else(|| panic!("{scales_name}: missing"));
                    assert_eq!(sshape.as_slice(), &[se, sn, groups], "{scales_name}");
                    let sc: &[u8] = sdata;
                    let sk = groups * 32;
                    let mut values = Vec::with_capacity(se * sn * sk);
                    for e in 0..se {
                        for r in 0..sn {
                            for g in 0..groups {
                                let scale = e8m0_value(sc[(e * sn + r) * groups + g]);
                                let at = (((e * sn + r) * groups) + g) * 16;
                                for b in 0..16 {
                                    let byte = data[at + b];
                                    values.push(e2m1_value(byte & 0xf) * scale);
                                    values.push(e2m1_value(byte >> 4) * scale);
                                }
                            }
                        }
                    }
                    assert_eq!(values.len(), se * sn * sk, "{name}");
                    w.insert(base.to_string(), values);
                }
                other => panic!("{name}: dtype {other:?}"),
            }
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

    fn opt(&self, name: &str) -> Option<&[f32]> {
        self.w.get(name).map(|v| v.as_slice())
    }

    /// Adds a projection bias (`None` without one) to the BF16-rounded output rows of a linear
    /// layer: the add rounds again, as the executor's `add` op does.
    fn bias(&self, mut out: Vec<f32>, b: Option<&[f32]>) -> Vec<f32> {
        if let Some(b) = b {
            for (v, &b) in out.iter_mut().zip(b.iter().cycle()) {
                *v = bf(*v + b);
            }
        }
        out
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

    /// HF `rotate_half` with cos and sin times YaRN's attention factor before BF16 rounding,
    /// over the first `rotary_dim` elements of each head (partial RoPE when it is below
    /// `head_dim`, Qwen3.5).
    fn rope(&self, x: &mut [f32], heads: usize) {
        let d = self.cfg.head_dim as usize;
        let half = self.cfg.rotary_dim.max(2) as usize / 2;
        let m = self.cfg.rope_attention_factor();
        for (pos, token) in x.chunks_exact_mut(heads * d).enumerate() {
            for head in token.chunks_exact_mut(d) {
                for i in 0..half {
                    let f = pos as f32 * self.inv_freq[i];
                    let (c, s) = (bf(f.cos() * m), bf(f.sin() * m));
                    let (x1, x2) = (head[i], head[i + half]);
                    head[i] = bf(bf(x1 * c) + bf(-x2 * s));
                    head[i + half] = bf(bf(x2 * c) + bf(x1 * s));
                }
            }
        }
    }

    /// Causal GQA attention over the whole sequence; query head `h` reads KV head
    /// `h / (heads / kv_heads)`. `window` (`Some(w)`: the last `w` positions only) and the
    /// per-head sink logits (which join the softmax denominator and add nothing to the output)
    /// are gpt-oss's; the f32 sum is of the unrounded exponentials, P·V uses them
    /// rounded to BF16, the row is normalised last (CK FMHA / PyTorch CPU flash order).
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        window: Option<u32>,
        sinks: Option<&[f32]>,
    ) -> Vec<f32> {
        let d = self.cfg.head_dim as usize;
        let hq = self.cfg.num_attention_heads as usize;
        let hkv = self.cfg.num_kv_heads as usize;
        let scale = self.cfg.attention_scale();
        let mut out = vec![0f32; t * hq * d];
        for i in 0..t {
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let qv = &q[(i * hq + h) * d..][..d];
                let first = window.map_or(0, |w| i.saturating_sub(w as usize - 1));
                let mut scores: Vec<f32> = (first..=i)
                    .map(|j| {
                        let kv = &k[(j * hkv + kvh) * d..][..d];
                        let mut dot = 0f32;
                        for (a, b) in qv.iter().zip(kv) {
                            dot += a * b;
                        }
                        dot * scale
                    })
                    .collect();
                let mut max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sink = sinks.map(|s| s[h]);
                if let Some(sink) = sink {
                    max = max.max(sink);
                }
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                if let Some(sink) = sink {
                    sum += (sink - max).exp();
                }
                let o = &mut out[(i * hq + h) * d..][..d];
                for (j, s) in scores.iter().enumerate() {
                    let p = bf(*s);
                    let j = j + first;
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

    /// The gpt-oss MoE block of layer prefix `p` over rows `h` (`mlp.router.{weight,bias}`,
    /// the fused dequantized `mlp.experts.{gate_up_proj,down_proj}`): the top-k of the biased
    /// BF16-rounded logits, a softmax over those, and the clamped activation of the
    /// interleaved gate/up rows, with the same rounding as the kernels' expert path.
    fn moe_gpt_oss(&self, h: &[f32], p: &str) -> Vec<f32> {
        let hidden = self.cfg.hidden as usize;
        let moe = self.cfg.moe.expect("a MoE config");
        let (inter, top_k) = (
            moe.expert_intermediate as usize,
            moe.experts_per_token as usize,
        );
        let limit = crate::executor::decoder::hooks::GPT_OSS_SWIGLU_LIMIT;
        let router = self.get(&format!("{p}.mlp.router.weight"));
        let bias = self.get(&format!("{p}.mlp.router.bias"));
        let gate_up = self.get(&format!("{p}.mlp.experts.gate_up_proj"));
        let gu_bias = self.get(&format!("{p}.mlp.experts.gate_up_proj_bias"));
        let down = self.get(&format!("{p}.mlp.experts.down_proj"));
        let down_bias = self.get(&format!("{p}.mlp.experts.down_proj_bias"));
        let mut out = Vec::with_capacity(h.len());
        for row in h.chunks_exact(hidden) {
            let logits: Vec<f32> = Naive::linear(row, router, hidden)
                .iter()
                .zip(bias)
                .map(|(l, b)| bf(l + b))
                .collect();
            let chosen = torch_topk(&logits, top_k);
            let m = chosen
                .iter()
                .map(|&e| logits[e])
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = chosen.iter().map(|&e| (logits[e] - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let weights: HashMap<usize, f32> = chosen
                .iter()
                .zip(&exps)
                .map(|(&e, x)| (e, x / sum))
                .collect();
            let mut acc = vec![0f32; hidden];
            let mut by_expert = chosen.clone();
            by_expert.sort_unstable();
            for e in by_expert {
                let gu = &gate_up[e * 2 * inter * hidden..][..2 * inter * hidden];
                let gu_b = &gu_bias[e * 2 * inter..][..2 * inter];
                let gu_out = self.bias(Naive::linear_bf(row, gu, hidden), Some(gu_b));
                let act: Vec<f32> = gu_out
                    .chunks_exact(2)
                    .map(|p| {
                        // `gptoss_glu` (the kernels' expert activation), rounded per step.
                        let gate = p[0].min(limit);
                        let up = p[1].clamp(-limit, limit);
                        let sig = bf(1.0 / (1.0 + (-bf(gate * 1.702)).exp()));
                        bf(bf(bf(up + 1.0) * bf(gate * sig)))
                    })
                    .collect();
                let dn = &down[e * hidden * inter..][..hidden * inter];
                let dn_b = &down_bias[e * hidden..][..hidden];
                let y = self.bias(Naive::linear_bf(&act, dn, inter), Some(dn_b));
                let w = bf(weights[&e]);
                for (a, y) in acc.iter_mut().zip(y) {
                    *a = bf(*a + bf(y * w));
                }
            }
            out.extend(acc);
        }
        out
    }

    /// The full-attention block of layer `layer` over the normalised rows `h`: Q/K/V, the
    /// per-head Q/K norm, partial RoPE, causal (windowed) attention, the per-head output gate
    /// (`attn_output_gate`) and the O projection. A gated layer's `q_proj` is per-head
    /// interleaved `[q_j | g_j]`; the checkpoint's rows are de-interleaved here.
    fn full_attention(&self, h: &[f32], p: &str, t: usize, layer: u32) -> Vec<f32> {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let d = c.head_dim as usize;
        let (heads, kv_heads) = (c.num_attention_heads as usize, c.num_kv_heads as usize);
        let q_dim = heads * d;
        let w = |s: &str| self.get(&format!("{p}.{s}.weight"));
        let b = |s: &str| self.opt(&format!("{p}.{s}.bias"));
        let (mut q, gate) = if c.attn_output_gate {
            let qg = self.bias(
                Naive::linear_bf(h, w("self_attn.q_proj"), hidden),
                b("self_attn.q_proj"),
            );
            let mut q = Vec::with_capacity(t * q_dim);
            let mut gate = Vec::with_capacity(t * q_dim);
            for tok in qg.chunks_exact(2 * q_dim) {
                for hh in 0..heads {
                    q.extend_from_slice(&tok[2 * hh * d..][..d]);
                    gate.extend_from_slice(&tok[(2 * hh + 1) * d..][..d]);
                }
            }
            (q, Some(gate))
        } else {
            (
                self.bias(
                    Naive::linear_bf(h, w("self_attn.q_proj"), hidden),
                    b("self_attn.q_proj"),
                ),
                None,
            )
        };
        let mut k = self.bias(
            Naive::linear_bf(h, w("self_attn.k_proj"), hidden),
            b("self_attn.k_proj"),
        );
        let v = self.bias(
            Naive::linear_bf(h, w("self_attn.v_proj"), hidden),
            b("self_attn.v_proj"),
        );
        if self.qk_norm {
            q = self.qk_norm(&q, w("self_attn.q_norm"), heads);
            k = self.qk_norm(&k, w("self_attn.k_norm"), kv_heads);
        }
        self.rope(&mut q, heads);
        self.rope(&mut k, kv_heads);
        let window = match c.layer_kinds[layer as usize] {
            LayerKind::SlidingWindow { window } => Some(window),
            _ => None,
        };
        let sinks = self.opt(&format!("{p}.self_attn.sinks"));
        let mut attn = self.attention(&q, &k, &v, t, window, sinks);
        if let Some(gate) = gate {
            for (a, g) in attn.iter_mut().zip(&gate) {
                *a = bf(bf(1.0 / (1.0 + (-*g).exp())) * *a);
            }
        }
        self.bias(
            Naive::linear_bf(&attn, w("self_attn.o_proj"), q_dim),
            b("self_attn.o_proj"),
        )
    }

    /// The Gated DeltaNet block of layer prefix `p` over the normalised rows `h` (Qwen3.5):
    /// `in_proj_qkv` → causal depthwise conv1d (kernel 4, silu) → `q | k | v`, `in_proj_z` /
    /// `in_proj_a` / `in_proj_b`, the delta-rule recurrence (l2-normalised q/k, `beta`,
    /// per-token decay `exp(g)`, f32 state) and the gated RMSNorm.
    pub fn gdn(&self, h: &[f32], p: &str) -> Vec<f32> {
        let c = &self.cfg;
        let g = c.gdn.expect("a Gated DeltaNet config");
        let (_, normed) = self.gdn_recurrence(h, p);
        let a = format!("{p}.linear_attn");
        Naive::linear_bf(
            &normed,
            self.get(&format!("{a}.out_proj.weight")),
            g.value_dim(),
        )
    }

    /// The GDN recurrence and gated norm of layer prefix `p`: `(out, normed)`.
    fn gdn_recurrence(&self, h: &[f32], p: &str) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let g = c.gdn.expect("a Gated DeltaNet config");
        let (kh, vh, kd, vd) = (
            g.k_heads as usize,
            g.v_heads as usize,
            g.k_dim as usize,
            g.v_dim as usize,
        );
        let (conv_dim, value_dim) = (g.conv_dim(), g.value_dim());
        let t = h.len() / hidden;
        let a = format!("{p}.linear_attn");
        let w = |s: &str| self.get(&format!("{a}.{s}.weight"));
        // in_proj_qkv → causal depthwise conv1d (kernel 4, silu).
        let qkv = Naive::linear_bf(h, w("in_proj_qkv"), hidden);
        let cw = w("conv1d");
        let mut conv = vec![0f32; t * conv_dim];
        for tok in 0..t {
            for ch in 0..conv_dim {
                let mut acc = 0f32;
                for j in 0..4usize {
                    let src = tok as isize - 3 + j as isize;
                    if src >= 0 {
                        acc += cw[ch * 4 + j] * qkv[src as usize * conv_dim + ch];
                    }
                }
                conv[tok * conv_dim + ch] = bf(acc / (1.0 + (-acc).exp()));
            }
        }
        let z = Naive::linear_bf(h, w("in_proj_z"), hidden);
        let av = Naive::linear_bf(h, w("in_proj_a"), hidden);
        let bv = Naive::linear_bf(h, w("in_proj_b"), hidden);
        let a_log = self.get(&format!("{a}.A_log"));
        let dt_bias = self.get(&format!("{a}.dt_bias"));
        let mut beta = vec![0f32; t * vh];
        let mut decay = vec![0f32; t * vh];
        for tok in 0..t {
            for hh in 0..vh {
                beta[tok * vh + hh] = 1.0 / (1.0 + (-bv[tok * vh + hh]).exp());
                let x = av[tok * vh + hh] + dt_bias[hh];
                let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
                decay[tok * vh + hh] = (-a_log[hh].exp() * sp).exp();
            }
        }
        let l2 =
            |row: &[f32]| -> f32 { 1.0 / (row.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt() };
        let qscale = 1.0 / (kd as f32).sqrt();
        let key_dim = kh * kd;
        let mut state = vec![0f32; vh * kd * vd];
        let mut out = vec![0f32; t * value_dim];
        for hh in 0..vh {
            let kvh = hh / (vh / kh);
            for tok in 0..t {
                let s = &mut state[hh * kd * vd..(hh + 1) * kd * vd];
                let dec = decay[tok * vh + hh];
                for x in s.iter_mut() {
                    *x *= dec;
                }
                let qrow = &conv[tok * conv_dim + kvh * kd..][..kd];
                let krow = &conv[tok * conv_dim + key_dim + kvh * kd..][..kd];
                let qs = l2(qrow) * qscale;
                let ks = l2(krow);
                let qn: Vec<f32> = qrow.iter().map(|v| v * qs).collect();
                let kn: Vec<f32> = krow.iter().map(|v| v * ks).collect();
                let bt = beta[tok * vh + hh];
                let vrow = &conv[tok * conv_dim + 2 * key_dim + hh * vd..][..vd];
                for col in 0..vd {
                    let kv: f32 = (0..kd).map(|i| s[i * vd + col] * kn[i]).sum();
                    let delta = (vrow[col] - kv) * bt;
                    for i in 0..kd {
                        s[i * vd + col] += kn[i] * delta;
                    }
                    out[tok * value_dim + hh * vd + col] =
                        bf((0..kd).map(|i| s[i * vd + col] * qn[i]).sum());
                }
            }
        }
        // Gated RMSNorm over v_dim per value head: weight · rms(x) · silu(z).
        let norm_w = self.get(&format!("{a}.norm.weight"));
        let eps = c.rms_norm_eps;
        let mut normed = vec![0f32; t * value_dim];
        for tok in 0..t {
            for hh in 0..vh {
                let row = &out[(tok * vh + hh) * vd..][..vd];
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let r = 1.0 / (ss / vd as f32 + eps).sqrt();
                for i in 0..vd {
                    let zz = z[(tok * vh + hh) * vd + i];
                    let silu = zz / (1.0 + (-zz).exp());
                    normed[(tok * vh + hh) * vd + i] =
                        bf(bf(silu) * bf(bf(row[i] * r) * norm_w[i]));
                }
            }
        }
        (out, normed)
    }

    /// Logits of every position of `tokens` (positions `0..tokens.len()`): row `i` depends
    /// only on `tokens[..=i]` (causal attention).
    pub fn forward(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let t = tokens.len();
        let tp = c.family.0.text_prefix();
        let embed = self.get(&format!("{tp}.embed_tokens.weight"));
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&id| embed[id as usize * hidden..][..hidden].iter().copied())
            .collect();
        for layer in 0..c.num_layers {
            let p = format!("{tp}.layers.{layer}");
            let w = |s: &str| self.get(&format!("{p}.{s}.weight"));
            let h = self.rmsnorm(&x, w("input_layernorm"));
            let mixer = match c.layer_kinds[layer as usize] {
                LayerKind::LinearAttention => self.gdn(&h, &p),
                _ => self.full_attention(&h, &p, t, layer),
            };
            x = x.iter().zip(&mixer).map(|(a, b)| bf(a + b)).collect();
            let h = self.rmsnorm(&x, w("post_attention_layernorm"));
            let down = if self.opt(&format!("{p}.mlp.experts.gate_up_proj")).is_some() {
                self.moe_gpt_oss(&h, &p)
            } else if c.moe.is_some() {
                self.moe(&h, &p)
            } else {
                self.swiglu(&h, w("mlp.gate_proj"), w("mlp.up_proj"), w("mlp.down_proj"))
            };
            x = x.iter().zip(&down).map(|(a, b)| bf(a + b)).collect();
        }
        let normed = self.rmsnorm(&x, self.get(&format!("{tp}.norm.weight")));
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
/// scaling, the plain `theta^(-2i/d)` without), in f64 then f32. The exponent's dimension `d` is
/// the rotary one (`head_dim * partial_rotary_factor`, transformers'
/// `compute_default_rope_parameters`), not `head_dim`.
fn inv_freq(cfg: &ModelArchConfig) -> Vec<f32> {
    let d = f64::from(cfg.rotary_dim);
    (0..cfg.rotary_dim / 2)
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
