//! The executors on the tiny synthetic checkpoints (P1 S-8, S-12; P2 S-4, S-5, S-9, S-16): the
//! `cpu-reference` provider against an independent naive implementation (Llama and OLMoE),
//! chunked prefill and batched paged decoding against unchunked single-sequence runs, and (lab
//! only) the HIP provider against the CPU provider.
use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use half::bf16;
use turbine_core::types::{BlockId, DeviceId, KvLayout, SeqId};
use turbine_distributed::collective::{Collective, HostCollective};
use turbine_kernels::torch_topk;
use turbine_kernels::{
    ActivationConfig, ActivationContext, ActivationKernel, AddRmsnormConfig, AddRmsnormContext,
    AddRmsnormKernel, AttentionConfig, AttentionContext, AttentionKernel, ElementwiseConfig,
    ElementwiseContext, ElementwiseKernel, EmbeddingConfig, EmbeddingContext, EmbeddingKernel,
    GemmConfig, GemmContext, GemmKernel, GraphHandle, KernelError, KernelMetrics, KernelProvider,
    KernelRegistry, KvCopyConfig, KvCopyContext, KvCopyKernel, MoeExpertsConfig, MoeExpertsContext,
    MoeKernel, MoeRouteConfig, MoeRouteContext, NormConfig, NormContext, NormKernel,
    PagedAttentionContext, ProviderId, RopeConfig, RopeContext, RopeKernel, ShimContext,
    ShimLibrary, cpu_reference_provider, shim_provider,
};
use turbine_model::config::{ModelArchConfig, RopeScaling};
use turbine_model::ep::{self, ExpertPlacement};
use turbine_model::executor::{
    self, BatchInput, DecodeGraphs, DecoderExecutor, DecoderSpec, ExecutorLimits, ExecutorOptions,
    GraphBackend, Logits, LogitsSlot, ModelExecutor, ReducedRow, RowReduce, SeqSlice, SequenceKv,
    TokenFeed, TraceTensor, build_executor, graphs,
};
use turbine_model::families;
use turbine_model::pp;
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::{
    TinyOptions, TinySpec, write_tiny_llama, write_tiny_llama_with, write_tiny_olmoe,
    write_tiny_olmoe_with_head_dim,
};
use turbine_model::testing::trace::{LocalChecker, compare_traces, read_bf16_weight, render};
use turbine_model::tp;
use turbine_model::{
    EpContext, MAX_STAGING_BYTES, ModelError, PpContext, SafetensorsIndex, TpContext, WeightLoader,
    llama_slots, olmoe_slots,
};
use turbine_observability::MetricsRegistry;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{
    DeviceBuffer, DeviceMemory, DevicePtr, KvPoolView, MemInfo, MemoryError, StreamRef,
};

const SEED: u64 = 7;
const PROMPT_LEN: usize = 20;
const DECODE_STEPS: usize = 30;
const MAX_SEQ_LEN: u32 = 64;
/// KV block size of every test pool: the `kv.block_tokens` default.
const BLOCK_TOKENS: u32 = 128;
/// Sequences per batch the test executors accept.
const MAX_SEQS: u32 = 4;
/// The only head_dim the HIP attention (CK FMHA) supports (P1 S-7): GPU executor tests use a
/// tiny checkpoint with this head_dim, hidden staying 64.
const GPU_HEAD_DIM: u32 = 128;

fn write_gpu_tiny(dir: &Path) -> TinySpec {
    let opts = TinyOptions {
        head_dim: GPU_HEAD_DIM,
        ..TinyOptions::default()
    };
    write_tiny_llama_with(dir, SEED, &opts)
}

fn prompt(vocab: u32) -> Vec<u32> {
    (0..PROMPT_LEN as u32)
        .map(|i| (i * 37 + 11) % vocab)
        .collect()
}

/// An executor plus the Phase 1 single-sequence KV it runs on; derefs to the executor.
struct Single {
    exec: DecoderExecutor,
    kv: SequenceKv,
}

impl std::ops::Deref for Single {
    type Target = DecoderExecutor;
    fn deref(&self) -> &DecoderExecutor {
        &self.exec
    }
}

impl std::ops::DerefMut for Single {
    fn deref_mut(&mut self) -> &mut DecoderExecutor {
        &mut self.exec
    }
}

impl Single {
    fn run(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Logits, ModelError> {
        self.kv.forward(&mut self.exec, tokens, positions)
    }
}

fn executor(
    spec: &TinySpec,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
) -> Single {
    let kv = SequenceKv::new(&mem, spec.config.kv_layout(BLOCK_TOKENS), MAX_SEQ_LEN).expect("kv");
    Single {
        exec: paged_executor(spec, provider, mem),
        kv,
    }
}

/// Every fusion on, the opt-in projection fusion included: the Llama helpers below run it, so
/// `hip_matches_cpu` and the trace comparisons exercise the fused path on HIP (the golden test
/// runs the default options).
const ALL_FUSED: ExecutorOptions = ExecutorOptions {
    fused_ops: true,
    fused_projections: true,
};

fn paged_executor(
    spec: &TinySpec,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
) -> DecoderExecutor {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let weights =
        WeightLoader::load(&index, &llama_slots(cfg), &mem, MAX_STAGING_BYTES).expect("load");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let card = provider.card_profile();
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(cfg, BLOCK_TOKENS, ALL_FUSED),
        &metrics,
        card,
    )
    .expect("every op has a provider");
    DecoderExecutor::new(
        cfg,
        families::llama::decoder_spec(),
        weights,
        Arc::new(registry),
        mem,
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: MAX_SEQ_LEN,
            max_seqs: MAX_SEQS,
        },
        ALL_FUSED,
    )
    .expect("executor")
}

fn cpu_executor(spec: &TinySpec) -> Single {
    executor(
        spec,
        cpu_reference_provider(),
        HostMemory::new(DeviceId(0), 1 << 30),
    )
}

fn forward(exec: &mut Single, tokens: &[u32], start: u32) -> Logits {
    let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
    exec.run(tokens, &positions).expect("forward")
}

/// Greedy argmax, ties to the lower id.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ----------------------------------------------------------------------- naive reference
// Plain f32 loops over the checkpoint read straight from `model.safetensors`, recomputing the
// whole sequence each step (no KV cache). Values are rounded to BF16 where the HF BF16 forward
// materialises a BF16 tensor: every projection output, RMSNorm (normalised x, then · weight),
// RoPE (cos, sin, each product and each sum), the unnormalised attention probabilities fed to
// P·V (CK FMHA / PyTorch CPU flash attention), attention output, silu(gate) and silu·up, and
// each residual sum. Logits stay f32.
//
// OLMoE (transformers `OlmoeSparseMoeBlock`): RMSNorm over the full Q and K projections before
// RoPE; the router logits are the BF16 output of the router linear, softmax in f32, the top-k
// set PyTorch's CPU `torch.topk` selects (`turbine_kernels::torch_topk`, pinned to torch by its
// own fixture; among tied weights not always the lower expert id), weights renormalised only
// with `norm_topk_prob`, then cast to BF16; each selected expert's SwiGLU output times its
// weight is rounded to BF16 and added in ascending expert order into a BF16 zero accumulator
// (`index_add_`).

fn bf(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

struct Naive {
    cfg: ModelArchConfig,
    w: HashMap<String, Vec<f32>>,
    inv_freq: Vec<f32>,
    /// Q/K norm (OLMoE); from the config, flipped only by the mutation checks.
    qk_norm: bool,
    /// Renormalise the selected routing weights (`norm_topk_prob`); likewise.
    renormalize: bool,
    /// Round the router logits to BF16 (transformers' BF16 router linear); likewise.
    bf16_router: bool,
}

impl Naive {
    fn load(dir: &Path, cfg: &ModelArchConfig) -> Naive {
        let bytes = std::fs::read(dir.join("model.safetensors")).expect("read safetensors");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let mut w = HashMap::new();
        for (name, view) in st.tensors() {
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name}");
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
            inv_freq: naive_inv_freq(cfg),
            qk_norm: cfg.qk_norm,
            renormalize: cfg.moe.is_some_and(|m| m.norm_topk_prob),
            bf16_router: true,
        }
    }

    fn get(&self, name: &str) -> &[f32] {
        self.w.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    /// `x [t, k] · wᵀ` with `w [n, k]`, sequential f32 accumulation.
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

    fn linear_bf(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
        Naive::linear(x, w, k).into_iter().map(bf).collect()
    }

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
    /// `h / (heads / kv_heads)`.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], t: usize) -> Vec<f32> {
        let d = self.cfg.head_dim as usize;
        let hq = self.cfg.num_attention_heads as usize;
        let hkv = self.cfg.num_kv_heads as usize;
        let scale = 1.0 / (d as f32).sqrt();
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
                // CK FMHA / PyTorch CPU flash order: the f32 sum above is of the unrounded
                // exponentials, P·V uses them rounded to BF16, the row is normalised last.
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

    /// The sparse MoE block of layer prefix `p` over rows `h`.
    fn moe(&self, h: &[f32], p: &str) -> Vec<f32> {
        let hidden = self.cfg.hidden as usize;
        let moe = self.cfg.moe.expect("an OLMoE config");
        let top_k = moe.experts_per_token as usize;
        let router = self.get(&format!("{p}.mlp.gate.weight"));
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
                let x = format!("{p}.mlp.experts.{e}");
                let y = self.swiglu(
                    row,
                    self.get(&format!("{x}.gate_proj.weight")),
                    self.get(&format!("{x}.up_proj.weight")),
                    self.get(&format!("{x}.down_proj.weight")),
                );
                for (a, y) in acc.iter_mut().zip(y) {
                    *a = bf(*a + bf(y * bf(weight)));
                }
            }
            out.extend(acc);
        }
        out
    }

    /// Logits of the last position of `tokens` (positions 0..len).
    fn logits(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let q_dim = c.num_attention_heads as usize * c.head_dim as usize;
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
                q = self.rmsnorm(&q, w("self_attn.q_norm"));
                k = self.rmsnorm(&k, w("self_attn.k_norm"));
            }
            self.rope(&mut q, c.num_attention_heads as usize);
            self.rope(&mut k, c.num_kv_heads as usize);
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
        let last = self.rmsnorm(&x[(t - 1) * hidden..], self.get("model.norm.weight"));
        let head = if c.tie_word_embeddings {
            embed
        } else {
            self.get("lm_head.weight")
        };
        Naive::linear(&last, head, hidden)
    }
}

/// transformers `_compute_llama3_parameters` (or the plain `theta^(-2i/d)` without scaling,
/// OLMoE), written out independently in f64.
fn naive_inv_freq(cfg: &ModelArchConfig) -> Vec<f32> {
    let d = f64::from(cfg.head_dim);
    let Some(RopeScaling::Llama3 {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    }) = cfg.rope_scaling
    else {
        assert!(cfg.rope_scaling.is_none(), "unknown rope scaling");
        return (0..cfg.head_dim / 2)
            .map(|i| cfg.rope_theta.powf(-(2.0 * f64::from(i)) / d) as f32)
            .collect();
    };
    let old = f64::from(original_max_position_embeddings);
    (0..cfg.head_dim / 2)
        .map(|i| {
            let f = cfg.rope_theta.powf(-(2.0 * f64::from(i)) / d);
            let wavelen = 2.0 * std::f64::consts::PI / f;
            let scaled = if wavelen < old / high_freq_factor {
                f
            } else if wavelen > old / low_freq_factor {
                f / factor
            } else {
                let smooth =
                    (old / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                (1.0 - smooth) * f / factor + smooth * f
            };
            scaled as f32
        })
        .collect()
}

// --------------------------------------------------------------------------------- tests

#[test]
fn cpu_forward_matches_naive() {
    let tmp = TempDir::new("tiny-model-naive");
    check_cpu_against_naive(&write_tiny_llama(tmp.path(), SEED));
}

/// The head_dim-128 variant the GPU tests use: q/o projections wider than hidden
/// (`[512, 64]` / `[64, 512]`).
#[test]
fn cpu_forward_matches_naive_head_dim_128() {
    let tmp = TempDir::new("tiny-model-naive-hd128");
    let spec = write_gpu_tiny(tmp.path());
    assert_eq!(
        (spec.config.head_dim, spec.config.hidden),
        (GPU_HEAD_DIM, 64)
    );
    check_cpu_against_naive(&spec);
}

/// Prefill, greedy decode past the rope original length and a restart: CPU provider logits
/// within 1e-4 of the naive reference.
fn check_cpu_against_naive(spec: &TinySpec) {
    let original = match spec.config.rope_scaling {
        Some(RopeScaling::Llama3 {
            original_max_position_embeddings,
            ..
        }) => original_max_position_embeddings as usize,
        _ => panic!("tiny config has llama3 rope scaling"),
    };
    assert!(
        PROMPT_LEN + DECODE_STEPS > original,
        "decode must pass the rope original length"
    );
    let naive = Naive::load(&spec.dir, &spec.config);
    let mut exec = cpu_executor(spec);
    assert_eq!(exec.shape().vocab, spec.vocab);
    assert_eq!(exec.kv_layout().num_layers, spec.config.num_layers);

    let mut tokens = prompt(spec.vocab);
    let logits = forward(&mut exec, &tokens, 0);
    assert_eq!((logits.rows, logits.vocab), (1, spec.vocab as usize));
    let mut row = logits.row(0).to_vec();
    let diff = max_abs_diff(&row, &naive.logits(&tokens));
    assert!(diff <= 1e-4, "prefill: max abs diff {diff}");

    for step in 0..DECODE_STEPS {
        let next = argmax(&row);
        let pos = tokens.len() as u32;
        tokens.push(next);
        row = forward(&mut exec, &[next], pos).row(0).to_vec();
        let diff = max_abs_diff(&row, &naive.logits(&tokens));
        assert!(
            diff <= 1e-4,
            "decode step {step} (position {pos}): max abs diff {diff}"
        );
    }

    // A new sequence restarts at position 0 and overwrites the cache.
    let restart = forward(&mut exec, &tokens[..PROMPT_LEN], 0);
    let diff = max_abs_diff(restart.row(0), &naive.logits(&tokens[..PROMPT_LEN]));
    assert!(diff <= 1e-4, "restart: max abs diff {diff}");
}

#[test]
fn forward_rejects_invalid_batches() {
    let tmp = TempDir::new("tiny-model-invalid");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let mut exec = cpu_executor(&spec);
    fn invalid<T>(r: Result<T, ModelError>) -> String {
        match r {
            Err(ModelError::Kernel(KernelError::InvalidArgument { message })) => message,
            Err(other) => panic!("expected InvalidArgument, got {other}"),
            Ok(_) => panic!("expected InvalidArgument, got success"),
        }
    }

    // The Phase 1 single-sequence rules (SequenceKv).
    let empty = invalid(exec.run(&[], &[]));
    assert!(empty.contains("empty"), "{empty}");
    let gap = invalid(exec.run(&[1, 2], &[0, 2]));
    assert!(gap.contains("consecutive"), "{gap}");
    // Nothing is cached yet: the batch cannot start past position 0.
    let ahead = invalid(exec.run(&[1], &[3]));
    assert!(ahead.contains("cached"), "{ahead}");
    let mismatch = invalid(exec.run(&[1, 2], &[0]));
    assert!(mismatch.contains("positions"), "{mismatch}");
    let out_of_vocab = invalid(exec.run(&[spec.vocab], &[0]));
    assert!(out_of_vocab.contains("vocab"), "{out_of_vocab}");

    let full: Vec<u32> = (0..MAX_SEQ_LEN).map(|i| i % spec.vocab).collect();
    forward(&mut exec, &full, 0);
    let beyond = invalid(exec.run(&[1], &[MAX_SEQ_LEN]));
    assert!(beyond.contains("max_seq_len"), "{beyond}");

    // The ragged-batch rules of the executor itself.
    let Single { exec, kv } = &mut exec;
    let view = kv.view();
    let table: Vec<BlockId> = (0..4).map(BlockId).collect();
    let seq = |id, q_start, q_len, kv_len| SeqSlice {
        seq: SeqId(id),
        q_start,
        q_len,
        kv_len,
        block_table: &table,
        reduce: None,
    };
    let mut run = |tokens: &[u32], positions: &[u32], seqs: &[SeqSlice<'_>]| {
        invalid(exec.forward(&BatchInput {
            tokens,
            positions,
            seqs,
            kv: &view,
        }))
    };
    let too_many: Vec<u32> = vec![1; MAX_SEQ_LEN as usize + 1];
    let err = run(&too_many, &too_many, &[seq(0, 0, 65, 65)]);
    assert!(err.contains("max_batch_tokens"), "{err}");
    let err = run(&[1, 2], &[0, 1], &[]);
    assert!(err.contains("no sequences"), "{err}");
    let five: Vec<SeqSlice<'_>> = (0..5).map(|i| seq(i, i as u32, 1, 1)).collect();
    let err = run(&[1; 5], &[0; 5], &five);
    assert!(err.contains("max_seqs"), "{err}");
    let err = run(&[1, 2], &[0, 1], &[seq(0, 0, 1, 1)]);
    assert!(err.contains("cover 1"), "{err}");
    let err = run(&[1, 2], &[5, 6], &[seq(0, 0, 2, 2)]);
    assert!(err.contains("expected 0"), "{err}");
    let err = run(&[1, 2], &[0, 0], &[seq(3, 0, 1, 1), seq(3, 1, 1, 1)]);
    assert!(err.contains("twice"), "{err}");
    // One token needs one block; the table has none.
    let err = run(
        &[1],
        &[0],
        &[SeqSlice {
            block_table: &[],
            ..seq(0, 0, 1, 1)
        }],
    );
    assert!(err.contains("needs 1 blocks"), "{err}");
    let outside = [BlockId(4)];
    let err = run(
        &[1],
        &[0],
        &[SeqSlice {
            block_table: &outside,
            ..seq(0, 0, 1, 1)
        }],
    );
    assert!(err.contains("outside the pool"), "{err}");
    // A pool laid out for another block size.
    let other = KvPoolView {
        layout: turbine_core::types::KvLayout {
            block_tokens: 64,
            ..view.layout
        },
        ..view
    };
    let err = invalid(exec.forward(&BatchInput {
        tokens: &[1],
        positions: &[0],
        seqs: &[seq(0, 0, 1, 1)],
        kv: &other,
    }));
    assert!(err.contains("layout"), "{err}");
    let err = invalid(exec.copy_blocks(&view, &[BlockId(0)], &[]));
    assert!(err.contains("destinations"), "{err}");
    let err = invalid(exec.copy_blocks(&view, &[BlockId(0)], &[BlockId(9)]));
    assert!(err.contains("outside the pool"), "{err}");
}

#[test]
fn requirements_and_workspace() {
    let tmp = TempDir::new("tiny-model-reqs");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let reqs = executor::requirements(&spec.config, BLOCK_TOKENS, ALL_FUSED);
    let rendered: Vec<String> = reqs
        .iter()
        .map(|r| format!("{} {}", r.op, r.config))
        .collect();
    // Distinct configs only, every op family the forward pass runs plus the block fork.
    let mut unique = rendered.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), rendered.len(), "{rendered:#?}");
    // One block of one layer: 2 × 128 tokens × 2 kv heads × 16 head_dim × 2 bytes.
    let layer_block = 2 * 128 * 2 * 16 * 2;
    let copy = format!("copy_blocks num_layers=2 block_bytes={layer_block}");
    // Fused: Q/K/V is one n = 64 + 2·32 GEMM, gate/up one n = 2·128 GEMM.
    for want in [
        "embedding hidden=64 vocab_rows=263 dtype=bf16",
        "rmsnorm dim=64 dtype=bf16",
        "gemm n=64 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=128 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=256 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=64 k=128 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=263 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32",
        "rope head_dim=16 rotary_dim=16 q_heads=4 kv_heads=2 dtype=bf16",
        "attention_prefill_paged head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1 block_tokens=128",
        "attention_decode_paged head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1 block_tokens=128",
        "silu_mul cols=128 dtype=bf16",
        "add dtype=bf16",
        "add_rmsnorm dim=64 dtype=bf16",
        copy.as_str(),
    ] {
        assert!(
            rendered.iter().any(|r| r == want),
            "missing {want}: {rendered:#?}"
        );
    }
    assert_eq!(rendered.len(), 14, "{rendered:#?}");
    // Unfused: separate Q (n = 64, as O), K/V (n = 32) and gate/up (n = 128) GEMMs.
    let unfused: Vec<String> = executor::requirements(
        &spec.config,
        BLOCK_TOKENS,
        ExecutorOptions::from_fused_ops(false),
    )
    .iter()
    .map(|r| format!("{} {}", r.op, r.config))
    .collect();
    for (want, present) in [
        ("gemm n=32 k=64", true),
        ("gemm n=128 k=64", true),
        ("gemm n=256 k=64", false),
    ] {
        assert_eq!(
            unfused.iter().any(|r| r.starts_with(want)),
            present,
            "{want}: {unfused:#?}"
        );
    }
    assert_eq!(unfused.len(), 13, "{unfused:#?}");

    // Workspace grows linearly in the batch token count and in the sequence count.
    let ws = |t, n| executor::workspace_bytes(&spec.config, BLOCK_TOKENS, t, n);
    let per_token = ws(2, 1) - ws(1, 1);
    assert_eq!(ws(64, 1) - ws(1, 1), 63 * per_token);
    // Per token: ids + positions (i32), x/h/proj [hidden], attn [q_dim], qkv [q_dim +
    // 2·kv_dim], gate_up [2·intermediate] and act [intermediate], all bf16.
    assert_eq!(per_token, 4 + 4 + 2 * (3 * 64 + 2 * 64 + 2 * 32 + 3 * 128));
    // Per sequence: last row [hidden] bf16, logits [vocab] f32, the logits_reduce results (64
    // ids and values, lse, sampled id and logit) and inputs (temperature, uniform, top_p, mode),
    // q_indptr + kv_lens entries and a block table for max_position_embeddings tokens (i32).
    let blocks = u64::from(spec.config.max_position_embeddings.div_ceil(BLOCK_TOKENS));
    let reduce = 4 * (2 * 64 + 3) + 4 * 4;
    assert_eq!(
        ws(1, 2) - ws(1, 1),
        2 * 64 + 4 * 263 + reduce + 4 * (2 + blocks)
    );
    // Fixed: the extra q_indptr entry and inv_freq [head_dim / 2] f32.
    assert_eq!(ws(1, 1) - per_token - (ws(1, 2) - ws(1, 1)), 4 + 4 * 8);
}

/// One pool shared by the paged tests: `blocks` blocks of `layout`.
fn pool(mem: &Arc<dyn DeviceMemory>, layout: &KvLayout, blocks: u32) -> DeviceBuffer {
    let bytes = layout.block_bytes() * u64::from(blocks);
    DeviceBuffer::alloc(mem, bytes as usize).expect("pool")
}

fn pool_view<'a>(storage: &'a DeviceBuffer, layout: &KvLayout, blocks: u32) -> KvPoolView<'a> {
    let layout = *layout;
    KvPoolView {
        storage,
        layout,
        num_blocks: blocks,
        layer_stride_bytes: layout.block_bytes() / u64::from(layout.num_layers) * u64::from(blocks),
    }
}

/// A 40-token prompt prefilled into a shared pool through a scattered block table, then 10
/// greedy decode steps, gives the logits of the Phase 1 single-sequence run (contiguous
/// blocks, one sequence per batch) on the same executor.
#[test]
fn paged_llama_single_sequence() {
    const PROMPT: u32 = 40;
    const STEPS: usize = 10;
    let tmp = TempDir::new("tiny-model-paged-single");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let mut single = executor(&spec, cpu_reference_provider(), Arc::clone(&mem));
    let prompt: Vec<u32> = (0..PROMPT).map(|i| (i * 53 + 5) % spec.vocab).collect();

    // Reference: the Phase 1 path, greedy tokens chosen by it.
    let mut want = vec![forward(&mut single, &prompt, 0).row(0).to_vec()];
    let mut tokens = prompt.clone();
    for _ in 0..STEPS {
        let next = argmax(want.last().expect("row"));
        let pos = tokens.len() as u32;
        tokens.push(next);
        want.push(forward(&mut single, &[next], pos).row(0).to_vec());
    }

    // The same executor on a 12-block pool; the sequence owns blocks 9, 2 and 7 then 4, in
    // that order, and the other blocks hold unrelated K/V that must not be read.
    let Single { exec, .. } = &mut single;
    let storage = pool(&mem, exec.kv_layout(), 12);
    let noise: Vec<u8> = (0..storage.len()).map(|i| (i * 7 % 251) as u8).collect();
    storage.whole().write_bytes(&noise).expect("fill pool");
    let kv = pool_view(&storage, exec.kv_layout(), 12);
    let table = [BlockId(9), BlockId(2), BlockId(7), BlockId(4)];
    let run = |exec: &mut DecoderExecutor, tokens: &[u32], start: u32| {
        let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
        let seqs = [SeqSlice {
            seq: SeqId(42),
            q_start: 0,
            q_len: tokens.len() as u32,
            kv_len: start + tokens.len() as u32,
            block_table: &table,
            reduce: None,
        }];
        exec.forward(&BatchInput {
            tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv,
        })
        .expect("paged forward")
    };
    let logits = run(exec, &prompt, 0);
    assert_eq!((logits.rows, logits.vocab), (1, spec.vocab as usize));
    let mut got = vec![logits.row(0).to_vec()];
    for step in 0..STEPS {
        let pos = PROMPT + step as u32;
        got.push(run(exec, &[tokens[pos as usize]], pos).row(0).to_vec());
    }
    // Both runs share the executor, so the reference itself is pinned to the naive model.
    let naive = Naive::load(&spec.dir, &spec.config);
    for (step, (g, w)) in got.iter().zip(&want).enumerate() {
        let diff = max_abs_diff(g, w);
        assert!(diff <= 1e-4, "step {step}: max abs diff {diff}");
        let context = &tokens[..PROMPT as usize + step];
        let diff = max_abs_diff(w, &naive.logits(context));
        assert!(
            diff <= 1e-4,
            "step {step}: reference vs naive max abs diff {diff}"
        );
    }
}

/// One ragged batch mixing a decode and two prefills of different lengths in one pool returns,
/// per sequence, the logits of that sequence run alone; `copy_blocks` forks a sequence whose
/// continuation then matches the original's.
#[test]
fn ragged_batch_rows_match_single_sequences() {
    let tmp = TempDir::new("tiny-model-ragged");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let mut single = executor(&spec, cpu_reference_provider(), Arc::clone(&mem));
    let v = spec.vocab;
    let a: Vec<u32> = (0..21).map(|i| (i * 31 + 3) % v).collect();
    let b: Vec<u32> = (0..17).map(|i| (i * 17 + 9) % v).collect();
    let c: Vec<u32> = (0..3).map(|i| (i * 11 + 1) % v).collect();
    let alone = |single: &mut Single, tokens: &[u32]| forward(single, tokens, 0).row(0).to_vec();
    let (want_a, want_b, want_c) = (
        alone(&mut single, &a),
        alone(&mut single, &b),
        alone(&mut single, &c),
    );

    let Single { exec, .. } = &mut single;
    let storage = pool(&mem, exec.kv_layout(), 10);
    let kv = pool_view(&storage, exec.kv_layout(), 10);
    let (ta, tb, tc) = (
        [BlockId(3), BlockId(8)],
        [BlockId(0), BlockId(5)],
        [BlockId(6)],
    );
    // Sequence A's first 20 tokens are prefilled alone; its 21st arrives as a decode in the
    // ragged batch, alongside B's and C's whole prompts.
    let prefill_a = [SeqSlice {
        seq: SeqId(1),
        q_start: 0,
        q_len: 20,
        kv_len: 20,
        block_table: &ta,
        reduce: None,
    }];
    let positions: Vec<u32> = (0..20).collect();
    exec.forward(&BatchInput {
        tokens: &a[..20],
        positions: &positions,
        seqs: &prefill_a,
        kv: &kv,
    })
    .expect("prefill a");
    let tokens: Vec<u32> = [&a[20..], &b[..], &c[..]].concat();
    let positions: Vec<u32> = std::iter::once(20).chain(0..17).chain(0..3).collect();
    let seqs = [
        SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: 1,
            kv_len: 21,
            block_table: &ta,
            reduce: None,
        },
        SeqSlice {
            seq: SeqId(2),
            q_start: 1,
            q_len: 17,
            kv_len: 17,
            block_table: &tb,
            reduce: None,
        },
        SeqSlice {
            seq: SeqId(3),
            q_start: 18,
            q_len: 3,
            kv_len: 3,
            block_table: &tc,
            reduce: None,
        },
    ];
    let logits = exec
        .forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv,
        })
        .expect("ragged forward");
    assert_eq!(logits.rows, 3);
    for (s, want) in [&want_a, &want_b, &want_c].into_iter().enumerate() {
        let diff = max_abs_diff(logits.row(s), want);
        assert!(diff <= 1e-4, "sequence {s}: max abs diff {diff}");
    }

    // Fork C (3 tokens, one partial block) into block 9 and decode both copies with the same
    // token: identical logits, and C's own blocks are untouched by the fork's append.
    exec.copy_blocks(&kv, &[BlockId(6)], &[BlockId(9)])
        .expect("copy_blocks");
    let fork = [BlockId(9)];
    let decode = |exec: &mut DecoderExecutor, table: &[BlockId]| {
        let seqs = [SeqSlice {
            seq: SeqId(3),
            q_start: 0,
            q_len: 1,
            kv_len: 4,
            block_table: table,
            reduce: None,
        }];
        exec.forward(&BatchInput {
            tokens: &[7],
            positions: &[3],
            seqs: &seqs,
            kv: &kv,
        })
        .expect("decode")
        .row(0)
        .to_vec()
    };
    let original = decode(exec, &tc);
    let forked = decode(exec, &fork);
    assert_eq!(
        original, forked,
        "the fork continues exactly like the original"
    );
}

// ------------------------------------------------------------- OLMoE and both architectures

/// Any tiny checkpoint's executor on the CPU provider, through `build_executor` and the
/// architecture's requirements, for batches of up to `max_batch_tokens` tokens and
/// [`MAX_SEQS`] sequences.
fn cpu_model(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    max_batch_tokens: u32,
) -> Box<dyn ModelExecutor> {
    cpu_model_with(
        spec,
        mem,
        max_batch_tokens,
        ExecutorOptions::default(),
        cpu_reference_provider(),
        false,
    )
}

/// [`cpu_model`] run with `opts` on `provider` (the CPU reference, or the HIP provider in lab
/// tests) alone, whose registry is built from the
/// requirements it can serve ([`executor::available_requirements`]), plus the optional
/// `logits_reduce` requirement when `reduce` is set.
fn cpu_model_with(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    max_batch_tokens: u32,
    opts: ExecutorOptions,
    provider: Arc<dyn KernelProvider>,
    reduce: bool,
) -> Box<dyn ModelExecutor> {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load");
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let mut reqs =
        executor::available_requirements(cfg, BLOCK_TOKENS, opts, std::slice::from_ref(&provider));
    if reduce {
        reqs.push(executor::logits::reduce_requirement(cfg));
    }
    let card = provider.card_profile();
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
        .expect("every op has a provider");
    build_executor(
        cfg,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        BLOCK_TOKENS,
        max_batch_tokens,
        MAX_SEQS,
        opts,
    )
    .expect("executor")
}

/// The tiny Llama and the tiny OLMoE checkpoint, in subdirectories of `tmp`.
fn both_checkpoints(tmp: &TempDir) -> [TinySpec; 2] {
    [
        write_tiny_llama(&tmp.path().join("llama"), SEED),
        write_tiny_olmoe(&tmp.path().join("olmoe"), SEED),
    ]
}

/// Runs one sequence's step on `exec`: `tokens` at positions `start..`, K/V in `table`.
fn run_seq(
    exec: &mut dyn ModelExecutor,
    kv: &KvPoolView<'_>,
    table: &[BlockId],
    tokens: &[u32],
    start: u32,
) -> Vec<f32> {
    let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
    let seqs = [SeqSlice {
        seq: SeqId(1),
        q_start: 0,
        q_len: tokens.len() as u32,
        kv_len: start + tokens.len() as u32,
        block_table: table,
        reduce: None,
    }];
    let logits = exec
        .forward(&BatchInput {
            tokens,
            positions: &positions,
            seqs: &seqs,
            kv,
        })
        .expect("forward");
    assert_eq!(logits.rows, 1);
    logits.row(0).to_vec()
}

/// The cpu-reference provider under its own id, with or without the kernel ABI v2.1
/// `add_rmsnorm` family (a v2.0 library's shape), counting `add_rmsnorm` launches.
struct Probe {
    inner: Arc<dyn KernelProvider>,
    add_rmsnorm: bool,
    calls: AtomicUsize,
}

impl Probe {
    fn new(add_rmsnorm: bool) -> Arc<Probe> {
        Arc::new(Probe {
            inner: cpu_reference_provider(),
            add_rmsnorm,
            calls: AtomicUsize::new(0),
        })
    }
}

impl KernelProvider for Probe {
    fn id(&self) -> ProviderId {
        ProviderId("probe")
    }
    fn gemm(&self) -> Option<&dyn GemmKernel> {
        self.inner.gemm()
    }
    fn attention(&self) -> Option<&dyn AttentionKernel> {
        self.inner.attention()
    }
    fn norm(&self) -> Option<&dyn NormKernel> {
        self.inner.norm()
    }
    fn rope(&self) -> Option<&dyn RopeKernel> {
        self.inner.rope()
    }
    fn activation(&self) -> Option<&dyn ActivationKernel> {
        self.inner.activation()
    }
    fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
        self.inner.embedding()
    }
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
        self.inner.elementwise()
    }
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
        self.inner.kv_copy()
    }
    fn moe(&self) -> Option<&dyn MoeKernel> {
        self.inner.moe()
    }
    fn add_rmsnorm(&self) -> Option<&dyn AddRmsnormKernel> {
        self.add_rmsnorm.then_some(self as &dyn AddRmsnormKernel)
    }
}

impl AddRmsnormKernel for Probe {
    fn supports(&self, cfg: &AddRmsnormConfig) -> bool {
        self.inner.add_rmsnorm().is_some_and(|k| k.supports(cfg))
    }
    fn implementation(&self, _: &AddRmsnormConfig) -> String {
        "probe".into()
    }
    fn execute(&self, ctx: &mut AddRmsnormContext<'_>) -> Result<(), KernelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .add_rmsnorm()
            .expect("cpu-reference add_rmsnorm")
            .execute(ctx)
    }
}

/// Phase 2c S-8/S-9: on both tiny checkpoints (CPU provider) every combination of the fusions
/// (projections: one Q/K/V GEMM and, for Llama, one gate/up GEMM into row-strided views;
/// `fused_ops`: each residual add a norm follows as one `add_rmsnorm`) gives bitwise the logits
/// of the unfused op sequence, for a 40-token prefill and 10 greedy decode steps; so does the
/// fused path on a provider without `add_rmsnorm` (the ABI v2 fallback: `add` then `rmsnorm`).
/// The runs really take different op sequences: only fused projections need the
/// `[q + 2·kv, hidden]` GEMM, and `fused_ops` launches `add_rmsnorm` 2·layers − 1 times per
/// forward, the fallback never.
#[test]
fn fused_ops_match_unfused() {
    const PROMPT: usize = 40;
    const DECODE: usize = 10;
    // (fused_ops, fused_projections, provider has add_rmsnorm); the first is the reference.
    const CASES: [(bool, bool, bool); 5] = [
        (true, true, true),
        (true, true, false),
        (false, false, true),
        (true, false, true),
        (false, true, true),
    ];
    let tmp = TempDir::new("tiny-model-fused");
    for spec in both_checkpoints(&tmp) {
        let cfg = &spec.config;
        let name = cfg.hf_architecture.as_str();
        let gemm = |n: u32| {
            format!(
                "gemm n={n} k={} trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
                cfg.hidden
            )
        };
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let rendered = |opts, provider: Arc<dyn KernelProvider>| -> Vec<String> {
            executor::available_requirements(cfg, BLOCK_TOKENS, opts, &[provider])
                .iter()
                .map(|r| format!("{} {}", r.op, r.config))
                .collect()
        };
        let fused = rendered(ALL_FUSED, Probe::new(true));
        let fallback = rendered(ALL_FUSED, Probe::new(false));
        let unfused = rendered(ExecutorOptions::from_fused_ops(false), Probe::new(true));
        let norm_only = ExecutorOptions {
            fused_ops: true,
            fused_projections: false,
        };
        let norm_only = rendered(norm_only, Probe::new(true));
        let qkv = gemm(cfg.num_attention_heads * cfg.head_dim + 2 * kv_dim);
        let add_norm = format!("add_rmsnorm dim={} dtype=bf16", cfg.hidden);
        assert!(fused.contains(&qkv), "{name}: {fused:#?}");
        assert!(fused.contains(&add_norm), "{name}: {fused:#?}");
        assert!(fallback.contains(&qkv), "{name}: {fallback:#?}");
        assert!(!fallback.contains(&add_norm), "{name}: {fallback:#?}");
        assert!(unfused.contains(&gemm(kv_dim)), "{name}: {unfused:#?}");
        assert!(!unfused.contains(&add_norm), "{name}: {unfused:#?}");
        assert!(norm_only.contains(&add_norm), "{name}: {norm_only:#?}");
        assert!(norm_only.contains(&gemm(kv_dim)), "{name}: {norm_only:#?}");
        assert_ne!(norm_only, fused, "{name}");
        assert_ne!(fused, unfused, "{name}");

        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let mut runs = Vec::new();
        for (i, &(fused_ops, fused_projections, with_add_norm)) in CASES.iter().enumerate() {
            let probe = Probe::new(with_add_norm);
            let opts = ExecutorOptions {
                fused_ops,
                fused_projections,
            };
            let mut exec = cpu_model_with(&spec, &mem, MAX_SEQ_LEN, opts, probe.clone(), false);
            let mut kv = SequenceKv::new(&mem, *exec.kv_layout(), MAX_SEQ_LEN).expect("kv");
            let mut tokens: Vec<u32> = (0..PROMPT as u32)
                .map(|i| (i * 53 + 5) % spec.vocab)
                .collect();
            let positions: Vec<u32> = (0..PROMPT as u32).collect();
            let prefill = kv.forward(exec.as_mut(), &tokens, &positions);
            let mut rows = vec![prefill.expect("prefill").data];
            for _ in 0..DECODE {
                let next = argmax(rows.last().expect("prefill row"));
                let pos = tokens.len() as u32;
                tokens.push(next);
                let decode = kv.forward(exec.as_mut(), &[next], &[pos]);
                rows.push(decode.expect("decode").data);
            }
            let per_forward = 2 * cfg.num_layers as usize - 1;
            let want = if fused_ops && with_add_norm {
                (1 + DECODE) * per_forward
            } else {
                0
            };
            let calls = probe.calls.load(Ordering::SeqCst);
            assert_eq!(calls, want, "{name} case {i}: {:?}", CASES[i]);
            runs.push(rows);
        }
        for (i, other) in runs.iter().enumerate().skip(1) {
            let label = format!("case {i} {:?}", CASES[i]);
            for (step, (fused, other)) in runs[0].iter().zip(other).enumerate() {
                let same = fused.len() == other.len()
                    && fused
                        .iter()
                        .zip(other)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    same,
                    "{name} step {step}: fused differs from {label} by up to {}",
                    max_abs_diff(fused, other)
                );
            }
        }
    }
}

/// The tiny OLMoE (8 experts, top-2, `norm_topk_prob: false`, Q/K norm) on the CPU provider:
/// prefill and greedy decode logits within 1e-4 of the naive model (BF16 router logits,
/// `torch.topk` selection), which in turn is far from the same model without Q/K norm, with
/// renormalised routing weights or with F32 router logits (so none can go missing unnoticed).
#[test]
fn olmoe_cpu_forward_matches_naive() {
    let tmp = TempDir::new("tiny-model-olmoe-naive");
    let spec = write_tiny_olmoe(tmp.path(), SEED);
    let cfg = &spec.config;
    let moe = cfg.moe.expect("OLMoE has experts");
    assert_eq!((moe.num_experts, moe.experts_per_token), (8, 2));
    assert!(!moe.norm_topk_prob && cfg.qk_norm && cfg.rope_scaling.is_none());

    let reqs: Vec<String> = executor::requirements(cfg, BLOCK_TOKENS, ExecutorOptions::default())
        .iter()
        .map(|r| format!("{} {}", r.op, r.config))
        .collect();
    for want in [
        "rmsnorm dim=64 dtype=bf16",
        "gemm n=8 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32",
        "moe_route experts=8 top_k=2 renormalize=0 bf16_logits=1",
        "moe_experts hidden=64 inter=32 experts=8 top_k=2 local=0..8 dtype=bf16",
        "gemm n=263 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32",
        "rope head_dim=16 rotary_dim=16 q_heads=4 kv_heads=4 dtype=bf16",
    ] {
        assert!(reqs.iter().any(|r| r == want), "missing {want}: {reqs:#?}");
    }
    assert!(
        !reqs.iter().any(|r| r.starts_with("silu_mul")),
        "OLMoE has no dense MLP: {reqs:#?}"
    );

    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let mut exec = cpu_model(&spec, &mem, MAX_SEQ_LEN);
    assert_eq!(exec.shape().num_experts, 8);
    let mut kv = SequenceKv::new(&mem, *exec.kv_layout(), MAX_SEQ_LEN).expect("kv");
    let naive = Naive::load(&spec.dir, cfg);

    let mut tokens = prompt(spec.vocab);
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();
    let mut row = kv
        .forward(exec.as_mut(), &tokens, &positions)
        .expect("prefill")
        .row(0)
        .to_vec();
    let diff = max_abs_diff(&row, &naive.logits(&tokens));
    assert!(diff <= 1e-4, "prefill: max abs diff {diff}");
    for step in 0..10 {
        let next = argmax(&row);
        let pos = tokens.len() as u32;
        tokens.push(next);
        row = kv
            .forward(exec.as_mut(), &[next], &[pos])
            .expect("decode")
            .row(0)
            .to_vec();
        let diff = max_abs_diff(&row, &naive.logits(&tokens));
        assert!(diff <= 1e-4, "decode step {step}: max abs diff {diff}");
    }

    let mut without_qk_norm = Naive::load(&spec.dir, cfg);
    without_qk_norm.qk_norm = false;
    let diff = max_abs_diff(&row, &without_qk_norm.logits(&tokens));
    assert!(diff > 1e-3, "Q/K norm changes nothing: {diff}");
    let mut renormalised = Naive::load(&spec.dir, cfg);
    renormalised.renormalize = true;
    let diff = max_abs_diff(&row, &renormalised.logits(&tokens));
    assert!(diff > 1e-3, "renormalisation changes nothing: {diff}");
    let mut f32_router = Naive::load(&spec.dir, cfg);
    f32_router.bf16_router = false;
    let diff = max_abs_diff(&row, &f32_router.logits(&tokens));
    assert!(diff > 1e-4, "BF16 router logits change nothing: {diff}");
}

/// Host memory without device-to-device copies, like the HIP shim under kernel ABI v2 (D2D
/// arrives in v3): an executor that needs `copy_d2d` fails on it exactly as on the R9700.
struct NoD2dMemory(Arc<HostMemory>);

impl DeviceMemory for NoD2dMemory {
    fn device(&self) -> DeviceId {
        self.0.device()
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError> {
        self.0.alloc(bytes)
    }
    fn free(&self, ptr: DevicePtr) {
        self.0.free(ptr)
    }
    fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
        self.0.copy_h2d(dst, src)
    }
    fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError> {
        self.0.copy_d2h(dst, src)
    }
    fn copy_d2d(&self, _: DevicePtr, _: DevicePtr, _: usize) -> Result<(), MemoryError> {
        Err(MemoryError::Unsupported(
            "device-to-device copy needs kernel ABI v3".into(),
        ))
    }
    fn synchronize(&self) -> Result<(), MemoryError> {
        self.0.synchronize()
    }
    fn mem_info(&self) -> Result<MemInfo, MemoryError> {
        self.0.mem_info()
    }
    fn compute_stream(&self) -> StreamRef {
        self.0.compute_stream()
    }
    fn as_host(&self) -> Option<&HostMemory> {
        Some(&self.0)
    }
}

/// Both executors load and serve (prefill, decode, block fork) on memory without
/// device-to-device copies (kernel ABI v2 on the HIP shim), with the logits of the same model on
/// plain host memory bit for bit.
#[test]
fn executors_run_without_device_to_device_copies() {
    const STEPS: usize = 4;
    let tmp = TempDir::new("tiny-model-no-d2d");
    for spec in both_checkpoints(&tmp) {
        let arch = spec.config.family;
        let mut rows: Vec<Vec<Vec<f32>>> = Vec::new();
        let plain: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let no_d2d: Arc<dyn DeviceMemory> =
            Arc::new(NoD2dMemory(HostMemory::new(DeviceId(0), 1 << 30)));
        for mem in [plain, no_d2d] {
            let mut exec = cpu_model(&spec, &mem, MAX_SEQ_LEN);
            let layout = *exec.kv_layout();
            let blocks = MAX_SEQ_LEN.div_ceil(BLOCK_TOKENS);
            let storage = pool(&mem, &layout, 2 * blocks);
            let kv = pool_view(&storage, &layout, 2 * blocks);
            let table: Vec<BlockId> = (0..blocks).map(BlockId).collect();
            let fork: Vec<BlockId> = (blocks..2 * blocks).map(BlockId).collect();
            let mut tokens = prompt(spec.vocab);
            let mut row = run_seq(exec.as_mut(), &kv, &table, &tokens, 0);
            let mut seen = vec![row.clone()];
            for _ in 0..STEPS {
                let next = argmax(&row);
                let pos = tokens.len() as u32;
                tokens.push(next);
                row = run_seq(exec.as_mut(), &kv, &table, &[next], pos);
                seen.push(row.clone());
            }
            exec.copy_blocks(&kv, &table, &fork).expect("fork");
            let next = argmax(&row);
            let pos = tokens.len() as u32;
            seen.push(run_seq(exec.as_mut(), &kv, &fork, &[next], pos));
            rows.push(seen);
        }
        assert_eq!(rows[0], rows[1], "{arch:?}: logits differ without copy_d2d");
    }
}

thread_local! {
    /// Set while a [`DeviceOffsetsProvider`] kernel runs: the CPU reference ops read their
    /// operands through the same memory, and those reads are not the executor's copies.
    static IN_KERNEL: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` as kernel work: device-to-host copies it makes are not counted.
fn in_kernel<R>(f: impl FnOnce() -> R) -> R {
    IN_KERNEL.set(true);
    let r = f();
    IN_KERNEL.set(false);
    r
}

/// Host memory that counts the device-to-host copies made outside kernels (the executor's own
/// blocking reads).
struct CountingMemory {
    inner: Arc<HostMemory>,
    d2h: AtomicUsize,
}

impl CountingMemory {
    fn d2h_copies(&self) -> usize {
        self.d2h.load(Ordering::SeqCst)
    }
}

impl DeviceMemory for CountingMemory {
    fn device(&self) -> DeviceId {
        self.inner.device()
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr, MemoryError> {
        self.inner.alloc(bytes)
    }
    fn free(&self, ptr: DevicePtr) {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, dst: DevicePtr, src: &[u8]) -> Result<(), MemoryError> {
        self.inner.copy_h2d(dst, src)
    }
    fn copy_d2h(&self, dst: &mut [u8], src: DevicePtr) -> Result<(), MemoryError> {
        if !IN_KERNEL.get() {
            self.d2h.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.copy_d2h(dst, src)
    }
    fn copy_d2d(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<(), MemoryError> {
        self.inner.copy_d2d(dst, src, bytes)
    }
    fn synchronize(&self) -> Result<(), MemoryError> {
        self.inner.synchronize()
    }
    fn mem_info(&self) -> Result<MemInfo, MemoryError> {
        self.inner.mem_info()
    }
    fn compute_stream(&self) -> StreamRef {
        self.inner.compute_stream()
    }
    fn as_host(&self) -> Option<&HostMemory> {
        Some(&self.inner)
    }
}

/// A family wrapper of [`DeviceOffsetsProvider`]: delegates to the inner provider's family,
/// running `execute` as kernel work.
macro_rules! kernel_family {
    ($name:ident, $family:ident, $kernel:ident, $cfg:ty, $ctx:ident) => {
        struct $name(Arc<dyn KernelProvider>);

        impl $name {
            fn inner(&self) -> &dyn $kernel {
                self.0.$family().expect(stringify!($family))
            }
        }

        impl $kernel for $name {
            fn supports(&self, cfg: &$cfg) -> bool {
                self.inner().supports(cfg)
            }
            fn implementation(&self, cfg: &$cfg) -> String {
                self.inner().implementation(cfg)
            }
            fn execute(&self, ctx: &mut $ctx<'_>) -> Result<(), KernelError> {
                in_kernel(|| self.inner().execute(ctx))
            }
        }
    };
}

kernel_family!(Gemm, gemm, GemmKernel, GemmConfig, GemmContext);
kernel_family!(Norm, norm, NormKernel, NormConfig, NormContext);
kernel_family!(Rope, rope, RopeKernel, RopeConfig, RopeContext);
kernel_family!(
    Activation,
    activation,
    ActivationKernel,
    ActivationConfig,
    ActivationContext
);
kernel_family!(
    Embedding,
    embedding,
    EmbeddingKernel,
    EmbeddingConfig,
    EmbeddingContext
);
kernel_family!(
    Elementwise,
    elementwise,
    ElementwiseKernel,
    ElementwiseConfig,
    ElementwiseContext
);
kernel_family!(KvCopy, kv_copy, KvCopyKernel, KvCopyConfig, KvCopyContext);

struct Attention(Arc<dyn KernelProvider>);

impl Attention {
    fn inner(&self) -> &dyn AttentionKernel {
        self.0.attention().expect("attention")
    }
}

impl AttentionKernel for Attention {
    fn supports(&self, cfg: &AttentionConfig) -> bool {
        self.inner().supports(cfg)
    }
    fn implementation(&self, cfg: &AttentionConfig) -> String {
        self.inner().implementation(cfg)
    }
    fn execute(&self, ctx: &mut AttentionContext<'_>) -> Result<(), KernelError> {
        in_kernel(|| self.inner().execute(ctx))
    }
    fn execute_paged(&self, ctx: &mut PagedAttentionContext<'_>) -> Result<(), KernelError> {
        in_kernel(|| self.inner().execute_paged(ctx))
    }
}

/// Routed rows up to which [`DeviceOffsetsProvider`] reads the expert offsets on the device,
/// like the HIP small-m path.
const DEVICE_OFFSETS_MAX_ROWS: usize = 512;

/// The CPU provider, except that `moe_experts` needs the host offsets above `max_rows` routed
/// rows only ([`DEVICE_OFFSETS_MAX_ROWS`] is the HIP small-m contract; 0 is the Phase 2
/// contract, always). It checks that the executor passes the host offsets exactly when they are
/// needed, and runs every op as kernel work for [`CountingMemory`].
struct DeviceOffsetsProvider {
    inner: Arc<dyn KernelProvider>,
    max_rows: usize,
    gemm: Gemm,
    attention: Attention,
    norm: Norm,
    rope: Rope,
    activation: Activation,
    embedding: Embedding,
    elementwise: Elementwise,
    kv_copy: KvCopy,
}

impl DeviceOffsetsProvider {
    fn new(inner: Arc<dyn KernelProvider>, max_rows: usize) -> DeviceOffsetsProvider {
        DeviceOffsetsProvider {
            max_rows,
            gemm: Gemm(Arc::clone(&inner)),
            attention: Attention(Arc::clone(&inner)),
            norm: Norm(Arc::clone(&inner)),
            rope: Rope(Arc::clone(&inner)),
            activation: Activation(Arc::clone(&inner)),
            embedding: Embedding(Arc::clone(&inner)),
            elementwise: Elementwise(Arc::clone(&inner)),
            kv_copy: KvCopy(Arc::clone(&inner)),
            inner,
        }
    }

    fn inner(&self) -> &dyn MoeKernel {
        self.inner.moe().expect("moe")
    }
}

impl MoeKernel for DeviceOffsetsProvider {
    fn supports_route(&self, cfg: &MoeRouteConfig) -> bool {
        self.inner().supports_route(cfg)
    }
    fn supports_experts(&self, cfg: &MoeExpertsConfig) -> bool {
        self.inner().supports_experts(cfg)
    }
    fn implementation_route(&self, cfg: &MoeRouteConfig) -> String {
        self.inner().implementation_route(cfg)
    }
    fn implementation_experts(&self, cfg: &MoeExpertsConfig) -> String {
        self.inner().implementation_experts(cfg)
    }
    fn route(&self, ctx: &mut MoeRouteContext<'_>) -> Result<(), KernelError> {
        in_kernel(|| self.inner().route(ctx))
    }
    fn experts(&self, ctx: &mut MoeExpertsContext<'_>) -> Result<(), KernelError> {
        let rows = ctx.cfg.routed_rows(ctx.x.shape[0]);
        assert_eq!(
            ctx.host_expert_offsets.is_empty(),
            !self.needs_host_offsets(&ctx.cfg, rows),
            "host offsets passed although not needed, or missing although needed ({rows} rows)"
        );
        in_kernel(|| self.inner().experts(ctx))
    }
    fn needs_host_offsets(&self, _cfg: &MoeExpertsConfig, routed_rows: usize) -> bool {
        routed_rows > self.max_rows
    }
}

impl KernelProvider for DeviceOffsetsProvider {
    fn id(&self) -> ProviderId {
        ProviderId("cpu-device-offsets")
    }
    fn gemm(&self) -> Option<&dyn GemmKernel> {
        Some(&self.gemm)
    }
    fn attention(&self) -> Option<&dyn AttentionKernel> {
        Some(&self.attention)
    }
    fn norm(&self) -> Option<&dyn NormKernel> {
        Some(&self.norm)
    }
    fn rope(&self) -> Option<&dyn RopeKernel> {
        Some(&self.rope)
    }
    fn activation(&self) -> Option<&dyn ActivationKernel> {
        Some(&self.activation)
    }
    fn embedding(&self) -> Option<&dyn EmbeddingKernel> {
        Some(&self.embedding)
    }
    fn elementwise(&self) -> Option<&dyn ElementwiseKernel> {
        Some(&self.elementwise)
    }
    fn kv_copy(&self) -> Option<&dyn KvCopyKernel> {
        Some(&self.kv_copy)
    }
    fn moe(&self) -> Option<&dyn MoeKernel> {
        Some(self)
    }
}

/// One sequence's part of a test batch: its index, its new tokens and the position of the
/// first of them.
type Part = (usize, Vec<u32>, u32);

/// P2c S-11: with a provider whose `moe_experts` reads the expert offsets on the device for up
/// to 512 routed rows, an OLMoE decode-only forward of 8 sequences makes exactly one
/// device-to-host copy (the logits), while a mixed forward of more than 512 routed rows reads
/// the offsets back once per layer, plus the logits. A provider that always needs them (the
/// Phase 2 contract) reads them in every forward; the logits of both are equal bit for bit.
#[test]
fn olmoe_decode_single_device_copy() {
    const SEQS: usize = 8;
    const PROMPT: u32 = 4;
    const LONG: u32 = 296;
    const MAX_TOKENS: u32 = 512;
    let tmp = TempDir::new("tiny-model-olmoe-d2h");
    let spec = write_tiny_olmoe(tmp.path(), SEED);
    let cfg = &spec.config;
    let top_k = cfg.moe.expect("tiny OLMoE has experts").experts_per_token as usize;
    let layers = cfg.num_layers as usize;
    let v = spec.vocab;

    let prompt_of = |s: usize| -> Vec<u32> {
        (0..PROMPT)
            .map(|i| (i * 13 + s as u32 * 7 + 1) % v)
            .collect()
    };
    let prefill: Vec<Part> = (0..SEQS).map(|s| (s, prompt_of(s), 0)).collect();
    let decode: Vec<Part> = (0..SEQS)
        .map(|s| (s, vec![(s as u32 * 5 + 2) % v], PROMPT))
        .collect();
    let long: Vec<u32> = (0..LONG).map(|i| (i * 31 + 3) % v).collect();
    let mut mixed: Vec<Part> = vec![(SEQS, long, 0)];
    mixed.extend((0..4).map(|s| (s, vec![(s as u32 * 3 + 5) % v], PROMPT + 1)));
    let mixed_tokens: usize = mixed.iter().map(|(_, t, _)| t.len()).sum();
    assert!(mixed_tokens * top_k > DEVICE_OFFSETS_MAX_ROWS);
    assert!(SEQS * PROMPT as usize * top_k <= DEVICE_OFFSETS_MAX_ROWS);

    let mut all_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut counts: Vec<[usize; 3]> = Vec::new();
    for max_rows in [DEVICE_OFFSETS_MAX_ROWS, 0] {
        let counting = Arc::new(CountingMemory {
            inner: HostMemory::new(DeviceId(0), 1 << 30),
            d2h: AtomicUsize::new(0),
        });
        let mem: Arc<dyn DeviceMemory> = counting.clone();
        let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
        let weights =
            WeightLoader::load(&index, &olmoe_slots(cfg), &mem, MAX_STAGING_BYTES).expect("load");
        let provider: Arc<dyn KernelProvider> = Arc::new(DeviceOffsetsProvider::new(
            cpu_reference_provider(),
            max_rows,
        ));
        let order = [provider.id()];
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let opts = ExecutorOptions::default();
        let reqs = executor::available_requirements(
            cfg,
            BLOCK_TOKENS,
            opts,
            std::slice::from_ref(&provider),
        );
        let card = provider.card_profile();
        let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider");
        let mut exec = build_executor(
            cfg,
            weights,
            Arc::new(registry),
            Arc::clone(&mem),
            BLOCK_TOKENS,
            MAX_TOKENS,
            SEQS as u32,
            opts,
        )
        .expect("executor");

        // Sequences 0..8 hold PROMPT tokens and then decode; sequence 8 is the long prompt.
        let per_seq = (LONG + 1).div_ceil(BLOCK_TOKENS);
        let blocks = per_seq * (SEQS as u32 + 1);
        let storage = pool(&mem, exec.kv_layout(), blocks);
        let kv = pool_view(&storage, exec.kv_layout(), blocks);
        let tables: Vec<Vec<BlockId>> = (0..=SEQS as u32)
            .map(|s| (s * per_seq..(s + 1) * per_seq).map(BlockId).collect())
            .collect();
        let mut seen = Vec::new();
        let mut step = |parts: &[Part]| -> usize {
            let before = counting.d2h_copies();
            let mut tokens = Vec::new();
            let mut positions = Vec::new();
            let mut seqs = Vec::new();
            for (s, toks, start) in parts {
                seqs.push(SeqSlice {
                    seq: SeqId(*s as u64 + 1),
                    q_start: tokens.len() as u32,
                    q_len: toks.len() as u32,
                    kv_len: start + toks.len() as u32,
                    block_table: &tables[*s],
                    reduce: None,
                });
                tokens.extend_from_slice(toks);
                positions.extend(*start..start + toks.len() as u32);
            }
            let logits = exec
                .forward(&BatchInput {
                    tokens: &tokens,
                    positions: &positions,
                    seqs: &seqs,
                    kv: &kv,
                })
                .expect("forward");
            assert_eq!(logits.rows, parts.len());
            seen.extend((0..logits.rows).map(|r| logits.row(r).to_vec()));
            counting.d2h_copies() - before
        };
        let copies = [step(&prefill), step(&decode), step(&mixed)];
        counts.push(copies);
        all_logits.push(seen);
    }
    assert_eq!(
        counts[0],
        [1, 1, layers + 1],
        "device offsets: copies of the prefill, decode and mixed forwards"
    );
    assert_eq!(
        counts[1],
        [layers + 1; 3],
        "host offsets: copies of the prefill, decode and mixed forwards"
    );
    assert_eq!(
        all_logits[0], all_logits[1],
        "skipping the host offsets changes no logit"
    );
}

/// For both tiny checkpoints, a 300-token prompt prefilled in chunks of 64 (each chunk
/// attending to the K/V the earlier chunks left in the pool) gives the last-position logits of
/// a single-chunk prefill within 1e-4.
#[test]
fn chunked_prefill_matches_unchunked() {
    const LONG: u32 = 300;
    const CHUNK: u32 = 64;
    let tmp = TempDir::new("tiny-model-chunked");
    for spec in both_checkpoints(&tmp) {
        let arch = spec.config.family;
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let mut exec = cpu_model(&spec, &mem, LONG);
        let layout = *exec.kv_layout();
        let blocks = LONG.div_ceil(BLOCK_TOKENS);
        let storage = pool(&mem, &layout, 2 * blocks);
        let kv = pool_view(&storage, &layout, 2 * blocks);
        // The single chunk uses blocks 0.., the chunked run the other half in reverse order.
        let whole_table: Vec<BlockId> = (0..blocks).map(BlockId).collect();
        let chunk_table: Vec<BlockId> = (0..blocks).map(|b| BlockId(2 * blocks - 1 - b)).collect();
        let prompt: Vec<u32> = (0..LONG).map(|i| (i * 41 + 7) % spec.vocab).collect();

        let whole = run_seq(exec.as_mut(), &kv, &whole_table, &prompt, 0);
        let mut last = Vec::new();
        for start in (0..LONG).step_by(CHUNK as usize) {
            let end = (start + CHUNK).min(LONG);
            last = run_seq(
                exec.as_mut(),
                &kv,
                &chunk_table,
                &prompt[start as usize..end as usize],
                start,
            );
        }
        let diff = max_abs_diff(&last, &whole);
        assert!(diff <= 1e-4, "{arch:?}: max abs diff {diff}");
    }
}

/// For both tiny checkpoints, 4 sequences with prompts of 3, 17, 40 and 65 tokens prefilled as
/// one ragged batch and decoded together over interleaved blocks of one pool (unused blocks
/// hold noise) give, per sequence and step, the logits of that sequence run alone on its own
/// contiguous Phase 1 KV within 1e-4.
#[test]
fn paged_matches_contiguous() {
    const LENS: [u32; 4] = [3, 17, 40, 65];
    const STEPS: u32 = 6;
    const MAX_TOKENS: u32 = 128;
    // Twice the blocks the sequences can need, a power of two so `i * 5 % POOL_BLOCKS` below
    // visits every block.
    const POOL_BLOCKS: u32 =
        (2 * LENS.len() as u32 * MAX_TOKENS.div_ceil(BLOCK_TOKENS)).next_power_of_two();
    let tmp = TempDir::new("tiny-model-paged-contiguous");
    for spec in both_checkpoints(&tmp) {
        let arch = spec.config.family;
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let prompts: Vec<Vec<u32>> = LENS
            .iter()
            .enumerate()
            .map(|(s, &len)| {
                (0..len)
                    .map(|i| (i * 29 + 13 * s as u32 + 1) % spec.vocab)
                    .collect()
            })
            .collect();

        // Reference: each sequence alone, greedy, on its own contiguous KV.
        let mut single = cpu_model(&spec, &mem, MAX_TOKENS);
        let mut want: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut seq_tokens: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut kv = SequenceKv::new(&mem, *single.kv_layout(), MAX_TOKENS).expect("kv");
            let positions: Vec<u32> = (0..p.len() as u32).collect();
            let mut rows = vec![
                kv.forward(single.as_mut(), p, &positions)
                    .expect("prefill")
                    .row(0)
                    .to_vec(),
            ];
            let mut tokens = p.clone();
            for _ in 0..STEPS {
                let next = argmax(rows.last().expect("row"));
                let pos = tokens.len() as u32;
                tokens.push(next);
                rows.push(
                    kv.forward(single.as_mut(), &[next], &[pos])
                        .expect("decode")
                        .row(0)
                        .to_vec(),
                );
            }
            want.push(rows);
            seq_tokens.push(tokens);
        }

        // Batched: one pool, blocks handed out round-robin in a scattered order.
        let mut exec = cpu_model(&spec, &mem, MAX_TOKENS);
        let layout = *exec.kv_layout();
        let storage = pool(&mem, &layout, POOL_BLOCKS);
        let noise: Vec<u8> = (0..storage.len()).map(|i| (i * 13 % 239) as u8).collect();
        storage.whole().write_bytes(&noise).expect("fill pool");
        let kv = pool_view(&storage, &layout, POOL_BLOCKS);
        let need: Vec<usize> = LENS
            .iter()
            .map(|&l| (l + STEPS).div_ceil(BLOCK_TOKENS) as usize)
            .collect();
        let mut free = (0..POOL_BLOCKS).map(|i| BlockId(i * 5 % POOL_BLOCKS));
        let mut tables: Vec<Vec<BlockId>> = vec![Vec::new(); LENS.len()];
        while tables.iter().zip(&need).any(|(t, &n)| t.len() < n) {
            for (t, &n) in tables.iter_mut().zip(&need) {
                if t.len() < n {
                    t.push(free.next().expect("enough blocks"));
                }
            }
        }

        let step =
            |exec: &mut dyn ModelExecutor, tokens: &[u32], q_lens: &[u32], kv_lens: &[u32]| {
                let mut positions = Vec::new();
                let mut seqs = Vec::new();
                let mut q_start = 0;
                for s in 0..LENS.len() {
                    positions.extend(kv_lens[s] - q_lens[s]..kv_lens[s]);
                    seqs.push(SeqSlice {
                        seq: SeqId(100 + s as u64),
                        q_start,
                        q_len: q_lens[s],
                        kv_len: kv_lens[s],
                        block_table: &tables[s],
                        reduce: None,
                    });
                    q_start += q_lens[s];
                }
                exec.forward(&BatchInput {
                    tokens,
                    positions: &positions,
                    seqs: &seqs,
                    kv: &kv,
                })
                .expect("batched forward")
            };
        let tokens: Vec<u32> = prompts.concat();
        let logits = step(exec.as_mut(), &tokens, &LENS, &LENS);
        assert_eq!(logits.rows, LENS.len());
        for (s, rows) in want.iter().enumerate() {
            let diff = max_abs_diff(logits.row(s), &rows[0]);
            assert!(diff <= 1e-4, "{arch:?} prefill of sequence {s}: {diff}");
        }
        for i in 0..STEPS {
            let tokens: Vec<u32> = (0..LENS.len())
                .map(|s| seq_tokens[s][(LENS[s] + i) as usize])
                .collect();
            let kv_lens: Vec<u32> = LENS.iter().map(|&l| l + i + 1).collect();
            let logits = step(exec.as_mut(), &tokens, &[1; 4], &kv_lens);
            for (s, rows) in want.iter().enumerate() {
                let diff = max_abs_diff(logits.row(s), &rows[i as usize + 1]);
                assert!(diff <= 1e-4, "{arch:?} decode {i} of sequence {s}: {diff}");
            }
        }
    }
}

/// Max |Δ logit| between the HIP and cpu-reference providers on the tiny checkpoint. With the
/// cpu-reference attention in CK's order (BF16 unnormalised probabilities for P·V, f32 row
/// sum), every op models the HIP numerics and the measured worst case over 32 steps on an R9700
/// is 3.8e-6 (0.097 while the cpu-reference kept f32 probabilities). 1e-4 leaves a 25× margin
/// for f32 accumulation order and is ~1000× below the error one numerics-model mismatch caused.
const HIP_MAX_ABS_LOGIT_DIFF: f32 = 1e-4;

/// A context on the first AMD device through the library `TURBINE_KERNEL_LIBRARY` names.
fn hip_context() -> Arc<ShimContext> {
    turbine_kernels::test_support::open_context("hip")
}

/// Decode graphs on `ctx` for an executor of up to `max_seqs` sequences.
fn hip_graphs(ctx: &Arc<ShimContext>, max_seqs: u32) -> DecodeGraphs {
    assert!(
        ctx.library().supports_graphs(),
        "the HIP kernel library exports the ABI v2.1 graph functions"
    );
    let backend: Arc<dyn GraphBackend<Graph = GraphHandle>> = Arc::<ShimContext>::clone(ctx);
    DecodeGraphs::new(backend, graphs::capacity_for(max_seqs))
}

/// Lab only (Task 21): HIP provider logits on the head_dim-128 tiny checkpoint match the CPU
/// provider within [`HIP_MAX_ABS_LOGIT_DIFF`] and 32 greedy tokens are identical. The HIP
/// executor runs with decode graphs on (P2c S-10), so the decode steps after the second replay
/// a captured graph.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_matches_cpu() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let ctx = hip_context();

    let tmp = TempDir::new("tiny-model-hip");
    let spec = write_gpu_tiny(tmp.path());
    let mut cpu = cpu_executor(&spec);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = executor(&spec, shim_provider(ctx.clone()), mem);
    hip.set_decode_graphs(Some(hip_graphs(&ctx, MAX_SEQS)));

    let mut tokens = prompt(spec.vocab);
    let mut cpu_row = forward(&mut cpu, &tokens, 0).row(0).to_vec();
    let mut hip_row = forward(&mut hip, &tokens, 0).row(0).to_vec();
    let mut worst = 0f32;
    for step in 0..32 {
        let diff = max_abs_diff(&hip_row, &cpu_row);
        worst = worst.max(diff);
        assert!(
            diff <= HIP_MAX_ABS_LOGIT_DIFF,
            "step {step}: max abs diff {diff}"
        );
        let (c, h) = (argmax(&cpu_row), argmax(&hip_row));
        assert_eq!(h, c, "greedy token {step} differs");
        let pos = tokens.len() as u32;
        tokens.push(c);
        cpu_row = forward(&mut cpu, &[c], pos).row(0).to_vec();
        hip_row = forward(&mut hip, &[c], pos).row(0).to_vec();
    }
    let graphs = hip.graph_counters();
    assert!(
        graphs.replayed > 0 && graphs.capture_failed == 0,
        "decode graphs were used: {graphs:?}"
    );
    println!("hip_matches_cpu: max abs logit diff over 32 steps {worst}; graphs {graphs:?}");
}

/// Decode steps of [`hip_decode_graph_matches_eager`] per batch.
const GRAPH_DECODE_STEPS: u32 = 40;

/// A tiny checkpoint's executor on the HIP provider for batches of up to 16 sequences of 128
/// tokens over KV blocks of `block_tokens`, reducing logits rows on the device when asked.
fn hip_graph_executor(
    spec: &TinySpec,
    ctx: &Arc<ShimContext>,
    block_tokens: u32,
) -> Box<dyn ModelExecutor> {
    let cfg = &spec.config;
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let provider = shim_provider(ctx.clone());
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("load");
    let opts = ExecutorOptions::default();
    let mut reqs =
        executor::available_requirements(cfg, block_tokens, opts, std::slice::from_ref(&provider));
    reqs.push(executor::logits::reduce_requirement(cfg));
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let card = provider.card_profile();
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
        .expect("every op has a provider");
    build_executor(
        cfg,
        weights,
        Arc::new(registry),
        mem,
        block_tokens,
        128,
        16,
        opts,
    )
    .expect("executor")
}

/// `seqs` sequences prefilled one by one (prompts of 100 to 119 tokens), then
/// [`GRAPH_DECODE_STEPS`] decode steps of all of them together, each sequence's blocks
/// scattered through the pool; even steps reduce every row on the device (a draw at
/// temperature 0.7, half the rows with a top-p nucleus), odd steps return full rows. Returns
/// every decode step's logits.
fn graph_decode_run(
    exec: &mut dyn ModelExecutor,
    kv: &KvPoolView<'_>,
    vocab: u32,
    seqs: usize,
) -> Vec<Logits> {
    let bt = kv.layout.block_tokens;
    let prompt_len = |s: usize| 100 + (s as u32 * 7) % 20;
    let per_seq = (119 + GRAPH_DECODE_STEPS).div_ceil(bt);
    // Sequence s owns blocks s, s + seqs, s + 2·seqs, … (interleaved), in reverse order.
    let tables: Vec<Vec<BlockId>> = (0..seqs)
        .map(|s| {
            (0..per_seq)
                .rev()
                .map(|j| BlockId(s as u32 + j * seqs as u32))
                .collect()
        })
        .collect();
    let token = |s: usize, p: u32| (p * 31 + 7 * s as u32 + 1) % vocab;
    for (s, table) in tables.iter().enumerate() {
        let len = prompt_len(s);
        let tokens: Vec<u32> = (0..len).map(|p| token(s, p)).collect();
        let positions: Vec<u32> = (0..len).collect();
        let slice = [SeqSlice {
            seq: SeqId(s as u64 + 1),
            q_start: 0,
            q_len: len,
            kv_len: len,
            block_table: table,
            reduce: None,
        }];
        exec.forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &slice,
            kv,
        })
        .expect("prefill");
    }
    let mut out = Vec::new();
    for step in 0..GRAPH_DECODE_STEPS {
        let positions: Vec<u32> = (0..seqs).map(|s| prompt_len(s) + step).collect();
        let tokens: Vec<u32> = (0..seqs).map(|s| token(s, positions[s])).collect();
        let slices: Vec<SeqSlice<'_>> = (0..seqs)
            .map(|s| SeqSlice {
                seq: SeqId(s as u64 + 1),
                q_start: s as u32,
                q_len: 1,
                kv_len: positions[s] + 1,
                block_table: &tables[s],
                reduce: (step % 2 == 0).then_some(RowReduce {
                    top_n: 5,
                    temperature: 0.7,
                    uniform: Some(0.1 + 0.05 * s as f32),
                    top_p: if s % 2 == 0 { 1.0 } else { 0.9 },
                }),
            })
            .collect();
        out.push(
            exec.forward(&BatchInput {
                tokens: &tokens,
                positions: &positions,
                seqs: &slices,
                kv,
            })
            .expect("decode"),
        );
    }
    out
}

/// Lab only (P2c S-10): on the head_dim-128 tiny Llama and OLMoE over KV blocks of 16 (the
/// Turbine paged kernel) and 128 tokens (CK), 40 decode steps of batches of 1, 5 and 16
/// sequences whose block tables grow across block boundaries give bitwise the logits (full
/// rows and device reductions) with decode graphs on as with every op launched eagerly, and
/// the graphs are replayed. Breaks if a replay reads stale metadata (tokens, positions,
/// `kv_lens`, block tables, reduction inputs) or a graph is never used.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_decode_graph_matches_eager() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let ctx = hip_context();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let tmp = TempDir::new("tiny-model-hip-graphs");
    let opts = TinyOptions {
        head_dim: GPU_HEAD_DIM,
        ..TinyOptions::default()
    };
    let specs = [
        write_tiny_llama_with(&tmp.path().join("llama"), SEED, &opts),
        write_tiny_olmoe_with_head_dim(&tmp.path().join("olmoe"), SEED, GPU_HEAD_DIM),
    ];
    for spec in &specs {
        for block_tokens in [16, 128] {
            let arch = spec.config.family;
            let mut exec = hip_graph_executor(spec, &ctx, block_tokens);
            assert!(exec.reduces_logits(), "{arch:?}");
            let layout = *exec.kv_layout();
            for seqs in [1usize, 5, 16] {
                let blocks = seqs as u32 * (119 + GRAPH_DECODE_STEPS).div_ceil(block_tokens);
                let storage = pool(&mem, &layout, blocks);
                let kv = pool_view(&storage, &layout, blocks);
                exec.set_decode_graphs(None);
                let eager = graph_decode_run(exec.as_mut(), &kv, spec.vocab, seqs);
                exec.set_decode_graphs(Some(hip_graphs(&ctx, 16)));
                let graphed = graph_decode_run(exec.as_mut(), &kv, spec.vocab, seqs);
                for (step, (e, g)) in eager.iter().zip(&graphed).enumerate() {
                    assert_eq!(
                        e, g,
                        "{arch:?} block_tokens {block_tokens} batch {seqs} step {step}"
                    );
                }
                let c = exec.graph_counters();
                assert!(
                    c.replayed > 0 && c.captured > 0 && c.capture_failed == 0,
                    "{arch:?} block_tokens {block_tokens} batch {seqs}: {c:?}"
                );
                println!(
                    "hip_decode_graph_matches_eager: {arch:?} block_tokens {block_tokens} \
                     batch {seqs}: {c:?}"
                );
            }
        }
    }
}

// ------------------------------------------------------------------------ trace diagnostics

/// Prefill plus `decode` greedy steps on every executor in `execs` (tokens chosen by the first),
/// tracing each forward. Returns per executor the trace of every step, and each step's first
/// position.
fn traced_run(
    execs: &mut [&mut Single],
    prompt: &[u32],
    decode: usize,
) -> (Vec<Vec<Vec<TraceTensor>>>, Vec<usize>) {
    for e in execs.iter_mut() {
        e.set_trace(true);
    }
    let mut traces = vec![Vec::new(); execs.len()];
    let mut starts = Vec::new();
    let mut batch = prompt.to_vec();
    let mut p0 = 0usize;
    for _ in 0..=decode {
        let mut next = None;
        for (i, e) in execs.iter_mut().enumerate() {
            let logits = forward(e, &batch, p0 as u32);
            next.get_or_insert(argmax(logits.row(0)));
            traces[i].push(e.take_trace());
        }
        starts.push(p0);
        p0 += batch.len();
        batch = vec![next.expect("at least one executor")];
    }
    (traces, starts)
}

/// The trace records every op, and a host recomputation of each op from the trace's own inputs
/// reproduces the cpu-reference provider bit for bit (the checker models its numerics exactly).
#[test]
fn cpu_trace_recomputes_exactly() {
    let tmp = TempDir::new("tiny-model-trace");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let mut exec = cpu_executor(&spec);
    assert!(exec.take_trace().is_empty(), "tracing is off by default");

    let (traces, starts) = traced_run(&mut [&mut exec], &prompt(spec.vocab), 3);
    let layers = spec.config.num_layers as usize;
    for (step, trace) in traces[0].iter().enumerate() {
        // embed + 15 per layer + final_norm + logits.
        assert_eq!(trace.len(), 1 + 15 * layers + 2, "step {step}");
        let t = if step == 0 { PROMPT_LEN } else { 1 };
        assert_eq!(trace[0].shape, [t, spec.config.hidden as usize]);
    }
    let weights = |name: &str| read_bf16_weight(&index, name);
    let mut checker = LocalChecker::new(&spec.config, &weights);
    let mut attn_p_f32_differs = false;
    for (step, trace) in traces[0].iter().enumerate() {
        for row in checker.check_step(step, starts[step], trace) {
            if row.name == "attn_p_f32" {
                attn_p_f32_differs |= row.stats.differing > 0;
                continue;
            }
            assert_eq!(row.stats.differing, 0, "{row:?}");
        }
    }
    assert!(
        attn_p_f32_differs,
        "f32 probabilities must be a different numerics model"
    );

    // The recorded logits are the returned ones; disabling drops the trace.
    let logits = forward(&mut exec, &prompt(spec.vocab), 0);
    let trace = exec.take_trace();
    let last = trace.last().expect("logits recorded");
    assert_eq!((last.name, last.data.as_slice()), ("logits", logits.row(0)));
    exec.set_trace(false);
    forward(&mut exec, &prompt(spec.vocab), 0);
    assert!(exec.take_trace().is_empty());
}

/// Tracing works for every family (Phase 2m S-3): the tiny OLMoE on the CPU provider with
/// tracing on records, for one 5-token prefill, `embed`, then per layer the attention outputs
/// with the full Q/K norms and the MoE block's router logits and output, then `final_norm` and
/// `logits`, each shaped as the op wrote it; the traced logits are bitwise those of the same
/// prefill untraced. Breaks if a family's executor loses tracing (OLMoE had none before the
/// shared decoder skeleton), a hook skips a trace point or recording changes an op.
#[test]
fn olmoe_trace_records_every_layer() {
    let tmp = TempDir::new("tiny-model-olmoe-trace");
    let spec = write_tiny_olmoe(tmp.path(), SEED);
    let cfg = &spec.config;
    let moe = cfg.moe.expect("OLMoE has experts");
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let mut exec = cpu_decoder(&spec, &mem);
    let mut kv = SequenceKv::new(&mem, *exec.kv_layout(), MAX_SEQ_LEN).expect("kv");
    let tokens = prompt(spec.vocab)[..5].to_vec();
    let positions: Vec<u32> = (0..5).collect();
    let untraced = kv
        .forward(&mut exec, &tokens, &positions)
        .expect("untraced prefill")
        .row(0)
        .to_vec();
    assert!(exec.take_trace().is_empty(), "tracing is off by default");

    exec.set_trace(true);
    // The same prefill again: it rewrites the same K/V slots.
    let traced = kv
        .forward(&mut exec, &tokens, &positions)
        .expect("traced prefill")
        .row(0)
        .to_vec();
    let trace = exec.take_trace();
    let per_layer = [
        "attn_norm",
        "q",
        "k",
        "v",
        "q_norm",
        "k_norm",
        "q_rope",
        "k_rope",
        "attn",
        "o_proj",
        "resid_attn",
        "mlp_norm",
        "router_logits",
        "moe_out",
        "resid_mlp",
    ];
    let layers = cfg.num_layers as usize;
    let mut want: Vec<(Option<usize>, &str)> = vec![(None, "embed")];
    for layer in 0..layers {
        want.extend(per_layer.iter().map(|&name| (Some(layer), name)));
    }
    want.extend([(None, "final_norm"), (None, "logits")]);
    let got: Vec<(Option<usize>, &str)> = trace.iter().map(|t| (t.layer, t.name)).collect();
    assert_eq!(got, want);

    let hidden = cfg.hidden as usize;
    let q_dim = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as usize;
    for t in &trace {
        let cols = match t.name {
            "q" | "q_norm" | "q_rope" | "attn" => q_dim,
            "k" | "v" | "k_norm" | "k_rope" => kv_dim,
            "router_logits" => moe.num_experts as usize,
            "logits" => spec.vocab as usize,
            _ => hidden,
        };
        let rows = if matches!(t.name, "final_norm" | "logits") {
            1
        } else {
            5
        };
        assert_eq!(t.shape, [rows, cols], "{:?} {}", t.layer, t.name);
        assert_eq!(t.data.len(), rows * cols, "{:?} {}", t.layer, t.name);
    }
    // Q/K norm rewrote the projections in place: the recorded rows differ.
    for (before, after) in [("q", "q_norm"), ("k", "k_norm")] {
        let get = |name: &str| {
            trace
                .iter()
                .find(|t| t.layer == Some(0) && t.name == name)
                .map(|t| t.data.clone())
                .expect("recorded")
        };
        assert_ne!(get(before), get(after), "{after} changed nothing");
    }
    let last = trace.last().expect("logits recorded");
    assert_eq!(last.data, traced);
    let bitwise = traced
        .iter()
        .zip(&untraced)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        bitwise,
        "tracing changed the logits by up to {}",
        max_abs_diff(&traced, &untraced)
    );

    exec.set_trace(false);
    kv.forward(&mut exec, &tokens, &positions).expect("prefill");
    assert!(exec.take_trace().is_empty());
}

/// Lab diagnostic (precision investigation): HIP vs cpu-reference on the head_dim-128 tiny
/// checkpoint op by op for the prefill and 3 decode steps (accumulated divergence), then every
/// HIP op recomputed on the host from the HIP trace's own inputs (the error each op adds).
/// Prints both tables; asserts only that the traces line up.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_trace_vs_cpu() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let ctx = turbine_kernels::test_support::open_context("hip");

    let tmp = TempDir::new("tiny-model-hip-trace");
    let spec = write_gpu_tiny(tmp.path());
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let mut cpu = cpu_executor(&spec);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = executor(&spec, shim_provider(ctx), mem);
    let (traces, starts) = traced_run(&mut [&mut cpu, &mut hip], &prompt(spec.vocab), 3);

    let mut accumulated = Vec::new();
    for (step, (c, h)) in traces[0].iter().zip(&traces[1]).enumerate() {
        accumulated.extend(compare_traces(step, c, h));
    }
    println!(
        "{}",
        render(
            "tiny hd128: HIP vs cpu-reference (accumulated)",
            &accumulated
        )
    );
    let weights = |name: &str| read_bf16_weight(&index, name);
    let mut checker = LocalChecker::new(&spec.config, &weights);
    let mut local = Vec::new();
    for (step, trace) in traces[1].iter().enumerate() {
        local.extend(checker.check_step(step, starts[step], trace));
    }
    println!(
        "{}",
        render(
            "tiny hd128: HIP op vs host recompute from its own inputs (local)",
            &local
        )
    );
}

/// The reduction `logits_reduce` defines for `row`: top-n by value (ties to the lower id),
/// log-sum-exp and the id-order inverse-CDF draw, sums in f64 in id order.
fn expected_reduction(row: &[f32], r: &RowReduce) -> ReducedRow {
    let mut pairs: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs.truncate(usize::from(r.top_n));
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&v| f64::from(v - max).exp()).sum();
    let sampled = r.uniform.map(|u| {
        let inv_t = 1.0 / r.temperature;
        let smax = row
            .iter()
            .map(|&v| v * inv_t)
            .fold(f32::NEG_INFINITY, f32::max);
        // Id order, or descending (ties by id) for a top-p nucleus.
        let mut order: Vec<usize> = (0..row.len()).collect();
        if r.top_p < 1.0 {
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        }
        let mut w: Vec<f64> = order
            .iter()
            .map(|&id| f64::from(row[id] * inv_t - smax).exp())
            .collect();
        if r.top_p < 1.0 {
            let target = f64::from(r.top_p) * w.iter().sum::<f64>();
            let mut cum = 0.0;
            let keep = w
                .iter()
                .position(|x| {
                    cum += x;
                    cum >= target
                })
                .expect("a nucleus below the total");
            w.truncate(keep + 1);
        }
        let target = f64::from(u) * w.iter().sum::<f64>();
        let mut cum = 0.0;
        let i = w
            .iter()
            .position(|x| {
                cum += x;
                target < cum
            })
            .expect("a draw below the total");
        (order[i] as u32, row[order[i]])
    });
    ReducedRow {
        lse: max + sum.ln() as f32,
        top: pairs,
        sampled,
    }
}

/// P2c S-4: on both tiny checkpoints, a batch mixing reduced and full rows (prefill chunks,
/// then decodes, with the reduced sequences not contiguous) returns the full rows bitwise equal
/// to the executor that copies every row, and each reduced row as the reduction of that row;
/// an executor whose registry lacks `logits_reduce` ignores `reduce` and copies every row.
#[test]
fn device_reduced_rows_match_full_rows() {
    let tmp = TempDir::new("tiny-model-reduced-rows");
    for spec in both_checkpoints(&tmp) {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let plain = cpu_model_with(
            &spec,
            &mem,
            16,
            ExecutorOptions::default(),
            cpu_reference_provider(),
            false,
        );
        let reducing = cpu_model_with(
            &spec,
            &mem,
            16,
            ExecutorOptions::default(),
            cpu_reference_provider(),
            true,
        );
        check_reduced_rows(&spec, &mem, plain, reducing);
    }
}

/// Lab only (P2c S-4): [`device_reduced_rows_match_full_rows`] on the HIP provider (the
/// head_dim-128 tiny Llama): the executor reducing on the device through the HIP
/// `logits_reduce` returns the full rows of the executor that copies every row, and each
/// reduced row as the reduction of that row.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_reduced_rows_match_full_rows() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let ctx = turbine_kernels::test_support::open_context("hip");
    assert!(
        ctx.library().abi_minor() >= 1,
        "logits_reduce needs kernel ABI v2.1"
    );
    let tmp = TempDir::new("tiny-model-hip-reduced-rows");
    let spec = write_gpu_tiny(tmp.path());
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let opts = ExecutorOptions::default();
    let plain = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx.clone()), false);
    let reducing = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx), true);
    check_reduced_rows(&spec, &mem, plain, reducing);
    println!("hip_reduced_rows_match_full_rows: ok");
}

/// Three steps of three sequences (prefill chunks, then decodes; reduced and full rows
/// interleaved) on `plain` (no `logits_reduce`) and `reducing`, each on its own pool.
fn check_reduced_rows(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    mut plain: Box<dyn ModelExecutor>,
    mut reducing: Box<dyn ModelExecutor>,
) {
    const LENS: [u32; 3] = [5, 3, 4];
    const POOL_BLOCKS: u32 = 8;
    let arch = spec.config.family;
    let mem = Arc::clone(mem);
    {
        assert!(
            !plain.reduces_logits() && reducing.reduces_logits(),
            "{arch:?}"
        );
        let layout = *plain.kv_layout();
        let (a, b) = (
            pool(&mem, &layout, POOL_BLOCKS),
            pool(&mem, &layout, POOL_BLOCKS),
        );
        let (kv_a, kv_b) = (
            pool_view(&a, &layout, POOL_BLOCKS),
            pool_view(&b, &layout, POOL_BLOCKS),
        );
        let tables: Vec<Vec<BlockId>> = (0..3).map(|s| vec![BlockId(s)]).collect();
        let greedy = |top_n| RowReduce {
            top_n,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        };
        let draw = RowReduce {
            top_n: 3,
            temperature: 0.8,
            uniform: Some(0.37),
            top_p: 1.0,
        };
        let nucleus = RowReduce {
            top_n: 2,
            temperature: 0.6,
            uniform: Some(0.81),
            top_p: 0.9,
        };
        let steps: [(&[u32], [Option<RowReduce>; 3]); 3] = [
            (&LENS, [Some(greedy(5)), None, Some(draw)]),
            (&[1, 1, 1], [None, Some(greedy(1)), Some(draw)]),
            (&[1, 1, 1], [Some(greedy(20)), Some(nucleus), None]),
        ];
        let mut kv_lens = [0u32; 3];
        for (step, (q_lens, reduce)) in steps.iter().enumerate() {
            let mut tokens = Vec::new();
            let mut positions = Vec::new();
            for s in 0..3 {
                let start = kv_lens[s];
                kv_lens[s] += q_lens[s];
                for p in start..kv_lens[s] {
                    tokens.push((p * 31 + 7 * s as u32 + 1) % spec.vocab);
                    positions.push(p);
                }
            }
            let run = |exec: &mut dyn ModelExecutor,
                       kv: &KvPoolView<'_>,
                       reduce: [Option<RowReduce>; 3]| {
                let mut q_start = 0;
                let seqs: Vec<SeqSlice<'_>> = (0..3)
                    .map(|s| {
                        let slice = SeqSlice {
                            seq: SeqId(s as u64 + 1),
                            q_start,
                            q_len: q_lens[s],
                            kv_len: kv_lens[s],
                            block_table: &tables[s],
                            reduce: reduce[s],
                        };
                        q_start += q_lens[s];
                        slice
                    })
                    .collect();
                exec.forward(&BatchInput {
                    tokens: &tokens,
                    positions: &positions,
                    seqs: &seqs,
                    kv,
                })
                .expect("forward")
            };
            // The plain executor ignores `reduce`.
            let full = run(plain.as_mut(), &kv_a, *reduce);
            assert_eq!((full.rows, full.slots()), (3, 3), "{arch:?}");
            let mixed = run(reducing.as_mut(), &kv_b, *reduce);
            assert_eq!(mixed.slots(), 3);
            for (s, want_reduce) in reduce.iter().enumerate() {
                match (mixed.slot(s), *want_reduce) {
                    (LogitsSlot::Full(row), None) => {
                        assert_eq!(row, full.row(s), "{arch:?} step {step} seq {s}");
                    }
                    (LogitsSlot::Reduced(got), Some(r)) => {
                        let want = expected_reduction(full.row(s), &r);
                        assert_eq!(got.top, want.top, "{arch:?} step {step} seq {s}");
                        assert_eq!(got.sampled, want.sampled, "{arch:?} step {step} seq {s}");
                        assert!(
                            (got.lse - want.lse).abs() <= 1e-6 * want.lse.abs().max(1.0),
                            "{arch:?} step {step} seq {s}: lse {} vs {}",
                            got.lse,
                            want.lse
                        );
                    }
                    (slot, r) => panic!("{arch:?} step {step} seq {s}: {slot:?} for {r:?}"),
                }
            }
        }
    }
}

// --------------------------------------------------------- launch ahead with token feeds (P2c)

/// P2c overlap scheduling: on both tiny checkpoints, steps launched ahead — each decode taking
/// its tokens on the device from the previous step's choices through [`TokenFeed`]s, launched
/// before the previous step is collected — return logits bitwise equal to the same steps run
/// one at a time with the tokens on the host (greedy rows with 1 and 3 candidates, drawn rows,
/// sequences reordered between steps). A third uncollected launch, a feed from a `top_k` row
/// (whose token the host draws) and a feed outside the batch are refused.
#[test]
fn launch_ahead_feeds_match_serial() {
    let tmp = TempDir::new("tiny-model-launch-ahead");
    for spec in both_checkpoints(&tmp) {
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let opts = ExecutorOptions::default();
        let serial = cpu_model_with(&spec, &mem, 16, opts, cpu_reference_provider(), true);
        let ahead = cpu_model_with(&spec, &mem, 16, opts, cpu_reference_provider(), true);
        check_launch_ahead(&spec, &mem, serial, ahead, false);
        // Greedy rows only, in one order: every decode is fed as one run.
        let serial = cpu_model_with(&spec, &mem, 16, opts, cpu_reference_provider(), true);
        let ahead = cpu_model_with(&spec, &mem, 16, opts, cpu_reference_provider(), true);
        check_launch_ahead(&spec, &mem, serial, ahead, true);
    }
}

/// Lab only (P2c overlap scheduling): [`launch_ahead_feeds_match_serial`] on the HIP provider
/// (the head_dim-128 tiny Llama): steps launched ahead through host staging (kernel ABI v2.3),
/// fed on the device, match the steps run one at a time bitwise.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_launch_ahead_feeds_match_serial() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let ctx = turbine_kernels::test_support::open_context("hip");
    assert!(
        ctx.library().abi_minor() >= 3,
        "host staging needs kernel ABI v2.3"
    );
    let tmp = TempDir::new("tiny-model-hip-launch-ahead");
    let spec = write_gpu_tiny(tmp.path());
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let opts = ExecutorOptions::default();
    let serial = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx.clone()), true);
    let ahead = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx.clone()), true);
    check_launch_ahead(&spec, &mem, serial, ahead, false);
    // Greedy rows fed as one run, with decode graphs on: the fed steps are captured (the feed's
    // word in the key) and replayed.
    let mut serial = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx.clone()), true);
    let mut ahead = cpu_model_with(&spec, &mem, 16, opts, shim_provider(ctx.clone()), true);
    serial.set_decode_graphs(Some(hip_graphs(&ctx, MAX_SEQS)));
    ahead.set_decode_graphs(Some(hip_graphs(&ctx, MAX_SEQS)));
    let counters = check_launch_ahead(&spec, &mem, serial, ahead, true);
    println!("hip_launch_ahead_feeds_match_serial: graphs {counters:?}");
    assert!(
        counters.replayed > 0,
        "fed greedy steps replay a graph: {counters:?}"
    );
    println!("hip_launch_ahead_feeds_match_serial: ok");
}

/// The token the device chose for a reduced row asking for `r`: the draw, or the best
/// candidate of a greedy row (the tiny models' logits are finite).
fn device_choice(row: &ReducedRow, r: &RowReduce) -> u32 {
    match r.uniform {
        Some(_) => row.sampled.expect("a drawn row has its draw").0,
        None => row.top[0].0,
    }
}

/// Three sequences prefilled (5, 3 and 4 tokens), then decoded for six steps, the last three
/// in rotated sequence orders: `serial` runs each step with the host's tokens; `ahead` launches
/// each decode with feeds before collecting the previous step. Each on its own pool.
fn check_launch_ahead(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    mut serial: Box<dyn ModelExecutor>,
    mut ahead: Box<dyn ModelExecutor>,
    greedy: bool,
) -> graphs::GraphCounters {
    const LENS: [u32; 3] = [5, 3, 4];
    const POOL_BLOCKS: u32 = 8;
    let decodes: usize = if greedy { 10 } else { 6 };
    let arch = spec.config.family;
    assert!(ahead.overlaps() && ahead.reduces_logits(), "{arch:?}");
    let layout = *serial.kv_layout();
    let (a, b) = (
        pool(mem, &layout, POOL_BLOCKS),
        pool(mem, &layout, POOL_BLOCKS),
    );
    let (kv_a, kv_b) = (
        pool_view(&a, &layout, POOL_BLOCKS),
        pool_view(&b, &layout, POOL_BLOCKS),
    );
    let tables: Vec<Vec<BlockId>> = (0..3).map(|s| vec![BlockId(2 * s + 1)]).collect();
    // Sequence 0 greedy with one candidate, 1 drawn (a fresh uniform each step), 2 greedy with
    // three candidates (its choice words are not adjacent to the others').
    let reduce = |s: usize, step: usize| match s {
        _ if greedy => RowReduce {
            top_n: 1,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        },
        0 => RowReduce {
            top_n: 1,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        },
        1 => RowReduce {
            top_n: 1,
            temperature: 0.8,
            uniform: Some(((step as f32) * 0.37 + 0.11).fract()),
            top_p: if step.is_multiple_of(2) { 1.0 } else { 0.9 },
        },
        _ => RowReduce {
            top_n: 3,
            temperature: 0.0,
            uniform: None,
            top_p: 1.0,
        },
    };
    // Sequence order of each step: prefill and the first decodes in order, then rotated.
    let order = |step: usize| -> [usize; 3] {
        match step {
            _ if greedy => [0, 1, 2],
            0..=3 => [0, 1, 2],
            4 => [2, 0, 1],
            5 => [1, 2, 0],
            _ => [0, 1, 2],
        }
    };
    let prompts: Vec<Vec<u32>> = (0..3)
        .map(|s| {
            (0..LENS[s])
                .map(|p| (p * 31 + 7 * s as u32 + 1) % spec.vocab)
                .collect()
        })
        .collect();
    let mut kv_lens = LENS;
    let mut last: [u32; 3] = [0; 3];
    // Step `step` of the run: its tokens (placeholders where `fed`), positions and slices.
    let batch = |step: usize, kv_lens: [u32; 3], last: [u32; 3], fed: bool| {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut q_lens = [0u32; 3];
        for &s in &order(step) {
            if step == 0 {
                tokens.extend_from_slice(&prompts[s]);
                positions.extend(0..LENS[s]);
                q_lens[s] = LENS[s];
            } else {
                tokens.push(if fed { 0 } else { last[s] });
                positions.push(kv_lens[s] - 1);
                q_lens[s] = 1;
            }
        }
        (tokens, positions, q_lens)
    };
    let slices = |step: usize, kv_lens: [u32; 3], q_lens: [u32; 3]| -> Vec<SeqSlice<'_>> {
        let mut q_start = 0;
        order(step)
            .iter()
            .map(|&s| {
                let slice = SeqSlice {
                    seq: SeqId(s as u64 + 1),
                    q_start,
                    q_len: q_lens[s],
                    kv_len: kv_lens[s],
                    block_table: &tables[s],
                    reduce: Some(reduce(s, step)),
                };
                q_start += q_lens[s];
                slice
            })
            .collect()
    };

    // Serial: host tokens, one step at a time.
    let mut want: Vec<Logits> = Vec::new();
    for step in 0..=decodes {
        if step > 0 {
            for len in &mut kv_lens {
                *len += 1;
            }
        }
        let (tokens, positions, q_lens) = batch(step, kv_lens, last, false);
        let seqs = slices(step, kv_lens, q_lens);
        let logits = serial
            .forward(&BatchInput {
                tokens: &tokens,
                positions: &positions,
                seqs: &seqs,
                kv: &kv_a,
            })
            .expect("serial forward");
        for (slot, &s) in order(step).iter().enumerate() {
            let LogitsSlot::Reduced(row) = logits.slot(slot) else {
                panic!("{arch:?} step {step}: sequence {s} not reduced");
            };
            last[s] = device_choice(row, &reduce(s, step));
        }
        want.push(logits);
    }

    // Ahead: every decode launched with feeds before the previous step is collected.
    let mut kv_lens = LENS;
    let mut got: Vec<Logits> = Vec::new();
    for step in 0..=decodes {
        let mut feeds = Vec::new();
        if step > 0 {
            for len in &mut kv_lens {
                *len += 1;
            }
            let prev = order(step - 1);
            for (token, s) in order(step).iter().enumerate() {
                let prev_slot = prev.iter().position(|p| p == s).expect("in both steps");
                feeds.push(TokenFeed {
                    token: token as u32,
                    prev_slot: prev_slot as u32,
                });
            }
        }
        let (tokens, positions, q_lens) = batch(step, kv_lens, [0; 3], true);
        let seqs = slices(step, kv_lens, q_lens);
        let input = BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv_b,
        };
        ahead.launch(&input, &feeds).expect("launch ahead");
        if step == 2 {
            let third = ahead.launch(&input, &feeds).expect_err("a third launch");
            assert!(
                third.to_string().contains("not collected"),
                "{arch:?}: {third}"
            );
        }
        if step > 0 {
            got.push(ahead.collect().expect("collect"));
        }
    }
    got.push(ahead.collect().expect("collect the last step"));
    assert_eq!(got.len(), want.len());
    for (step, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{arch:?} step {step}: launched ahead vs serial");
    }
    assert!(
        ahead.collect().is_err(),
        "{arch:?}: nothing left to collect"
    );

    // A feed from a `top_k` row (host draw) or outside the batch is refused.
    let top_k = RowReduce {
        top_n: 4,
        temperature: 0.8,
        uniform: None,
        top_p: 1.0,
    };
    for len in &mut kv_lens {
        *len += 1;
    }
    let seqs: Vec<SeqSlice<'_>> = (0..3)
        .map(|s| SeqSlice {
            seq: SeqId(s as u64 + 1),
            q_start: s as u32,
            q_len: 1,
            kv_len: kv_lens[s],
            block_table: &tables[s],
            reduce: Some(if s == 1 { top_k } else { reduce(s, 0) }),
        })
        .collect();
    let positions: Vec<u32> = (0..3).map(|s| kv_lens[s] - 1).collect();
    let input = BatchInput {
        tokens: &[1, 2, 3],
        positions: &positions,
        seqs: &seqs,
        kv: &kv_b,
    };
    ahead.launch(&input, &[]).expect("launch with a top_k row");
    for len in &mut kv_lens {
        *len += 1;
    }
    let seqs: Vec<SeqSlice<'_>> = seqs
        .iter()
        .map(|s| SeqSlice {
            kv_len: s.kv_len + 1,
            ..*s
        })
        .collect();
    let positions: Vec<u32> = positions.iter().map(|p| p + 1).collect();
    let input = BatchInput {
        tokens: &[1, 2, 3],
        positions: &positions,
        seqs: &seqs,
        kv: &kv_b,
    };
    for (feed, what) in [
        (
            TokenFeed {
                token: 1,
                prev_slot: 1,
            },
            "no device-chosen token",
        ),
        (
            TokenFeed {
                token: 3,
                prev_slot: 0,
            },
            "3-token batch",
        ),
    ] {
        let err = ahead.launch(&input, &[feed]).expect_err(what).to_string();
        assert!(err.contains(what), "{arch:?}: {err}");
    }
    ahead.collect().expect("collect the top_k step");
    ahead.graph_counters()
}

// ------------------------------------------------------------------------ op profile (P2c S-2)

/// A tiny checkpoint's decoder executor on the CPU provider with the default options (batches
/// of up to [`MAX_SEQ_LEN`] tokens and [`MAX_SEQS`] sequences), and the registry it runs on.
fn cpu_profiled(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
) -> (DecoderExecutor, Arc<KernelRegistry>) {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load");
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let opts = ExecutorOptions::default();
    let reqs =
        executor::available_requirements(cfg, BLOCK_TOKENS, opts, std::slice::from_ref(&provider));
    let card = provider.card_profile();
    let registry = Arc::new(
        KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider"),
    );
    let exec = DecoderExecutor::new(
        cfg,
        decoder_spec(cfg),
        weights,
        Arc::clone(&registry),
        Arc::clone(mem),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: MAX_SEQ_LEN,
            max_seqs: MAX_SEQS,
        },
        opts,
    )
    .expect("decoder executor");
    (exec, registry)
}

/// [`cpu_profiled`]'s executor alone.
fn cpu_decoder(spec: &TinySpec, mem: &Arc<dyn DeviceMemory>) -> DecoderExecutor {
    cpu_profiled(spec, mem).0
}

/// The decoder spec of a tiny checkpoint's family.
fn decoder_spec(cfg: &ModelArchConfig) -> DecoderSpec {
    match cfg.family.0.name() {
        "llama" => families::llama::decoder_spec(),
        "olmoe" => families::olmoe::decoder_spec(),
        other => panic!("no tiny decoder spec for family {other}"),
    }
}

/// For both tiny checkpoints on the CPU provider, a decode forward of 3 sequences with profile
/// mode on yields one entry per op kind the forward runs (each at its selected implementation)
/// with the per-layer call pattern × layers, plus the batch upload and the logits read (and
/// OLMoE's expert-offset read and accumulator reset once per layer); the profile's total is
/// within the forward's wall time, the logits are bitwise those of the same forward with
/// profile mode off, and a profile-off forward records nothing.
#[test]
fn op_profile_accounts_forward() {
    const PROMPT: u32 = 20;
    let tmp = TempDir::new("tiny-model-op-profile");
    for spec in both_checkpoints(&tmp) {
        let arch = spec.config.family;
        let layers = spec.config.num_layers;
        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let (mut exec, registry) = cpu_profiled(&spec, &mem);
        let layout = *exec.kv_layout();
        let per_seq = (PROMPT + 1).div_ceil(BLOCK_TOKENS);
        let storage = pool(&mem, &layout, 3 * per_seq);
        let kv = pool_view(&storage, &layout, 3 * per_seq);
        let tables: Vec<Vec<BlockId>> = (0..3)
            .map(|s| (0..per_seq).map(|b| BlockId(s * per_seq + b)).collect())
            .collect();
        for (s, table) in tables.iter().enumerate() {
            let tokens: Vec<u32> = (0..PROMPT)
                .map(|i| (i * 13 + 5 * s as u32 + 1) % spec.vocab)
                .collect();
            run_seq(&mut exec, &kv, table, &tokens, 0);
        }
        // The same decode step twice (it rewrites the same K/V slot): profile off, then on.
        let tokens = [3u32, 7, 11];
        let positions = [PROMPT; 3];
        let seqs: Vec<SeqSlice<'_>> = tables
            .iter()
            .enumerate()
            .map(|(s, table)| SeqSlice {
                seq: SeqId(s as u64 + 1),
                q_start: s as u32,
                q_len: 1,
                kv_len: PROMPT + 1,
                block_table: table,
                reduce: None,
            })
            .collect();
        let batch = BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv,
        };
        let off = exec.forward(&batch).expect("profile-off decode");
        assert!(
            exec.take_profile().entries.is_empty(),
            "{arch:?}: profile-off forward recorded ops"
        );
        exec.set_profile(true);
        let start = std::time::Instant::now();
        let on = exec.forward(&batch).expect("profiled decode");
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let profile = exec.take_profile();
        assert_eq!(on, off, "{arch:?}: profiling changed the logits");

        // Each op kind runs one implementation on the CPU provider: the one selected for it.
        let cpu = |op: &str| -> &str {
            let impls: Vec<&str> = registry
                .selections()
                .iter()
                .filter(|s| s.op.as_str() == op)
                .map(|s| s.implementation.as_str())
                .collect();
            assert!(!impls.is_empty(), "{op} was never selected");
            assert!(impls.iter().all(|i| *i == impls[0]), "{op}: {impls:?}");
            impls[0]
        };
        let mut want: Vec<(&str, &str, u32)> = vec![
            ("batch_upload", "host", 1),
            ("embedding", cpu("embedding"), 1),
            ("rope", cpu("rope"), layers),
            (
                "attention_decode_paged",
                cpu("attention_decode_paged"),
                layers,
            ),
            // Fused ops: every residual add a norm follows is one add_rmsnorm (the post-attention
            // norm, the next layer's input norm); the last layer's last residual is a plain add.
            ("add_rmsnorm", cpu("add_rmsnorm"), 2 * layers - 1),
            ("add", cpu("add"), 1),
            ("logits_read", "d2h", 1),
        ];
        // The default options: Q/K/V (and Llama's gate/up) one GEMM each when the projections are
        // fused, one per projection otherwise.
        let fused = ExecutorOptions::default().fused_projections;
        let (qkv, gate_up) = if fused { (1, 1) } else { (3, 2) };
        match arch.0.name() {
            "llama" => want.extend([
                // The first input norm and the final norm of the (consecutive) last rows.
                ("rmsnorm", cpu("rmsnorm"), 2),
                // Q/K/V, O, gate/up and down per layer plus the LM head.
                ("gemm", cpu("gemm"), (qkv + gate_up + 2) * layers + 1),
                ("silu_mul", cpu("silu_mul"), layers),
            ]),
            _ => want.extend([
                // The first input norm, Q and K norms per layer and the final norm.
                ("rmsnorm", cpu("rmsnorm"), 2 * layers + 2),
                // Q/K/V, O and the router per layer plus the LM head.
                ("gemm", cpu("gemm"), (qkv + 2) * layers + 1),
                ("moe_route", cpu("moe_route"), layers),
                // The CPU provider reads the group sizes on the device: no offsets read.
                ("moe_zero", "add", layers),
                ("moe_experts", cpu("moe_experts"), layers),
            ]),
        }
        let mut got: Vec<(&str, &str, u32)> = profile
            .entries
            .iter()
            .map(|e| (e.op.as_str(), e.r#impl.as_str(), e.calls))
            .collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "{arch:?}: profile entries");
        assert!(
            profile.entries.iter().all(|e| e.total_ms >= 0.0),
            "{arch:?}: negative time"
        );
        let total = profile.total_ms();
        assert!(
            total > 0.0 && total <= wall_ms,
            "{arch:?}: profiled {total} ms against a {wall_ms} ms forward"
        );

        // Profiling off again: nothing is recorded and nothing is left over.
        exec.set_profile(false);
        let again = exec.forward(&batch).expect("decode after profiling");
        assert_eq!(again, off, "{arch:?}");
        assert!(exec.take_profile().entries.is_empty(), "{arch:?}");
    }
}

/// One sequence through `exec` on a pool of its own: the first `prompt_len` tokens of `feed` as
/// one prefill, then one decode step per token. With `greedy` each step's token is the argmax of
/// the previous row (for `steps` steps; `feed` holds only the prompt), else the next token of
/// `feed`. Returns the tokens fed and every forward's logits row.
fn single_sequence(
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
    feed: &[u32],
    prompt_len: usize,
    steps: usize,
    greedy: bool,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let layout = *exec.kv_layout();
    let blocks = ((prompt_len + steps) as u32).div_ceil(layout.block_tokens);
    let storage = pool(mem, &layout, blocks);
    let kv = pool_view(&storage, &layout, blocks);
    let table: Vec<BlockId> = (0..blocks).map(BlockId).collect();
    let run = |exec: &mut dyn ModelExecutor, tokens: &[u32], start: u32| {
        let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
        let seqs = [SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: tokens.len() as u32,
            kv_len: start + tokens.len() as u32,
            block_table: &table,
            reduce: None,
        }];
        exec.forward(&BatchInput {
            tokens,
            positions: &positions,
            seqs: &seqs,
            kv: &kv,
        })
        .expect("forward")
        .row(0)
        .to_vec()
    };
    let mut tokens = feed[..prompt_len].to_vec();
    let mut rows = vec![run(exec, &tokens, 0)];
    for step in 0..steps {
        let next = if greedy {
            argmax(rows.last().expect("a row"))
        } else {
            feed[prompt_len + step]
        };
        let pos = tokens.len() as u32;
        tokens.push(next);
        rows.push(run(exec, &[next], pos));
    }
    (tokens, rows)
}

/// HIP vs CPU logit bound of the head_dim-128 tiny OLMoE: its expert GEMMs and weighted sum
/// round to BF16 in another order than the reference; 3.7e-3 measured on the R9700 (lab run
/// 0926202901-0eb0e349), where the tiny Llama stays within `HIP_MAX_ABS_LOGIT_DIFF`.
const HIP_MAX_ABS_LOGIT_DIFF_MOE: f32 = 1e-2;

/// Phase 2m S-5: a kernel library without the ABI v2.4 group still loads and serves.
/// `libturbine_hip_v23.so` (the same kernels, minor 3, built beside `TURBINE_KERNEL_LIBRARY`)
/// enumerates nothing, so every selection on it is the library's own (`provider_internal`); it
/// chooses the implementations the Rust registry binds on `libturbine_hip.so`, and on the tiny
/// Llama and OLMoE (head_dim 128) a greedy run on it, like one on the bound v2.4 library,
/// matches the CPU reference greedily (every step's argmax) and within a logit bound: the
/// `hip_matches_cpu` bound for Llama, [`HIP_MAX_ABS_LOGIT_DIFF_MOE`] for OLMoE (the two libraries
/// tune their GEMMs per context, so they are compared through the reference, not bitwise). Breaks if an older library no longer loads
/// or serves, or if the Rust selection departs from the library's own.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_v23_library_matches_cpu() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let opened = turbine_kernels::test_support::open_backend("hip");
    let device = opened
        .device
        .clone()
        .expect("the hip backend runs on a device");
    let main_ctx = opened.context.clone().expect("a kernel library context");
    assert!(main_ctx.library().enumerates_implementations());
    let v23_path = main_ctx
        .library()
        .path()
        .with_file_name("libturbine_hip_v23.so");
    let lib = ShimLibrary::load(&v23_path, "hip")
        .unwrap_or_else(|e| panic!("load {}: {e}", v23_path.display()));
    assert_eq!(lib.abi_minor(), 3);
    assert!(!lib.enumerates_implementations());
    let ctx = lib.create_context(&device).expect("v2.3 context");
    if let Some(card) = opened.card {
        ctx.set_profile(card)
            .expect("a v2.3 library ignores the card profile");
    }
    let old_mem: Arc<dyn DeviceMemory> = ctx.clone();
    let new_mem: Arc<dyn DeviceMemory> = main_ctx.clone();
    let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
    let tmp = TempDir::new("tiny-model-hip-v23");
    let specs = [
        write_gpu_tiny(&tmp.path().join("llama")),
        write_tiny_olmoe_with_head_dim(&tmp.path().join("olmoe"), SEED, GPU_HEAD_DIM),
    ];
    let opts = ExecutorOptions::default();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let selections = |provider: &Arc<dyn KernelProvider>, spec: &TinySpec| {
        let reqs = executor::available_requirements(
            &spec.config,
            BLOCK_TOKENS,
            opts,
            std::slice::from_ref(provider),
        );
        KernelRegistry::build(
            vec![Arc::clone(provider)],
            &[provider.id()],
            &reqs,
            &metrics,
            provider.card_profile(),
        )
        .expect("the library serves every op")
        .selections()
        .to_vec()
    };
    for (spec, bound_diff) in specs
        .iter()
        .zip([HIP_MAX_ABS_LOGIT_DIFF, HIP_MAX_ABS_LOGIT_DIFF_MOE])
    {
        let name = spec.dir.display().to_string();
        let old = shim_provider(ctx.clone());
        let new = shim_provider(main_ctx.clone());
        let (old_sel, new_sel) = (selections(&old, spec), selections(&new, spec));
        assert!(
            old_sel.iter().all(|s| s.reason_code == "provider_internal"),
            "{name}: {old_sel:?}"
        );
        assert!(
            new_sel.iter().all(|s| s.reason_code != "provider_internal"),
            "{name}: {new_sel:?}"
        );
        let chosen = |sel: &[turbine_kernels::Selection]| {
            sel.iter()
                .map(|s| (s.op, s.config.clone(), s.implementation.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(chosen(&old_sel), chosen(&new_sel), "{name}");

        let prompt = prompt(spec.vocab);
        let n = prompt.len();
        let mut cpu = cpu_model_with(spec, &host, 128, opts, cpu_reference_provider(), false);
        let (feed, want) = single_sequence(cpu.as_mut(), &host, &prompt, n, 16, true);
        let mut v23 = cpu_model_with(spec, &old_mem, 128, opts, old, false);
        let (_, got) = single_sequence(v23.as_mut(), &old_mem, &feed, n, 16, false);
        let mut v24 = cpu_model_with(spec, &new_mem, 128, opts, new, false);
        let (_, bound) = single_sequence(v24.as_mut(), &new_mem, &feed, n, 16, false);
        for (lib, rows) in [("v2.3", &got), ("v2.4", &bound)] {
            let mut worst = 0f32;
            for (step, (g, w)) in rows.iter().zip(&want).enumerate() {
                let diff = max_abs_diff(g, w);
                worst = worst.max(diff);
                assert!(
                    diff <= bound_diff,
                    "{name} {lib} step {step}: max abs diff {diff}"
                );
                assert_eq!(
                    argmax(g),
                    argmax(w),
                    "{name} {lib}: greedy token {step} differs"
                );
            }
            println!("hip_v23_library_matches_cpu {name} {lib}: max abs logit diff {worst}");
        }
    }
}

// ----------------------------------------------------------- tensor parallelism (P5 S-6)

/// Greedy decode steps after the prefill in the tensor-parallel comparisons.
const TP_STEPS: usize = 16;

/// One run's greedy tokens per step and every step's logits rows.
type GreedyRun = (Vec<[u32; 2]>, Vec<Vec<f32>>);

/// Two ragged sequences (a 20- and a 13-token prompt) prefilled in one batch, then
/// [`TP_STEPS`] greedy decode steps of both in one batch: the greedy tokens per step and every
/// step's two logits rows. Sequence `s` keeps its K/V in pool block `s`.
fn greedy_pair(exec: &mut dyn ModelExecutor, kv: &KvPoolView<'_>, vocab: u32) -> GreedyRun {
    let prompts: [Vec<u32>; 2] = [
        prompt(vocab),
        (0..13u32).map(|i| (i * 29 + 3) % vocab).collect(),
    ];
    let tables = [[BlockId(0)], [BlockId(1)]];
    let mut lens = [prompts[0].len() as u32, prompts[1].len() as u32];
    let tokens: Vec<u32> = prompts.concat();
    let positions: Vec<u32> = prompts.iter().flat_map(|p| 0..p.len() as u32).collect();
    let seqs = [
        SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: lens[0],
            kv_len: lens[0],
            block_table: &tables[0],
            reduce: None,
        },
        SeqSlice {
            seq: SeqId(2),
            q_start: lens[0],
            q_len: lens[1],
            kv_len: lens[1],
            block_table: &tables[1],
            reduce: None,
        },
    ];
    let mut logits = exec
        .forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv,
        })
        .expect("prefill");
    let (mut chosen, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..TP_STEPS {
        assert_eq!(logits.rows, 2);
        rows.extend([logits.row(0).to_vec(), logits.row(1).to_vec()]);
        let next = [argmax(logits.row(0)), argmax(logits.row(1))];
        chosen.push(next);
        let positions = lens;
        lens = [lens[0] + 1, lens[1] + 1];
        let seqs: Vec<SeqSlice<'_>> = (0..2)
            .map(|s| SeqSlice {
                seq: SeqId(s as u64 + 1),
                q_start: s as u32,
                q_len: 1,
                kv_len: lens[s],
                block_table: &tables[s],
                reduce: None,
            })
            .collect();
        logits = exec
            .forward(&BatchInput {
                tokens: &next,
                positions: &positions,
                seqs: &seqs,
                kv,
            })
            .expect("decode");
    }
    rows.extend([logits.row(0).to_vec(), logits.row(1).to_vec()]);
    (chosen, rows)
}

/// Rank `s` of `spec`'s model on `provider` over `mem`, its collectives on `collective`.
fn tp_rank(
    spec: &TinySpec,
    s: tp::ShardSpec,
    collective: Arc<dyn Collective>,
    provider: Arc<dyn KernelProvider>,
    mem: &Arc<dyn DeviceMemory>,
) -> Box<dyn ModelExecutor> {
    let cfg = &spec.config;
    let opts = ExecutorOptions::default();
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = tp::weight_slots(cfg, s).expect("shard slots");
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load shard");
    let reqs =
        tp::available_requirements(cfg, s, BLOCK_TOKENS, opts, std::slice::from_ref(&provider))
            .expect("requirements");
    let order = [provider.id()];
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
        .expect("every op has a provider");
    tp::build_executor(
        cfg,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: 64,
            max_seqs: MAX_SEQS,
        },
        opts,
        TpContext {
            rank: s.rank,
            world: s.world,
            collective,
            stream: mem.compute_stream(),
        },
    )
    .expect("rank executor")
}

/// Runs [`greedy_pair`] on every rank of a `world`-rank group over the host collective, one
/// thread per rank, each on the memory and provider `env(rank)` gives, with its own pool of its
/// KV layout; returns every rank's run.
fn tp_group(
    spec: &TinySpec,
    world: u32,
    env: impl Fn(u32) -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) + Sync,
) -> Vec<GreedyRun> {
    let group = HostCollective::group(world as usize, std::time::Duration::from_secs(60));
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(rank, collective)| {
                let env = &env;
                scope.spawn(move || {
                    let s = tp::ShardSpec {
                        rank: rank as u32,
                        world,
                    };
                    let (mem, provider) = env(s.rank);
                    let mut exec = tp_rank(spec, s, Arc::new(collective), provider, &mem);
                    let layout = tp::kv_layout(&spec.config, s, BLOCK_TOKENS).expect("layout");
                    assert_eq!(*exec.kv_layout(), layout);
                    let storage = pool(&mem, &layout, 4);
                    greedy_pair(exec.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    })
}

/// How a tensor-parallel group's logits must match one device's.
#[derive(Clone, Copy)]
enum TpBound {
    /// A group of one: the tensor-parallel path with identity collectives, bitwise.
    Bitwise,
    /// The golden rule (user decision 2026-09-28, "P5: tensor-parallel accuracy bound"):
    /// greedy tokens identical and, for each of one device's top-k candidates of every row,
    /// |Δ logprob| within the strict golden bounds of `tests/golden/llama-3.2-3b-instruct/
    /// tolerance.json` — the bound the served model is held to at concurrency 1. Raw logits are
    /// not bounded: each all-reduce rounds BF16 partial sums (one device rounds once, after an
    /// F32 accumulation over the whole reduction), which moves them by ~1 % relative.
    Golden(GoldenLogprobBounds),
}

/// The strict (concurrency-1) logprob bounds of a golden `tolerance.json`.
#[derive(Clone, Copy, Debug)]
struct GoldenLogprobBounds {
    top_k: usize,
    likely: f64,
    tail: f64,
    likely_floor: f64,
}

impl GoldenLogprobBounds {
    /// The committed Llama bounds (0.15 above logprob −2, 0.55 below, top 5): the tightest of
    /// the golden references, used for both tiny families.
    fn llama() -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/llama-3.2-3b-instruct/tolerance.json");
        let text = std::fs::read_to_string(&path).expect("golden tolerance.json");
        let v: serde_json::Value = serde_json::from_str(&text).expect("tolerance.json parses");
        let num = |key: &str| {
            v[key]
                .as_f64()
                .unwrap_or_else(|| panic!("tolerance.json {key}"))
        };
        GoldenLogprobBounds {
            top_k: num("top_k") as usize,
            likely: num("max_abs_logprob_diff_likely"),
            tail: num("max_abs_logprob_diff_tail"),
            likely_floor: num("likely_logprob_floor"),
        }
    }
}

/// Log-softmax of one logits row, in f64.
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
    let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    let log_z = max + sum.ln();
    row.iter().map(|&x| x as f64 - log_z).collect()
}

/// Checks a group's runs against one device's: every rank ends each step with bitwise the same
/// logits rows, the greedy tokens are exactly one device's, and the logits obey `bound`.
/// Returns the largest |Δ logprob| over the compared candidates as a fraction of its golden
/// bound (0 for [`TpBound::Bitwise`]).
fn check_tp_against_one_device(
    what: &str,
    ranks: &[GreedyRun],
    want: &GreedyRun,
    bound: TpBound,
) -> f64 {
    for (rank, (tokens, rows)) in ranks.iter().enumerate() {
        assert!(
            rows == &ranks[0].1,
            "{what}: rank {rank} logits differ from rank 0"
        );
        assert_eq!(
            tokens, &want.0,
            "{what}: greedy tokens differ from one device"
        );
    }
    let bounds = match bound {
        TpBound::Bitwise => {
            assert!(
                ranks[0].1 == want.1,
                "{what}: logits differ from one device"
            );
            return 0.0;
        }
        TpBound::Golden(b) => b,
    };
    let mut worst = 0f64;
    for (row, (got, w)) in ranks[0].1.iter().zip(&want.1).enumerate() {
        let (got, w) = (log_softmax(got), log_softmax(w));
        let mut order: Vec<usize> = (0..w.len()).collect();
        order.sort_by(|&a, &b| w[b].total_cmp(&w[a]));
        for &cand in order.iter().take(bounds.top_k) {
            let diff = (got[cand] - w[cand]).abs();
            let limit = if w[cand] > bounds.likely_floor {
                bounds.likely
            } else {
                bounds.tail
            };
            worst = worst.max(diff / limit);
            assert!(
                diff <= limit,
                "{what} row {row} token {cand}: |Δ logprob| = {diff} > {limit} \
                 (one device {:.4})",
                w[cand]
            );
        }
    }
    worst
}

/// P5 S-6: the tiny Llama (4 heads, 2 KV heads, vocabulary 263 — not divisible by the group,
/// so the last vocabulary shard is padded) and the tiny OLMoE (4 = 4 KV heads, full-projection
/// Q/K norm, 8 experts top-2, untied LM head) on the cpu-reference provider, split over 2 and 4
/// ranks on the host collective (each rank its own host "device" and KV pool, fed the same
/// batches on its own thread; tp 4 replicates the Llama KV heads), give one device's greedy
/// tokens exactly for a ragged two-sequence prefill and 16 batched decode steps, with logits
/// bitwise equal on every rank and every row's top-5 logprobs within the strict golden bounds
/// ([`TpBound::Golden`]); a group of one is bitwise one device. Breaks if any sharding rule,
/// collective insertion point, vocabulary offset, the sharded norm or the logits reorder is
/// wrong.
#[test]
fn tp2_matches_tp1_on_host() {
    let tmp = TempDir::new("tiny-model-tp");
    let golden = TpBound::Golden(GoldenLogprobBounds::llama());
    for spec in both_checkpoints(&tmp) {
        let name = spec.config.family.0.name();
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let mut one = cpu_model(&spec, &host, 64);
        let layout = spec.config.kv_layout(BLOCK_TOKENS);
        let storage = pool(&host, &layout, 4);
        let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
        for world in [1u32, 2, 4] {
            let ranks = tp_group(&spec, world, |rank| {
                let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(rank), 1 << 30);
                (mem, cpu_reference_provider())
            });
            let what = format!("{name} tp {world}");
            let bound = if world == 1 { TpBound::Bitwise } else { golden };
            let worst = check_tp_against_one_device(&what, &ranks, &want, bound);
            println!(
                "tp2_matches_tp1_on_host {what}: worst top-k |Δ logprob| / golden bound {worst:.3}"
            );
        }
    }
}

/// Lab only: the tensor-parallel path on the HIP provider without RCCL — the head_dim-128 tiny
/// Llama split over two ranks, each its own context on the same GPU, through the host
/// collective (which reaches any device memory), against one HIP context: identical greedy
/// tokens and top-5 logprobs within the strict golden bounds ([`TpBound::Golden`]). Proves the
/// HIP shard loading, the
/// `vocab_offset` embedding, the per-rank GEMM shapes, the all-reduce insertion points and the
/// device-to-device logits reorder (kernel ABI v2.5 `turbine_memcpy_async`).
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_tp2_matches_tp1() {
    let tmp = TempDir::new("tiny-model-hip-tp");
    let spec = write_gpu_tiny(tmp.path());
    let ctx = hip_context();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let opts = ExecutorOptions::default();
    let mut one = cpu_model_with(&spec, &mem, 64, opts, shim_provider(ctx), false);
    let layout = spec.config.kv_layout(BLOCK_TOKENS);
    let storage = pool(&mem, &layout, 4);
    let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
    let ranks = tp_group(&spec, 2, |_| {
        let ctx = hip_context();
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        (mem, shim_provider(ctx))
    });
    let golden = TpBound::Golden(GoldenLogprobBounds::llama());
    let worst = check_tp_against_one_device("hip tp 2", &ranks, &want, golden);
    println!("hip_tp2_matches_tp1: worst top-k |Δ logprob| / golden bound {worst:.3}");
}

/// Lab only, two GPUs (P5 Task 32, tensor-parallel decode graphs): the head_dim-128 tiny Llama
/// at tp 2 over the `hostmem` collective on both R9700s, [`greedy_pair`] eagerly and then again
/// with decode graphs on every rank (the second decode step captures its collectives, the rest
/// replay): the greedy tokens and every logits row of the graph run are bitwise the eager run's
/// on both ranks, and the graphs were replayed. Breaks if a replay reuses a captured sequence
/// number (stale slots or an early wait), the logits reorder copies are not captured, or the
/// ranks capture different keys.
#[test]
#[ignore = "needs two HIP devices and libturbine_hip.so (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_tp2_graphs_match_eager() {
    use turbine_distributed::collective::{self as coll, CollectiveInit};

    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let tmp = TempDir::new("tiny-model-hip-tp-graphs");
    let spec = write_gpu_tiny(tmp.path());
    let lib = coll::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load hostmem");
    let id = lib.unique_id().expect("id");
    let runs: Vec<(GreedyRun, GreedyRun, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2u32)
            .map(|rank| {
                let (lib, spec) = (Arc::clone(&lib), &spec);
                scope.spawn(move || {
                    let ctx = turbine_kernels::test_support::open_context_nth("hip", rank as usize);
                    let mem: Arc<dyn DeviceMemory> = ctx.clone();
                    let collective = lib
                        .open(CollectiveInit {
                            rank: rank as usize,
                            world: 2,
                            unique_id: id,
                            init_timeout: std::time::Duration::from_secs(60),
                            op_timeout: std::time::Duration::from_secs(60),
                            clock: Arc::new(turbine_core::clock::SystemClock::new()),
                            metrics: None,
                            memory: Some(Arc::clone(&mem)),
                            // hostmem's kernels for every message.
                            route_max_bytes: Some(u64::MAX),
                        })
                        .expect("open hostmem");
                    let s = tp::ShardSpec { rank, world: 2 };
                    let layout = tp::kv_layout(&spec.config, s, BLOCK_TOKENS).expect("layout");
                    let run = |graphs: bool| {
                        let provider = shim_provider(ctx.clone());
                        let mut exec = tp_rank(spec, s, Arc::clone(&collective), provider, &mem);
                        if graphs {
                            exec.set_decode_graphs(Some(hip_graphs(&ctx, MAX_SEQS)));
                        }
                        let storage = pool(&mem, &layout, 4);
                        let got = greedy_pair(
                            exec.as_mut(),
                            &pool_view(&storage, &layout, 4),
                            spec.vocab,
                        );
                        (got, exec.graph_counters().replayed)
                    };
                    let (eager, none) = run(false);
                    assert_eq!(none, 0, "rank {rank}: no graphs in the eager run");
                    let (graphed, replayed) = run(true);
                    (eager, graphed, replayed)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    });
    for (rank, (eager, graphed, replayed)) in runs.iter().enumerate() {
        assert!(
            *replayed >= (TP_STEPS - 2) as u64,
            "rank {rank}: {replayed} replays"
        );
        assert_eq!(graphed.0, eager.0, "rank {rank}: greedy tokens differ");
        assert!(
            graphed.1 == eager.1,
            "rank {rank}: graph logits differ from the eager run"
        );
        assert!(
            graphed.1 == runs[0].1.1,
            "rank {rank}: logits differ from rank 0"
        );
        println!("hostmem_tp2_graphs_match_eager rank {rank}: {replayed} replays, bitwise eager");
    }
}

/// P5 Task 32 (tensor-parallel prefill overlap): the tiny Llama at tp 2 on the host collective
/// (cpu-reference provider, no side stream, so the halves run one after the other), with the
/// prefill split at row 16 of the ragged 20 + 13-token prefill — the first sequence continues in
/// the second half after its first 16 rows' K/V — gives [`greedy_pair`]'s tokens and every
/// logits row bitwise equal to the unsplit run on both ranks. Breaks if a half's attention
/// metadata, its row window, the order of a half's norm and all-reduce, or the continuation of
/// the straddling sequence is wrong.
#[test]
fn tp2_prefill_split_matches_unsplit_on_host() {
    let tmp = TempDir::new("tiny-model-tp-split");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let run = |split: bool| -> Vec<GreedyRun> {
        let group = HostCollective::group(2, std::time::Duration::from_secs(60));
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .enumerate()
                .map(|(rank, collective)| {
                    let spec = &spec;
                    scope.spawn(move || {
                        let s = tp::ShardSpec {
                            rank: rank as u32,
                            world: 2,
                        };
                        let mem: Arc<dyn DeviceMemory> =
                            HostMemory::new(DeviceId(rank as u32), 1 << 30);
                        let collective: Arc<dyn Collective> = Arc::new(collective);
                        let mut exec = tp_decoder(spec, Some((s, collective)), &mem);
                        if split {
                            // Any prefill of 2 rows or more splits.
                            assert!(exec.set_prefill_overlap(Some(2)), "rank {rank}: overlap on");
                        }
                        let layout = *exec.kv_layout();
                        let storage = pool(&mem, &layout, 4);
                        let got =
                            greedy_pair(&mut exec, &pool_view(&storage, &layout, 4), spec.vocab);
                        // The one prefill split; the decode steps never do.
                        assert_eq!(exec.prefill_splits(), u64::from(split), "rank {rank}");
                        got
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        })
    };
    let (serial, split) = (run(false), run(true));
    for rank in 0..2 {
        assert_eq!(split[rank].0, serial[rank].0, "rank {rank}: greedy tokens");
        assert!(
            split[rank].1 == serial[rank].1,
            "rank {rank}: split prefill logits differ from the unsplit run"
        );
    }
    assert!(split[0].1 == split[1].1, "both ranks hold the same logits");
}

/// Lab only, two GPUs (P5 Task 32): [`tp2_prefill_split_matches_unsplit_on_host`] on both
/// R9700s over the `hostmem` collective with every message routed to its RCCL delegate
/// (`route_max_bytes` 0), so the split prefill's all-reduces run on the collective side stream
/// while the compute stream works on the other half (the real overlap): tokens and logits
/// bitwise the unsplit run's on both ranks. Breaks if the side stream's fences let an all-reduce
/// read rows before the compute stream wrote them, or the compute stream read them before the
/// all-reduce finished.
#[test]
#[ignore = "needs two HIP devices, libturbine_hip.so and RCCL (scripts/lab-test.sh novanas --gpus 2)"]
fn hostmem_tp2_prefill_overlap_matches_serial() {
    use turbine_distributed::collective::{self as coll, CollectiveInit};

    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let tmp = TempDir::new("tiny-model-hip-tp-overlap");
    let spec = write_gpu_tiny(tmp.path());
    let lib = coll::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load hostmem with its RCCL delegate");
    let id = lib.unique_id().expect("id");
    let runs: Vec<(GreedyRun, GreedyRun)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2u32)
            .map(|rank| {
                let (lib, spec) = (Arc::clone(&lib), &spec);
                scope.spawn(move || {
                    let ctx = turbine_kernels::test_support::open_context_nth("hip", rank as usize);
                    let mem: Arc<dyn DeviceMemory> = ctx.clone();
                    let collective = lib
                        .open(CollectiveInit {
                            rank: rank as usize,
                            world: 2,
                            unique_id: id,
                            init_timeout: std::time::Duration::from_secs(60),
                            op_timeout: std::time::Duration::from_secs(60),
                            clock: Arc::new(turbine_core::clock::SystemClock::new()),
                            metrics: None,
                            memory: Some(Arc::clone(&mem)),
                            // Every message to RCCL, which runs on the stream it is given.
                            route_max_bytes: Some(0),
                        })
                        .expect("open hostmem");
                    let s = tp::ShardSpec { rank, world: 2 };
                    let layout = tp::kv_layout(&spec.config, s, BLOCK_TOKENS).expect("layout");
                    let run = |overlap: bool| {
                        let provider = shim_provider(ctx.clone());
                        let mut exec = tp_rank(spec, s, Arc::clone(&collective), provider, &mem);
                        if overlap {
                            assert!(exec.set_prefill_overlap(Some(2)), "rank {rank}: overlap on");
                        }
                        let storage = pool(&mem, &layout, 4);
                        greedy_pair(exec.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab)
                    };
                    (run(false), run(true))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    });
    for (rank, (serial, overlapped)) in runs.iter().enumerate() {
        assert_eq!(overlapped.0, serial.0, "rank {rank}: greedy tokens differ");
        assert!(
            overlapped.1 == serial.1,
            "rank {rank}: overlapped prefill logits differ from the serial run"
        );
    }
    println!("hostmem_tp2_prefill_overlap_matches_serial: bitwise the serial run on both ranks");
}

/// `spec`'s decoder on the cpu-reference provider over `mem`: one device (`rank: None`) or rank
/// `s` of a tensor-parallel group on `collective`.
fn tp_decoder(
    spec: &TinySpec,
    rank: Option<(tp::ShardSpec, Arc<dyn Collective>)>,
    mem: &Arc<dyn DeviceMemory>,
) -> DecoderExecutor {
    let cfg = &spec.config;
    let opts = ExecutorOptions::default();
    let decoder = cfg
        .family
        .0
        .tp_decoder_spec()
        .expect("tensor-parallel family");
    let shard = rank.as_ref().map(|(s, _)| *s);
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = match shard {
        Some(s) => tp::weight_slots(cfg, s).expect("shard slots"),
        None => cfg.family.0.weight_slots(cfg),
    };
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load");
    let reqs = DecoderExecutor::requirements_for(cfg, &decoder, BLOCK_TOKENS, opts, shard)
        .expect("requirements");
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, None)
        .expect("every op has a provider");
    DecoderExecutor::new_tp(
        cfg,
        decoder,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: 64,
            max_seqs: MAX_SEQS,
        },
        opts,
        rank.map(|(s, collective)| TpContext {
            rank: s.rank,
            world: s.world,
            collective,
            stream: mem.compute_stream(),
        }),
    )
    .expect("executor")
}

/// The traced prefill of `prompt(vocab)` on `exec` over a fresh pool of its layout.
fn traced_prefill(
    exec: &mut DecoderExecutor,
    mem: &Arc<dyn DeviceMemory>,
    vocab: u32,
) -> Vec<TraceTensor> {
    let layout = *exec.kv_layout();
    let storage = pool(mem, &layout, 1);
    let kv = pool_view(&storage, &layout, 1);
    exec.set_trace(true);
    run_seq(exec, &kv, &[BlockId(0)], &prompt(vocab), 0);
    exec.take_trace()
}

/// Columns `cols` of every row of trace tensor `t`.
fn trace_cols(t: &TraceTensor, cols: std::ops::Range<usize>) -> Vec<f32> {
    t.data
        .chunks_exact(t.shape[1])
        .flat_map(|row| row[cols.clone()].to_vec())
        .collect()
}

/// Max |a − b| over the largest |b|: how far apart two tensors are at their own scale.
fn scaled_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = b
        .iter()
        .fold(0f32, |m, x| m.max(x.abs()))
        .max(f32::MIN_POSITIVE);
    max_abs_diff(a, b) / scale
}

/// P5 S-6, the sharding itself: up to the first all-reduce a tensor-parallel rank computes
/// exactly one device's numbers for its heads. On both tiny checkpoints at tp 2 (host
/// collective, cpu-reference provider), layer 0 of a 20-token prefill: the embedding
/// (vocabulary-parallel, all-reduced: one rank's row plus zeros) is bitwise one device's; each
/// rank's Q, K and V projections are bitwise the columns of its heads and KV head of one
/// device's (the Q/K/V shard loading and fused stacks); for Llama the RoPE outputs and the
/// attention output are too (the rank's KV, positions and head mapping); OLMoE's sharded Q/K
/// norm and everything after the first all-reduce (O projection, the gate/up inputs of the
/// rank's intermediate columns, the residual) agree at the BF16 rounding scale (2^-6 of the
/// tensor's largest value; the partial sums round before they are added). Breaks on a head,
/// KV-head or column taken from the wrong place, which the end-to-end logits comparison only
/// bounds.
#[test]
fn tp_layer0_matches_one_device_slices() {
    let tmp = TempDir::new("tiny-model-tp-trace");
    for spec in both_checkpoints(&tmp) {
        let name = spec.config.family.0.name();
        let olmoe = spec.config.qk_norm;
        let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
        let mut one = tp_decoder(&spec, None, &host);
        let want = traced_prefill(&mut one, &host, spec.vocab);
        let world = 2u32;
        let group = HostCollective::group(world as usize, std::time::Duration::from_secs(60));
        let traces: Vec<Vec<TraceTensor>> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .enumerate()
                .map(|(rank, c)| {
                    let spec = &spec;
                    scope.spawn(move || {
                        let s = tp::ShardSpec {
                            rank: rank as u32,
                            world,
                        };
                        let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(s.rank), 1 << 30);
                        let mut exec = tp_decoder(spec, Some((s, Arc::new(c))), &mem);
                        traced_prefill(&mut exec, &mem, spec.vocab)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        let find = |trace: &[TraceTensor], layer: Option<usize>, op: &str| -> TraceTensor {
            trace
                .iter()
                .find(|t| t.layer == layer && t.name == op)
                .unwrap_or_else(|| panic!("{name}: no {op} in the trace"))
                .clone()
        };
        let cfg = &spec.config;
        let hd = cfg.head_dim as usize;
        let (q_dim, kv_heads) = (
            cfg.num_attention_heads as usize * hd,
            cfg.num_kv_heads as usize,
        );
        let inter = cfg.moe.map_or(cfg.intermediate, |m| m.expert_intermediate) as usize;
        let mut worst = 0f32;
        for (rank, trace) in traces.iter().enumerate() {
            let what = |op: &str| format!("{name} tp {world} rank {rank} layer 0 {op}");
            assert_eq!(
                find(trace, None, "embed").data,
                find(&want, None, "embed").data,
                "{}",
                what("embed")
            );
            let heads = rank * q_dim / 2..(rank + 1) * q_dim / 2;
            let kv_head = rank * kv_heads / 2 * hd..(rank + 1) * kv_heads / 2 * hd;
            let mut exact = vec![
                ("q", heads.clone()),
                ("k", kv_head.clone()),
                ("v", kv_head.clone()),
            ];
            let mut close = vec![
                ("o_proj", 0..cfg.hidden as usize),
                ("resid_mlp", 0..cfg.hidden as usize),
            ];
            if olmoe {
                close.extend([
                    ("q_norm", heads.clone()),
                    ("k_norm", kv_head.clone()),
                    ("attn", heads.clone()),
                ]);
            } else {
                exact.extend([
                    ("q_rope", heads.clone()),
                    ("k_rope", kv_head.clone()),
                    ("attn", heads.clone()),
                ]);
                let cols = rank * inter / 2..(rank + 1) * inter / 2;
                close.extend([("gate", cols.clone()), ("up", cols)]);
            }
            for (op, cols) in exact {
                let got = find(trace, Some(0), op);
                assert_eq!(got.shape[1], cols.len(), "{}", what(op));
                assert!(
                    got.data == trace_cols(&find(&want, Some(0), op), cols),
                    "{} is not one device's bitwise",
                    what(op)
                );
            }
            for (op, cols) in close {
                let got = find(trace, Some(0), op);
                let d = scaled_diff(&got.data, &trace_cols(&find(&want, Some(0), op), cols));
                worst = worst.max(d);
                assert!(d <= 1.0 / 64.0, "{}: scaled diff {d}", what(op));
            }
        }
        println!("tp_layer0_matches_one_device_slices {name}: worst scaled diff {worst:e}");
    }
}

// ----------------------------------------------------------- expert parallelism (P5 S-11)

/// One EP rank's run: its greedy run, its router's choices (every layer's `topk_ids` and
/// `topk_weights` trace, in order) and its token counts afterwards.
struct EpRun {
    run: GreedyRun,
    router: Vec<TraceTensor>,
    counts: ep::ExpertCountsSnapshot,
}

/// Rank `s` of `spec`'s model (an EP rank; with tp = ep also the tensor-parallel rank of the
/// same position, both on `collective`) on `provider` over `mem`, its experts placed by
/// `placement`; the executor and the counts it records.
fn ep_decoder(
    spec: &TinySpec,
    s: ep::EpShard,
    placement: &Arc<ExpertPlacement>,
    collective: Arc<dyn Collective>,
    provider: Arc<dyn KernelProvider>,
    mem: &Arc<dyn DeviceMemory>,
) -> (DecoderExecutor, Arc<ep::ExpertTokenCounts>) {
    let cfg = &spec.config;
    let opts = ExecutorOptions::default();
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = ep::weight_slots(cfg, s, placement).expect("rank slots");
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load rank");
    let reqs = ep::available_requirements(
        cfg,
        s,
        placement,
        BLOCK_TOKENS,
        opts,
        std::slice::from_ref(&provider),
    )
    .expect("requirements");
    let order = [provider.id()];
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
        .expect("every op has a provider");
    let counts = Arc::new(ep::ExpertTokenCounts::default());
    let tp = s.tp_shard().map(|t| TpContext {
        rank: t.rank,
        world: t.world,
        collective: Arc::clone(&collective),
        stream: mem.compute_stream(),
    });
    let ctx = EpContext {
        rank: s.rank,
        world: s.world,
        placement: Arc::clone(placement),
        collective,
        stream: mem.compute_stream(),
        counts: Arc::clone(&counts),
    };
    let decoder = cfg.family.0.tp_decoder_spec().expect("olmoe hooks");
    let exec = DecoderExecutor::new_parallel(
        cfg,
        decoder,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: 64,
            max_seqs: MAX_SEQS,
        },
        opts,
        tp,
        Some(ctx),
    )
    .expect("rank executor");
    (exec, counts)
}

/// Runs `body` on every rank of a `world`-rank EP group (attention `attention`, experts placed
/// by `placement`) over the host collective, one thread per rank, each on the memory and
/// provider `env(rank)` gives, with its own pool of its KV layout; returns every rank's result.
fn ep_group<T: Send>(
    spec: &TinySpec,
    world: u32,
    attention: ep::EpAttention,
    placement: &Arc<ExpertPlacement>,
    env: impl Fn(u32) -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) + Sync,
    body: impl Fn(&mut DecoderExecutor, &KvPoolView<'_>, &ep::ExpertTokenCounts) -> T + Sync,
) -> Vec<T> {
    let group = HostCollective::group(world as usize, std::time::Duration::from_secs(60));
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(rank, collective)| {
                let (env, body) = (&env, &body);
                scope.spawn(move || {
                    let s = ep::EpShard {
                        rank: rank as u32,
                        world,
                        attention,
                    };
                    let (mem, provider) = env(s.rank);
                    let (mut exec, counts) =
                        ep_decoder(spec, s, placement, Arc::new(collective), provider, &mem);
                    let layout = ep::kv_layout(&spec.config, s, BLOCK_TOKENS).expect("layout");
                    assert_eq!(*exec.kv_layout(), layout);
                    let storage = pool(&mem, &layout, 4);
                    body(&mut exec, &pool_view(&storage, &layout, 4), &counts)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    })
}

/// [`greedy_pair`] traced on an EP rank: its run, router choices and counts.
fn ep_greedy_pair(
    exec: &mut DecoderExecutor,
    kv: &KvPoolView<'_>,
    counts: &ep::ExpertTokenCounts,
    vocab: u32,
) -> EpRun {
    exec.set_trace(true);
    let run = greedy_pair(exec, kv, vocab);
    let router = exec
        .take_trace()
        .into_iter()
        .filter(|t| t.name == "topk_ids" || t.name == "topk_weights")
        .collect();
    EpRun {
        run,
        router,
        counts: counts.snapshot(),
    }
}

/// Checks an EP group's runs against one device's run `want` (as
/// [`check_tp_against_one_device`]) and among themselves: the same router choices on every
/// rank at every layer and step, and the same token counts, which account for every routed row
/// of the run exactly once (`tokens · top_k` per layer) and none to a rank without the expert.
fn check_ep_group(what: &str, ranks: &[EpRun], want: &GreedyRun, bound: TpBound, rows: u64) -> f64 {
    for (rank, r) in ranks.iter().enumerate() {
        assert!(!r.router.is_empty(), "{what}: rank {rank} traced no router");
        assert!(
            r.router == ranks[0].router,
            "{what}: rank {rank}'s router choices differ from rank 0's"
        );
        assert_eq!(r.counts, ranks[0].counts, "{what}: rank {rank}'s counts");
    }
    let c = &ranks[0].counts;
    let layers = c.per_layer.len() as u64;
    assert_eq!(
        c.per_rank.iter().sum::<u64>(),
        rows * layers,
        "{what}: {c:?}"
    );
    assert_eq!(
        c.per_expert.iter().sum::<u64>(),
        rows * layers,
        "{what}: {c:?}"
    );
    for (_, per_expert) in &c.per_layer {
        assert_eq!(per_expert.iter().sum::<u64>(), rows, "{what}: {c:?}");
    }
    let runs: Vec<GreedyRun> = ranks.iter().map(|r| r.run.clone()).collect();
    check_tp_against_one_device(what, &runs, want, bound)
}

/// A single-token placement file text for `world` ranks: in every layer the experts
/// `chosen[layer]` on rank 0, the others dealt round-robin over ranks 1.. (rank 0 too when
/// `world` is 1).
fn one_rank_placement(chosen: &[Vec<u32>], experts: u32, world: u32) -> String {
    chosen
        .iter()
        .enumerate()
        .map(|(layer, picked)| {
            let mut next = 0u32;
            let ranks: Vec<String> = (0..experts)
                .map(|e| {
                    if world == 1 || picked.contains(&e) {
                        return "0".to_string();
                    }
                    let r = 1 + next % (world - 1);
                    next += 1;
                    r.to_string()
                })
                .collect();
            format!("{layer}: [{}]\n", ranks.join(", "))
        })
        .collect()
}

/// P5 S-11: the tiny OLMoE (8 experts top-2, Q/K norm, untied LM head) on the cpu-reference
/// provider split into EP 2 and EP 4 (contiguous: 4 or 2 experts per rank) over the host
/// collective, with attention replicated (tp = 1) and tensor-parallel over the same ranks
/// (tp = ep), gives one device's greedy tokens exactly for a ragged two-sequence prefill and 16
/// batched decode steps, with logits bitwise equal on every rank, the same router choices
/// (`topk_ids`, `topk_weights`) on every rank at every layer and step, every routed row counted
/// once, and every row's top-5 logprobs within the strict golden bounds ([`TpBound::Golden`]);
/// an EP group of one is bitwise one device. Then a batch whose every token routes to rank 0's
/// experts: one token, its layers' two chosen experts placed on rank 0 by a placement file
/// (the others dealt over the other ranks, non-contiguously), at tp = 1 gives one device's
/// logits bitwise (the other ranks add exact zeros) and counts every routed row on rank 0.
/// Breaks if a rank drops or double-counts an expert, loads another rank's expert, the combine
/// is missing or doubled (tp = ep) or the router differs between ranks.
#[test]
fn ep2_matches_ep1_on_host() {
    let tmp = TempDir::new("tiny-model-ep");
    let spec = write_tiny_olmoe(tmp.path(), SEED);
    let cfg = &spec.config;
    let moe = cfg.moe.expect("tiny OLMoE has experts");
    let layers = ep::moe_layers(cfg);
    let host = |rank: u32| -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) {
        (
            HostMemory::new(DeviceId(rank), 1 << 30),
            cpu_reference_provider(),
        )
    };
    let (mem, _) = host(0);
    let mut one = cpu_model(&spec, &mem, 64);
    let layout = cfg.kv_layout(BLOCK_TOKENS);
    let storage = pool(&mem, &layout, 4);
    let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
    // greedy_pair routes 20 + 13 prefill and 16 × 2 decode tokens.
    let rows = u64::from((20 + 13 + 2 * TP_STEPS as u32) * moe.experts_per_token);
    let golden = TpBound::Golden(GoldenLogprobBounds::llama());
    use ep::EpAttention::{Replicated, TensorParallel};
    for (world, attention) in [
        (1, Replicated),
        (2, Replicated),
        (4, Replicated),
        (2, TensorParallel),
        (4, TensorParallel),
    ] {
        let placement = Arc::new(
            ExpertPlacement::contiguous(moe.num_experts, &layers, world).expect("placement"),
        );
        let ranks = ep_group(
            &spec,
            world,
            attention,
            &placement,
            host,
            |exec, kv, counts| ep_greedy_pair(exec, kv, counts, spec.vocab),
        );
        let what = format!("ep {world} {attention:?}");
        let bound = if world == 1 { TpBound::Bitwise } else { golden };
        let worst = check_ep_group(&what, &ranks, &want, bound, rows);
        let per_rank = &ranks[0].counts.per_rank;
        assert!(
            per_rank.iter().all(|&r| r > 0),
            "{what}: every rank computes some rows: {per_rank:?}"
        );
        println!(
            "ep2_matches_ep1_on_host {what}: worst top-k |Δ logprob| / golden bound {worst:.3}, \
             rows per rank {per_rank:?}"
        );
    }
    let fresh = pool(&mem, &layout, 1);
    check_one_rank_batch(&spec, one.as_mut(), &pool_view(&fresh, &layout, 1), host);
}

/// The batch whose every token routes to rank 0's experts: one token, whose chosen experts in
/// every layer (read from an EP group of one, itself bitwise one device) a placement file puts
/// on rank 0 (the others dealt over the other ranks, non-contiguously). At EP 2 and EP 4 with
/// tp = 1 every rank ends with one device's logits `one` gives, bitwise (the other ranks add
/// exact zeros), and every routed row is counted on rank 0.
fn check_one_rank_batch(
    spec: &TinySpec,
    one: &mut dyn ModelExecutor,
    kv: &KvPoolView<'_>,
    env: impl Fn(u32) -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) + Sync + Copy,
) {
    use ep::EpAttention::Replicated;
    let cfg = &spec.config;
    let moe = cfg.moe.expect("tiny OLMoE has experts");
    let layers = ep::moe_layers(cfg);
    let token = [prompt(spec.vocab)[3]];
    let one_token = |exec: &mut dyn ModelExecutor, kv: &KvPoolView<'_>| {
        run_seq(exec, kv, &[BlockId(0)], &token, 0)
    };
    let want = one_token(one, kv);
    let single = Arc::new(ExpertPlacement::contiguous(moe.num_experts, &layers, 1).expect("one"));
    let traced = ep_group(spec, 1, Replicated, &single, env, |exec, kv, _| {
        exec.set_trace(true);
        let logits = one_token(exec, kv);
        (logits, exec.take_trace())
    });
    let (logits, trace) = &traced[0];
    assert!(logits == &want, "an EP group of one is one device's logits");
    let chosen: Vec<Vec<u32>> = trace
        .iter()
        .filter(|t| t.name == "topk_ids")
        .map(|t| t.data.iter().map(|&e| e as u32).collect())
        .collect();
    assert_eq!(chosen.len(), layers.len());
    for world in [2u32, 4] {
        let text = one_rank_placement(&chosen, moe.num_experts, world);
        let placement = Arc::new(
            ExpertPlacement::parse(&text, moe.num_experts, &layers, world).expect("placement"),
        );
        let got = ep_group(
            spec,
            world,
            Replicated,
            &placement,
            env,
            |exec, kv, counts| (one_token(exec, kv), counts.snapshot()),
        );
        let per_layer = u64::from(moe.experts_per_token);
        for (rank, (logits, counts)) in got.iter().enumerate() {
            assert!(
                logits == &want,
                "ep {world} rank {rank}: a batch on rank 0's experts is one device's logits"
            );
            let mut on_rank = vec![0u64; world as usize];
            on_rank[0] = per_layer * layers.len() as u64;
            assert_eq!(counts.per_rank, on_rank, "ep {world} rank {rank}");
        }
    }
}

/// P5 S-12: expert parallelism over a dense model (the tiny Llama) is refused before anything
/// is loaded or built, as `parallel.expert_parallel_size` with the reason code `ep_moe_only`;
/// so is a placement over another group size. Breaks if a dense model plans an EP group.
#[test]
fn ep_refuses_dense_model() {
    let tmp = TempDir::new("tiny-model-ep-dense");
    let llama = write_tiny_llama(&tmp.path().join("llama"), SEED);
    let olmoe = write_tiny_olmoe(&tmp.path().join("olmoe"), SEED);
    let s = ep::EpShard {
        rank: 0,
        world: 2,
        attention: ep::EpAttention::Replicated,
    };
    let layers: Vec<u32> = (0..llama.config.num_layers).collect();
    let placement = Arc::new(ExpertPlacement::contiguous(8, &layers, 2).expect("placement"));
    let dense = |e: ModelError| match e {
        ModelError::Unsupported { field, value, .. } => {
            assert_eq!(field, ep::EP_KEY);
            assert!(value.contains(ep::EP_MOE_ONLY), "{value}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    };
    dense(ep::check(&llama.config, s, &placement).unwrap_err());
    dense(ep::weight_slots(&llama.config, s, &placement).unwrap_err());
    dense(
        ep::requirements(
            &llama.config,
            s,
            &placement,
            BLOCK_TOKENS,
            ExecutorOptions::default(),
        )
        .unwrap_err(),
    );
    let four = ExpertPlacement::contiguous(8, &layers, 4).expect("placement");
    let err = ep::check(&olmoe.config, s, &four).unwrap_err().to_string();
    assert!(err.contains(ep::EP_KEY) && err.contains("4 ranks"), "{err}");
}

/// Lab only: expert parallelism on the HIP provider without RCCL — the head_dim-128 tiny OLMoE
/// split into EP 2 over two contexts on the same GPU, through the host collective, at tp = 1
/// and tp = 2, against one HIP context: identical greedy tokens, logits bitwise equal on both
/// ranks, the same router choices and counts on both, top-5 logprobs within the strict golden
/// bounds ([`TpBound::Golden`]). Proves the local-expert `moe_experts` calls (the rank's weight
/// stacks at the range's offset), the per-layer offsets and their read-back, and the combine
/// on device memory. Then the one-token batch on rank 0's experts, bitwise one device's.
///
/// Seed `EP_HIP_SEED` (1), not [`SEED`]: the head_dim-128 tiny OLMoE's random weights leave
/// near-ties between the top two greedy candidates (at seed 7, decode step 1 of sequence 0:
/// one device's logprobs −3.5855 / −3.5860), which the tensor-parallel attention's BF16
/// partial sums flip at tp = 2 (as tensor parallelism alone does at step 14) while every
/// logprob stays ≤ 0.01 from one device's. A sweep of seeds 1–12 on the R9700 and on the
/// cpu-reference provider (2026-09-28): tp = 1 matches one device's tokens at every seed,
/// tp = 2 at seeds 1, 5, 6, 9, 10 and 11.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_ep2_matches_ep1() {
    const EP_HIP_SEED: u64 = 1;
    let tmp = TempDir::new("tiny-model-hip-ep");
    let spec = write_tiny_olmoe_with_head_dim(tmp.path(), EP_HIP_SEED, GPU_HEAD_DIM);
    let cfg = &spec.config;
    let moe = cfg.moe.expect("tiny OLMoE has experts");
    let ctx = hip_context();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let opts = ExecutorOptions::default();
    let mut one = cpu_model_with(&spec, &mem, 64, opts, shim_provider(ctx), false);
    let layout = cfg.kv_layout(BLOCK_TOKENS);
    let storage = pool(&mem, &layout, 4);
    let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
    let rows = u64::from((20 + 13 + 2 * TP_STEPS as u32) * moe.experts_per_token);
    let layers = ep::moe_layers(cfg);
    let placement =
        Arc::new(ExpertPlacement::contiguous(moe.num_experts, &layers, 2).expect("placement"));
    let golden = TpBound::Golden(GoldenLogprobBounds::llama());
    let hip = |_| -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) {
        let ctx = hip_context();
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        (mem, shim_provider(ctx))
    };
    for attention in [ep::EpAttention::Replicated, ep::EpAttention::TensorParallel] {
        let ranks = ep_group(&spec, 2, attention, &placement, hip, |exec, kv, counts| {
            ep_greedy_pair(exec, kv, counts, spec.vocab)
        });
        let what = format!("hip ep 2 {attention:?}");
        let worst = check_ep_group(&what, &ranks, &want, golden, rows);
        println!(
            "hip_ep2_matches_ep1 {what}: worst top-k |Δ logprob| / golden bound {worst:.3}, \
             rows per rank {:?}",
            ranks[0].counts.per_rank
        );
    }
    let fresh = pool(&mem, &layout, 1);
    check_one_rank_batch(&spec, one.as_mut(), &pool_view(&fresh, &layout, 1), hip);
}

// ------------------------------------------------------------ pipeline parallelism (P5 S-10)

/// Every fusion off: residual adds and norms as separate ops, one GEMM per projection.
const UNFUSED: ExecutorOptions = ExecutorOptions {
    fused_ops: false,
    fused_projections: false,
};

/// Stage `stage` of `spec`'s model split into the contiguous `ranges` (one per stage), on
/// `provider` over `mem`, its hand-off on `collective`, run with `opts`.
fn pp_stage(
    spec: &TinySpec,
    ranges: &[std::ops::Range<u32>],
    stage: u32,
    collective: Arc<dyn Collective>,
    provider: Arc<dyn KernelProvider>,
    mem: &Arc<dyn DeviceMemory>,
    opts: ExecutorOptions,
) -> Box<dyn ModelExecutor> {
    let cfg = &spec.config;
    let layers = ranges[stage as usize].clone();
    let stages = ranges.len() as u32;
    let plan: Vec<pp::StageSpec> = ranges
        .iter()
        .enumerate()
        .map(|(s, r)| pp::StageSpec {
            stage: s as u32,
            device: DeviceId(s as u32),
            layers: r.clone(),
            embedding: s == 0,
            lm_head: s + 1 == ranges.len(),
        })
        .collect();
    pp::check(cfg, &plan).expect("pipeline plan");
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = pp::weight_slots(cfg, &plan[stage as usize]).expect("stage slots");
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load stage");
    let reqs = pp::available_requirements(
        cfg,
        &layers,
        BLOCK_TOKENS,
        opts,
        std::slice::from_ref(&provider),
    )
    .expect("requirements");
    let order = [provider.id()];
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
        .expect("every op has a provider");
    pp::build_executor(
        cfg,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: 64,
            max_seqs: MAX_SEQS,
        },
        opts,
        PpContext {
            stage,
            stages,
            layers,
            collective,
            stream: mem.compute_stream(),
        },
    )
    .expect("stage executor")
}

/// [`greedy_pair`]'s batches with the tokens of `chosen` fed after each step (the leader samples
/// from the last stage and feeds every stage the same batch): every step's logits rows (none on
/// a stage before the last, whose every step must return 0 rows).
fn scripted_pair(
    exec: &mut dyn ModelExecutor,
    kv: &KvPoolView<'_>,
    vocab: u32,
    chosen: &[[u32; 2]],
) -> Vec<Vec<f32>> {
    let prompts: [Vec<u32>; 2] = [
        prompt(vocab),
        (0..13u32).map(|i| (i * 29 + 3) % vocab).collect(),
    ];
    let tables = [[BlockId(0)], [BlockId(1)]];
    let mut lens = [prompts[0].len() as u32, prompts[1].len() as u32];
    let tokens: Vec<u32> = prompts.concat();
    let positions: Vec<u32> = prompts.iter().flat_map(|p| 0..p.len() as u32).collect();
    let seqs = [
        SeqSlice {
            seq: SeqId(1),
            q_start: 0,
            q_len: lens[0],
            kv_len: lens[0],
            block_table: &tables[0],
            reduce: None,
        },
        SeqSlice {
            seq: SeqId(2),
            q_start: lens[0],
            q_len: lens[1],
            kv_len: lens[1],
            block_table: &tables[1],
            reduce: None,
        },
    ];
    let mut steps = vec![
        exec.forward(&BatchInput {
            tokens: &tokens,
            positions: &positions,
            seqs: &seqs,
            kv,
        })
        .expect("prefill"),
    ];
    for next in chosen {
        let positions = lens;
        lens = [lens[0] + 1, lens[1] + 1];
        let seqs: Vec<SeqSlice<'_>> = (0..2)
            .map(|s| SeqSlice {
                seq: SeqId(s as u64 + 1),
                q_start: s as u32,
                q_len: 1,
                kv_len: lens[s],
                block_table: &tables[s],
                reduce: None,
            })
            .collect();
        steps.push(
            exec.forward(&BatchInput {
                tokens: next,
                positions: &positions,
                seqs: &seqs,
                kv,
            })
            .expect("decode"),
        );
    }
    let mut rows = Vec::new();
    for logits in &steps {
        if logits.rows == 0 {
            assert!(logits.reduced.is_empty());
            continue;
        }
        assert_eq!(logits.rows, 2);
        rows.extend([logits.row(0).to_vec(), logits.row(1).to_vec()]);
    }
    assert!(
        rows.is_empty() || rows.len() == 2 * steps.len(),
        "a stage returns logits at every step or at none"
    );
    rows
}

/// Runs [`scripted_pair`] with one device's greedy tokens on every stage of `spec`'s model
/// split into `ranges`, over the host collective's `send` / `recv`, one thread per stage, each
/// on the memory and provider `env(stage)` gives with its own pool of its layers' KV layout;
/// returns every stage's rows.
fn pp_group(
    spec: &TinySpec,
    ranges: &[std::ops::Range<u32>],
    opts: ExecutorOptions,
    chosen: &[[u32; 2]],
    env: impl Fn(u32) -> (Arc<dyn DeviceMemory>, Arc<dyn KernelProvider>) + Sync,
) -> Vec<Vec<Vec<f32>>> {
    let group = HostCollective::group(ranges.len(), std::time::Duration::from_secs(60));
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .enumerate()
            .map(|(stage, collective)| {
                let env = &env;
                scope.spawn(move || {
                    let stage = stage as u32;
                    let range = &ranges[stage as usize];
                    let (mem, provider) = env(stage);
                    let collective: Arc<dyn Collective> = Arc::new(collective);
                    let mut exec = pp_stage(spec, ranges, stage, collective, provider, &mem, opts);
                    let layout =
                        pp::kv_layout(&spec.config, range, BLOCK_TOKENS).expect("stage layout");
                    assert_eq!(*exec.kv_layout(), layout);
                    assert_eq!(layout.num_layers as usize, range.len());
                    let storage = pool(&mem, &layout, 4);
                    let kv = pool_view(&storage, &layout, 4);
                    scripted_pair(exec.as_mut(), &kv, spec.vocab, chosen)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("stage thread"))
            .collect()
    })
}

/// Checks a pipeline group's rows against one device's run `want`: every stage before the last
/// returned no logits, the last stage's rows are bitwise one device's (pipeline parallelism
/// moves the residual stream's bytes; it does not change arithmetic) and their greedy tokens
/// are the tokens fed.
fn check_pp_against_one_device(what: &str, stages: &[Vec<Vec<f32>>], want: &GreedyRun) {
    let (last, before) = stages.split_last().expect("a stage");
    for (s, rows) in before.iter().enumerate() {
        assert!(rows.is_empty(), "{what}: stage {s} returned logits");
    }
    assert_eq!(last.len(), want.1.len(), "{what}: logits rows");
    for (i, (got, w)) in last.iter().zip(&want.1).enumerate() {
        assert!(got == w, "{what}: row {i} differs from one device");
    }
    let tokens: Vec<[u32; 2]> = last
        .chunks_exact(2)
        .take(want.0.len())
        .map(|r| [argmax(&r[0]), argmax(&r[1])])
        .collect();
    assert_eq!(tokens, want.0, "{what}: greedy tokens");
}

/// P5 S-10: the tiny Llama (tied embedding: on the first and the last stage) and the tiny
/// OLMoE (untied LM head, Q/K norm, 8 experts top-2) on the cpu-reference provider, split into
/// 2 stages of one layer each (the tiny checkpoints have 2 layers, so 3 or 4 stages do not fit)
/// over the host collective's `send` / `recv`, each stage its own host "device" and KV pool of
/// its layer on its own thread, fed the same batches (a ragged two-sequence prefill and 16
/// batched decode steps, one device's greedy tokens fed back): the last stage's logits are
/// bitwise one device's, the first stage returns none; a single stage over both layers is
/// bitwise one device too. Both run with every fusion on (fused Q/K/V and the fused residual
/// add + norm, which one device runs across the stage boundary and a stage splits into its
/// last add and the next stage's input norm) and with every fusion off. Breaks if a stage
/// boundary drops or repeats a layer, the hand-off moves the wrong rows, a stage's KV holds
/// another stage's layers, or the tied embedding is missing on the last stage.
#[test]
fn pp2_matches_pp1_on_host() {
    let tmp = TempDir::new("tiny-model-pp");
    for spec in both_checkpoints(&tmp) {
        let name = spec.config.family.0.name();
        let layers = spec.config.num_layers;
        for opts in [UNFUSED, ALL_FUSED] {
            let host: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(0), 1 << 30);
            let mut one = cpu_model_with(&spec, &host, 64, opts, cpu_reference_provider(), false);
            let layout = spec.config.kv_layout(BLOCK_TOKENS);
            let storage = pool(&host, &layout, 4);
            let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
            let splits: Vec<Vec<std::ops::Range<u32>>> =
                vec![vec![0..layers], (0..layers).map(|l| l..l + 1).collect()];
            for ranges in splits {
                let stages = pp_group(&spec, &ranges, opts, &want.0, |stage| {
                    let mem: Arc<dyn DeviceMemory> = HostMemory::new(DeviceId(stage), 1 << 30);
                    (mem, cpu_reference_provider())
                });
                let what = format!("{name} pp {} {opts:?}", ranges.len());
                check_pp_against_one_device(&what, &stages, &want);
                println!("pp2_matches_pp1_on_host {what}: bitwise");
            }
        }
    }
}

/// Lab only: the pipeline stages on the HIP provider without RCCL — the head_dim-128 tiny Llama
/// split into 2 stages, each its own context on the same GPU, over the host collective (which
/// reaches any device memory), against one HIP context, with every fusion on (the fused
/// residual add + norm across the boundary against the stage's add and the next stage's norm)
/// and off: the last stage's logits bitwise one device's. Proves the HIP stage loading, the
/// per-stage KV pools and the hand-off of device rows.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_pp2_matches_pp1() {
    let tmp = TempDir::new("tiny-model-hip-pp");
    let spec = write_gpu_tiny(tmp.path());
    let layers = spec.config.num_layers;
    for opts in [UNFUSED, ALL_FUSED] {
        let ctx = hip_context();
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let mut one = cpu_model_with(&spec, &mem, 64, opts, shim_provider(ctx), false);
        let layout = spec.config.kv_layout(BLOCK_TOKENS);
        let storage = pool(&mem, &layout, 4);
        let want = greedy_pair(one.as_mut(), &pool_view(&storage, &layout, 4), spec.vocab);
        let ranges: Vec<std::ops::Range<u32>> = (0..layers).map(|l| l..l + 1).collect();
        let stages = pp_group(&spec, &ranges, opts, &want.0, |_| {
            let ctx = hip_context();
            let mem: Arc<dyn DeviceMemory> = ctx.clone();
            (mem, shim_provider(ctx))
        });
        let what = format!("hip pp 2 {opts:?}");
        check_pp_against_one_device(&what, &stages, &want);
        println!("hip_pp2_matches_pp1 {what}: bitwise");
    }
}
