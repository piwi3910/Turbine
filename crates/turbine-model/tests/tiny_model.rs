//! The Llama executor on the tiny synthetic checkpoint (P1 S-8, S-12): the `cpu-reference`
//! provider against an independent naive implementation, and (lab only) the HIP provider
//! against the CPU provider.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use half::bf16;
use turbine_core::types::{DeviceId, ExecutionBackend, Vendor};
use turbine_kernels::{
    KernelError, KernelMetrics, KernelProvider, KernelRegistry, cpu_reference_provider,
    shim_provider,
};
use turbine_model::config::{ModelArchConfig, RopeScaling};
use turbine_model::executor::{BatchInput, LlamaExecutor, Logits, ModelExecutor};
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::{
    TinyOptions, TinySpec, write_tiny_llama, write_tiny_llama_with,
};
use turbine_model::{MAX_STAGING_BYTES, ModelError, SafetensorsIndex, WeightLoader, llama_slots};
use turbine_observability::MetricsRegistry;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

const SEED: u64 = 7;
const PROMPT_LEN: usize = 20;
const DECODE_STEPS: usize = 30;
const MAX_SEQ_LEN: u32 = 64;
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

fn executor(
    spec: &TinySpec,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
) -> LlamaExecutor {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let weights =
        WeightLoader::load(&index, &llama_slots(cfg), &mem, MAX_STAGING_BYTES).expect("load");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &LlamaExecutor::requirements(cfg),
        &metrics,
    )
    .expect("every op has a provider");
    LlamaExecutor::new(
        cfg,
        weights,
        Arc::new(registry),
        mem,
        MAX_SEQ_LEN,
        MAX_SEQ_LEN,
    )
    .expect("executor")
}

fn cpu_executor(spec: &TinySpec) -> LlamaExecutor {
    executor(
        spec,
        cpu_reference_provider(),
        HostMemory::new(DeviceId(0), 1 << 30),
    )
}

fn forward(exec: &mut LlamaExecutor, tokens: &[u32], start: u32) -> Logits {
    let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
    exec.forward(&BatchInput {
        tokens,
        positions: &positions,
    })
    .expect("forward")
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
// RoPE (cos, sin, each product and each sum), attention output, silu(gate) and silu·up, and
// each residual sum. Logits stay f32.

fn bf(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

struct Naive {
    cfg: ModelArchConfig,
    w: HashMap<String, Vec<f32>>,
    inv_freq: Vec<f32>,
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
                let o = &mut out[(i * hq + h) * d..][..d];
                for (j, s) in scores.iter().enumerate() {
                    let p = s / sum;
                    let vv = &v[(j * hkv + kvh) * d..][..d];
                    for (acc, x) in o.iter_mut().zip(vv) {
                        *acc += p * x;
                    }
                }
            }
        }
        out.into_iter().map(bf).collect()
    }

    /// Logits of the last position of `tokens` (positions 0..len).
    fn logits(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let hidden = c.hidden as usize;
        let q_dim = c.num_attention_heads as usize * c.head_dim as usize;
        let inter = c.intermediate as usize;
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
            self.rope(&mut q, c.num_attention_heads as usize);
            self.rope(&mut k, c.num_kv_heads as usize);
            let attn = self.attention(&q, &k, &v, t);
            let o = Naive::linear_bf(&attn, w("self_attn.o_proj"), q_dim);
            x = x.iter().zip(&o).map(|(a, b)| bf(a + b)).collect();
            let h = self.rmsnorm(&x, w("post_attention_layernorm"));
            let gate = Naive::linear_bf(&h, w("mlp.gate_proj"), hidden);
            let up = Naive::linear_bf(&h, w("mlp.up_proj"), hidden);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| bf(bf(g / (1.0 + (-g).exp())) * u))
                .collect();
            let down = Naive::linear_bf(&act, w("mlp.down_proj"), inter);
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

/// transformers `_compute_llama3_parameters`, written out independently in f64.
fn naive_inv_freq(cfg: &ModelArchConfig) -> Vec<f32> {
    let d = f64::from(cfg.head_dim);
    let Some(RopeScaling::Llama3 {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    }) = cfg.rope_scaling
    else {
        panic!("the tiny checkpoint uses llama3 rope scaling");
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
    let invalid = |r: Result<Logits, ModelError>| match r {
        Err(ModelError::Kernel(KernelError::InvalidArgument { message })) => message,
        Err(other) => panic!("expected InvalidArgument, got {other}"),
        Ok(_) => panic!("expected InvalidArgument, got logits"),
    };

    let empty = invalid(exec.forward(&BatchInput {
        tokens: &[],
        positions: &[],
    }));
    assert!(empty.contains("empty"), "{empty}");
    let gap = invalid(exec.forward(&BatchInput {
        tokens: &[1, 2],
        positions: &[0, 2],
    }));
    assert!(gap.contains("consecutive"), "{gap}");
    // Nothing is cached yet: the batch cannot start past position 0.
    let ahead = invalid(exec.forward(&BatchInput {
        tokens: &[1],
        positions: &[3],
    }));
    assert!(ahead.contains("cached"), "{ahead}");
    let mismatch = invalid(exec.forward(&BatchInput {
        tokens: &[1, 2],
        positions: &[0],
    }));
    assert!(mismatch.contains("positions"), "{mismatch}");
    let out_of_vocab = invalid(exec.forward(&BatchInput {
        tokens: &[spec.vocab],
        positions: &[0],
    }));
    assert!(out_of_vocab.contains("vocab"), "{out_of_vocab}");

    let full: Vec<u32> = (0..MAX_SEQ_LEN).map(|i| i % spec.vocab).collect();
    forward(&mut exec, &full, 0);
    let beyond = invalid(exec.forward(&BatchInput {
        tokens: &[1],
        positions: &[MAX_SEQ_LEN],
    }));
    assert!(beyond.contains("max_seq_len"), "{beyond}");
}

#[test]
fn requirements_and_workspace() {
    let tmp = TempDir::new("tiny-model-reqs");
    let spec = write_tiny_llama(tmp.path(), SEED);
    let reqs = LlamaExecutor::requirements(&spec.config);
    let rendered: Vec<String> = reqs
        .iter()
        .map(|r| format!("{} {}", r.op, r.config))
        .collect();
    // Distinct configs only, every op family the forward pass runs.
    let mut unique = rendered.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), rendered.len(), "{rendered:#?}");
    for want in [
        "embedding hidden=64 vocab_rows=263 dtype=bf16",
        "rmsnorm dim=64 dtype=bf16",
        "gemm n=64 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=32 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=128 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=64 k=128 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=263 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32",
        "rope head_dim=16 rotary_dim=16 q_heads=4 kv_heads=2 dtype=bf16",
        "attention_prefill head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1",
        "attention_decode head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1",
        "silu_mul cols=128 dtype=bf16",
        "add dtype=bf16",
    ] {
        assert!(rendered.iter().any(|r| r == want), "missing {want}");
    }
    assert_eq!(rendered.len(), 12, "{rendered:#?}");

    // Workspace grows linearly in the forward token count, with a fixed part for the last
    // row, the logits and inv_freq.
    let w1 = LlamaExecutor::workspace_bytes(&spec.config, 1);
    let w2 = LlamaExecutor::workspace_bytes(&spec.config, 2);
    let w64 = LlamaExecutor::workspace_bytes(&spec.config, 64);
    assert_eq!(w64 - w1, 63 * (w2 - w1));
    // Per token: ids + positions (i32), x/h/proj [hidden], q/attn [q_dim], gate/up/act
    // [intermediate], all bf16.
    assert_eq!(w2 - w1, 4 + 4 + 2 * (3 * 64 + 2 * 64 + 3 * 128));
    // Fixed: last row [hidden] bf16, logits [vocab] f32, inv_freq [head_dim / 2] f32.
    assert_eq!(w1 - (w2 - w1), 2 * 64 + 4 * 263 + 4 * 8);
}

/// Lab only (Task 21): HIP provider logits on the head_dim-128 tiny checkpoint match the CPU
/// provider within 2e-2 and 32 greedy tokens are identical.
#[test]
#[ignore = "needs a HIP device and TURBINE_KERNEL_LIBRARY"]
fn hip_matches_cpu() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .expect("TURBINE_KERNEL_LIBRARY is not set; point it at libturbine_hip.so");
    let lib = turbine_kernels::ShimLibrary::load(Path::new(&library), ExecutionBackend::Hip)
        .expect("load the HIP kernel library");
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let device = inventory
        .devices
        .iter()
        .find(|d| d.vendor == Vendor::Amd)
        .expect("an AMD device");
    let ctx = lib.create_context(device).expect("HIP context");

    let tmp = TempDir::new("tiny-model-hip");
    let spec = write_gpu_tiny(tmp.path());
    let mut cpu = cpu_executor(&spec);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = executor(&spec, shim_provider(ctx), mem);

    let mut tokens = prompt(spec.vocab);
    let mut cpu_row = forward(&mut cpu, &tokens, 0).row(0).to_vec();
    let mut hip_row = forward(&mut hip, &tokens, 0).row(0).to_vec();
    for step in 0..32 {
        let diff = max_abs_diff(&hip_row, &cpu_row);
        assert!(diff <= 2e-2, "step {step}: max abs diff {diff}");
        let (c, h) = (argmax(&cpu_row), argmax(&hip_row));
        assert_eq!(h, c, "greedy token {step} differs");
        let pos = tokens.len() as u32;
        tokens.push(c);
        cpu_row = forward(&mut cpu, &[c], pos).row(0).to_vec();
        hip_row = forward(&mut hip, &[c], pos).row(0).to_vec();
    }
}
