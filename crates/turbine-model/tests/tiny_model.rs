//! The Llama executor on the tiny synthetic checkpoint (P1 S-8, S-12): the `cpu-reference`
//! provider against an independent naive implementation, and (lab only) the HIP provider
//! against the CPU provider.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use half::bf16;
use turbine_core::types::{BlockId, SeqId};
use turbine_core::types::{DeviceId, ExecutionBackend, Vendor};
use turbine_kernels::{
    KernelError, KernelMetrics, KernelProvider, KernelRegistry, cpu_reference_provider,
    shim_provider,
};
use turbine_model::config::{ModelArchConfig, RopeScaling};
use turbine_model::executor::{
    BatchInput, LlamaExecutor, Logits, ModelExecutor, SeqSlice, SequenceKv, TraceTensor,
};
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::{
    TinyOptions, TinySpec, write_tiny_llama, write_tiny_llama_with,
};
use turbine_model::testing::trace::{LocalChecker, compare_traces, read_bf16_weight, render};
use turbine_model::{MAX_STAGING_BYTES, ModelError, SafetensorsIndex, WeightLoader, llama_slots};
use turbine_observability::MetricsRegistry;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

const SEED: u64 = 7;
const PROMPT_LEN: usize = 20;
const DECODE_STEPS: usize = 30;
const MAX_SEQ_LEN: u32 = 64;
/// KV block size of every test pool.
const BLOCK_TOKENS: u32 = 16;
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
    exec: LlamaExecutor,
    kv: SequenceKv,
}

impl std::ops::Deref for Single {
    type Target = LlamaExecutor;
    fn deref(&self) -> &LlamaExecutor {
        &self.exec
    }
}

impl std::ops::DerefMut for Single {
    fn deref_mut(&mut self) -> &mut LlamaExecutor {
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

fn paged_executor(
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
        &LlamaExecutor::requirements(cfg, BLOCK_TOKENS),
        &metrics,
    )
    .expect("every op has a provider");
    LlamaExecutor::new(
        cfg,
        weights,
        Arc::new(registry),
        mem,
        BLOCK_TOKENS,
        MAX_SEQ_LEN,
        MAX_SEQS,
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
    let short = [BlockId(0)];
    let err = run(
        &[1],
        &[16],
        &[SeqSlice {
            block_table: &short,
            ..seq(0, 0, 1, 17)
        }],
    );
    assert!(err.contains("needs 2 blocks"), "{err}");
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
            block_tokens: 8,
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
    let reqs = LlamaExecutor::requirements(&spec.config, BLOCK_TOKENS);
    let rendered: Vec<String> = reqs
        .iter()
        .map(|r| format!("{} {}", r.op, r.config))
        .collect();
    // Distinct configs only, every op family the forward pass runs plus the block fork.
    let mut unique = rendered.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), rendered.len(), "{rendered:#?}");
    // One block of one layer: 2 × 16 tokens × 2 kv heads × 16 head_dim × 2 bytes.
    let layer_block = 2 * 16 * 2 * 16 * 2;
    let copy = format!("copy_blocks num_layers=2 block_bytes={layer_block}");
    for want in [
        "embedding hidden=64 vocab_rows=263 dtype=bf16",
        "rmsnorm dim=64 dtype=bf16",
        "gemm n=64 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=32 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=128 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=64 k=128 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16",
        "gemm n=263 k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=f32",
        "rope head_dim=16 rotary_dim=16 q_heads=4 kv_heads=2 dtype=bf16",
        "attention_prefill_paged head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1 block_tokens=16",
        "attention_decode_paged head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1 block_tokens=16",
        "silu_mul cols=128 dtype=bf16",
        "add dtype=bf16",
        copy.as_str(),
    ] {
        assert!(
            rendered.iter().any(|r| r == want),
            "missing {want}: {rendered:#?}"
        );
    }
    assert_eq!(rendered.len(), 13, "{rendered:#?}");

    // Workspace grows linearly in the batch token count and in the sequence count.
    let ws = |t, n| LlamaExecutor::workspace_bytes(&spec.config, BLOCK_TOKENS, t, n);
    let per_token = ws(2, 1) - ws(1, 1);
    assert_eq!(ws(64, 1) - ws(1, 1), 63 * per_token);
    // Per token: ids + positions (i32), x/h/proj [hidden], q/attn [q_dim], k/v [kv_dim],
    // gate/up/act [intermediate], all bf16.
    assert_eq!(per_token, 4 + 4 + 2 * (3 * 64 + 2 * 64 + 2 * 32 + 3 * 128));
    // Per sequence: last row [hidden] bf16, logits [vocab] f32, q_indptr + kv_lens entries and
    // a block table for max_position_embeddings tokens (i32).
    let blocks = u64::from(spec.config.max_position_embeddings.div_ceil(BLOCK_TOKENS));
    assert_eq!(ws(1, 2) - ws(1, 1), 2 * 64 + 4 * 263 + 4 * (2 + blocks));
    // Fixed: the extra q_indptr entry and inv_freq [head_dim / 2] f32.
    assert_eq!(ws(1, 1) - per_token - (ws(1, 2) - ws(1, 1)), 4 + 4 * 8);
}

/// One pool shared by the paged tests: `blocks` blocks of the executor's layout.
fn pool(mem: &Arc<dyn DeviceMemory>, exec: &LlamaExecutor, blocks: u32) -> DeviceBuffer {
    let layout = exec.kv_layout();
    let bytes = layout.block_bytes() * u64::from(blocks);
    DeviceBuffer::alloc(mem, bytes as usize).expect("pool")
}

fn pool_view<'a>(storage: &'a DeviceBuffer, exec: &LlamaExecutor, blocks: u32) -> KvPoolView<'a> {
    let layout = *exec.kv_layout();
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
    let storage = pool(&mem, exec, 12);
    let noise: Vec<u8> = (0..storage.len()).map(|i| (i * 7 % 251) as u8).collect();
    storage.whole().write_bytes(&noise).expect("fill pool");
    let kv = pool_view(&storage, exec, 12);
    let table = [BlockId(9), BlockId(2), BlockId(7), BlockId(4)];
    let run = |exec: &mut LlamaExecutor, tokens: &[u32], start: u32| {
        let positions: Vec<u32> = (start..start + tokens.len() as u32).collect();
        let seqs = [SeqSlice {
            seq: SeqId(42),
            q_start: 0,
            q_len: tokens.len() as u32,
            kv_len: start + tokens.len() as u32,
            block_table: &table,
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
    let storage = pool(&mem, exec, 10);
    let kv = pool_view(&storage, exec, 10);
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
        },
        SeqSlice {
            seq: SeqId(2),
            q_start: 1,
            q_len: 17,
            kv_len: 17,
            block_table: &tb,
        },
        SeqSlice {
            seq: SeqId(3),
            q_start: 18,
            q_len: 3,
            kv_len: 3,
            block_table: &tc,
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
    let decode = |exec: &mut LlamaExecutor, table: &[BlockId]| {
        let seqs = [SeqSlice {
            seq: SeqId(3),
            q_start: 0,
            q_len: 1,
            kv_len: 4,
            block_table: table,
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

/// Max |Δ logit| between the HIP and cpu-reference providers on the tiny checkpoint. With the
/// cpu-reference attention in CK's order (BF16 unnormalised probabilities for P·V, f32 row
/// sum), every op models the HIP numerics and the measured worst case over 32 steps on an R9700
/// is 3.8e-6 (0.097 while the cpu-reference kept f32 probabilities). 1e-4 leaves a 25× margin
/// for f32 accumulation order and is ~1000× below the error one numerics-model mismatch caused.
const HIP_MAX_ABS_LOGIT_DIFF: f32 = 1e-4;

/// Lab only (Task 21): HIP provider logits on the head_dim-128 tiny checkpoint match the CPU
/// provider within [`HIP_MAX_ABS_LOGIT_DIFF`] and 32 greedy tokens are identical.
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
    println!("hip_matches_cpu: max abs logit diff over 32 steps {worst}");
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
