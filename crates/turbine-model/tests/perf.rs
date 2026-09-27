//! Lab performance profile (P2c S-2, S-15): the real executors on the R9700 with profile mode
//! on. `forward_profile` loads Llama-3.2-3B-Instruct (`TURBINE_TEST_MODEL_DIR`) and
//! OLMoE-1B-7B-0125-Instruct (`TURBINE_TEST_MOE_MODEL_DIR`) on the HIP provider and prints, per
//! model, one `forward_profile: <json>` line: for a decode step of 1, 16 and 64 sequences at a
//! 768-token context and for one 2,048-token prefill chunk, the unprofiled forward time (median
//! of 5) and the per-op profile ([`OpProfile`]). The kernel-level counterpart is
//! `hip_ops decode_forward_timing` (a synthetic forward of the same shapes): the difference
//! between the two forward times is the executor's own overhead. Llama runs twice, with the
//! default options and with every fusion on (the opt-in projection fusion included); each line
//! names its `fused_ops` and `fused_projections`. `serving_mix` (below) times decode steps under
//! a server-like continuous batch for every fusion combination. `moe_prefill_timings` times every
//! `moe_experts` implementation alone at the served prefill sizes with OLMoE's real routing and
//! weights (perf item #2).
//!
//! Run with `scripts/lab-test.sh novanas -- --release -p turbine-model --test perf -- forward_profile --nocapture`.
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde::Serialize;
use turbine_core::types::{BlockId, KvLayout, SeqId};
use turbine_kernels::{
    KernelMetrics, KernelRegistry, OpKind, shim_provider, test_support::require_backend,
    test_support::require_env_dir,
};
use turbine_model::config::ModelArchConfig;
use turbine_model::executor::{
    self, BatchInput, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, OpProfile,
    OpProfileEntry, SeqSlice,
};
use turbine_model::families;
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, load_model_config};
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

/// KV page size of the profile (the Phase 2c default: both paged ops on CK `fmha_fwd_pagedkv`).
const BLOCK_TOKENS: u32 = 128;
/// Context of every decode case, the new token included.
const DECODE_CTX: u32 = 768;
/// Tokens of the prefill case (one chunk of one sequence).
const PREFILL: u32 = 2048;
const DECODE_BATCHES: [u32; 3] = [1, 16, 64];
const MAX_SEQS: u32 = 64;
/// Unprofiled runs per case; the median is reported.
const RUNS: usize = 5;

/// Serializes the tests of this binary: each loads a model and a KV pool that together fill
/// most of the R9700's memory, so two at once run out of it.
static GPU: Mutex<()> = Mutex::new(());

fn lock_gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The profile switches of both executors, so one body drives either.
trait Profiled: ModelExecutor {
    fn set_profile(&mut self, on: bool);
    fn take_profile(&mut self) -> OpProfile;
}

impl Profiled for DecoderExecutor {
    fn set_profile(&mut self, on: bool) {
        DecoderExecutor::set_profile(self, on);
    }
    fn take_profile(&mut self) -> OpProfile {
        DecoderExecutor::take_profile(self)
    }
}

#[derive(Serialize)]
struct CaseReport {
    case: String,
    forward_ms: f64,
    profiled_ms: f64,
    ops: Vec<OpProfileEntry>,
}

#[derive(Serialize)]
struct ModelReport {
    model: String,
    block_tokens: u32,
    fused_ops: bool,
    fused_projections: bool,
    cases: Vec<CaseReport>,
}

/// The requirements the HIP provider serves for `cfg` with the default (fused) op sequence.
fn hip_requirements(
    cfg: &ModelArchConfig,
    ctx: &Arc<turbine_kernels::ShimContext>,
    opts: ExecutorOptions,
) -> Vec<turbine_kernels::OpRequirement> {
    executor::available_requirements(cfg, BLOCK_TOKENS, opts, &[shim_provider(ctx.clone())])
}

/// The model's executor on the HIP provider, for batches of up to [`PREFILL`] tokens and
/// [`MAX_SEQS`] sequences, running the default (fused) op sequence.
fn hip_executor(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
    opts: ExecutorOptions,
    max_batch_tokens: u32,
) -> Box<dyn Profiled> {
    Box::new(hip_decoder(cfg, dir, ctx, opts, max_batch_tokens))
}

/// [`hip_executor`] unboxed (for tracing).
fn hip_decoder(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
    opts: ExecutorOptions,
    max_batch_tokens: u32,
) -> DecoderExecutor {
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let card = provider.card_profile();
    let registry = Arc::new(
        KernelRegistry::build(
            vec![provider],
            &order,
            &hip_requirements(cfg, ctx, opts),
            &metrics,
            card,
        )
        .expect("every op has a provider"),
    );
    let spec = match cfg.family.0.name() {
        "llama" => families::llama::decoder_spec(),
        _ => families::olmoe::decoder_spec(),
    };
    let limits = ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens,
        max_seqs: MAX_SEQS,
    };
    DecoderExecutor::new(cfg, spec, weights, registry, mem, limits, opts).expect("decoder executor")
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

/// One ragged step over `seqs` (each `(table, tokens, first position)`).
fn step(
    exec: &mut dyn Profiled,
    kv: &KvPoolView<'_>,
    seqs: &[(&[BlockId], &[u32], u32)],
) -> Result<(), turbine_model::ModelError> {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut slices = Vec::with_capacity(seqs.len());
    for (s, &(table, toks, start)) in seqs.iter().enumerate() {
        slices.push(SeqSlice {
            seq: SeqId(s as u64 + 1),
            q_start: tokens.len() as u32,
            q_len: toks.len() as u32,
            kv_len: start + toks.len() as u32,
            block_table: table,
            reduce: None,
        });
        tokens.extend_from_slice(toks);
        positions.extend(start..start + toks.len() as u32);
    }
    exec.forward(&BatchInput {
        tokens: &tokens,
        positions: &positions,
        seqs: &slices,
        kv,
    })?;
    Ok(())
}

/// Median of `RUNS` unprofiled runs of `run`, then one profiled run (after a profiled warm-up).
fn profile_case(
    name: &str,
    exec: &mut dyn Profiled,
    mut run: impl FnMut(&mut dyn Profiled),
) -> CaseReport {
    run(exec);
    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run(exec);
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    let forward_ms = times[RUNS / 2];
    exec.set_profile(true);
    run(exec);
    exec.take_profile();
    run(exec);
    let profile = exec.take_profile();
    exec.set_profile(false);
    CaseReport {
        case: name.to_string(),
        forward_ms,
        profiled_ms: profile.total_ms(),
        ops: profile.entries,
    }
}

/// Profiles the model in `dir`: fills the KV of [`MAX_SEQS`] sequences with distinct
/// `DECODE_CTX − 1`-token prompts, then runs the decode and prefill cases.
fn profile_model(
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
    opts: ExecutorOptions,
) -> ModelReport {
    let cfg = load_model_config(dir).expect("config.json");
    let mut exec = hip_executor(&cfg, dir, ctx, opts, PREFILL);
    let layout = *exec.kv_layout();
    let per_decode = DECODE_CTX.div_ceil(BLOCK_TOKENS);
    let per_prefill = PREFILL.div_ceil(BLOCK_TOKENS);
    let blocks = MAX_SEQS * per_decode + per_prefill;
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let storage = DeviceBuffer::alloc(&mem, (layout.block_bytes() * u64::from(blocks)) as usize)
        .expect("KV pool");
    let kv = pool_view(&storage, &layout, blocks);
    let vocab = cfg.vocab_size;
    let tables: Vec<Vec<BlockId>> = (0..MAX_SEQS)
        .map(|s| {
            (0..per_decode)
                .map(|b| BlockId(s * per_decode + b))
                .collect()
        })
        .collect();
    let prompts: Vec<Vec<u32>> = (0..MAX_SEQS)
        .map(|s| {
            (0..DECODE_CTX - 1)
                .map(|i| (i * 7919 + s * 104_729 + 1000) % vocab)
                .collect()
        })
        .collect();
    // Two prompts per prefill step (2 × 767 ≤ PREFILL tokens).
    for pair in (0..MAX_SEQS as usize).collect::<Vec<_>>().chunks(2) {
        let seqs: Vec<(&[BlockId], &[u32], u32)> = pair
            .iter()
            .map(|&s| (tables[s].as_slice(), prompts[s].as_slice(), 0))
            .collect();
        step(exec.as_mut(), &kv, &seqs).expect("fill the KV");
    }

    let mut cases = Vec::new();
    let next: Vec<[u32; 1]> = (0..MAX_SEQS).map(|s| [(s * 97 + 13) % vocab]).collect();
    for batch in DECODE_BATCHES {
        let seqs: Vec<(&[BlockId], &[u32], u32)> = (0..batch as usize)
            .map(|s| (tables[s].as_slice(), next[s].as_slice(), DECODE_CTX - 1))
            .collect();
        cases.push(profile_case(
            &format!("decode_b{batch}_ctx{DECODE_CTX}"),
            exec.as_mut(),
            |e| step(e, &kv, &seqs).expect("decode step"),
        ));
    }
    let table: Vec<BlockId> = (0..per_prefill)
        .map(|b| BlockId(MAX_SEQS * per_decode + b))
        .collect();
    let prompt: Vec<u32> = (0..PREFILL).map(|i| (i * 31 + 7) % vocab).collect();
    cases.push(profile_case(
        &format!("prefill_{PREFILL}"),
        exec.as_mut(),
        |e| step(e, &kv, &[(&table, &prompt, 0)]).expect("prefill step"),
    ));

    // Every op kind the executor's requirements name, per case: the decode cases run no
    // prefill attention and the prefill case no decode attention; no case forks blocks.
    let kinds: Vec<OpKind> = hip_requirements(&cfg, ctx, opts)
        .iter()
        .map(|r| r.op)
        .filter(|op| *op != OpKind::CopyBlocks)
        .collect();
    for case in &cases {
        let decode = case.case.starts_with("decode");
        for kind in &kinds {
            let unused = if decode {
                OpKind::AttentionPrefillPaged
            } else {
                OpKind::AttentionDecodePaged
            };
            if *kind == unused {
                continue;
            }
            assert!(
                case.ops.iter().any(|e| e.op == kind.as_str()),
                "{} {}: no {kind} in the profile",
                dir.display(),
                case.case
            );
        }
        assert!(
            case.profiled_ms >= 0.8 * case.forward_ms,
            "{} {}: the profile accounts for {:.3} ms of a {:.3} ms forward",
            dir.display(),
            case.case,
            case.profiled_ms,
            case.forward_ms
        );
    }
    let model = dir
        .file_name()
        .map_or_else(|| dir.display().to_string(), |n| n.to_string_lossy().into());
    ModelReport {
        model,
        block_tokens: BLOCK_TOKENS,
        fused_ops: opts.fused_ops,
        fused_projections: opts.fused_projections,
        cases,
    }
}

/// Lab only: the op-level profile of both real models on the R9700 (see the module docs).
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY, TURBINE_TEST_MODEL_DIR and TURBINE_TEST_MOE_MODEL_DIR"]
fn forward_profile() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let dirs = [
        require_env_dir("TURBINE_TEST_MODEL_DIR"),
        require_env_dir("TURBINE_TEST_MOE_MODEL_DIR"),
    ];
    let ctx = turbine_kernels::test_support::open_context("hip");
    // Llama also with the opt-in projection fusion, for the side-by-side profile.
    let all_fused = ExecutorOptions {
        fused_ops: true,
        fused_projections: true,
    };
    let runs = [
        (&dirs[0], ExecutorOptions::default()),
        (&dirs[0], all_fused),
        (&dirs[1], ExecutorOptions::default()),
    ];
    for (dir, opts) in runs {
        let report = profile_model(dir, &ctx, opts);
        for case in &report.cases {
            let moe: f64 = case
                .ops
                .iter()
                .filter(|e| e.op.starts_with("moe_"))
                .map(|e| e.total_ms)
                .sum();
            if moe > 0.0 {
                println!(
                    "forward_profile_moe: model={} case={} moe_path_ms={moe:.3} ({:.0}% of the profiled {:.3} ms)",
                    report.model,
                    case.case,
                    100.0 * moe / case.profiled_ms,
                    case.profiled_ms
                );
            }
        }
        println!(
            "forward_profile: {}",
            serde_json::to_string(&report).expect("serialize the profile")
        );
    }
}

/// Lab diagnostic (P2c Task 10, no assertion on speed): Llama-3.2-3B under continuous
/// batching shaped like the baseline workload — 16 sequences decoding; whenever one finishes its
/// 256 tokens a new 600–760-token prompt is prefilled in the same forward as the others' decode
/// rows; 2 ms of host time between forwards; `max_batch_tokens` 8,192 — for every combination of
/// `fused_ops` and `fused_projections`, each executor run twice in mirrored order (so a drift in
/// GPU clocks cancels out). Prints one `serving_mix:` line per run: the mean and median
/// decode-only forward time, its launch and logits-wait parts, and the mean mixed-step time.
/// Run with `scripts/lab-test.sh novanas -- --release -p turbine-model --test perf -- serving_mix --nocapture`.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MODEL_DIR"]
fn serving_mix() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let dir = require_env_dir("TURBINE_TEST_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");
    let cfg = load_model_config(&dir).expect("config.json");
    const RUNNING: u32 = 16;
    const OUTPUT: u32 = 256;
    const SLOTS: u32 = 48;
    let per_seq = (800 + OUTPUT).div_ceil(BLOCK_TOKENS);
    let o = |fused_ops, fused_projections| ExecutorOptions {
        fused_ops,
        fused_projections,
    };
    let (none, proj, norm, both) = (
        o(false, false),
        o(false, true),
        o(true, false),
        o(true, true),
    );
    for opts in [none, proj, norm, both, both, norm, proj, none] {
        let mut exec = hip_executor(&cfg, &dir, &ctx, opts, 8192);
        let layout = *exec.kv_layout();
        let blocks = SLOTS * per_seq;
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let storage =
            DeviceBuffer::alloc(&mem, (layout.block_bytes() * u64::from(blocks)) as usize)
                .expect("KV pool");
        let kv = pool_view(&storage, &layout, blocks);
        let vocab = cfg.vocab_size;
        let tables: Vec<Vec<BlockId>> = (0..SLOTS)
            .map(|s| (0..per_seq).map(|b| BlockId(s * per_seq + b)).collect())
            .collect();
        // (slot, prompt, context so far, tokens left); a new sequence has an empty context.
        let new_seq = |n: u32, left: u32| {
            let slot = (n % SLOTS) as usize;
            let len = 600 + (n * 37) % 161;
            let prompt: Vec<u32> = (0..len)
                .map(|i| (i * 7919 + n * 104_729 + 1000) % vocab)
                .collect();
            (slot, prompt, 0u32, left)
        };
        let mut next = RUNNING;
        let mut running: Vec<(usize, Vec<u32>, u32, u32)> = (0..RUNNING)
            .map(|i| new_seq(i, OUTPUT - i * (OUTPUT / RUNNING)))
            .collect();
        let mut decode_ms = Vec::new();
        let mut launch_ms = Vec::new();
        let mut wait_ms = Vec::new();
        let mut prefill_ms = Vec::new();
        for _ in 0..1200 {
            let tok: Vec<[u32; 1]> = running.iter().map(|r| [(r.2 * 31 + 7) % vocab]).collect();
            let mut seqs: Vec<(&[BlockId], &[u32], u32)> = Vec::new();
            let mut prefill = None;
            for (i, r) in running.iter().enumerate() {
                if r.2 == 0 {
                    if prefill.is_none() {
                        seqs.push((tables[r.0].as_slice(), r.1.as_slice(), 0));
                        prefill = Some(i);
                    }
                } else {
                    seqs.push((tables[r.0].as_slice(), tok[i].as_slice(), r.2));
                }
            }
            let start = Instant::now();
            step(exec.as_mut(), &kv, &seqs).expect("step");
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if prefill.is_some() {
                prefill_ms.push(ms);
            } else {
                decode_ms.push(ms);
                let t = exec.last_timings();
                launch_ms.push(t.launch.as_secs_f64() * 1e3);
                wait_ms.push(t.device_wait.as_secs_f64() * 1e3);
            }
            // The server's host work between forwards (sampling, streaming).
            std::thread::sleep(std::time::Duration::from_millis(2));
            for (i, r) in running.iter_mut().enumerate() {
                if r.2 == 0 {
                    if prefill == Some(i) {
                        r.2 = r.1.len() as u32;
                    }
                } else {
                    r.2 += 1;
                    r.3 -= 1;
                }
            }
            for r in running.iter_mut() {
                if r.3 == 0 {
                    *r = new_seq(next, OUTPUT);
                    next += 1;
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let mut sorted = decode_ms.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "serving_mix: fused_ops={} fused_projections={} decode_steps={} decode_mean_ms={:.3} decode_median_ms={:.3} launch_mean_ms={:.3} logits_wait_mean_ms={:.3} prefill_steps={} prefill_mean_ms={:.3}",
            opts.fused_ops,
            opts.fused_projections,
            decode_ms.len(),
            mean(&decode_ms),
            sorted[sorted.len() / 2],
            mean(&launch_ms),
            mean(&wait_ms),
            prefill_ms.len(),
            mean(&prefill_ms)
        );
    }
}

// ------------------------------------------------------------------ MoE prefill (perf item #2)

/// Tokens (× top-8 = routed rows) `moe_prefill_timings` times: 512, 2,048 and 16,384 routed
/// rows — the first grouped-tier size, a mid-sized chunk and a full 2,048-token chunk.
const MOE_PREFILL_TOKENS: [usize; 3] = [64, 256, 2048];

/// The prompt token ids of the committed OLMoE golden reference (real chat-formatted text),
/// concatenated and cut to `n`.
fn golden_prompt_tokens(n: usize) -> Vec<u32> {
    #[derive(serde::Deserialize)]
    struct Record {
        prompt_token_ids: Vec<u32>,
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl");
    let text = std::fs::read_to_string(&path).expect("OLMoE reference.jsonl");
    let mut tokens: Vec<u32> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .flat_map(|l| {
            serde_json::from_str::<Record>(l)
                .expect("reference line")
                .prompt_token_ids
        })
        .collect();
    assert!(
        tokens.len() >= n,
        "the golden prompts hold {} tokens",
        tokens.len()
    );
    tokens.truncate(n);
    tokens
}

/// The raw BF16 bytes of layer `layer`'s expert projection `proj`, stacked `[experts, rows,
/// cols]` as the loader stacks them.
fn stacked_expert_bytes(
    index: &SafetensorsIndex,
    layer: usize,
    proj: &str,
    experts: usize,
) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut out = Vec::new();
    for e in 0..experts {
        let name = format!("model.layers.{layer}.mlp.experts.{e}.{proj}.weight");
        let entry = index
            .get(&name)
            .unwrap_or_else(|| panic!("checkpoint lacks {name}"));
        let mut f = std::fs::File::open(&entry.file).expect("open shard");
        f.seek(SeekFrom::Start(entry.range.start)).expect("seek");
        let start = out.len();
        out.resize(start + entry.byte_len() as usize, 0);
        f.read_exact(&mut out[start..]).expect("read expert weight");
    }
    out
}

/// Lab timing (perf tier, perf item #2): `moe_experts` at the served prefill shapes with **real
/// routing**. One 2,048-token prefill of OLMoE-1B-7B over the golden prompts (real text) is
/// traced; for the layers of `TURBINE_MOE_LAYERS` (default `0,5,10,15`) its `mlp_norm` rows (the
/// MoE input) and `router_logits` are routed again by `moe_route`, and every enumerated
/// `moe_experts` implementation that supports the shape runs alone on the layer's real expert
/// weights over the first 64 / 256 / 2,048 tokens (512 / 2,048 / 16,384 routed rows: what a
/// prefill of those tokens routes). One `moe_prefill_timing` line per (layer, size,
/// implementation): mean time per call, TFLOPS, the routing skew (active experts, the largest
/// expert's rows) and how many outputs differ from `turbine_hip_moe_wmma` (0: the same WMMA
/// chain, so compatible with the small-m tier's batch invariance). No bound. Run natively on
/// GPU 0 under `scripts/bench-lock.sh`; the expert weights (805 MB per layer) stream from memory.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MOE_MODEL_DIR"]
fn moe_prefill_timings() {
    use turbine_core::types::DType;
    use turbine_kernels::{
        ImplChoice, KernelProvider, MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig,
        MoeRouteContext, OpConfig, OpRequirement,
    };
    use turbine_tensor::{Tensor, TensorView};

    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let dir = require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let cfg = load_model_config(&dir).expect("config.json");
    let moe = cfg.moe.expect("an MoE model");
    let (experts, top_k, inter, hidden) = (
        moe.num_experts as usize,
        moe.experts_per_token as usize,
        moe.expert_intermediate as usize,
        cfg.hidden as usize,
    );
    let layers: Vec<usize> = std::env::var("TURBINE_MOE_LAYERS")
        .unwrap_or_else(|_| "0,5,10,15".into())
        .split(',')
        .map(|s| s.trim().parse().expect("TURBINE_MOE_LAYERS: layer numbers"))
        .collect();
    let max_t = *MOE_PREFILL_TOKENS.iter().max().expect("sizes");

    // 1. The real MoE inputs and router logits of one 2,048-token prefill.
    let (norms, logits) = {
        let mut exec = hip_decoder(&cfg, &dir, &ctx, ExecutorOptions::default(), max_t as u32);
        let layout = *exec.kv_layout();
        let blocks = (max_t as u32).div_ceil(BLOCK_TOKENS);
        let storage =
            DeviceBuffer::alloc(&mem, (layout.block_bytes() * u64::from(blocks)) as usize)
                .expect("KV pool");
        let kv = pool_view(&storage, &layout, blocks);
        let table: Vec<BlockId> = (0..blocks).map(BlockId).collect();
        let tokens = golden_prompt_tokens(max_t);
        exec.set_trace(true);
        step(&mut exec, &kv, &[(&table, &tokens, 0)]).expect("traced prefill");
        let trace = exec.take_trace();
        let pick = |name: &str, layer: usize| {
            trace
                .iter()
                .find(|t| t.layer == Some(layer) && t.name == name)
                .unwrap_or_else(|| panic!("trace lacks layer {layer} {name}"))
                .data
                .clone()
        };
        let norms: Vec<Vec<f32>> = layers.iter().map(|&l| pick("mlp_norm", l)).collect();
        let logits: Vec<Vec<f32>> = layers.iter().map(|&l| pick("router_logits", l)).collect();
        (norms, logits)
    };

    // 2. Every moe_experts implementation, bound alone.
    let provider = shim_provider(ctx.clone());
    let route_cfg = MoeRouteConfig {
        num_experts: experts as u32,
        top_k: top_k as u32,
        renormalize: moe.norm_topk_prob,
        bf16_logits: true,
    };
    let experts_cfg = MoeExpertsConfig {
        hidden: hidden as u32,
        inter: inter as u32,
        num_experts: experts as u32,
        top_k: top_k as u32,
        expert_begin: 0,
        expert_end: experts as u32,
        dtype: DType::BF16,
    };
    let registry = KernelRegistry::build(
        vec![Arc::clone(&provider)],
        &[provider.id()],
        &[OpRequirement::from(OpConfig::MoeRoute(route_cfg))],
        &KernelMetrics::register(&MetricsRegistry::new()),
        provider.card_profile(),
    )
    .expect("moe_route");
    let router = registry.moe_route(&route_cfg);
    let spec = OpConfig::MoeExperts(experts_cfg);
    let reference = "turbine_hip_moe_wmma";
    let mut impls: Vec<(String, Arc<dyn KernelProvider>)> = provider
        .implementations(OpKind::MoeExperts)
        .into_iter()
        .filter_map(|i| Some((i.name, provider.bind(&spec, &ImplChoice::Single(i.index))?)))
        .collect();
    // The reference first: every other output is compared with it.
    impls.sort_by_key(|(n, _)| n != reference);
    assert_eq!(impls[0].0, reference, "the library enumerates {reference}");

    let upload = |shape: &[usize], dtype: DType, bytes: &[u8]| {
        let mut t = Tensor::empty(&mem, shape, dtype).expect("tensor");
        t.storage.copy_from_host(0, bytes).expect("upload");
        t
    };
    let index = SafetensorsIndex::open(&dir).expect("open safetensors");
    for (li, &layer) in layers.iter().enumerate() {
        let gate_shape = [experts, inter, hidden];
        let w_gate = upload(
            &gate_shape,
            DType::BF16,
            &stacked_expert_bytes(&index, layer, "gate_proj", experts),
        );
        let w_up = upload(
            &gate_shape,
            DType::BF16,
            &stacked_expert_bytes(&index, layer, "up_proj", experts),
        );
        let w_down = upload(
            &[experts, hidden, inter],
            DType::BF16,
            &stacked_expert_bytes(&index, layer, "down_proj", experts),
        );
        for t in MOE_PREFILL_TOKENS {
            let rows = t * top_k;
            let x_bytes: Vec<u8> = norms[li][..t * hidden]
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            let x = upload(&[t, hidden], DType::BF16, &x_bytes);
            let l_bytes: Vec<u8> = logits[li][..t * experts]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let l = upload(&[t, experts], DType::F32, &l_bytes);
            let ids = Tensor::empty(&mem, &[t, top_k], DType::I32).expect("ids");
            let tw = Tensor::empty(&mem, &[t, top_k], DType::F32).expect("weights");
            let sorted = Tensor::empty(&mem, &[rows], DType::I32).expect("sorted");
            let offsets = Tensor::empty(&mem, &[experts + 1], DType::I32).expect("offsets");
            router
                .route(&mut MoeRouteContext {
                    cfg: route_cfg,
                    router_logits: l.view(),
                    topk_ids: ids.view(),
                    topk_weights: tw.view(),
                    sorted_rows: sorted.view(),
                    expert_offsets: offsets.view(),
                })
                .expect("moe_route");
            let host: Vec<i32> = offsets
                .storage
                .whole()
                .read_bytes()
                .expect("offsets")
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let per_expert: Vec<i32> = host.windows(2).map(|w| w[1] - w[0]).collect();
            let active = per_expert.iter().filter(|&&c| c > 0).count();
            let largest = per_expert.iter().copied().max().unwrap_or(0);
            // The routed rows per expert, for offline kernel work on the same distribution.
            println!("moe_prefill_routing layer={layer} tokens={t} rows_per_expert={per_expert:?}");
            let zeros = vec![0u8; t * hidden * 2];
            let mut o = upload(&[t, hidden], DType::BF16, &zeros);
            let sorted_view =
                TensorView::contiguous(sorted.storage.whole(), 0, &[rows], DType::I32);
            let mut want: Option<Vec<u8>> = None;
            for (name, p) in &impls {
                let kernel = p.moe().expect("moe");
                if !kernel.supports_experts(&experts_cfg) {
                    continue;
                }
                // One run into a zeroed output, compared with the reference implementation's.
                o.storage.copy_from_host(0, &zeros).expect("zero out");
                let run = || {
                    kernel.experts(&mut MoeExpertsContext {
                        cfg: experts_cfg,
                        x: x.view(),
                        w_gate: w_gate.view(),
                        w_up: w_up.view(),
                        w_down: w_down.view(),
                        sorted_rows: sorted_view.clone(),
                        expert_offsets: offsets.view(),
                        topk_weights: tw.view(),
                        host_expert_offsets: &host,
                        out: o.view(),
                        workspace: None,
                    })
                };
                // A library may enumerate an implementation it cannot run on this device
                // (hipblaslt_grouped without a grouped-GEMM solution): reported, not timed.
                if let Err(e) = run() {
                    println!(
                        "moe_prefill_timing layer={layer} tokens={t} rows={rows} impl={name} \
                         unsupported: {e}"
                    );
                    continue;
                }
                let run = || run().expect("moe_experts");
                let got = o.storage.whole().read_bytes().expect("read out");
                let differing = match &want {
                    None => {
                        want = Some(got);
                        0
                    }
                    Some(w) => w
                        .chunks_exact(2)
                        .zip(got.chunks_exact(2))
                        .filter(|(a, b)| a != b)
                        .count(),
                };
                run();
                mem.synchronize().expect("synchronize");
                let iters: u32 = if t >= 1024 { 20 } else { 50 };
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                mem.synchronize().expect("synchronize");
                let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
                let tflops = 6.0 * (rows * hidden * inter) as f64 / (us * 1e6);
                println!(
                    "moe_prefill_timing layer={layer} tokens={t} rows={rows} \
                     active_experts={active} largest_expert_rows={largest} impl={name} \
                     us={us:.1} tflops={tflops:.1} differing_vs_{reference}={differing}/{}",
                    t * hidden
                );
            }
        }
    }
}
