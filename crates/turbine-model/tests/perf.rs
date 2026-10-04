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
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Barrier, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;
use turbine_core::clock::SystemClock;
use turbine_core::types::{BlockId, KvLayout, SeqId};
use turbine_distributed::collective::{
    self, Collective, CollectiveError, CollectiveInit, CollectiveMetrics, CollectiveOp, ReduceOp,
    hostmem,
};
use turbine_kernels::{
    KernelMetrics, KernelRegistry, OpKind, shim_provider, test_support::require_backend,
    test_support::require_env_dir,
};
use turbine_model::config::ModelArchConfig;
use turbine_model::executor::{
    self, BatchInput, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, OpProfile,
    OpProfileEntry, RowReduce, SeqSlice,
};
use turbine_model::families;
use turbine_model::tp;
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, load_model_config};
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DType, DeviceBuffer, DeviceMemory, DeviceSlice, KvPoolView, StreamRef};

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
        classes: None,
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
            block_formats: &[],
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

/// The MoE inputs (`mlp_norm` rows) and router logits of layers `layers` of one traced
/// `tokens`-token OLMoE prefill over the golden prompts (real text), on `ctx`.
fn traced_moe_inputs(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
    layers: &[usize],
    tokens: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut exec = hip_decoder(cfg, dir, ctx, ExecutorOptions::default(), tokens as u32);
    let layout = *exec.kv_layout();
    let blocks = (tokens as u32).div_ceil(BLOCK_TOKENS);
    let storage = DeviceBuffer::alloc(&mem, (layout.block_bytes() * u64::from(blocks)) as usize)
        .expect("KV pool");
    let kv = pool_view(&storage, &layout, blocks);
    let table: Vec<BlockId> = (0..blocks).map(BlockId).collect();
    let prompt = golden_prompt_tokens(tokens);
    exec.set_trace(true);
    step(&mut exec, &kv, &[(&table, &prompt, 0)]).expect("traced prefill");
    let trace = exec.take_trace();
    let pick = |name: &str, layer: usize| {
        trace
            .iter()
            .find(|t| t.layer == Some(layer) && t.name == name)
            .unwrap_or_else(|| panic!("trace lacks layer {layer} {name}"))
            .data
            .clone()
    };
    let norms = layers.iter().map(|&l| pick("mlp_norm", l)).collect();
    let logits = layers.iter().map(|&l| pick("router_logits", l)).collect();
    (norms, logits)
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
    let (norms, logits) = traced_moe_inputs(&cfg, &dir, &ctx, &layers, max_t);

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

/// Tokens `moe_ep_local_timings` times: a 16-sequence decode step (128 routed rows, the small-m
/// tier) and one 2,048-token prefill chunk (16,384 routed rows, the prefill tier).
const MOE_EP_TOKENS: [usize; 2] = [16, 2048];

/// Lab timing (P5 Task 25, the EP provider evaluation): an expert-parallel rank of OLMoE-1B-7B
/// runs `moe_experts` over only its experts through the local expert range
/// `[expert_begin, expert_end)` every Phase 2 provider already takes (kernel ABI v2: the rows
/// of remote experts lie outside the range's part of the sorted list, so they are neither
/// gathered, multiplied nor scattered). With OLMoE's real routing (the traced 2,048-token
/// golden prefill; its first 16 tokens stand for a 16-sequence decode step) and real expert
/// weights of layers `TURBINE_MOE_LAYERS` (default `0,5,10,15`), the library's default choice
/// at each shape is timed for: all 64 experts (one device, or an EP rank giving remote experts
/// weight 0), rank 0 and rank 1 of contiguous EP 2 (experts 0–31, 32–63; weights `[32, …]`), and
/// rank 0 of an interleaved placement (the even experts: 32 one-expert runs, what a placement
/// file without contiguous runs costs). Checks, bitwise: rank 0's range then rank 1's range into
/// one accumulator equals the 64-expert call (the scatter adds each token's experts in
/// ascending order either way), and so do the 64 one-expert runs in ascending order. One
/// `moe_ep_timing` line per (layer, tokens, case). Run natively on GPU 0 under
/// `scripts/bench-lock.sh`.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MOE_MODEL_DIR"]
fn moe_ep_local_timings() {
    use turbine_core::types::DType;
    use turbine_kernels::{
        MoeExpertsConfig, MoeExpertsContext, MoeRouteConfig, MoeRouteContext, OpConfig,
        OpRequirement,
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
    let mid = experts / 2;
    let layers: Vec<usize> = std::env::var("TURBINE_MOE_LAYERS")
        .unwrap_or_else(|_| "0,5,10,15".into())
        .split(',')
        .map(|s| s.trim().parse().expect("TURBINE_MOE_LAYERS: layer numbers"))
        .collect();
    let max_t = *MOE_EP_TOKENS.iter().max().expect("sizes");
    let (norms, logits) = traced_moe_inputs(&cfg, &dir, &ctx, &layers, max_t);

    let range = |begin: usize, end: usize| MoeExpertsConfig {
        hidden: hidden as u32,
        inter: inter as u32,
        num_experts: experts as u32,
        top_k: top_k as u32,
        expert_begin: begin as u32,
        expert_end: end as u32,
        dtype: DType::BF16,
    };
    let route_cfg = MoeRouteConfig {
        num_experts: experts as u32,
        top_k: top_k as u32,
        renormalize: moe.norm_topk_prob,
        bf16_logits: true,
    };
    // The library's default choice (its card profile's row tiers) for every range used.
    let mut reqs = vec![
        OpRequirement::from(OpConfig::MoeRoute(route_cfg)),
        OpRequirement::from(OpConfig::MoeExperts(range(0, experts))),
        OpRequirement::from(OpConfig::MoeExperts(range(0, mid))),
        OpRequirement::from(OpConfig::MoeExperts(range(mid, experts))),
    ];
    reqs.extend((0..experts).map(|e| OpRequirement::from(OpConfig::MoeExperts(range(e, e + 1)))));
    let provider = shim_provider(ctx.clone());
    let registry = KernelRegistry::build(
        vec![Arc::clone(&provider)],
        &[provider.id()],
        &reqs,
        &KernelMetrics::register(&MetricsRegistry::new()),
        provider.card_profile(),
    )
    .expect("the library serves every range");
    let router = registry.moe_route(&route_cfg);

    let upload = |shape: &[usize], dtype: DType, bytes: &[u8]| {
        let mut t = Tensor::empty(&mem, shape, dtype).expect("tensor");
        t.storage.copy_from_host(0, bytes).expect("upload");
        t
    };
    let index = SafetensorsIndex::open(&dir).expect("open safetensors");
    for (li, &layer) in layers.iter().enumerate() {
        let gate_shape = [experts, inter, hidden];
        let stack = |proj: &str| stacked_expert_bytes(&index, layer, proj, experts);
        let w_gate = upload(&gate_shape, DType::BF16, &stack("gate_proj"));
        let w_up = upload(&gate_shape, DType::BF16, &stack("up_proj"));
        let w_down = upload(&[experts, hidden, inter], DType::BF16, &stack("down_proj"));
        for t in MOE_EP_TOKENS {
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
            let rows_in = |b: usize, e: usize| host[e] - host[b];
            let zeros = vec![0u8; t * hidden * 2];
            let o = upload(&[t, hidden], DType::BF16, &zeros);
            let sorted_view =
                TensorView::contiguous(sorted.storage.whole(), 0, &[rows], DType::I32);
            // One call over the experts `[b, e)`, accumulating into `o`.
            let call = |b: usize, e: usize| {
                let cfg = range(b, e);
                registry
                    .moe_experts(&cfg)
                    .experts(&mut MoeExpertsContext {
                        cfg,
                        x: x.view(),
                        w_gate: w_gate.view().rows(b, e - b),
                        w_up: w_up.view().rows(b, e - b),
                        w_down: w_down.view().rows(b, e - b),
                        sorted_rows: sorted_view.clone(),
                        expert_offsets: offsets.view(),
                        topk_weights: tw.view(),
                        host_expert_offsets: &host,
                        out: o.view(),
                        workspace: None,
                    })
                    .expect("moe_experts");
            };
            let zero = || o.storage.whole().write_bytes(&zeros).expect("zero out");
            let read = || o.storage.whole().read_bytes().expect("read out");
            let differing = |a: &[u8], b: &[u8]| {
                a.chunks_exact(2)
                    .zip(b.chunks_exact(2))
                    .filter(|(x, y)| x != y)
                    .count()
            };
            // Bitwise: the split ranges and the one-expert runs reproduce the 64-expert call.
            zero();
            call(0, experts);
            let full = read();
            zero();
            call(0, mid);
            call(mid, experts);
            let split = differing(&full, &read());
            zero();
            (0..experts).for_each(|e| call(e, e + 1));
            let singles = differing(&full, &read());
            assert_eq!(
                (split, singles),
                (0, 0),
                "layer {layer} tokens {t}: split ranges / one-expert runs differ from the \
                 64-expert call"
            );
            let even: Vec<usize> = (0..experts).step_by(2).collect();
            let run_even = || even.iter().for_each(|&e| call(e, e + 1));
            let cases: [(&str, &dyn Fn(), i32); 4] = [
                ("all_64", &|| call(0, experts), rows_in(0, experts)),
                ("ep2_rank0_0..32", &|| call(0, mid), rows_in(0, mid)),
                (
                    "ep2_rank1_32..64",
                    &|| call(mid, experts),
                    rows_in(mid, experts),
                ),
                (
                    "ep2_rank0_even_32_runs",
                    &run_even,
                    even.iter().map(|&e| rows_in(e, e + 1)).sum(),
                ),
            ];
            let imp = registry
                .moe_experts(&range(0, experts))
                .implementation_experts(&range(0, experts));
            for (case, run, local_rows) in cases {
                run();
                mem.synchronize().expect("synchronize");
                let iters: u32 = if t >= 1024 { 20 } else { 100 };
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                mem.synchronize().expect("synchronize");
                let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
                println!(
                    "moe_ep_timing layer={layer} tokens={t} rows={rows} case={case} \
                     local_rows={local_rows} us={us:.1} impl={imp}"
                );
            }
        }
    }
}

// ---- Tensor-parallel step profile (P5 Task 32) ----

/// Sequences of the tensor-parallel decode case (the c16 serving workload).
const TP_DECODE_SEQS: u32 = 16;
/// Forwards pass through to the real communicator.
const COLL_PASS: u8 = 0;
/// Every call is timed: the rank's stream is drained, the call enqueued and the stream drained
/// again; the host instants of arrival and completion are recorded (both ranks are threads of
/// this process, so the instants compare across ranks).
const COLL_TIME: u8 = 1;
/// Every call is skipped (both ranks skip the same calls; the numbers are wrong, the timing of
/// everything else is not): the forward without collectives.
const COLL_SKIP_ALL: u8 = 2;
/// The calls `hostmem` routes to RCCL are skipped: the forward with free large collectives.
const COLL_SKIP_RCCL: u8 = 3;

/// One timed collective call.
#[derive(Clone, Copy, Debug)]
struct CollCall {
    op: CollectiveOp,
    /// nccl-tests bytes (the routing size).
    bytes: usize,
    route: &'static str,
    arrive: Instant,
    end: Instant,
}

/// A rank's communicator as the profile sees it: the real one (`hostmem` with its RCCL delegate)
/// behind a switch ([`COLL_PASS`] …). Test-only, so the serving path pays nothing for it.
struct TimedCollective {
    inner: Arc<dyn Collective>,
    mem: Arc<dyn DeviceMemory>,
    mode: AtomicU8,
    calls: Mutex<Vec<CollCall>>,
}

impl TimedCollective {
    fn new(inner: Arc<dyn Collective>, mem: Arc<dyn DeviceMemory>) -> Arc<TimedCollective> {
        Arc::new(TimedCollective {
            inner,
            mem,
            mode: AtomicU8::new(COLL_PASS),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn set_mode(&self, mode: u8) {
        self.mode.store(mode, Ordering::Release);
    }

    fn take_calls(&self) -> Vec<CollCall> {
        std::mem::take(&mut *self.calls.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Where `hostmem` sends `op` over `bytes` at `parallel.collective.hostmem_max_bytes: auto`
    /// (checked against `turbine_collective_route_total` after the run).
    fn route(&self, op: CollectiveOp, bytes: usize) -> &'static str {
        let backend = self.inner.backend();
        if backend == "hostmem" && bytes as u64 > hostmem::auto_max_bytes(op) {
            "rccl"
        } else {
            backend
        }
    }

    fn call(
        &self,
        op: CollectiveOp,
        bytes: usize,
        f: impl FnOnce() -> Result<(), CollectiveError>,
    ) -> Result<(), CollectiveError> {
        let route = self.route(op, bytes);
        match self.mode.load(Ordering::Acquire) {
            COLL_SKIP_ALL => Ok(()),
            COLL_SKIP_RCCL if route == "rccl" => Ok(()),
            COLL_TIME => {
                let drain = || {
                    self.mem
                        .synchronize()
                        .map_err(|e| CollectiveError::Backend {
                            code: -1,
                            message: e.to_string(),
                        })
                };
                drain()?;
                let arrive = Instant::now();
                f()?;
                drain()?;
                let end = Instant::now();
                self.calls
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(CollCall {
                        op,
                        bytes,
                        route,
                        arrive,
                        end,
                    });
                Ok(())
            }
            _ => f(),
        }
    }
}

impl Collective for TimedCollective {
    fn backend(&self) -> &'static str {
        self.inner.backend()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_reduce(
        &self,
        buf: &mut DeviceSlice<'_>,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let n = buf.len();
        self.call(CollectiveOp::AllReduce, n, || {
            self.inner.all_reduce(buf, dtype, op, stream)
        })
    }
    fn all_gather(
        &self,
        send: &DeviceSlice<'_>,
        recv: &mut DeviceSlice<'_>,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let n = recv.len();
        self.call(CollectiveOp::AllGather, n, || {
            self.inner.all_gather(send, recv, stream)
        })
    }
    fn reduce_scatter(
        &self,
        send: &DeviceSlice<'_>,
        recv: &mut DeviceSlice<'_>,
        dtype: DType,
        op: ReduceOp,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let n = send.len();
        self.call(CollectiveOp::ReduceScatter, n, || {
            self.inner.reduce_scatter(send, recv, dtype, op, stream)
        })
    }
    fn broadcast(
        &self,
        buf: &mut DeviceSlice<'_>,
        root: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        let n = buf.len();
        self.call(CollectiveOp::Broadcast, n, || {
            self.inner.broadcast(buf, root, stream)
        })
    }
    fn barrier(&self, stream: &StreamRef) -> Result<(), CollectiveError> {
        self.inner.barrier(stream)
    }
    fn send(
        &self,
        buf: &DeviceSlice<'_>,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.send(buf, peer, stream)
    }
    fn recv(
        &self,
        buf: &mut DeviceSlice<'_>,
        peer: usize,
        stream: &StreamRef,
    ) -> Result<(), CollectiveError> {
        self.inner.recv(buf, peer, stream)
    }
    fn step_begin(&self) {
        self.inner.step_begin();
    }
    fn step_end(&self) -> Result<(), CollectiveError> {
        self.inner.step_end()
    }
    fn abort(&self) {
        self.inner.abort();
    }
}

/// Aborts the test process when the guard lives longer than `limit`: a rank stuck in a
/// collective would otherwise hold both GPUs of the lab Job.
struct TpWatchdog(Arc<AtomicBool>);

fn tp_watchdog(name: &'static str, limit: Duration) -> TpWatchdog {
    let done = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&done);
    std::thread::spawn(move || {
        let started = Instant::now();
        while !seen.load(Ordering::Acquire) {
            if started.elapsed() > limit {
                eprintln!("perf: {name} ran longer than {limit:?}; aborting the process");
                std::process::abort();
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });
    TpWatchdog(done)
}

impl Drop for TpWatchdog {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// The kernel-library context of every AMD device, in index order.
fn hip_contexts() -> Vec<Arc<turbine_kernels::ShimContext>> {
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from);
    let backend = turbine_kernels::backends::registry()
        .get("hip")
        .expect("the hip execution backend");
    inventory
        .devices
        .iter()
        .filter(|d| d.vendor.as_str() == backend.vendor())
        .map(|d| {
            backend
                .open(&turbine_kernels::backends::BackendRequest {
                    device: d.index,
                    kernel_library: library.as_deref(),
                    inventory: &inventory,
                    meminfo: Path::new("/proc/meminfo"),
                    card_profile: "auto",
                })
                .unwrap_or_else(|e| panic!("open device {}: {e}", d.index.0))
                .context
                .expect("a kernel-library context")
        })
        .collect()
}

/// Rank `s` of `cfg` on `ctx` over `collective` (the [`turbine_model::tp`] shard, the family's
/// tensor-parallel hooks), for batches of up to [`PREFILL`] tokens and [`TP_DECODE_SEQS`]
/// sequences, as the server builds it (no decode graphs, no overlapped launches under TP).
fn tp_rank_decoder(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
    s: tp::ShardSpec,
    collective: Arc<dyn Collective>,
) -> DecoderExecutor {
    let opts = ExecutorOptions::default();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let slots = tp::weight_slots(cfg, s).expect("shard slots");
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let order = [provider.id()];
    let reqs =
        tp::available_requirements(cfg, s, BLOCK_TOKENS, opts, std::slice::from_ref(&provider))
            .expect("rank requirements");
    let card = provider.card_profile();
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = Arc::new(
        KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider"),
    );
    let spec = cfg
        .family
        .0
        .tp_decoder_spec()
        .expect("tensor-parallel hooks");
    let limits = ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens: PREFILL,
        max_seqs: TP_DECODE_SEQS,
    };
    let stream = mem.compute_stream();
    DecoderExecutor::new_tp(
        cfg,
        spec,
        weights,
        registry,
        mem,
        limits,
        opts,
        Some(tp::TpContext {
            rank: s.rank,
            world: s.world,
            collective,
            stream,
        }),
    )
    .expect("rank executor")
}

/// One rank's measurements of one case.
struct RankCase {
    case: String,
    /// Median of [`RUNS`] unprofiled forwards (pipelined, as served), and of their step stages.
    forward_ms: f64,
    launch_ms: f64,
    device_wait_ms: f64,
    /// The same with every collective skipped, and with the RCCL-routed ones skipped.
    no_collectives_ms: f64,
    no_rccl_ms: f64,
    profile: OpProfile,
    calls: Vec<CollCall>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Measures one case on one rank; `barrier` (both ranks) precedes every forward, so the ranks
/// start each one together.
fn tp_case(
    name: &str,
    exec: &mut DecoderExecutor,
    coll: Option<&TimedCollective>,
    barrier: Option<&Barrier>,
    run: &mut dyn FnMut(&mut DecoderExecutor),
) -> RankCase {
    let sync = || {
        if let Some(b) = barrier {
            b.wait();
        }
    };
    let set = |mode| {
        if let Some(c) = coll {
            c.set_mode(mode);
        }
    };
    let mut pipelined = |mode: u8, exec: &mut DecoderExecutor| -> (f64, f64, f64) {
        set(mode);
        sync();
        run(exec);
        let (mut total, mut launch, mut wait) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..RUNS {
            sync();
            let start = Instant::now();
            run(exec);
            total.push(start.elapsed().as_secs_f64() * 1e3);
            let t = ModelExecutor::last_timings(exec);
            launch.push(t.launch.as_secs_f64() * 1e3);
            wait.push(t.device_wait.as_secs_f64() * 1e3);
        }
        set(COLL_PASS);
        (median(total), median(launch), median(wait))
    };
    let (forward_ms, launch_ms, device_wait_ms) = pipelined(COLL_PASS, exec);
    let (no_collectives_ms, _, _) = pipelined(COLL_SKIP_ALL, exec);
    let (no_rccl_ms, _, _) = pipelined(COLL_SKIP_RCCL, exec);
    set(COLL_TIME);
    exec.set_profile(true);
    sync();
    run(exec);
    exec.take_profile();
    let _ = coll.map(TimedCollective::take_calls);
    sync();
    run(exec);
    let profile = exec.take_profile();
    let calls = coll.map(TimedCollective::take_calls).unwrap_or_default();
    exec.set_profile(false);
    set(COLL_PASS);
    RankCase {
        case: name.to_string(),
        forward_ms,
        launch_ms,
        device_wait_ms,
        no_collectives_ms,
        no_rccl_ms,
        profile,
        calls,
    }
}

/// Fills the KV of [`TP_DECODE_SEQS`] sequences with distinct `DECODE_CTX − 1`-token prompts
/// (two per step), then measures a decode step of all of them and one [`PREFILL`]-token chunk.
fn tp_rank_cases(
    exec: &mut DecoderExecutor,
    mem: &Arc<dyn DeviceMemory>,
    vocab: u32,
    coll: Option<&TimedCollective>,
    barrier: Option<&Barrier>,
) -> Vec<RankCase> {
    let layout = *ModelExecutor::kv_layout(exec);
    let per_decode = DECODE_CTX.div_ceil(BLOCK_TOKENS);
    let per_prefill = PREFILL.div_ceil(BLOCK_TOKENS);
    let blocks = TP_DECODE_SEQS * per_decode + per_prefill;
    let storage = DeviceBuffer::alloc(mem, (layout.block_bytes() * u64::from(blocks)) as usize)
        .expect("KV pool");
    let kv = pool_view(&storage, &layout, blocks);
    let tables: Vec<Vec<BlockId>> = (0..TP_DECODE_SEQS)
        .map(|s| {
            (0..per_decode)
                .map(|b| BlockId(s * per_decode + b))
                .collect()
        })
        .collect();
    let prompts: Vec<Vec<u32>> = (0..TP_DECODE_SEQS)
        .map(|s| {
            (0..DECODE_CTX - 1)
                .map(|i| (i * 7919 + s * 104_729 + 1000) % vocab)
                .collect()
        })
        .collect();
    for pair in (0..TP_DECODE_SEQS as usize).collect::<Vec<_>>().chunks(2) {
        let seqs: Vec<(&[BlockId], &[u32], u32)> = pair
            .iter()
            .map(|&s| (tables[s].as_slice(), prompts[s].as_slice(), 0))
            .collect();
        step(exec, &kv, &seqs).expect("fill the KV");
    }
    let next: Vec<[u32; 1]> = (0..TP_DECODE_SEQS)
        .map(|s| [(s * 97 + 13) % vocab])
        .collect();
    let decode: Vec<(&[BlockId], &[u32], u32)> = (0..TP_DECODE_SEQS as usize)
        .map(|s| (tables[s].as_slice(), next[s].as_slice(), DECODE_CTX - 1))
        .collect();
    let table: Vec<BlockId> = (0..per_prefill)
        .map(|b| BlockId(TP_DECODE_SEQS * per_decode + b))
        .collect();
    let prompt: Vec<u32> = (0..PREFILL).map(|i| (i * 31 + 7) % vocab).collect();
    vec![
        tp_case(
            &format!("decode_b{TP_DECODE_SEQS}_ctx{DECODE_CTX}"),
            exec,
            coll,
            barrier,
            &mut |e| served_step(e, &kv, &decode).expect("decode step"),
        ),
        tp_case(
            &format!("prefill_{PREFILL}"),
            exec,
            coll,
            barrier,
            &mut |e| served_step(e, &kv, &[(&table, &prompt, 0)]).expect("prefill step"),
        ),
    ]
}

/// [`step`] as the server runs it: every row reduced on the device (greedy, one candidate) when
/// the executor reduces, so only the reductions are read back (under tensor parallelism the
/// vocabulary shards are still all-gathered first).
fn served_step(
    exec: &mut DecoderExecutor,
    kv: &KvPoolView<'_>,
    seqs: &[(&[BlockId], &[u32], u32)],
) -> Result<(), turbine_model::ModelError> {
    let reduce = ModelExecutor::reduces_logits(exec).then_some(RowReduce {
        top_n: 1,
        temperature: 0.0,
        uniform: None,
        top_p: 1.0,
    });
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
            block_formats: &[],
            reduce,
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

/// One (op, bytes, route) group of a case's collective calls on one rank.
#[derive(Serialize)]
struct CollGroup {
    op: &'static str,
    bytes: usize,
    route: &'static str,
    calls: u32,
    /// From the later rank's arrival until this rank's call completed, summed.
    transfer_ms: f64,
    /// This rank waiting for the later one to arrive, summed.
    peer_wait_ms: f64,
}

#[derive(Serialize)]
struct TpCaseReport {
    model: String,
    tp: u32,
    rank: u32,
    case: String,
    forward_ms: f64,
    launch_ms: f64,
    device_wait_ms: f64,
    no_collectives_ms: f64,
    no_rccl_ms: f64,
    profiled_ms: f64,
    /// Profiled collective ops (`tp_all_reduce`, `tp_all_gather`), peer wait included.
    collective_ms: f64,
    collective_transfer_ms: f64,
    collective_peer_wait_ms: f64,
    /// Profiled `gemm` and `moe_experts`.
    gemm_ms: f64,
    other_ms: f64,
    collectives: Vec<CollGroup>,
    ops: Vec<OpProfileEntry>,
}

const COLLECTIVE_OPS: [&str; 3] = ["tp_all_reduce", "tp_all_gather", "ep_combine"];

/// The report of rank `rank`'s `case`; `peer` is the other rank's call list of the same case.
fn tp_report(model: &str, tp: u32, rank: u32, case: &RankCase, peer: &[CollCall]) -> TpCaseReport {
    let sum = |f: &dyn Fn(&str) -> bool| -> f64 {
        case.profile
            .entries
            .iter()
            .filter(|e| f(&e.op))
            .map(|e| e.total_ms)
            .sum()
    };
    let collective_ms = sum(&|op| COLLECTIVE_OPS.contains(&op));
    let gemm_ms = sum(&|op| op == "gemm" || op == "moe_experts");
    let profiled_ms = case.profile.total_ms();
    assert!(
        peer.is_empty() || peer.len() == case.calls.len(),
        "{model} {}: rank {rank} made {} collective calls, its peer {}",
        case.case,
        case.calls.len(),
        peer.len()
    );
    let mut groups: Vec<CollGroup> = Vec::new();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    for (i, c) in case.calls.iter().enumerate() {
        let later = match peer.get(i) {
            Some(p) => {
                assert!(
                    p.op == c.op && p.bytes == c.bytes,
                    "{model} {}: call {i} differs across ranks",
                    case.case
                );
                p.arrive.max(c.arrive)
            }
            None => c.arrive,
        };
        let transfer = ms(c.end.saturating_duration_since(later));
        let wait = ms(later.saturating_duration_since(c.arrive));
        match groups
            .iter_mut()
            .find(|g| g.op == c.op.as_str() && g.bytes == c.bytes && g.route == c.route)
        {
            Some(g) => {
                g.calls += 1;
                g.transfer_ms += transfer;
                g.peer_wait_ms += wait;
            }
            None => groups.push(CollGroup {
                op: c.op.as_str(),
                bytes: c.bytes,
                route: c.route,
                calls: 1,
                transfer_ms: transfer,
                peer_wait_ms: wait,
            }),
        }
    }
    let mut ops = case.profile.entries.clone();
    ops.sort_by(|a, b| b.total_ms.total_cmp(&a.total_ms));
    TpCaseReport {
        model: model.to_string(),
        tp,
        rank,
        case: case.case.clone(),
        forward_ms: case.forward_ms,
        launch_ms: case.launch_ms,
        device_wait_ms: case.device_wait_ms,
        no_collectives_ms: case.no_collectives_ms,
        no_rccl_ms: case.no_rccl_ms,
        profiled_ms,
        collective_ms,
        collective_transfer_ms: groups.iter().map(|g| g.transfer_ms).sum(),
        collective_peer_wait_ms: groups.iter().map(|g| g.peer_wait_ms).sum(),
        gemm_ms,
        other_ms: profiled_ms - collective_ms - gemm_ms,
        collectives: groups,
        ops,
    }
}

fn print_tp_report(r: &TpCaseReport) {
    let top: Vec<String> = r
        .ops
        .iter()
        .take(6)
        .map(|e| format!("{}/{}={:.3}x{}", e.op, e.r#impl, e.total_ms, e.calls))
        .collect();
    println!(
        "tp_forward_profile_split: model={} tp={} rank={} case={} forward_ms={:.3} launch_ms={:.3} \
         device_wait_ms={:.3} no_collectives_ms={:.3} no_rccl_ms={:.3} profiled_ms={:.3} \
         collective_ms={:.3} (transfer {:.3}, peer_wait {:.3}) gemm_ms={:.3} other_ms={:.3} top=[{}]",
        r.model,
        r.tp,
        r.rank,
        r.case,
        r.forward_ms,
        r.launch_ms,
        r.device_wait_ms,
        r.no_collectives_ms,
        r.no_rccl_ms,
        r.profiled_ms,
        r.collective_ms,
        r.collective_transfer_ms,
        r.collective_peer_wait_ms,
        r.gemm_ms,
        r.other_ms,
        top.join(" ")
    );
    for g in &r.collectives {
        println!(
            "tp_forward_profile_collective: model={} rank={} case={} op={} bytes={} route={} \
             calls={} transfer_us_per_call={:.1} peer_wait_us_per_call={:.1} transfer_ms={:.3}",
            r.model,
            r.rank,
            r.case,
            g.op,
            g.bytes,
            g.route,
            g.calls,
            1e3 * g.transfer_ms / f64::from(g.calls),
            1e3 * g.peer_wait_ms / f64::from(g.calls),
            g.transfer_ms
        );
    }
    println!(
        "tp_forward_profile: {}",
        serde_json::to_string(r).expect("serialize the report")
    );
}

/// What one rank thread hands back: its cases and the route of every profiled call.
type RankRun = (Vec<RankCase>, Vec<&'static str>);

/// Lab only (novanas, both R9700s; P5 Task 32): where a tensor-parallel (tp 2) step spends its
/// time, for Llama-3.2-3B-Instruct and OLMoE-1B-7B-0125-Instruct. Per model it runs tp 1 on GPU 0
/// for reference, then tp 2 on GPUs 0 and 1 (ranks as threads, `hostmem` with its RCCL delegate
/// at the `auto` threshold, as `parallel.ranks.mode: local` serves) over a decode step of 16
/// sequences at a 768-token context and one 2,048-token prefill chunk, each rank printing:
///
/// - the pipelined forward (median of 5, as served: no per-op synchronisation), its step stages
///   (`launch`, `device_wait`), and the same forward with every collective skipped and with only
///   the RCCL-routed ones skipped (wrong numbers, right timing of everything else: what a free
///   collective would save);
/// - the profiled forward (every op synchronised): collective, GEMM (`gemm`, `moe_experts`) and
///   other time, the top ops, and every collective call grouped by (op, bytes, route) with its
///   transfer time (from the later rank's arrival) and its wait for the peer.
///
/// Asserts only that both ranks issue the same calls and that every route the report names was
/// counted by `turbine_collective_route_total`. Run with
/// `scripts/lab-test.sh novanas --gpus 2 -- --release -p turbine-model --test perf -- tp_forward_profile --nocapture`.
#[test]
#[ignore = "needs two HIP devices, TURBINE_KERNEL_LIBRARY, RCCL, TURBINE_TEST_MODEL_DIR and TURBINE_TEST_MOE_MODEL_DIR"]
fn tp_forward_profile() {
    if !require_backend("hip") {
        return;
    }
    let _gpu = lock_gpu();
    let _watchdog = tp_watchdog("tp_forward_profile", Duration::from_secs(1800));
    let dirs = [
        require_env_dir("TURBINE_TEST_MODEL_DIR"),
        require_env_dir("TURBINE_TEST_MOE_MODEL_DIR"),
    ];
    let ctxs = hip_contexts();
    assert!(ctxs.len() >= 2, "two AMD devices, found {}", ctxs.len());
    let lib = collective::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("hostmem (with RCCL as the delegate)");
    for dir in &dirs {
        let cfg = load_model_config(dir).expect("config.json");
        let model = dir
            .file_name()
            .map_or_else(|| dir.display().to_string(), |n| n.to_string_lossy().into());
        // tp 1 on GPU 0, for reference.
        {
            let mut exec = hip_decoder(&cfg, dir, &ctxs[0], ExecutorOptions::default(), PREFILL);
            let mem: Arc<dyn DeviceMemory> = ctxs[0].clone();
            for case in tp_rank_cases(&mut exec, &mem, cfg.vocab_size, None, None) {
                print_tp_report(&tp_report(&model, 1, 0, &case, &[]));
            }
        }
        // tp 2 on GPUs 0 and 1.
        let registry = MetricsRegistry::new();
        let metrics = CollectiveMetrics::register(&registry);
        let id = lib.unique_id().expect("group id");
        let barrier = Barrier::new(2);
        let ranks: Vec<RankRun> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2u32)
                .map(|rank| {
                    let (lib, ctx, cfg, barrier, metrics) = (
                        Arc::clone(&lib),
                        &ctxs[rank as usize],
                        &cfg,
                        &barrier,
                        &metrics,
                    );
                    scope.spawn(move || {
                        let mem: Arc<dyn DeviceMemory> = ctx.clone();
                        // Makes this thread's current device the rank's (RCCL's init uses it).
                        mem.mem_info().expect("device memory info");
                        let inner = lib
                            .open(CollectiveInit {
                                rank: rank as usize,
                                world: 2,
                                unique_id: id,
                                init_timeout: Duration::from_secs(120),
                                op_timeout: Duration::from_secs(60),
                                clock: Arc::new(SystemClock::new()),
                                metrics: Some(metrics.clone()),
                                memory: Some(Arc::clone(&mem)),
                                route_max_bytes: None,
                            })
                            .expect("open the communicator");
                        let coll = TimedCollective::new(inner, Arc::clone(&mem));
                        let s = tp::ShardSpec { rank, world: 2 };
                        let mut exec = tp_rank_decoder(
                            cfg,
                            dir,
                            ctx,
                            s,
                            Arc::clone(&coll) as Arc<dyn Collective>,
                        );
                        let cases = tp_rank_cases(
                            &mut exec,
                            &mem,
                            cfg.vocab_size,
                            Some(&coll),
                            Some(barrier),
                        );
                        let routes = cases
                            .iter()
                            .flat_map(|c| c.calls.iter().map(|k| k.route))
                            .collect();
                        (cases, routes)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank"))
                .collect()
        });
        for (rank, (cases, _)) in ranks.iter().enumerate() {
            let peer = &ranks[1 - rank].0;
            for (case, other) in cases.iter().zip(peer) {
                print_tp_report(&tp_report(&model, 2, rank as u32, case, &other.calls));
            }
        }
        let text = registry.render().expect("render the collective metrics");
        for line in text
            .lines()
            .filter(|l| l.starts_with("turbine_collective_route_total"))
        {
            println!("tp_forward_profile_route: model={model} {line}");
        }
        for route in ["hostmem", "rccl"] {
            if ranks[0].1.contains(&route) {
                assert!(
                    text.contains(&format!("backend=\"{route}\"")),
                    "{model}: the report names route {route}, the route metric never counted it"
                );
            }
        }
    }
}
