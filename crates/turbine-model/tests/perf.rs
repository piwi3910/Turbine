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
//! a server-like continuous batch for every fusion combination.
//!
//! Run with `scripts/lab-test.sh novanas -- --release -p turbine-model --test perf -- forward_profile --nocapture`.
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde::Serialize;
use turbine_core::types::{BlockId, KvLayout, SeqId, Vendor};
use turbine_kernels::{
    KernelMetrics, KernelRegistry, OpKind, shim_provider, test_support::require_backend,
    test_support::require_env_dir,
};
use turbine_model::config::{Architecture, ModelArchConfig};
use turbine_model::executor::{
    self, BatchInput, ExecutorOptions, LlamaExecutor, ModelExecutor, OlmoeExecutor, OpProfile,
    OpProfileEntry, SeqSlice,
};
use turbine_model::{
    MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, llama_slots, load_model_config, olmoe_slots,
};
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

impl Profiled for LlamaExecutor {
    fn set_profile(&mut self, on: bool) {
        LlamaExecutor::set_profile(self, on);
    }
    fn take_profile(&mut self) -> OpProfile {
        LlamaExecutor::take_profile(self)
    }
}

impl Profiled for OlmoeExecutor {
    fn set_profile(&mut self, on: bool) {
        OlmoeExecutor::set_profile(self, on);
    }
    fn take_profile(&mut self) -> OpProfile {
        OlmoeExecutor::take_profile(self)
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
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let index = SafetensorsIndex::open(dir).expect("open safetensors");
    let slots = match cfg.architecture {
        Architecture::Llama => llama_slots(cfg),
        Architecture::Olmoe => olmoe_slots(cfg),
        other => panic!("no profile for {other:?}"),
    };
    let weights = WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("weights");
    let provider = shim_provider(ctx.clone());
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let registry = Arc::new(
        KernelRegistry::build(
            vec![provider],
            &order,
            &hip_requirements(cfg, ctx, opts),
            &metrics,
        )
        .expect("every op has a provider"),
    );
    match cfg.architecture {
        Architecture::Llama => Box::new(
            LlamaExecutor::new(
                cfg,
                weights,
                registry,
                mem,
                BLOCK_TOKENS,
                max_batch_tokens,
                MAX_SEQS,
                opts,
            )
            .expect("llama executor"),
        ),
        _ => Box::new(
            OlmoeExecutor::new(
                cfg,
                weights,
                registry,
                mem,
                BLOCK_TOKENS,
                max_batch_tokens,
                MAX_SEQS,
                opts,
            )
            .expect("olmoe executor"),
        ),
    }
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
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .expect("TURBINE_KERNEL_LIBRARY is not set; point it at libturbine_hip.so");
    let lib = turbine_kernels::ShimLibrary::load(Path::new(&library), "hip")
        .expect("load the HIP kernel library");
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let device = inventory
        .devices
        .iter()
        .find(|d| d.vendor == Vendor::Amd)
        .expect("an AMD device");
    let ctx = lib.create_context(device).expect("HIP context");
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
    let library = std::env::var_os("TURBINE_KERNEL_LIBRARY")
        .filter(|v| !v.is_empty())
        .expect("TURBINE_KERNEL_LIBRARY is not set; point it at libturbine_hip.so");
    let lib = turbine_kernels::ShimLibrary::load(Path::new(&library), "hip")
        .expect("load the HIP kernel library");
    let inventory = turbine_device::discover(&turbine_device::DiscoveryOptions::default())
        .expect("device discovery");
    let device = inventory
        .devices
        .iter()
        .find(|d| d.vendor == Vendor::Amd)
        .expect("an AMD device");
    let ctx = lib.create_context(device).expect("HIP context");
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
