//! Lab performance profile (P2c S-2, S-15): the real executors on the R9700 with profile mode
//! on. `forward_profile` loads Llama-3.2-3B-Instruct (`TURBINE_TEST_MODEL_DIR`) and
//! OLMoE-1B-7B-0125-Instruct (`TURBINE_TEST_MOE_MODEL_DIR`) on the HIP provider and prints, per
//! model, one `forward_profile: <json>` line: for a decode step of 1, 16 and 64 sequences at a
//! 768-token context and for one 2,048-token prefill chunk, the unprofiled forward time (median
//! of 5) and the per-op profile ([`OpProfile`]). The kernel-level counterpart is
//! `hip_ops decode_forward_timing` (a synthetic forward of the same shapes): the difference
//! between the two forward times is the executor's own overhead.
//!
//! Run with `scripts/lab-test.sh novanas -- --release -p turbine-model --test perf -- forward_profile --nocapture`.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use turbine_core::types::{BlockId, ExecutionBackend, KvLayout, SeqId, Vendor};
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
    cases: Vec<CaseReport>,
}

/// The requirements the HIP provider serves for `cfg` with the default (fused) op sequence.
fn hip_requirements(
    cfg: &ModelArchConfig,
    ctx: &Arc<turbine_kernels::ShimContext>,
) -> Vec<turbine_kernels::OpRequirement> {
    executor::available_requirements(
        cfg,
        BLOCK_TOKENS,
        ExecutorOptions::default(),
        &[shim_provider(ctx.clone())],
    )
}

/// The model's executor on the HIP provider, for batches of up to [`PREFILL`] tokens and
/// [`MAX_SEQS`] sequences, running the default (fused) op sequence.
fn hip_executor(
    cfg: &ModelArchConfig,
    dir: &Path,
    ctx: &Arc<turbine_kernels::ShimContext>,
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
            &hip_requirements(cfg, ctx),
            &metrics,
        )
        .expect("every op has a provider"),
    );
    let opts = ExecutorOptions::default();
    match cfg.architecture {
        Architecture::Llama => Box::new(
            LlamaExecutor::new(
                cfg,
                weights,
                registry,
                mem,
                BLOCK_TOKENS,
                PREFILL,
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
                PREFILL,
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
fn profile_model(dir: &Path, ctx: &Arc<turbine_kernels::ShimContext>) -> ModelReport {
    let cfg = load_model_config(dir).expect("config.json");
    let mut exec = hip_executor(&cfg, dir, ctx);
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
    let kinds: Vec<OpKind> = hip_requirements(&cfg, ctx)
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
    let dirs = [
        require_env_dir("TURBINE_TEST_MODEL_DIR"),
        require_env_dir("TURBINE_TEST_MOE_MODEL_DIR"),
    ];
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
    for dir in &dirs {
        let report = profile_model(dir, &ctx);
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
