//! The Phase 8 model families (Qwen3, Qwen3-MoE, Mistral, Mixtral; Phase 2m S-11, from the
//! run-ahead's `tests/families.rs`) on their tiny synthetic checkpoints, through the family
//! registry: the `cpu-reference` provider against the naive decoder
//! ([`turbine_model::testing::naive`], which reads the checkpoint under its own tensor names),
//! fused against unfused op sequences, ragged batches against single-sequence runs, and each
//! family's weight slots and op lists.
use std::sync::Arc;

use turbine_core::types::{BlockId, DeviceId, KvLayout, SeqId};
use turbine_kernels::{KernelMetrics, KernelRegistry, cpu_reference_provider};
use turbine_model::config::ModelArchConfig;
use turbine_model::executor::{
    self, BatchInput, ExecutorOptions, Logits, ModelExecutor, SeqSlice, SequenceKv, build_executor,
};
use turbine_model::families::{self, FamilyRef, Llama};
use turbine_model::testing::TempDir;
use turbine_model::testing::naive::Naive;
use turbine_model::testing::tiny::{TINY_PHASE8_FAMILIES, TinySpec, write_tiny_family};
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader};
use turbine_observability::MetricsRegistry;
use turbine_tensor::host::HostMemory;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

const SEED: u64 = 11;
const PROMPT_LEN: usize = 20;
const DECODE_STEPS: usize = 10;
const MAX_SEQ_LEN: u32 = 64;
const BLOCK_TOKENS: u32 = 128;
const MAX_SEQS: u32 = 4;

fn tiny(tmp: &TempDir, family: &str) -> TinySpec {
    write_tiny_family(&tmp.path().join(family), family, SEED)
}

fn host_mem() -> Arc<dyn DeviceMemory> {
    HostMemory::new(DeviceId(0), 1 << 30)
}

/// The family's executor (through the registry) on the CPU provider, run with `opts`.
fn cpu_model(
    spec: &TinySpec,
    mem: &Arc<dyn DeviceMemory>,
    opts: ExecutorOptions,
) -> Box<dyn ModelExecutor> {
    let cfg = &spec.config;
    let index = SafetensorsIndex::open(&spec.dir).expect("open tiny index");
    let slots = cfg.family.0.weight_slots(cfg);
    let weights = WeightLoader::load(&index, &slots, mem, MAX_STAGING_BYTES).expect("load");
    assert!(
        weights.unexpected.is_empty() && weights.ignored.is_empty(),
        "{}: every checkpoint tensor maps to a parameter: {:?} {:?}",
        cfg.hf_architecture,
        weights.unexpected,
        weights.ignored
    );
    let provider = cpu_reference_provider();
    let order = [provider.id()];
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let reqs =
        executor::available_requirements(cfg, BLOCK_TOKENS, opts, std::slice::from_ref(&provider));
    let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics)
        .expect("every op has a provider");
    build_executor(
        cfg,
        weights,
        Arc::new(registry),
        Arc::clone(mem),
        BLOCK_TOKENS,
        MAX_SEQ_LEN,
        MAX_SEQS,
        opts,
    )
    .expect("executor")
}

fn prompt(vocab: u32, salt: u32) -> Vec<u32> {
    (0..PROMPT_LEN as u32)
        .map(|i| (i * 37 + 11 + salt) % vocab)
        .collect()
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

/// Prefill then greedy decode on `exec`; every row within 1e-4 of the naive model. Returns
/// the rows and the tokens.
fn check_against_naive(
    spec: &TinySpec,
    exec: &mut dyn ModelExecutor,
    mem: &Arc<dyn DeviceMemory>,
    naive: &Naive,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let name = &spec.config.hf_architecture;
    let mut kv = SequenceKv::new(mem, *exec.kv_layout(), MAX_SEQ_LEN).expect("kv");
    let mut tokens = prompt(spec.vocab, 0);
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();
    let mut rows = vec![kv.forward(exec, &tokens, &positions).expect("prefill").data];
    let diff = max_abs_diff(&rows[0], &naive.last_logits(&tokens));
    assert!(diff <= 1e-4, "{name} prefill: max abs diff {diff}");
    for step in 0..DECODE_STEPS {
        let next = argmax(rows.last().expect("row"));
        let pos = tokens.len() as u32;
        tokens.push(next);
        let row = kv.forward(exec, &[next], &[pos]).expect("decode").data;
        let diff = max_abs_diff(&row, &naive.last_logits(&tokens));
        assert!(
            diff <= 1e-4,
            "{name} decode step {step}: max abs diff {diff}"
        );
        rows.push(row);
    }
    (rows, tokens)
}

/// Every Phase 8 family resolves through the registry from its tiny checkpoint, under its
/// registry name and Hugging Face name.
#[test]
fn families_resolve_through_the_registry() {
    let tmp = TempDir::new("families-resolve");
    for (name, hf) in TINY_PHASE8_FAMILIES.into_iter().zip([
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
    ]) {
        let spec = tiny(&tmp, name);
        assert_eq!(spec.config.family.0.name(), name);
        assert_eq!(spec.config.hf_architecture, hf);
        let registered = families::registry().get(name).expect("registered");
        assert_eq!(spec.config.family, FamilyRef(registered));
    }
}

/// Every family on the CPU provider matches the naive model for a prefill and 10 greedy decode
/// steps, with the default options and with every fusion off, bit for bit between the two; and
/// the naive model is far from the same model without its distinguishing feature (per-head
/// Q/K norm, renormalised routing, the untied LM head), so none can go missing unnoticed.
#[test]
fn families_cpu_forward_matches_naive() {
    let tmp = TempDir::new("families-naive");
    let mem = host_mem();
    for family in TINY_PHASE8_FAMILIES {
        let spec = tiny(&tmp, family);
        let name = spec.config.hf_architecture.clone();
        let naive = Naive::load(&spec.dir, &spec.config);
        let mut fused = cpu_model(&spec, &mem, ExecutorOptions::default());
        let (rows, tokens) = check_against_naive(&spec, fused.as_mut(), &mem, &naive);
        let mut unfused = cpu_model(&spec, &mem, ExecutorOptions::from_fused_ops(false));
        let (unfused_rows, _) = check_against_naive(&spec, unfused.as_mut(), &mem, &naive);
        for (step, (a, b)) in rows.iter().zip(&unfused_rows).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{name} step {step}: fused differs from unfused by {}",
                max_abs_diff(a, b)
            );
        }

        let last = rows.last().expect("rows");
        let mutated = |edit: &dyn Fn(&mut Naive)| {
            let mut m = Naive::load(&spec.dir, &spec.config);
            edit(&mut m);
            max_abs_diff(last, &m.last_logits(&tokens))
        };
        let mut checks: Vec<(&str, f32)> = Vec::new();
        match family {
            "qwen3" => checks.push(("per-head Q/K norm", mutated(&|m| m.qk_norm = false))),
            "qwen3_moe" => {
                checks.push(("per-head Q/K norm", mutated(&|m| m.qk_norm = false)));
                checks.push(("renormalisation", mutated(&|m| m.renormalize = false)));
            }
            "mistral" => checks.push(("the untied head", mutated(&|m| m.tied = true))),
            _ => checks.push(("renormalisation", mutated(&|m| m.renormalize = false))),
        }
        for (feature, diff) in checks {
            assert!(diff > 1e-3, "{name}: {feature} changes nothing: {diff}");
        }
    }
}

/// The families' op lists: Qwen3 families add an RMSNorm over `head_dim` (16) and keep Q/K/V
/// unfused even with fused projections (Qwen3's gate/up still fuse); Mixtral has no Q/K norm and
/// renormalised routing; Mistral is exactly the Llama op list.
#[test]
fn family_requirements() {
    let tmp = TempDir::new("families-reqs");
    let rendered = |cfg: &ModelArchConfig| -> Vec<String> {
        executor::requirements(cfg, BLOCK_TOKENS, ExecutorOptions::default())
            .iter()
            .map(|r| format!("{} {}", r.op, r.config))
            .collect()
    };
    let gemm = |n: u32| format!("gemm n={n} k=64 trans_b=1 a_dtype=bf16 b_dtype=bf16 c_dtype=bf16");
    // q 4 × 16, k and v 2 × 16 each.
    let (fused_qkv, q, kv) = (gemm(128), gemm(64), gemm(32));
    let head_norm = "rmsnorm dim=16 dtype=bf16".to_string();

    let qwen3 = rendered(&tiny(&tmp, "qwen3").config);
    for want in [&q, &kv, &head_norm, &gemm(256)] {
        assert!(qwen3.contains(want), "Qwen3 lacks {want}: {qwen3:#?}");
    }
    assert!(!qwen3.contains(&fused_qkv), "{qwen3:#?}");

    let qwen3_moe = rendered(&tiny(&tmp, "qwen3_moe").config);
    for want in [
        &q,
        &kv,
        &head_norm,
        &"moe_route experts=8 top_k=2 renormalize=1 bf16_logits=1".to_string(),
        &"moe_experts hidden=64 inter=32 experts=8 top_k=2 local=0..8 dtype=bf16".to_string(),
    ] {
        assert!(
            qwen3_moe.iter().any(|r| r.starts_with(want.as_str())),
            "Qwen3-MoE lacks {want}: {qwen3_moe:#?}"
        );
    }
    assert!(!qwen3_moe.contains(&fused_qkv), "{qwen3_moe:#?}");

    let mixtral = rendered(&tiny(&tmp, "mixtral").config);
    assert!(mixtral.contains(&fused_qkv), "{mixtral:#?}");
    assert!(
        mixtral
            .iter()
            .any(|r| r.starts_with("moe_route experts=8 top_k=2 renormalize=1 bf16_logits=1")),
        "{mixtral:#?}"
    );
    assert!(
        !mixtral
            .iter()
            .any(|r| r.starts_with("rmsnorm dim=32") || r.starts_with("rmsnorm dim=16")),
        "Mixtral has no Q/K norm: {mixtral:#?}"
    );

    let mistral = tiny(&tmp, "mistral").config;
    let mut as_llama = mistral.clone();
    as_llama.family = FamilyRef(&Llama);
    assert_eq!(rendered(&mistral), rendered(&as_llama));
}

/// Mixtral's slots map its `block_sparse_moe` names onto the MoE hook's parameters: the router
/// under `mlp.gate`, experts `w1` / `w3` / `w2` into the gate / up / down stacks.
#[test]
fn mixtral_weight_map() {
    let tmp = TempDir::new("families-mixtral-map");
    let spec = tiny(&tmp, "mixtral");
    let slots = spec.config.family.0.weight_slots(&spec.config);
    let find = |name: &str| {
        slots
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no slot {name}"))
    };
    let router = find("model.layers.1.block_sparse_moe.gate.weight");
    let stack = router.stack.as_ref().expect("renamed through a stack");
    assert_eq!(
        (stack.name.as_str(), stack.shape.as_slice(), stack.offset),
        ("model.layers.1.mlp.gate.weight", &[8usize, 64][..], 0)
    );
    for (proj, stacked, shape) in [
        ("w1", "gate_proj", [32usize, 64]),
        ("w3", "up_proj", [32, 64]),
        ("w2", "down_proj", [64, 32]),
    ] {
        let slot = find(&format!(
            "model.layers.0.block_sparse_moe.experts.3.{proj}.weight"
        ));
        let stack = slot.stack.as_ref().expect("stacked");
        assert_eq!(
            stack.name,
            format!("model.layers.0.mlp.experts.{stacked}.weight")
        );
        assert_eq!(stack.offset, 3 * shape[0] * shape[1]);
    }
    assert!(!slots.iter().any(|s| s.name.contains("q_norm")));
    // The checkpoint the tiny writer made from these slots loads with nothing left over.
    let mem = host_mem();
    cpu_model(&spec, &mem, ExecutorOptions::default());
}

/// One pool of `blocks` blocks of `layout`.
fn pool(mem: &Arc<dyn DeviceMemory>, layout: &KvLayout, blocks: u32) -> DeviceBuffer {
    DeviceBuffer::alloc(mem, (layout.block_bytes() * u64::from(blocks)) as usize).expect("pool")
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

fn forward(
    exec: &mut dyn ModelExecutor,
    kv: &KvPoolView<'_>,
    seqs: &[(&[u32], u32, &[BlockId])],
) -> Logits {
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut slices = Vec::new();
    for (i, &(new, start, table)) in seqs.iter().enumerate() {
        slices.push(SeqSlice {
            seq: SeqId(i as u64 + 1),
            q_start: tokens.len() as u32,
            q_len: new.len() as u32,
            kv_len: start + new.len() as u32,
            block_table: table,
            reduce: None,
        });
        tokens.extend_from_slice(new);
        positions.extend(start..start + new.len() as u32);
    }
    exec.forward(&BatchInput {
        tokens: &tokens,
        positions: &positions,
        seqs: &slices,
        kv,
    })
    .expect("forward")
}

/// A ragged batch — two prompts of different lengths prefilled together, then decoded
/// together — gives each sequence the logits of its own single-sequence run (within 1e-5: the
/// per-head norm and every other op act on rows independently), so the per-head Q/K norm's
/// `[tokens · heads, head_dim]` rows never mix sequences.
#[test]
fn families_ragged_batch_matches_single() {
    let tmp = TempDir::new("families-ragged");
    let mem = host_mem();
    for family in TINY_PHASE8_FAMILIES {
        let spec = tiny(&tmp, family);
        let name = &spec.config.hf_architecture;
        let mut exec = cpu_model(&spec, &mem, ExecutorOptions::default());
        let layout = *exec.kv_layout();
        let storage = pool(&mem, &layout, 4);
        let kv = pool_view(&storage, &layout, 4);
        let a = prompt(spec.vocab, 0);
        let b: Vec<u32> = prompt(spec.vocab, 5)[..13].to_vec();
        let (ta, tb) = ([BlockId(2)], [BlockId(0)]);

        let batch = forward(exec.as_mut(), &kv, &[(&a, 0, &ta), (&b, 0, &tb)]);
        let (na, nb) = (argmax(batch.row(0)), argmax(batch.row(1)));
        let decode = forward(
            exec.as_mut(),
            &kv,
            &[(&[na], a.len() as u32, &ta), (&[nb], b.len() as u32, &tb)],
        );

        for (i, (p, next, prefill_row, decode_row)) in [
            (&a, na, batch.row(0), decode.row(0)),
            (&b, nb, batch.row(1), decode.row(1)),
        ]
        .into_iter()
        .enumerate()
        {
            let table = [BlockId(3)];
            let single = forward(exec.as_mut(), &kv, &[(p, 0, &table)]);
            let diff = max_abs_diff(prefill_row, single.row(0));
            assert!(diff <= 1e-5, "{name} seq {i} prefill: {diff}");
            let single = forward(exec.as_mut(), &kv, &[(&[next], p.len() as u32, &table)]);
            let diff = max_abs_diff(decode_row, single.row(0));
            assert!(diff <= 1e-5, "{name} seq {i} decode: {diff}");
        }
    }
}

/// The naive decoder's per-position rows: row `i` of a whole-sequence forward equals the last
/// row of the forward over `tokens[..=i]` (causal), for a dense and a mixture-of-experts family.
#[test]
fn naive_forward_rows_are_causal() {
    let tmp = TempDir::new("families-naive-rows");
    for family in ["qwen3", "mixtral"] {
        let spec = tiny(&tmp, family);
        let tokens = prompt(spec.vocab, 3)[..6].to_vec();
        let rows = turbine_model::testing::naive::forward(&spec.config, &spec.dir, &tokens);
        assert_eq!(rows.len(), tokens.len());
        let naive = Naive::load(&spec.dir, &spec.config);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), spec.vocab as usize);
            assert_eq!(row, &naive.last_logits(&tokens[..=i]), "{family} row {i}");
        }
    }
}
