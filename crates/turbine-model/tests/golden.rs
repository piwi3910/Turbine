//! In-process golden tests (P1 S-11): Turbine's greedy generation replayed against a Hugging
//! Face transformers reference produced by `scripts/golden/hf_reference.py`.
//!
//! - `hf_reference_matches_cpu` (lab only, needs `uv`): the tiny synthetic checkpoint on the
//!   `cpu-reference` provider against a reference generated on the spot.
//! - `logits_match_reference` (lab only, needs a HIP device and the weights): Llama-3.2-3B on the
//!   HIP provider against the committed `tests/golden/llama-3.2-3b-instruct/reference.jsonl`.
//!
//! Both judge with the committed `tolerance.json` and the same rule as `turbine-golden compare`,
//! re-implemented in [`compare_prompt`] because this crate cannot depend on `turbine-bench`
//! (contract §1.2).
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use serde::Deserialize;
use smallvec::SmallVec;
use turbine_core::request::{
    CancelFlag, Endpoint, FinishReason, GenerationEvent, GenerationRequest, SamplingParams,
    StopConditions,
};
use turbine_core::types::{ExecutionBackend, RequestId, Vendor};
use turbine_kernels::{
    KernelMetrics, KernelProvider, KernelRegistry, cpu_reference_provider, shim_provider,
};
use turbine_model::executor::{BatchInput, LlamaExecutor, ModelExecutor};
use turbine_model::generate::{GenerateOptions, generate};
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::write_tiny_llama;
use turbine_model::testing::trace::{LocalChecker, compare_traces, read_bf16_weight, render};
use turbine_model::{
    ChatTemplate, MAX_STAGING_BYTES, SafetensorsIndex, Tokenizer, WeightLoader, llama_slots,
    load_model_config,
};
use turbine_observability::MetricsRegistry;
use turbine_tensor::DeviceMemory;
use turbine_tensor::host::HostMemory;

/// Alternatives requested per generated position: the reference's `--top-logprobs` default and
/// the most `turbine-golden compare` asks for.
const TOP_LOGPROBS: u32 = 20;

/// The logprob bound of the tiny-checkpoint test (the committed Llama-3.2 tolerance otherwise
/// applies unchanged). The random tiny weights give logits of magnitude 16–32 with top-1 to
/// top-5 gaps of 6–10 nats, where one BF16 ulp is 0.125: the reference log-softmaxes BF16
/// logits and differs from Turbine's F32 LM head by up to half an ulp per logit, and BF16
/// rounding differences inside the forward pass are amplified by the same scale. Measured
/// against transformers 4.57.1 on CPU: all 512 greedy tokens and every prompt id identical,
/// max |Δ logprob| 0.21 over the top 5; 0.3 keeps that property meaningful (a wrong layer,
/// template or tokenization breaks the tokens or moves logprobs by whole nats).
const TINY_MAX_ABS_LOGPROB_DIFF: f32 = 0.3;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_dir() -> PathBuf {
    repo_root().join("tests/golden")
}

// ------------------------------------------------------------------------------ fixtures

/// One line of `tests/golden/prompts.jsonl`.
#[derive(Debug, Deserialize)]
struct PromptRecord {
    id: String,
    kind: String,
    prompt: Option<String>,
    messages: Option<Vec<serde_json::Value>>,
    max_tokens: u32,
    #[serde(default)]
    chat_template_kwargs: serde_json::Map<String, serde_json::Value>,
}

/// One line of a `reference.jsonl` written by `hf_reference.py`.
#[derive(Debug, Deserialize)]
struct ReferenceRecord {
    id: String,
    prompt_token_ids: Vec<u32>,
    tokens: Vec<u32>,
    top_logprobs: Vec<Vec<(u32, f32)>>,
}

/// `tolerance.json`.
#[derive(Debug, Deserialize)]
struct Tolerance {
    min_identical_prefix: usize,
    min_prompts_passing: usize,
    top_k: usize,
    max_abs_logprob_diff: f32,
    margin_nats: f32,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), n + 1))
        })
        .collect()
}

fn read_tolerance(path: &Path) -> Tolerance {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// ----------------------------------------------------------------------- tolerance rule

/// Per-prompt result of [`compare_prompt`].
#[derive(Debug)]
struct Verdict {
    id: String,
    reference_len: usize,
    identical_prefix: usize,
    first_divergence: Option<usize>,
    margin_at_divergence: Option<f32>,
    max_abs_logprob_diff: f32,
    /// `(position, token id)` of the first reference top-k id missing from Turbine's top list.
    missing_top_k: Option<(usize, u32)>,
    logprob_within_bound: bool,
    passed: bool,
}

/// The `turbine-golden compare` rule for one prompt: the greedy prefix must be at least
/// `min_identical_prefix` long unless the reference margin (top-1 minus top-2) at the first
/// divergence is below `margin_nats`; at every position before the divergence each reference
/// top-k id must appear in Turbine's top list within `max_abs_logprob_diff`. A candidate that
/// stops early diverges at its length.
fn compare_prompt(
    reference: &ReferenceRecord,
    got_tokens: &[u32],
    got_top: &[Vec<(u32, f32)>],
    tol: &Tolerance,
) -> Verdict {
    let n = reference.tokens.len();
    let identical_prefix = reference
        .tokens
        .iter()
        .zip(got_tokens)
        .take_while(|(r, g)| r == g)
        .count();
    let first_divergence = (identical_prefix < n).then_some(identical_prefix);
    let sorted = |row: &[(u32, f32)]| {
        let mut row = row.to_vec();
        row.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        row
    };
    let margin_at_divergence = first_divergence
        .and_then(|d| reference.top_logprobs.get(d))
        .map(|row| sorted(row))
        .filter(|row| row.len() >= 2)
        .map(|row| row[0].1 - row[1].1);

    let mut max_abs_logprob_diff = 0f32;
    let mut missing_top_k = None;
    for pos in 0..first_divergence.unwrap_or(n) {
        let Some(ref_row) = reference.top_logprobs.get(pos) else {
            break;
        };
        let got_row = got_top.get(pos).map(Vec::as_slice).unwrap_or(&[]);
        for &(id, lp) in sorted(ref_row).iter().take(tol.top_k) {
            match got_row.iter().find(|e| e.0 == id) {
                Some(&(_, got_lp)) => {
                    max_abs_logprob_diff = max_abs_logprob_diff.max((got_lp - lp).abs());
                }
                None => {
                    missing_top_k.get_or_insert((pos, id));
                }
            }
        }
    }
    let logprob_within_bound =
        missing_top_k.is_none() && max_abs_logprob_diff <= tol.max_abs_logprob_diff;
    let prefix_ok = identical_prefix >= tol.min_identical_prefix.min(n);
    let excused = margin_at_divergence.is_some_and(|m| m < tol.margin_nats);
    Verdict {
        id: reference.id.clone(),
        reference_len: n,
        identical_prefix,
        first_divergence,
        margin_at_divergence,
        max_abs_logprob_diff,
        missing_top_k,
        logprob_within_bound,
        passed: (prefix_ok || excused) && logprob_within_bound,
    }
}

/// The whole set holds: the logprob bound on every prompt and enough prompts passing.
fn assert_tolerance(verdicts: &[Verdict], tol: &Tolerance) {
    let mut report = String::new();
    for v in verdicts {
        report.push_str(&format!(
            "{} {} identical_prefix={}/{} first_divergence={:?} margin={:?} \
             max_abs_logprob_diff={:.4} missing_top_k={:?}\n",
            if v.passed { "PASS" } else { "FAIL" },
            v.id,
            v.identical_prefix,
            v.reference_len,
            v.first_divergence,
            v.margin_at_divergence,
            v.max_abs_logprob_diff,
            v.missing_top_k,
        ));
    }
    println!("{report}");
    let passing = verdicts.iter().filter(|v| v.passed).count();
    assert!(!verdicts.is_empty(), "no prompts compared");
    assert!(
        verdicts.iter().all(|v| v.logprob_within_bound),
        "|Δ logprob| over the reference top-{} exceeds {} (or an id is missing):\n{report}",
        tol.top_k,
        tol.max_abs_logprob_diff,
    );
    assert!(
        passing >= tol.min_prompts_passing,
        "{passing}/{} prompts passing, need {}:\n{report}",
        verdicts.len(),
        tol.min_prompts_passing,
    );
}

#[test]
fn compare_prompt_applies_the_tolerance_rule() {
    let tol = Tolerance {
        min_identical_prefix: 4,
        min_prompts_passing: 1,
        top_k: 2,
        max_abs_logprob_diff: 0.15,
        margin_nats: 0.5,
    };
    // Four positions: token 10+i with runner-up 20+i at the given margin.
    let reference = |margins: [f32; 4]| ReferenceRecord {
        id: "p".into(),
        prompt_token_ids: vec![1],
        tokens: (0..4).map(|i| 10 + i).collect(),
        top_logprobs: margins
            .iter()
            .zip(0u32..)
            .map(|(&m, i)| vec![(20 + i, -1.0 - m), (10 + i, -1.0), (30 + i, -9.0)])
            .collect(),
    };
    let top = |r: &ReferenceRecord| -> Vec<Vec<(u32, f32)>> { r.top_logprobs.clone() };

    let exact = reference([2.0; 4]);
    let v = compare_prompt(&exact, &exact.tokens, &top(&exact), &tol);
    assert!(v.passed && v.first_divergence.is_none(), "{v:?}");

    // A flip at a decisive position fails; at a near tie it is excused.
    let flipped = [10, 11, 99, 13];
    let v = compare_prompt(&exact, &flipped, &top(&exact), &tol);
    assert_eq!(
        (v.first_divergence, v.margin_at_divergence),
        (Some(2), Some(2.0))
    );
    assert!(!v.passed && v.logprob_within_bound, "{v:?}");
    let tie = reference([2.0, 2.0, 0.3, 2.0]);
    let v = compare_prompt(&tie, &flipped, &top(&tie), &tol);
    assert!(v.passed, "{v:?}");

    // Stopping early diverges at the candidate's length.
    let v = compare_prompt(&exact, &exact.tokens[..3], &top(&exact), &tol);
    assert_eq!(v.first_divergence, Some(3));
    assert!(!v.passed);

    // A shifted top-k logprob or a missing top-k id violates the bound; rank 3 is not checked.
    let mut shifted = top(&exact);
    shifted[1][0].1 += 0.2;
    let v = compare_prompt(&exact, &exact.tokens, &shifted, &tol);
    assert!(!v.logprob_within_bound && !v.passed, "{v:?}");
    let mut missing = top(&exact);
    missing[3].retain(|e| e.0 != 23);
    let v = compare_prompt(&exact, &exact.tokens, &missing, &tol);
    assert_eq!(v.missing_top_k, Some((3, 23)));
    let mut rank3 = top(&exact);
    rank3[0][2].1 += 5.0;
    let v = compare_prompt(&exact, &exact.tokens, &rank3, &tol);
    assert!(v.passed, "{v:?}");
}

// ------------------------------------------------------------------------ replay helpers

/// Prompt token ids exactly as the server builds them: chat prompts through the model's chat
/// template (`add_generation_prompt`, the prompt's `chat_template_kwargs`) tokenized without
/// special tokens (the template emits BOS); completion prompts tokenized with them.
fn prompt_token_ids(
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    prompt: &PromptRecord,
) -> Vec<u32> {
    match prompt.kind.as_str() {
        "completion" => {
            let text = prompt.prompt.as_deref().expect("completion prompt text");
            tokenizer.encode(text, true).expect("encode")
        }
        "chat" => {
            let messages = prompt.messages.as_deref().expect("chat messages");
            let text = template
                .render(messages, None, true, &prompt.chat_template_kwargs)
                .unwrap_or_else(|e| panic!("{}: render: {e}", prompt.id));
            tokenizer.encode(&text, false).expect("encode")
        }
        other => panic!("{}: unknown kind {other}", prompt.id),
    }
}

/// Greedy generation through `turbine_model::generate` for exactly `max_tokens` positions
/// (`ignore_eos`, as `turbine-golden compare` requests), with the top-20 raw logprobs per
/// position.
fn greedy(
    exec: &mut dyn ModelExecutor,
    tokenizer: &Arc<Tokenizer>,
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    max_seq_len: u32,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    let req = GenerationRequest {
        id: RequestId::new_v4(),
        endpoint: Endpoint::Completions,
        http_request_id: "golden".into(),
        prompt_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            seed: Some(0),
            logprobs: Some(TOP_LOGPROBS),
        },
        stop: StopConditions {
            eos_token_ids: SmallVec::new(),
            stop_strings: Vec::new(),
            max_tokens,
            ignore_eos: true,
        },
    };
    let cancel = CancelFlag::default();
    let opts = GenerateOptions {
        max_seq_len,
        metrics: None,
    };
    let mut tokens = Vec::new();
    let mut tops = Vec::new();
    let mut finished = None;
    for event in generate(exec, Arc::clone(tokenizer), &req, &cancel, opts) {
        match event {
            GenerationEvent::Token {
                token_id,
                top_logprobs,
                ..
            } => {
                tokens.push(token_id);
                tops.push(top_logprobs);
            }
            GenerationEvent::Finished { reason, .. } => finished = Some(reason),
            GenerationEvent::Error { code, message } => {
                panic!("generation failed: {code:?}: {message}")
            }
            _ => {}
        }
    }
    assert_eq!(
        finished,
        Some(FinishReason::Length),
        "generation must run to max_tokens"
    );
    (tokens, tops)
}

/// Replays every prompt on `exec`, asserting Turbine's prompt ids equal the reference's, and
/// returns one verdict per prompt.
fn replay(
    exec: &mut dyn ModelExecutor,
    model_dir: &Path,
    prompts: &[PromptRecord],
    references: &[ReferenceRecord],
    tol: &Tolerance,
    max_seq_len: u32,
) -> Vec<Verdict> {
    let tokenizer =
        Arc::new(Tokenizer::from_file(&model_dir.join("tokenizer.json")).expect("tokenizer"));
    let template = ChatTemplate::resolve(model_dir, None).expect("chat template");
    assert_eq!(
        prompts.len(),
        references.len(),
        "prompts.jsonl and reference.jsonl differ in length"
    );
    let mut verdicts = Vec::with_capacity(prompts.len());
    for (prompt, reference) in prompts.iter().zip(references) {
        assert_eq!(prompt.id, reference.id, "prompt and reference order differ");
        let ids = prompt_token_ids(&tokenizer, &template, prompt);
        assert_eq!(
            ids, reference.prompt_token_ids,
            "{}: Turbine's prompt token ids differ from the reference",
            prompt.id
        );
        assert_eq!(
            reference.tokens.len(),
            prompt.max_tokens as usize,
            "{}",
            prompt.id
        );
        let (tokens, tops) = greedy(exec, &tokenizer, ids, prompt.max_tokens, max_seq_len);
        if std::env::var_os("TURBINE_GOLDEN_DUMP").is_some() {
            // Diagnostics: the candidate in reference.jsonl shape, one line per prompt.
            let line = serde_json::json!({
                "id": prompt.id,
                "tokens": tokens,
                "top_logprobs": tops,
            });
            println!("golden-candidate {line}");
        }
        verdicts.push(compare_prompt(reference, &tokens, &tops, tol));
    }
    verdicts
}

/// The context every prompt needs: its prompt ids plus `max_tokens`.
fn needed_seq_len(references: &[ReferenceRecord]) -> u32 {
    references
        .iter()
        .map(|r| (r.prompt_token_ids.len() + r.tokens.len()) as u32)
        .max()
        .expect("at least one reference")
}

fn build_executor(
    model_dir: &Path,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> LlamaExecutor {
    let cfg = load_model_config(model_dir).expect("config.json");
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let weights = WeightLoader::load(&index, &llama_slots(&cfg), &mem, MAX_STAGING_BYTES)
        .expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &LlamaExecutor::requirements(&cfg),
        &metrics,
    )
    .expect("every op has a provider");
    LlamaExecutor::new(
        &cfg,
        weights,
        Arc::new(registry),
        mem,
        max_seq_len,
        max_seq_len,
    )
    .expect("executor")
}

// --------------------------------------------------------------------------------- tests

/// Lab only (Task 21, needs `uv` on PATH): writes the tiny checkpoint, generates its reference
/// with `hf_reference.py`, and replays every committed prompt on the CPU executor.
#[test]
#[ignore = "needs uv (runs scripts/golden/hf_reference.py)"]
fn hf_reference_matches_cpu() {
    let tmp = TempDir::new("golden-tiny");
    let model_dir = tmp.path().join("tiny");
    write_tiny_llama(&model_dir, 7);
    let reference_path = tmp.path().join("reference.jsonl");
    let prompts_path = golden_dir().join("prompts.jsonl");
    let status = Command::new("uv")
        .current_dir(repo_root())
        .args(["run", "scripts/golden/hf_reference.py", "--model-dir"])
        .arg(&model_dir)
        .arg("--prompts")
        .arg(&prompts_path)
        .arg("--out")
        .arg(&reference_path)
        .status()
        .expect("spawn `uv` (install it: https://docs.astral.sh/uv/)");
    assert!(status.success(), "hf_reference.py failed: {status}");

    let prompts: Vec<PromptRecord> = read_jsonl(&prompts_path);
    let references: Vec<ReferenceRecord> = read_jsonl(&reference_path);
    let tol = Tolerance {
        max_abs_logprob_diff: TINY_MAX_ABS_LOGPROB_DIFF,
        ..read_tolerance(&golden_dir().join("llama-3.2-3b-instruct/tolerance.json"))
    };
    let max_seq_len = needed_seq_len(&references);
    let mem = HostMemory::new(turbine_core::types::DeviceId(0), 1 << 30);
    let mut exec = build_executor(&model_dir, cpu_reference_provider(), mem, max_seq_len);
    let verdicts = replay(
        &mut exec,
        &model_dir,
        &prompts,
        &references,
        &tol,
        max_seq_len,
    );
    assert_tolerance(&verdicts, &tol);
}

/// Lab only (Task 21): Llama-3.2-3B-Instruct on the HIP provider against the committed
/// transformers reference.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MODEL_DIR"]
fn logits_match_reference() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
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

    let fixture = golden_dir().join("llama-3.2-3b-instruct");
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    // Diagnostics: TURBINE_GOLDEN_REFERENCE names another reference file in the fixture
    // directory (e.g. one generated with `hf_reference.py --fp32-logits`).
    let reference_file =
        std::env::var("TURBINE_GOLDEN_REFERENCE").unwrap_or_else(|_| "reference.jsonl".into());
    println!("reference: {reference_file}");
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join(&reference_file));
    let tol = read_tolerance(&fixture.join("tolerance.json"));
    let max_seq_len = needed_seq_len(&references);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut exec = build_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    let verdicts = replay(
        &mut exec,
        &model_dir,
        &prompts,
        &references,
        &tol,
        max_seq_len,
    );
    assert_tolerance(&verdicts, &tol);
}

/// Lab diagnostic (precision investigation), a no-op unless `TURBINE_GOLDEN_TRACE` names
/// prompts (comma-separated ids, e.g. `p01,p03`); run it alone in a release build, the
/// cpu-reference executor on the 3B model is scalar. For each named prompt, Llama-3.2-3B on the
/// HIP and the cpu-reference providers runs the reference's prompt ids and then its first
/// `TRACE_DECODE` tokens (teacher-forced, so both see identical inputs), traced; prints the
/// op-by-op HIP-vs-CPU table (accumulated divergence) and every HIP op recomputed on the host
/// from the HIP trace's own inputs (the error each op adds).
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY, TURBINE_TEST_MODEL_DIR and TURBINE_GOLDEN_TRACE"]
fn hip_trace_vs_cpu_3b() {
    const TRACE_DECODE: usize = 2;
    let Some(ids) = std::env::var("TURBINE_GOLDEN_TRACE")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        println!("TURBINE_GOLDEN_TRACE is not set: nothing traced");
        return;
    };
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
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

    let fixture = golden_dir().join("llama-3.2-3b-instruct");
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join("reference.jsonl"));
    let selected: Vec<&ReferenceRecord> = ids
        .split(',')
        .map(|id| {
            references
                .iter()
                .find(|r| r.id == id.trim())
                .unwrap_or_else(|| panic!("no reference prompt {id}"))
        })
        .collect();
    let max_seq_len = selected
        .iter()
        .map(|r| (r.prompt_token_ids.len() + TRACE_DECODE) as u32)
        .max()
        .expect("at least one prompt");
    let cfg = load_model_config(&model_dir).expect("config.json");
    let index = SafetensorsIndex::open(&model_dir).expect("open safetensors");
    let host = HostMemory::new(turbine_core::types::DeviceId(0), 16 << 30);
    let mut cpu = build_executor(&model_dir, cpu_reference_provider(), host, max_seq_len);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = build_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    cpu.set_trace(true);
    hip.set_trace(true);
    let weights = |name: &str| read_bf16_weight(&index, name);

    for r in selected {
        let mut accumulated = Vec::new();
        let mut local = Vec::new();
        let mut checker = LocalChecker::new(&cfg, &weights);
        let mut batch = r.prompt_token_ids.clone();
        let mut p0 = 0usize;
        for step in 0..=TRACE_DECODE {
            let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
            let input = BatchInput {
                tokens: &batch,
                positions: &positions,
            };
            cpu.forward(&input).expect("cpu forward");
            hip.forward(&input).expect("hip forward");
            let (c, h) = (cpu.take_trace(), hip.take_trace());
            accumulated.extend(compare_traces(step, &c, &h));
            local.extend(checker.check_step(step, p0, &h));
            p0 += batch.len();
            batch = vec![r.tokens[step]];
        }
        println!(
            "{}",
            render(
                &format!("3B {}: HIP vs cpu-reference (accumulated)", r.id),
                &accumulated
            )
        );
        println!(
            "{}",
            render(
                &format!(
                    "3B {}: HIP op vs host recompute from its own inputs (local)",
                    r.id
                ),
                &local
            )
        );
    }
}
