//! In-process golden tests (P1 S-11): Turbine's greedy generation replayed against a Hugging
//! Face transformers reference produced by `scripts/golden/hf_reference.py`.
//!
//! - `hf_reference_matches_cpu` (lab only, needs `uv`): the tiny synthetic checkpoint on the
//!   `cpu-reference` provider against a reference generated on the spot.
//! - `logits_match_reference` (lab only, needs a HIP device and the weights): Llama-3.2-3B on the
//!   HIP provider against the committed `tests/golden/llama-3.2-3b-instruct/reference.jsonl`.
//! - `olmoe_logits_match_reference` (lab only, likewise): OLMoE-1B-7B on the HIP provider against
//!   `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl` under its own calibrated
//!   `tolerance.json` (see the README beside it).
//! - `hip_trace_vs_cpu_3b` and `olmoe_teacher_forced_vs_reference` (lab diagnostics, no-ops
//!   unless their environment variable names prompts): op-by-op HIP-vs-CPU traces of the 3B,
//!   and teacher-forced OLMoE log-probabilities of both providers against its reference.
//!
//! The first three judge with the committed `tolerance.json` and the same rule as `turbine-golden compare`,
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
use turbine_core::types::RequestId;
use turbine_kernels::{
    KernelMetrics, KernelProvider, KernelRegistry, cpu_reference_provider, shim_provider,
};
use turbine_model::executor::{
    self, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, SequenceKv,
};
use turbine_model::families;
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
#[serde(deny_unknown_fields)]
struct Tolerance {
    min_identical_prefix: usize,
    min_prompts_passing: usize,
    top_k: usize,
    /// Bound for reference top-k candidates with reference logprob > `likely_logprob_floor`.
    max_abs_logprob_diff_likely: f32,
    /// Bound for the other (tail) reference top-k candidates.
    max_abs_logprob_diff_tail: f32,
    likely_logprob_floor: f32,
    margin_nats: f32,
    /// Bounds `turbine-golden compare` applies above concurrency 1 only; these tests generate
    /// one sequence at a time and judge by the strict bounds.
    #[serde(default)]
    #[allow(dead_code)]
    max_abs_logprob_diff_likely_batched: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    max_abs_logprob_diff_tail_batched: Option<f32>,
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
    max_abs_logprob_diff_likely: f32,
    max_abs_logprob_diff_tail: f32,
    /// `(position, token id)` of the first reference top-k id missing from Turbine's top list.
    missing_top_k: Option<(usize, u32)>,
    logprob_within_bound: bool,
    passed: bool,
}

/// The `turbine-golden compare` rule for one prompt: the greedy prefix must be at least
/// `min_identical_prefix` long unless the reference margin (top-1 minus top-2) at the first
/// divergence is below `margin_nats`; at every position before the divergence each reference
/// top-k id must appear in Turbine's top list, within `max_abs_logprob_diff_likely` when its
/// reference logprob is above `likely_logprob_floor` and within `max_abs_logprob_diff_tail`
/// otherwise (at the floor exactly counts as tail). A candidate that stops early diverges at its
/// length.
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

    let mut max_abs_logprob_diff_likely = 0f32;
    let mut max_abs_logprob_diff_tail = 0f32;
    let mut missing_top_k = None;
    for pos in 0..first_divergence.unwrap_or(n) {
        let Some(ref_row) = reference.top_logprobs.get(pos) else {
            break;
        };
        let got_row = got_top.get(pos).map(Vec::as_slice).unwrap_or(&[]);
        for &(id, lp) in sorted(ref_row).iter().take(tol.top_k) {
            match got_row.iter().find(|e| e.0 == id) {
                Some(&(_, got_lp)) => {
                    let tier = if lp > tol.likely_logprob_floor {
                        &mut max_abs_logprob_diff_likely
                    } else {
                        &mut max_abs_logprob_diff_tail
                    };
                    *tier = tier.max((got_lp - lp).abs());
                }
                None => {
                    missing_top_k.get_or_insert((pos, id));
                }
            }
        }
    }
    let logprob_within_bound = missing_top_k.is_none()
        && max_abs_logprob_diff_likely <= tol.max_abs_logprob_diff_likely
        && max_abs_logprob_diff_tail <= tol.max_abs_logprob_diff_tail;
    let prefix_ok = identical_prefix >= tol.min_identical_prefix.min(n);
    let excused = margin_at_divergence.is_some_and(|m| m < tol.margin_nats);
    Verdict {
        id: reference.id.clone(),
        reference_len: n,
        identical_prefix,
        first_divergence,
        margin_at_divergence,
        max_abs_logprob_diff_likely,
        max_abs_logprob_diff_tail,
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
             max_abs_logprob_diff_likely={:.4} max_abs_logprob_diff_tail={:.4} \
             missing_top_k={:?}\n",
            if v.passed { "PASS" } else { "FAIL" },
            v.id,
            v.identical_prefix,
            v.reference_len,
            v.first_divergence,
            v.margin_at_divergence,
            v.max_abs_logprob_diff_likely,
            v.max_abs_logprob_diff_tail,
            v.missing_top_k,
        ));
    }
    println!("{report}");
    let passing = verdicts.iter().filter(|v| v.passed).count();
    assert!(!verdicts.is_empty(), "no prompts compared");
    assert!(
        verdicts.iter().all(|v| v.logprob_within_bound),
        "|Δ logprob| over the reference top-{} exceeds {} (reference logprob > {}) or {} \
         (tail), or an id is missing:\n{report}",
        tol.top_k,
        tol.max_abs_logprob_diff_likely,
        tol.likely_logprob_floor,
        tol.max_abs_logprob_diff_tail,
    );
    println!(
        "golden verdict: {passing}/{} prompts passing (need {}); logprob bound held on every prompt",
        verdicts.len(),
        tol.min_prompts_passing,
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
        max_abs_logprob_diff_likely: 0.15,
        max_abs_logprob_diff_tail: 0.55,
        likely_logprob_floor: -2.0,
        margin_nats: 0.5,
        max_abs_logprob_diff_likely_batched: None,
        max_abs_logprob_diff_tail_batched: None,
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

    // A shifted likely top-k logprob (reference −1.0) or a missing top-k id violates the bound;
    // the tail runner-up (reference −3.0) gets 0.55; rank 3 is not checked.
    let mut shifted = top(&exact);
    shifted[1][1].1 += 0.2;
    let v = compare_prompt(&exact, &exact.tokens, &shifted, &tol);
    assert!(!v.logprob_within_bound && !v.passed, "{v:?}");
    let mut tail = top(&exact);
    tail[1][0].1 += 0.5;
    let v = compare_prompt(&exact, &exact.tokens, &tail, &tol);
    assert!(v.passed, "{v:?}");
    tail[1][0].1 += 0.1;
    let v = compare_prompt(&exact, &exact.tokens, &tail, &tol);
    assert!(!v.logprob_within_bound, "{v:?}");
    let mut missing = top(&exact);
    missing[3].retain(|e| e.0 != 23);
    let v = compare_prompt(&exact, &exact.tokens, &missing, &tol);
    assert_eq!(v.missing_top_k, Some((3, 23)));
    let mut rank3 = top(&exact);
    rank3[0][2].1 += 5.0;
    let v = compare_prompt(&exact, &exact.tokens, &rank3, &tol);
    assert!(v.passed, "{v:?}");
}

/// The same boundary test as `turbine-golden compare`: the tier is chosen by the reference
/// logprob; just above −2 is likely (0.15), −2 exactly and below are tail (0.55). Breaks if −2
/// counts as likely, if the candidate's logprob picks the tier, or if one bound covers both.
#[test]
fn logprob_bound_has_a_likely_and_a_tail_tier() {
    let tol = Tolerance {
        min_identical_prefix: 1,
        min_prompts_passing: 1,
        top_k: 2,
        max_abs_logprob_diff_likely: 0.15,
        max_abs_logprob_diff_tail: 0.55,
        likely_logprob_floor: -2.0,
        margin_nats: 0.5,
        max_abs_logprob_diff_likely_batched: None,
        max_abs_logprob_diff_tail_batched: None,
    };
    // One position: top-1 token 10, runner-up 20 at `ref_lp`, moved by `delta` in the candidate.
    let check = |ref_lp: f32, delta: f32| {
        let r = ReferenceRecord {
            id: "p".into(),
            prompt_token_ids: vec![1],
            tokens: vec![10],
            top_logprobs: vec![vec![(10, -0.1), (20, ref_lp)]],
        };
        let mut top = r.top_logprobs.clone();
        top[0][1].1 += delta;
        compare_prompt(&r, &r.tokens, &top, &tol)
    };
    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;

    let v = check(-1.99, 0.1);
    assert!(v.logprob_within_bound && v.passed, "{v:?}");
    assert!(close(v.max_abs_logprob_diff_likely, 0.1), "{v:?}");
    assert_eq!(v.max_abs_logprob_diff_tail, 0.0);
    let v = check(-1.99, 0.5);
    assert!(!v.logprob_within_bound && !v.passed, "{v:?}");
    for ref_lp in [-2.0, -2.01] {
        let v = check(ref_lp, 0.5);
        assert!(v.logprob_within_bound && v.passed, "{ref_lp}: {v:?}");
        assert!(close(v.max_abs_logprob_diff_tail, 0.5), "{v:?}");
        assert_eq!(v.max_abs_logprob_diff_likely, 0.0);
        let v = check(ref_lp, 0.6);
        assert!(!v.logprob_within_bound && !v.passed, "{ref_lp}: {v:?}");
    }
    let v = check(-1.99, -0.5);
    assert!(!v.logprob_within_bound, "{v:?}");
}

/// The committed tolerance is the two-tier rule of the user decision 2026-09-26 and the
/// committed reference is the FP32-final-logit transformers reference.
#[test]
fn committed_tolerance_and_reference() {
    let fixture = golden_dir().join("llama-3.2-3b-instruct");
    let tol = read_tolerance(&fixture.join("tolerance.json"));
    assert_eq!(
        (tol.min_identical_prefix, tol.min_prompts_passing, tol.top_k),
        (32, 14, 5)
    );
    assert_eq!(
        (
            tol.max_abs_logprob_diff_likely,
            tol.max_abs_logprob_diff_tail,
            tol.likely_logprob_floor,
            tol.margin_nats
        ),
        (0.15, 0.55, -2.0, 0.5)
    );
    let text = std::fs::read_to_string(fixture.join("reference.jsonl")).expect("reference");
    let engines: Vec<String> = text
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("json")["engine"].to_string())
        .collect();
    assert_eq!(engines.len(), 16);
    assert!(
        engines.iter().all(|e| e.ends_with("-fp32-logits\"")),
        "{engines:?}"
    );
}

/// OLMoE's committed tolerance is transformers' own spread (decision "OLMoE golden gate",
/// 2026-09-26; measurement in the README beside it): the same rule and floor as Llama, bounds
/// 1.01 / 1.66 that every transformers variant (sdpa/eager × BF16/FP32 × incremental/full)
/// meets against the reference, at least 14/16 prompts. Breaks if the OLMoE file drifts from
/// the documented calibration or Llama's file is loosened with it.
#[test]
fn olmoe_tolerance_is_the_calibrated_self_spread() {
    let fixture = golden_dir().join("olmoe-1b-7b-0125-instruct");
    let tol = read_tolerance(&fixture.join("tolerance.json"));
    assert_eq!(
        (tol.min_identical_prefix, tol.min_prompts_passing, tol.top_k),
        (32, 14, 5)
    );
    assert_eq!(
        (
            tol.max_abs_logprob_diff_likely,
            tol.max_abs_logprob_diff_tail,
            tol.likely_logprob_floor,
            tol.margin_nats
        ),
        (1.01, 1.66, -2.0, 0.5)
    );
    // The spread already spans GEMM shapes and kernels, so concurrency 16 gets the same bounds.
    assert_eq!(
        (
            tol.max_abs_logprob_diff_likely_batched,
            tol.max_abs_logprob_diff_tail_batched
        ),
        (Some(1.01), Some(1.66))
    );
    let readme = std::fs::read_to_string(fixture.join("README.md")).expect("README.md");
    for bound in ["1.01", "1.66", "1.0092", "1.6569"] {
        assert!(readme.contains(bound), "README.md lacks {bound}");
    }
    let llama = read_tolerance(&golden_dir().join("llama-3.2-3b-instruct/tolerance.json"));
    assert_eq!(
        (
            llama.max_abs_logprob_diff_likely,
            llama.max_abs_logprob_diff_tail
        ),
        (0.15, 0.55)
    );
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
    kv: &mut SequenceKv,
    tokenizer: &Arc<Tokenizer>,
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    max_seq_len: u32,
) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
    let req = GenerationRequest {
        id: RequestId::new_v4(),
        n: 1,
        priority: turbine_core::types::Priority::default(),
        echo: false,
        constraint: None,
        deadline_ms: u64::MAX,
        endpoint: Endpoint::Completions,
        http_request_id: "golden".into(),
        prompt_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            seed: Some(0),
            logprobs: Some(TOP_LOGPROBS),
            ..SamplingParams::default()
        },
        stop: StopConditions {
            eos_token_ids: SmallVec::new(),
            stop_strings: Vec::new(),
            max_tokens,
            ignore_eos: true,
            ..StopConditions::default()
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
    for event in generate(exec, kv, Arc::clone(tokenizer), &req, &cancel, opts) {
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
    kv: &mut SequenceKv,
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
        let (tokens, tops) = greedy(exec, kv, &tokenizer, ids, prompt.max_tokens, max_seq_len);
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

/// KV block size of the golden runs: the `kv.block_tokens` default (CK paged attention on HIP).
const BLOCK_TOKENS: u32 = 128;

/// Held by every test that loads a 3B-class model onto the GPU, so the test harness's threads
/// never hold Llama-3.2-3B and OLMoE-1B-7B (and their KV) on one card at once.
static GPU_MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_model_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

/// An executor and the single-sequence KV it generates on.
struct Runner {
    exec: DecoderExecutor,
    kv: SequenceKv,
}

fn build_executor(
    model_dir: &Path,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> Runner {
    let cfg = load_model_config(model_dir).expect("config.json");
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let weights = WeightLoader::load(&index, &llama_slots(&cfg), &mem, MAX_STAGING_BYTES)
        .expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, ExecutorOptions::default()),
        &metrics,
    )
    .expect("every op has a provider");
    let kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
    let exec = DecoderExecutor::new(
        &cfg,
        families::llama::decoder_spec(),
        weights,
        Arc::new(registry),
        mem,
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: max_seq_len,
            max_seqs: 1,
        },
        ExecutorOptions::default(),
    )
    .expect("executor");
    Runner { exec, kv }
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
    // The committed Llama tolerance applies unchanged. Measured (transformers 4.57.1 CPU,
    // FP32-logit reference, CK-order BF16 attention probabilities in the cpu-reference
    // provider): 16/16 prompts with all 32 greedy tokens identical, max |Δ logprob| 0.0041 on
    // likely and 0.1154 on tail candidates.
    let tol = read_tolerance(&golden_dir().join("llama-3.2-3b-instruct/tolerance.json"));
    let max_seq_len = needed_seq_len(&references);
    let mem = HostMemory::new(turbine_core::types::DeviceId(0), 1 << 30);
    let mut runner = build_executor(&model_dir, cpu_reference_provider(), mem, max_seq_len);
    let verdicts = replay(
        &mut runner.exec,
        &mut runner.kv,
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
    let _gpu = gpu_model_lock();
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");

    let fixture = golden_dir().join("llama-3.2-3b-instruct");
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join("reference.jsonl"));
    let tol = read_tolerance(&fixture.join("tolerance.json"));
    let max_seq_len = needed_seq_len(&references);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut runner = build_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    let verdicts = replay(
        &mut runner.exec,
        &mut runner.kv,
        &model_dir,
        &prompts,
        &references,
        &tol,
        max_seq_len,
    );
    assert_tolerance(&verdicts, &tol);
}

/// Lab only: OLMoE-1B-7B-0125-Instruct on the HIP provider against the committed transformers
/// reference, judged with OLMoE's calibrated `tolerance.json` (transformers' own spread across
/// attention kernel, compute precision and decode shape; README beside it).
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MOE_MODEL_DIR"]
fn olmoe_logits_match_reference() {
    let _gpu = gpu_model_lock();
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");

    let fixture = golden_dir().join("olmoe-1b-7b-0125-instruct");
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join("reference.jsonl"));
    let tol = read_tolerance(&fixture.join("tolerance.json"));
    let max_seq_len = needed_seq_len(&references);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut runner = build_any_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    let verdicts = replay(
        runner.exec.as_mut(),
        &mut runner.kv,
        &model_dir,
        &prompts,
        &references,
        &tol,
        max_seq_len,
    );
    for v in &verdicts {
        println!("olmoe_golden {v:?}");
    }
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
    let _gpu = gpu_model_lock();
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
    let ctx = turbine_kernels::test_support::open_context("hip");

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
    cpu.exec.set_trace(true);
    hip.exec.set_trace(true);
    let weights = |name: &str| read_bf16_weight(&index, name);

    for r in selected {
        let mut accumulated = Vec::new();
        let mut local = Vec::new();
        let mut checker = LocalChecker::new(&cfg, &weights);
        let mut batch = r.prompt_token_ids.clone();
        let mut p0 = 0usize;
        for step in 0..=TRACE_DECODE {
            let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
            for r in [&mut cpu, &mut hip] {
                r.kv.forward(&mut r.exec, &batch, &positions)
                    .expect("forward");
            }
            let (c, h) = (cpu.exec.take_trace(), hip.exec.take_trace());
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

// ------------------------------------------------------------------------------- OLMoE

/// An executor of any architecture (through [`executor::build_executor`]) and the
/// single-sequence KV it runs on.
struct AnyRunner {
    exec: Box<dyn ModelExecutor>,
    kv: SequenceKv,
}

fn build_any_executor(
    model_dir: &Path,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> AnyRunner {
    let cfg = load_model_config(model_dir).expect("config.json");
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let slots = cfg.family.0.weight_slots(&cfg);
    let weights =
        WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, ExecutorOptions::default()),
        &metrics,
    )
    .expect("every op has a provider");
    let kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
    let exec = executor::build_executor(
        &cfg,
        weights,
        Arc::new(registry),
        mem,
        BLOCK_TOKENS,
        max_seq_len,
        1,
        ExecutorOptions::default(),
    )
    .expect("executor");
    AnyRunner { exec, kv }
}

/// FP32 log-softmax of one logits row (accumulated in f64).
fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&l| f64::from(l - max).exp()).sum();
    let lse = f64::from(max) + sum.ln();
    row.iter().map(|&l| (f64::from(l) - lse) as f32).collect()
}

/// One teacher-forced position: Turbine's argmax and the largest |Δ logprob| over the
/// reference's top-5 (likely and tail tiers as in `tolerance.json`).
#[derive(Debug)]
struct ForcedPosition {
    argmax: u32,
    likely: f32,
    tail: f32,
}

/// Runs `r`'s prompt and then its reference tokens (teacher-forced: every position sees the
/// reference's own prefix, so one bad position does not cascade) and compares each position's
/// log-probabilities with the reference's top-5.
fn teacher_forced(runner: &mut AnyRunner, r: &ReferenceRecord, floor: f32) -> Vec<ForcedPosition> {
    let mut out = Vec::with_capacity(r.tokens.len());
    let mut batch = r.prompt_token_ids.clone();
    let mut p0 = 0usize;
    for step in 0..r.tokens.len() {
        let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
        let logits = runner
            .kv
            .forward(runner.exec.as_mut(), &batch, &positions)
            .expect("forward");
        let lp = log_softmax(logits.row(0));
        let argmax = (0..lp.len())
            .max_by(|&a, &b| lp[a].total_cmp(&lp[b]).then(b.cmp(&a)))
            .expect("non-empty vocab") as u32;
        let (mut likely, mut tail) = (0f32, 0f32);
        for &(id, ref_lp) in r.top_logprobs[step].iter().take(5) {
            let d = (lp[id as usize] - ref_lp).abs();
            if ref_lp > floor {
                likely = likely.max(d);
            } else {
                tail = tail.max(d);
            }
        }
        out.push(ForcedPosition {
            argmax,
            likely,
            tail,
        });
        p0 += batch.len();
        batch = vec![r.tokens[step]];
    }
    out
}

/// Lab diagnostic (OLMoE numerics vs semantics), a no-op unless `TURBINE_GOLDEN_OLMOE` names
/// prompts (comma-separated ids); run it alone in a release build (the cpu-reference provider
/// is scalar). For each named prompt, OLMoE-1B-7B on the cpu-reference and the HIP provider
/// runs the reference's prompt and then its tokens teacher-forced, and prints per position the
/// largest |Δ logprob| against the transformers reference's top-5 for both providers.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY, TURBINE_TEST_MOE_MODEL_DIR and TURBINE_GOLDEN_OLMOE"]
fn olmoe_teacher_forced_vs_reference() {
    let Some(ids) = std::env::var("TURBINE_GOLDEN_OLMOE")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        println!("TURBINE_GOLDEN_OLMOE is not set: nothing compared");
        return;
    };
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let _gpu = gpu_model_lock();
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");

    let fixture = golden_dir().join("olmoe-1b-7b-0125-instruct");
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join("reference.jsonl"));
    let tol = read_tolerance(&fixture.join("tolerance.json"));
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
        .map(|r| (r.prompt_token_ids.len() + r.tokens.len()) as u32)
        .max()
        .expect("at least one prompt");
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = build_any_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    let host = HostMemory::new(turbine_core::types::DeviceId(0), 24 << 30);
    let mut cpu = build_any_executor(&model_dir, cpu_reference_provider(), host, max_seq_len);
    for r in selected {
        let h = teacher_forced(&mut hip, r, tol.likely_logprob_floor);
        let worst = |v: &[ForcedPosition]| {
            v.iter()
                .fold((0f32, 0f32), |(l, t), p| (l.max(p.likely), t.max(p.tail)))
        };
        println!("OLMoE {} hip max {:?}", r.id, worst(&h));
        let c = teacher_forced(&mut cpu, r, tol.likely_logprob_floor);
        println!(
            "OLMoE {} teacher-forced vs transformers (|Δ| likely / tail):",
            r.id
        );
        println!("pos  ref_tok  cpu_argmax cpu_likely cpu_tail  hip_argmax hip_likely hip_tail");
        for (pos, (c, h)) in c.iter().zip(&h).enumerate() {
            println!(
                "{pos:>3} {:>8} {:>10} {:>10.4} {:>8.4} {:>11} {:>10.4} {:>8.4}",
                r.tokens[pos], c.argmax, c.likely, c.tail, h.argmax, h.likely, h.tail
            );
        }
        println!(
            "OLMoE {} max: cpu {:?}, hip {:?}",
            r.id,
            worst(&c),
            worst(&h)
        );
    }
}
