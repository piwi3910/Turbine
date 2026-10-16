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
//! - `hip_trace_vs_cpu_3b`, `olmoe_teacher_forced_vs_reference` and
//!   `mxfp4_decode_vs_prefill_trace` (lab diagnostics, no-ops unless their environment variable
//!   names prompts): op-by-op HIP-vs-CPU traces of the 3B, teacher-forced OLMoE
//!   log-probabilities of both providers against its reference, and the MXFP4-A16 8B's decode
//!   steps against one prefill of the same tokens, op by op.
//!
//! - `hip_trace_vs_cpu_qwen3moe` (lab, two GPUs, release): Qwen3-30B-A3B FP8's first 4 layers at
//!   tensor parallelism 2 on HIP against the cpu-reference executor, op by op (P7 S-5).
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
    self, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, SequenceKv, TraceTensor,
};
use turbine_model::families;
use turbine_model::generate::{GenerateOptions, generate};
use turbine_model::testing::TempDir;
use turbine_model::testing::tiny::write_tiny_llama;
use turbine_model::testing::trace::{
    LocalChecker, RoutingRow, compare_traces, read_bf16_weight, render,
};
use turbine_model::{
    ChatTemplate, MAX_STAGING_BYTES, SafetensorsIndex, Tokenizer, WeightLoader, llama_slots,
    load_model_config, load_model_config_with,
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
    /// Bounds for an MoE model over several GPUs (`turbine-golden compare --multi-gpu-bounds`).
    #[serde(default)]
    #[allow(dead_code)]
    max_abs_logprob_diff_likely_multi_gpu: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    max_abs_logprob_diff_tail_multi_gpu: Option<f32>,
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
        max_abs_logprob_diff_likely_multi_gpu: None,
        max_abs_logprob_diff_tail_multi_gpu: None,
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
        max_abs_logprob_diff_likely_multi_gpu: None,
        max_abs_logprob_diff_tail_multi_gpu: None,
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
        (0.15, 0.56, -2.0, 0.5)
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
/// 1.01 / 1.66 (tail 1.72 since decision "5p golden tolerance vs engine summation order" A) that every transformers variant (sdpa/eager × BF16/FP32 × incremental/full)
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
        (1.01, 1.72, -2.0, 0.5)
    );
    // The spread already spans GEMM shapes and kernels, so concurrency 16 gets the same bounds.
    assert_eq!(
        (
            tol.max_abs_logprob_diff_likely_batched,
            tol.max_abs_logprob_diff_tail_batched
        ),
        (Some(1.01), Some(1.72))
    );
    let readme = std::fs::read_to_string(fixture.join("README.md")).expect("README.md");
    for bound in ["1.01", "1.66", "1.72", "1.0092", "1.6569"] {
        assert!(readme.contains(bound), "README.md lacks {bound}");
    }
    let llama = read_tolerance(&golden_dir().join("llama-3.2-3b-instruct/tolerance.json"));
    assert_eq!(
        (
            llama.max_abs_logprob_diff_likely,
            llama.max_abs_logprob_diff_tail
        ),
        (0.15, 0.56)
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
        session: None,
        cache_salt: None,
        kv_policy: None,
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

/// A traceable executor of a dense model (Llama or Qwen3) in `model_dir`.
fn build_executor(
    model_dir: &Path,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> Runner {
    let cfg = load_model_config(model_dir).expect("config.json");
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let weights = WeightLoader::load_format(
        cfg.weight_format.get(),
        &index,
        &cfg.family.0.weight_slots(&cfg),
        &mem,
        MAX_STAGING_BYTES,
    )
    .expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let card = provider.card_profile();
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, ExecutorOptions::default()),
        &metrics,
        card,
    )
    .expect("every op has a provider");
    let kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
    let spec = match cfg.family.0.name() {
        "llama" => families::llama::decoder_spec(),
        "qwen3" => families::qwen3::decoder_spec(),
        other => panic!("no traced decoder for family {other}"),
    };
    let exec = DecoderExecutor::new(
        &cfg,
        spec,
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

/// Largest BF16-ulp error a HIP per-head Q/K norm may add over its host recompute from the HIP
/// trace's own input ([`LocalChecker`]: `bf(bf(x · rsqrt) · w)`): an f32 `rsqrt` computed in
/// another summation order can flip the first BF16 rounding, which the second (the weight
/// product) can carry one more ulp. Calibration (lab, R9700, p7-t10, p01 prefill + 2 decodes):
/// 1 ulp on most layers, 2 on one element of 14,336 (layer 18 `k_norm`, |Δ| 4.9e-4).
const QWEN3_QK_NORM_MAX_ULPS: f32 = 2.0;

/// Lab only (P7 S-2; needs `TURBINE_TEST_QWEN3_MODEL_DIR` = Qwen3-4B-Instruct-2507, which the
/// lab test Job sets to `/models/qwen3-4b-instruct-2507`): a no-op in a debug build (the
/// cpu-reference executor on the 4B model is scalar; run it with `--release`). Prompt p01 of
/// `tests/golden/prompts.jsonl` on the HIP and the cpu-reference providers, the prefill and 2
/// decode steps teacher-forced with the cpu-reference's greedy tokens, traced. Prints the
/// op-by-op HIP-vs-CPU table and every HIP op recomputed on the host from the HIP trace's own
/// inputs; asserts the Phase 1 trace bounds where they are per-op: every HIP `q_norm` /
/// `k_norm` (one `rmsnorm` over `[tokens · heads, head_dim]` rows) within
/// [`QWEN3_QK_NORM_MAX_ULPS`] of its host recompute, and each step's top-5 logits in the
/// cpu-reference's order. Breaks if the HIP per-head norm reads the wrong rows or is skipped,
/// or the HIP path departs from the reference enough to reorder the likely tokens.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_QWEN3_MODEL_DIR, release"]
fn hip_trace_vs_cpu_qwen3() {
    let _gpu = gpu_model_lock();
    const TRACE_DECODE: usize = 2;
    if cfg!(debug_assertions) {
        println!("hip_trace_vs_cpu_qwen3: debug build, nothing traced (run it with --release)");
        return;
    }
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_QWEN3_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");
    let cfg = load_model_config(&model_dir).expect("config.json");
    assert_eq!(
        cfg.family.0.name(),
        "qwen3",
        "TURBINE_TEST_QWEN3_MODEL_DIR must hold a Qwen3 checkpoint"
    );
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    let p01 = prompts.iter().find(|p| p.id == "p01").expect("prompt p01");
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).expect("tokenizer");
    let template = ChatTemplate::resolve(&model_dir, None).expect("chat template");
    let ids = prompt_token_ids(&tokenizer, &template, p01);
    let max_seq_len = (ids.len() + TRACE_DECODE + 1) as u32;
    let index = SafetensorsIndex::open(&model_dir).expect("open safetensors");
    let host = HostMemory::new(turbine_core::types::DeviceId(0), 16 << 30);
    let mut cpu = build_executor(&model_dir, cpu_reference_provider(), host, max_seq_len);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = build_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    cpu.exec.set_trace(true);
    hip.exec.set_trace(true);
    let weights = |name: &str| read_bf16_weight(&index, name);
    let mut checker = LocalChecker::new(&cfg, &weights);
    let (mut accumulated, mut local) = (Vec::new(), Vec::new());
    let mut batch = ids;
    let mut p0 = 0usize;
    let top5 = |row: &[f32]| -> Vec<usize> {
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        order.truncate(5);
        order
    };
    let mut reordered = Vec::new();
    for step in 0..=TRACE_DECODE {
        let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
        let mut rows = Vec::new();
        for r in [&mut cpu, &mut hip] {
            let logits =
                r.kv.forward(&mut r.exec, &batch, &positions)
                    .expect("forward");
            rows.push(logits.row(0).to_vec());
        }
        let (c, h) = (cpu.exec.take_trace(), hip.exec.take_trace());
        accumulated.extend(compare_traces(step, &c, &h));
        local.extend(checker.check_step(step, p0, &h));
        let (want, got) = (top5(&rows[0]), top5(&rows[1]));
        if want != got {
            reordered.push((step, want.clone(), got));
        }
        p0 += batch.len();
        batch = vec![want[0] as u32];
    }
    println!(
        "{}",
        render(
            "Qwen3-4B p01: HIP vs cpu-reference (accumulated)",
            &accumulated
        )
    );
    println!(
        "{}",
        render(
            "Qwen3-4B p01: HIP op vs host recompute from its own inputs (local)",
            &local
        )
    );
    let norms: Vec<_> = local
        .iter()
        .filter(|r| r.name == "q_norm" || r.name == "k_norm")
        .collect();
    assert_eq!(
        norms.len(),
        2 * cfg.num_layers as usize * (TRACE_DECODE + 1),
        "a q_norm and a k_norm per layer and step"
    );
    for r in norms {
        assert!(
            r.stats.max_ulps <= QWEN3_QK_NORM_MAX_ULPS,
            "step {} layer {:?} {}: {} BF16 ulps from the host recompute ({:?})",
            r.step,
            r.layer,
            r.name,
            r.stats.max_ulps,
            r.stats
        );
    }
    assert!(
        reordered.is_empty(),
        "top-5 logits order differs from the cpu-reference (step, cpu, hip): {reordered:?}"
    );
}

// ------------------------------------------------------------------------------ gpt-oss

/// Layers of gpt-oss-20b the trace runs (the cpu-reference executor is scalar): two sliding
/// layers and two full ones.
const GPT_OSS_TRACE_LAYERS: u32 = 4;

/// Positions the traced gpt-oss prompt reaches before its decode steps: past the 128-position
/// window, so the sliding layers drop positions in the prefill and in every decode step.
const GPT_OSS_MIN_PROMPT: usize = 130;

/// The decoder of the first [`GPT_OSS_TRACE_LAYERS`] layers of the gpt-oss checkpoint in
/// `model_dir` on `provider` (the MXFP4 experts stay packed), with the single-sequence KV.
fn build_gpt_oss_executor(
    model_dir: &Path,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> Runner {
    let mut cfg = load_model_config(model_dir).expect("config.json");
    assert_eq!(
        cfg.family.0.name(),
        "gpt_oss",
        "TURBINE_TEST_GPT_OSS_MODEL_DIR must hold a gpt-oss checkpoint"
    );
    assert!(cfg.num_layers > GPT_OSS_TRACE_LAYERS);
    cfg.num_layers = GPT_OSS_TRACE_LAYERS;
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let weights = WeightLoader::load_format(
        cfg.weight_format.get(),
        &index,
        &cfg.family.0.weight_slots(&cfg),
        &mem,
        MAX_STAGING_BYTES,
    )
    .expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let card = provider.card_profile();
    let opts = ExecutorOptions::default();
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, opts),
        &metrics,
        card,
    )
    .expect("every op has a provider");
    let kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
    let exec = DecoderExecutor::new(
        &cfg,
        families::gpt_oss::decoder_spec(),
        weights,
        Arc::new(registry),
        mem,
        ExecutorLimits {
            block_tokens: BLOCK_TOKENS,
            max_batch_tokens: max_seq_len,
            max_seqs: 1,
        },
        opts,
    )
    .expect("executor");
    Runner { exec, kv }
}

/// The router-logit gap under which two experts count as a tie: the same absolute bound the
/// per-op comparison uses. The providers' router logits differ by at most their own trace |d|
/// (measured 1.6e-3 on run 1013172834), so a flip between experts whose logits tie this closely
/// is rounding; a gap above the bound cannot be closed by rounding and is a routing bug.
const GPT_OSS_ROUTER_TIE: f32 = 2e-2;

/// The gpt-oss router selects on the biased logits rounded to BF16, so the tie gap is measured
/// there too: the raw logits' gap can be large where the bias pulls two experts to the same
/// BF16 value (run 1014101047: token 97 flipped between experts whose raw-logit gap was 0.316
/// while their biased, rounded scores were half a BF16 ulp apart — the flip was rounding, not
/// routing). `bias` is the layer's `router_bias` trace (`[1, experts]`).
fn gpt_oss_router_score(logits: &TraceTensor, tok: usize, e: usize, bias: &TraceTensor) -> f32 {
    half::bf16::from_f32(logits.data[tok * logits.shape[1] + e] + bias.data[e]).to_f32()
}

/// The routed experts of one step and layer: a flip of a token's expert set is tolerated only
/// between experts whose biased, BF16-rounded router scores tie within [`GPT_OSS_ROUTER_TIE`]
/// in BOTH providers (rounding may swap them); any larger gap is a routing bug and fails.
/// Multiset pairing: a token's expert ids are distinct, so the symmetric difference pairs
/// one-to-one in id order.
fn check_routing(
    step: usize,
    c: &TraceTensor,
    h: &TraceTensor,
    cl: &TraceTensor,
    hl: &TraceTensor,
    bias: &TraceTensor,
    failures: &mut Vec<String>,
) {
    let tokens = c.shape[0];
    let topk = c.shape[1];
    let experts = cl.shape[1];
    for tok in 0..tokens {
        let (cs, hs) = (
            &c.data[tok * topk..(tok + 1) * topk],
            &h.data[tok * topk..(tok + 1) * topk],
        );
        let cpu_only: Vec<f32> = cs.iter().filter(|x| !hs.contains(x)).copied().collect();
        let hip_only: Vec<f32> = hs.iter().filter(|y| !cs.contains(y)).copied().collect();
        for (x, y) in cpu_only.iter().zip(&hip_only) {
            let (x, y) = (*x as usize, *y as usize);
            let gap_c = (gpt_oss_router_score(cl, tok, x, bias)
                - gpt_oss_router_score(cl, tok, y, bias))
            .abs();
            let gap_h = (gpt_oss_router_score(hl, tok, x, bias)
                - gpt_oss_router_score(hl, tok, y, bias))
            .abs();
            if gap_c > GPT_OSS_ROUTER_TIE || gap_h > GPT_OSS_ROUTER_TIE {
                failures.push(format!(
                    "step {step} layer {:?} token {tok}: routed experts {x} vs {y} differ by a \
                     biased-BF16 router-score gap {gap_c} (cpu) / {gap_h} (hip) > \
                     {GPT_OSS_ROUTER_TIE}\n\
                     cpu ids {cs:?}\nhip ids {hs:?}\ncpu logits {:?}\nhip logits {:?}\n\
                     cpu logits all {:?}\nhip logits all {:?}",
                    c.layer,
                    &cl.data[tok * experts..(tok + 1) * experts],
                    &hl.data[tok * experts..(tok + 1) * experts],
                    cl.data,
                    hl.data,
                ));
            }
        }
    }
}
/// Job sets to `/models/gpt-oss-20b`, and a HIP device; a no-op in a debug build: the
/// cpu-reference is scalar). The checkpoint's first [`GPT_OSS_TRACE_LAYERS`] layers (two
/// sliding-window layers,
/// two full) run prompt p01 of `tests/golden/prompts.jsonl`, its Harmony-rendered tokens
/// repeated to at least [`GPT_OSS_MIN_PROMPT`] positions so the window is active in the
/// prefill, then 2 teacher-forced decode steps past position 128 (the cpu-reference's greedy
/// tokens, decode through CK split-KV with the window and the sinks), on the HIP and the
/// cpu-reference providers, traced. Prints the op-by-op HIP-vs-CPU table and asserts every op
/// within the Phase 1 trace bound: max |d| at most 2e-2 or a quarter of the op's largest value
/// (the bound of `tiny_model gpt_oss_against_transformers`; the routed experts may flip only
/// between experts whose biased, BF16-rounded router scores tie within the trace bound). Breaks
/// if the HIP path drops a bias, the sinks or the window, reads the fused expert layout wrongly
/// or routes to other experts.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_GPT_OSS_MODEL_DIR, release"]
fn hip_trace_vs_cpu_gpt_oss() {
    let _gpu = gpu_model_lock();
    const TRACE_DECODE: usize = 2;
    if cfg!(debug_assertions) {
        println!("hip_trace_vs_cpu_gpt_oss: debug build, nothing traced (run it with --release)");
        return;
    }
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir =
        turbine_kernels::test_support::require_env_dir("TURBINE_TEST_GPT_OSS_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    let p01 = prompts.iter().find(|p| p.id == "p01").expect("prompt p01");
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).expect("tokenizer");
    let template = ChatTemplate::resolve(&model_dir, None).expect("chat template");
    let base = prompt_token_ids(&tokenizer, &template, p01);
    let ids: Vec<u32> = base
        .iter()
        .cycle()
        .take(base.len().max(GPT_OSS_MIN_PROMPT))
        .copied()
        .collect();
    let max_seq_len = (ids.len() + TRACE_DECODE + 1) as u32;
    let host = HostMemory::new(turbine_core::types::DeviceId(0), 32 << 30);
    let mut cpu = build_gpt_oss_executor(&model_dir, cpu_reference_provider(), host, max_seq_len);
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = build_gpt_oss_executor(&model_dir, shim_provider(ctx), mem, max_seq_len);
    cpu.exec.set_trace(true);
    hip.exec.set_trace(true);

    let mut accumulated = Vec::new();
    let mut failures = Vec::new();
    let mut batch = ids;
    let mut p0 = 0usize;
    for step in 0..=TRACE_DECODE {
        let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
        let mut rows = Vec::new();
        for r in [&mut cpu, &mut hip] {
            let logits =
                r.kv.forward(&mut r.exec, &batch, &positions)
                    .expect("forward");
            rows.push(logits.row(0).to_vec());
        }
        let (c, h) = (cpu.exec.take_trace(), hip.exec.take_trace());
        assert_eq!(c.len(), h.len(), "step {step}: op counts");
        for (w, g) in c.iter().zip(&h) {
            if w.name == "topk_ids" {
                let Some(cl) = c
                    .iter()
                    .find(|t| t.name == "router_logits" && t.layer == w.layer)
                else {
                    failures.push(format!("step {step}: no router_logits trace"));
                    continue;
                };
                let Some(hl) = h
                    .iter()
                    .find(|t| t.name == "router_logits" && t.layer == w.layer)
                else {
                    failures.push(format!("step {step}: no HIP router_logits trace"));
                    continue;
                };
                let Some(bias) = c
                    .iter()
                    .find(|t| t.name == "router_bias" && t.layer == w.layer)
                else {
                    failures.push(format!("step {step}: no router_bias trace"));
                    continue;
                };
                check_routing(step, w, g, cl, hl, bias, &mut failures);
            }
        }
        accumulated.extend(compare_traces(step, &c, &h));
        p0 += batch.len();
        let next = rows[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i as u32)
            .expect("a row");
        batch = vec![next];
    }
    println!(
        "{}",
        render(
            "gpt-oss-20b (4 layers) p01 past position 128: HIP vs cpu-reference (accumulated)",
            &accumulated
        )
    );
    for r in &accumulated {
        let s = &r.stats;
        if r.name != "topk_ids" && s.max_abs > 2e-2 && s.max_abs > s.max_ref / 4.0 {
            failures.push(format!(
                "step {} layer {:?} {}: max |d| {} ({} ulps), max |ref| {}",
                r.step, r.layer, r.name, s.max_abs, s.max_ulps, s.max_ref
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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
    build_any_executor_with(model_dir, None, provider, mem, max_seq_len)
}

/// [`build_any_executor`] with a `model.rope_scaling` override (the YaRN golden).
fn build_any_executor_with(
    model_dir: &Path,
    rope_scaling: Option<&serde_json::Value>,
    provider: Arc<dyn KernelProvider>,
    mem: Arc<dyn DeviceMemory>,
    max_seq_len: u32,
) -> AnyRunner {
    let cfg = load_model_config_with(model_dir, rope_scaling).expect("config.json");
    let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
    let slots = cfg.family.0.weight_slots(&cfg);
    let weights =
        WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("load weights");
    let metrics = KernelMetrics::register(&MetricsRegistry::new());
    let order = [provider.id()];
    let card = provider.card_profile();
    let registry = KernelRegistry::build(
        vec![provider],
        &order,
        &executor::requirements(&cfg, BLOCK_TOKENS, ExecutorOptions::default()),
        &metrics,
        card,
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
    /// The reference's top-5 at this position: (token, reference logprob, ours).
    candidates: Vec<(u32, f32, f32)>,
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
        let mut candidates = Vec::with_capacity(5);
        for &(id, ref_lp) in r.top_logprobs[step].iter().take(5) {
            candidates.push((id, ref_lp, lp[id as usize]));
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
            candidates,
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

// ------------------------------------------------------------------------------- YaRN

/// The factor-16 YaRN override of `tests/golden/llama-3.2-3b-instruct-yarn16/` (its README and
/// `scripts/lab/phase6-novanas-llama-yarn16.yaml`).
fn yarn16_override() -> serde_json::Value {
    serde_json::json!({
        "rope_type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32,
        "beta_slow": 1
    })
}

/// Prompts at most this long also run on the (scalar) cpu-reference provider in
/// [`yarn_teacher_forced_vs_reference`]; longer ones run on HIP only.
const YARN_CPU_MAX_TOKENS: usize = 1024;

/// Lab diagnostic (P6a Tasks 28 / 28a, YaRN tail numerics), a no-op unless `TURBINE_GOLDEN_YARN`
/// names prompts of the YaRN golden (comma-separated ids); run it alone in a release build.
/// For each named prompt, Llama-3.2-3B with the factor-16 YaRN override runs the reference's
/// prompt and then its tokens teacher-forced, and prints per position the largest |Δ logprob|
/// against the transformers reference's top-5 for `hip` (the HIP provider) and `cpu` (the
/// cpu-reference provider, only for prompts up to [`YARN_CPU_MAX_TOKENS`]), then every
/// candidate of the positions whose tail |Δ| exceeds 0.3. Both place the attention factor on
/// cos/sin as transformers does (Task 28a; before it, the fold into the softmax scale measured
/// p16 tails of 1.49 cpu / 1.38 hip against 0.40 for transformers' placement).
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY, TURBINE_TEST_MODEL_DIR and TURBINE_GOLDEN_YARN"]
fn yarn_teacher_forced_vs_reference() {
    let Some(ids) = std::env::var("TURBINE_GOLDEN_YARN")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        println!("TURBINE_GOLDEN_YARN is not set: nothing compared");
        return;
    };
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let _gpu = gpu_model_lock();
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");

    let fixture = golden_dir().join("llama-3.2-3b-instruct-yarn16");
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
    let seq_len = |r: &&ReferenceRecord| (r.prompt_token_ids.len() + r.tokens.len()) as u32;
    let max_seq_len = selected.iter().map(seq_len).max().expect("a prompt");
    let cpu_seq_len = selected
        .iter()
        .filter(|r| r.prompt_token_ids.len() <= YARN_CPU_MAX_TOKENS)
        .map(seq_len)
        .max();
    let rope = yarn16_override();
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let mut hip = build_any_executor_with(
        &model_dir,
        Some(&rope),
        shim_provider(ctx),
        mem,
        max_seq_len,
    );
    let host = || HostMemory::new(turbine_core::types::DeviceId(0), 16 << 30);
    let mut cpu = cpu_seq_len.map(|len| {
        build_any_executor_with(
            &model_dir,
            Some(&rope),
            cpu_reference_provider(),
            host(),
            len,
        )
    });
    let worst = |v: &[ForcedPosition]| {
        v.iter()
            .fold((0f32, 0f32), |(l, t), p| (l.max(p.likely), t.max(p.tail)))
    };
    let cols = |v: &[ForcedPosition], pos: usize| {
        v.get(pos).map_or((u32::MAX, f32::NAN, f32::NAN), |p| {
            (p.argmax, p.likely, p.tail)
        })
    };
    for r in selected {
        let short = r.prompt_token_ids.len() <= YARN_CPU_MAX_TOKENS;
        let h = teacher_forced(&mut hip, r, tol.likely_logprob_floor);
        println!("YaRN {} hip max {:?}", r.id, worst(&h));
        let forced = |runner: Option<&mut AnyRunner>| match runner {
            Some(runner) if short => teacher_forced(runner, r, tol.likely_logprob_floor),
            _ => Vec::new(),
        };
        let c = forced(cpu.as_mut());
        println!(
            "YaRN {} teacher-forced vs transformers (|Δ| likely / tail):",
            r.id
        );
        println!("pos  ref_tok  cpu_argmax cpu_likely cpu_tail  hip_argmax hip_likely hip_tail");
        for (pos, h) in h.iter().enumerate() {
            let (ca, cl, ct) = cols(&c, pos);
            println!(
                "{pos:>3} {:>8} {ca:>10} {cl:>10.4} {ct:>8.4} {:>11} {:>10.4} {:>8.4}",
                r.tokens[pos], h.argmax, h.likely, h.tail
            );
        }
        for (pos, h) in h.iter().enumerate() {
            let cpu_pos = c.get(pos);
            if h.tail <= 0.3 && cpu_pos.is_none_or(|p| p.tail <= 0.3) {
                continue;
            }
            println!("YaRN {} pos {pos} candidates (token ref cpu hip):", r.id);
            for (i, &(id, ref_lp, hip_lp)) in h.candidates.iter().enumerate() {
                let cpu_lp = cpu_pos.map_or(f32::NAN, |p| p.candidates[i].2);
                println!("    {id:>8} {ref_lp:>9.4} {cpu_lp:>9.4} {hip_lp:>9.4}");
            }
        }
        if short {
            println!(
                "YaRN {} max: cpu {:?}, hip {:?}",
                r.id,
                worst(&c),
                worst(&h)
            );
        }
    }
}

// ---------------------------------------------------------------------- MXFP4 decode

/// Lab diagnostic (the MXFP4-A16 8B p05 golden miss), a no-op unless `TURBINE_GOLDEN_MXFP4`
/// names `<prompt id>:<position>` (e.g. `p05:5`); needs `TURBINE_TEST_MXFP4_MODEL_DIR` (the
/// `llama-3.1-8b-instruct-mxfp4a16` checkpoint). On the HIP provider, the prompt and the
/// reference's first `position` tokens run twice, traced, on one executor and two KV caches:
/// (A) the prompt as one prefill, then one decode step per token, as the server generates;
/// (B) everything as one prefill. Prints, for the row predicting `position`, each op's
/// difference A − B per layer (the first layer that departs is where decode and prefill
/// disagree), the same for the K/V rows the decode steps appended, and both logprob rows.
#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY, TURBINE_TEST_MXFP4_MODEL_DIR and TURBINE_GOLDEN_MXFP4"]
fn mxfp4_decode_vs_prefill_trace() {
    use turbine_model::executor::TraceTensor;
    use turbine_model::testing::trace::DiffStats;

    let Some(spec) = std::env::var("TURBINE_GOLDEN_MXFP4")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        println!("TURBINE_GOLDEN_MXFP4 is not set: nothing traced");
        return;
    };
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let _gpu = gpu_model_lock();
    let (id, pos) = spec
        .split_once(':')
        .expect("TURBINE_GOLDEN_MXFP4=<id>:<position>");
    let pos: usize = pos.parse().expect("position");
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MXFP4_MODEL_DIR");
    let ctx = turbine_kernels::test_support::open_context("hip");
    let fixture = golden_dir().join("llama-3.1-8b-instruct-mxfp4a16");
    let references: Vec<ReferenceRecord> = read_jsonl(&fixture.join("reference.jsonl"));
    let r = references
        .iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("no reference prompt {id}"));
    let n_prompt = r.prompt_token_ids.len();
    let max_seq_len = (n_prompt + pos + 1) as u32;
    let mem: Arc<dyn DeviceMemory> = ctx.clone();
    let cfg = load_model_config(&model_dir).expect("config.json");
    let index = SafetensorsIndex::open(&model_dir).expect("open safetensors");
    let build = |provider: Arc<dyn KernelProvider>, mem: Arc<dyn DeviceMemory>| -> Runner {
        let weights = WeightLoader::load_format(
            cfg.weight_format.get(),
            &index,
            &llama_slots(&cfg),
            &mem,
            MAX_STAGING_BYTES,
        )
        .expect("load weights");
        let order = [provider.id()];
        let card = provider.card_profile();
        let reqs = executor::available_requirements(
            &cfg,
            BLOCK_TOKENS,
            ExecutorOptions::default(),
            std::slice::from_ref(&provider),
        );
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider");
        let exec = DecoderExecutor::new(
            &cfg,
            families::llama::decoder_spec(),
            weights,
            Arc::new(registry),
            mem.clone(),
            ExecutorLimits {
                block_tokens: BLOCK_TOKENS,
                max_batch_tokens: max_seq_len,
                max_seqs: 1,
            },
            ExecutorOptions::default(),
        )
        .expect("executor");
        let kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
        Runner { exec, kv }
    };
    let mut a = build(shim_provider(ctx), mem.clone());
    let mut kv_b = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
    a.exec.set_trace(true);

    // (A) prefill, then decode steps; keep every step's trace.
    let mut steps: Vec<Vec<TraceTensor>> = Vec::new();
    let mut batch = r.prompt_token_ids.clone();
    let mut p0 = 0usize;
    let mut logits_a = Vec::new();
    for step in 0..=pos {
        let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
        let logits =
            a.kv.forward(&mut a.exec, &batch, &positions)
                .expect("forward");
        logits_a = log_softmax(logits.row(0));
        steps.push(a.exec.take_trace());
        p0 += batch.len();
        if step < pos {
            batch = vec![r.tokens[step]];
        }
    }
    // (B) one prefill of the same tokens.
    let mut all = r.prompt_token_ids.clone();
    all.extend_from_slice(&r.tokens[..pos]);
    let positions: Vec<u32> = (0..all.len() as u32).collect();
    let logits = kv_b
        .forward(&mut a.exec, &all, &positions)
        .expect("forward");
    let logits_b = log_softmax(logits.row(0));
    let trace_b = a.exec.take_trace();

    let row = |t: &TraceTensor, i: usize| -> Vec<f32> {
        let cols = t.shape[1];
        t.data[i * cols..(i + 1) * cols].to_vec()
    };
    let find = |trace: &[TraceTensor], layer: Option<usize>, name: &str| -> Option<TraceTensor> {
        trace
            .iter()
            .find(|e| e.layer == layer && e.name == name)
            .cloned()
    };
    const OPS: [&str; 14] = [
        "attn_norm",
        "q",
        "k",
        "v",
        "q_rope",
        "k_rope",
        "attn",
        "o_proj",
        "resid_attn",
        "mlp_norm",
        "gate",
        "up",
        "down",
        "resid_mlp",
    ];
    // Step s of (A) holds positions n_prompt + s − 1 (s ≥ 1); (B) holds them all.
    for (s, step) in steps.iter().enumerate().skip(1) {
        let b_row = n_prompt + s - 1;
        println!("== {id} decode step {s} (position {b_row}) A − B, per layer: max |Δ| (max ulps)");
        println!("layer {}", OPS.join(" "));
        for layer in 0..cfg.num_layers as usize {
            let l = Some(layer);
            let cells: Vec<String> = OPS
                .iter()
                .map(
                    |name| match (find(step, l, name), find(&trace_b, l, name)) {
                        (Some(ta), Some(tb)) => {
                            let d = DiffStats::of(&row(&tb, b_row), &row(&ta, 0));
                            format!("{:.3e}({:.0})", d.max_abs, d.max_ulps)
                        }
                        _ => "-".into(),
                    },
                )
                .collect();
            println!("{layer:>3} {}", cells.join(" "));
        }
    }
    for name in ["final_norm", "logits"] {
        if let (Some(ta), Some(tb)) = (find(&steps[pos], None, name), find(&trace_b, None, name)) {
            let d = DiffStats::of(&row(&tb, 0), &row(&ta, 0));
            println!("{name}: max |Δ| {:.4e} ({:.0} ulps)", d.max_abs, d.max_ulps);
        }
    }
    println!("position {pos}: reference top-5 / decode (A) / prefill (B) logprobs");
    for &(tok, lp) in r.top_logprobs[pos].iter().take(5) {
        println!(
            "  {tok:>7} ref {lp:>9.4} A {:>9.4} B {:>9.4}",
            logits_a[tok as usize], logits_b[tok as usize]
        );
    }
    let top = |lp: &[f32]| {
        let mut v: Vec<(usize, f32)> = lp.iter().copied().enumerate().collect();
        v.sort_by(|x, y| y.1.total_cmp(&x.1));
        v.truncate(5);
        v
    };
    println!("  A top-5 {:?}", top(&logits_a));
    println!("  B top-5 {:?}", top(&logits_b));
    // (C) with TURBINE_GOLDEN_MXFP4_CPU set: the same prefill on the cpu-reference provider
    // (scalar, sequential f32 sums: a third, independent rounding of the same model).
    if std::env::var_os("TURBINE_GOLDEN_MXFP4_CPU").is_some() {
        let host: Arc<dyn DeviceMemory> =
            HostMemory::new(turbine_core::types::DeviceId(0), 24 << 30);
        let mut c = build(cpu_reference_provider(), host);
        let logits =
            c.kv.forward(&mut c.exec, &all, &positions)
                .expect("forward");
        let logits_c = log_softmax(logits.row(0));
        println!("position {pos}: reference / cpu-reference prefill (C)");
        for &(tok, lp) in r.top_logprobs[pos].iter().take(5) {
            println!("  {tok:>7} ref {lp:>9.4} C {:>9.4}", logits_c[tok as usize]);
        }
        println!("  C top-5 {:?}", top(&logits_c));
    }
}

// -------------------------------------------------- MXFP4 p05 CPU vs transformers (P7 Task 28)

/// One op of the p05 transformers trace ([`scripts/golden/tiny_trace.py --prompt-file`]): its
/// layer (`None` for `embed`/`final_norm`/`logits`), name, shape and decoded values.
struct P05Op {
    layer: Option<usize>,
    name: String,
    cols: usize,
    values: Vec<f32>,
}

/// Two hex chars per byte, little-endian per value.
fn decode_hex(hex: &str) -> Vec<u8> {
    let bytes = hex.as_bytes();
    assert!(bytes.len().is_multiple_of(2), "odd hex length");
    bytes
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| (b as char).to_digit(16).expect("hex digit") as u8;
            (digit(pair[0]) << 4) | digit(pair[1])
        })
        .collect()
}

/// The MXFP4-A16 8B checkpoint and the p05 trace, from the lab environment: `None` when neither
/// `TURBINE_TEST_MXFP4_MODEL_DIR` nor an `mxfp4`-named `TURBINE_TEST_MODEL_DIR` is set (the
/// command line the fixture was made with, `TURBINE_TEST_MODEL_DIR=/models/llama-3.1-8b-instruct-mxfp4a16`,
/// included). The trace path is `TURBINE_P05_TRACE` (repo-relative) or the committed fixture.
fn p05_inputs() -> Option<(PathBuf, PathBuf)> {
    let model_dir = if let Some(v) = std::env::var_os("TURBINE_TEST_MXFP4_MODEL_DIR") {
        PathBuf::from(v)
    } else {
        let dir = PathBuf::from(std::env::var_os("TURBINE_TEST_MODEL_DIR")?);
        let is_mxfp4 = dir
            .file_name()
            .map(|n| n.to_string_lossy().contains("mxfp4"))
            .unwrap_or(false);
        if !is_mxfp4 {
            return None;
        }
        dir
    };
    let rel = std::env::var("TURBINE_P05_TRACE")
        .unwrap_or_else(|_| "tests/golden/llama-3.1-8b-instruct-mxfp4a16/p05-trace.json".into());
    let trace = PathBuf::from(rel);
    let trace = if trace.is_absolute() {
        trace
    } else {
        repo_root().join(trace)
    };
    Some((model_dir, trace))
}

/// Lab diagnostic (P7 Task 28 / S-14; release only, a no-op in a debug build and without
/// `TURBINE_TEST_MXFP4_MODEL_DIR`): the MXFP4-A16 8B checkpoint's p05 knife edge — position 25
/// (the 21-token prompt plus the reference's first 5 greedy tokens), where Turbine's HIP decode
/// and its cpu-reference prefill land in one "bad" state and the HIP prefill, the transformers
/// reference and the committed tolerance in the other — checked op by op. Runs the cpu-reference
/// (scalar) prefill of the trace's tokens and prints, per layer and op, the max |Δ| against the
/// transformers trace and the first op whose |Δ| exceeds 1e-2 (whether a single op injects the
/// divergence or the CPU path merely rounds differently from the reference throughout).
#[test]
#[ignore = "needs a release build, the MXFP4-A16 8B checkpoint and the p05 trace fixture"]
fn mxfp4_cpu_vs_transformers_p05() {
    use turbine_model::testing::trace::DiffStats;

    if cfg!(debug_assertions) {
        println!(
            "mxfp4_cpu_vs_transformers_p05: debug build (the cpu-reference prefill is scalar); run with --release"
        );
        return;
    }
    let Some((model_dir, trace_path)) = p05_inputs() else {
        println!(
            "mxfp4_cpu_vs_transformers_p05: TURBINE_TEST_MXFP4_MODEL_DIR is not set: nothing traced"
        );
        return;
    };
    let raw =
        std::fs::read(&trace_path).unwrap_or_else(|e| panic!("read {}: {e}", trace_path.display()));
    let json: serde_json::Value = serde_json::from_slice(&raw).expect("trace JSON");
    let prompt_ids: Vec<u32> = json["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().expect("prompt id") as u32)
        .collect();
    let ops: Vec<P05Op> = json["ops"]
        .as_array()
        .expect("ops")
        .iter()
        .map(|o| {
            let layer = o["layer"].as_u64().map(|l| l as usize);
            let name = o["name"].as_str().expect("name").to_string();
            let shape = o["shape"].as_array().expect("shape");
            let (rows, cols) = (
                shape[0].as_u64().expect("rows") as usize,
                shape[1].as_u64().expect("cols") as usize,
            );
            let bytes = decode_hex(o["data_hex"].as_str().expect("data_hex"));
            let values: Vec<f32> = match o["dtype"].as_str().unwrap_or("f32") {
                "bf16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                _ => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            };
            assert_eq!(values.len(), rows * cols, "{name}: values vs shape");
            P05Op {
                layer,
                name,
                cols,
                values,
            }
        })
        .collect();
    println!(
        "mxfp4_cpu_vs_transformers_p05: {} tokens, {} traced ops, {}\n  (Turbine cpu-reference prefill vs transformers; the trace holds each op's last row)",
        prompt_ids.len(),
        ops.len(),
        trace_path.display()
    );

    let max_seq_len = prompt_ids.len() as u32;
    let mem: Arc<dyn DeviceMemory> = HostMemory::new(turbine_core::types::DeviceId(0), 24 << 30);
    let mut runner = build_executor(&model_dir, cpu_reference_provider(), mem, max_seq_len);
    runner.exec.set_trace(true);
    let positions: Vec<u32> = (0..max_seq_len).collect();
    runner
        .kv
        .forward(&mut runner.exec, &prompt_ids, &positions)
        .expect("cpu-reference prefill");
    let trace = runner.exec.take_trace();

    let label = |layer: Option<usize>| layer.map_or_else(|| "-".to_string(), |l| l.to_string());
    println!(
        "{:>5}  {:<11} {:>11} {:>9} {:>10}",
        "layer", "op", "max|Δ|", "ulps", "max |ref|"
    );
    let mut first_over: Option<(Option<usize>, String, f32)> = None;
    for op in &ops {
        let Some(got) = trace
            .iter()
            .find(|t| t.layer == op.layer && t.name == op.name)
        else {
            println!(
                "{:>5}  {:<11} (Turbine records no such op)",
                label(op.layer),
                op.name
            );
            continue;
        };
        assert_eq!(
            got.shape[1],
            op.cols,
            "{} {}: {} columns vs {}",
            label(op.layer),
            op.name,
            got.shape[1],
            op.cols
        );
        let last = &got.data[(got.shape[0] - 1) * got.shape[1]..];
        let stats = DiffStats::of(&op.values, last);
        println!(
            "{:>5}  {:<11} {:>11.4e} {:>9.2} {:>10.4e}",
            label(op.layer),
            op.name,
            stats.max_abs,
            stats.max_ulps,
            stats.max_ref
        );
        if stats.max_abs > 1e-2 && first_over.is_none() {
            first_over = Some((op.layer, op.name.clone(), stats.max_abs));
        }
    }
    match first_over {
        None => println!("first op above 1e-2: none"),
        Some((layer, name, d)) => println!(
            "first op above 1e-2: layer {} {name} max|Δ| {d:.4e}",
            label(layer)
        ),
    }
}

// ------------------------------------------------------------------- Qwen3-MoE (P7 S-5)

/// The traced tensors every tensor-parallel rank holds whole (identical rows on all ranks, the
/// CPU executor's shapes): the residual after attention, the router's logits, the all-reduced
/// expert output and the final norm. The router's choices themselves are compared through
/// [`DecoderExecutor::set_routing_trace`] (the op trace does not carry them).
const QWEN3MOE_REPLICATED_TRACES: [&str; 4] =
    ["resid_attn", "router_logits", "moe_out", "final_norm"];

/// One tensor-parallel rank's kept op traces, its per-step logits rows and its routing trace.
struct RankRun {
    steps: Vec<Vec<TraceTensor>>,
    logits: Vec<Vec<f32>>,
    routing: Vec<RoutingRow>,
}

/// Layers of Qwen3-30B-A3B the trace runs (the cpu-reference executor is scalar).
const QWEN3MOE_TRACE_LAYERS: u32 = 4;

/// Lab only (P7 S-5; two HIP devices, release build; needs `TURBINE_TEST_QWEN3MOE_MODEL_DIR` =
/// Qwen3-30B-A3B-Instruct-2507-FP8, which the lab test Job sets to
/// `/models/qwen3-30b-a3b-instruct-2507-fp8`, and `TURBINE_KERNEL_LIBRARY`; a no-op in a debug
/// build). The checkpoint's first [`QWEN3MOE_TRACE_LAYERS`] layers run prompt p01 of
/// `tests/golden/prompts.jsonl` (prefill and 2 teacher-forced decode steps, the cpu-reference's
/// greedy tokens) on the cpu-reference executor (one device) and on a two-rank tensor-parallel
/// group of HIP devices (the `hostmem` collective, FP8-block experts split column-wise per rank);
/// both ranks' replicated tensors ([`QWEN3MOE_REPLICATED_TRACES`]) are compared with the
/// cpu-reference's op by op and printed. Asserts the ranks end every step with equal logits and
/// that the routers choose the cpu-reference's experts for all but a small share of the
/// (token, layer) rows. Breaks if a rank loads the wrong shard of a quantized expert, routes on
/// a partial hidden row, or the HIP FP8-block experts depart from the host reference.
#[test]
#[ignore = "needs two HIP devices, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_QWEN3MOE_MODEL_DIR, release"]
fn hip_trace_vs_cpu_qwen3moe() {
    use turbine_distributed::collective::{self as coll, CollectiveInit};
    use turbine_model::executor::TraceTensor;
    use turbine_model::tp::{self, ShardSpec, TpContext};

    let _gpu = gpu_model_lock();
    const TRACE_DECODE: usize = 2;
    if cfg!(debug_assertions) {
        println!("hip_trace_vs_cpu_qwen3moe: debug build, nothing traced (run it with --release)");
        return;
    }
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir =
        turbine_kernels::test_support::require_env_dir("TURBINE_TEST_QWEN3MOE_MODEL_DIR");
    let mut cfg = load_model_config(&model_dir).expect("config.json");
    assert_eq!(
        cfg.family.0.name(),
        "qwen3_moe",
        "TURBINE_TEST_QWEN3MOE_MODEL_DIR must hold a Qwen3-MoE checkpoint"
    );
    assert!(cfg.num_layers > QWEN3MOE_TRACE_LAYERS);
    cfg.num_layers = QWEN3MOE_TRACE_LAYERS;
    let prompts: Vec<PromptRecord> = read_jsonl(&golden_dir().join("prompts.jsonl"));
    let p01 = prompts.iter().find(|p| p.id == "p01").expect("prompt p01");
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).expect("tokenizer");
    let template = ChatTemplate::resolve(&model_dir, None).expect("chat template");
    let ids = prompt_token_ids(&tokenizer, &template, p01);
    let max_seq_len = (ids.len() + TRACE_DECODE + 1) as u32;
    let limits = ExecutorLimits {
        block_tokens: BLOCK_TOKENS,
        max_batch_tokens: max_seq_len,
        max_seqs: 1,
    };
    let opts = ExecutorOptions::default();
    let index = SafetensorsIndex::open(&model_dir).expect("open safetensors");
    let keep = |trace: Vec<TraceTensor>| -> Vec<TraceTensor> {
        trace
            .into_iter()
            .filter(|t| QWEN3MOE_REPLICATED_TRACES.contains(&t.name))
            .collect()
    };

    // One device on the cpu-reference provider: the traces and the greedy tokens to force.
    let (cpu_steps, forced, cpu_logits, _cpu_routing) = {
        let provider = cpu_reference_provider();
        let mem: Arc<dyn DeviceMemory> =
            HostMemory::new(turbine_core::types::DeviceId(0), 32 << 30);
        let slots = cfg.family.0.weight_slots(&cfg);
        let weights = WeightLoader::load_format(
            cfg.weight_format.get(),
            &index,
            &slots,
            &mem,
            MAX_STAGING_BYTES,
        )
        .expect("load weights");
        let reqs = executor::available_requirements(
            &cfg,
            BLOCK_TOKENS,
            opts,
            std::slice::from_ref(&provider),
        );
        let order = [provider.id()];
        let card = provider.card_profile();
        let metrics = KernelMetrics::register(&MetricsRegistry::new());
        let registry = KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
            .expect("every op has a provider");
        let mut exec = DecoderExecutor::new(
            &cfg,
            families::qwen3_moe::decoder_spec(),
            weights,
            Arc::new(registry),
            Arc::clone(&mem),
            limits,
            opts,
        )
        .expect("cpu executor");
        let mut kv = SequenceKv::new(&mem, cfg.kv_layout(BLOCK_TOKENS), max_seq_len).expect("kv");
        exec.set_trace(true);
        exec.set_routing_trace(true);
        let (mut steps, mut forced, mut logits, mut routing) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut batch = ids.clone();
        let mut p0 = 0usize;
        for _ in 0..=TRACE_DECODE {
            let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
            let row = kv
                .forward(&mut exec, &batch, &positions)
                .expect("forward")
                .row(0)
                .to_vec();
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .map(|(i, _)| i as u32)
                .expect("a row");
            steps.push(keep(exec.take_trace()));
            routing.extend(exec.take_routing_trace());
            logits.push(row);
            forced.push(next);
            p0 += batch.len();
            batch = vec![next];
        }
        (steps, forced, logits, routing)
    };

    // Two HIP ranks, each its own device, the same batches in the same order.
    let lib = coll::registry()
        .get("hostmem")
        .expect("registered")
        .load(None)
        .expect("load hostmem");
    let unique_id = lib.unique_id().expect("id");
    type RankTraces = Vec<RankRun>;
    let ranks: RankTraces = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2u32)
            .map(|rank| {
                let (lib, cfg, index, ids, forced) =
                    (Arc::clone(&lib), &cfg, &index, &ids, &forced);
                scope.spawn(move || {
                    let ctx = turbine_kernels::test_support::open_context_nth("hip", rank as usize);
                    let mem: Arc<dyn DeviceMemory> = ctx.clone();
                    let collective = lib
                        .open(CollectiveInit {
                            rank: rank as usize,
                            world: 2,
                            unique_id,
                            init_timeout: std::time::Duration::from_secs(120),
                            op_timeout: std::time::Duration::from_secs(120),
                            clock: Arc::new(turbine_core::clock::SystemClock::new()),
                            metrics: None,
                            memory: Some(Arc::clone(&mem)),
                            route_max_bytes: Some(u64::MAX),
                        })
                        .expect("open hostmem");
                    let s = ShardSpec { rank, world: 2 };
                    let provider = shim_provider(ctx);
                    let slots = tp::weight_slots(cfg, s).expect("shard slots");
                    let mut rank_cfg = cfg.clone();
                    turbine_model::weights::resolve_for_providers(
                        &mut rank_cfg,
                        &slots,
                        std::slice::from_ref(&provider),
                    );
                    let weights = WeightLoader::load_format(
                        rank_cfg.weight_format.get(),
                        index,
                        &slots,
                        &mem,
                        MAX_STAGING_BYTES,
                    )
                    .expect("load shard");
                    let reqs = tp::available_requirements(
                        &rank_cfg,
                        s,
                        BLOCK_TOKENS,
                        opts,
                        std::slice::from_ref(&provider),
                    )
                    .expect("requirements");
                    let order = [provider.id()];
                    let card = provider.card_profile();
                    let metrics = KernelMetrics::register(&MetricsRegistry::new());
                    let registry =
                        KernelRegistry::build(vec![provider], &order, &reqs, &metrics, card)
                            .expect("every op has a provider");
                    let mut exec = DecoderExecutor::new_tp(
                        &rank_cfg,
                        families::qwen3_moe::decoder_spec(),
                        weights,
                        Arc::new(registry),
                        Arc::clone(&mem),
                        limits,
                        opts,
                        Some(TpContext {
                            rank,
                            world: 2,
                            collective,
                            stream: mem.compute_stream(),
                        }),
                    )
                    .expect("rank executor");
                    let layout = tp::kv_layout(cfg, s, BLOCK_TOKENS).expect("layout");
                    let mut kv = SequenceKv::new(&mem, layout, max_seq_len).expect("kv");
                    exec.set_trace(true);
                    exec.set_routing_trace(true);
                    let (mut steps, mut logits, mut rank_routing) =
                        (Vec::new(), Vec::new(), Vec::new());
                    let mut batch = ids.clone();
                    let mut p0 = 0usize;
                    for &next in forced.iter().take(TRACE_DECODE + 1) {
                        let positions: Vec<u32> = (p0 as u32..(p0 + batch.len()) as u32).collect();
                        let row = kv
                            .forward(&mut exec, &batch, &positions)
                            .expect("forward")
                            .row(0)
                            .to_vec();
                        steps.push(keep(exec.take_trace()));
                        rank_routing.extend(exec.take_routing_trace());
                        logits.push(row);
                        p0 += batch.len();
                        batch = vec![next];
                    }
                    RankRun {
                        steps,
                        logits,
                        routing: rank_routing,
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    });

    assert!(
        ranks[0].logits == ranks[1].logits,
        "the ranks' logits differ from each other"
    );
    // The routers run on the all-reduced hidden row: the two ranks must choose identical experts
    // for every row (the topk_gates too, they weigh the expert sums). Against the cpu-reference
    // the choices are not compared: the router logits are F32 GEMMs over a BF16 hidden row and
    // the all-reduce's rounding moves them by ~0.3 % relative (measured, layer 0: 3.4e-2 on a
    // 10.2 logit), enough to flip a borderline top-k member of a row — observed on 15 of the
    // 64 rows, without moving the top-5 logits (asserted below as the same set).
    for (rank, r) in ranks.iter().enumerate() {
        assert_eq!(
            r.routing, ranks[0].routing,
            "rank {rank}: routing differs from rank 0"
        );
    }
    let choices = ranks[0].routing.len();
    for (rank, r) in ranks.iter().enumerate() {
        let mut rows = Vec::new();
        for (step, (want, got)) in cpu_steps.iter().zip(&r.steps).enumerate() {
            rows.extend(compare_traces(step, want, got));
        }
        println!(
            "{}",
            render(
                &format!(
                    "Qwen3-30B-A3B p01 ({QWEN3MOE_TRACE_LAYERS} layers) rank {rank}: HIP tp 2 vs cpu-reference"
                ),
                &rows
            )
        );
    }
    // The router logits stay close to the cpu-reference's, and every step's top-5 logits keep
    // the cpu-reference's set (a near-tie may reorder them).
    let mut worst_router = 0f32;
    let mut router_rows = 0usize;
    for r in &ranks {
        for (step, (want, got)) in cpu_steps.iter().zip(&r.steps).enumerate() {
            let find = |t: &[TraceTensor], name: &str| {
                t.iter()
                    .find(|e| e.name == name)
                    .cloned()
                    .unwrap_or_else(|| panic!("step {step}: no {name} trace"))
            };
            let (w, g) = (find(want, "router_logits"), find(got, "router_logits"));
            router_rows += w.data.len();
            worst_router = worst_router.max(
                w.data
                    .iter()
                    .zip(&g.data)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
        }
    }
    println!(
        "routing rows: {choices}; router logit |Δ| worst {worst_router:.4} over {router_rows} \
         elements"
    );
    assert!(choices > 0, "no router choice was traced");
    for (step, (c, h)) in cpu_logits.iter().zip(&ranks[0].logits).enumerate() {
        let top = |row: &[f32]| {
            let mut v: Vec<usize> = (0..row.len()).collect();
            v.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            v.truncate(5);
            v.iter().map(|&i| i as u32).collect::<Vec<u32>>()
        };
        let mut want = top(c);
        let mut got = top(h);
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(
            want, got,
            "step {step}: the top-5 logits are not the cpu-reference's set"
        );
    }
    assert!(
        worst_router <= 0.05,
        "router logits depart from the cpu-reference by {worst_router:.4} (at most 0.05; the \
         measured spread of the all-reduced row is 0.034)"
    );
}
