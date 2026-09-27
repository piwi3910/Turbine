//! Lab only (novanas R9700, OLMoE-1B-7B weights): a sequence's logits do not depend on the
//! batch it runs in (decision 2026-09-27, "Before Phase 5": the OLMoE golden flip at
//! concurrency 16). A golden prompt runs its prefill and then its reference tokens
//! teacher-forced, one step at a time; every step runs alone and inside batches of other
//! compositions — other prefills, 3 or 15 decode rows, a decode row next to a prefill chunk,
//! the prompt split into two chunks — with the target's KV history always the alone run's. Each
//! batched step is traced ([`DecoderExecutor::set_trace`]) and compared with the alone step on
//! the target's rows op by op: the first op whose rows differ is the op that depends on the
//! batch, and everything after it inherits the difference.
//!
//! `olmoe_rows_are_batch_invariant` prints one line per (step, scenario) with the first differing
//! op, then a per-scenario summary, then asserts the target's logits are bit-identical in every
//! scenario, except the ones marked pending the per-card GEMM table (the prompt's prefill rows at
//! the tail of a 2,048-row batch, where hipBLASLt's dense GEMMs change the rows' bits), which it
//! only reports. `TURBINE_BATCH_TARGETS` (default `p10,p14`) names the prompts and
//! `TURBINE_BATCH_STEPS` (default 32, the whole reference) the decode steps.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use turbine_core::types::{BlockId, SeqId};
use turbine_kernels::{KernelMetrics, KernelRegistry, shim_provider};
use turbine_model::executor::{
    self, BatchInput, DecoderExecutor, ExecutorLimits, ExecutorOptions, ModelExecutor, SeqSlice,
    TraceTensor,
};
use turbine_model::families;
use turbine_model::{MAX_STAGING_BYTES, SafetensorsIndex, WeightLoader, load_model_config};
use turbine_observability::MetricsRegistry;
use turbine_tensor::{DeviceBuffer, DeviceMemory, KvPoolView};

const BLOCK_TOKENS: u32 = 128;
/// Blocks each sequence of the test owns (p09's 2,004-token prompt fits in 16).
const BLOCKS_PER_SEQ: u32 = 17;
const MAX_SEQS: u32 = 21;
/// Sequence of the long companion prefill (p09 whole).
const LONG_SEQ: u32 = 20;
const MAX_BATCH_TOKENS: u32 = 2048;
/// Companion prompts are cut to this many tokens.
const COMPANION_MAX: usize = 100;

#[derive(Debug, Deserialize)]
struct ReferenceRecord {
    id: String,
    prompt_token_ids: Vec<u32>,
    tokens: Vec<u32>,
}

fn reference_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl")
}

fn references() -> Vec<ReferenceRecord> {
    let text = std::fs::read_to_string(reference_path()).expect("reference.jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("reference line"))
        .collect()
}

/// One sequence of a batch: its tokens this step at positions `start..`, its blocks.
#[derive(Clone)]
struct Part {
    seq: u32,
    tokens: Vec<u32>,
    start: u32,
}

/// A batched step: the parts in row order and which one is the target.
struct Scenario {
    name: String,
    /// Known to differ until the per-card GEMM table pins batch-invariant hipBLASLt algorithms
    /// (dense GEMM rows at the tail of a 2,048-row batch): reported, not asserted.
    pending_gemm_table: bool,
    parts: Vec<Part>,
    target: usize,
    /// The alone step's rows the target's rows correspond to (a chunk covers part of them).
    alone_rows: (usize, usize),
}

struct Harness {
    exec: DecoderExecutor,
    storage: DeviceBuffer,
    layout: turbine_core::types::KvLayout,
    num_blocks: u32,
}

impl Harness {
    fn new(model_dir: &Path) -> Harness {
        let ctx = turbine_kernels::test_support::open_context("hip");
        let mem: Arc<dyn DeviceMemory> = ctx.clone();
        let provider = shim_provider(ctx);
        let cfg = load_model_config(model_dir).expect("config.json");
        let index = SafetensorsIndex::open(model_dir).expect("open safetensors");
        let slots = cfg.family.0.weight_slots(&cfg);
        let weights =
            WeightLoader::load(&index, &slots, &mem, MAX_STAGING_BYTES).expect("load weights");
        let opts = ExecutorOptions::default();
        let order = [provider.id()];
        let card = provider.card_profile();
        let registry = KernelRegistry::build(
            vec![provider],
            &order,
            &executor::requirements(&cfg, BLOCK_TOKENS, opts),
            &KernelMetrics::register(&MetricsRegistry::new()),
            card,
        )
        .expect("every op has a provider");
        let exec = DecoderExecutor::new(
            &cfg,
            families::olmoe::decoder_spec(),
            weights,
            Arc::new(registry),
            Arc::clone(&mem),
            ExecutorLimits {
                block_tokens: BLOCK_TOKENS,
                max_batch_tokens: MAX_BATCH_TOKENS,
                max_seqs: MAX_SEQS,
            },
            opts,
        )
        .expect("executor");
        let layout = *exec.kv_layout();
        let num_blocks = MAX_SEQS * BLOCKS_PER_SEQ;
        let storage = DeviceBuffer::alloc(
            &mem,
            (layout.block_bytes() * u64::from(num_blocks)) as usize,
        )
        .expect("pool");
        Harness {
            exec,
            storage,
            layout,
            num_blocks,
        }
    }

    /// Runs `parts` as one batch; returns every sequence's logits row and the trace.
    fn run(&mut self, parts: &[Part], trace: bool) -> (Vec<Vec<f32>>, Vec<TraceTensor>) {
        let kv = KvPoolView {
            storage: &self.storage,
            layout: self.layout,
            num_blocks: self.num_blocks,
            layer_stride_bytes: self.layout.block_bytes() / u64::from(self.layout.num_layers)
                * u64::from(self.num_blocks),
        };
        let tables: Vec<Vec<BlockId>> = parts
            .iter()
            .map(|p| {
                (0..BLOCKS_PER_SEQ)
                    .map(|b| BlockId(p.seq * BLOCKS_PER_SEQ + b))
                    .collect()
            })
            .collect();
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut seqs = Vec::new();
        for (p, table) in parts.iter().zip(&tables) {
            let q_start = tokens.len() as u32;
            tokens.extend_from_slice(&p.tokens);
            positions.extend(p.start..p.start + p.tokens.len() as u32);
            seqs.push(SeqSlice {
                seq: SeqId(u64::from(p.seq) + 1),
                q_start,
                q_len: p.tokens.len() as u32,
                kv_len: p.start + p.tokens.len() as u32,
                block_table: table,
                reduce: None,
            });
        }
        self.exec.set_trace(trace);
        let logits = self
            .exec
            .forward(&BatchInput {
                tokens: &tokens,
                positions: &positions,
                seqs: &seqs,
                kv: &kv,
            })
            .expect("forward");
        let trace = self.exec.take_trace();
        self.exec.set_trace(false);
        let rows = (0..parts.len()).map(|s| logits.row(s).to_vec()).collect();
        (rows, trace)
    }
}

/// The target's rows of one traced tensor: token rows `[q_start, q_start + len)`, or its
/// sequence row for `final_norm` and `logits`.
fn target_rows(t: &TraceTensor, seq_index: usize, q_start: usize, len: usize) -> &[f32] {
    let cols = t.shape[1];
    if t.layer.is_none() && matches!(t.name, "final_norm" | "logits") {
        &t.data[seq_index * cols..(seq_index + 1) * cols]
    } else {
        &t.data[q_start * cols..(q_start + len) * cols]
    }
}

/// The alone rows `[r0, r0 + len)` of a traced tensor of the alone step (one sequence).
fn alone_rows(t: &TraceTensor, r0: usize, len: usize) -> &[f32] {
    let cols = t.shape[1];
    if t.layer.is_none() && matches!(t.name, "final_norm" | "logits") {
        &t.data[..cols]
    } else {
        &t.data[r0 * cols..(r0 + len) * cols]
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter().zip(b).fold((0f32, 0usize), |(m, n), (x, y)| {
        if x.to_bits() == y.to_bits() {
            (m, n)
        } else {
            (m.max((x - y).abs()), n + 1)
        }
    })
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Top-1 minus top-2 logit.
fn margin(row: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in row {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

#[derive(Default)]
struct Summary {
    steps: usize,
    logits_differ: usize,
    argmax_differ: usize,
    max_logits_abs: f32,
    first_ops: std::collections::BTreeMap<String, usize>,
}

#[test]
#[ignore = "needs a HIP device, TURBINE_KERNEL_LIBRARY and TURBINE_TEST_MOE_MODEL_DIR"]
fn olmoe_rows_are_batch_invariant() {
    if !turbine_kernels::test_support::require_backend("hip") {
        return;
    }
    let model_dir = turbine_kernels::test_support::require_env_dir("TURBINE_TEST_MOE_MODEL_DIR");
    let targets: Vec<String> = std::env::var("TURBINE_BATCH_TARGETS")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "p10,p14".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let steps: usize = std::env::var("TURBINE_BATCH_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let refs = references();
    let mut h = Harness::new(&model_dir);
    let mut failures = Vec::new();

    for target_id in &targets {
        let t_index = refs
            .iter()
            .position(|r| &r.id == target_id)
            .unwrap_or_else(|| panic!("no reference prompt {target_id}"));
        let target = &refs[t_index];
        // Sequence 0 is the target; companions 1..=15 are the other prompts (cut to
        // COMPANION_MAX tokens), prefilled once so they can decode.
        let companions: Vec<(u32, Vec<u32>)> = refs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != t_index)
            .take(15)
            .enumerate()
            .map(|(c, (_, r))| {
                let n = r.prompt_token_ids.len().min(COMPANION_MAX);
                (c as u32 + 1, r.prompt_token_ids[..n].to_vec())
            })
            .collect();
        for (seq, prompt) in &companions {
            h.run(
                &[Part {
                    seq: *seq,
                    tokens: prompt.clone(),
                    start: 0,
                }],
                false,
            );
        }
        let companion_decode = |c: usize| {
            let (seq, prompt) = &companions[c];
            Part {
                seq: *seq,
                tokens: vec![prompt[0]],
                start: prompt.len() as u32,
            }
        };
        // A prefill chunk that re-prefills a companion (its KV is rewritten in place).
        let chunk_seq = 16u32;
        let chunk_prompt: Vec<u32> = refs
            .iter()
            .find(|r| r.id == "p15" && target_id != "p15")
            .or_else(|| refs.iter().find(|r| r.id == "p13"))
            .map(|r| r.prompt_token_ids.clone())
            .expect("a companion prompt");
        let prefill_companions: Vec<Part> = companions
            .iter()
            .take(3)
            .map(|(seq, prompt)| Part {
                seq: seq + 16,
                tokens: prompt.clone(),
                start: 0,
            })
            .collect();

        let mut summaries: std::collections::BTreeMap<String, Summary> = Default::default();
        let prompt_len = target.prompt_token_ids.len();
        for step in 0..=steps.min(target.tokens.len()) {
            let (tokens, start) = if step == 0 {
                (target.prompt_token_ids.clone(), 0u32)
            } else {
                (
                    vec![target.tokens[step - 1]],
                    (prompt_len + step - 1) as u32,
                )
            };
            let me = Part {
                seq: 0,
                tokens: tokens.clone(),
                start,
            };
            let q_len = tokens.len();
            let mut scenarios = Vec::new();
            if step == 0 {
                let mut parts = vec![me.clone()];
                parts.extend(prefill_companions.iter().cloned());
                // The serving shape of concurrency 16: the prompt behind p09's long prefill in
                // one 2,048-token batch, its rows past row 2,000.
                let long: Vec<u32> = refs
                    .iter()
                    .find(|r| r.id == "p09")
                    .map(|r| r.prompt_token_ids.clone())
                    .expect("p09");
                if long.len() + q_len <= MAX_BATCH_TOKENS as usize && target_id != "p09" {
                    scenarios.push(Scenario {
                        name: "prefill behind a 2004-token prefill".into(),
                        pending_gemm_table: false,
                        parts: vec![
                            Part {
                                seq: LONG_SEQ,
                                tokens: long,
                                start: 0,
                            },
                            me.clone(),
                        ],
                        target: 1,
                        alone_rows: (0, q_len),
                    });
                }
                scenarios.push(Scenario {
                    name: "prefill + 3 prefills".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 0,
                    alone_rows: (0, q_len),
                });
                let mut parts: Vec<Part> = (0..15).map(companion_decode).collect();
                parts.push(me.clone());
                scenarios.push(Scenario {
                    name: "15 decodes + prefill".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 15,
                    alone_rows: (0, q_len),
                });
                // The prompt in two chunks: the first alone, untraced (writes the same KV the
                // alone prefill writes up to split), the second compared with the alone rows.
                let split = q_len * 2 / 3;
                scenarios.push(Scenario {
                    name: "second chunk".into(),
                    pending_gemm_table: false,
                    parts: vec![Part {
                        seq: 0,
                        tokens: tokens[split..].to_vec(),
                        start: split as u32,
                    }],
                    target: 0,
                    alone_rows: (split, q_len - split),
                });
            } else {
                let mut parts = vec![me.clone()];
                parts.extend((0..15).map(companion_decode));
                scenarios.push(Scenario {
                    name: "16 decodes, first".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 0,
                    alone_rows: (0, 1),
                });
                let mut parts: Vec<Part> = (0..15).map(companion_decode).collect();
                parts.push(me.clone());
                scenarios.push(Scenario {
                    name: "16 decodes, last".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 15,
                    alone_rows: (0, 1),
                });
                let mut parts = vec![me.clone()];
                parts.extend((0..3).map(companion_decode));
                scenarios.push(Scenario {
                    name: "4 decodes".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 0,
                    alone_rows: (0, 1),
                });
                scenarios.push(Scenario {
                    name: "decode + prefill chunk".into(),
                    pending_gemm_table: false,
                    parts: vec![
                        me.clone(),
                        Part {
                            seq: chunk_seq,
                            tokens: chunk_prompt.clone(),
                            start: 0,
                        },
                    ],
                    target: 0,
                    alone_rows: (0, 1),
                });
                // Past the first 16 rows of a larger batch (a decode after 15 decodes and a
                // prefill chunk, as the engine orders rows only by arrival).
                let mut parts: Vec<Part> = (0..15).map(companion_decode).collect();
                parts.push(Part {
                    seq: chunk_seq,
                    tokens: chunk_prompt.clone(),
                    start: 0,
                });
                parts.push(me.clone());
                scenarios.push(Scenario {
                    name: "decode at row 85".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 16,
                    alone_rows: (0, 1),
                });
                let mut extra = companion_decode(0);
                extra.seq = chunk_seq;
                let mut parts: Vec<Part> = (0..15).map(companion_decode).collect();
                parts.push(extra);
                parts.push(me.clone());
                scenarios.push(Scenario {
                    name: "17 decodes, last".into(),
                    pending_gemm_table: false,
                    parts,
                    target: 16,
                    alone_rows: (0, 1),
                });
            }
            let mut batched = Vec::new();
            for sc in &scenarios {
                if sc.name == "second chunk" {
                    let split = sc.alone_rows.0;
                    h.run(
                        &[Part {
                            seq: 0,
                            tokens: tokens[..split].to_vec(),
                            start: 0,
                        }],
                        false,
                    );
                }
                let q_start: usize = sc.parts[..sc.target].iter().map(|p| p.tokens.len()).sum();
                let (rows, trace) = h.run(&sc.parts, true);
                let own: Vec<(Option<usize>, &'static str, Vec<f32>)> = trace
                    .iter()
                    .map(|t| {
                        (
                            t.layer,
                            t.name,
                            target_rows(t, sc.target, q_start, sc.parts[sc.target].tokens.len())
                                .to_vec(),
                        )
                    })
                    .collect();
                batched.push((rows[sc.target].clone(), own));
            }
            // Alone last: its K/V is the history every later step reads.
            let (alone_logits, alone_trace) = h.run(std::slice::from_ref(&me), true);
            let alone_row = &alone_logits[0];
            for (sc, (row, own)) in scenarios.iter().zip(&batched) {
                let (r0, len) = sc.alone_rows;
                let mut first = None;
                let mut worst: Vec<String> = Vec::new();
                assert_eq!(own.len(), alone_trace.len(), "{}: trace lengths", sc.name);
                for ((layer, name, got), want) in own.iter().zip(&alone_trace) {
                    assert_eq!((*layer, *name), (want.layer, want.name));
                    let want = alone_rows(want, r0, len);
                    let (d, n) = max_abs(want, got);
                    if n > 0 {
                        let op = format!(
                            "{}.{}",
                            layer.map_or_else(|| "-".to_string(), |l| l.to_string()),
                            name
                        );
                        if first.is_none() {
                            first = Some(format!("{op} (max_abs {d:.3e}, {n}/{})", want.len()));
                        }
                        if worst.len() < 6 {
                            worst.push(format!("{op}={d:.2e}"));
                        }
                    }
                }
                let (dl, nl) = max_abs(alone_row, row);
                let flip = argmax(alone_row) != argmax(row);
                println!(
                    "batch_invariance {target_id} step {step:>2} [{}]: logits max_abs {dl:.3e} ({nl} differ) argmax_flip={flip} margin={:.3} first_op={} ops={}",
                    sc.name,
                    margin(alone_row),
                    first.as_deref().unwrap_or("none"),
                    worst.join(" ")
                );
                let s = summaries.entry(sc.name.clone()).or_default();
                s.steps += 1;
                if nl > 0 {
                    s.logits_differ += 1;
                    if sc.pending_gemm_table {
                        println!(
                            "batch_invariance pending the per-card GEMM table: {target_id} step {step} [{}]",
                            sc.name
                        );
                    } else {
                        failures.push(format!("{target_id} step {step} [{}]", sc.name));
                    }
                }
                s.argmax_differ += usize::from(flip);
                s.max_logits_abs = s.max_logits_abs.max(dl);
                if let Some(f) = &first {
                    let op = f.split(' ').next().unwrap_or("").to_string();
                    let op = op
                        .split_once('.')
                        .map_or(op.clone(), |(_, n)| n.to_string());
                    *s.first_ops.entry(op).or_default() += 1;
                }
            }
        }
        for (name, s) in &summaries {
            println!(
                "batch_invariance summary {target_id} [{name}]: {}/{} steps with differing logits, {} argmax flips, max |Δ logit| {:.3e}, first differing op {:?}",
                s.logits_differ, s.steps, s.argmax_differ, s.max_logits_abs, s.first_ops
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{} (target, step, scenario) runs changed the target's logits bits:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
