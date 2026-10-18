# gpt-oss-20b golden fixture

Phase 7 Task 27 (spec S-8, S-13), fixture half. The GSM8K pair, throughput and the
`openai_mxfp4` support rows are the gate half, run against the committed reference after Task 26
merges.

## Reference

- Checkpoint: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`
  (`/home/piwi/turbine-models/gpt-oss-20b` on `novanas`; `quant_method: mxfp4`,
  `modules_to_not_convert`: `model.layers.*.self_attn`, `mlp.router`, `embed_tokens`, `lm_head`).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, FP32 LM head on the BF16 final-norm output
  (`scripts/golden/hf_reference.py`, engine string
  `transformers-4.57.1-bf16-cpu-fp32-logits`), 16 prompts × 32 tokens, greedy, `ignore_eos`. The
  weights it reads are the checkpoint dequantized to BF16
  (`/home/piwi/turbine-models/gpt-oss-20b-bf16`, 39 GB), written by
  `scripts/golden/dequantize_checkpoint.py` extended for the checkpoint's fused-expert storage:
  the experts live as `…mlp.experts.<proj>_blocks` U8 `[E, n, k/32, 16]` with `<proj>_scales`
  `[E, n, k/32]` (the same rule as `turbine-model/src/weights/mxfp4.rs`
  `fused_expert_tensor`), decoded with the script's standard MXFP4 decode (E2M1, low nibble
  first, E8M0 ldexp, F32 math rounded to BF16) and written as the fused BF16 parameter
  `<proj>` `[E, k, n]` that plain transformers' `GptOssExperts` loads — the layout
  transformers' own dequantize (`convert_moe_packed_tensors`, ending in `transpose(1, 2)`)
  produces. The decode is bit-exact against transformers' own: the generator ran with
  `--verify-transformers 2` (first 2 and last fused expert tensors, gate_up and down of the
  first and last layer, bit-equal; `turbine_dequant.json` beside the BF16 copy records the
  layers). Attention, router, embedding and `lm_head` are BF16 in the checkpoint and are copied
  as stored.
- `reasoning_effort: low` in the system message of both runs (decision Harmony A): the fixture's
  `prompts.jsonl` is the shared `tests/golden/prompts.jsonl` with `reasoning_effort: "low"`
  added to every record's `chat_template_kwargs` — `hf_reference.py` passes them to
  `apply_chat_template` and `turbine-golden` replays the same kwargs through the chat route, so
  reference and candidate tokenize identically. The `date_string` kwarg of the shared file is
  kept (unused by the gpt-oss template).
- `chat_template.jinja` (this directory): the checkpoint's Harmony template with the system
  message's `strftime_now("%Y-%m-%d")` pinned to the literal `"2026-10-16"`, byte-identical
  otherwise; `model.chat_template` in `scripts/lab/phase7-novanas-gpt-oss-20b.yaml` names it and
  `hf_reference.py --chat-template` renders the reference with it (see "Date independence"
  below). `reference.jsonl` was regenerated with it on 2026-10-18 (engine string unchanged).
## Tolerance

`tolerance.json` was seeded at the quantized floor — Llama-3.2-3B's strict and batched bounds
(decision "Golden tolerance floor for quantized checkpoints", B) — which was never measured
against a Turbine run of this checkpoint, and then re-derived from the engine spread below
(the Qwen3-MoE method, `tests/golden/qwen3-30b-a3b-instruct-2507-fp8/README.md`), not a
transformers self-spread: gpt-oss-20b is a quantized sparse-MoE checkpoint whose reference is
the dequantized MXFP4 → BF16 copy, so how far Turbine's own serving of the checkpoint moves
under a change of summation order is the metric that matters, and it is one card (no
`<family>-spread` lab-cluster scenario), so the captures come from the one-card
`scripts/lab/golden-engine-spread.sh`. `min_identical_prefix` 32, `min_prompts_passing` 14,
`top_k` 5, `likely_logprob_floor` −2 and `margin_nats` 0.5 are unchanged.

## Engine summation-order spread (2026-10-16)

`scripts/lab/golden-engine-spread.sh p7t27-gptoss gpt-oss-20b` (novanas GPU 0, commit
`a4d79dd9`) captured this slug free-running (top-20) 20 times: `c1-cold`, `c1-warm` (prefix
reuse), `c1-heuristic` (`execution.gemm_autotune=false`), `c1-unfused`
(`execution.fused_ops=false`) and `par-00`…`par-15` (16 prompt-rotated captures at once:
batched GEMMs and CK split-KV decode attention instead of the b1 kernel). `python3
scripts/golden/engine_spread.py <captures>/spread --slug gpt-oss-20b` (data: `engine_spread.json`)
judges each capture against `reference.jsonl` with the compare rule (observed) and measures the
engine's own noise: per position and reference top-5 candidate, the range of its logprob over
every capture still on the reference's tokens there. Max engine noise: likely 0.3413 (p12
position 29), tail 2.7258 (p12 position 1).

| bound          | old (floor) | observed | triangle | proposed |
| -------------- | ----------- | -------- | -------- | -------- |
| likely         | 0.15        | 0.35     | 0.50     | 0.35 **new** |
| tail           | 0.56        | 2.41     | 3.29     | 2.41 **new** |
| likely batched | 0.25        | 0.48     | 0.60     | 0.48 **new** |
| tail batched   | 0.75        | 2.64     | 3.48     | 2.64 **new** |

Every capture keeps 16/16 prompts (the token prefix rule) and no capture misses a reference
top-5 id from its top-20, so nothing hinges on the margin excuse; `c1-warm` is at the reference
level identical to `c1-cold` (both likely 0.3406 / tail 2.4044), i.e. prefix reuse changes
nothing the gate judges. The proposed column is `max(today, observed)` capped at the triangle
(today's bound plus the slug's largest order noise, the most the measured noise justifies);
nothing hit the cap, so the bounds are the observed spread. The token rule is unchanged.
Re-run the script and re-derive the bounds if the reference, the prompts or the gate placement
change.

## Date independence (2026-10-18)

**The fixture is date-independent now.** The checkpoint's Harmony `chat_template.jinja` renders the
system message as `{{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}` — the host clock in
UTC (`crates/turbine-model/src/chat_template.rs`) — for the `kind: chat` records (p10–p16); the
`date_string` kwarg the shared prompt set carries is **unused by this template**. So the rendered
prefix (and the served logprobs) used to change with the day the server ran, and the c1 gate only
passed on the reference's capture date (`kind: completion` p01–p09 send a pre-rendered `prompt`
string and were always date-free). The drift, the bisect and the root cause are in
`.procoder/perf-log.md`, "P7 gpt-oss golden c1 drift — root cause" (Tasks 39/39b).

Fix (lead call 2026-10-18, `.procoder/ask/decisions.md`, "P7 exit: gpt-oss c1 date-dependent
fixture — lead call"): `tests/golden/gpt-oss-20b/chat_template.jinja` is the checkpoint template
with `strftime_now("%Y-%m-%d")` replaced by the literal `"2026-10-16"` (the date the reference and
the Task 27 gate were captured at; one line differs, everything else byte-identical). The slug's
lab config sets `model.chat_template` to it (relative to the tree root, the server's cwd in
`lab-bench.sh` and the lab-serve Job) and `scripts/golden/hf_reference.py --chat-template` renders
the reference with the same file, so reference and candidate render identically on any day. This
keeps the fixture exercising the real server-side chat path; the alternative — pre-rendering
p10–p16 as `kind: completion` records — would drop that coverage.

The regenerated `reference.jsonl` is self-consistent at the pinned date: its p10–p16 records render
`Current date: 2026-10-16` (the committed records had p10–p13 at `2026-10-14` and p14–p16 at
`2026-10-16`, because the fixture's first generation was interrupted and resumed two days later, so
the gate itself only ever matched the p14–p16 half). The tolerance is unaffected and stays
conservative: the engine-spread captures were taken `2026-10-16` and so matched the p14–p16 records
exactly while judging p10–p13 across the date change.

## Fixture provenance

- novanas CPU, one fixture job at a time under `fixture.lock` (queue entry `p7-ref-gpt-oss`),
  `nice -n 19`, cores 12–15, 4 threads; guarded script
  `.procoder/handoff/p7_ref_gptoss.sh`, scratch `/home/piwi/turbine-ci/scratch/p7-t27`.
- Dequantize + transformers cross-check: 2026-10-14 18:34:36–18:37:21 UTC (2 m 45 s; the first
  two attempts failed on the fused-expert support the extension adds — root causes in git
  history).
- Reference: 2026-10-14, 18:40:04 UTC start, 16 prompts, ~2 min/prompt (interrupted after p13 and
  resumed on 2026-10-16 for p14–p16, which is why the committed `captured` stamps were mixed).
- Reference regenerated 2026-10-18 with the pinned template (Task 39c): guarded script
  `.procoder/handoff/p7_t39c_ref.sh`, scratch `/home/piwi/turbine-ci/scratch/p7-t39c`, same
  `fixture.lock` queue entry `p7-ref-gpt-oss` and pinning; the dequantized BF16 copy already
  existed, so only the reference stage ran.
