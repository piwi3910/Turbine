# Qwen3-30B-A3B-Instruct-2507-FP8 golden fixture

- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, FP32 LM head on the BF16 final-norm output
  (`scripts/golden/hf_reference.py`), weights `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` at revision
  `5a5a776300a41aaa681dd7ff0106608ef2bc90db`. The BF16 copy the reference reads is the
  checkpoint dequantized with `scripts/golden/dequantize_checkpoint.py` (novanas CPU, one fixture
  job at a time under `fixture.lock`); it is deleted after the run.
- `tolerance.json`: seeded at the quantized floor — Llama's strict and batched bounds (decision
  "Golden tolerance floor for quantized checkpoints", B) — and then re-derived from the engine
  spread below; `min_prompts_passing` 14, `min_identical_prefix` 32, `top_k` 5,
  `likely_logprob_floor` −2 and `margin_nats` 0.5 are unchanged. The rule that applies it is the
  same as every slug's (`turbine-golden compare`, `crates/turbine-model/tests/golden.rs`).

## Why the tolerance is derived from the engine spread

A 128-expert MoE at FP8 moves further against a transformers reference than any dense-model
bound: like OLMoE, a rounding difference anywhere upstream can swap the 8th routed expert of a
token, and prompts sitting near a routing or greedy flip (p13, p14, p16 here) then move by
nats, not hundredths. The floor the tolerance was seeded at was never measured against a real
Turbine run of this checkpoint, so — as for OLMoE (decision "5p golden tolerance vs engine
summation order", 2026-10-06 B) — the bounds are the spread's proposed column: max(today,
observed), capped at the triangle (today's bound + the engine's own noise, the most the
measured noise can justify). Nothing hit the cap.

## Engine summation-order and placement spread (2026-10-13)

`scripts/lab-cluster.sh qwen3moe-spread-novanas` (run `1013171613-19c22291`, tree `0497a6c2`)
captured this slug free-running (top-20) on two R9700s, 35 captures in all. The model does not
fit one GPU with KV headroom, so the family-spread scenario takes **tensor parallel 2 as the
reference-capture placement**: `c1-cold`, `c1-warm` (prefix reuse) and `par-00`…`par-15`
(16 prompt-rotated captures at once: batched GEMMs). The second placement is captured as
**expert parallel 2**: `mg-ep2-c1` and `mg-ep2-c16-00`…`mg-ep2-c16-15`.
`python3 scripts/golden/engine_spread.py <captures>/spread --slug
qwen3-30b-a3b-instruct-2507-fp8` (data: `engine_spread.json`) judges each capture against
`reference.jsonl` with the compare rule (observed) and measures the engine's own noise: per
position and reference top-5 candidate, the range of its logprob over every capture still on
the reference's tokens there. Max engine noise: likely 1.52 (p13 position 10), tail 8.13
(p16 position 12).

| placement            | prefix ok | likely (prompt)     | tail (prompt)       |
| -------------------- | --------- | ------------------- | ------------------- |
| tp2 c1 (cold = warm) | 16/16     | 0.53 (p05)          | 2.59 (p14)          |
| tp2 batched (par-*)  | min 14/16 | 1.31 (p13, par-10)  | 6.22 (p16, par-04)  |
| ep2 c1               | 16/16     | 0.46 (p02)          | 2.82 (p16)          |
| ep2 batched (mg-*)   | min 15/16 | 1.27 (p13, mg-…-03) | 4.82 (p16, mg-…-03) |

`c1-warm` is bitwise equal to `c1-cold` — prefix reuse is exact at tp2. Every capture keeps at
least 14/16 prompts (`min_prompts_passing`), with the losses excused divergences at reference
near-ties, not bound violations; no capture misses a reference top-5 id from its top-20.

| bound            | today (floor) | observed | triangle | proposed     |
| ---------------- | ------------- | -------- | -------- | ------------ |
| likely           | 0.15          | 0.54     | 1.67     | 0.54 **new** |
| tail             | 0.56          | 2.59     | 8.70     | 2.59 **new** |
| likely batched   | 0.25          | 1.32     | 1.77     | 1.32 **new** |
| tail batched     | 0.75          | 6.23     | 8.89     | 6.23 **new** |
| likely multi-GPU | 0.25          | 1.32     | 1.77     | 1.32 **new** |
| tail multi-GPU   | 0.75          | 6.23     | 8.89     | 6.23 **new** |

The strict and batched bounds move because the floor was a guess and the measurement is now
real; the derivation is the OLMoE method (decision A, 2026-10-06), not hand-picked. The
`_multi_gpu` bounds (`max_abs_logprob_diff_{likely,tail}_multi_gpu`, applied by
`turbine-golden compare --multi-gpu-bounds` at every concurrency — the lab-cluster family
scenarios gate this slug's tp2/ep2 golden legs with that flag, falling back to the batched
bounds while the keys are absent) equal the batched ones here: ep2 moves within tp2's own
batched spread (ep2 worst tail 4.82 vs tp2-batched 6.22), so the tp2 par-* captures set the
bound. Re-run the scenario and the script, and re-derive, if the reference, the prompts or the
gate placements change.
