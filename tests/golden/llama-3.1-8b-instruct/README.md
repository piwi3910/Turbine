# Llama-3.1-8B-Instruct (BF16) golden fixture

The BF16 baseline of the Phase 6a `mxfp4` proof (spec S-11; the MXFP4 checkpoint is a quantized
Llama-3.1-8B-Instruct).

- Checkpoint: `unsloth/Llama-3.1-8B-Instruct` at revision
  `4699cc75b550f9c6f3173fb80f4703b62d946aa5` (an ungated mirror of
  `meta-llama/Llama-3.1-8B-Instruct`; `/home/piwi/turbine-models/llama-3.1-8b-instruct` on
  `novanas`).
- `reference.jsonl`: `uv run scripts/golden/hf_reference.py --model-dir
  /home/piwi/turbine-models/llama-3.1-8b-instruct --prompts tests/golden/prompts.jsonl --out
  tests/golden/llama-3.1-8b-instruct/reference.jsonl --model-name unsloth/Llama-3.1-8B-Instruct`
  (novanas CPU, `nice -n 19`, 6 then 4 threads, 83 min): transformers 4.57.1, BF16, SDPA,
  incremental decode with the KV cache, FP32 LM head on the BF16 final-norm output.

## Tolerance

Provisional: `tolerance.json` holds the Llama-3.2-3B-Instruct values unchanged. The 8B BF16
self-spread (`uv run scripts/golden/self_spread.py`, the OLMoE method of
`tests/golden/olmoe-1b-7b-0125-instruct/README.md`) is queued on novanas behind the other Phase 6a
fixtures; the bounds are recalibrated from it. Until then golden verdicts against this fixture are
informational (a `--golden16` run failed once with these bounds).

## Engine summation-order spread (2026-10-06)

Decision "5p golden tolerance vs engine summation order" (B). `scripts/lab/golden-engine-spread.sh`
captured this slug on novanas GPU 0 at 2ba95514 (20 captures, free-running greedy,
top-20): `c1-cold`, `c1-warm` (prefix reuse), `c1-heuristic` (`execution.gemm_autotune=false`),
`c1-unfused` (`execution.fused_ops=false`) and `par-00`…`par-15` (16 captures at once: batched
GEMMs and CK split-KV decode attention instead of the b1 kernel). `scripts/golden/engine_spread.py`
(data: `engine_spread.json`) judges each capture against `reference.jsonl` with the compare rule
(observed) and measures the engine's own noise: per position and reference top-5 candidate, the
range of its logprob over every capture still on the reference's tokens there. Max noise: likely
0.402 (p16 position 27), tail 0.580
(p16 position 27). Triangle = today's bound + that noise (a correct engine is
within the reference's spread of the reference and within its order noise of any reordering of
itself, so the most the noise justifies); proposed = max(today, observed), capped at the triangle.
The token-identity rule (`min_identical_prefix`, `min_prompts_passing`, `margin_nats`) is unchanged.
Decision A (user, 2026-10-06): the proposed column is applied to `tolerance.json` (unchanged where proposed = today).

| bound | today | observed | triangle | proposed |
|---|---|---|---|---|
| likely | 0.15 | 0.28 | 0.56 | 0.28 **changed** |
| tail | 0.55 | 0.52 | 1.14 | 0.55 |
| likely batched | 0.25 | 0.28 | 0.66 | 0.28 **changed** |
| tail batched | 0.75 | 0.89 | 1.34 | 0.89 **changed** |

Default server c1 against the reference: prefix ok 16/16, likely 0.274 (p16),
tail 0.516 (p16).

Today's bounds are the 3B values copied provisionally; the default server already exceeds the strict likely bound at c1 (0.274 on p16) and the batched tail (0.88 in batched captures). All three moves sit well inside the triangle.
