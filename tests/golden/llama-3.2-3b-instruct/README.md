# Llama-3.2-3B-Instruct (BF16) golden fixture

The Phase 1 reference (`scripts/golden/hf_reference.py`, transformers BF16 on CPU, FP32 LM head on
the BF16 final-norm output; AGENTS.md "Golden reference from Hugging Face transformers"). The
bounds in `tolerance.json` are the Phase 1 values (0.15 / 0.55, likely floor −2) and the batched
bounds 0.25 / 0.75 (user decision 2026-09-26).

## Engine summation-order spread (2026-10-06)

Decision "5p golden tolerance vs engine summation order" (B). `scripts/lab/golden-engine-spread.sh`
captured this slug on novanas GPU 0 at 2ba95514 (20 captures, free-running greedy,
top-20): `c1-cold`, `c1-warm` (prefix reuse), `c1-heuristic` (`execution.gemm_autotune=false`),
`c1-unfused` (`execution.fused_ops=false`) and `par-00`…`par-15` (16 captures at once: batched
GEMMs and CK split-KV decode attention instead of the b1 kernel). `scripts/golden/engine_spread.py`
(data: `engine_spread.json`) judges each capture against `reference.jsonl` with the compare rule
(observed) and measures the engine's own noise: per position and reference top-5 candidate, the
range of its logprob over every capture still on the reference's tokens there. Max noise: likely
0.181 (p11 position 8), tail 0.889
(p16 position 28). Triangle = today's bound + that noise (a correct engine is
within the reference's spread of the reference and within its order noise of any reordering of
itself, so the most the noise justifies); proposed = max(today, observed), capped at the triangle.
The token-identity rule (`min_identical_prefix`, `min_prompts_passing`, `margin_nats`) is unchanged.
`tolerance.json` is unchanged until the decision.

| bound | today | observed | triangle | proposed |
|---|---|---|---|---|
| likely | 0.15 | 0.12 | 0.34 | 0.15 |
| tail | 0.55 | 0.56 | 1.44 | 0.56 **changed** |
| likely batched | 0.25 | 0.14 | 0.44 | 0.25 |
| tail batched | 0.75 | 0.56 | 1.64 | 0.75 |

Default server c1 against the reference: prefix ok 16/16, likely 0.118 (p14),
tail 0.464 (p16).

Only the strict tail moves (0.55 → 0.56: `fused_ops=false` reaches 0.554 on p16); the default server stays inside today's bounds at c1 and in every batched capture.
