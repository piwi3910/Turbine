# Llama-3.2-3B-Instruct FP8 (per-tensor) golden fixture

- Checkpoint: `RedHatAI/Llama-3.2-3B-Instruct-FP8` at revision
  `377571d314b30f1d58448499e4100e2deafe7d7d` (compressed-tensors `float-quantized`: FP8 e4m3 weights
  with one scale per tensor, static per-tensor FP8 activation scales `input_scale`; `lm_head` BF16),
  on `novanas` in `/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8`. Turbine serves it as `fp8`
  (W8A8 on `hipblaslt_fp8`); the loader expands the per-tensor weight scales per row and a fused
  stack takes the largest `input_scale` of its parts.
- `reference.jsonl`: `scripts/golden/quant_reference.py` (transformers 4.57.1, BF16 on CPU, FP32 LM
  head on the BF16 final-norm output) on the checkpoint decoded exactly as `cpu::quant` does, with
  FP8 per-tensor activation fake-quantization (`--act-quant fp8_tensor`) on every decoded linear
  layer, generated 2026-09-28 on `novanas`:

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8 \
      --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct-fp8/reference.jsonl \
      --act-quant fp8_tensor --work-dir /dev/shm --keep-dequantized --model-name RedHatAI/Llama-3.2-3B-Instruct-FP8
  ```

  Captured 2026-09-28 22:22Z, before `quant_reference.py` gave a fused projection's parts their
  largest `input_scale` (4729281, the loader's and vLLM's rule). Task 14 regenerated it with the
  current script on 2026-09-29 (`.procoder/handoff/t14_spread.sh`): same tokens, max |Δ logprob|
  0.000000 (`reference-diff.txt`: SAME), so this file is kept.
- `tolerance.json`: from the self-spread with the per-tensor activation hooks on the dequantized
  copy (`novanas` CPU, `fixture.lock`, 2026-09-29):

  ```
  uv run scripts/golden/self_spread.py <dequantized-dir> tests/golden/llama-3.2-3b-instruct-fp8/reference.jsonl \
      spread.json --act-quant fp8_tensor
  ```

  | variant                         | prefix ok | max likely   | max tail     | missing |
  | ------------------------------- | --------- | ------------ | ------------ | ------- |
  | bf16 sdpa incremental (control) | 16/16     | 0.0000 (p02) | 0.0000 (p08) | —       |
  | bf16 sdpa full sequence         | 16/16     | 0.4955 (p11) | 1.7626 (p10) | —       |
  | bf16 eager incremental          | 16/16     | 1.3005 (p12) | 2.0090 (p14) | —       |
  | bf16 eager full sequence        | 16/16     | 1.3005 (p12) | 2.0090 (p14) | —       |
  | fp32 sdpa incremental           | 15/16     | 0.8070 (p12) | 2.9875 (p16) | —       |
  | fp32 sdpa full sequence         | 15/16     | 1.3562 (p14) | 1.8248 (p04) | —       |
  | fp32 eager incremental          | 16/16     | 1.2951 (p12) | 1.5262 (p04) | —       |
  | fp32 eager full sequence        | 16/16     | 1.1913 (p12) | 1.5618 (p14) | —       |

  The smallest two-decimal bounds every variant meets are likely 1.36 and tail 2.99, above the
  BF16 Llama floor (0.15 / 0.55, batched 0.25 / 0.75), and the same values serve as the batched
  bounds, as in `../llama-3.2-3b-instruct-fp8-dynamic/README.md`; `min_prompts_passing` stays 14.
  Static per-tensor activation scales spread wider on the likely candidates than dynamic
  per-token ones (0.78) and narrower on the tail (5.08).
- Task 14 verdict (saved runs `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/run/fp8-tensor/`, judged
  against these bounds from the per-prompt maxima of `golden1.txt` / `golden16.txt`): 16/16 at c1
  and c16 (worst likely 1.1077 at p14, worst tail 2.8894 at p16; every early divergence at a
  reference margin < 0.5 nats).

## Engine summation-order spread (2026-10-06)

Decision "5p golden tolerance vs engine summation order" (B). `scripts/lab/golden-engine-spread.sh`
captured this slug on novanas GPU 0 at 2ba95514 (20 captures, free-running greedy,
top-20): `c1-cold`, `c1-warm` (prefix reuse), `c1-heuristic` (`execution.gemm_autotune=false`),
`c1-unfused` (`execution.fused_ops=false`) and `par-00`…`par-15` (16 captures at once: batched
GEMMs and CK split-KV decode attention instead of the b1 kernel). `scripts/golden/engine_spread.py`
(data: `engine_spread.json`) judges each capture against `reference.jsonl` with the compare rule
(observed) and measures the engine's own noise: per position and reference top-5 candidate, the
range of its logprob over every capture still on the reference's tokens there. Max noise: likely
1.281 (p14 position 18), tail 2.845
(p10 position 12). Triangle = today's bound + that noise (a correct engine is
within the reference's spread of the reference and within its order noise of any reordering of
itself, so the most the noise justifies); proposed = max(today, observed), capped at the triangle.
The token-identity rule (`min_identical_prefix`, `min_prompts_passing`, `margin_nats`) is unchanged.
`tolerance.json` is unchanged until the decision.

| bound | today | observed | triangle | proposed |
|---|---|---|---|---|
| likely | 1.36 | 1.09 | 2.65 | 1.36 |
| tail | 2.99 | 1.75 | 5.84 | 2.99 |
| likely batched | 1.36 | 1.42 | 2.65 | 1.42 **changed** |
| tail batched | 2.99 | 5.34 | 5.84 | 5.34 **changed** |

Default server c1 against the reference: prefix ok 15/16, likely 1.062 (p14),
tail 1.742 (p13).

The two c16 failures (p14 likely 1.40, p16 tail 5.27 in the 2026-10-06 golden c16 run): p14's likely 1.42 is order noise (range 1.28 at p14 position 18 alone). p16 is different: the reference has a 0.10-margin near-tie at position 13; most captures take the other branch there (excused, positions ≥ 13 unjudged), 3 of 16 batched captures take the reference's branch, and on it the −9.09 tail candidate (token 198) at position 28 sits at −13.10, −14.29 and −14.43 in them (range 1.33, all 4.0–5.3 below the reference): a systematic offset on that branch, not order noise. It passes the proposed batched tail only under the uniform reading of the triangle (today's 2.99 plus the slug's largest order noise, 2.85 at p10 position 12).
