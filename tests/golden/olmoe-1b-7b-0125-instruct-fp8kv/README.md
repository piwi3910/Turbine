# OLMoE-1B-7B-0125-Instruct with FP8 KV (`fp8_e4m3`) golden fixture

Phase 6a Task 24, spec S-13 (decision "FP8 KV golden", B′: a transformers reference with FP8 KV
emulated and a tolerance calibrated from its self-spread). The served model is the BF16 checkpoint
with `kv.dtype: fp8_e4m3` (lab-bench model `olmoe-fp8kv`).

- Checkpoint: `allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` (the BF16 model, under
  `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct` on `novanas`). The checkpoint ships no K/V scales; the
  server calibrates per-layer K and V scales at load (Phase 7 Task 36, `kv_scales::calibrate`: absmax / 448 over 32
  fixed prompts, cached as `~/.cache/turbine/kv-scales-v1-*.json`).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output; K (after RoPE) and V are
  quantize-dequantized to e4m3 with the served, calibrated scales (`--kv-scales`, the f32 bit
  patterns of the server's cache file) as they enter the KV cache and read back as BF16. Engine string `transformers-4.57.1-bf16-cpu-act-none-kv-fp8_e4m3-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Command (`novanas`, CPU, from the repository root):

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/olmoe-1b-7b-0125-instruct \
    --prompts tests/golden/prompts.jsonl --out reference.jsonl --kv-quant fp8_e4m3 \
    --kv-scales ~/.cache/turbine/kv-scales-v1-3e3d0833feedaa64-*.json \
    --model-name allenai/OLMoE-1B-7B-0125-Instruct
  ```

  To regenerate: serve the model once with `kv.dtype: fp8_e4m3` (the server logs
  `event="kv_scales_calibrated"` and writes the cache file; the key changes with the checkpoint, the
  prompt set or the calibration code), then run the command with that file. The reference of
  2026-09-29 to 2026-10-06 used scale 1.0 for every layer (the checkpoint's scales); the reference in
  the tree was regenerated on 2026-10-08 with the calibrated scales (user decision "P7 OLMoE FP8 KV
  golden reference", A). The tolerance calibration and engine-spread sections below were measured on
  the scale-1.0 reference.

  The OLMoE reference was regenerated 2026-09-29 by `fp8kv_eager_regen.sh` (the first one had no
  KV quantization applied and was bitwise equal to the BF16 reference), then all eight spread
  variants were rerun on it.
  The driver runs each step as `flock /home/piwi/turbine-ci/fixture.lock nice -n 19 taskset -c 12-15`
  with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and the GPU visibility variables empty, from a source
  tree at 0883d9b (`fix(golden): FP8 KV emulation hooks the KV cache`, the fix of the emulation that
  the first OLMoE reference and the first Llama eager rows lacked; handoff `.procoder/handoff/p6a-fp8kv-eager.md`).

- `tolerance.json`: calibrated below, floored at the BF16 OLMoE-1B-7B-0125-Instruct bounds.

## Tolerance calibration (the OLMoE method, `--kv-quant fp8_e4m3`)

`scripts/golden/self_spread.py <model-dir> reference.jsonl self_spread.json --kv-quant fp8_e4m3`
scores eight transformers 4.57.1 variants (attention `sdpa` / `eager` × BF16 / FP32 × incremental
/ full sequence), each with the same FP8 KV emulation, against `reference.jsonl`, teacher-forced
on the reference tokens with the exact `turbine-golden compare` rule.

Measured 2026-09-29 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats):

| variant                         | prefix ok | max likely   | max tail     |
| ------------------------------- | --------- | ------------ | ------------ |
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p11) | 0.0000 (p10) |
| bf16 sdpa full sequence         | 15/16     | 0.6191 (p13) | 1.3822 (p14) |
| bf16 eager incremental          | 16/16     | 1.0958 (p10) | 1.5289 (p06) |
| bf16 eager full sequence        | 16/16     | 1.0958 (p10) | 1.5289 (p06) |
| fp32 sdpa incremental           | 14/16     | 0.7643 (p05) | 1.9716 (p08) |
| fp32 sdpa full sequence         | 15/16     | 0.7894 (p05) | 1.9525 (p08) |
| fp32 eager incremental          | 15/16     | 0.7731 (p05) | 1.9544 (p08) |
| fp32 eager full sequence        | 15/16     | 0.7734 (p05) | 2.0042 (p08) |

The four FP32 variants miss one reference top-5 id of p06 from their top-20. The control
reproduces the reference to within 0.00001 nats. Prompts failing a variant's prefix rule: p05
(five variants) and p08 (fp32 sdpa incremental); the other early divergences sit at reference
margins below 0.5 nats and are excused, as expected for OLMoE's top-8 routing (see
`../olmoe-1b-7b-0125-instruct/README.md`).

The worst spread is likely 1.0958 (bf16 eager, p10) and tail 2.0042 (fp32 eager full, p08), above
the BF16 OLMoE bounds (1.01 / 1.66, batched the same). By the floor rule (decision "Golden
tolerance floor for quantized checkpoints", B) `tolerance.json` sets likely 1.10 and tail 2.01,
strict and batched alike; `min_prompts_passing` stays 14, which the worst variant (14/16) meets.

Re-run the spread and re-derive the bounds if the reference, the transformers version, the
emulation or the prompts change. The spread JSON stays a working artifact on `novanas`
(`/home/piwi/turbine-ci/golden-work/fp8kv/olmoe-1b-7b-0125-instruct/self_spread.json`), as in the AWQ precedent.

## Engine summation-order spread (2026-10-06)

Decision "5p golden tolerance vs engine summation order" (B). `scripts/lab/golden-engine-spread.sh`
captured this slug on novanas GPU 0 at 2ba95514 (20 captures, free-running greedy,
top-20): `c1-cold`, `c1-warm` (prefix reuse), `c1-heuristic` (`execution.gemm_autotune=false`),
`c1-unfused` (`execution.fused_ops=false`) and `par-00`…`par-15` (16 captures at once: batched
GEMMs and CK split-KV decode attention instead of the b1 kernel). `scripts/golden/engine_spread.py`
(data: `engine_spread.json`) judges each capture against `reference.jsonl` with the compare rule
(observed) and measures the engine's own noise: per position and reference top-5 candidate, the
range of its logprob over every capture still on the reference's tokens there. Max noise: likely
0.956 (p13 position 16), tail 1.401
(p09 position 15). Triangle = today's bound + that noise (a correct engine is
within the reference's spread of the reference and within its order noise of any reordering of
itself, so the most the noise justifies); proposed = max(today, observed), capped at the triangle.
The token-identity rule (`min_identical_prefix`, `min_prompts_passing`, `margin_nats`) is unchanged.
Decision A (user, 2026-10-06): the proposed column is applied to `tolerance.json` (unchanged where proposed = today).

| bound          | today | observed        | triangle | proposed         |
| -------------- | ----- | --------------- | -------- | ---------------- |
| likely         | 1.10  | 1.14            | 2.06     | 1.14 **changed** |
| tail           | 2.01  | 5.25 (over cap) | 3.42     | 3.42 **changed** |
| likely batched | 1.10  | 1.14            | 2.06     | 1.14 **changed** |
| tail batched   | 2.01  | 5.25 (over cap) | 3.42     | 3.42 **changed** |

Default server c1 against the reference: prefix ok 16/16, likely 0.951 (p13),
tail 5.162 (p03). All of this section was measured on the scale-1.0 reference.

Over the cap: p03 position 20 puts the −12.62 tail candidate at −17.78 in every default capture (−17.07 … −17.86 over all variants), 5.16 from the reference — a systematic offset of the FP8 KV path, not order noise, so the triangle (3.42) does not cover it and no bound is proposed beyond the cap. The `OlmoeForCausalLM/bf16/fp8_e4m3` row is `experimental`; the strict likely moves 1.10 → 1.14.

Applied bounds (decision A): likely 1.14, tail 3.42 (the cap: 2.01 plus the largest noise 1.40), strict and batched. The slug still fails the gate: c1 tail is 5.16 at p03 position 20, where the -12.62 candidate sits at -17.07 ... -17.86 in every capture, a systematic offset above the cap that no noise-justified bound covers. It stays failing (experimental row) until the Phase 7 investigation.

## Calibrated-scale reference (2026-10-08)

The reference in the tree was regenerated with the served, calibrated scales (decision "P7 OLMoE
FP8 KV golden reference", A; `--kv-scales`, the section above). The tolerance bounds were **not**
re-derived for it yet, per this README's own rule. Against the calibrated-scale reference the gate
misses only on p10: golden c1 / c16 15/16 (token rule met, need 14), p10's strict likely |Δ| 1.1450
against the 1.14 bound (0.005 over; a teacher-forced `turbine-golden positions` re-run gives worst
likely 1.0663 at position 5, tail 2.0606 at position 18 — order noise at the bound's edge, not a
systematic offset). The full-GSM8K pair with the calibrated scales passes (0.6603 vs 0.6573, drop
0.0030). Follow-up: the engine-spread re-run against this reference to re-derive the bounds
(20 captures), then the gate again.

## Engine summation-order spread, calibrated-scale reference (2026-10-09)

The spread was re-run against the regenerated reference (`golden-engine-spread.sh p7t36-fp8kv
olmoe-1b-7b-0125-instruct-fp8kv`, 20 captures at 171d243c, novanas GPU 0; data
`engine_spread.json`). Every variant is prefix ok 16/16 — with matching scales the engine and the
reference agree on the greedy tokens everywhere, and the old p03 systematic tail offset (a
scale-1.0 artifact) is gone: worst tail vs the reference is 1.3190 (p16), worst likely 1.1450
(p10). Engine noise: likely 0.9390 (p09 position 15), tail 1.4970 (p16 position 30).

| bound          | old  | observed | triangle | proposed         |
| -------------- | ---- | -------- | -------- | ---------------- |
| likely         | 1.14 | 1.15     | 2.08     | 1.15 **changed** |
| tail           | 3.42 | 2.08     | 4.92     | 3.42 unchanged   |
| likely batched | 1.14 | 1.15     | 2.08     | 1.15 **changed** |
| tail batched   | 3.42 | 2.08     | 4.92     | 3.42 unchanged   |

Decision A ("5p golden tolerance vs engine summation order") applied: strict and batched likely
1.14 → 1.15, tail 3.42 unchanged.
x
