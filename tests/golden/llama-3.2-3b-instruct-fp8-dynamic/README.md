# Llama-3.2-3B-Instruct FP8-dynamic golden fixture

Phase 6a Task 14, spec S-11 (the `fp8` gate checkpoint).

- Checkpoint: `RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic` at revision
  `c308a86de78778c5f904a1d82401ac85e18ca205` (compressed-tensors `float-quantized`: FP8 e4m3 weights
  with one scale per output channel, dynamic per-token FP8 activations; `lm_head` BF16), on
  `novanas` in `/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic`, downloaded with
  `hf download RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic --revision c308a86de78778c5f904a1d82401ac85e18ca205 --local-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic`.
  Turbine serves it as `fp8` (W8A8 on `hipblaslt_fp8`, activations quantized per token by
  `quantize_act`).
- `reference.jsonl`: `scripts/golden/quant_reference.py` (transformers 4.57.1, BF16 on CPU, SDPA
  attention, incremental decode with the KV cache, FP32 LM head on the BF16 final-norm output) on
  the checkpoint decoded exactly as `cpu::quant` does, with FP8 per-token activation
  fake-quantization (`--act-quant fp8_token`) on every decoded linear layer (196 hooks). Engine
  string `transformers-4.57.1-bf16-cpu-dequant-ct_fp8-act-fp8_token-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Captured 2026-09-29 (03:02Z) on `novanas` (CPU, under `fixture.lock`);
  an earlier capture of 2026-09-28 has the same tokens:

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic \
      --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct-fp8-dynamic/reference.jsonl \
      --act-quant fp8_token --work-dir /dev/shm --keep-dequantized --model-name RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic
  ```

- `tolerance.json`: calibrated below (decision "Golden tolerance for activation-quantized
  checkpoints": the OLMoE method with the activation fake-quantization hooks), floored at the BF16
  Llama-3.2-3B-Instruct bounds (decision "Golden tolerance floor for quantized checkpoints").

## Tolerance calibration

`scripts/golden/self_spread.py` scores eight transformers 4.57.1 variants (attention `sdpa` /
`eager` × BF16 / FP32 × incremental / full sequence) on the dequantized copy
`quant_reference.py --keep-dequantized` left behind, with the same per-token FP8 activation hooks,
against `reference.jsonl`, teacher-forced on the reference tokens with the exact `turbine-golden
compare` rule (`novanas` CPU, `fixture.lock`, `nice 19`):

```
uv run scripts/golden/self_spread.py <dequantized-dir> \
  tests/golden/llama-3.2-3b-instruct-fp8-dynamic/reference.jsonl spread.json --act-quant fp8_token
```

Measured 2026-09-29 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats; "missing" = a reference top-5 id absent from the variant's top-20):

| variant                         | prefix ok | max likely   | max tail     | missing        |
| ------------------------------- | --------- | ------------ | ------------ | -------------- |
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p14) | 0.0000 (p16) | —              |
| bf16 sdpa full sequence         | 15/16     | 0.7570 (p04) | 2.0143 (p16) | —              |
| bf16 eager incremental          | 16/16     | 0.5328 (p14) | 2.1969 (p10) | p05 pos 23     |
| bf16 eager full sequence        | 16/16     | 0.4043 (p14) | 2.1969 (p10) | p05 pos 23     |
| fp32 sdpa incremental           | 16/16     | 0.6078 (p16) | 5.0718 (p16) | —              |
| fp32 sdpa full sequence         | 15/16     | 0.7742 (p14) | 2.7322 (p16) | —              |
| fp32 eager incremental          | 16/16     | 0.5191 (p16) | 2.3941 (p16) | —              |
| fp32 eager full sequence        | 15/16     | 0.6200 (p16) | 4.4880 (p16) | —              |

The control reproduces the reference (|Δ| ≤ 1.4e-5), so the harness adds nothing. The spread is
far wider than for the weight-only formats (GPTQ 0.14 / 0.91): with dynamic per-token FP8
activations, any upstream rounding difference moves values across FP8 bins, and that step is
amplified layer by layer. Divergences before token 32 sit at reference margins of 0.013–0.43 nats
and are excused, except p16 under full-sequence SDPA (diverges at token 29, margin 1.28) and p12
under FP32 eager full sequence (token 0, margin 0.84) — one unexcused prompt per variant at most.
The tail maximum 5.07 is p16 under FP32 SDPA; every other variant stays at or below 4.49.

The smallest two-decimal bounds every variant meets are likely 0.78 and tail 5.08, both above the
BF16 Llama bounds (0.15 / 0.55, batched 0.25 / 0.75), so `tolerance.json` holds likely 0.78, tail
5.08, and the same values for the batched bounds (the spread already spans a GEMM-shape change,
attention kernel and precision, which is what batch composition changes). `min_prompts_passing`
stays 14 (every variant keeps ≥ 15/16; never tighter than the BF16 model's 14); the other keys
are unchanged.

One spread fact the bounds cannot express: under BF16 eager attention, p05's reference top-5
candidate 9478 at position 23 (reference logprob −11.14, 0.003 nats above the sixth candidate)
drops out of the top-20, which `turbine-golden compare` counts as a logprob violation for the
whole set. No SDPA or FP32 variant misses it; a Turbine run failing on exactly this id is the
reference's near-tie (check with `turbine-golden positions --prompt-id p05`), not a kernel defect.

Re-run the spread and re-derive the bounds if the reference, the transformers version or the
prompts change.
