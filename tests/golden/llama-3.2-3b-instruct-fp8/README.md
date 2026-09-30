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
