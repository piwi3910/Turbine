# Llama-3.2-3B-Instruct FP8 block-scaled (`fp8_block`) golden fixture

- Checkpoint: `unsloth/Llama-3.2-3B-Instruct-FP8-Block` at revision
  `08cf804398b23fab4a1df02fbe8d4d5a11a800cc` (compressed-tensors `float-quantized`: FP8 e4m3
  linear weights with F32 128×128 block scales; BF16 embedding/norms, tied head skipped), on
  `novanas` in `/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-block`. Turbine serves it as
  `fp8_block` (W8A16: own kernel `turbine_hip_fp8_block`, fused WMMA for decode m ≤ 64,
  dequantize-to-BF16 + tuned hipBLASLt GEMM above / in prefill). Activations stay BF16 — the
  checkpoint's own group-128 activation quantization scheme is not applied by the served path, so
  the reference must be generated with `--act-quant none` rather than the checkpoint's
  `fp8_group128` scheme (the dequant path multiplies `bf16(q·s)` exactly as
  `dequantize_checkpoint.py` would; the fused decode path uses the exact `q·s`).
- `reference.jsonl`: `scripts/golden/quant_reference.py` (transformers 4.57.1, BF16 on CPU, FP32
  LM head on the BF16 final-norm output) on the checkpoint decoded exactly as `cpu::quant` does,
  with no activation fake-quantization (`--act-quant none`), generated 2026-09-29 on `novanas`:

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-block \
      --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct-fp8-block/reference.jsonl \
      --act-quant none --work-dir /dev/shm --keep-dequantized --model-name unsloth/Llama-3.2-3B-Instruct-FP8-Block
  ```

  The script warns `--act-quant none differs from the checkpoint's fp8_group128` — expected: the
  served kernel is W8A16, so the reference must not apply the checkpoint's activation scheme.
- `tolerance.json`: from the self-spread with no activation hooks on the dequantized copy
  (`novanas` CPU, `fixture.lock`, 2026-09-30, `--act-quant none`):

  ```
  uv run scripts/golden/self_spread.py <dequantized-dir> tests/golden/llama-3.2-3b-instruct-fp8-block/reference.jsonl \
      spread.json --act-quant none
  ```

  | variant                         | prefix ok | max likely   | max tail     | missing |
  | -------------------------------- | --------- | ------------ | ------------ | ------- |
  | bf16 sdpa incremental (control)  | 16/16     | 0.0000 (p07) | 0.0000 (p12) | —       |
  | bf16 sdpa full sequence          | 16/16     | 0.1829 (p16) | 0.2538 (p10) | —       |
  | bf16 eager incremental           | 16/16     | 0.0827 (p14) | 0.3107 (p04) | —       |
  | bf16 eager full sequence         | 16/16     | 0.0827 (p14) | 0.3107 (p04) | —       |
  | fp32 sdpa incremental            | 16/16     | 0.1068 (p16) | 0.2650 (p04) | —       |
  | fp32 sdpa full sequence          | 16/16     | 0.1068 (p16) | 0.2650 (p04) | —       |
  | fp32 eager incremental           | 16/16     | 0.1069 (p16) | 0.2650 (p04) | —       |
  | fp32 eager full sequence         | 16/16     | 0.1068 (p16) | 0.2650 (p04) | —       |

  The smallest two-decimal bounds every variant meets are likely 0.19 and tail 0.32, both below the
  BF16 Llama floor (0.15 strict likely / 0.55 strict tail, 0.25 / 0.75 batched — user decision
  "Golden tolerance floor for quantized checkpoints"), so the tolerance is kept at the floor: likely
  0.19 (max(0.19, 0.15)), tail 0.55 (max(0.32, 0.55)), batched 0.25 / 0.75 unchanged (spread stays
  under the batched floor on both). `min_prompts_passing` stays 14.
