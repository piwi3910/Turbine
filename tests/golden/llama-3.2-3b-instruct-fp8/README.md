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
  largest `input_scale` (4729281, the loader's and vLLM's rule); the engine string does not say
  which rule produced it. Task 14 regenerates it with the current script
  (`.procoder/handoff/t14_spread.sh`, output `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/spread/`
  on `novanas`) and replaces this file when the regenerated one differs.
- `tolerance.json`: the BF16 Llama tolerance (|Δ logprob| ≤ 0.15 likely / 0.55 tail, batched
  0.25 / 0.75), provisional until the self-spread with the per-tensor activation hooks has run on
  the dequantized copy:

  ```
  uv run scripts/golden/self_spread.py <dequantized-dir> tests/golden/llama-3.2-3b-instruct-fp8/reference.jsonl \
      spread.json --act-quant fp8_tensor
  ```

  The bounds are then re-derived as for `../llama-3.2-3b-instruct-fp8-dynamic/README.md` (the OLMoE
  method, floored at the BF16 Llama bounds). The `fp8` per-tensor support row stays unflipped
  until then.
