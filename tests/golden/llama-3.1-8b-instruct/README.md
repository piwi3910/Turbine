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
