# Llama-3.2-3B-Instruct with FP8 KV (`fp8_e4m3`) golden fixture

Phase 6a Task 24, spec S-13 (decision "FP8 KV golden", B′: a transformers reference with FP8 KV
emulated and a tolerance calibrated from its self-spread). The served model is the BF16 checkpoint
with `kv.dtype: fp8_e4m3` (lab-bench model `llama-fp8kv`).

- Checkpoint: `unsloth/Llama-3.2-3B-Instruct` at revision `006f5dcd1393c3add266de40994ba96225e9689d` (the BF16 model, under
  `/home/piwi/turbine-models/llama-3.2-3b-instruct` on `novanas`). The checkpoint ships no K/V scales, so the
  KV scales are 1.0.
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output; K (after RoPE) and V are
  quantize-dequantized to e4m3 with the served scales as they enter the KV cache and read back as
  BF16. Engine string `transformers-4.57.1-bf16-cpu-act-none-kv-fp8_e4m3-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Command (`novanas`, CPU, from the repository root):

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct \
    --prompts tests/golden/prompts.jsonl --out reference.jsonl --kv-quant fp8_e4m3 \
    --model-name unsloth/Llama-3.2-3B-Instruct
  ```

  The Llama reference was captured 2026-09-29 by the first Task 24 fixture job
  (`fp8kv_fixtures_t24.sh`, sdpa, which the later emulation fix does not change); the regeneration
  kept it and its four sdpa spread rows and reran only the four eager rows.
  The driver runs each step as `flock /home/piwi/turbine-ci/fixture.lock nice -n 19 taskset -c 12-15`
  with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and the GPU visibility variables empty, from a source
  tree at 0883d9b (`fix(golden): FP8 KV emulation hooks the KV cache`, the fix of the emulation that
  the first OLMoE reference and the first Llama eager rows lacked; handoff `.procoder/handoff/p6a-fp8kv-eager.md`).

- `tolerance.json`: calibrated below, floored at the BF16 Llama-3.2-3B-Instruct bounds.

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
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p12) | 0.0000 (p01) |
| bf16 sdpa full sequence         | 16/16     | 0.1410 (p10) | 0.8840 (p12) |
| bf16 eager incremental          | 16/16     | 0.3693 (p10) | 2.4302 (p16) |
| bf16 eager full sequence        | 16/16     | 0.3693 (p10) | 2.3132 (p16) |
| fp32 sdpa incremental           | 16/16     | 0.2646 (p08) | 1.4765 (p16) |
| fp32 sdpa full sequence         | 16/16     | 0.3955 (p05) | 1.6468 (p16) |
| fp32 eager incremental          | 16/16     | 0.3495 (p12) | 1.6600 (p16) |
| fp32 eager full sequence        | 16/16     | 0.3709 (p05) | 1.7272 (p16) |

No variant misses a reference top-5 id from its top-20. The control reproduces the reference to
within 0.00001 nats. Divergences before token 32 (p05, p07, p15; earliest at token 9) sit at
reference margins of at most 0.07 nats and are excused by the margin rule.

The worst spread is likely 0.3955 (fp32 sdpa full, p05) and tail 2.4302 (bf16 eager incremental,
p16), both above the BF16 Llama-3.2-3B-Instruct bounds (0.15 / 0.55, batched 0.25 / 0.75). By the
floor rule (decision "Golden tolerance floor for quantized checkpoints", B: max of the spread and
the BF16 bounds) `tolerance.json` sets likely 0.40 and tail 2.44, strict and batched alike, and
keeps `min_prompts_passing` 14 and the other keys of the BF16 file.

Re-run the spread and re-derive the bounds if the reference, the transformers version, the
emulation or the prompts change. The spread JSON stays a working artifact on `novanas`
(`/home/piwi/turbine-ci/golden-work/fp8kv/llama-3.2-3b-instruct/self_spread.json`), as in the AWQ precedent.
