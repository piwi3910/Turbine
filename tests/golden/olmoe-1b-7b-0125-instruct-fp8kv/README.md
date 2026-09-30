# OLMoE-1B-7B-0125-Instruct with FP8 KV (`fp8_e4m3`) golden fixture

Phase 6a Task 24, spec S-13 (decision "FP8 KV golden", B′: a transformers reference with FP8 KV
emulated and a tolerance calibrated from its self-spread). The served model is the BF16 checkpoint
with `kv.dtype: fp8_e4m3` (lab-bench model `olmoe-fp8kv`).

- Checkpoint: `allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` (the BF16 model, under
  `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct` on `novanas`). The checkpoint ships no K/V scales, so the
  KV scales are 1.0.
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output; K (after RoPE) and V are
  quantize-dequantized to e4m3 with the served scales as they enter the KV cache and read back as
  BF16. Engine string `transformers-4.57.1-bf16-cpu-act-none-kv-fp8_e4m3-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Command (`novanas`, CPU, from the repository root):

  ```
  uv run scripts/golden/quant_reference.py --model-dir /home/piwi/turbine-models/olmoe-1b-7b-0125-instruct \
    --prompts tests/golden/prompts.jsonl --out reference.jsonl --kv-quant fp8_e4m3 \
    --model-name allenai/OLMoE-1B-7B-0125-Instruct
  ```

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
