# Llama-3.2-3B-Instruct GPTQ (`gptq_int4`) golden fixture

Phase 6a Task 18, spec S-11 (user decisions 2026-09-28 Q4 and Q8).

- Checkpoint: `shuyuej/Llama-3.2-3B-Instruct-GPTQ` at revision
  `dd5a311f040728fbc612eb03c8dadfae0a90552f` (AutoGPTQ, 4 bits, group 128, symmetric,
  `desc_act` false), downloaded on `novanas` with
  `hf download shuyuej/Llama-3.2-3B-Instruct-GPTQ --revision dd5a311f040728fbc612eb03c8dadfae0a90552f --local-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-gptq`.
  Turbine serves it as `gptq_int4` (weight-only W4A16, BF16 activations).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization of the
  checkpoint (each weight decoded in F32 as `(q - z) * s`, `z` the stored v1 zero plus one — 8 for
  this symmetric checkpoint, checked — rounded to nearest-even BF16; decision "Fixture scripts (Task 11)"). Engine string
  `transformers-4.57.1-bf16-cpu-dequant-gptq-act-none-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Captured 2026-09-29 on `novanas` (CPU, under `fixture.lock`):

  ```
  uv run scripts/golden/quant_reference.py \
    --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-gptq \
    --prompts tests/golden/prompts.jsonl \
    --out tests/golden/llama-3.2-3b-instruct-gptq/reference.jsonl \
    --act-quant none --work-dir /dev/shm --keep-dequantized \
    --model-name shuyuej/Llama-3.2-3B-Instruct-GPTQ
  ```

- `tolerance.json`: calibrated below (the OLMoE method), floored at the BF16 Llama-3.2-3B-Instruct
  bounds.

## Tolerance calibration (the OLMoE method)

As for OLMoE (`../olmoe-1b-7b-0125-instruct/README.md`): `scripts/golden/self_spread.py` scores
eight transformers 4.57.1 variants (attention `sdpa` / `eager` × BF16 / FP32 × incremental / full
sequence) on the dequantized copy `quant_reference.py --keep-dequantized` left behind, against
`reference.jsonl`, teacher-forced on the reference tokens with the exact `turbine-golden compare`
rule (`novanas` CPU, `fixture.lock`, `nice 19`, 12 threads):

```
uv run scripts/golden/self_spread.py <dequantized-dir> \
  tests/golden/llama-3.2-3b-instruct-gptq/reference.jsonl gptq-spread.json
```

Measured 2026-09-29 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats; no variant misses a reference top-5 id from its top-20):

| variant                         | prefix ok | max likely   | max tail     |
| ------------------------------- | --------- | ------------ | ------------ |
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p12) | 0.0000 (p16) |
| bf16 sdpa full sequence         | 16/16     | 0.1358 (p10) | 0.3033 (p04) |
| bf16 eager incremental          | 16/16     | 0.0975 (p11) | 0.9092 (p04) |
| bf16 eager full sequence        | 16/16     | 0.1356 (p10) | 0.9092 (p04) |
| fp32 sdpa incremental           | 16/16     | 0.1111 (p16) | 0.3276 (p04) |
| fp32 sdpa full sequence         | 16/16     | 0.1111 (p16) | 0.3276 (p04) |
| fp32 eager incremental          | 16/16     | 0.1111 (p16) | 0.3276 (p04) |
| fp32 eager full sequence        | 16/16     | 0.1111 (p16) | 0.3276 (p04) |

The control reproduces the reference, so the harness adds nothing. Divergences before token 32
(p02, p06, p09, p12, p14; earliest p14 at token 3) all sit at reference margins of 0.005–0.053
nats and are excused by the margin rule. The tail maximum is one prompt (p04) under eager
attention in BF16; every other variant stays at or below 0.33.

The smallest two-decimal bounds every variant meets are likely 0.14 and tail 0.91. A quantized
slug's tolerance is never tighter than its BF16 model's (the Turbine-side rounding that the BF16
bounds absorb is the same with INT4 weights), so each bound is the larger of the spread and the
BF16 Llama-3.2-3B-Instruct value (0.15 / 0.55, batched 0.25 / 0.75): `tolerance.json` holds
likely 0.15, tail 0.91, batched likely 0.25, batched tail 0.91, `min_prompts_passing` 14, the other
keys unchanged. Re-run the spread and re-derive the bounds if the reference, the transformers
version or the prompts change.
