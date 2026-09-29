# Llama-3.2-3B-Instruct AWQ (`awq_int4`) golden fixture

Phase 6a Task 18, spec S-11 (user decisions 2026-09-28 Q4 and Q8).

- Checkpoint: `casperhansen/llama-3.2-3b-instruct-awq` at revision
  `272b3bde867b606760447deb9a4d2719fbdfd3ae` (AutoAWQ GEMM, 4 bits, group 128, zero points; F16
  scales, embeddings and norms), downloaded on `novanas` with
  `hf download casperhansen/llama-3.2-3b-instruct-awq --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae --local-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-awq`.
  Turbine serves it as `awq_int4` (weight-only W4A16, BF16 activations).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization of the
  checkpoint (each weight decoded in F32 as `(q - z) * s`, rounded to nearest-even BF16; F16
  embeddings and norms rounded to BF16; decision "Fixture scripts (Task 11)"). Engine string
  `transformers-4.57.1-bf16-cpu-dequant-awq-act-none-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Captured 2026-09-28 on `novanas` (CPU, under `fixture.lock`):

  ```
  uv run scripts/golden/quant_reference.py \
    --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-awq \
    --prompts tests/golden/prompts.jsonl \
    --out tests/golden/llama-3.2-3b-instruct-awq/reference.jsonl \
    --act-quant none --work-dir /dev/shm --keep-dequantized \
    --model-name casperhansen/llama-3.2-3b-instruct-awq
  ```

- `tolerance.json`: calibrated below (the OLMoE method), floored at the BF16 Llama-3.2-3B-Instruct
  bounds.

## Tolerance calibration (the OLMoE method)

As for OLMoE (`../olmoe-1b-7b-0125-instruct/README.md`): `scripts/golden/self_spread.py` scores
eight transformers 4.57.1 variants (attention `sdpa` / `eager` × BF16 / FP32 × incremental / full
sequence) on the dequantized copy against `reference.jsonl`, teacher-forced on the reference
tokens with the exact `turbine-golden compare` rule. Run on `novanas` (CPU, `fixture.lock`,
`nice 19`, 12 threads), in two parts because the reboots of 2026-09-29 interrupted the first:

```
uv run scripts/golden/dequantize_checkpoint.py \
  --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-awq --out /dev/shm/p6a-int4-awq-bf16
uv run scripts/golden/self_spread.py /dev/shm/p6a-int4-awq-bf16 \
  tests/golden/llama-3.2-3b-instruct-awq/reference.jsonl awq-spread.json \
  bf16-sdpa-incremental,bf16-sdpa-full
uv run scripts/golden/self_spread.py /dev/shm/p6a-int4-awq-bf16 \
  tests/golden/llama-3.2-3b-instruct-awq/reference.jsonl awq-spread-b.json \
  bf16-eager-incremental,bf16-eager-full,fp32-sdpa-incremental,fp32-sdpa-full,fp32-eager-incremental,fp32-eager-full
```

Measured 2026-09-29 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats; no variant misses a reference top-5 id from its top-20):

| variant                         | prefix ok | max likely   | max tail     |
| ------------------------------- | --------- | ------------ | ------------ |
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p01) | 0.0000 (p03) |
| bf16 sdpa full sequence         | 16/16     | 0.1169 (p11) | 0.2268 (p03) |
| bf16 eager incremental          | 16/16     | 0.0767 (p02) | 0.3557 (p08) |
| bf16 eager full sequence        | 16/16     | 0.0945 (p11) | 0.2941 (p16) |
| fp32 sdpa incremental           | 16/16     | 0.0816 (p10) | 0.2719 (p10) |
| fp32 sdpa full sequence         | 16/16     | 0.0816 (p10) | 0.2719 (p10) |
| fp32 eager incremental          | 16/16     | 0.0816 (p10) | 0.2719 (p10) |
| fp32 eager full sequence        | 16/16     | 0.0816 (p10) | 0.2719 (p10) |

The control reproduces the reference, so the harness adds nothing. Divergences before token 32
(p05, p07, p11, p13, p15; earliest at token 5) all sit at reference margins of 0.012–0.105 nats
and are excused by the margin rule.

The smallest two-decimal bounds every variant meets are likely 0.12 and tail 0.36, both inside
the BF16 Llama-3.2-3B-Instruct tolerance (0.15 / 0.55, batched 0.25 / 0.75). A quantized slug's
tolerance is never tighter than its BF16 model's (the Turbine-side rounding that the BF16 bounds
absorb — GEMM shapes and summation order, the attention kernels — is the same with INT4 weights),
so `tolerance.json` keeps the BF16 values: likely 0.15, tail 0.55, batched 0.25 / 0.75,
`min_prompts_passing` 14, the other keys unchanged. Re-run the spread and re-derive the bounds if
the reference, the transformers version or the prompts change.
