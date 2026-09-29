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

- `tolerance.json`: **provisional** — the BF16 Llama-3.2-3B-Instruct tolerance
  (`../llama-3.2-3b-instruct/tolerance.json`, unchanged) until the self-spread below completes. The
  golden verdicts measured under it are informational, not the gate.

## Tolerance calibration (the OLMoE method; pending)

As for OLMoE (`../olmoe-1b-7b-0125-instruct/README.md`): `scripts/golden/self_spread.py` scores
eight transformers 4.57.1 variants (attention `sdpa` / `eager` × BF16 / FP32 × incremental / full
sequence) on the dequantized copy against `reference.jsonl`, teacher-forced on the reference
tokens with the exact `turbine-golden compare` rule; the tolerance becomes the smallest
two-decimal bound every variant meets.

```
uv run scripts/golden/dequantize_checkpoint.py \
  --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct-awq --out <dequantized-dir>
uv run scripts/golden/self_spread.py <dequantized-dir> \
  tests/golden/llama-3.2-3b-instruct-awq/reference.jsonl <out.json> [<variants>]
```

Measured so far (2026-09-29; max |Δ logprob| over the reference top-5 before the first
divergence; "prefix ok" as in the OLMoE README):

| variant                          | prefix ok | max likely | max tail |
| -------------------------------- | --------- | ---------- | -------- |
| bf16 sdpa incremental (control)  | 16/16     | 0.0000     | 0.0000   |
| bf16 sdpa full sequence          | 16/16     | 0.1169     | 0.2268   |

The remaining six variants (`bf16-eager-*`, `fp32-*`) were interrupted by the novanas reboots of
2026-09-29 and are queued again on `fixture.lock`; the bounds are derived once all eight are in.
