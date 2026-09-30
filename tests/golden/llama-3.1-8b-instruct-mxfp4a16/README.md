# Llama-3.1-8B-Instruct MXFP4-A16 (`mxfp4`) golden fixture

Phase 6a Task 20, spec S-11 / S-17.

- Checkpoint: `FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16` at revision
  `14c3aca849a72df8fcc8b3a30ab8d9eed86ee646` (compressed-tensors `mxfp4` weights, 4-bit
  microscaled with a 32-element block scale, BF16 activations — W4A16; BF16 KV despite the
  checkpoint's KV16 name, embeddings and norms), downloaded on `novanas` at
  `/home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4a16`. Turbine serves it as `mxfp4`
  (weight-only, `ct_mxfp4` packaging).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with
  the KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization
  of the checkpoint (each MXFP4 block decoded to BF16, no activation fake-quantization — weight
  only). Engine string `transformers-4.57.1-bf16-cpu-dequant-ct_mxfp4-act-none-fp32-logits`,
  prompts `tests/golden/prompts.jsonl`. Captured 2026-09-30 on `novanas` (CPU, `fixture.lock`):

  ```
  uv run scripts/golden/quant_reference.py \
    --model-dir /home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4a16 \
    --prompts tests/golden/prompts.jsonl \
    --out tests/golden/llama-3.1-8b-instruct-mxfp4a16/reference.jsonl \
    --act-quant none --work-dir /dev/shm --keep-dequantized \
    --model-name FabioTrindade/Llama-3.1-8B-Instruct-W4A16KV16-MXFP4A16
  ```

- `tolerance.json`: calibrated below (the OLMoE method, restricted to the full-sequence variants
  to bound the 8B fixture's CPU time), floored at the BF16 Llama tolerance.

## Tolerance calibration (the OLMoE method, full-sequence variants only)

As for AWQ / GPTQ (`../llama-3.2-3b-instruct-awq/README.md`): `scripts/golden/self_spread.py`
scores transformers 4.57.1 variants on the dequantized copy against `reference.jsonl`,
teacher-forced on the reference tokens with the exact `turbine-golden compare` rule. Only the
full-sequence variants ran (`bf16-sdpa-full`, `bf16-eager-full`, `fp32-sdpa-full`,
`fp32-eager-full`; the incremental variants were skipped to keep the 8B fixture's CPU time down —
full-sequence already exercises a different GEMM shape than the reference's own incremental
decode, which is the source of spread this method measures). Run on `novanas` (CPU,
`fixture.lock`, `fixtures-r5c.sh`):

```
uv run scripts/golden/dequantize_checkpoint.py \
  --model-dir /home/piwi/turbine-models/llama-3.1-8b-instruct-mxfp4a16 --out <work>/bf16
uv run scripts/golden/self_spread.py <work>/bf16 \
  tests/golden/llama-3.1-8b-instruct-mxfp4a16/reference.jsonl spread.json \
  bf16-sdpa-full,bf16-eager-full,fp32-sdpa-full,fp32-eager-full
```

Measured 2026-09-30 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats; no variant misses a reference top-5 id from its top-20):

| variant                  | prefix ok | max likely   | max tail     |
| ------------------------ | --------- | ------------ | ------------ |
| bf16 sdpa full sequence  | 16/16     | 0.1369 (p08) | 0.8914 (p05) |
| bf16 eager full sequence | 16/16     | 0.1057 (p13) | 0.3715 (p16) |
| fp32 sdpa full sequence  | 16/16     | 0.0787 (p12) | 0.5366 (p06) |
| fp32 eager full sequence | 16/16     | 0.0787 (p12) | 0.5366 (p06) |

The smallest two-decimal bounds every variant meets are likely 0.14 and tail 0.90. Per the AWQ /
GPTQ rule a quantized slug's tolerance is never tighter than its BF16 model's (0.15 / 0.55,
batched 0.25 / 0.75): the measured likely bound (0.14) is inside the BF16 floor, so
`tolerance.json` keeps 0.15; the measured tail bound (0.90) is above both the strict (0.55) and
batched (0.75) BF16 floors, so `tolerance.json` uses the measured 0.90 for both the strict and
the batched tail (as `../llama-3.2-3b-instruct-fp8/README.md` does when the measured spread
exceeds the batched floor too). Final: likely 0.15 / 0.15 batched, tail 0.90 / 0.90 batched,
`min_prompts_passing` 14, the other keys unchanged. Re-run the spread and re-derive the bounds if
the reference, the transformers version or the prompts change.

## p05: knife-edge decode position (2026-09-30) — `mxfp4` stays `experimental`

`lab-bench --model llama8b-mxfp4 --golden16` passes 15/16 prompts at c1 and c16 (identical numbers); p05 misses at one
position: prefix 32/32, likely 0.3066 (bound 0.15), tail 6.0156 (bound 0.90), reference top-5 id 18476 missing at
position 5. Teacher-forced prefill at that position is within bounds (tail 0.58); every execution switch off changes
nothing; decode and prefill first differ by ≈ 1e-3 in layer-1 attention and the gap doubles per layer from layer 8
(`mxfp4_decode_vs_prefill_trace` in `crates/turbine-model/tests/golden.rs`); Turbine's scalar CPU prefill flips the
position too. No kernel bug was found.

transformers' decode-shaped (incremental) variants on p05 do NOT flip it (`self_spread.py` with the four
`*-incremental` variants on a reference file holding only p05, `scratch/mxfp4_inc_spread.sh`, novanas 2026-09-30):

| variant                         | prefix ok | max likely | max tail | missing |
| ------------------------------- | --------- | ---------- | -------- | ------- |
| bf16 sdpa incremental (control) | 1/1       | 0.0000     | 0.0000   | —       |
| bf16 eager incremental          | 1/1       | 0.0520     | 0.1985   | —       |
| fp32 sdpa incremental           | 1/1       | 0.0407     | 0.4003   | —       |
| fp32 eager incremental          | 1/1       | 0.0407     | 0.4003   | —       |

So the tolerance above is unchanged and the `mxfp4` row stays `experimental` (user decision 2026-09-30,
`.procoder/ask/decisions.md`). Phase 7 traces Turbine's CPU path against transformers op by op on p05 (where
intermediates round to BF16) before the row is revisited.
