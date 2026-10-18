# Llama-3.2-3B-Instruct MXFP4 W4A4 (`mxfp4_a4`) golden fixture

Phase 7 Task 28, the `mxfp4_a4` gate.

- Checkpoint: `matmelis/Llama_3_2_3B_w_mxfp4_a_mxfp4_gptq` at revision
  `91925ffda6977d097354a99718a20e035f8af80a` (AMD Quark W4A4: fp4 weights and dynamic fp4 input
  activations per 32, E8M0 scales, GPTQ with static groups, SmoothQuant folded into the weights),
  on `novanas` at `/home/piwi/turbine-models/llama-3.2-3b-mxfp4-a4`. Turbine serves it as
  `mxfp4_a4` (weights on `turbine_hip_mxfp4`, each activation group quantize-dequantized to
  MXFP4; support row `amd/gfx1201/LlamaForCausalLM/mxfp4_a4/bf16/none`).
- The checkpoint is a **base** Llama-3.2-3B (no chat template and no `chat_template` key), so the
  golden is measured in **completion form**: prompts `tests/golden/llama-3.2-3b-instruct-mxfp4-a4/prompts.jsonl`
  (16 prompts, `kind: completion`, max_tokens 32), not the shared chat-form `tests/golden/prompts.jsonl`.
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization of the
  checkpoint with the checkpoint's own activation fake-quantization (196 `mxfp4` activation
  hooks). Engine string `transformers-4.57.1-bf16-cpu-dequant-quark_mxfp4-act-mxfp4-fp32-logits`,
  prompts `tests/golden/llama-3.2-3b-instruct-mxfp4-a4/prompts.jsonl`. Captured 2026-10-18 on
  `novanas` (CPU, `fixture.lock`):

  ```
  uv run scripts/golden/quant_reference.py \
    --model-dir /home/piwi/turbine-models/llama-3.2-3b-mxfp4-a4 \
    --prompts tests/golden/llama-3.2-3b-instruct-mxfp4-a4/prompts.jsonl \
    --out tests/golden/llama-3.2-3b-instruct-mxfp4-a4/reference.jsonl \
    --act-quant mxfp4 --work-dir <work> --keep-dequantized \
    --model-name matmelis/Llama_3_2_3B_w_mxfp4_a_mxfp4_gptq
  ```

- `tolerance.json`: calibrated to transformers' own spread below (the OLMoE method, restricted to
  the full-sequence variants to bound the fixture's CPU time, as
  `../llama-3.1-8b-instruct-mxfp4a16/README.md` does), floored at the BF16 Llama-3.2-3B tolerance.

## Tolerance calibration (the OLMoE method, full-sequence variants only)

`scripts/golden/self_spread.py` scores transformers 4.57.1 variants on the dequantized copy against
`reference.jsonl`, teacher-forced on the reference tokens with the exact `turbine-golden compare`
rule. The incremental variants were skipped to bound the CPU time (full-sequence already changes
the GEMM shape from the reference's incremental decode; the W4A4 activation fake-quant is the
dominant source of spread here). Run on `novanas` (CPU, `fixture.lock`, `p7-t42/mxfp4a4_fixture.sh`):

```
uv run scripts/golden/self_spread.py <work>/bf16 \
  tests/golden/llama-3.2-3b-instruct-mxfp4-a4/reference.jsonl spread.json \
  bf16-sdpa-full,bf16-eager-full,fp32-sdpa-full,fp32-eager-full --act-quant mxfp4
```

Measured 2026-10-18 (max |Δ logprob| over the reference top-5 before the first divergence; "prefix
ok" = first 32 greedy tokens identical or a divergence at a reference margin < 0.5 nats; "missing"
= prompts where the variant's top-20 misses a reference top-5 id):

| variant                  | prefix ok | max likely | max tail | missing (top-20)              |
| ------------------------ | --------- | ---------- | -------- | ----------------------------- |
| bf16 sdpa full sequence  | 11/16     | 0.8533     | 2.7592   | 2 (p04, p07)                  |
| bf16 eager full sequence | 10/16     | 1.1251     | 1.4903   | 2 (p09, p16)                  |
| fp32 sdpa full sequence  | 11/16     | 1.4483     | 4.0080   | 2 (p06, p10)                  |
| fp32 eager full sequence | 10/16     | 1.4483     | 4.0080   | 2 (p06, p10)                  |

The smallest two-decimal bounds every variant meets are likely 1.45 and tail 4.01, with the fewest
passing prompts 10/16. Floored at the BF16 model's strict bound (0.15 / 0.56) both measured values
are already above it, so `tolerance.json` uses 1.45 / 4.01 for the strict and the batched bounds,
`min_prompts_passing` 10, and the other keys unchanged (`min_identical_prefix` 32, `top_k` 5,
`likely_logprob_floor` −2, `margin_nats` 0.5). This is a loose bound: the W4A4 activation
fake-quantization moves the reference's own top-5 further than the dense BF16 spread does, so the
gate is weak against the transformers reference by construction; the accuracy gate (GSM8K) is the
binding one for this row. Re-run the spread and re-derive the bounds if the reference, the
transformers version or the prompts change.
