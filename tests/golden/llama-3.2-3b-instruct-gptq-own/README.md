# Llama-3.2-3B-Instruct own GPTQ (`gptq_int4`) golden fixture

Phase 6a Task 18, spec S-11 (user decision 2026-09-30 "GPTQ INT4: which better checkpoint", option A).

- Checkpoint: our own llm-compressor GPTQ INT4 checkpoint of `unsloth/Llama-3.2-3B-Instruct`
  at revision `006f5dcd1393c3add266de40994ba96225e9689d`, made with
  `scripts/eval/gptq_calibrate.py` (`GPTQModifier`, W4A16: 4-bit int, symmetric, group 128,
  dampening 0.01, no act order, `lm_head` kept BF16). Calibration used 512 conversations of
  `HuggingFaceH4/ultrachat_200k` at revision `8049631c405ae6576f93f445c6b8166f76f5505a`
  (split `train_sft`, shard `train_sft-00000-of-00003`, shuffle seed 42, chat template,
  ≤ 2048 tokens), on one R9700 with llmcompressor 0.14.0 and compressed-tensors 0.19.0.
  Written as compressed-tensors `pack-quantized` under
  `/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own` on `novanas`;
  Turbine's `ct_pack_int4` packaging serves it as `gptq_int4` (weight-only W4A16, BF16
  activations). The calibration used the eager-torch GPTQ and pack-quantize paths;
  full provenance and package versions are in the [GSM8K reports](../../eval/llama-3.2-3b-instruct/README.md).
  The committed full-GSM8K results at concurrency 16 are Turbine 988/1319 = 0.7491 vs
  BF16 0.7801, drop 0.0311 ≤ 0.04 **PASS**; vLLM-ROCm on the same checkpoint also scores
  988/1319 = 0.7491 (Turbine vs vLLM drop 0.0000).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization of the
  checkpoint (each weight decoded in F32 as `(q - z) * s`, rounded to nearest-even BF16;
  decision "Fixture scripts (Task 11)"). Engine string
  `transformers-4.57.1-bf16-cpu-dequant-ct_pack_int4-act-none-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Captured 2026-09-30 on `novanas` (CPU, under `fixture.lock`)
  by `.procoder/handoff/gptq_own_fixture.sh`, with these variables and command:

  ```
  UV=/home/piwi/.local/bin/uv
  M=/home/piwi/turbine-models/llama-3.2-3b-instruct-gptq-own
  OUT=/home/piwi/turbine-ci/scratch/p6a-gptq-own/fixture
  WORK=$(mktemp -d /dev/shm/turbine-gptq-own-XXXXXX)
  run 2h "$OUT/reference.log" scripts/golden/quant_reference.py --model-dir "$M" \
    --prompts tests/golden/prompts.jsonl --out "$OUT/reference.jsonl" --act-quant none \
    --work-dir "$WORK" --keep-dequantized \
    --model-name turbine/Llama-3.2-3B-Instruct-GPTQ-own
  ```

  The driver's `run` wrapper executes `timeout "$t" "$UV" run "$@"`, redirects to the named
  log, and sets `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and empty `CUDA_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`. It copies the resulting reference into this
  directory. RoPE: the checkpoint's `config.json` carries the classic top-level `rope_theta`
  500000 and `rope_scaling` `llama3` (factor 32, original 8192; no `rope_parameters` block), so
  transformers 4.57.1 reads it as the BF16 model does (checked 2026-09-30).

- `tolerance.json`: calibrated below (the OLMoE method), floored at the BF16 Llama-3.2-3B-Instruct
  bounds.

## Tolerance calibration (the OLMoE method)

As for OLMoE (`../olmoe-1b-7b-0125-instruct/README.md`): `scripts/golden/self_spread.py` scores
eight transformers 4.57.1 variants (attention `sdpa` / `eager` × BF16 / FP32 × incremental / full
sequence) on the dequantized copy against `reference.jsonl`, teacher-forced on the reference
tokens with the exact `turbine-golden compare` rule. Run on `novanas` (CPU, `fixture.lock`,
`nice 19`, cores 12–15, 4 threads), all eight variants in one pass by the same driver:

```
ALL="bf16-sdpa-incremental,bf16-sdpa-full,bf16-eager-incremental,bf16-eager-full,fp32-sdpa-incremental,fp32-sdpa-full,fp32-eager-incremental,fp32-eager-full"
bf16=$(find "$WORK" -mindepth 1 -maxdepth 3 -name bf16 -type d | head -1)
run 20h "$OUT/spread.log" scripts/golden/self_spread.py "$bf16" "$OUT/reference.jsonl" \
  "$OUT/spread.json" "$ALL"
```

Measured 2026-09-30 (max |Δ logprob| over the reference top-5 before the first divergence, with
the prompt that sets it; "prefix ok" = first 32 greedy tokens identical or a divergence at a
reference margin < 0.5 nats; no variant misses a reference top-5 id from its top-20):

| variant                         | prefix ok | max likely   | max tail     |
| ------------------------------- | --------- | ------------ | ------------ |
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p15) | 0.0000 (p04) |
| bf16 sdpa full sequence         | 16/16     | 0.1020 (p16) | 0.6580 (p16) |
| bf16 eager incremental          | 16/16     | 0.1031 (p11) | 0.5124 (p16) |
| bf16 eager full sequence        | 16/16     | 0.0876 (p15) | 0.5790 (p16) |
| fp32 sdpa incremental           | 16/16     | 0.1012 (p05) | 1.0545 (p16) |
| fp32 sdpa full sequence         | 16/16     | 0.1012 (p05) | 1.0545 (p16) |
| fp32 eager incremental          | 16/16     | 0.1012 (p05) | 1.0546 (p16) |
| fp32 eager full sequence        | 16/16     | 0.1012 (p05) | 1.0545 (p16) |

The control reproduces the reference to within 0.000011 nats. Divergences before token 32
(p01, p02, p06, p08, p13; earliest at token 5) all sit at reference margins of 0.004–0.041 nats
and are excused by the margin rule.

The smallest two-decimal bounds every variant meets are likely 0.11 and tail 1.06. The likely
spread is inside the BF16 Llama-3.2-3B-Instruct tolerance (0.15 / 0.55, batched 0.25 / 0.75),
but the worst tail spread is 1.054593563079834 nats (fp32 eager incremental, p16), exceeding
both tail bounds. A quantized slug's tolerance is never tighter than its BF16 model's (the
Turbine-side rounding that the BF16 bounds absorb — GEMM shapes and summation order, the
attention kernels — is the same with INT4 weights), so `tolerance.json` keeps likely 0.15,
batched likely 0.25, `min_prompts_passing` 14 and the other keys unchanged, widening only tail
to 1.06 (+0.51) and batched tail to 1.06 (+0.31). Re-run the spread and re-derive the bounds if
the reference, the transformers version or the prompts change. `spread.json` remains a local
working artifact, as in the AWQ precedent.
