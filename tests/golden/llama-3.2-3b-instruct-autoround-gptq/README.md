# Llama-3.2-3B-Instruct AutoRound GPTQ (`gptq_int4`) golden fixture

Phase 6a Task 18, spec S-11 (user decision 2026-09-30 "gptq_int4 proof checkpoint after the long-prompt probe", option B).

- Checkpoint: `kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit` at revision
  `e11f15d2291d8c343a4de84d6bb16ebf7c871dfc`, downloaded with `hf download … --revision <sha> --local-dir`
  into `/home/piwi/turbine-models/llama-3.2-3b-instruct-autoround-gptq` on `novanas`. AutoRound 0.4.5
  (500 iterations, 512 samples of 2048 tokens) exported in the AutoGPTQ format: 4-bit int, symmetric,
  group 128, `desc_act` false, FP16 scales; Turbine serves it as `gptq_int4` (weight-only W4A16, BF16
  activations). Full GSM8K at concurrency 16: 0.7566, drop 0.0235 ≤ 0.04 against BF16 0.7801
  (decision above).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, run on an exact BF16 dequantization of the
  checkpoint (196 layers decoded in F32 as `(q - z) * s`, rounded to nearest-even BF16; decision
  "Fixture scripts (Task 11)"). Engine string
  `transformers-4.57.1-bf16-cpu-dequant-gptq-act-none-fp32-logits`, prompts
  `tests/golden/prompts.jsonl`. Captured 2026-09-30 on `novanas` (CPU, under `fixture.lock`)
  by `.procoder/handoff/gptq_autoround_fixture.sh`, with these variables and command:

  ```
  UV=/home/piwi/.local/bin/uv
  M=/home/piwi/turbine-models/llama-3.2-3b-instruct-autoround-gptq
  OUT=/home/piwi/turbine-ci/scratch/p6a-gptq-autoround/fixture
  WORK=$(mktemp -d /dev/shm/turbine-gptq-ar-XXXXXX)
  run 2h "$OUT/reference.log" scripts/golden/quant_reference.py --model-dir "$M" \
    --prompts tests/golden/prompts.jsonl --out "$OUT/reference.jsonl" --act-quant none \
    --work-dir "$WORK" --keep-dequantized \
    --model-name turbine/Llama-3.2-3B-Instruct-AutoRound-GPTQ
  ```

  The driver's `run` wrapper executes `timeout "$t" "$UV" run "$@"`, redirects to the named
  log, and sets `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and empty `CUDA_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`. It copies the resulting reference into this
  directory. RoPE: the checkpoint's `config.json` carries the classic top-level `rope_theta`
  500000 and `rope_scaling` `llama3` (factor 32, original 8192; no `rope_parameters` block), so
  transformers 4.57.1 reads it as the BF16 model does (checked 2026-09-30).

- `tolerance.json`: the BF16 Llama-3.2-3B-Instruct bounds, unchanged; the spread below fits inside them.

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
| bf16 sdpa incremental (control) | 16/16     | 0.0000 (p04) | 0.0000 (p03) |
| bf16 sdpa full sequence         | 16/16     | 0.1166 (p10) | 0.2730 (p16) |
| bf16 eager incremental          | 16/16     | 0.1072 (p14) | 0.3185 (p16) |
| bf16 eager full sequence        | 16/16     | 0.1157 (p14) | 0.3349 (p16) |
| fp32 sdpa incremental           | 16/16     | 0.1017 (p12) | 0.4000 (p04) |
| fp32 sdpa full sequence         | 16/16     | 0.1017 (p12) | 0.4000 (p04) |
| fp32 eager incremental          | 16/16     | 0.1017 (p12) | 0.4000 (p04) |
| fp32 eager full sequence        | 16/16     | 0.1017 (p12) | 0.4000 (p04) |

The control reproduces the reference to within 0.00001 nats. Divergences before token 32
(p05, p06, p09, p15; earliest at token 6) all sit at reference margins of 0.0003–0.018 nats
and are excused by the margin rule.

The worst spread, likely 0.1166 (bf16 sdpa full, p10) and tail 0.4000 (fp32 eager full, p04), is
inside the BF16 Llama-3.2-3B-Instruct tolerance (0.15 / 0.55, batched 0.25 / 0.75). A quantized
slug's tolerance is never tighter than its BF16 model's (the Turbine-side rounding that the BF16
bounds absorb — GEMM shapes and summation order, the attention kernels — is the same with INT4
weights), so `tolerance.json` is the BF16 file unchanged. Re-run the spread and re-derive the
bounds if the reference, the transformers version or the prompts change. `spread.json` remains a
local working artifact, as in the AWQ precedent.
