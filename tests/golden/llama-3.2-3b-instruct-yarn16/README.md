# Llama-3.2-3B-Instruct with factor-16 YaRN: golden fixture

The Phase 6a YaRN proof (spec S-16, plan Task 28). Turbine serves the BF16 Llama checkpoint with
`model.rope_scaling` replacing its `llama3` scaling
(`scripts/lab/phase6-novanas-llama-yarn16.yaml`):

```yaml
rope_scaling: {rope_type: yarn, factor: 16.0, original_max_position_embeddings: 8192, beta_fast: 32, beta_slow: 1}
```

and is compared with transformers' native YaRN on the same weights carrying that `config.json`.

- `prompts.jsonl`: the 16 golden prompts (`tests/golden/prompts.jsonl`, byte for byte) plus
  `p17-long`, a 12,030-token chat prompt made of the first 190 GSM8K questions of
  `tests/eval/gsm8k-200.jsonl`, numbered and concatenated, followed by a request to repeat
  question 1 word for word. Answering it needs attention from the last position back to the
  start of the context, past the 8,192 positions the override treats as the original context.
  `turbine-golden compare` reads `prompts.jsonl` beside the reference by default, so one file
  holds both sets. Written by
  `uv run scripts/golden/yarn_long_prompt.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct --out tests/golden/llama-3.2-3b-instruct-yarn16/prompts.jsonl`
  (the tokenizer only counts tokens; the first prompt reaching 12,000 tokens is kept).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output, weights `unsloth/Llama-3.2-3B-Instruct` at
  revision `006f5dcd1393c3add266de40994ba96225e9689d`, written on novanas (under a shared
  `flock` on the benchmark lock) by

  ```sh
  uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct \
    --prompts tests/golden/llama-3.2-3b-instruct-yarn16/prompts.jsonl \
    --out tests/golden/llama-3.2-3b-instruct-yarn16/reference.jsonl \
    --config-override '{"rope_scaling":{"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":8192,"beta_fast":32,"beta_slow":1}}'
  ```

  `--config-override` replaces `config.json`'s `rope_scaling` before the model loads (the same
  as a copy of the checkpoint carrying that `config.json`, without copying the weights); the
  script logged `rotary yarn, attention scaling 1.2772588722239782`, the attention factor that
  multiplies cos and sin (transformers and, from plan Task 28a, Turbine's rope op: kernel ABI
  v2.10 `turbine_rope_desc.attn_factor`).
- `tolerance.json`: max(transformers' own spread on this fixture with the factor on cos/sin,
  `uv run scripts/golden/yarn_self_spread.py` without `--fold`, the Llama BF16 bounds) — user
  decision 2026-09-29, "YaRN attention factor: on cos/sin". The spread run (Task 28a,
  `bf16-sdpa-incremental` / `bf16-sdpa-full` / `bf16-eager-incremental` / `fp32-sdpa-incremental`
  on p01–p16, `bf16-sdpa-incremental` alone on p17-long: it is the only variant that finishes
  12,030 tokens in reasonable CPU time) landed at max |Δ| likely 0.1864, tail 0.8378 across every
  prompt and variant — above the Llama BF16 floor (0.15 / 0.55) on both tiers, and above the
  batched floor (0.25 / 0.75) on tail but not on likely. `tolerance.json` (values rounded up from
  the spread) is therefore strict likely 0.187 (max(0.1864, 0.15)), strict tail 0.838
  (max(0.8378, 0.55)), batched likely 0.25 (max(0.1864, 0.25), the batched floor wins) and batched
  tail 0.838 (max(0.8378, 0.75), the spread wins again). `min_identical_prefix` 32, `top_k` 5,
  `margin_nats` 0.5, `min_prompts_passing` 15 of the 17 prompts (Llama's 14 of 16 plus the long
  prompt) are unchanged.

  A/B that decided the placement (Task 28, teacher-forced p16 against this reference, lab test
  `turbine-model --test golden yarn_teacher_forced_vs_reference`, max |Δ| likely / tail):
  cpu-reference with the factor folded into the softmax scale (`scale × factor²`, Q19) 0.190 /
  1.488; cpu-reference with the factor on cos/sin before BF16 rounding (transformers' placement)
  0.067 / 0.395; HIP with the fold 0.099 / 1.382; transformers' own spread (`bf16-sdpa-full`)
  0.43 tail. The fold rounded q·k before the factor; Task 28a moved the factor onto cos/sin
  (kernel ABI v2.10) and the softmax scale back to `head_dim^-0.5`.

  GPU proof (Task 28a lab run, kernel ABI v2.10, `attn_factor` on cos/sin): teacher-forced p16 max
  |Δ| likely/tail cpu (0.0667, 0.3945) hip (0.0968, 0.2922), p17-long hip (0.0229, 0.4038) — all
  under the new tolerance. Served `llama-yarn16` golden concurrency 1 and 16: 17/17 prompts
  passing (need 15) on both, tok/s 854.7 (64-request bench). Served `llama` (BF16, unchanged)
  golden concurrency 1 and 16: 16/16 passing (need 14) on both, tok/s 871.9 — the plain-Llama gate
  is unaffected by the change.

Gate: `LABBOOK_SET=phase-6a-quantization scripts/lab-bench.sh --model llama-yarn16 --golden16`
(GPU 0), golden at concurrency 1 and 16.
