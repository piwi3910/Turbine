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
- `tolerance.json`: the BF16 Llama bounds (|Δ logprob| ≤ 0.15 likely, ≤ 0.55 tail; batched
  0.25 / 0.75; `min_identical_prefix` 32, `top_k` 5, `margin_nats` 0.5), with
  `min_prompts_passing` 15 of the 17 prompts (Llama's 14 of 16 plus the long prompt). Rule (user
  decision 2026-09-29, "YaRN attention factor: on cos/sin"): max(transformers' own spread on this
  fixture with the factor on cos/sin, `uv run scripts/golden/yarn_self_spread.py` without
  `--fold`, the Llama BF16 bounds). The spread run (Task 28a) is recorded below; until it lands
  the bounds are the Llama floor.

  A/B that decided the placement (Task 28, teacher-forced p16 against this reference, lab test
  `turbine-model --test golden yarn_teacher_forced_vs_reference`, max |Δ| likely / tail):
  cpu-reference with the factor folded into the softmax scale (`scale × factor²`, Q19) 0.190 /
  1.488; cpu-reference with the factor on cos/sin before BF16 rounding (transformers' placement)
  0.067 / 0.395; HIP with the fold 0.099 / 1.382; transformers' own spread (`bf16-sdpa-full`)
  0.43 tail. The fold rounded q·k before the factor; Task 28a moved the factor onto cos/sin
  (kernel ABI v2.10) and the softmax scale back to `head_dim^-0.5`.

Gate: `LABBOOK_SET=phase-6a-quantization scripts/lab-bench.sh --model llama-yarn16 --golden16`
(GPU 0), golden at concurrency 1 and 16.
