# Qwen3-4B-Instruct-2507 (BF16) golden fixture

The Phase 7 Qwen3 dense gate (spec S-2, S-13; plan Task 11).

- Checkpoint: `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554` (`/home/piwi/turbine-models/qwen3-4b-instruct-2507`
  on `novanas`, fetched by `scripts/fetch-checkpoint.sh novanas qwen3-4b-instruct-2507`).
- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output (`scripts/golden/hf_reference.py`). Made on
  novanas as a fixture job under `fixture.lock` (command line `p7-ref-qwen3-4b`, `nice -n 19`,
  `taskset -c 12-15`, 4 threads):

  ```sh
  flock /home/piwi/turbine-ci/fixture.lock p7-ref-qwen3-4b.sh
  # p7-ref-qwen3-4b.sh:
  nice -n 19 taskset -c 12-15 uv run scripts/golden/hf_reference.py \
    --model-dir /home/piwi/turbine-models/qwen3-4b-instruct-2507 \
    --prompts tests/golden/prompts.jsonl \
    --out tests/golden/qwen3-4b-instruct-2507/reference.jsonl \
    --model-name Qwen/Qwen3-4B-Instruct-2507
  ```

## Tolerance

`tolerance.json` is the Llama-3.2-3B-Instruct file unchanged (plan Constraints: a Phase 7 slug
starts from the Llama bounds): `max_abs_logprob_diff_likely` 0.15, `_tail` 0.56 (the Llama tail
after the 2026-10-06 engine-spread decision A; 0.55 before it), batched 0.25 / 0.75,
`min_prompts_passing` 14, `min_identical_prefix` 32, `top_k` 5, `likely_logprob_floor` −2,
`margin_nats` 0.5. No self-spread has been run for this checkpoint; the bounds are widened only by
the `scripts/golden/self_spread.py` method (`tests/golden/olmoe-1b-7b-0125-instruct/README.md`),
with its derivation recorded here.
