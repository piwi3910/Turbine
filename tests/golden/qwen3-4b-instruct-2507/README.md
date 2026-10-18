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

The slug started from the Llama-3.2-3B-Instruct bounds (plan Constraints), but the first gate run
against them failed (c1 9/16 prompts, likely |Δ| up to 0.233 vs 0.15, p14 tail 0.92; all 32 greedy
tokens identical on every prompt), so the bounds are calibrated to transformers' own spread with
the `scripts/golden/self_spread.py` method (`tests/golden/olmoe-1b-7b-0125-instruct/README.md`),
following the Task 10 decision "Qwen3 bounds C — accept now, fused norm later" (2026-10-08): the
QK-norm stays unfused and the tolerance absorbs the difference.

## Self-spread derivation

`self_spread.py` (2026-10-08, novanas CPU under `fixture.lock`, job `p7-spread-qwen3-4b`,
`nice -n 19 taskset -c 12-15`, 4 threads; ~2.7 h wall; `self_spread.json` beside this README, its
log on novanas under `/home/piwi/turbine-ci/p7-t11/spread.log`) scores the eight transformers
4.57.1 variants (attention `sdpa`/`eager` × BF16/FP32 ×
incremental/full-sequence) against the committed reference with the exact `turbine-golden compare`
rule. Max |Δ logprob| over the reference top-5 before the first divergence; "prefix ok" = first 32
greedy tokens identical or a divergence at a reference margin < 0.5 nats (the only margins hit are
0.0122, 0.0416 and 0.0737, on p02/p07/p09); no variant misses a reference top-5 id from its top-20:

| variant                          | prefix ok | max likely   | max tail   |
| -------------------------------- | --------- | ------------ | ---------- |
| bf16 sdpa incremental (control)  | 16/16     | 0.0000 (9e-6) | 0.0000 (2e-5) |
| bf16 sdpa full sequence          | 16/16     | 0.2990 (p11) | 0.5046 (p10) |
| bf16 eager incremental           | 16/16     | 0.6381 (p10) | 1.0355 (p14) |
| bf16 eager full sequence         | 16/16     | 0.4422 (p10) | 0.6801 (p14) |
| fp32 sdpa incremental            | 16/16     | 0.4349 (p11) | 0.4197 (p14) |
| fp32 sdpa full sequence          | 16/16     | 0.4350 (p11) | 0.4198 (p14) |
| fp32 eager incremental           | 16/16     | 0.4350 (p11) | 0.4198 (p14) |
| fp32 eager full sequence         | 16/16     | 0.4350 (p11) | 0.4198 (p14) |

The control reproduces the reference (|Δ| ≤ 9e-6), so the harness itself adds nothing. Every
variant keeps all 16 prompts. The tolerance is the smallest two-decimal bound every variant meets:
`max_abs_logprob_diff_likely` 0.64, `max_abs_logprob_diff_tail` 1.04, `min_prompts_passing` 16;
`min_identical_prefix` 32, `top_k` 5, `likely_logprob_floor` −2 and `margin_nats` 0.5 are
unchanged. The batched bounds (`*_batched`, used above concurrency 1) are the same 0.64 / 1.04:
the spread already covers a change of GEMM shape (full sequence vs incremental), attention kernel
and precision, which is what batch composition changes. Re-run the script and re-derive the bounds
if the reference, the transformers version or the prompts change.
