# Handoff: p6a-fp8kv-eager (Task 24 FP8 KV golden fixtures, reference emulation fix)

Written 2026-09-29 by the fp8kv-eager builder. Branch `p6a-fp8kv-eager` (off `phase-6a-quantization`,
merged f6756e4).

## Root cause

`quant_reference.install_kv_quant` registered a custom `AttentionInterface` name
(`turbine_fp8_kv_<base>`) and pointed `config._attn_implementation` at it. In transformers 4.57.1
that broke two things:

1. **Eager attention ran without a causal mask (Llama).** `masking_utils._preprocess_mask_arguments`
   returns no mask at all for an implementation name missing from `AttentionMaskInterface`. sdpa
   copes (`is_causal` when the mask is None). `eager_attention_forward` adds no mask, so the
   prefill attended to future tokens. This caused the Llama eager spread rows: 3–7/16 prompts,
   likely |Δ| 2.4–8.8.
2. **OLMoE never ran the wrapper.** `modeling_olmoe` in 4.57.1 still picks an attention _class_
   at construction (`OLMOE_ATTENTION_CLASSES`) and never calls a registered attention function.
   Its sdpa path therefore ran **without KV quantization**, and its eager path crashed looking up
   `eager_attention_forward`. Proof: the old `golden-work/fp8kv/olmoe-1b-7b-0125-instruct/reference.jsonl`
   is bitwise identical to the committed BF16 `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl`
   (16/16 tokens, 2560/2560 logprobs exactly equal).

The Llama FP8 KV reference is valid. It was built with sdpa, where the old wrapper did quantize.
The fixed code reproduces it bitwise: bf16-sdpa-incremental, likely |Δ| 0.0000 on p01 × 8 tokens.
The Llama sdpa spread rows are kept.

## Fix — 0883d9b `fix(golden): FP8 KV emulation hooks the KV cache …`

- The emulation now sits on the KV cache, not on the attention function. A forward pre-hook on
  each `*.self_attn` hands it a `_Fp8KvCache` view of `past_key_values`, and that view's
  `update` returns the whole K/V quantize-dequantized.
- Every family calls `update` after RoPE and before attention, both Llama (the interface) and
  OLMoE (legacy classes). The attention implementation and the mask are left untouched.
- A call without a cache raises. `self_spread.py`'s full mode now passes `use_cache=True`.
- `--self-test` gains `kv_self_test` (tiny random Llama and OLMoE, F32). It checks four things:
  - eager and sdpa agree to within 1e-4, with and without FP8 KV, in both incremental and full mode;
  - the FP8 KV changes the output by more than 1e-3;
  - each layer's scale pair reaches its own layer;
  - a call without a cache is refused.
- Mutation check: the old `install_kv_quant` fails the new self-test (Llama: eager vs sdpa off by
  4.25; OLMoE: the AttributeError).
- Small novanas CPU check (p01, 8 tokens, new code):
  - Llama: sdpa-incremental 0.0000 vs the old reference; eager 0.041 (BF16 noise).
  - OLMoE: sdpa 0.17 against the old, unquantized reference, so the quantization is now applied;
    eager runs (0.19).
  - Log: `/home/piwi/turbine-ci/scratch/fp8kv-eager/check.log`.

## Detached run (not waited for)

Script `/home/piwi/turbine-ci/scratch/fp8kv-eager/fp8kv_eager_regen.sh` runs from source tree
`…/fp8kv-eager/src`, which holds 0883d9b's `scripts/golden`. Log:
`/home/piwi/turbine-ci/scratch/fp8kv-eager/fp8kv_eager_regen.log`.

- The script started 13:57 on novanas. Its jobs run sequentially under `fixture.lock` at nice 19,
  on cores 12-15, with 4 threads.
- Queue regex: `fp8kv-eager`. Every waiter carries `FIXTURE_JOB=fp8kv-eager:<job>`, and its paths
  avoid `golden-work/fp8kv`. The lead ranks it.
- It resumes: a rerun skips whatever is complete.

Steps, with work dir `/home/piwi/turbine-ci/golden-work/kv8-eager/<slug>/`:

1. **Llama:** reruns only the 4 eager variants (`eager.json`), then merges them with the kept sdpa
   rows (`sdpa.json`) into `self_spread.json`. It keeps the old reference.
2. **OLMoE:** regenerates `reference.jsonl` (6 h timeout), then prints a sanity line: the new
   reference vs BF16 must _not_ be bitwise equal. It then runs all 8 spread variants into
   `self_spread.json` (12 h timeout).
3. **Install (only if all of it succeeded):** the old files in
   `golden-work/fp8kv/<slug>/` move to `old-kvhook/`, and the new `reference.jsonl` and
   `self_spread.json` replace them.

The last line reads `fp8kv-eager: done rc=<rc> …`, preceded by one spread table per model.

## How to judge

- `rc=0`. The line `olmoe fp8kv reference vs bf16: …` should show fewer than 16/16 bitwise-equal
  logprobs; 0/16 is expected.
- In each table, the eager rows should be at the level of the sdpa rows (Llama sdpa rows are
  16/16; the old eager rows were 3–7/16 with likely |Δ| 2.4–8.8).
- Then Task 24's finish takes over: tolerance from the spread (floor max(spread, BF16 bounds),
  decision 807aabc), and commit `tests/golden/<slug>-fp8kv/`. **The OLMoE FP8 KV golden must use
  the new reference.** Any earlier OLMoE FP8 KV golden or eval judgement against the old one
  compared against plain BF16.
- The handoff `p6a-kv-t24.md` §"If OLMoE still misses" names this reference as the comparison
  point for OLMoE's GSM8K drop. Until the rerun lands, that comparison is not valid.
- Queue: the coordinator put `fp8kv-eager` at the top of `fixture.queue` (2026-09-29 ~14:00), behind the running yarn-fold job; a collector judges the result when the log shows done.
