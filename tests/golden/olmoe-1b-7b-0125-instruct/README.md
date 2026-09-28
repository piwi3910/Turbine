# OLMoE-1B-7B-0125-Instruct golden fixture

- `reference.jsonl`: transformers 4.57.1, BF16 on CPU, SDPA attention, incremental decode with the
  KV cache, FP32 LM head on the BF16 final-norm output (`scripts/golden/hf_reference.py`), weights
  `allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
- `tolerance.json`: calibrated to transformers' own spread (user decision "OLMoE golden gate",
  2026-09-26). It differs from Llama's on purpose; the rule that applies it is the same
  (`turbine-golden compare`, `crates/turbine-model/tests/golden.rs`).

## Why OLMoE gets its own tolerance

Each OLMoE layer routes every token to 8 of 64 experts by `torch.topk` over the softmax of BF16
router logits. A BF16 rounding difference anywhere upstream (another attention kernel, another
GEMM shape or summation order, another precision) can swap the 8th and 9th expert of a token,
which moves that token's hidden state by far more than the rounding itself. The dense Llama
tolerance (|Δ logprob| ≤ 0.15 likely, ≤ 0.55 tail) is tighter than transformers can hold against
its own reference once anything but the exact reference computation changes: the Turbine router
already matches transformers (BF16 router logits, the top-k set `torch.topk` selects,
`turbine_kernels::torch_topk`), so the remaining gap is this sensitivity, not a semantics
difference.

## Method

`uv run scripts/golden/self_spread.py <model-dir> tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl <out.json>`
(novanas CPU, one job at a time under the benchmark lock; the FP32 variants need ~28 GB of RAM)
scores eight transformers 4.57.1 variants against the committed reference with the exact
`turbine-golden compare` rule: attention `sdpa` or `eager` × compute dtype BF16 or FP32 ×
incremental decode (as the reference) or one full-sequence forward. Each variant is
teacher-forced on the reference tokens, which is exactly what the greedy comparison sees: up to
the first divergence the generated context equals the reference's, and the rule compares
nothing after it.

Measured 2026-09-26 (max |Δ logprob| over the reference top-5 before the first divergence;
"prefix ok" = first 32 greedy tokens identical or a divergence at a reference margin < 0.5 nats;
no variant misses a reference top-5 id from its top-20):

| variant                          | prefix ok | max likely | max tail |
| -------------------------------- | --------- | ---------- | -------- |
| bf16 sdpa incremental (control)  | 16/16     | 0.0000     | 0.0000   |
| bf16 sdpa full sequence          | 16/16     | 0.4236     | 1.2588   |
| bf16 eager incremental           | 14/16     | 1.0092     | 1.6569   |
| bf16 eager full sequence         | 14/16     | 1.0092     | 1.6569   |
| fp32 sdpa incremental            | 14/16     | 0.9891     | 1.5281   |
| fp32 sdpa full sequence          | 14/16     | 0.9891     | 1.5281   |
| fp32 eager incremental           | 14/16     | 0.9891     | 1.5281   |
| fp32 eager full sequence         | 14/16     | 0.9891     | 1.5281   |

The control reproduces the reference (|Δ| ≤ 8e-6), so the harness itself adds nothing. The
largest deviations come from prompts that sit near a routing or greedy flip (p10 likely 1.01,
p08 tail 1.66); every variant keeps at least 14/16 prompts.

The tolerance is the smallest two-decimal bound that every variant meets: `max_abs_logprob_diff_likely`
1.01, `max_abs_logprob_diff_tail` 1.66, `min_prompts_passing` 14; `min_identical_prefix` 32,
`top_k` 5, `likely_logprob_floor` −2 and `margin_nats` 0.5 are unchanged. The batched bounds
(`*_batched`, used by `turbine-golden compare` above concurrency 1) are the same 1.01 / 1.66: the
spread already covers a change of GEMM shape (full sequence vs incremental), attention kernel and
precision, which is what batch composition changes. Re-run the script and
re-derive the bounds if the reference, the transformers version or the prompts change.

## Multi-GPU modes (Phase 5)

This reference and tolerance gate **one GPU**. A multi-GPU run (expert, tensor or pipeline
parallelism) is gated against a one-GPU capture of the same model and commit taken in the same
lab run (`turbine-golden capture` on the one-GPU server, then `turbine-golden compare
--reference <capture> --tolerance tests/golden/<slug>/tolerance.json` at concurrency 1 with the
strict bounds and at 16 with the batched bounds); its verdict against this transformers reference
is reported for information only (user decision 2026-09-28, "P5: OLMoE golden tolerance under
expert parallelism"). The reason: this tolerance is transformers' own spread, and one GPU already
sits at its edge on some prompts (p14: likely |Δ| 1.0003 of 1.01 before diverging at token 14),
so a multi-GPU run that follows the reference further is scored on positions one GPU never
reaches. `turbine-golden positions --url <server> --reference … --prompt-id <id>` prints a
prompt's per-position |Δ| teacher-forced on the reference's tokens, to check such a case.
