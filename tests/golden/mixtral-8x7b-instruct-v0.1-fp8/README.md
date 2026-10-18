# Mixtral-8x7B-Instruct-v0.1 FP8 (per-tensor) golden fixture

- Checkpoint: `RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8` at revision
  `9a978d5cc7dabc37b3d13d6501426081c99c4cbd` (compressed-tensors FP8 e4m3 weights with static
  per-tensor activation scales; `lm_head` BF16), on `novanas` in
  `/home/piwi/turbine-models/mixtral-8x7b-instruct-v0.1-fp8`. Turbine serves it as `fp8` (W8A8);
  the ~47 GB of weights need both R9700s, so the gated placements are tensor parallel 2 and
  expert parallel 2 (no one-GPU support row).
- `reference.jsonl`: `scripts/golden/quant_reference.py` (transformers 4.57.1, BF16 on CPU, FP32
  LM head on the BF16 final-norm output) on the dequantized copy with FP8 per-tensor activation
  fake-quantization (`--act-quant fp8_tensor`, 896 activation hooks; fused projections share the
  largest static `input_scale`, the loader's and vLLM's rule), generated 2026-10-14 on `cpuhost`
  (guarded chain `p7_tol3_mixtral.sh`, stage 1):

  ```
  uv run scripts/golden/quant_reference.py --model-dir <bf16> --dequantized <bf16> \
      --prompts prompts.jsonl --out reference-actquant.jsonl --act-quant fp8_tensor \
      --model-name RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8
  ```

  It replaces the 2026-10-13 reference generated WITHOUT the act-quant hooks (commit `64c254f5`,
  kept in git history): no FP8-serving run — Turbine's own tp2-c1 capture and transformers with
  the same hooks alike — matched that one on 7/16 prompts (tail |Δ| to 20.1 nats at reference
  margins > 0.5); the act-quant gap, not engine noise, moves those prompts (user decision
  2026-10-13 B, `.procoder/ask/decisions.md` "P7 Mixtral golden tolerance base").

- `tolerance.json`: from the self-spread with the same per-tensor activation hooks
  (`scripts/golden/self_spread.py --act-quant fp8_tensor`, teacher-forced on the reference
  tokens, `cpuhost`, 2026-10-14). Four bf16 variants were planned (fp32 needs ~187 GB; `cpuhost`
  has 125 GB):

  | variant                         | prefix ok                                               | max likely   | max tail     |
  | ------------------------------- | ------------------------------------------------------- | ------------ | ------------ |
  | bf16 sdpa incremental (control) | 16/16                                                   | 0.0000       | 0.0000       |
  | bf16 sdpa full sequence         | 11/16                                                   | 1.1395 (p13) | 7.0402 (p11) |
  | bf16 eager incremental          | 13/16                                                   | 3.3457 (p14) | 6.0961 (p14) |
  | bf16 eager full sequence        | 12/16 of the 13 prompts done so far — PARTIAL, DEFERRED | 1.0269 (p10) | 4.7314 (p12) |

  **The fourth variant is deferred, not derived from** (user rule 2026-10-14: never load fixture
  models that do not fit comfortably in host RAM). The ~93 GB eager-full residency thrashed
  against a busy Pool0 (the LUN storm wedged the zvol taskq six times); `self_spread.py`
  checkpoints per prompt since `bd026dc6`, so its banked rows accumulate across the quiet windows
  the demoted watcher catches. Its maxima so far sit under the completed variants'; if completion
  raises them materially, the bounds are revisited. The bounds come from the three completed
  variants: the smallest two-decimal bounds every one meets are likely 3.35 and tail 7.05, and the
  same values serve as the batched bounds (as in `../llama-3.2-3b-instruct-fp8/README.md`).

- `min_prompts_passing` is 10, set from the engine captures below, not the self-spread (worst
  completed variant 11/16): under fp8 act-quant the placement captures pass the token-identity
  rule on 10–13/16 prompts (the excursions concentrate at p14, a reference near-tie that
  placements and batch compositions resolve differently). Llama-fp8 uses 14; this slug's act-quant
  spread is far wider.
- `missing_top_k_margin_nats` is 0.5 (the same as `margin_nats`): a reference top-5 id missing
  from the candidate's top-20 at a compared position is excused when the reference margin there is
  below it. The ep2 gate's only golden-c16 failure is such a case — at p06 position 16, reference
  top-5 candidate token 28743 (margin 0.28 nats) is absent from the run's top-20 in 6/16 banked
  ep2-c16 captures (0/16 tp2, no c1), with every judged |Δ| inside the `_multi_gpu` bounds: the
  same EP summation-order rounding the multi-GPU bounds absorb flips a near-tie's top-list
  membership (lead call 2026-10-18, `.procoder/ask/decisions.md` "P7 Mixtral ep2 missing_top_k
  judging rule").

## Engine spread and multi-GPU bound (2026-10-14)

`scripts/lab-cluster.sh` run `1013172007-1c724adf` captured the slug free-running greedy (top-20)
on `novanas`: `c1-cold` on one GPU and the placement captures `mg-ep2-c1`,
`mg-tp2-c16-00..15`, `mg-ep2-c16-00..15` (34 files; no `par-*` captures — one GPU cannot hold the
weights, so the strict and batched columns stand on `c1-cold` alone).
`python3 scripts/golden/engine_spread.py <captures>/spread --slug mixtral-8x7b-instruct-v0.1-fp8`
(data: `engine_spread.json`) derives each bound: old = the tolerance today, observed = the largest
engine-vs-reference |Δ| over the captures, triangle = old + the slug's engine noise, proposed =
min(triangle, max(old, observed)) — applied per decision A of "5p golden tolerance vs engine
summation order" (2026-10-06). The `_multi_gpu` keys follow decision A of "Phase 7 Task 13: OLMoE
EP×TP drift" (2026-10-08): old = the batched bound, derived from the `mg-*` captures, applied by
`turbine-golden compare --multi-gpu-bounds` at every concurrency.

| bound            | old  | observed | triangle | proposed          |
| ---------------- | ---- | -------- | -------- | ----------------- |
| likely           | 3.35 | 1.32     | 3.35     | 3.35              |
| tail             | 7.05 | 4.93     | 7.05     | 7.05              |
| likely batched   | 3.35 | 1.32     | 3.35     | 3.35              |
| tail batched     | 7.05 | 4.93     | 7.05     | 7.05              |
| likely multi-GPU | 3.35 | 5.50     | 8.92     | 5.50 **changed**  |
| tail multi-GPU   | 7.05 | 10.89    | 15.44    | 10.89 **changed** |

Against the new reference the one-GPU `c1-cold` capture passes 12/16 (likely 1.31 at p10, tail
4.93 at p12); the ep2 captures run 10–13/16 with their worst excursions at p14 (likely 5.49,
c1; tail 10.89, c16) and the tp2 captures 10–13/16 (likely 2.04 at p10, tail 4.94 at p14) — the
multi-GPU bounds carry those; the batched bounds cover everything one GPU produces.
