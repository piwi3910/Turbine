# Handoff: p6a-yarn (plan Task 28, YaRN proof)

Written 2026-09-29 ~08:00 by the new 6a lead (builder killed without a handoff).
Worktree `.claude/worktrees/agent-ad603c5a7f228a0ed`, tip 9e28f1f (13 commits behind integration; all its commits
are on `phase-6a-quantization`: Tasks 26 (6e4b6ed, 42ee226), 27 (3e621c5, 7504011), 9e28f1f GSM8K CoT).
Merge `phase-6a-quantization` in before continuing.

## Uncommitted (in progress)

- `scripts/golden/hf_reference.py`: `--config-override <json>` (plan Task 28 file; looks complete).
- `scripts/golden/yarn_self_spread.py` (new): self_spread with the factor-16 override; `--fold` puts the attention
  factor where Turbine does (softmax scale × factor², cos/sin unscaled, user decision Q19).
- `scripts/golden/yarn_long_prompt.py` (new): builds `p17-long` (≈ 12k tokens from GSM8K questions).
- `tests/golden/llama-3.2-3b-instruct-yarn16/{prompts.jsonl,reference.jsonl,tolerance.json,README.md}`
  (17 prompts incl. p17-long; tolerance = BF16 Llama with min_prompts_passing 15).
- `scripts/lab/phase6-novanas-llama-yarn16.yaml`.

## Finding (open)

`lab-bench --model llama-yarn16` (commit 58fe8c0): 787.4 tok/s, golden c1 FAIL and c16 FAIL. c16: 15/17 passing
but p16 tail 1.38 and p17-long tail 1.42 (> 0.75); likely candidates are fine (≤ 0.15), greedy tokens identical.
transformers' own spread (`/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-spread.log`, bf16-sdpa-full):
p16 tail 0.43 — so plain BF16 noise does not explain 1.38.
Hypothesis under test: the folded attention factor (softmax scale × mscale² instead of scaling cos/sin, i.e.
rounding q·k differently in BF16). Running now under fixture.lock:
`yarn_self_spread.py --fold … bf16-sdpa-incremental` → `/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-fold.json`.
- If the fold alone reproduces ~1.4 tail: the fold placement costs accuracy → a design question for the
  coordinator (keep the fold with a calibrated tolerance from `--fold` spread, or scale cos/sin like transformers).
- If not: debug with `DecoderExecutor::set_trace` + the `golden hip_trace_vs_cpu_3b` lab test on p16 (release,
  `TURBINE_GOLDEN_TRACE=p16`) to find the first diverging op. Use the procoder:debug discipline.

## Exact next steps (fresh builder, default model — numerics)

1. Read the fold result; act per the two branches above; don't loosen the tolerance without the spread evidence
   and a lead/coordinator decision.
2. When golden passes: `golden.rs` slug list, `quant_fixtures_valid`/fixture test, `lab-bench --model llama-yarn16
   --golden16` and `--model llama --golden16` (unchanged), perf-log row, commit `test(golden): YaRN …`.

## Lab traps

- The long prompt's eager transformers run peaked ~30 GB RAM — a candidate cause of the 06:23 crash (unconfirmed).
  Keep fixture jobs on fixture.lock with 12 threads max and nothing else big alongside.
- ONE GPU job at a time.
