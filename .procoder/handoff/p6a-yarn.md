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

## Rotation 5 (2026-09-29 ~13:45, T28 builder, reads only)

Worktree `.claude/worktrees/agent-aa94fe3bfacab12b1`, branch `p6a-yarn-t28` (c06a1c8). The A/B work is
STAGED, NOT COMMITTED: the Bash PreToolUse procoder hook runs in the session cwd (the memory dir), where
prettier applies, and blocks golden.rs / README.md / tolerance.json as "unformatted"; the repo's own
`procoder check` in the worktree says they are clean (0 blocking). Commit from a session rooted in the
worktree (message in the lead's notes: `test(golden): YaRN factor-16 fixture, --config-override and the fold
A/B diagnostic (Task 28, provisional tolerance)`), then merge `phase-6a-quantization`, then gate. README's
"YaRN changes no kernel or precision" claim is corrected and tolerance marked provisional (staged).
Plan amendment (prompts.jsonl instead of prompts-long.jsonl, crates/turbine-model/tests/golden.rs instead of
benches/…) is an unstaged edit of `.procoder/plans/phase-6a-quantization.md` for a separate
`handoff(.procoder/plans/phase-6a-quantization.md): …` commit.

yarn-tf1 CPU-column NaN: not a numerical NaN. p17-long (12,030 prompt tokens) is above
`YARN_CPU_MAX_TOKENS` (1024), so the cpu/unf runners are skipped and `cols()` prints the placeholder
(argmax 4294967295, NaN). p16 in yarn-tf1: cpu-reference (folded) max likely/tail 0.190 / 1.488, hip
0.099 / 1.382; worst position 22 (tail cpu 1.488, hip 1.382). So the HIP kernels are not the cause: the
scalar reference with the same fold shows the same tail.

Running detached (do not duplicate):

- Read 1, transformers fold spread: fixture.lock job `yarn_self_spread.py --fold … bf16-sdpa-incremental`
  (pid 89591 at 13:45, 4 threads on cores 12-15), log `/home/piwi/turbine-ci/scratch/fixtures-r5/yarn-fold-r5.log`
  (ends `yarn-fold-r5: ALLDONE`), output `/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-fold.json`.
  Judge: p16 tail (and every prompt's max |Δ| likely/tail) of the fold variant vs the reference. ≈1.4 on p16
  ⇒ the fold placement alone explains Turbine's tail (design question Q19 for the coordinator); ≈0.43 (the
  unfolded bf16-sdpa-full spread) ⇒ it does not, trace per the steps above.
- Read 2, HIP/CPU/unf A/B: `/home/piwi/turbine-ci/remote/agent-aa94fe3bfacab12b1/yarn_tf_r5.sh` (copy in the
  lead scratchpad `yarn_tf_r5.sh`), log `…/agent-aa94fe3bfacab12b1/yarn-tf-r5.log` (last line
  `yarn-tf-r5: done rc=<rc>`; summary `YaRN p16 max: cpu (l, t), unf (l, t), hip (l, t)`), full table
  `…/yarn-tf-r5.out`. Queued at 09:43Z on port18000.lock → bench.gate → bench.lock behind the full-GSM8K runs.
  Judge: `unf` (cpu-reference with the attention factor on cos/sin, softmax scale unscaled) against `cpu`
  (same provider, folded). unf tail ≈ 0.43 and cpu ≈ 1.49 ⇒ the fold is the cause; both ≈ 1.49 ⇒ not the fold.
