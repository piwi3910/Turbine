# Handoff: Phase 6a lead (rotation at ~270k tokens, 2026-09-29 ~08:40 +04)

Integration worktree `.claude/worktrees/agent-a4b513efedb95892f`, branch `phase-6a-quantization`, tip 4f45232, clean.
Lab runs from the clean detached worktree `agent-a4b513efedb95892f-lab` (lab scripts rsync uncommitted files).
Per-branch detail: `.procoder/handoff/<branch>.md` in each worktree; this file is the lead's overview.

## Integration state

- Merged: everything up to Tasks 1–13, 16, 17, 19, 21 (host), 22, 23, 25, 26, 27; 2116221 (Quark
  `kv_cache_quant_ignored`, verified by the lead: loader + detect test); 33c5d9a (handoffs, TurboQuant S final);
  4f45232 (decisions: full GSM8K for the gate misses).
- Pending merge: `p6a-kv-t24` (tip 244ce98: ca45bd9 reworded kv_gpu FP8 tests, 8fb9e90 GSM8K-200 evals, 0e877d6
  handoff, 185ccee decisions addendum extending the full-GSM8K rule to OLMoE, 244ce98 handoff). Gate ok 774.
  **185ccee re-adds the "gate misses" entry with OLMoE**; integration already has 4f45232's version — on merge,
  keep ONE entry (185ccee's text supersedes 4f45232's; resolve the conflict by hand).
- Not merged: `p6b-groundwork` (waits for 6b, starts from main after 6a closes).

## Running builders (agent IDs for SendMessage)

| Agent             | Branch                               | Task                                                       | Waiting on                                                                                                                                                  |
| ----------------- | ------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a4baec4689b995379 | p6a-fp8-t15 (or its worktree branch) | 15 fp8_block kernel + loader FP8 layout, quick tier        | its lab `hip_qgemm` (old queued run → `scratchpad/fp8/labtest_block2.txt`), gate, bench.lock                                                                |
| aa94fe3bfacab12b1 | p6a-yarn-t28                         | 28 YaRN golden tail (p16 1.38, p17-long 1.42)              | fold spread `/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-fold.json` (fixture.lock); may stop with a design question (fold vs cos/sin scaling) |
| a19ca024910301566 | p6a-gsm8k-full (worktree branch)     | full GSM8K fixture + generator                             | its gate; generated `scratchpad/gsm8k-full.jsonl` already, commit not yet on `p6a-gsm8k-full`                                                               |
| a67eec8abc8f117eb | p6a-kv-t24-full                      | full-GSM8K FP8 KV gate, Llama + OLMoE (4 runs, sequential) | the dataset commit, then bench/port locks; ~6× a 200-run each                                                                                               |

Finished (don't reuse): ad2d9c8c7cc380c42 (Task 24 wrap-up).

## Background runs (scratchpad = `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad`)

- fixture.lock queue on novanas (one at a time, hours each): FP8-dynamic `self_spread` (running; then fp8-block
  reference) via `scratchpad/fp8/fixture_chain2.sh`, outputs `/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/`;
  MXFP4 8B `fixq.sh` (`/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/scratch/fixtures/out/fixq3.log`);
  YaRN fold (above); INT4 `fixtures6.sh` (`/home/piwi/turbine-ci/scratch/p6a-int4/fixtures6.log`);
  FP8 KV golden `fp8kv_fixtures_t24.sh` (`/home/piwi/turbine-ci/remote/agent-ad2d9c8c7cc380c42/fp8kv_fixtures_t24.log`,
  output `/home/piwi/turbine-ci/golden-work/fp8kv/<slug>/`).
- GPU queue: `scratchpad/mxfp4/queue3.sh` (W4A4 8B eval → vLLM 8B-a4 → lab-bench llama8b / -mxfp4 / -mxfp4-a4);
  `scratchpad/int4/vllm_session.sh … gptq` (vLLM GPTQ, log `scratchpad/int4/vllm-session-gptq.log`);
  the old FP8 builder's `lab-test hip_qgemm` (local pid 72411).
- I killed only one stale local ssh (duplicate fp8-block reference with no remote side).

## Remaining tasks, in order

1. Merge `p6a-kv-t24` (see conflict note). Then merge Task 15 / 28 builders' branches when they report; cherry-pick
   their `handoff(<file>)` commits for lead-owned files; lead restores nothing else by hand.
2. Task 15 lead part: if the builder didn't finish the loader FP8 layout for `fp8_block`, do it (decode fallback per
   unsupported shape, `event="fp8_block_decoded"`). Then the fp8_block proof (new builder).
3. Task 14 proof (new builder, after the FP8 spreads land): tolerances, `lab-bench --model llama-fp8 / llama-fp8-tensor
--golden16 --c1`, vLLM (`scratchpad/fp8/vllm_fp8.sh`), eval, soak (ask coordinator).
4. **Task 18 builder (not started):** gated by `fixtures6.sh` (AWQ spread, GPTQ reference + spread) and the vLLM GPTQ
   session. Brief: `.claude/worktrees/agent-abee0e2542a317f3c/.procoder/handoff/p6a-int4.md`.
5. **Task 20 builder (not started):** gated by `fixq.sh` and `queue3.sh`; must run full GSM8K on MXFP4-A16 8B and
   BF16 8B (decision 4f45232) once `gsm8k-full.jsonl` lands; numerics check before any status change. Brief:
   `.claude/worktrees/agent-a4784842b25c93376/.procoder/handoff/p6a-mxfp4.md` (ec9fce4). Note the misnamed eval file there.
6. Task 24 finish: FP8 KV golden fixtures → tolerance → `lab-bench --model llama-fp8kv / olmoe-fp8kv --golden16`;
   full-GSM8K verdicts (a67eec8abc8f117eb); rows only after.
7. Task 28 per builder outcome.
8. Task 21 two-GPU leg and Task 29's two-GPU tier: **blocked on the PSU**.
9. Task 29 exit; merge into local main; 6b from main with `p6b-groundwork` (YELLOW ladder change, its handoff).

## Open questions

- None open with the coordinator right now. Possible soon: YaRN fold placement (from aa94fe3bfacab12b1); OLMoE FP8 KV
  full-GSM8K result / per-layer scale finding; MXFP4-A16 full-GSM8K result.
- Task 15 test tolerance: the old builder widened `qgemm_fp8_block_matches_cpu` for the dequant path; the new builder
  was told to prefer a BF16-rounded CPU reference with a tight bound — check what it did before merging.

## Rules in force

- PSU: ONE GPU-heavy job at a time (bench.lock, `TURBINE_LAB_ONE_GPU_JOB`); no two-GPU runs; crash → wait bounded, don't debug.
- Disk: cleanup from ~100 GB free (209 GB at 07:56): `git worktree remove` merged clean finished trees →
  `TURBINE_PRUNE_IDLE_HOURS=1 scripts/lab-prune.sh --report` → real run only if it lists finished agents' trees only.
- Rotation: one task per builder; replace at ~300k tokens with a handoff; sonnet for mechanical work.
- Ownership: the lead owns the kernel header, `ffi.rs`, ops, registry, turbine-model config/decoder/loader/weights, core
  support/config, `.procoder/`; builders send `handoff(<file>)` commits. No push; merge into local main at 6a close.
- Design questions go to the coordinator via SendMessage; keep working on anything independent.

## Review file

`.procoder/review-2026-09-29.md` is as the previous lead left it (overnight state, through 2116221). Not yet updated
with: the agent loss and rebuild, the new builders, the Task 24 / 18 / 20 numbers in this file, the GSM8K decisions.
Update it at the next clean point.
