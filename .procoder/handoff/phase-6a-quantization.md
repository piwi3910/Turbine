# Handoff: phase-6a-quantization (integration branch)

Written 2026-09-29 ~08:00 by the new 6a lead (previous lead killed at ~07:40 without a handoff).
Worktree `.claude/worktrees/agent-a4b513efedb95892f`, tip 62ab8b7, clean. Lab runs go from the clean detached
worktree `agent-a4b513efedb95892f-lab` (lab scripts rsync uncommitted files; never lab from a dirty tree).
Status table, perf table and incidents: `.procoder/review-2026-09-29.md`.

## Landed (90 commits over main 38703b9)

Tasks 1–13, 21 (host), 22, 23, 25, 26, 27 done; 16, 17, 19 (INT4 / MXFP4 kernels) done; 1c85bfd `fp8_block`
BF16 decode (now fallback only); 2116221 Quark `kv_cache_quant_ignored` + one-GPU-job rule; 2a8d6ec the user's
answers (spec/plan amended; `procoder spec check` / `plan check` COMPLETE for 6a and 6b on 2026-09-29 08:00).

## Remaining per task (branch handoffs in each worktree's `.procoder/handoff/<branch>.md`)

| Task | State | Where |
|---|---|---|
| 14 fp8 proof | fixtures + spread running; tolerance calibration, full benches, vLLM, eval, soak | p6a-fp8 |
| 15 fp8_block own kernel | kernel written (fused ≤ 64 rows, dequant+BF16 GEMM above), lab matches_cpu pending | p6a-fp8 |
| 15 loader | lead: restore the FP8 layout for `fp8_block`, decode fallback per unsupported shape | lead |
| 18 awq/gptq proof | AWQ measured (1.47× tok/s, 0.48× c1 ITL, golden c1 PASS, GSM8K 0.775 vs vLLM 0.755); GPTQ fixtures + vLLM queued | p6a-int4 |
| 20 mxfp4 proofs | 8B BF16 baseline done; MXFP4-A16 eval drop 0.055 > 0.04 (open); W4A4 8B eval queued; fixtures running | p6a-mxfp4 |
| 21 TP lab leg | **blocked: PSU** (no two-GPU runs until the user says the PSU is in) | — |
| 24 FP8 KV proof | kv_gpu PASS, perf ✓; Llama GSM8K drop 0.015 > 0.01 (open); FP8 KV golden fixtures lost in a crash, re-queue | p6a-kv |
| 28 YaRN proof | golden tail fails on p16 / p17-long (1.38 / 1.42); fold hypothesis spread running | p6a-yarn |
| 29 exit | after all above; the two-GPU tier is **blocked: PSU** | lead |

## Lead-owned shared files

Kernel header, `ffi.rs`, ops, registry, turbine-model config/decoder/loader/weights, core support/config,
`.procoder/`. Builders send changes to them as `handoff(<file>)` commits; the lead cherry-picks.

## Rules in force (coordinator brief 2026-09-29)

ONE GPU-heavy job at a time (bench.lock, `TURBINE_LAB_ONE_GPU_JOB`); disk cleanup from ~100 GB free
(worktree remove of merged clean trees → `TURBINE_PRUNE_IDLE_HOURS=1 scripts/lab-prune.sh --report` → real run if
only finished agents' trees); no push; merge into local main at 6a close; builders rotate after one task.
