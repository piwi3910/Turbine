# Handoff: Phase 6b lead

Started 2026-09-30 ~21:00 +04 after the user said "ok start" (6a closed, local `main` f8599f4 + later docs commits,
not pushed). Integration branch `p6b-stack` (worktree `.claude/worktrees/agent-p6b-t2`); every builder branches from
it and lands by merge. Decisions: `.procoder/ask/decisions.md` (all 6b entries 2026-09-30 / 10-01 are answered).

## State (2026-10-01 ~03:00)

- Done on `p6b-stack`: plan Tasks 1–8, 10, 11, 14, 15 (Task 5 server side included). Also: ABI group renumbered v2.11
  (6a took v2.10); encoded-size eviction scores + `make_room` in-flight fix; sim and backend copy timing (own copies,
  poll-bounded stream copies); prefetch-on-idle-engine hang fix; staging in the memory budget; HIP graph-capture
  SIGSEGV fixed (shim lock; upstream drafts in `.procoder/handoff/upstream/`, user reviewing, nothing filed);
  shared-prefix GSM8K variant `tests/eval/gsm8k-200-shared-prefix.jsonl` with lossy-reuse guards.
- Running (each with its own worktree and handoff file):
  - `p6b-planner2`: shared-wait once per plan + promotions first, estimate decay, `tp_tiers` timing, A/B below
    SURVIVAL (medians of 3), FP8-tier eval on the shared-prefix variant
  - `p6b-d2h`: why device→host copies calibrate at 1.6 GB/s vs 10.45 GB/s host→device
  - `p6b-t12`: Task 12 mixed-format paged attention (own decode `turbine_hip_mixed`, staged CK prefill, S-5 amended)
- Next: server-side TurboQuant table upload (~15–17 MiB/rank; needed by Tasks 9 and 13) → Task 9 TurboQuant tier
  proof → Task 13 TurboQuant in L0 proof → Task 16 ladder proof + soak → Task 17 L0 ladder step → 18 → 19 exit.
- Known items: rung reading can jump two steps in one tick (blocks move one rung); tq4 encode 0.64 ms/block (~4× the
  host link) may cap demotion rate; `server_cli::sigterm_drains_then_cancels` flaked once under load; one-off SIGSEGV
  at exit of `turbine-device --test lab` (6a, not investigated).

## Rules in force

Builders rotate near ~300k context (handoff, fresh builder). No Mac builds. Lab on novanas only; stop only own serve
Jobs by run id. Ask the user (structured question, recorded in decisions.md first) for every design decision.
