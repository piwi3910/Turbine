# Handoff: Phase 6b lead

## PAUSED 2026-10-02 (user: "when possible lets pause our work") — START HERE

Integration branch `p6b-stack` (worktree `agent-p6b-t2`), last gate ok 935 passed. Nothing of ours runs on novanas.
Every decision so far is answered and recorded in `.procoder/ask/decisions.md`.

**Supported now (lower tier):** `fp8_e4m3`, `tq4`. **Experimental:** lower-tier `tq2`; L0 `tq4`. **Refused:** L0 `tq2`.
**Done:** plan Tasks 1–16 except the open fixes below. **Left:** Task 17 (L0 ladder step + BF16 recent window, user
decision 2026-10-02), Task 18 (L0 ladder proof + soak), Task 19 (phase exit).

Three unmerged branches, each with its own handoff and exact next steps:

1. `p6b-ladderperf` — ladder-on regression root-caused (new-rung demotions fail `Full`, L0 can't drain); fix = fallback to
   the tier's own format (user decision A), drafted in `.procoder/handoff/p6b-ladderperf.patch` (also a stash
   2c01160d…, the patch is the canonical copy); finish, test, gate, rerun the ladder A/B.
2. `p6b-copykernel` — promotion copy kernel validated on OLMoE + Llama; the default flip to `kernel` (4ba941c) was
   committed BEFORE `lab-bench --golden16` (both models) and the demotion-stall check ran: run both, revert the flip if
   either fails, then merge.
3. `p6b-greenhead` — GREEN admission headroom capped at RED 0.90 (user decision A) committed but not gate-clean:
   `overload_sim survival_liveness_seed_6` no longer reaches SURVIVAL (peaks at RED); the regression needs a scenario
   that still exercises SURVIVAL (don't weaken it); then mutation check, lab burst check, soak, quick bench.

Upstream: rocm-systems#12677 (HIP, filed), rocm-libraries#12895 (hipBLASLt, filed) + fix PR rocm-libraries#12900.

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
