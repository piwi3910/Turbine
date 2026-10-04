# Handoff: Phase 5p start (2026-10-04)

## What this worktree (`agent-p6b-t2`, branch `p6b-stack`) carries

The 5p start work, per decision "Phase 5p start: amendment scope and the track gate" (2026-10-04:
1 A, 2 A, 3 A):

1. **Track gate accepts the interlude** (`feat(scripts)` commit): `scripts/track-gate.sh
   phase-5p-serving-efficiency` gates the start on the 5p spec AND plan being COMPLETE — the
   "previous track closed" rule does not apply to the interlude. Usage string, tracks regex, the
   order case (empty for 5p) and the plan check (`procoder plan check` COMPLETE + the plan's own
   `Status: complete`; the plan is read from the sibling `plans` directory) are in the script;
   `lab_scripts track_gate` feeds it the 5p spec+plan (passes on a BF16-only matrix — the
   order rule truly absent), a draft plan (fails `plan check is not COMPLETE`), a missing plan
   file (fails the same way) and a draft spec (fails as every track). Mutation-checked (5p
   dropped from the tracks regex → test red). AGENTS.md's track-start line and the umbrella
   plan's state paragraph record the amendment.
2. **The 5p spec amended for the 6b reality**: the Source paragraph now records the start on the
   6b closing state (the old "runs before phase-6-quantization" is gone); a 6b amendment bullet
   covers token-granular reuse against classed pages (exact classes only: `bf16`/`fp8_e4m3`
   bases, the BF16 recent-window class on TurboQuant bases; a ladder-rewritten child is outcome
   `lossy_source`; the private tail block follows its source's page class; the ladder never
   rewrites a referenced block; partial entries are untouched by tail-tag expiry); S-7 notes the
   pressure gate (6b's GREEN headroom cap at RED 0.90) runs before the policy; the two stale
   citations fixed (`modules.scheduling_policy` → the existing `turbine_server::modules::ModuleChoices`
   entry; `scheduler.policy` → the `policy` field of `SchedulerConfig`, Phase 2m, registering
   `cache_aware`); the partial-reuse metric gains `lossy_source` (six outcomes). New scope items
   S-14–S-18 with targets and acceptance criteria carry the folded follow-ups.
3. **The plan amended and complete** (`Status: complete`): 6b starting-point bullet (ABI v2.11,
   the ladder opt-in, the recent window, per-class pages, lower-tier formats, tail-tag expiry,
   128 MiB slabs), Task 3 baselines on the 6b tree with the ladder off, the S-4/S-5/S-6/S-7 task
   texts carrying the interactions, and five new tasks (each with files, steps, criteria):
   Task 13 ladder serving cost (instrument → the measured fix: rewrite budget / ORANGE
   hysteresis / copy-kernel path), Task 14 FP8 c1 ITL (profile → fix; the 6a 0.846×/0.855×
   record against ≤ 0.75×), Task 15 MXFP4 prefill (reuse-first provider evaluation → ≤ 1.10×
   hipBLASLt BF16, TTFT ≤ 1.05× BF16), Task 16 OLMoE BF16 margin (0.982× → ≥ 608 tok/s median or
   a recorded re-bound), Task 17 slab mixing (`kv.cpu.slab_drains_per_tick`, drain-and-reformat,
   rung-format copies actually store). Exit task renumbered 13 → 18. Decision-bounded notes:
   Task 13 is bounded by 2026-10-04 A (a follow-up, not a blocker), Task 17 by 2026-10-03 C
   (slabs landed; B held "only if compression must land sooner" — 2 A is that go-ahead) — both
   noted in the spec's S-14/S-18 bullets.

`procoder spec check phase-5p-serving-efficiency` → COMPLETE; `procoder plan check` → COMPLETE;
`scripts/track-gate.sh phase-5p-serving-efficiency` → GATE PASS (evidence in the commits).
`scripts/gate.sh` → `gate: ok` per commit. `scripts/gate.sh phase-7-model-families` still passes.

## Next (fresh builder, branch from this tip)

- Task 1 (`turbine-bench stage-share`) starts the phase; Tasks 1–2 are host-only, Task 3 is the
  first lab run (standing novanas approval, GPU 0, ladder-off baselines).
- The 6b follow-ups' remaining recorded items that did NOT fold into 5p: the upstream ROCm items
  (rocm-systems#12677, rocm-libraries#12895 + PR #12900) stay with the lead.
- Branch name for the work: `phase-5p` (branched from this tip), per plan Task 3's step.
