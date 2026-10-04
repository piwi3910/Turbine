# Handoff: p6b-staticresume — the static flood test's schedule equality flaked; pinned the timing-robust invariants instead

Branch `p6b-staticresume` at 9105861 (one test-only commit on `p6b-stack` HEAD 7a79341), gate
`ok crates=all passed=940 failed=0`. Worktree `.claude/worktrees/agent-p6b-staticresume`. No lab
Jobs, servers or locks left; no lab work at all (host-only test, run via `scripts/remote-cargo.sh`).
`target/debug` deleted on novanas after the commit.

## Root cause (the flake, not a static-mode defect)

`static_tiers_flood_then_resume` failed ~30% of runs at `assert_eq!(stat.resumed, local.resumed)`
with local `[32, 32, 32, 48]` vs static `[32, 48, 48, 48]`. Bisect by revision (8 runs each):

- 4465a66 (copykernel merged, **before** ladderperf + greenhead): both modes deterministic
  `[32, 32, 32, 32]`, promoted 2 — passes.
- d04b4bc (ladderperf merged): both modes deterministic `[32, 32, 32, 32]` — passes.
- 7a79341 (greenhead merged, `p6b-stack` HEAD): local bimodal — usually `[32, 48, 48, 48]`
  (promoted 5, retrieve plans 8) but ~30% `[32, 32, 32, 48]` (promoted 3, plans 6); static
  `[32, 48, 48, 48]` in every observed run. Fails when local lands low.
- 7a79341 with the GREEN arm of `within_headroom` removed (temporary edit, restored): both modes
  deterministic `[32, 32, 32, 32]` again — causal confirmation.

Cause: 7d3ae8c (GREEN headroom capped at RED 0.90, a deliberate user decision) queues a burst
admission when live `kv_utilization` (used + reserved + **held** bytes) would pass 0.90. Held bytes
shrink as the reclaimer demotes/frees asynchronously, so a filler's admission races the previous
filler's still-draining demotion queue; which blocks survive in L2 becomes schedule-dependent, and
on the cpu backend the planner's retrieve-vs-recompute choice flips between runs anyway (a copy and
a prefill both take about one engine turn — the suite already prints `plans`, not compares, for
exactly this reason). Those flips now reach the resumed cached-token vector. Each mode's own
realization varies run to run; the mode comparison just sampled the noise. Not promotion_copy
(in the diff at 4465a66, before the other two merges, the test is deterministic) and not the
ladderperf make_room (d04b4bc clean).

## What landed (9105861, test-only, `crates/turbine-server/src/engine/tp.rs`)

- `Flood` gained `answers: Vec<Vec<u32>>` — each resumed session's greedy tokens.
- The resumed cached-token vector is **floored, not compared**: every session in both modes must
  clear the shared system prefix (2 blocks = 32 tokens), and `promoted > 0` in both modes.
- New mode-equivalence invariant: `assert_eq!(stat.answers, local.answers)` — a retrieved and a
  recomputed block hold identical bytes, so the resumed sessions must serve identical outputs.
  Observed identical across 10 runs while the cached vectors differed in 5 of them.
- `later` equality kept (deterministic in ~60 observed runs).
- Doc comment tells the whole story (GREEN headroom, the flip mechanism, why floored).

## Evidence

- HEAD without the fix, 6 + 8 runs: static 6/6 `[32, 48, 48, 48]`; local 5/14 runs `[32, 32, 32,
48]` (the failures), rest `[32, 48, 48, 48]`.
- Fixed test, 10 runs: 10/10 pass; `later` `[48, 64, 48, 64, …]` and answers
  `[[104]×8, [105]×8, [106]×8, [107]×8]` identical in both modes every run; the `resumed` vectors
  differed between modes in 5 of the 10 runs and the test correctly passed.
- Mutation check (temporary engine mutation, reverted): every static worker demotion acks as failed
  → static `resumed` `[0, 32, 32, 32]` / `promoted 0.0` → test fails on "static resumes with the
  shared prefix" (2/2 runs). The 1 MiB-L2 mutation attempt is refused at L2 open, not useful.
- Suites: `engine::tp` (12), `engine::tp_tiers` (5), `engine::r#loop` (22) green; gate
  `ok crates=all passed=940 failed=0`.

## Not changed

- `promotion_copy` kernel default, ladderperf make_room, the GREEN headroom arm — none is the
  cause; the GREEN arm behaves as designed (the test's burst schedule is what changed meaning).
- If the exact vector equality is wanted back, the schedule must be pinned in the engine (e.g. a
  deterministic planner for the test backend) — a design question, not a test tweak; I did not
  attempt it.
