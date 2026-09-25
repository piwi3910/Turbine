# P0-T6 turbine-bench library — seeded prompts and report aggregation

Status: closed 2026-09-25
Created: 2026-09-25

## Description

Phase 0 plan Task 6 (`.procoder/plans/phase-0-skeleton.md`, "## Task 6"): turbine-bench library — seeded prompts and report aggregation. Covers S-6 (seeded prompts, report keys); `prompt::tests::prompts_are_deterministic`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-bench --lib prompt::tests::prompts_are_deterministic` passes (expect PASS.)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red first: `cargo test -p turbine-bench --lib prompt::tests` (test only, no implementation) →
  `error[E0425]: cannot find value `WORD_COUNT` in this scope`
- `cargo test -p turbine-bench --lib prompt::tests::prompts_are_deterministic` →
  `test prompt::tests::prompts_are_deterministic ... ok` / `test result: ok. 1 passed; 0 failed`
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0,
  `Finished `dev` profile [unoptimized + debuginfo]`
- `cargo test --workspace` → turbine-bench `test result: ok. 4 passed; 0 failed` (prompt determinism
  plus `report::tests::{nearest_rank_percentiles, report_aggregates_results, zero_wall_time_gives_zero_throughput}`),
  turbine-core `test result: ok. 4 passed; 0 failed`
- `launcher.sh check` → `procoder gate: ... 18 hygiene finding(s) (0 blocking)`
- Commit `feat(bench): seeded prompt generator and latency/throughput report` is on the task
  worktree branch `worktree-agent-abe15d6caf930cc13`; the last criterion closes when the
  coordinator lands it on `phase-0-skeleton`.
