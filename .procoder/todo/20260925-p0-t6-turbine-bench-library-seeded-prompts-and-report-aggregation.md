# P0-T6 turbine-bench library — seeded prompts and report aggregation

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 6 (`.procoder/plans/phase-0-skeleton.md`, "## Task 6"): turbine-bench library — seeded prompts and report aggregation. Covers S-6 (seeded prompts, report keys); `prompt::tests::prompts_are_deterministic`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-bench --lib prompt::tests::prompts_are_deterministic` passes (expect PASS.)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
