# P0-T7 turbine-bench binary — streaming client, fixed concurrency, exit codes

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 7 (`.procoder/plans/phase-0-skeleton.md`, "## Task 7"): turbine-bench binary — streaming client, fixed concurrency, exit codes. Covers S-6; `bench mock_endpoint_measurements`, `bench failures_counted`, manual run against dgx-spark vLLM.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-bench` passes (expect PASS (unit + 2 integration tests).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
