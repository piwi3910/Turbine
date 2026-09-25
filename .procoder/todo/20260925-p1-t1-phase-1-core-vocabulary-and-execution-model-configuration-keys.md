# P1-T1 Phase 1 core vocabulary and execution / model configuration keys

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 1 (`.procoder/plans/phase-1-single-request.md`, "## Task 1"): Phase 1 core vocabulary and execution / model configuration keys. Covers S-1 (shared vocabulary), config keys of §Configuration additions; the exit-2 half of `tiny_server startup_failures_exit_1` is asserted end to end in Task 17. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-core` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
