# P1-T18 turbine-golden compare and capture

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 18 (`.procoder/plans/phase-1-single-request.md`, "## Task 18"): turbine-golden compare and capture. Covers S-11 AC `golden capture_and_compare_roundtrip`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-bench` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
