# P1-T10 Startup memory budget

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 10 (`.procoder/plans/phase-1-single-request.md`, "## Task 10"): Startup memory budget. Covers S-5 AC `budget::tests::refuses_before_loading`, `budget::tests::available_memory_by_kind`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model budget::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
