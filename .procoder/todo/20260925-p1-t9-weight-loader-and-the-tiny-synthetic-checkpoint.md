# P1-T9 Weight loader and the tiny synthetic checkpoint

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 9 (`.procoder/plans/phase-1-single-request.md`, "## Task 9"): Weight loader and the tiny synthetic checkpoint. Covers S-2 AC `loader::tests::tensor_mapping`, `loader::tests::never_opens_pickle`; S-12 (tiny synthetic checkpoint). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model loader::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
