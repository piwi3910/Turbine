# P1-T2 turbine-tensor — device memory handles, host backend, tensors

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 2 (`.procoder/plans/phase-1-single-request.md`, "## Task 2"): turbine-tensor — device memory handles, host backend, tensors. Covers S-5 (`Tensor` per TS §6, `DeviceBuffer` owning one allocation freed on `Drop`). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-tensor` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
