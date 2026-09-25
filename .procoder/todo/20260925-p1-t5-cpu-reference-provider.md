# P1-T5 cpu-reference provider

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 5 (`.procoder/plans/phase-1-single-request.md`, "## Task 5"): cpu-reference provider. Covers S-6 (`cpu-reference`, pure Rust, f32 accumulation, always available). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kernels cpu::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
