# P2-T19 turbine-golden compare --concurrency

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 19 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 19"): turbine-golden compare --concurrency. Covers §Interfaces `turbine-golden compare (change)` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-bench` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-bench): concurrent golden comparison`)

## Evidence

