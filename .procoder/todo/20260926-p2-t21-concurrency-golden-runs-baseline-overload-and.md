# P2-T21 Concurrency golden runs, baseline, overload and acceptance on macOS

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 21 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 21"): Concurrency golden runs, baseline, overload and acceptance on macOS. Covers S-1 AC (macOS build/test/clippy/fmt and `cargo tree -p turbine-scheduler`); S-15/S-16 AC manual golden runs with `--concurrency 16`; S-15 AC manual baseline run; S-15 AC manual overload run Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message

## Evidence

