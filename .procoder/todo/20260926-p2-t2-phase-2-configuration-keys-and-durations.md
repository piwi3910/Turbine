# P2-T2 Phase 2 configuration keys and durations

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 2 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 2"): Phase 2 configuration keys and durations. Covers S-7 (bounds are configurable), §Configuration additions and changes Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-core config::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-core): phase 2 scheduler, kv, timeout and structured-output keys`)

## Evidence

