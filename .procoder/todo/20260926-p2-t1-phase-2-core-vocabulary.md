# P2-T1 Phase 2 core vocabulary

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 1 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 1"): Phase 2 core vocabulary. Covers S-2 (request vocabulary), S-10/S-17/S-18 (field types) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-core` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-core): phase 2 vocabulary, clock and resource estimate`)

## Evidence

