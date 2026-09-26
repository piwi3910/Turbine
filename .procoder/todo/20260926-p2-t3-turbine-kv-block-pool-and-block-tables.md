# P2-T3 turbine-kv block pool and block tables

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 3 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 3"): turbine-kv block pool and block tables. Covers S-5 AC `pool::tests::blocks_conserved`; S-1 (crate boundary) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kv` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-kv): preallocated block pool and block tables`)

## Evidence

