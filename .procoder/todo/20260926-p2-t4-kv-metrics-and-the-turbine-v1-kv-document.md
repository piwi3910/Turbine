# P2-T4 KV metrics and the /turbine/v1/kv document

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 4 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 4"): KV metrics and the /turbine/v1/kv document. Covers S-5 (KV accounting), S-11 (KV document shape), S-14 (`turbine_kv_blocks`) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kv` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-kv): kv metrics and diagnostics document`)

## Evidence

