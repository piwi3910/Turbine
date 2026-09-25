# P1-T8 Safetensors index with header validation

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 8 (`.procoder/plans/phase-1-single-request.md`, "## Task 8"): Safetensors index with header validation. Covers S-2 AC `safetensors::tests::rejects_malformed_headers`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model safetensors::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
