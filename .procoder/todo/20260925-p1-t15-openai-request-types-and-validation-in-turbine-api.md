# P1-T15 OpenAI request types and validation in turbine-api

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 15 (`.procoder/plans/phase-1-single-request.md`, "## Task 15"): OpenAI request types and validation in turbine-api. Covers S-10 (supported request fields, unsupported → 400); the end-to-end assertions are `tiny_server request_validation` in Task 17. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-api` passes (expect PASS (Phase 0 `api` tests unchanged))
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
