# P2-T14 OpenAI Phase 2 fields, tool calls and new errors in turbine-api

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 14 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 14"): OpenAI Phase 2 fields, tool calls and new errors in turbine-api. Covers S-10, S-13 (readiness reason), S-17/S-18 (API surface); end-to-end assertions in Tasks 15–17 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-api` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-api): phase 2 OpenAI fields, tool calls and timeout errors`)

## Evidence

