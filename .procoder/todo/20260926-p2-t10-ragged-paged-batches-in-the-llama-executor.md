# P2-T10 Ragged paged batches in the Llama executor

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 10 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 10"): Ragged paged batches in the Llama executor. Covers S-9 (batched execution, one D2H copy), S-5 (executor on the paged pool); the named paged/chunked ACs are delivered in Task 11 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model --test tiny_model` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): ragged paged batches for the Llama executor`)

## Evidence

