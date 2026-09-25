# P1-T11 Tokenizer and incremental detokenization

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 11 (`.procoder/plans/phase-1-single-request.md`, "## Task 11"): Tokenizer and incremental detokenization. Covers S-4 AC `tokenizer::tests::incremental_detokenize_utf8`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model tokenizer::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
