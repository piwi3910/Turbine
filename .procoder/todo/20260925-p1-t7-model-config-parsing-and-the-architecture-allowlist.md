# P1-T7 Model config parsing and the architecture allowlist

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 7 (`.procoder/plans/phase-1-single-request.md`, "## Task 7"): Model config parsing and the architecture allowlist. Covers S-3 AC `config::tests::parses_target_config`, `config::tests::rejects_unsupported`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model config::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
