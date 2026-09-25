# P1-T12 Chat template rendering (minijinja + pycompat)

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 12 (`.procoder/plans/phase-1-single-request.md`, "## Task 12"): Chat template rendering (minijinja + pycompat). Covers S-4 AC `chat_template::tests::renders_target_template`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model chat_template::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
