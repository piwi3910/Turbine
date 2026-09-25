# P1-T17 turbine-server model startup, single generation slot and Phase 1 metrics

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 17 (`.procoder/plans/phase-1-single-request.md`, "## Task 17"): turbine-server model startup, single generation slot and Phase 1 metrics. Covers S-10 AC `tiny_server completions_stream_and_non_stream`, `tiny_server single_slot_and_cancel`, `tiny_server request_validation`, `tiny_server startup_failures_exit_1`; S-14 AC `tiny_server phase1_metrics`; S-5 (budget before weights). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-server` passes (expect PASS (Phase 0 `server_cli` tests unchanged))
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
