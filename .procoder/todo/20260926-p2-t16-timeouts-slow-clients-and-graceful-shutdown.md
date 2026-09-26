# P2-T16 Timeouts, slow clients and graceful shutdown

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 16 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 16"): Timeouts, slow clients and graceful shutdown. Covers S-7 AC `tiny_server slow_client_paused_then_cancelled`, `tiny_server request_and_queue_timeouts`; S-13 AC `server_cli sigterm_drains_then_cancels`; S-8 (timeout and shutdown cancellation) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-server` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): request/queue timeouts, slow clients and graceful shutdown`)

## Evidence

