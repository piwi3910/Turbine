# P2-T16 Timeouts, slow clients and graceful shutdown

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 16 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 16"): Timeouts, slow clients and graceful shutdown. Covers S-7 AC `tiny_server slow_client_paused_then_cancelled`, `tiny_server request_and_queue_timeouts`; S-13 AC `server_cli sigterm_drains_then_cancels`; S-8 (timeout and shutdown cancellation) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-server` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): request/queue timeouts, slow clients and graceful shutdown`)

## Evidence

- Red first (engine deadline check disabled): `cargo test -p turbine-server --test tiny_server -- slow_client_paused_then_cancelled request_and_queue_timeouts` → `test result: FAILED. 0 passed; 2 failed` ("the slow client is cancelled: not reached within 5s", "the timed-out stream is dropped: not reached within 5s"); `cargo test -p turbine-server --test server_cli sigterm` before the shutdown sequence → `test result: FAILED. 0 passed; 2 failed`.
- `cargo test -p turbine-server --test tiny_server -- slow_client_paused_then_cancelled request_and_queue_timeouts` → `test result: ok. 2 passed; 0 failed`
- `cargo test -p turbine-server --test server_cli sigterm` → `test sigterm_drains_then_cancels ... ok`, `test sigterm_graceful_shutdown ... ok`
- `cargo test -p turbine-server` → unit `15 passed`, lab_openai `3 passed; 1 ignored`, server_cli `4 passed`, tiny_server `10 passed`; 0 failed (deadlines unit tests on `FakeClock`: `engine::deadlines::tests::*` 5 passed)
- `cargo test --workspace` → every `test result: ok`, 0 failed
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished`, exit 0
- `launcher.sh check` → `procoder gate: 6 clean, 0 unformatted, ... (0 blocking)`
