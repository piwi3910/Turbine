# P2-T15 Engine thread with continuous batching, bounded channels and diagnostics

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 15 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 15"): Engine thread with continuous batching, bounded channels and diagnostics. Covers S-1 (engine thread in `turbine-server`), S-7 AC `tiny_server queue_full_429`; S-8 AC `tiny_server disconnect_releases_kv`; S-11 AC `tiny_server diagnostics_shapes`; S-3 (continuous batching end to end) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-server` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): engine thread with continuous batching and paged KV`)

## Evidence

- Red first: `CARGO_INCREMENTAL=0 cargo test -p turbine-server --test tiny_server` on the Phase 1 engine → `single_slot_and_cancel` and `queue_full_429` fail with `{"error":{..."code":"not_implemented"}}` (`/turbine/v1/scheduler` 501), `diagnostics_shapes` and `disconnect_releases_kv` with `http/1.1 429 too many requests` (`engine_busy` on the second stream), `startup_failures_exit_1` with `turbine-server did not exit within 60s` (no `kv.gpu.enabled` / `kv.gpu.max_bytes` checks) → `test result: FAILED. 3 passed; 5 failed`.
- `CARGO_INCREMENTAL=0 cargo test --workspace` → exit 0, 37 `test result: ok` lines; `tests/tiny_server.rs` → `single_slot_and_cancel ... ok`, `queue_full_429 ... ok`, `disconnect_releases_kv ... ok`, `diagnostics_shapes ... ok`, `startup_failures_exit_1 ... ok`, `test result: ok. 8 passed; 0 failed`; engine unit tests `engine::r#loop::tests::{batched_greedy_matches_single_request_generate, three_consecutive_failed_iterations_stop_the_engine, engine_panic_fails_every_request, full_channel_pauses_and_closed_channel_cancels, refused_submissions_answer_before_any_event} ... ok`, `engine::requests::tests::{history_stop_strings_and_held_events, length_at_max_tokens_or_context} ... ok`.
- `cargo fmt --all --check && CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets -- -D warnings` → exit 0 (`Finished dev profile`).
- `launcher.sh check` → `procoder gate: 9 clean, 0 unformatted, 0 unchecked, 2 out of scope, 43 hygiene finding(s) (0 blocking)`.
- Deviations: `EngineCommand::Submit` carries a third field, a `oneshot` admission answer (`Result<(), SubmitError>`), so a scheduler refusal (`queue_full` and the other submission checks, counted in `turbine_admission_total`) is a plain HTTP error before any event, also for streams; `EngineCommand::Cancel` is not added (no sender in this task: client disconnects are detected from the closed output channel every turn). `ModelBackend` moved to `src/backend.rs` (the plan removes `generation.rs` without naming its new home); `src/main.rs` declares the modules; `turbine_stream_paused_total` is registered in `src/metrics.rs`. The executor is built through `model::build_executor` (delegating to Task 11's `turbine_model::executor::build_executor`; registry and budget use `executor::requirements` / `executor::workspace_bytes`), so OLMoE checkpoints serve too.

