# P1-T17 turbine-server model startup, single generation slot and Phase 1 metrics

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 17 (`.procoder/plans/phase-1-single-request.md`, "## Task 17"): turbine-server model startup, single generation slot and Phase 1 metrics. Covers S-10 AC `tiny_server completions_stream_and_non_stream`, `tiny_server single_slot_and_cancel`, `tiny_server request_validation`, `tiny_server startup_failures_exit_1`; S-14 AC `tiny_server phase1_metrics`; S-5 (budget before weights). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-server` passes (expect PASS (Phase 0 `server_cli` tests unchanged))
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- `CARGO_INCREMENTAL=0 cargo test -p turbine-server` (on top of T14 `f21fe79`):
  - `Running unittests src/main.rs` → `test result: ok. 4 passed` (`generation::tests::three_consecutive_failures_stop_the_engine`, `model::tests::{served_name_from_hf_snapshot_or_last_component, max_seq_len_defaults_and_bounds}`, `metrics::tests::renders_phase1_names_and_labels`)
  - `Running tests/server_cli.rs` → `test result: ok. 3 passed` (`invalid_config_exits_2_before_bind`, `port_in_use_exits_1`, `sigterm_graceful_shutdown`)
  - `Running tests/tiny_server.rs` → `test result: ok. 5 passed` (`completions_stream_and_non_stream`, `single_slot_and_cancel`, `request_validation`, `startup_failures_exit_1`, `phase1_metrics`); run 3× in a row, all green
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `clippy exit 0`
- `CARGO_INCREMENTAL=0 cargo test --workspace` → `test exit 0`, 145 passed, 0 failed, 2 ignored (GPU/lab)
- `launcher.sh check` → `procoder gate: 3 clean, 0 unformatted, 0 unchecked, 0 out of scope, 15 hygiene finding(s) (0 blocking)`
- Mutation check: making the generation thread ignore a failed `blocking_send` (client gone) fails `single_slot_and_cancel` (no `outcome="cancelled"`); the first real-T14 run failed `completions_stream_and_non_stream` with 429 on a back-to-back request, which led to releasing the slot before the final event (now covered by 20 back-to-back requests in `single_slot_and_cancel`).
- Deviation: `tests/server_cli.rs` `port_in_use_exits_1` and `sigterm_graceful_shutdown` now serve the tiny checkpoint on the cpu backend (`served_name: m`): the Phase 1 startup order loads the provider and model before binding, so `model.path: /m` can no longer reach the bind step, and the in-flight request now completes 200 instead of 503 `model_not_loaded`. `invalid_config_exits_2_before_bind` is unchanged.
