# P0-T5 turbine-server binary — CLI, startup order, exit codes, graceful shutdown

Status: done
Created: 2026-09-25

## Description

Phase 0 plan Task 5 (`.procoder/plans/phase-0-skeleton.md`, "## Task 5"): turbine-server binary — CLI, startup order, exit codes, graceful shutdown. Covers S-2 (reject before bind, `--set`, `--check-config`, `RUST_LOG`), S-5 (graceful shutdown); `server_cli invalid_config_exits_2_before_bind`, `server_cli sigterm_graceful_shutdown`, `server_cli port_in_use_exits_1`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-server --test server_cli` passes (expect PASS (3 passed).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-server --test server_cli` with an empty `main` → `left: Some(0)` / `right: Some(2)` (also `right: Some(1)` and `exited early (exit status: 0)`); `test result: FAILED. 0 passed; 3 failed`.
- Green: `cargo test -p turbine-server --test server_cli` → `test result: ok. 3 passed; 0 failed` (15 consecutive runs, all ok, ~0.85 s each).
- Discovery failure: `cargo run -q -p turbine-server -- --config turbine-p0.yaml --set devices.amd_smi_library=/nonexistent/libamd_smi.so` → `turbine-server: device discovery failed: cannot load /nonexistent/libamd_smi.so: dlopen failed: …` and `exit=1`.
- CLI usage error: `turbine-server --bogus` → `error: unexpected argument '--bogus' found`, `exit=2`.
- Gate: `cargo fmt --all --check` exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished`; `cargo test --workspace` → every `test result: ok` (26 tests); `launcher.sh check` → `(0 blocking)`.
- Committed on the Task 5 worktree branch (from `phase-0-skeleton`), to be merged into `phase-0-skeleton`.
