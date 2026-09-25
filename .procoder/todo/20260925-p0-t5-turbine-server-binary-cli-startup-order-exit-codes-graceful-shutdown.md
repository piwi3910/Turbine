# P0-T5 turbine-server binary — CLI, startup order, exit codes, graceful shutdown

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 5 (`.procoder/plans/phase-0-skeleton.md`, "## Task 5"): turbine-server binary — CLI, startup order, exit codes, graceful shutdown. Covers S-2 (reject before bind, `--set`, `--check-config`, `RUST_LOG`), S-5 (graceful shutdown); `server_cli invalid_config_exits_2_before_bind`, `server_cli sigterm_graceful_shutdown`, `server_cli port_in_use_exits_1`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-server --test server_cli` passes (expect PASS (3 passed).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
