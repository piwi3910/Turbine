# P0-T1 Workspace root and turbine-core configuration model

Status: done
Created: 2026-09-25

## Description

Phase 0 plan Task 1 (`.procoder/plans/phase-0-skeleton.md`, "## Task 1"): Workspace root and turbine-core configuration model. Covers S-1 (workspace, edition, rust-version, license, `LICENSE`), S-2; `config::tests::example_config_loads`, `config::tests::byte_size_parsing`, `config::tests::impossible_configs_rejected`, `config::tests::set_overrides_apply`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-core config::tests` passes (expect PASS (4 passed).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red first: `cargo test -p turbine-core config::tests` with an empty `config/mod.rs` failed to compile with `error[E0425]: cannot find type `Config` in this scope` (plus the other missing `ByteSize`/`Override`/`load_from_str` items).
- `cargo test -p turbine-core config::tests` → `test result: ok. 4 passed; 0 failed` (example_config_loads, byte_size_parsing, impossible_configs_rejected, set_overrides_apply).
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished `dev` profile`, exit 0; `cargo test --workspace` → 4 passed, 0 failed.
- `launcher.sh check` → `procoder gate: 7 clean, 0 unformatted, 0 unchecked, 4 out of scope, 57 hygiene finding(s) (0 blocking)`.
- `shasum -a 256 LICENSE` → `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` (downloaded from apache.org).
- Commit `feat(core): workspace root and validated configuration model` on `phase-0-skeleton` (`git log -1 --oneline`).
