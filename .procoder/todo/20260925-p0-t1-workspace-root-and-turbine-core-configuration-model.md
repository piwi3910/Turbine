# P0-T1 Workspace root and turbine-core configuration model

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 1 (`.procoder/plans/phase-0-skeleton.md`, "## Task 1"): Workspace root and turbine-core configuration model. Covers S-1 (workspace, edition, rust-version, license, `LICENSE`), S-2; `config::tests::example_config_loads`, `config::tests::byte_size_parsing`, `config::tests::impossible_configs_rejected`, `config::tests::set_overrides_apply`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-core config::tests` passes (expect PASS (4 passed).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
