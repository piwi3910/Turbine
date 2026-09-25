# P0-T3 turbine-device — NVML and amd-smi discovery with deadlines

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 3 (`.procoder/plans/phase-0-skeleton.md`, "## Task 3"): turbine-device — NVML and amd-smi discovery with deadlines. Covers S-4; `discovery::tests::no_libraries_means_empty_inventory`, `discovery::tests::explicit_missing_library_is_fatal`, `discovery::tests::backend_timeout`, `discovery::tests::unified_memory_uses_host_total`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-device --lib` passes (expect PASS (6 passed; `backend_timeout` takes ~10 s).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
