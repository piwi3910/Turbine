# P1-T1 Phase 1 core vocabulary and execution / model configuration keys

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 1 (`.procoder/plans/phase-1-single-request.md`, "## Task 1"): Phase 1 core vocabulary and execution / model configuration keys. Covers S-1 (shared vocabulary), config keys of §Configuration additions; the exit-2 half of `tiny_server startup_failures_exit_1` is asserted end to end in Task 17. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-core` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-core config::tests::execution_and_model_keys` before the implementation → `error[E0432]: unresolved import crate::types::ExecutionBackend` and `error[E0609]: no field execution on type config::Config` (compile failure: keys and types absent).
- Green: `cargo test -p turbine-core` → `test result: ok. 6 passed; 0 failed` (includes `config::tests::execution_and_model_keys` and `types::tests::dtype_codes_and_kv_layout_sizes`: 114_688 / 1_835_008 bytes for Llama-3.2-3B BF16).
- Workspace: `cargo test --workspace` → 28 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → `0 blocking`.
- Commit: `feat(turbine-core): phase 1 vocabulary and execution configuration` on the Task 1 worktree branch (branched from phase-1-single-request; lands there on merge).
