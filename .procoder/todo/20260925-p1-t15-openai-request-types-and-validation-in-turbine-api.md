# P1-T15 OpenAI request types and validation in turbine-api

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 15 (`.procoder/plans/phase-1-single-request.md`, "## Task 15"): OpenAI request types and validation in turbine-api. Covers S-10 (supported request fields, unsupported → 400); the end-to-end assertions are `tiny_server request_validation` in Task 17. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-api` passes (expect PASS (Phase 0 `api` tests unchanged))
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed with the plan's commit message on worktree branch `worktree-agent-aac519a1f1e3cd786` (branched from phase-1-single-request; coordinator merges it)

## Evidence

- Red first: `cargo test -p turbine-api --test openai request_validation_rules` -> `error[E0433]: cannot find openai in turbine_api`, `error[E0407]: method submit is not a member of trait InferenceBackend`, `error[E0599]: no associated function ... named unsupported_parameter` (quotes stripped)
- `cargo test -p turbine-api` -> `tests/api.rs: test result: ok. 5 passed; 0 failed` (Phase 0, unchanged) and `tests/openai.rs: test result: ok. 4 passed; 0 failed` (request_validation_rules, request_helpers, error_constructors_follow_the_code_table, backend_submit_seam)
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` -> exit 0 (`Finished \`dev\` profile`)
- `cargo test --workspace` -> every `test result: ok`, 32 passed, 0 failed
- `launcher.sh check` -> `procoder gate: 6 clean, 0 unformatted, 0 unchecked, 3 out of scope, 46 hygiene finding(s) (0 blocking)`
