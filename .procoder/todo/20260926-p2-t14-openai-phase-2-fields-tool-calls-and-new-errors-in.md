# P2-T14 OpenAI Phase 2 fields, tool calls and new errors in turbine-api

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 14 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 14"): OpenAI Phase 2 fields, tool calls and new errors in turbine-api. Covers S-10, S-13 (readiness reason), S-17/S-18 (API surface); end-to-end assertions in Tasks 15–17 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-api` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-api): phase 2 OpenAI fields, tool calls and timeout errors`)

## Evidence

- Red first: `cargo test -p turbine-api --test openai phase2_shapes` failed to compile before the implementation — `error[E0599]: no associated function or constant named queue_full found for struct ApiError`, `unresolved imports turbine_api::openai::request::ResponseFormat, ToolChoiceMode`, `no variant ... named ShuttingDown found for enum NotReadyReason`.
- `cargo test -p turbine-api` → `test phase2_shapes ... ok`, `test phase2_request_fields ... ok`, `test phase2_error_constructors ... ok`; `test result: ok. 9 passed; 0 failed` (tests/openai.rs) and `test result: ok. 5 passed; 0 failed` (tests/api.rs).
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0 (`Finished dev profile`).
- `cargo test --workspace` → exit 0, 200 passed, 0 failed, 11 ignored (after rebasing onto 3648a43); includes `tiny_server request_validation ... ok` (the Phase 1 engine still refuses the Phase 2 fields it does not serve yet with 400 `unsupported_parameter`).
- `launcher.sh check` → `procoder gate: 8 clean, 0 unformatted, 0 unchecked, 0 out of scope, 36 hygiene finding(s) (0 blocking)`.
