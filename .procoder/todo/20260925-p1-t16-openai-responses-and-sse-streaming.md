# P1-T16 OpenAI responses and SSE streaming

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 16 (`.procoder/plans/phase-1-single-request.md`, "## Task 16"): OpenAI responses and SSE streaming. Covers S-10 (streaming and non-streaming shapes); end-to-end in Task 17 `tiny_server completions_stream_and_non_stream`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-api` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed with the plan's commit message on worktree branch `worktree-agent-ab142049d3f73b2cd` (branched from phase-1-single-request; coordinator merges it)

## Evidence

- Red first: `cargo test -p turbine-api --test openai response_shapes_and_stream_order` -> `assertion left == right failed: {"error":{"code":"model_not_loaded",...}} left: 503 right: 200` (the Phase 0 503 handler bodies)
- `cargo test -p turbine-api` -> `tests/api.rs: test result: ok. 5 passed; 0 failed` (Phase 0, unchanged) and `tests/openai.rs: test result: ok. 6 passed; 0 failed` (adds response_shapes_and_stream_order, handler_errors_before_the_stream)
- C-3: `response_shapes_and_stream_order` asserts a mid-stream `Error{internal_error}` yields the content chunk, `{"error":{"message":"kernel failed","type":"server_error","code":"internal_error"}}`, then `[DONE]`
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` -> exit 0 (`Finished \`dev\` profile`)
- `cargo test --workspace` -> every `test result: ok`, 54 passed, 0 failed
- `launcher.sh check` -> `procoder gate: 5 clean, 0 unformatted, 0 unchecked, 0 out of scope, 19 hygiene finding(s) (0 blocking)`
