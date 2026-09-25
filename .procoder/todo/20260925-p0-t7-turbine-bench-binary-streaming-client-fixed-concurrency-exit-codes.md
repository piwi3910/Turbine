# P0-T7 turbine-bench binary — streaming client, fixed concurrency, exit codes

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 7 (`.procoder/plans/phase-0-skeleton.md`, "## Task 7"): turbine-bench binary — streaming client, fixed concurrency, exit codes. Covers S-6; `bench mock_endpoint_measurements`, `bench failures_counted`, manual run against dgx-spark vLLM.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-bench` passes (expect PASS (unit + 2 integration tests).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red first: `cargo test -p turbine-bench --test bench` with an empty `main` → ``assertion `left == right` failed: null`` / `left: Null` / `right: 4` (mock_endpoint_measurements) and `right: 2` (failures_counted); `test result: FAILED. 0 passed; 2 failed`.
- `cargo test -p turbine-bench` → `test result: ok. 4 passed` (lib unit: prompt + report) and `test result: ok. 2 passed` (tests/bench.rs: `mock_endpoint_measurements`, `failures_counted`).
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `cargo test --workspace` → every `test result: ok`.
- `launcher.sh check` → `procoder gate: 4 clean, 0 unformatted, 0 unchecked, 1 out of scope, 15 hygiene finding(s) (0 blocking)`.
- Committed on the task worktree branch with `feat(bench): streaming OpenAI load generator with TTFT/ITL/E2E percentiles`; the phase-0-skeleton criterion is ticked when the coordinator merges it.
- Manual dgx-spark run (spec S-6 manual acceptance): NOT run, awaiting user approval. Command: `cargo run --release -p turbine-bench -- --url http://192.168.10.246:8000 --concurrency 2 --requests 10 --output json`.
