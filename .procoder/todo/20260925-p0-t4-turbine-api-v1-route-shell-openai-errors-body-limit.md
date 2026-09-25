# P0-T4 turbine-api — V1 route shell, OpenAI errors, body limit

Status: done
Created: 2026-09-25

## Description

Phase 0 plan Task 4 (`.procoder/plans/phase-0-skeleton.md`, "## Task 4"): turbine-api — V1 route shell, OpenAI errors, body limit. Covers S-3, S-5; `api metrics_counts_requests`, `api unmatched_route_label_is_bounded`, `api request_id_echoed_or_generated`, `api route_table_phase0`, `api body_limit_413`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-api --test api` passes (expect PASS (5 passed).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed with the plan's commit message on worktree branch `worktree-agent-aa919b6bdfa0c48e8` (branched from phase-0-skeleton; coordinator merges it)

## Evidence

- Red first: `cargo test -p turbine-api --test api` with an empty lib.rs -> `error[E0432]: unresolved imports \`turbine_api::ApiError\`, \`turbine_api::ApiLimits\`, ...`
- `cargo test -p turbine-api --test api` -> `test result: ok. 5 passed; 0 failed` (body_limit_413, metrics_counts_requests, request_id_echoed_or_generated, route_table_phase0, unmatched_route_label_is_bounded)
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` -> exit 0 (`Finished \`dev\` profile`)
- `cargo test --workspace` -> every `test result: ok`, 0 failed
- `launcher.sh check` -> `procoder gate: ... 4 hygiene finding(s) (0 blocking)`
