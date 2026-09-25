# P0-T4 turbine-api — V1 route shell, OpenAI errors, body limit

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 4 (`.procoder/plans/phase-0-skeleton.md`, "## Task 4"): turbine-api — V1 route shell, OpenAI errors, body limit. Covers S-3, S-5; `api metrics_counts_requests`, `api unmatched_route_label_is_bounded`, `api request_id_echoed_or_generated`, `api route_table_phase0`, `api body_limit_413`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-api --test api` passes (expect PASS (5 passed).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
