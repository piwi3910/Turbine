# P0-T2 turbine-observability — tracing, metrics registry, HTTP layers

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 2 (`.procoder/plans/phase-0-skeleton.md`, "## Task 2"): turbine-observability — tracing, metrics registry, HTTP layers. Covers S-3 (subscriber, registry, request-id, bounded labels; the S-3 acceptance tests exercise these through the router in Task 4).. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-observability` passes (expect PASS (2 passed).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
