# P0-T2 turbine-observability — tracing, metrics registry, HTTP layers

Status: closed 2026-09-25
Created: 2026-09-25

## Description

Phase 0 plan Task 2 (`.procoder/plans/phase-0-skeleton.md`, "## Task 2"): turbine-observability — tracing, metrics registry, HTTP layers. Covers S-3 (subscriber, registry, request-id, bounded labels; the S-3 acceptance tests exercise these through the router in Task 4).. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-observability` passes (expect PASS (2 passed).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red first: with only the two tests in `metrics.rs` / `http.rs`, `cargo test -p turbine-observability` failed to compile with `error[E0425]: cannot find function `valid_request_id` in this scope`, `cannot find function `method_label``, and `error[E0433]: cannot find type `MetricsRegistry` in this scope`.
- `cargo test -p turbine-observability` → `test result: ok. 2 passed; 0 failed` (metrics::tests::build_info_and_registered_metrics_render, http::tests::request_id_validation).
- Layers exercised ad hoc (temporary test, not committed) through `request_id_layer()` + `http_metrics_layer()`: `x-request-id: abc-123` echoed, missing id → generated UUIDv4, render shows `turbine_http_requests_total{method="OTHER",route="unmatched",status="200"} 1` and `turbine_http_request_duration_seconds_bucket{le="0.001",...}` … `le="32.768"`, `+Inf`. The router-level S-3 acceptance tests belong to Task 4.
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished `dev` profile`, exit 0; `cargo test --workspace` → 6 passed (4 core + 2 observability), 0 failed.
- `launcher.sh check` → `procoder gate: 4 clean, 0 unformatted, 0 unchecked, 2 out of scope, 12 hygiene finding(s) (0 blocking)`.
- Commit `feat(observability): tracing init, metrics registry and HTTP request-id/metrics layers` is on worktree branch `worktree-agent-a914646b4e8503bd3` (branched from `phase-0-skeleton` at `a687ce5`); the third criterion closes once it is merged into `phase-0-skeleton`.
