# P1-T4 Kernel registry with logged, metered selection

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 4 (`.procoder/plans/phase-1-single-request.md`, "## Task 4"): Kernel registry with logged, metered selection. Covers S-6 AC `registry::tests::selection_order_and_reason`, `registry::tests::no_provider_is_startup_error`; S-14 (kernel selection log + gauge). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kernels registry::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
