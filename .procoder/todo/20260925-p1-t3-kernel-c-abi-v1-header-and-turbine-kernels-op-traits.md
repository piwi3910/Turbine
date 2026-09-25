# P1-T3 Kernel C ABI v1 header and turbine-kernels op traits

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 3 (`.procoder/plans/phase-1-single-request.md`, "## Task 3"): Kernel C ABI v1 header and turbine-kernels op traits. Covers S-1 AC `cargo test -p turbine-kernels --test unsafe_isolation`; S-1/S-7 AC `cargo test -p turbine-kernels --test abi_header_neutral`; S-6 (traits carry no vendor types). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kernels --test unsafe_isolation --test abi_header_neutral` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
