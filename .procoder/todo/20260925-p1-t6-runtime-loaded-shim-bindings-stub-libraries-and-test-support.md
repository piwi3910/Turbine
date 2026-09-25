# P1-T6 Runtime-loaded shim bindings, stub libraries and test_support

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 6 (`.procoder/plans/phase-1-single-request.md`, "## Task 6"): Runtime-loaded shim bindings, stub libraries and test_support. Covers S-7 AC `shim::tests::abi_and_arch_mismatch_are_fatal`; S-1 (nothing links HIP at build time). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kernels` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
