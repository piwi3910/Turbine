# P1-T20 libturbine_hip.so — CMake build and ABI v1 ops

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 20 (`.procoder/plans/phase-1-single-request.md`, "## Task 20"): libturbine_hip.so — CMake build and ABI v1 ops. Covers S-7 (HIP shim, hipBLASLt GEMM, CK FMHA, CK rmsnorm2d where available, Turbine kernels, CMake + runtime load); build verified in Task 21. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cmake -S kernels/rocm -B /home/piwi/turbine-ci/target/kernels -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 && cmake --build /home/piwi/turbine-ci/target/kernels` inside the lab Job (Task 21)`passes (expect PASS with`libturbine_hip.so`listed by`nm -D --defined-only | grep ' T turbine_'` and no non-`turbine_` exports)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
