# P0-T3 turbine-device — NVML and amd-smi discovery with deadlines

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 3 (`.procoder/plans/phase-0-skeleton.md`, "## Task 3"): turbine-device — NVML and amd-smi discovery with deadlines. Covers S-4; `discovery::tests::no_libraries_means_empty_inventory`, `discovery::tests::explicit_missing_library_is_fatal`, `discovery::tests::backend_timeout`, `discovery::tests::unified_memory_uses_host_total`.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-device --lib` passes (expect PASS (6 passed; `backend_timeout` takes ~10 s).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

- Red first: with only `discovery/tests.rs` and the inventory types in place, `cargo test -p turbine-device --lib` failed to compile (21 errors), e.g. `error[E0405]: cannot find trait `DiscoveryBackend` in this scope`, `error[E0425]: cannot find function `run_backends` in this scope`, `error[E0433]: cannot find module or crate `nvml` in this scope`.
- `cargo test -p turbine-device --lib` → `test result: ok. 6 passed; 0 failed; ... finished in 10.00s` (discovery::tests::{no_libraries_means_empty_inventory, explicit_missing_library_is_fatal, backend_timeout, unified_memory_uses_host_total}, discovery::amd_smi::tests::{struct_layouts_match_amdsmi_h, bdf_and_arch_format}).
- FFI declarations re-checked against ROCm 7.14.1 `amdsmi.h` (AMDSMI_LIB_VERSION 26.5.0): all ten function signatures, `AMDSMI_PROCESSOR_TYPE_AMD_GPU = 1`, `AMDSMI_MEM_TYPE_VRAM = 0`, `AMDSMI_GPU_UUID_SIZE 38`, the `amdsmi_asic_info_t` / `amdsmi_driver_info_t` field order and the `amdsmi_bdf_t` bitfield widths (3/5/8/48); sizes/offsets 896/256/520/800/768 pinned by `struct_layouts_match_amdsmi_h`.
- `unsafe`: 17 `unsafe {` blocks in `crates/turbine-device/src`, each preceded by a `// SAFETY:` comment; no `unsafe` in any other crate's `src`.
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished `dev` profile`, exit 0; `cargo test --workspace` → 18 passed, 0 failed.
- `launcher.sh check` → `procoder gate: 6 clean, 0 unformatted, 0 unchecked, 3 out of scope, 20 hygiene finding(s) (0 blocking)`.
- Commit `feat(device): runtime-loaded NVML and amd-smi discovery with per-backend deadline` is on worktree branch `worktree-agent-a846a89058794542a` (branched from `phase-0-skeleton` at `b1127ae`); the third criterion closes once it is merged into `phase-0-skeleton`.
- Not covered here: real-hardware discovery (`tests/lab.rs`, `inventory_matches_expectation`) is Task 8.
