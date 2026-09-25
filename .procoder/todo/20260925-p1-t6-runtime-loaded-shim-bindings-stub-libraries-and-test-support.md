# P1-T6 Runtime-loaded shim bindings, stub libraries and test_support

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 6 (`.procoder/plans/phase-1-single-request.md`, "## Task 6"): Runtime-loaded shim bindings, stub libraries and test_support. Covers S-7 AC `shim::tests::abi_and_arch_mismatch_are_fatal`; S-1 (nothing links HIP at build time). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-kernels` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: with only the test module in `shim.rs` (build.rs and the stub already compiling `libturbine_stub_abi999.so` / `libturbine_stub_gfx942.so` into OUT_DIR), `cargo test -p turbine-kernels shim::tests::abi_and_arch_mismatch_are_fatal` failed to compile: `error[E0433]: cannot find type ShimLibrary in this scope`.
- Green: `cargo test -p turbine-kernels` → `test shim::tests::abi_and_arch_mismatch_are_fatal ... ok` (asserts `kernel ABI version mismatch: library 999, expected 1`, `build_archs() == ["gfx942"]`, `device arch gfx1201 not in library build archs gfx942`, and a message starting `cannot load /nonexistent/libturbine_hip.so`), plus `backend_mismatch_is_fatal`, `context_memory_provider_and_destroy_once` (h2d/d2h round trip, OOM with the requested size, d2d Unsupported, provider id `hip`, `stub_add` impl, `-2` mapped to `Unsupported` with the `turbine_last_error` message, host memory refused, live stub contexts back to the baseline after the last `Arc` drops), `descriptors_match_the_c_layout`, `search_order`, and `test_support::tests::{backend_match_skip_and_unset, unset_model_dir_fails}` → `test result: ok. 27 passed; 0 failed` (unit) plus `unsafe_isolation` 3 passed and `abi_header_neutral` 3 passed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` → 88 passed, 0 failed, 1 ignored (pre-existing lab test); `launcher.sh check` → `0 blocking`.
- Committed on the worktree branch `worktree-agent-a1c5f1be04a46dbde` (branched from `phase-1-single-request`) for the coordinator to land.
