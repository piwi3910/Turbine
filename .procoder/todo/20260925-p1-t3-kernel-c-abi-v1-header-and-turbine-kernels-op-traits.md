# P1-T3 Kernel C ABI v1 header and turbine-kernels op traits

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 3 (`.procoder/plans/phase-1-single-request.md`, "## Task 3"): Kernel C ABI v1 header and turbine-kernels op traits. Covers S-1 AC `cargo test -p turbine-kernels --test unsafe_isolation`; S-1/S-7 AC `cargo test -p turbine-kernels --test abi_header_neutral`; S-6 (traits carry no vendor types). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-kernels --test unsafe_isolation --test abi_header_neutral` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: before `lib.rs`/`ops` existed, `cargo test -p turbine-kernels --test abi_header_neutral` failed to compile with `error[E0432]: unresolved import turbine_kernels::TURBINE_KERNELS_ABI_VERSION` / `could not find ops in turbine_kernels`; `--test unsafe_isolation` first failed with `crates/turbine-device/src/discovery/amd_smi.rs:200: unsafe block or impl without a preceding // SAFETY: comment` (a rustfmt-wrapped `let driver_ok =\n    unsafe { … }` whose SAFETY comment sits above the statement) — the scanner now walks over continuation lines of the same statement (covered by `scanner_flags_leaks_and_missing_safety_comments`).
- Injection check: an uncompiled `crates/turbine-tensor/src/injected.rs` with `unsafe { std::mem::zeroed() }` made the scan fail with `crates/turbine-tensor/src/injected.rs:2: unsafe outside ["crates/turbine-device/src", "crates/turbine-kernels/src"]`; the same injection into compiled `dtype.rs` is already rejected at build time by `unsafe_code = "forbid"`. Appending `typedef int hipStream_t;` to the header made `header_has_no_vendor_identifiers` fail with `vendor identifiers in turbine_kernels.h: ["hipStream_t"]`. Both injections reverted.
- Green: `cargo test -p turbine-kernels --test unsafe_isolation --test abi_header_neutral` → `test result: ok. 3 passed` (abi_header_neutral) and `test result: ok. 3 passed` (unsafe_isolation); unit tests `test result: ok. 5 passed`.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0 (`Finished dev profile`); `cargo test --workspace` 59 passed, 0 failed; `launcher.sh check` → `procoder gate: 5 clean, 0 unformatted, 0 unchecked, 3 out of scope, 43 hygiene finding(s) (0 blocking)`.
- Header compiles standalone: `cc -fsyntax-only -x c` and `-x c++` on `kernels/include/turbine_kernels.h` exit 0.
- Commit `feat(turbine-kernels): vendor-neutral kernel ABI v1 and op capability traits` on the Task 3 worktree branch `worktree-agent-a37a622a3d460cea3` (branched from `phase-1-single-request`; the coordinator merges it).
