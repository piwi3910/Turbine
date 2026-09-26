# P2-T18 libturbine_hip.so ABI v2 — paged attention, block copy and MoE

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 18 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 18"): libturbine_hip.so ABI v2 — paged attention, block copy and MoE. Covers S-5, S-16 (HIP paged-KV and MoE providers); the ignored `hip_ops paged_and_moe_ops` is written here and accepted by its lab run in Task 20 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(kernels): hip ABI v2 paged attention, block copy and MoE ops`)

## Evidence

- Gate (macOS, worktree of `phase-2-serving-runtime` at 9e66ff3 + this task): `cargo fmt --all --check` exit 0; `CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets -- -D warnings` exit 0 (Finished dev profile); `CARGO_INCREMENTAL=0 cargo test --workspace` → 37 binaries `test result: ok`, 219 passed, 0 failed; `launcher.sh check` → `procoder gate: 7 clean, 0 unformatted, 0 unchecked, 3 out of scope, 43 hygiene finding(s) (0 blocking)`.
- Red step on macOS (no R9700): `TURBINE_TEST_BACKEND=hip cargo test -p turbine-kernels --test hip_ops paged_and_moe_ops -- --ignored` → `test paged_and_moe_ops ... FAILED` at `no AMD device in the inventory` (without the variable: `TURBINE_TEST_BACKEND is not set`).
- HIP sources front-end check (not a build): Homebrew clang 22 `-x hip --offload-arch=gfx1201 -fsyntax-only` against the ROCm 7.14.1 headers copied from novanas and the pinned CK checkout — `paged_attention.hip`, `moe.hip`, `copy_blocks.cpp`, `context.cpp`, `moe.cpp`, `gemm.cpp`, `memory.cpp`: no diagnostics; `paged_attention.cpp` reports only the same 4 macOS-host errors (`__assert_rtn` in `amd_tdm_descriptor.hpp`, `min(size_t, uint64_t)` in `rotating_buffers.hpp`) as the lab-proven `attention.cpp`.
- CK generator filter checked with the pinned `generate.py --targets gfx1201 --api fwd,pagedkv_prefill --filter …` → 20 blobs, including `fmha_fwd_pagedkv_d128_bf16_group_…_nmask_…` and `…_mask_…` and `fmha_fwd_pagedkv_api.cpp`, whose dispatcher accepts the traits `paged_attention.cpp` passes.
- Not run here: the lab criterion (`scripts/lab-test.sh novanas`, expect `test paged_and_moe_ops ... ok` with `hipblaslt_per_expert`, `turbine_hip` and `ck_tile_fmha_pagedkv` in the log) — left to the coordinator's lab build.
