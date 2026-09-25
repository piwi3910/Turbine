# P1-T20 libturbine_hip.so — CMake build and ABI v1 ops

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 20 (`.procoder/plans/phase-1-single-request.md`, "## Task 20"): libturbine_hip.so — CMake build and ABI v1 ops. Covers S-7 (HIP shim, hipBLASLt GEMM, CK FMHA, CK rmsnorm2d where available, Turbine kernels, CMake + runtime load); build verified in Task 21. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cmake -S kernels/rocm -B /home/piwi/turbine-ci/target/kernels -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 && cmake --build /home/piwi/turbine-ci/target/kernels` inside the lab Job (Task 21)`passes (expect PASS with`libturbine_hip.so`listed by`nm -D --defined-only | grep ' T turbine_'` and no non-`turbine_` exports)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: novanas k3s Job `turbine-ci/turbine-t20-hip` (one R9700, `amd.com/gpu: 1`; checked first that no pod held `amd.com/gpu`) with the base tree (no `kernels/rocm`): `CMake Error: The source directory "/home/piwi/turbine-ci/src-t20/kernels/rocm" does not exist.`, Job failed. (The plan's literal red step, `scripts/lab-test.sh novanas` failing in `hip_ops`, needs Task 6's loader and Task 21's `hip_ops`, neither on this branch.)
- Green, same Job (rust:1.97-trixie, `apt-get install cmake python3`, hostPath `/opt/rocm/rocm` + `/etc/alternatives/rocm-lib` read-only, env `TURBINE_ROCM_PATH=/opt/rocm/rocm/core-7.14`): `cmake -S kernels/rocm -B /home/piwi/turbine-ci/target/kernels -DCMAKE_HIP_COMPILER=/opt/rocm/rocm/bin/hipcc -DGPU_TARGETS=gfx1201 && cmake --build /home/piwi/turbine-ci/target/kernels -j 16` → `-- CK FMHA: 17 generated sources`, `libturbine_hip.so` built without warnings; `nm -D --defined-only libturbine_hip.so | grep -c ' T turbine_'` → `36` (every ABI v1 symbol); `nm -D --defined-only … | grep -v ' turbine_'` empty → `no non-turbine_ exports`.
- C-level op check on the R9700 (throwaway harness linked against the header and the library, CPU reference with the cpu-reference numerics; the committed Rust `hip_ops` tests are Task 21): `turbine_hip: libturbine_hip.so built for gfx1201, ROCm 7.14.1, hipBLASLt 1.4.1, CK cd9574023093742434e8c992d13b89ab9a6c1cf8`; `ctx_create(99) -> -1: turbine_ctx_create: device ordinal 99 out of range (1 HIP devices visible)`; `malloc(1 PiB) -> -3: hipErrorOutOfMemory: out of memory (hipMalloc 1125899906842624 bytes)`; `attention head_dim 64 -> -2`; `test add (bit-exact) ... ok`, `embedding`, `silu_mul`, `rope` max |diff| 0 (`turbine_hip`); `rmsnorm rows=17 dim=3072 ... ok (max |diff| 0.00781, impl ck_tile_rmsnorm2d)`, dims 128/1000 `turbine_hip` max |diff| 0; `gemm` m∈{1,17} × (n,k)∈{(3072,3072),(1024,3072),(8192,3072),(3072,8192)} BF16 ok (max 0.0156 = one BF16 ulp above 2), (128256,3072) F32 ok (max 1.14e-05) (`hipblaslt`); `attention_prefill` q_len 1/17/512/4096 and `attention_decode` q_start 0/16/511/4095 (GQA 24/8) plus chunked `q_len=17 q_start=100` ok, max |diff| ≤ 0.0156 (`ck_tile_fmha_fwd`); `PASS: 0 failure(s)`, Job succeeded.
- Found on the way: the compiler's `__float2bfloat16` lowering in the RoPE kernel rounded a tie (2.0078125) up to 2.015625 while the same conversion in isolation rounded to even, so the Turbine kernels spell out BF16 round-to-nearest-even on raw bits; RoPE then matched bit for bit. hipGetLastError also returned a stale hipMalloc OOM to the next launch check, so failed runtime calls now reset it.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` 69 passed, 0 failed; `launcher.sh check` → `procoder gate: 6 clean, 0 unformatted, 0 unchecked, 4 out of scope, 37 hygiene finding(s) (0 blocking)`; `plan check phase-1-single-request` COMPLETE after recording the two extra files and the sparse CK fetch in Task 20.
- Commit `feat(kernels): libturbine_hip.so with hipBLASLt, CK FMHA and Turbine HIP kernels` on worktree branch `worktree-agent-aa80044b511d1ecfe` (branched from `phase-1-single-request`; the coordinator merges it).
