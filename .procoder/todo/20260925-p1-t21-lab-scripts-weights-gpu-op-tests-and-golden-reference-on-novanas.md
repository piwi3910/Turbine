# P1-T21 Lab scripts, weights, GPU op tests and golden reference on novanas

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 21 (`.procoder/plans/phase-1-single-request.md`, "## Task 21"): Lab scripts, weights, GPU op tests and golden reference on novanas. Covers S-7/S-13 AC `scripts/lab-test.sh novanas` with `hip_ops`; S-8/S-12/S-13 `tiny_model hip_matches_cpu`; S-11 `golden hf_reference_matches_cpu`; S-11/S-13 `golden logits_match_reference`; S-13 (lab execution, weights). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS with log lines `libturbine_hip.so` built for `gfx1201`, `hipBLASLt 1.4.1`, `CK cd9574023093742434e8c992d13b89ab9a6c1cf8`, `test gemm_matches_cpu ... ok`, `test attention_matches_cpu ... ok`, `test hip_matches_cpu ... ok`, `test hf_reference_matches_cpu ... ok`, `test logits_match_reference ... ok`, Job exit 0)
- [x] Lab Job builds `libturbine_hip.so` for `gfx1201` with hipBLASLt 1.4.1 and CK `cd9574023093742434e8c992d13b89ab9a6c1cf8`
- [x] `hip_ops` passes on the R9700: `test gemm_matches_cpu ... ok`, `test attention_matches_cpu ... ok`, `test norm_rope_silu_embedding_add_match_cpu ... ok`, each printing the `_impl` name
- [x] Device inventory lab test passes on novanas (`test inventory_matches_expectation ... ok`, 2 × R9700 gfx1201)
- [ ] `tiny_model hip_matches_cpu` passes on novanas (blocked: HIP attention has no head_dim 16 kernel, see Evidence)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

Partial: the script/manifest part of the task (no Job has been run on novanas; the lab-run criterion and the `hip_ops.rs` / reference steps stay open). Weights are already at `/home/piwi/turbine-models/llama-3.2-3b-instruct` (ungated mirror `unsloth/Llama-3.2-3B-Instruct` @ `006f5dcd1393c3add266de40994ba96225e9689d`, user decision; scripts never read or pass a token); the golden reference is already committed under `tests/golden/llama-3.2-3b-instruct/`.

- `scripts/lab/novanas-test-job.yaml`: setup script (`apt-get install -y -qq cmake python3`, `cmake -S kernels/rocm -B /home/piwi/turbine-ci/target/kernels … -DGPU_TARGETS=gfx1201`, uv 0.11.2 with `UV_CACHE_DIR=/home/piwi/turbine-ci/uv-cache`), env `TURBINE_KERNEL_LIBRARY`, `TURBINE_TEST_BACKEND=hip`, `TURBINE_TEST_MODEL_DIR=/models/llama-3.2-3b-instruct`, `TURBINE_ROCM_PATH=/opt/rocm/rocm/core-7.14` (T20 finding), read-only hostPath `/home/piwi/turbine-models` → `/models`; `activeDeadlineSeconds` 1800 → 5400 for a cold CK + torch run.
- `scripts/lab/novanas-serve-job.yaml`, `scripts/lab/phase1-novanas.yaml`, `scripts/lab-serve.sh` (with `--dry-run`) added.
- `bash -n scripts/lab-serve.sh && shellcheck scripts/lab-serve.sh` → exit 0, no findings.
- `ssh piwi@192.168.10.203 'kubectl apply --dry-run=client -o name -f -' < scripts/lab/novanas-{test,serve}-job.yaml` → `job.batch/turbine-lab-test`, `job.batch/turbine-lab-serve`.
- `cargo test -p turbine-bench --test lab_scripts` → `test result: ok. 6 passed; 0 failed` (config loads via `turbine_core::config::load`; manifests' env/mounts/script order; `lab-serve.sh --dry-run` start/stop sequences with `ssh`/`rsync`/`curl`/`kubectl` stubbed to fail if called; usage errors exit 2). Mutations `hostNetwork: false` and `--stop` deleting `turbine-lab-test` each fail one test.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `cargo test --workspace` → exit 0 (117 passed, 1 ignored).

### Lab run (part 2: `hip_ops.rs`, 2026-09-26)

`crates/turbine-kernels/tests/hip_ops.rs`: three ignored tests through `ShimLibrary`/`shim_provider` against `cpu_reference_provider` on seeded splitmix64 inputs; tolerance |Δ| ≤ 1e-2 BF16 (one BF16 ulp where |ref| > 2), ≤ 1e-4 F32. The tests hold a process-wide lock: run 2 showed three parallel in-process `discover` calls all finding no AMD device, while the inventory test alone saw 2. Locally, without `TURBINE_KERNEL_LIBRARY`, the tests fail as expected (`test result: FAILED. 0 passed; 1 failed`). `novanas-test-job.yaml`: drops a kernel build cache that was configured from another source tree (run 1 failed on T20's `src-t20` cache: `CMake Error: The source ".../src/kernels/rocm/CMakeLists.txt" does not match the source ".../src-t20/kernels/rocm/CMakeLists.txt"`).

`scripts/lab-test.sh novanas`, run 3 (GPUs free: no pod requested `amd.com/gpu` except a Succeeded T20 pod):

- `turbine_hip: libturbine_hip.so built for gfx1201, ROCm 7.14.1, hipBLASLt 1.4.1, CK cd9574023093742434e8c992d13b89ab9a6c1cf8`
- `test inventory_matches_expectation ... ok`
- `test attention_matches_cpu ... ok`, `test gemm_matches_cpu ... ok`, `test norm_rope_silu_embedding_add_match_cpu ... ok`
- e.g. `gemm m=17 n=128256 k=3072 … c_dtype=f32: impl=hipblaslt max |Δ| 1.073e-5`, `attention_prefill q_len=4096 q_start=0 rows 0..16: impl=ck_tile_fmha_fwd max |Δ| 1.562e-2 ok`, `rmsnorm rows=17 dim=3072 dtype=bf16: impl=ck_tile_rmsnorm2d max |Δ| 1.562e-2 ok`, rope/silu_mul/embedding/add `impl=turbine_hip max |Δ| 0.000e0`
- FAIL: `test hip_matches_cpu ... FAILED`: `every op has a provider: NoProvider { op: AttentionPrefill, config: "head_dim=16 kv_heads=2 dtype=bf16 q_heads=4 causal=1" }`. The tiny checkpoint has head_dim 16, but spec S-7 gives the HIP attention head_dim 128 only. This is a spec gap between S-7 and S-8/S-13 and needs a decision: a head_dim-128 tiny variant for the HIP test, a Turbine HIP attention kernel for other head dims, or a CPU fallback in the test registry. `lab-test: novanas: tests failed (exit 101)`. The golden tests `hf_reference_matches_cpu` / `logits_match_reference` do not exist in the tree yet.
