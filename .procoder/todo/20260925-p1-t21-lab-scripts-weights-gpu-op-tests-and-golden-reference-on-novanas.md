# P1-T21 Lab scripts, weights, GPU op tests and golden reference on novanas

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 21 (`.procoder/plans/phase-1-single-request.md`, "## Task 21"): Lab scripts, weights, GPU op tests and golden reference on novanas. Covers S-7/S-13 AC `scripts/lab-test.sh novanas` with `hip_ops`; S-8/S-12/S-13 `tiny_model hip_matches_cpu`; S-11 `golden hf_reference_matches_cpu`; S-11/S-13 `golden logits_match_reference`; S-13 (lab execution, weights). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS with log lines `libturbine_hip.so` built for `gfx1201`, `hipBLASLt 1.4.1`, `CK cd9574023093742434e8c992d13b89ab9a6c1cf8`, `test gemm_matches_cpu ... ok`, `test attention_matches_cpu ... ok`, `test hip_matches_cpu ... ok`, `test hf_reference_matches_cpu ... ok`, `test logits_match_reference ... ok`, Job exit 0)
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
