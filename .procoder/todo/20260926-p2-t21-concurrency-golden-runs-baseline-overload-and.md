# P2-T21 Concurrency golden runs, baseline, overload and acceptance on macOS

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 21 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 21"): Concurrency golden runs, baseline, overload and acceptance on macOS. Covers S-1 AC (macOS build/test/clippy/fmt and `cargo tree -p turbine-scheduler`); S-15/S-16 AC manual golden runs with `--concurrency 16`; S-15 AC manual baseline run; S-15 AC manual overload run Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] macOS: `cargo build --workspace`, `cargo test --workspace`, clippy and fmt exit 0; `cargo tree -p turbine-scheduler` has no GPU/vendor crate
- [x] Llama `turbine-golden compare --concurrency 16` against `scripts/lab/phase2-novanas-llama.yaml` exits 0
- [x] OLMoE `turbine-golden compare --concurrency 16` against `scripts/lab/phase2-novanas-olmoe.yaml` exits 0
- [x] Llama baseline (`turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos`) exits 0 with `"requests_failed": 0`; vLLM-ROCm recorded
- [x] OLMoE baseline exits 0 with `"requests_failed": 0`; vLLM-ROCm recorded
- [x] Overload run (`--concurrency 512 --requests 2000`) exits 0 with refusals, pod `Running`, `/ready` 200, `blocks_used: 0` once idle
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message

## Evidence

- Llama golden (2026-09-26, Phase 2 engine at 7850162, `scripts/lab-serve.sh novanas scripts/lab/phase2-novanas-llama.yaml`): `turbine-golden compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --concurrency 16` → exit 0, `PASS: 16/16 prompts passing (need 14)`, every prompt identical_prefix 32/32, max likely Δ 0.1352 (p12), max tail Δ 0.2974 (p16).
- Llama baseline, Turbine (same server): exit 0, `requests_ok` 200, `requests_failed` 0, output_token_throughput 92.2 tok/s, TTFT p50 1528 ms, ITL p50 164 ms, e2e p50 44.3 s. Server metrics: decode forward 157.2 s / 3322 = 47 ms per step, batch mostly 16; KV afterwards `blocks_used: 0` of 4681 (16-token blocks). Impls: gemm hipblaslt, rmsnorm ck_tile_rmsnorm2d, paged attention turbine_hip.
- vLLM-ROCm baseline (`scripts/lab-serve.sh novanas --vllm <slug>`, image rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0@sha256:19ad8dc5…, ROCm 7.14.1, one Radeon AI PRO R9700 gfx1201, same bench command against :18100): Llama-3.2-3B 200/200 ok, 738.0 tok/s, TTFT p50 338 ms, ITL p50 17.0 ms; OLMoE-1B-7B 200/200 ok, 534.9 tok/s, TTFT p50 201 ms, ITL p50 26.5 ms. Performance gap handled by Phase 2c (user decision 2026-09-26: ≥ 75% of vLLM-ROCm).
- Overload (Llama, `scheduler.max_queued_requests: 256`): `turbine-bench --concurrency 512 --requests 2000 --max-tokens 256 --ignore-eos --output json` → exit 0, requests_ok 64, requests_failed 1936 (429 `queue_full` and 503 `queue_timeout` refusals, e.g. `stream error: {"code":"queue_timeout",…}`); afterwards `/ready` 200, pod `turbine-lab-serve-0926062308-13c3621e` `1/1 Running`, `/turbine/v1/kv` `blocks_used: 0`.
- OLMoE: first serve failed at load (`device-to-device copy needs kernel ABI v3` — OLMoE executor used copy_d2d); fix in progress, then golden and baseline.
- OLMoE fix: executor stacks expert weights with H2D copies (ABI v2, no copy_d2d); router computes BF16 logits and ports torch.topk tie order (bee6ab5); tolerance calibrated to transformers' self-spread, likely 1.01 / tail 1.66 (c5c0d3b, user decision 2026-09-26).
- Final served runs (2026-09-26, `scripts/lab-bench.sh --gpu 0`, native release server pinned to GPU 0, configs `scripts/lab/phase2-novanas-<model>.yaml`):
  - OLMoE (c5c0d3b tree + 8acce3e test fix): golden c1 `PASS: 16/16 prompts passing (need 14); strict bounds (concurrency 1)`, golden c16 `PASS: 16/16 prompts passing (need 14); batched bounds (concurrency 16)`; baseline `turbine-bench --concurrency 16 --requests 200 --prompt-words 512 --max-tokens 256 --ignore-eos` ok=200 failed=0, 567.1 tok/s, ITL p50 25.5 ms, TTFT p50 405 ms. vLLM-ROCm on the same card (GPU 0): 535.5 tok/s, ITL p50 26.6 ms, TTFT p50 200 ms.
  - Llama (8acce3e): golden c1 and c16 PASS; baseline ok=200 failed=0, 745.0 tok/s, ITL p50 17.8 ms, TTFT p50 573 ms. vLLM-ROCm on GPU 0: 715.1 tok/s.
- S-1 gate (spec amended: novanas build host via `scripts/remote-cargo.sh`, user decision 2026-09-26), 8acce3e: `cargo fmt --all --check` ok; `build --workspace` ok; `test --workspace` 310 passed, 0 failed; `clippy --workspace --all-targets -- -D warnings` ok; `cargo tree -p turbine-scheduler` contains no turbine-kernels, turbine-model, hip, rocm or cuda.
- Commits landed on `phase-2c-performance` (Phase 2 and Phase 2c share the branch; merged to main together, user decision 2026-09-26).
