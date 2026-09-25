# P1-T5 cpu-reference provider

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 5 (`.procoder/plans/phase-1-single-request.md`, "## Task 5"): cpu-reference provider. Covers S-6 (`cpu-reference`, pure Rust, f32 accumulation, always available). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-kernels cpu::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: with only the test module in `cpu/mod.rs`, `cargo test -p turbine-kernels cpu::tests` failed to compile: `error[E0425]: cannot find function cpu_reference_provider in this scope` (also `load`, `store`).
- Green: `cargo test -p turbine-kernels cpu::` → `test cpu::tests::gemm_and_causal_gqa_attention_reference ... ok` plus `decode_attends_the_cache_prefix_only`, `rmsnorm_rounds_like_hf_llama`, `rope_half_split_rotates_q_and_k`, `silu_mul_embedding_and_add`, `strided_store_preserves_the_bytes_between_rows`, `bad_shapes_and_dtypes_are_errors`, `round_to_matches_the_dtype_precision` and three `cpu::math::tests` → `test result: ok. 11 passed; 0 failed`. The plan test asserts GEMM F32 `[4, 4, 1, -1.5]` and causal GQA attention token 1 head 0 = `[10·σ(1), 20·(1−σ(1))]` within 1e-5.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` → 80 passed, 0 failed; `launcher.sh check` → `0 blocking`.
- Committed on the worktree branch `worktree-agent-a1c5f1be04a46dbde` (branched from `phase-1-single-request`) for the coordinator to land.
