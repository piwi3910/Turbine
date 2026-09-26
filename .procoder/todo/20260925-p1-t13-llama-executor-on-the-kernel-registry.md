# P1-T13 Llama executor on the kernel registry

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 13 (`.procoder/plans/phase-1-single-request.md`, "## Task 13"): Llama executor on the kernel registry. Covers S-8, S-12 AC `tiny_model cpu_forward_matches_naive` (the ignored `tiny_model hip_matches_cpu` is written here; its acceptance run belongs to Task 21). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model --test tiny_model cpu_forward_matches_naive && cargo test -p turbine-model rope::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red (rope): `cargo test -p turbine-model rope::tests` with only the test written → `error[E0425]: cannot find function inv_freq in this scope` (compile failure: `rope::inv_freq` absent).
- Red (executor): `cargo test -p turbine-model --test tiny_model cpu_forward_matches_naive` with the test written → `error[E0432]: unresolved imports turbine_model::executor::BatchInput, turbine_model::executor::LlamaExecutor, turbine_model::executor::Logits, turbine_model::executor::ModelExecutor` (executor absent).
- Green: `cargo test -p turbine-model rope::tests` → `test executor::rope::tests::llama3_bands_follow_transformers ... ok` (theta 500000, dim 128, factor 32/1/4/8192: band 0 unchanged, band 63 = plain / 32, every band with 2048 < wavelength < 8192 strictly between).
- Green: `cargo test -p turbine-model --test tiny_model` → `test result: ok. 3 passed; 0 failed; 1 ignored`:
  - `cpu_forward_matches_naive`: seed 7, CPU registry, a 20-token prefill, 30 greedy decode steps up to position 49 (past the original length 32), then a restart at position 0. Every step's logits are within 1e-4 of the in-test naive f32 model, which reads `model.safetensors` directly and computes its own llama3 inv_freq.
  - `forward_rejects_invalid_batches`: an empty batch, a token/position count mismatch, non-consecutive positions, a start past the cached length, a token ≥ vocab and a position ≥ max_seq_len each return `KernelError::InvalidArgument`.
  - `requirements_and_workspace`: 12 distinct op configs; workspace is 8 + 2·(3·hidden + 2·q_dim + 3·inter) bytes per token, plus a fixed part for the last row, the F32 logits and inv_freq.
  - `hip_matches_cpu ... ignored`: its acceptance run belongs to Task 21.
- Mutation check: replacing the low-frequency `base / factor` with `base` in `rope::inv_freq` makes `cpu_forward_matches_naive` panic with `prefill: max abs diff 3.1566944` (restored afterwards).
- Workspace: `cargo test --workspace` → 115 passed, 0 failed, 2 ignored.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → `0 blocking`.
- Deviation: `crates/turbine-model/Cargo.toml` gains `[dev-dependencies] turbine-device` (plus `Cargo.lock`), because `hip_matches_cpu` needs a `DeviceInfo` from `turbine_device::discover` for `ShimLibrary::create_context`. No new external crate.
- `kv_layout()` is `cfg.kv_layout(max_seq_len)`: the contiguous cache is one block of `max_seq_len` tokens, so `block_bytes()` equals the cache allocation.
- Commit: `feat(turbine-model): Llama executor over the kernel registry` on the worktree branch (branched from phase-1-single-request; lands there on merge).
