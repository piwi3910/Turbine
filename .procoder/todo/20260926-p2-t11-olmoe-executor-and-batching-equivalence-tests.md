# P2-T11 OLMoE executor and batching equivalence tests

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 11 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 11"): OLMoE executor and batching equivalence tests. Covers S-16 AC `tiny_model olmoe_cpu_forward_matches_naive`; S-4/S-9 AC `tiny_model chunked_prefill_matches_unchunked`; S-5/S-9 AC `tiny_model paged_matches_contiguous` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-model --test tiny_model` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): OLMoE executor and batching equivalence tests`)

## Evidence

- Red first: `cargo test -p turbine-model --test tiny_model olmoe_cpu_forward_matches_naive` before the implementation → `error[E0432]: unresolved imports turbine_model::executor::OlmoeExecutor, turbine_model::executor::build_executor` and `error[E0425]: cannot find function requirements in module executor`.
- `CARGO_INCREMENTAL=0 cargo test -p turbine-model --test tiny_model` → `test olmoe_cpu_forward_matches_naive ... ok`, `test chunked_prefill_matches_unchunked ... ok`, `test paged_matches_contiguous ... ok`, `test result: ok. 10 passed; 0 failed; 2 ignored`.
- Mutation check (each applied alone to `executor/olmoe.rs`, then reverted): routing forced to `renormalize: true`, Q projection without its `q_norm`, MoE accumulator not reset to zero → each gives `test olmoe_cpu_forward_matches_naive ... FAILED`.
- `cargo fmt --all --check && CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets -- -D warnings` → exit 0 (`Finished dev profile`).
- `CARGO_INCREMENTAL=0 cargo test --workspace` → 37 `test result: ok` lines, no failures.
- `launcher.sh check` → `procoder gate: 4 clean, 0 unformatted, 0 unchecked, 0 out of scope, 22 hygiene finding(s) (0 blocking)`.
- Signatures follow the tree's Task 10 deviation (`block_tokens` argument): `OlmoeExecutor::requirements(cfg, block_tokens)`, `OlmoeExecutor::workspace_bytes(cfg, block_tokens, max_batch_tokens, max_seqs)`, `OlmoeExecutor::new(cfg, weights, registry, mem, block_tokens, max_batch_tokens, max_seqs)`, `build_executor(cfg, weights, registry, mem, block_tokens, max_batch_tokens, max_seqs) -> Result<Box<dyn ModelExecutor>, ModelError>`; plus the dispatchers `executor::requirements(cfg, block_tokens)` and `executor::workspace_bytes(cfg, block_tokens, max_batch_tokens, max_seqs)`.
