# P2-T10 Ragged paged batches in the Llama executor

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 10 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 10"): Ragged paged batches in the Llama executor. Covers S-9 (batched execution, one D2H copy), S-5 (executor on the paged pool); the named paged/chunked ACs are delivered in Task 11 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-model --test tiny_model` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): ragged paged batches for the Llama executor`)

## Evidence

- Red: `cargo test -p turbine-model --test tiny_model paged_llama_single_sequence` against the Phase 1 executor fails to compile (`BatchInput` has no `seqs`/`kv`, no `SeqSlice`, `LlamaExecutor::new` takes no `block_tokens`/`max_seqs`) — the test drives the P2 interface.
- `cargo test -p turbine-model --test tiny_model` → `test paged_llama_single_sequence ... ok`, `test ragged_batch_rows_match_single_sequences ... ok`, `test result: ok. 7 passed; 0 failed; 2 ignored`
- Mutation check: packing every block-table entry as block 1 makes `paged_llama_single_sequence` fail (`reference vs naive` assertion) along with `ragged_batch_rows_match_single_sequences`, `cpu_forward_matches_naive*` and `cpu_trace_recomputes_exactly`.
- `cargo test -p turbine-model --lib executor::batch` → `packs_a_ragged_batch ... ok`, `rejects_malformed_batches ... ok`
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0 (`Finished \`dev\` profile`)
- `cargo test --workspace` → 37 test binaries, 201 passed, 0 failed (Phase 1 tiny_model, golden, generate and turbine-server tests included)
- `launcher.sh check` → `procoder gate: 3 clean, 0 unformatted, 0 unchecked, 0 out of scope, 21 hygiene finding(s) (0 blocking)`
