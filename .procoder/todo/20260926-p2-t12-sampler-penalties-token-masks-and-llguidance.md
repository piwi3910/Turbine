# P2-T12 Sampler penalties, token masks and llguidance structured output

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 12 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 12"): Sampler penalties, token masks and llguidance structured output. Covers S-17 AC `structured::tests::mask_applied_before_sampling`; S-10 (penalties, bias, min_tokens, stop_token_ids in sampling); S-6 (sampler state kept across preemption) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-model` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): sampler penalties, token masks and llguidance structured output`)

## Evidence

- Red: `cargo test -p turbine-model structured::tests::mask_applied_before_sampling` and `… structured::tests::json_schema_matcher_on_llama_tokenizer` before the implementation → `error[E0425]: cannot find type TokenMask in this scope` (also `TokenMatcher`, `step_mask`, `GrammarCompiler`, `GrammarLimits`); `error: could not compile turbine-model (lib test) due to 10 previous errors`.
- Green: `cargo test -p turbine-model` → `test structured::tests::mask_applied_before_sampling ... ok`, `test structured::tests::json_schema_matcher_on_llama_tokenizer ... ok`, `test sampler::tests::state_round_trip_continues_the_stream ... ok`; lib `test result: ok. 56 passed; 0 failed`, `tests/golden.rs` `ok. 3 passed; 0 failed; 3 ignored`, `tests/tiny_model.rs` `ok. 5 passed; 0 failed; 2 ignored`.
- Gate: `cargo fmt --all --check` exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished dev profile`; `cargo test --workspace` → every `test result: ok`, 0 failed; `launcher.sh check` → `0 blocking`.
- Pins: `llguidance = "=1.8.0"` and `toktrie_hf_tokenizers = "=1.8.0"` resolve and build on Rust 1.97 against `tokenizers =0.21.4` (no tokenizer upgrade); dev-dep `jsonschema 0.58.0`.
