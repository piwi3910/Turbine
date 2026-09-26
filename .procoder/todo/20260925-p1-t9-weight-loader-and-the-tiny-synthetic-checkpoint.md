# P1-T9 Weight loader and the tiny synthetic checkpoint

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 9 (`.procoder/plans/phase-1-single-request.md`, "## Task 9"): Weight loader and the tiny synthetic checkpoint. Covers S-2 AC `loader::tests::tensor_mapping`, `loader::tests::never_opens_pickle`; S-12 (tiny synthetic checkpoint). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model loader::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: tests and implementation were written in one pass, so red was shown by mutation instead of a pre-implementation run: reading every chunk from the tensor's start (dropping `+ done`) → `cargo test -p turbine-model loader::tests::tensor_mapping` fails with `model.embed_tokens.weight bytes differ from the file` (the 64-byte staging case); forcing `require_bf16` to accept every dtype → `config::tests::rejects_unsupported` fails with `called Result::unwrap_err() on an Ok value`. Both reverted.
- Green: `cargo test -p turbine-model` → `test result: ok. 11 passed; 0 failed` — `loader::tests::tensor_mapping` (tiny checkpoint on host memory: all 20 slots filled with expected shapes and bytes equal to the file, `take` works once; 64-byte staging gives identical bytes; `omit: [model.layers.1.mlp.down_proj.weight]` → `missing tensor model.layers.1.mlp.down_proj.weight` with 0 allocations on a counting memory; `extra: [model.extra.weight]` → `unexpected`; tied + `ship_lm_head` → `ignored`; untied loads `lm_head.weight` `[263, 64]`), `loader::tests::never_opens_pickle` (mode-0o000 `pytorch_model.bin` → `<path>: pickle formats are not supported`), `loader::tests::rejects_wrong_shape_before_allocating`, `testing::tiny::tests::{tiny_checkpoint_is_deterministic_and_parses, tiny_tokenizer_is_byte_level_with_llama_specials}`, `config::tests::rejects_unsupported` (F8_E4M3 case, Task 7's deferred criterion).
- Tiny checkpoint checked in transformers 4.57.1 (`uv run`, scratch script, not committed): `AutoTokenizer` encodes `"ab é"` to `[256, 97, 98, 32, 195, 169]`, `apply_chat_template` renders the Llama-3.2 system/user/assistant layout, decode skips specials; `AutoModelForCausalLM` loads it (llama3 rope scaling, tied, vocab 263) and produces logits `[1, 6, 263]`. `tokenizer_class` is `PreTrainedTokenizerFast` (coordinator note). The embedded `LLAMA32_CHAT_TEMPLATE` equals the `chat_template` of the unsloth `tokenizer_config.json` byte for byte.
- `TinyOptions::template_with_tools: false` selects a plain Llama-3-style template with no tool rendering (the plan names the option but not the alternative template).
- Workspace: `cargo test --workspace` → 80 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → `0 blocking`.
- Commit: `feat(turbine-model): positioned-read weight loader and tiny synthetic checkpoint` on the worktree branch (branched from phase-1-single-request; lands there on merge).
