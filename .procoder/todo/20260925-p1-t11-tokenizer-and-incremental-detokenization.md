# P1-T11 Tokenizer and incremental detokenization

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 11 (`.procoder/plans/phase-1-single-request.md`, "## Task 11"): Tokenizer and incremental detokenization. Covers S-4 AC `tokenizer::tests::incremental_detokenize_utf8`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model tokenizer::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-model tokenizer::tests::incremental_detokenize_utf8` with `push`/`flush` returning `None` → `assertion left == right failed; left: "" right: "Hello 世界 👩‍👩‍👧‍👦 🇧🇪 naïve café 日本語テキスト"` (nothing streamed).
- Green: `cargo test -p turbine-model tokenizer::tests` → `test result: ok. 4 passed; 0 failed` (`incremental_detokenize_utf8`: streamed chunks + `flush()` == `decode(all)`, no chunk holds U+FFFD, at least one `push` returned `None`; `special_tokens_and_lookups`: vocab 128256, `<|begin_of_text|>` 128000, `<|eot_id|>` 128009, BOS added only with `add_special_tokens`; `special_token_is_skipped_in_stream`; `missing_file_is_io_error`).
- Fixtures: `tokenizer.json` and `tokenizer_config.json` fetched raw from `unsloth/Llama-3.2-3B-Instruct` at `006f5dcd1393c3add266de40994ba96225e9689d`; `shasum -a 256` → `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b  tokenizer.json`, `9ddd255c19fe319c8d4e891163540382e9fbda99f394674f2a929efc47d57458  tokenizer_config.json`; committed byte-exact (`.prettierignore`, 20 MB limit).
- Workspace: `cargo test --workspace` → 76 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → 0 blocking.
- Commit: `feat(turbine-model): tokenizer and UTF-8-safe incremental detokenizer` on a worktree branch from phase-1-single-request (lands there on merge).
