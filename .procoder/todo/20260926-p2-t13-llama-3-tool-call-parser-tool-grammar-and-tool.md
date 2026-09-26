# P2-T13 Llama-3 tool-call parser, tool grammar and tool rendering

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 13 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 13"): Llama-3 tool-call parser, tool grammar and tool rendering. Covers S-18 AC `tools::tests::llama3_json_parser`, `chat_template::tests::renders_llama_tools` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-model tools::tests chat_template::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): llama3_json tool-call parser and tool grammar`)

## Evidence

- Red first: `cargo test -p turbine-model --lib chat_template::tests::renders_llama_tools` → panicked `…/tests/fixtures/llama-3.2-3b-instruct/expected_renders.json: No such file or directory`; `cargo test -p turbine-model --lib tools::tests::llama3_json_parser` → `error[E0425]: cannot find type ToolParse` (and the other tools names) before `tools.rs` had an implementation.
- Fixture regenerated locally from the committed tokenizer files (no weights, no lab host): `uv run --with 'transformers==4.57.1' --with jinja2 python3 scripts/golden/render_fixture.py crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct` → `system_user: 65 tokens`, `user_only: 38 tokens`, `tools: 254 tokens`, exit 0, `transformers_version` 4.57.1; the `system_user` / `user_only` cases are identical to `tests/golden/llama-3.2-3b-instruct/expected_renders.json`.
- The plan's run line passes two filters to cargo, which cargo rejects (`error: unexpected argument 'chat_template::tests' found`); run as `cargo test -p turbine-model -- tools::tests chat_template::tests` → `test result: ok. 10 passed; 0 failed` (tools: `llama3_json_parser`, `call_ids_are_24_alphanumerics`, `tool_grammar_shapes`; chat_template: 7 incl. `renders_llama_tools`).
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0 (`Finished dev profile`); `cargo test --workspace` → exit 0, 202 tests ok (after rebasing onto `3648a43`); `launcher.sh check` → `0 blocking`.
- Grammar checked against the real llguidance 1.8.0 (scratch crate, not committed; `ParserFactory::new_simple` over `toktrie_hf_tokenizers::ByteTokenizer` of the committed Llama tokenizer, forcing the tokens of each text): accepts `{"name": "get_weather", "parameters": {"location": "Paris"}}`, `<|python_tag|>` + two `;`-separated calls (parallel), a named `get_time` call; rejects a second call without parallel, a missing required `location`, `unit: "kelvin"`, an unknown name, compact `{"location":"Paris"}` (no free whitespace), the wrong function under a named choice, and leading prose.

