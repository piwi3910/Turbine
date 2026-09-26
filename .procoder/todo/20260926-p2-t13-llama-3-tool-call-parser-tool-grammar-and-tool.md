# P2-T13 Llama-3 tool-call parser, tool grammar and tool rendering

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 13 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 13"): Llama-3 tool-call parser, tool grammar and tool rendering. Covers S-18 AC `tools::tests::llama3_json_parser`, `chat_template::tests::renders_llama_tools` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model tools::tests chat_template::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): llama3_json tool-call parser and tool grammar`)

## Evidence

