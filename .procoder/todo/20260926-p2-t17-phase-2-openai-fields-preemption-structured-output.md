# P2-T17 Phase 2 OpenAI fields, preemption, structured output and tools in the engine

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 17 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 17"): Phase 2 OpenAI fields, preemption, structured output and tools in the engine. Covers S-10 AC `tiny_server openai_phase2_fields`; S-6/S-9 AC `tiny_server preempted_output_unchanged`; S-17 AC `tiny_server response_format_json_schema`; S-18 AC `tiny_server tool_choice_modes`; S-14 AC `tiny_server phase2_metrics_and_reasons` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-server` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): n, penalties, preemption state, structured output and tool calls`)

## Evidence

