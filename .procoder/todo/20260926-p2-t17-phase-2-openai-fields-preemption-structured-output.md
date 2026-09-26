# P2-T17 Phase 2 OpenAI fields, preemption, structured output and tools in the engine

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 17 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 17"): Phase 2 OpenAI fields, preemption, structured output and tools in the engine. Covers S-10 AC `tiny_server openai_phase2_fields`; S-6/S-9 AC `tiny_server preempted_output_unchanged`; S-17 AC `tiny_server response_format_json_schema`; S-18 AC `tiny_server tool_choice_modes`; S-14 AC `tiny_server phase2_metrics_and_reasons` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-server` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): n, penalties, preemption state, structured output and tool calls`)

## Evidence

- Failing first (tests written against the Task 15 engine, before the Task 17 code):
  `cargo test -p turbine-server --test tiny_server -- openai_phase2_fields preempted_output_unchanged response_format_json_schema tool_choice_modes phase2_metrics_and_reasons`
  → `openai_phase2_fields ... FAILED` (`"unsupported parameter: n"`), `preempted_output_unchanged ... FAILED`
  (`"unsupported parameter: response_format"`), `tool_choice_modes ... FAILED` (`"unsupported parameter: tools"`),
  `response_format_json_schema ... FAILED`, `phase2_metrics_and_reasons ... FAILED`
- After (rebased on Task 16): `cargo test -p turbine-server` → `test result: ok. 23 passed` (unit: grammar service,
  requests, engine loop incl. `n_choices_fork_after_one_prefill`, `tool_call_parser_resolution`), `test result: ok. 15 passed` (tiny_server:
  `openai_phase2_fields`, `preempted_output_unchanged`, `response_format_json_schema`, `tool_choice_modes`,
  `phase2_metrics_and_reasons` and the rewritten `request_validation` ok), `server_cli` ok
- `cargo test -p turbine-model --lib tools::` → `test result: ok. 3 passed` (tool grammar errors are `ModelError::Constraint`)
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0
- `cargo test --workspace` → every target ok. Two of five full runs (host load average about 20 from parallel agent
  builds) saw the Task 15/16 timing tests `queue_full_429`, `disconnect_releases_kv`, `slow_client_paused_then_cancelled`
  time out; each passes alone and the other full runs were clean. The five Task 17 tests take 2.5–4.4 s each.
- `launcher.sh check` → `procoder gate: 9 clean, 0 unformatted, 0 unchecked, 2 out of scope, 42 hygiene finding(s) (0 blocking)` (after the rebase: `11 clean … (0 blocking)`)

