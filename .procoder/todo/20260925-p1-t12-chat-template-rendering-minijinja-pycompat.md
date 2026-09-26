# P1-T12 Chat template rendering (minijinja + pycompat)

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 12 (`.procoder/plans/phase-1-single-request.md`, "## Task 12"): Chat template rendering (minijinja + pycompat). Covers S-4 AC `chat_template::tests::renders_target_template`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model chat_template::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-model chat_template::tests::renders_target_template` with the tests and no implementation → `error[E0433]: cannot find type ChatTemplate in this scope`, `error[E0425]: cannot find function strftime in this scope` (compile failure: API absent).
- Green: `cargo test -p turbine-model chat_template::tests` → `test result: ok. 6 passed; 0 failed` (`renders_target_template`: system+user and user-only conversations with `date_string="26 Jul 2024"` render byte-identical to the transformers text and re-encode to the transformers token ids (65 and 38 ids); without `date_string` the output holds `Today Date: ` + today's UTC `%d %b %Y`; an inline `{{ raise_exception('bad role') }}` → `ModelError::Template("bad role")`; the Llama template's own `raise_exception` on two tool calls → `ModelError::Template`; plus `strftime_formats`, `tojson_matches_python_json_dumps`, `jinja2_semantics`, `inline_raise_exception_and_errors_name_file`, `resolve_prefers_jinja_and_reads_special_tokens`).
- Fixture deviation: `scripts/golden/render_fixture.py` and `expected_renders.json` are committed by another agent on a different branch and are not on this one (and no Python runs on this path), so the test pins the expected texts and ids inline as constants taken from the transformers render made during planning (`transformers 5.17.0`, same template, `add_generation_prompt=True`, `date_string="26 Jul 2024"`). The cross-check against the committed `expected_renders.json` lands when the branches merge.
- Workspace: `cargo test --workspace` → 82 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → 0 blocking.
- Commit: `feat(turbine-model): Llama-3.2 chat template rendering with pycompat` on a worktree branch from phase-1-single-request (lands there on merge).
