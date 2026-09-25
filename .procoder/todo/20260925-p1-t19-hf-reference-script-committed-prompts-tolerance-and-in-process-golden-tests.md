# P1-T19 HF reference script, committed prompts, tolerance and in-process golden tests

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 19 (`.procoder/plans/phase-1-single-request.md`, "## Task 19"): HF reference script, committed prompts, tolerance and in-process golden tests. Covers S-11 (reference generator, committed prompts and tolerance; the ignored tests `golden hf_reference_matches_cpu` and `golden logits_match_reference` are written here and accepted by their lab run in Task 21). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `uv run scripts/golden/hf_reference.py --model-dir /tmp/tiny --prompts tests/golden/prompts.jsonl --out /tmp/ref.jsonl --top-logprobs 5` after `cargo test -p turbine-model --test tiny_model` wrote `/tmp/tiny` passes (expect PASS (exit 0, 16 lines))
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
