# P1-T19 HF reference script, committed prompts, tolerance and in-process golden tests

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 19 (`.procoder/plans/phase-1-single-request.md`, "## Task 19"): HF reference script, committed prompts, tolerance and in-process golden tests. Covers S-11 (reference generator, committed prompts and tolerance; the ignored tests `golden hf_reference_matches_cpu` and `golden logits_match_reference` are written here and accepted by their lab run in Task 21). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `uv run scripts/golden/hf_reference.py --model-dir /tmp/tiny --prompts tests/golden/prompts.jsonl --out /tmp/ref.jsonl --top-logprobs 5` after `cargo test -p turbine-model --test tiny_model` wrote `/tmp/tiny` passes (expect PASS (exit 0, 16 lines))
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

Progress (first half: script, prompts, tolerance, fixtures; the Rust golden tests `hf_reference_matches_cpu` / `logits_match_reference` are not written yet — they need Tasks 13/14 — so the tiny-checkpoint and commit criteria stay open):

- `scripts/golden/hf_reference.py` (PEP 723: torch==2.9.0, transformers==4.57.1, safetensors==0.6.2, jinja2==3.1.6 — transformers 4.57.1 does not pull jinja2, which `apply_chat_template` needs) and `scripts/golden/render_fixture.py` (transformers==4.57.1, jinja2==3.1.6; fixture-only, decision "P1: allow a second fixture-only Python script").
- `tests/golden/prompts.jsonl` (16 prompts; p09 = 1989 tokens with the Llama-3.2 tokenizer), `tests/golden/llama-3.2-3b-instruct/tolerance.json` (plan values).
- Tiny-checkpoint dry run (prototype checkpoint from planning, `tokenizer_class` set to `PreTrainedTokenizerFast` because 4.57.1 does not know `TokenizersBackend`): `uv run scripts/golden/hf_reference.py --model-dir <tiny> --prompts tests/golden/prompts.jsonl --out <tmp> --top-logprobs 5` → exit 0, `16` lines; tokens and prompt ids identical to the planning-time transformers 5.17.0 run.
- Reference on novanas CPU (as piwi, in /home/piwi/turbine-ci/golden-work, HF_HUB_OFFLINE=1): `uv run scripts/golden/hf_reference.py --model-dir /home/piwi/turbine-models/llama-3.2-3b-instruct --model-name unsloth/Llama-3.2-3B-Instruct --prompts tests/golden/prompts.jsonl --out tests/golden/llama-3.2-3b-instruct/reference.jsonl --top-logprobs 20 --device cpu` → `engine transformers-4.57.1-bf16-cpu, model unsloth/Llama-3.2-3B-Instruct, revision 006f5dcd1393c3add266de40994ba96225e9689d`, `p09: 1989 prompt tokens, 32 generated`, `EXIT 0`; 16 lines, 289439 bytes; top-1 equals the chosen token at all 512 positions; a second run produced identical tokens and logprobs (differences only in which id fills an exact tie at rank 20 before the tie-ordering fix).
- Render fixture on novanas: `uv run scripts/golden/render_fixture.py /home/piwi/turbine-models/llama-3.2-3b-instruct --out tests/golden/llama-3.2-3b-instruct/expected_renders.json` → `system_user: 65 tokens`, `user_only: 38 tokens`, exit 0; transformers_version 4.57.1. Task 12 expects it at `crates/turbine-model/tests/fixtures/llama-3.2-3b-instruct/expected_renders.json` (copy there when that crate lands).
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` 69 passed, 0 failed; `launcher.sh check` 0 blocking.
