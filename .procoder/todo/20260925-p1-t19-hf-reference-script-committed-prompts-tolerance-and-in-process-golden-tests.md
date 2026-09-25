# P1-T19 HF reference script, committed prompts, tolerance and in-process golden tests

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 19 (`.procoder/plans/phase-1-single-request.md`, "## Task 19"): HF reference script, committed prompts, tolerance and in-process golden tests. Covers S-11 (reference generator, committed prompts and tolerance; the ignored tests `golden hf_reference_matches_cpu` and `golden logits_match_reference` are written here and accepted by their lab run in Task 21). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `uv run scripts/golden/hf_reference.py --model-dir /tmp/tiny --prompts tests/golden/prompts.jsonl --out /tmp/ref.jsonl --top-logprobs 5` after `cargo test -p turbine-model --test tiny_model` wrote `/tmp/tiny` passes (expect PASS (exit 0, 16 lines))
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

Second half (in-process golden tests, `crates/turbine-model/tests/golden.rs`, on top of Task 14 `f21fe79`):

- Tests: `compare_prompt_applies_the_tolerance_rule` (not ignored; the `turbine-golden compare` rule re-implemented in the test file, no `turbine-bench` dependency), `hf_reference_matches_cpu` (ignored, needs `uv`), `logits_match_reference` (ignored, `require_backend("hip")`, `require_env_dir("TURBINE_TEST_MODEL_DIR")`, HIP provider, committed `reference.jsonl` and `tolerance.json`). Replay goes through `turbine_model::generate::generate` (temperature 0, `ignore_eos`, `logprobs: 20`); prompt ids are built like the server (chat template then `encode(.., false)`; completion `encode(.., true)`) and asserted equal to `prompt_token_ids`.
- Tiny-checkpoint criterion: the test itself writes the tiny checkpoint (seed 7) and runs `uv run scripts/golden/hf_reference.py --model-dir <tiny> --prompts tests/golden/prompts.jsonl --out <tmp>` (`tiny_model` does not write `/tmp/tiny`; this is the equivalent run). Verified from the Mac with `uv` on PATH replaced by a scratch shim that copies the checkpoint to novanas (piwi, /home/piwi/turbine-ci/golden-work, CPU, HF_HUB_OFFLINE=1) and runs the committed script there: `cargo test -p turbine-model --test golden -- --include-ignored hf_reference_matches_cpu compare_prompt` → script log `engine transformers-4.57.1-bf16-cpu`, `p01`…`p16` each `32 generated` (p09: 9145 prompt tokens), exit 0; `test hf_reference_matches_cpu ... ok`, `test compare_prompt_applies_the_tolerance_rule ... ok`, `2 passed`. Every prompt: prompt ids identical, identical_prefix 32/32.
- Deviation: the tiny test applies the committed tolerance with `max_abs_logprob_diff` 0.3 instead of 0.15 (`TINY_MAX_ABS_LOGPROB_DIFF`): the random tiny weights give logits of magnitude 16–32 where HF's BF16 logits are quantised at 0.125; measured max |Δ logprob| over the top 5 was 0.1226–0.2095 per prompt (0.15 failed 12/16 prompts on the logprob bound alone, with all 512 tokens identical). The Llama-3.2 test uses `tolerance.json` unchanged.
- `logits_match_reference` is accepted by its lab run in Task 21 (not run here).
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` 137 passed, 0 failed; `launcher.sh check` → `0 blocking`.
