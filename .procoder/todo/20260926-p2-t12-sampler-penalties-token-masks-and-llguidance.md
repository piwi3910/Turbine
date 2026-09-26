# P2-T12 Sampler penalties, token masks and llguidance structured output

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 12 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 12"): Sampler penalties, token masks and llguidance structured output. Covers S-17 AC `structured::tests::mask_applied_before_sampling`; S-10 (penalties, bias, min_tokens, stop_token_ids in sampling); S-6 (sampler state kept across preemption) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): sampler penalties, token masks and llguidance structured output`)

## Evidence

