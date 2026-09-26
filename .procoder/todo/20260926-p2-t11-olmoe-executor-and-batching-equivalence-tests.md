# P2-T11 OLMoE executor and batching equivalence tests

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 11 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 11"): OLMoE executor and batching equivalence tests. Covers S-16 AC `tiny_model olmoe_cpu_forward_matches_naive`; S-4/S-9 AC `tiny_model chunked_prefill_matches_unchunked`; S-5/S-9 AC `tiny_model paged_matches_contiguous` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model --test tiny_model` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): OLMoE executor and batching equivalence tests`)

## Evidence

