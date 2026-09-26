# P2-T9 OLMoE config, weight slots and tiny OLMoE checkpoint

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 9 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 9"): OLMoE config, weight slots and tiny OLMoE checkpoint. Covers S-16 AC `config::tests::parses_olmoe_config` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model config::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-model): OLMoE config, weight slots and tiny checkpoint`)

## Evidence

