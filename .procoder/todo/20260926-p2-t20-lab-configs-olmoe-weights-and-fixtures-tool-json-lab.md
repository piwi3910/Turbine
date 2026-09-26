# P2-T20 Lab configs, OLMoE weights and fixtures, tool/JSON lab test on novanas

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 20 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 20"): Lab configs, OLMoE weights and fixtures, tool/JSON lab test on novanas. Covers S-15/S-17/S-18 AC `lab_openai tools_and_json_schema`; S-5/S-16/S-15 `hip_ops paged_and_moe_ops` (lab run); S-15 (lab configs and jobs) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`test(lab): phase 2 configs, OLMoE reference and tool/JSON lab test`)

## Evidence

