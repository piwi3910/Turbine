# P2-T18 libturbine_hip.so ABI v2 — paged attention, block copy and MoE

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 18 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 18"): libturbine_hip.so ABI v2 — paged attention, block copy and MoE. Covers S-5, S-16 (HIP paged-KV and MoE providers); the ignored `hip_ops paged_and_moe_ops` is written here and accepted by its lab run in Task 20 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(kernels): hip ABI v2 paged attention, block copy and MoE ops`)

## Evidence

