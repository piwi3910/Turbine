# P2-T8 Kernel ABI v2 on the Rust side and CPU reference ops

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 8 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 8"): Kernel ABI v2 on the Rust side and CPU reference ops. Covers S-5 (paged attention ops, `copy_blocks`), S-16 (`moe_route`, `moe_experts` with CPU reference) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-kernels` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-kernels): kernel ABI v2 with paged attention, block copy and MoE ops`)

## Evidence

