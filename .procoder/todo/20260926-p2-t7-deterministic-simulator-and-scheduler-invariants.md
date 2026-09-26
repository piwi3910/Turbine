# P2-T7 Deterministic simulator and scheduler invariants

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 7 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 7"): Deterministic simulator and scheduler invariants. Covers S-3 AC `sim::tests::ts_section7_iteration_pattern`; S-3/S-12 AC `sim::tests::decode_never_starved`; S-4 AC `sim::tests::chunk_budget_respected`; S-6 AC `sim::tests::preemption_by_recompute`; S-8 AC `sim::tests::cancellation_frees_within_one_iteration`; S-12 AC `sim::tests::bounded_under_overload` Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-scheduler sim::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-scheduler): deterministic simulator and scheduling invariants`)

## Evidence

