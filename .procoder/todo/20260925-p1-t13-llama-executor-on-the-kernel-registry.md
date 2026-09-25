# P1-T13 Llama executor on the kernel registry

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 13 (`.procoder/plans/phase-1-single-request.md`, "## Task 13"): Llama executor on the kernel registry. Covers S-8, S-12 AC `tiny_model cpu_forward_matches_naive` (the ignored `tiny_model hip_matches_cpu` is written here; its acceptance run belongs to Task 21). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `cargo test -p turbine-model --test tiny_model cpu_forward_matches_naive && cargo test -p turbine-model rope::tests` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
