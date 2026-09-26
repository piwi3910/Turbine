# P2-T15 Engine thread with continuous batching, bounded channels and diagnostics

Status: open
Created: 2026-09-26

## Description

Phase 2 plan Task 15 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 15"): Engine thread with continuous batching, bounded channels and diagnostics. Covers S-1 (engine thread in `turbine-server`), S-7 AC `tiny_server queue_full_429`; S-8 AC `tiny_server disconnect_releases_kv`; S-11 AC `tiny_server diagnostics_shapes`; S-3 (continuous batching end to end) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [ ] `cargo test -p turbine-server` passes (expect PASS)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-server): engine thread with continuous batching and paged KV`)

## Evidence

