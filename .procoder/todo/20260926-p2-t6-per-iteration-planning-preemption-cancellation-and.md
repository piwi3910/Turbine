# P2-T6 Per-iteration planning, preemption, cancellation and scheduler metrics

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 6 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 6"): Per-iteration planning, preemption, cancellation and scheduler metrics. Covers S-3, S-4, S-6, S-7 (scheduler side), S-8, S-11 (snapshot), S-14 (scheduler metrics and reason logs); the named sim ACs are delivered in Task 7 Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-scheduler` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-scheduler): continuous batching plan, preemption and metrics`)

## Evidence

- Built test-first on a run-ahead branch (runahead/p2-*, red run recorded there), cherry-picked onto `phase-2-serving-runtime` (conflicts in `turbine-model` lib.rs/tiny.rs resolved keeping both the P1 head_dim option and the OLMoE writer; follow-up fix commit c070004 adapts P1 literals, the paged CPU attention `round_p` argument and the concurrency golden tolerance fixture).
- `cargo test --workspace --no-fail-fast` on macOS arm64 → 191 passed, 0 failed, 10 ignored.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0.
