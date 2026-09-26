# P2-T8 Kernel ABI v2 on the Rust side and CPU reference ops

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 8 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 8"): Kernel ABI v2 on the Rust side and CPU reference ops. Covers S-5 (paged attention ops, `copy_blocks`), S-16 (`moe_route`, `moe_experts` with CPU reference) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `cargo test -p turbine-kernels` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`feat(turbine-kernels): kernel ABI v2 with paged attention, block copy and MoE ops`)

## Evidence

- Built test-first on a run-ahead branch (runahead/p2-*, red run recorded there), cherry-picked onto `phase-2-serving-runtime` (conflicts in `turbine-model` lib.rs/tiny.rs resolved keeping both the P1 head_dim option and the OLMoE writer; follow-up fix commit c070004 adapts P1 literals, the paged CPU attention `round_p` argument and the concurrency golden tolerance fixture).
- `cargo test --workspace --no-fail-fast` on macOS arm64 → 191 passed, 0 failed, 10 ignored.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0.
