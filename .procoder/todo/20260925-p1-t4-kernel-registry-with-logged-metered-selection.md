# P1-T4 Kernel registry with logged, metered selection

Status: done
Created: 2026-09-25

## Description

Phase 1 plan Task 4 (`.procoder/plans/phase-1-single-request.md`, "## Task 4"): Kernel registry with logged, metered selection. Covers S-6 AC `registry::tests::selection_order_and_reason`, `registry::tests::no_provider_is_startup_error`; S-14 (kernel selection log + gauge). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-kernels registry::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: with only the test module in `registry.rs`, `cargo test -p turbine-kernels registry::tests` failed to compile: `error[E0433]: cannot find type KernelRegistry in this scope` (also `KernelMetrics`, `OpConfig`, `OpRequirement`, `Selection`).
- Green: `cargo test -p turbine-kernels registry::tests` → `test registry::tests::selection_order_and_reason ... ok`, `test registry::tests::no_provider_is_startup_error ... ok` (plus `duplicate_requirements_select_once`, `accessor_panics_on_unselected_config`) → `test result: ok. 4 passed; 0 failed`. The selection test asserts the JSON log carries `"event":"kernel_selected"`, `"op":"attention_prefill"`, the rendered config, `"provider":"second"`, `"impl":"second_fmha"`, `"reason":"first provider in order supporting config; unsupported by: first"`, and `/metrics` contains `turbine_kernel_provider_selected{op="attention_prefill",provider="second",impl="second_fmha"} 1`; the error test asserts `no kernel provider supports attention_prefill head_dim=128 kv_heads=8 dtype=bf16 q_heads=24 causal=1`.
- Mutation: replacing the attention `supports()` filter with `true` made `selection_order_and_reason` and `no_provider_is_startup_error` FAIL (`test result: FAILED. 2 passed; 2 failed`); reverted.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` 69 passed, 0 failed; `launcher.sh check` → `procoder gate: 2 clean, 0 unformatted, 0 unchecked, 0 out of scope, 15 hygiene finding(s) (0 blocking)`.
- Commit `feat(turbine-kernels): kernel registry with selection log and gauge` on the Task 4 worktree branch `worktree-agent-a497152ada1f938d3` (branched from `phase-1-single-request`; the coordinator merges it).
