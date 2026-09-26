# P1-T10 Startup memory budget

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 10 (`.procoder/plans/phase-1-single-request.md`, "## Task 10"): Startup memory budget. Covers S-5 AC `budget::tests::refuses_before_loading`, `budget::tests::available_memory_by_kind`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model budget::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-model budget::tests` with the tests written and no implementation → `error[E0422]: cannot find struct, variant or union type BudgetTerms in this scope`, `error[E0425]: cannot find function available_bytes / check_budget / host_mem_available in this scope` (compile failure: budget API absent). First green run then failed on the plan's meminfo figure (`left: Some(129923002368)`, `right: Some(129922002368)`).
- Green: `cargo test -p turbine-model budget::tests` → `test result: ok. 2 passed; 0 failed` (`refuses_before_loading`: tiny index opened, `model.safetensors` truncated to its header, terms weights = `index.total_bytes()`, kv = `kv_layout(1).bytes_per_token() × 512`, workspace/reserve 1 MiB vs 1 KiB available → `ModelError::Budget("weights … B + kv_reservation … B + workspace 1048576 B + emergency_reserve 1048576 B = … B > available 1024 B")`; a real load of the truncated file still fails with `Io` naming it (budget read no weights); an exactly-fitting budget passes. `available_memory_by_kind`: dedicated 10 GiB free / host 5 GiB → 10 GiB; unified host 5 GiB → 5 GiB, host 20 GiB → 10 GiB, host unknown → 10 GiB; `host_mem_available` on a written meminfo with `MemAvailable: 126877932 kB` → 129923002368, missing line / missing file → `None`).
- Deviation: the plan states `MemAvailable: 126877932 kB → 129922002368`; 126877932 × 1024 = 129923002368. The test asserts the correct product and says so in a comment; the plan line needs the same correction (outside this task's files).
- `memory_budget` INFO event carries weights, kv_reservation, workspace, emergency_reserve, required, `available_bytes`, fits.
- Workspace: `cargo test --workspace` → 82 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → `0 blocking`.
- Commit: `feat(turbine-model): startup memory budget` on the worktree branch (branched from phase-1-single-request; lands there on merge).
