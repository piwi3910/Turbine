# P1-T8 Safetensors index with header validation

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 8 (`.procoder/plans/phase-1-single-request.md`, "## Task 8"): Safetensors index with header validation. Covers S-2 AC `safetensors::tests::rejects_malformed_headers`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-model safetensors::tests` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red: `cargo test -p turbine-model safetensors::tests::rejects_malformed_headers` before the implementation → `error[E0425]: cannot find type SafetensorsIndex in this scope` (plus `SINGLE_FILE`, `INDEX_FILE`, `HEADER_TENSOR`; compile failure: index API absent).
- Green: `cargo test -p turbine-model safetensors::tests` → `test result: ok. 2 passed; 0 failed` (`rejects_malformed_headers`: out-of-file range → `range outside file`, overlap → `overlapping ranges with a`, `[2,2]` BF16 over 6 bytes → `shape [2, 2] × BF16 size 8 B != byte range 6 B`, dtype `Q4` → `unknown dtype Q4`, 101 MiB length prefix in a 10-byte file → `header of 105906176 bytes exceeds 100 MiB`, duplicate key → `tensor listed twice`, index naming a missing shard → `shard listed in index is missing`, tensor in two shards → `tensor listed twice`, index entry absent from its shard; each asserted to name file, tensor and rule; a valid sharded checkpoint opens with absolute ranges; `single_file_and_directory_errors`: single file, missing/empty dir → `Io`, pickle-only dir → `Pickle` without opening, symlink to a directory → `not a regular file`, non-JSON header).
- Deviation: `crates/turbine-model/src/testing/mod.rs` (`TempDir`, a Task 9 file) lands here because these tests need it; Task 9 adds `testing::tiny`.
- Workspace: `cargo test --workspace` → 75 passed, 0 failed.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `launcher.sh check` → `0 blocking`.
- Commit: `feat(turbine-model): validated safetensors index` on the worktree branch (branched from phase-1-single-request; lands there on merge).
