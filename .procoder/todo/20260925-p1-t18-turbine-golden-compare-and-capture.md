# P1-T18 turbine-golden compare and capture

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 18 (`.procoder/plans/phase-1-single-request.md`, "## Task 18"): turbine-golden compare and capture. Covers S-11 AC `golden capture_and_compare_roundtrip`. Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] `cargo test -p turbine-bench` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

- Red first: `cargo test -p turbine-bench --test golden capture_and_compare_roundtrip` with the binary still an empty `main` → `assertion \`left == right\` failed: capture failed: left: Some(2) right: Some(0)`.
- `cargo test -p turbine-bench` → `test capture_and_compare_roundtrip ... ok`; lib unit tests `test result: ok. 13 passed; 0 failed`; `tests/bench.rs` `test result: ok. 2 passed; 0 failed`; `tests/golden.rs` `test result: ok. 1 passed; 0 failed`.
- The roundtrip test covers: capture into a temp file (engine from `system_fingerprint`, both logprob shapes, RFC 3339 `captured`, no `.tmp` left); compare of the same mock exits 0; flip at position 5 with reference margin 2 nats exits 1 with `first_divergence` 5 and `margin_at_divergence` 2.0 (JSON) and `first_divergence=5 margin=2.000` (text); flip where the margin is 0.3 exits 0; one top-5 logprob shifted by 0.2 exits 1; capture against a 500-ing mock exits 1 leaving no file; usage errors exit 2.
- `cargo fmt --all --check` → exit 0; `cargo clippy --workspace --all-targets -- -D warnings` → `Finished \`dev\` profile` with no warnings; `cargo test --workspace` → 36 passed, 0 failed.
- `launcher.sh check` → `procoder gate: 7 clean, 0 unformatted, 0 unchecked, 2 out of scope, 25 hygiene finding(s) (0 blocking)`.
- Commit is on the worktree branch `worktree-agent-ae4787ba405c9af22` (branched from `phase-1-single-request`); the third criterion closes when the coordinator lands it on `phase-1-single-request`.
