# P1-T22 Manual golden and baseline runs, AGENTS.md commands

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 22 (`.procoder/plans/phase-1-single-request.md`, "## Task 22"): Manual golden and baseline runs, AGENTS.md commands. Covers S-1 AC (`cargo build --workspace`, `cargo test --workspace`, clippy and fmt on macOS arm64 with no ROCm and no weights); S-11/S-13 AC manual `turbine-golden compare --url http://192.168.10.203:18000 …`; S-14 AC manual `turbine-bench … --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json`; S-14 AC manual real-stream check `turbine-bench … --concurrency 1 --requests 10 --output json` (moved from phase-0 S-6, decision 2026-09-25). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
