# P0-T8 Lab runner — scripts/lab-test.sh, novanas Job, lab inventory test

Status: closed 2026-09-25
Created: 2026-09-25

## Description

Phase 0 plan Task 8 (`.procoder/plans/phase-0-skeleton.md`, "## Task 8"): Lab runner — scripts/lab-test.sh, novanas Job, lab inventory test. Covers S-7, S-4 on real hardware; `lab inventory_matches_expectation` and the three `scripts/lab-test.sh` runs.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [x] `cargo test -p turbine-device --test lab` before the file exists` passes (expect FAIL (`no test target named `lab``); after creating it, `cargo test -p turbine-device --test lab -- --include-ignored` — expect PASS on macOS (no expectations set).)
- [x] `shellcheck scripts/lab-test.sh && bash -n scripts/lab-test.sh; scripts/lab-test.sh; echo exit=$?` passes (expect PASS (no findings) and `exit=2` with the usage line.)
- [x] `ssh -o BatchMode=yes piwi@192.168.10.203 'kubectl apply --dry-run=client -o name -f -' < scripts/lab/novanas-test-job.yaml` passes (expect PASS (`job.batch/turbine-lab-test`, nothing created).)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-0-skeleton with the plan's commit message (committed on worktree branch `worktree-agent-a5d67b07b4b8a82ca`, branched from phase-0-skeleton; ticks when merged)
- [x] Lab novanas: `scripts/lab-test.sh novanas` reports 2 AMD devices, arch gfx1201, memory.kind dedicated, and `lab-test: novanas: PASS`

## Evidence

- `cargo test -p turbine-device --test lab` (before the file existed) → ``error: no test target named `lab` in `turbine-device` package``; after creating it, `cargo test -p turbine-device --test lab -- --include-ignored` → `test inventory_matches_expectation ... ok` / `test result: ok. 1 passed`.
- `shellcheck scripts/lab-test.sh && bash -n scripts/lab-test.sh` → no findings; `scripts/lab-test.sh; echo exit=$?` → `usage: scripts/lab-test.sh <dgx-spark|dgx-spark2|novanas>` / `exit=2` (same for an unknown host).
- `ssh -o BatchMode=yes piwi@192.168.10.203 'kubectl apply --dry-run=client -o name -f -' < scripts/lab/novanas-test-job.yaml` → `job.batch/turbine-lab-test`, exit 0.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exit 0; `cargo test --workspace` → 26 passed, 1 ignored (the lab test); `launcher.sh check` → `0 blocking`.
- Lab novanas pre-check (read-only, 2026-09-25): `kubectl describe node | grep amd.com/gpu` → `amd.com/gpu  0  0`; no pod had an amd.com/gpu limit; `amd-smi monitor -v` → both GPUs `57 MB` used of `32624 MB`.
- `scripts/lab-test.sh novanas` (first attempt; Job Complete 1/1 in 55 s; afterwards `amd.com/gpu 0 0`):
  - `lab-inventory: index=0 vendor=amd name="AMD Radeon AI PRO R9700" arch=gfx1201 memory.kind=dedicated total_bytes=34208743424`
  - `lab-inventory: index=1 vendor=amd name="AMD Radeon AI PRO R9700" arch=gfx1201 memory.kind=dedicated total_bytes=34208743424`
  - backends: nvidia `unavailable` (libnvidia-ml.so.1 not present), amd `ok` / `2 device(s)`
  - `test inventory_matches_expectation ... ok`; every other workspace test binary `ok` (27 passed on novanas)
  - `lab-test: novanas: PASS`, script exit 0
