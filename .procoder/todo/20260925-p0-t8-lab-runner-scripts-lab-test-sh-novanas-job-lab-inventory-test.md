# P0-T8 Lab runner — scripts/lab-test.sh, novanas Job, lab inventory test

Status: open
Created: 2026-09-25

## Description

Phase 0 plan Task 8 (`.procoder/plans/phase-0-skeleton.md`, "## Task 8"): Lab runner — scripts/lab-test.sh, novanas Job, lab inventory test. Covers S-7, S-4 on real hardware; `lab inventory_matches_expectation` and the three `scripts/lab-test.sh` runs.. Done when every test step of the task passes, the gate is clean and the task's commit is on `phase-0-skeleton`.

## Acceptance criteria

- [ ] `cargo test -p turbine-device --test lab` before the file exists` passes (expect FAIL (`no test target named `lab``); after creating it, `cargo test -p turbine-device --test lab -- --include-ignored` — expect PASS on macOS (no expectations set).)
- [ ] `shellcheck scripts/lab-test.sh && bash -n scripts/lab-test.sh; scripts/lab-test.sh; echo exit=$?` passes (expect PASS (no findings) and `exit=2` with the usage line.)
- [ ] `ssh -o BatchMode=yes piwi@192.168.10.203 'kubectl apply --dry-run=client -o name -f -' < scripts/lab/novanas-test-job.yaml` passes (expect PASS (`job.batch/turbine-lab-test`, nothing created).)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-0-skeleton with the plan's commit message

## Evidence

<!-- Filled at close time. -->
