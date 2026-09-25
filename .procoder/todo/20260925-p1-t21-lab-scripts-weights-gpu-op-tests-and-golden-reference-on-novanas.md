# P1-T21 Lab scripts, weights, GPU op tests and golden reference on novanas

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 21 (`.procoder/plans/phase-1-single-request.md`, "## Task 21"): Lab scripts, weights, GPU op tests and golden reference on novanas. Covers S-7/S-13 AC `scripts/lab-test.sh novanas` with `hip_ops`; S-8/S-12/S-13 `tiny_model hip_matches_cpu`; S-11 `golden hf_reference_matches_cpu`; S-11/S-13 `golden logits_match_reference`; S-13 (lab execution, weights). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [ ] `scripts/lab-test.sh novanas` passes (expect PASS with log lines `libturbine_hip.so` built for `gfx1201`, `hipBLASLt 1.4.1`, `CK cd9574023093742434e8c992d13b89ab9a6c1cf8`, `test gemm_matches_cpu ... ok`, `test attention_matches_cpu ... ok`, `test hip_matches_cpu ... ok`, `test hf_reference_matches_cpu ... ok`, `test logits_match_reference ... ok`, Job exit 0)
- [ ] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

<!-- Filled at close time. -->
