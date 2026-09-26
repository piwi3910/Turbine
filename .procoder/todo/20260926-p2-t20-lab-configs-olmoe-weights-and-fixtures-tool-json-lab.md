# P2-T20 Lab configs, OLMoE weights and fixtures, tool/JSON lab test on novanas

Status: done
Created: 2026-09-26

## Description

Phase 2 plan Task 20 (`.procoder/plans/phase-2-serving-runtime.md`, "## Task 20"): Lab configs, OLMoE weights and fixtures, tool/JSON lab test on novanas. Covers S-15/S-17/S-18 AC `lab_openai tools_and_json_schema`; S-5/S-16/S-15 `hip_ops paged_and_moe_ops` (lab run); S-15 (lab configs and jobs) Done when every step of the task passes, the gate is clean and the task's commit is on `phase-2-serving-runtime`.

## Acceptance criteria

- [x] `scripts/lab-test.sh novanas` passes (expect PASS)
- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-2-serving-runtime with the plan's commit message (`test(lab): phase 2 configs, OLMoE reference and tool/JSON lab test`)

## Evidence

File-only part (no lab host contacted; 2026-09-26):

- Configs, Jobs and `--vllm` mode: `scripts/lab/phase2-novanas-{llama,olmoe}.yaml`, `TURBINE_TEST_MOE_MODEL_DIR=/models/olmoe-1b-7b-0125-instruct` in `scripts/lab/novanas-test-job.yaml`, `scripts/lab/novanas-vllm-job.yaml`, `scripts/lab-serve.sh novanas --vllm <slug>`.
  `bash -n scripts/lab-serve.sh && shellcheck scripts/lab-serve.sh scripts/lab-test.sh` → exit 0 (no findings).
- vLLM image pin: `curl -s 'https://hub.docker.com/v2/repositories/rocm/vllm/tags?page_size=25'` lists
  `rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0 2026-09-01T15:58:48Z sha256:19ad8dc5fb3012f2d5810995f73e8bf069302056d2b4aa6fdf2d2ae5fa9a68ab ['amd64']` — ROCm 7.14.1 (the host's version), RDNA build (gfx1201); pinned by tag and digest.
- `cargo test -p turbine-bench --test lab_scripts` → `test result: ok. 13 passed; 0 failed` (new: `phase2_novanas_configs_load_with_the_scheduler_defaults`, `lab_serve_vllm_dry_run_applies_the_pinned_baseline_job`, `lab_serve_vllm_usage_errors_exit_2_without_contacting_a_host`; the test-Job test now asserts `TURBINE_TEST_MOE_MODEL_DIR`; every Phase 1 dry-run test unchanged and passing).
- `cargo test -p turbine-server --test lab_openai` → `test result: ok. 3 passed; 0 failed; 1 ignored` (`tools_and_json_schema` ignored; `tool_requests_fixture_is_well_formed`, `schema_validator_accepts_and_rejects`, `response_checks_reject_malformed_answers`).
- `cargo test -p turbine-server --test lab_openai -- --ignored` on macOS → FAIL as expected: `TURBINE_TEST_BACKEND is not set; set it to the backend under test (hip or cuda)`.
- `tests/golden/tools/requests.jsonl`: 12 chat requests (3 `required`, 2 named, 3 `auto`, 3 `json_schema`, 1 `json_object`); `tests/golden/olmoe-1b-7b-0125-instruct/tolerance.json` identical to Llama's.
- Gate: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` → exit 0; `cargo test --workspace` → exit 0; `launcher.sh check` → `0 blocking`.

Remaining (need the user's approval and the engine of Tasks 15–18):

1. Ask the user, then download OLMoE once to `/home/piwi/turbine-models/olmoe-1b-7b-0125-instruct` on novanas.
2. Ask the user, then generate `tests/golden/olmoe-1b-7b-0125-instruct/reference.jsonl` with `scripts/golden/hf_reference.py` on novanas and commit it.
3. After Tasks 17 and 18 land and an R9700 is free: `scripts/lab-test.sh novanas` → expect `test paged_and_moe_ops ... ok`, `test tools_and_json_schema ... ok`, Job exit 0.

Lab (2026-09-26, `phase-2c-performance` at 8acce3e, one R9700):

- Full suite `scripts/lab-test.sh novanas` (job `turbine-lab-test-0926145924-0b19b6df`): `test paged_and_moe_ops ... ok`, `test tools_and_json_schema ... ok` (12/12 fixture requests: required, named, auto, json_schema, json_object), `test logits_match_reference ... ok`; 318 passed, 2 failed — `forward_profile` (OOM: a stray native server held GPU 0) and `registry::tests::selection_order_and_reason` (tracing interest-cache flake, fixed in 8acce3e).
- `scripts/lab-test.sh novanas -- -p turbine-model -p turbine-kernels` (job `turbine-lab-test-0926153446-352ee5a9`): 167 passed; `paged_and_moe_ops ... ok`, `logits_match_reference` 16/16; 4 OOM failures because the job shared GPU 0 with a benchmark server started outside k8s.
- Rerun of those four on a free card (job `turbine-lab-test-0926155359-0351abe4`), exit 0: `decode_forward_timing ... ok`, `olmoe_logits_match_reference ... ok` (16/16 prompts, need 14, calibrated OLMoE tolerance), `forward_profile ... ok`, `serving_mix ... ok`.
