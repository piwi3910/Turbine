# P1-T22 Manual golden and baseline runs, AGENTS.md commands

Status: open
Created: 2026-09-25

## Description

Phase 1 plan Task 22 (`.procoder/plans/phase-1-single-request.md`, "## Task 22"): Manual golden and baseline runs, AGENTS.md commands. Covers S-1 AC (`cargo build --workspace`, `cargo test --workspace`, clippy and fmt on macOS arm64 with no ROCm and no weights); S-11/S-13 AC manual `turbine-golden compare --url http://192.168.10.203:18000 …`; S-14 AC manual `turbine-bench … --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json`; S-14 AC manual real-stream check `turbine-bench … --concurrency 1 --requests 10 --output json` (moved from phase-0 S-6, decision 2026-09-25). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message

## Evidence

Partial (documentation/config part, 2026-09-26). Task stays **open**: the manual lab runs need
Task 17 (turbine-server with a model), which has not landed, so none of them was run.

Done and verified on the macOS arm64 workstation (no ROCm, no `TURBINE_TEST_MODEL_DIR`,
`CARGO_INCREMENTAL=0`):

- AGENTS.md Commands: HIP kernel build, GPU/weights test env vars, server with a model,
  `turbine-bench --bin` fix (the package has two binaries; `cargo run -p turbine-bench -- --help`
  failed with "could not determine which binary to run"), `turbine-golden compare|capture`,
  `hf_reference.py` / `render_fixture.py`, `scripts/lab-test.sh novanas`,
  `scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml|--stop|--dry-run`, the Phase 1
  golden, baseline and real-stream runs at `--concurrency 1` against
  http://192.168.10.203:18000, weights provenance (unsloth/Llama-3.2-3B-Instruct @
  006f5dcd1393c3add266de40994ba96225e9689d in /home/piwi/turbine-models/llama-3.2-3b-instruct;
  HF token stays on novanas), ask-first rule plus the 2026-09-25 standing approval.
- `examples/turbine.yaml`: `execution` block with the contract defaults (`backend: hip`,
  `device: 0`, `kernel_library: null`).
- `cargo build --workspace` → `Finished dev profile`, exit 0
- `cargo test --workspace` → exit 0, 121 passed, 0 failed, 2 ignored (lab-only)
- `cargo clippy --workspace --all-targets -- -D warnings` → exit 0
- `cargo fmt --all --check` → exit 0
- `! cargo tree --workspace | grep -Ei 'hip|rocm|cuda'` → exit 0 (no match)
- `cargo test -p turbine-core` → `6 passed`; `cargo test -p turbine-core config::tests::byte_size_parsing` → `1 passed`; `cargo test -p turbine-api --test api route_table_phase0` → `1 passed`
- `cargo run -p turbine-server -- --config examples/turbine.yaml --check-config` → `config ok`, exit 0; with `--set execution.backend=cuda` → exit 2 naming `phase-2b-nvidia`
- `cargo run -p turbine-bench --bin turbine-bench -- --help` → exit 0; `turbine-golden compare --help` / `capture --help` → exit 0
- `scripts/lab-serve.sh --dry-run novanas scripts/lab/phase1-novanas.yaml` → `lab-serve: novanas: dry run: nothing contacted`, exit 0; `--dry-run novanas --stop` → exit 0
- `launcher.sh agents --host claude` → `every agent rule file matches AGENTS.md`; `launcher.sh check` → `0 blocking`

Still open (plan Task 22 steps 2–6, lab and model required): `scripts/lab-serve.sh novanas
scripts/lab/phase1-novanas.yaml` to `/ready` 200; `turbine-golden compare` (≥ 14/16, max
|Δ logprob| ≤ 0.15); the `--max-tokens 128 --ignore-eos` baseline JSON; the real-stream check
JSON; `--stop`; and running the with-model server command as written. Commit lands on the
worktree branch for merge into phase-1-single-request.
