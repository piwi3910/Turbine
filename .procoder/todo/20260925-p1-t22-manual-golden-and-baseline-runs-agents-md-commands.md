# P1-T22 Manual golden and baseline runs, AGENTS.md commands

Status: closed 2026-09-26
Created: 2026-09-25

## Description

Phase 1 plan Task 22 (`.procoder/plans/phase-1-single-request.md`, "## Task 22"): Manual golden and baseline runs, AGENTS.md commands. Covers S-1 AC (`cargo build --workspace`, `cargo test --workspace`, clippy and fmt on macOS arm64 with no ROCm and no weights); S-11/S-13 AC manual `turbine-golden compare --url http://192.168.10.203:18000 …`; S-14 AC manual `turbine-bench … --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json`; S-14 AC manual real-stream check `turbine-bench … --concurrency 1 --requests 10 --output json` (moved from phase-0 S-6, decision 2026-09-25). Done when every step of the task passes, the gate is clean and the task's commit is on `phase-1-single-request`.

## Acceptance criteria

- [x] Gate clean: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [x] Committed on branch phase-1-single-request with the plan's commit message
- [x] `scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml` logs `listening` and reports `/ready` 200
- [x] `turbine-golden compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` exits 0 (≥ 14/16 prompts, max |Δ logprob| ≤ 0.15 on likely / ≤ 0.55 on tail candidates — two-tier bound and FP32-logit reference, user decisions 2026-09-26) — PASS 16/16 (see Evidence, "Golden rerun")
- [x] Baseline `turbine-bench … --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json` exits 0 with `"requests_ok": 10`; JSON pasted as the Phase 1 single-request baseline
- [x] Real-stream check `turbine-bench … --concurrency 1 --requests 10 --output json` exits 0 with `"requests_ok": 10`, `ttft_ms.p50` > 0, `output_token_throughput` > 0
- [x] `scripts/lab-serve.sh novanas --stop` deletes the serve Job and nothing else

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

### Lab runs (2026-09-26, tree at 6d72957)

GPUs free before start: polled `turbine-ci` and all namespaces from 00:43 to 01:18 (+04); the T21
`turbine-lab-test` Job ran and failed repeatedly, then no pod requested `amd.com/gpu` for 16 min.

`scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml` → exit 0:

- `turbine_hip: libturbine_hip.so built for gfx1201, ROCm 7.14.1, hipBLASLt 1.4.1, CK cd9574023093742434e8c992d13b89ab9a6c1cf8`
- `kernel library loaded … abi_version=1 build_archs=gfx1201 device_arch="gfx1201"`; every op on provider `hip` (`hipblaslt`, `ck_tile_rmsnorm2d`, `ck_tile_fmha_fwd`, `turbine_hip`)
- `memory_budget weights=6425499648 kv_reservation=3758096384 workspace=2618027264 emergency_reserve=2147483648 required=14949106944 available_bytes=33917239296 fits=true`
- `listening; loading the model addr=0.0.0.0:18000 devices=1` → `model loaded and warmed up load_seconds=2.207633493` → `ready`
- `lab-serve: novanas: ready at http://192.168.10.203:18000 (stop with: scripts/lab-serve.sh novanas --stop)`

`cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl --tolerance tests/golden/llama-3.2-3b-instruct/tolerance.json` → **exit 1** (identical numbers on a second `--output json` run; same verdicts as the in-process `logits_match_reference` of T21 run 4):

```
FAIL p01 identical_prefix=30/32 first_divergence=30 margin=0.000 max_abs_logprob_diff=0.2068
PASS p02 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1157
FAIL p03 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.3940
FAIL p04 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1958
PASS p05 identical_prefix=17/32 first_divergence=17 margin=0.000 max_abs_logprob_diff=0.0947
PASS p06 identical_prefix=13/32 first_divergence=13 margin=0.000 max_abs_logprob_diff=0.1012
FAIL p07 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1565
FAIL p08 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1532
FAIL p09 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1826
FAIL p10 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.3484
FAIL p11 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.2999
FAIL p12 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.2630
FAIL p13 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.1897
FAIL p14 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.2681
FAIL p15 identical_prefix=18/32 first_divergence=18 margin=0.000 max_abs_logprob_diff=0.2169
FAIL p16 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff=0.4581
FAIL: 3/16 prompts passing (need 14); |Δ logprob| over top-5 ≤ 0.15 on every prompt: no
```

Diagnosis (per-position comparison of the served top-20 logprobs against `reference.jsonl`, script
kept outside the tree): greedy tokens match 32/32 on 12 prompts and every divergence (p01@30,
p05@17, p06@13, p15@18) is at a reference top-1/top-2 margin of 0.000 (a BF16 tie), so tokens are
within tolerance on all 16 prompts. Only the |Δ logprob| ≤ 0.15 bound fails, and the drift grows
with distance from the top-1. Max |Δ| per prompt, bucketed by the reference logprob of the top-5
entry. `pos` = positions covered: up to the first position where the reference top-1 diverges or a
reference top-5 id is absent from the served top-20 (p16 stops at 25, so its golden 0.4581 lies at
positions 25–31, which this table does not cover):

```
id  pos  lp>-2   -2..-8  lp<-8
p01 31  0.0932  0.2068  0.1257
p02 32  0.0631  0.1157  0.0889
p03 32  0.0793  0.1405  0.3940
p04 32  0.0618  0.1785  0.1958
p05 18  0.0473  0.0947  0.0930
p06 14  0.0556  0.1012  0.0851
p07 32  0.0838  0.1565  0.0000
p08 32  0.0876  0.1532  0.1209
p09 32  0.0998  0.1826  0.1222
p10 32  0.0939  0.2043  0.3484
p11 32  0.1089  0.1700  0.2999
p12 32  0.0480  0.2630  0.1668
p13 32  0.0753  0.1897  0.1061
p14 32  0.1383  0.1867  0.2681
p15 19  0.0680  0.2169  0.1974
p16 25  0.0118  0.2253  0.2323
```

Over the covered positions every top-5 entry with reference logprob > -2 is within 0.15 (max 0.1383) on all 16 prompts; the
worst cases are tail candidates 13–18 nats below a near-certain top-1 (p03 pos 15: 0.394 on a
14.75-nat gap, 2.7 % of the logit gap). The top-1-anchored difference equals the absolute one
(log-sum-exp offset ≤ 0.03), so the drift is in logit differences, not normalisation. The reference
logits are BF16-quantised (its logprob gaps are multiples of 0.0625/0.125) while Turbine computes
the LM head in F32. The pattern fits BF16 accumulation-order drift scaled by logit magnitude, not a
defect such as a wrong RoPE/softmax/GQA mapping (that would move the top-1 and the high-probability
candidates). Resolving it needs a decision outside Task 22's files (calibrating tolerance.json,
e.g. bounding |Δ| only for top-5 entries above a logprob floor or relative to the logit gap,
versus hunting the drift in the HIP path); Task 22 stays open.

Baseline: `cargo run --release -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18000 --concurrency 1 --requests 10 --max-tokens 128 --ignore-eos --output json` → exit 0 (the Phase 1 single-request baseline):

```json
{
  "requests_ok": 10,
  "requests_failed": 0,
  "wall_seconds": 45.125575333,
  "request_throughput": 0.22160382280349727,
  "output_token_throughput": 28.36528931884765,
  "ttft_ms": {
    "p50": 62.614666,
    "p95": 118.6185,
    "p99": 118.6185
  },
  "itl_ms": {
    "p50": 32.151332999999994,
    "p95": 48.375208,
    "p99": 95.062042
  },
  "e2e_ms": {
    "p50": 4248.534874999999,
    "p95": 6300.681458999999,
    "p99": 6300.681458999999
  }
}
```

Real-stream check (natural EOS): `cargo run --release -p turbine-bench --bin turbine-bench -- --url http://192.168.10.203:18000 --concurrency 1 --requests 10 --output json` → exit 0, `"requests_ok": 10`, `ttft_ms.p50` 60.85, `output_token_throughput` 29.31:

```json
{
  "requests_ok": 10,
  "requests_failed": 0,
  "wall_seconds": 43.087227583,
  "request_throughput": 0.2320873391247267,
  "output_token_throughput": 29.312630931452983,
  "ttft_ms": {
    "p50": 60.854459,
    "p95": 67.274,
    "p99": 67.274
  },
  "itl_ms": {
    "p50": 31.945584,
    "p95": 44.353333,
    "p99": 45.846542
  },
  "e2e_ms": {
    "p50": 4116.677125,
    "p95": 5407.366625,
    "p99": 5407.366625
  }
}
```

`scripts/lab-serve.sh novanas --stop` → exit 0: `job.batch "turbine-lab-serve" deleted from turbine-ci namespace`, `lab-serve: novanas: stopped`; `/ready` then answers `000` (connection refused). A `turbine-lab-test-0925212027-24a30632` Job that another agent started during the runs was left untouched.

### Golden rerun (2026-09-26, after the user decisions of 2026-09-26)

The diagnosis above led to three user decisions (`.procoder/ask/decisions.md`, end): the committed
reference is the FP32-final-logit transformers reference, the logprob bound is 0.15 for reference
top-5 candidates with reference logprob > −2 and 0.55 for the tail, and the cpu-reference attention
rounds its probabilities to BF16 like CK. Applied in `2793c23` and `7399e97` (spec, plan, contract
and AGENTS.md updated in the docs commit).

`scripts/lab-serve.sh novanas scripts/lab/phase1-novanas.yaml` → exit 0 (second attempt; the first
Job failed at `install cmake ninja python3 rsync` while a `lab-test.sh` Job was installing packages
at the same time): `ready`, `lab-serve: novanas: ready at http://192.168.10.203:18000`.

`cargo run --release -p turbine-bench --bin turbine-golden -- compare --url http://192.168.10.203:18000 --reference tests/golden/llama-3.2-3b-instruct/reference.jsonl` → **exit 0**:

```
PASS p01 identical_prefix=30/32 first_divergence=30 margin=0.009 max_abs_logprob_diff_likely=0.0537 max_abs_logprob_diff_tail=0.1232
PASS p02 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0339 max_abs_logprob_diff_tail=0.0728
PASS p03 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0322 max_abs_logprob_diff_tail=0.2359
PASS p04 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0871 max_abs_logprob_diff_tail=0.1725
PASS p05 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0599 max_abs_logprob_diff_tail=0.1197
PASS p06 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0705 max_abs_logprob_diff_tail=0.1403
PASS p07 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0555 max_abs_logprob_diff_tail=0.1105
PASS p08 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0620 max_abs_logprob_diff_tail=0.1493
PASS p09 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0430 max_abs_logprob_diff_tail=0.0988
PASS p10 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0750 max_abs_logprob_diff_tail=0.3394
PASS p11 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0407 max_abs_logprob_diff_tail=0.2563
PASS p12 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0368 max_abs_logprob_diff_tail=0.2256
PASS p13 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0643 max_abs_logprob_diff_tail=0.1454
PASS p14 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0801 max_abs_logprob_diff_tail=0.1632
PASS p15 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0393 max_abs_logprob_diff_tail=0.1416
PASS p16 identical_prefix=32/32 first_divergence=none margin=- max_abs_logprob_diff_likely=0.0646 max_abs_logprob_diff_tail=0.5133
PASS: 16/16 prompts passing (need 14); |Δ logprob| over top-5 ≤ 0.15 (reference logprob > -2) and ≤ 0.55 (tail) on every prompt: yes
```

The only divergence (p01 at 30) sits at a reference margin of 0.009 nats. Headroom: likely max
0.0871 (p04) against 0.15; tail max 0.5133 (p16) against 0.55 — the closest margin. Identical to the
in-process `golden logits_match_reference` verdicts in T21.

`scripts/lab-serve.sh novanas --stop` → `job.batch "turbine-lab-serve-0926034958-36ce491f" deleted`,
`job.batch "turbine-lab-serve-0926035115-021dd1f7" deleted`, `lab-serve: novanas: stopped`.
