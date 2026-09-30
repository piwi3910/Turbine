# Handoff: p6a-fp8-t15 (plan Task 15, `fp8_block` kernel + FP8 layout in the loader)

Written 2026-09-29 by the Task 15 builder. Branch `p6a-fp8-t15` (from `p6a-fp8` 7174d4c), worktree
`.claude/worktrees/agent-a4baec4689b995379`. Not pushed. The proof (fixtures, bench, eval, soak) is the
next agent's.

## Done (committed)

- ca0bad2 `feat(rocm)`: own kernel `turbine_hip_fp8_block` (W8A16; fused WMMA for decode m <= 64,
  dequantize-to-BF16 + tuned hipBLASLt GEMM above / in prefill), plus the `qgemm_eval --block 1` timing.
- Handoffs, one per shared file: db9aea0 impl_table.cpp, 9cf6b63 turbine_hip.hpp, f8f3bbd CMakeLists.txt,
  82a6cd6 cards/gfx1201.rs, c3493ad hip_qgemm.rs, 6f99811 weights/mod.rs, 983c670 weights/fp8.rs
  (+ ct_fp8/hf_fp8 constructors, docs/extending/weight-format.md, the mod.rs test expectations),
  43f4c8d turbine-server model.rs, 7b6c977 tiny_model.rs.
- Test judgement: the old builder's widened bound (2e-2 + |c|/128) is gone. The dequantize path is
  judged against the CPU BF16 gemm of `bf16(e4m3(q)·s)` (what it multiplies), the fused path against
  the CPU qgemm (exact q·s), both with the Phase 1 BF16 tolerance. All 80 cases pass
  (max |Δ| 1.56e-2 = one BF16 ulp at |c| ≥ 2).
- Loader: `fp8_block` keeps e4m3 + F32 block scales; `WeightFormat::for_kernels` /
  `weights::resolve_for_providers` (called in `prepare_with` before the requirements) decodes to BF16
  only the stacks no selected provider runs (`kernel_unsupported`) or whose TP shard cuts a block
  (`shard_misaligned`), logged `event="fp8_block_decoded"`. Activations stay BF16 for block
  (W8A16, the user's option B); the checkpoint's group-128 activation scheme is not applied, so its
  golden reference must be made with `--act-quant none` (the dequant path multiplies
  bf16(q·s) exactly as `dequantize_checkpoint.py` would; the fused decode path uses exact q·s).

## Results

- `scripts/gate.sh`: ok, 774 passed (crates all).
- `scripts/lab-test.sh novanas --tier quick`: PASS (job turbine-lab-test-0929043937-1be4f7b8),
  including `qgemm_fp8_block_matches_cpu`, `qgemm_fp8_block_rows_do_not_depend_on_m`,
  `fp8_block_decode_fallback_per_stack`, `tp2_quantized_matches_tp1_on_host`.
- Weight bytes, computed from the checkpoint headers (the FP8 layout the loader uploads: e4m3
  linears, F32 block scales, BF16 embedding/norms, tied head skipped): **3,607,615,488 B** for
  `llama-3.2-3b-instruct-fp8-block` vs **6,425,499,648 B** BF16 = 0.56× (the 788 MB BF16 embedding
  keeps it above 0.5; the linear layers are exactly half).

## Pending: the served number (detached, queued on port18000 + bench.lock)

- Launched `scripts/bench-lock.sh --name port18000 scripts/bench-lock.sh scratchpad/t15_serve.sh`
  (nohup). It serves `scratchpad/t15-fp8-block.yaml` (copy of the old worktree's
  `phase6-novanas-llama-fp8-block.yaml`) with `lab-serve.sh`, greps the log, and always stops its own
  serve Job.
- Watch: `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad/t15_weight_bytes.txt`
  (last line `t15-serve: done rc=<rc> run=<id>`); full log `…/scratchpad/t15_serve.txt`.
- Judge: `rc=0`; the `weight_format` event's `weight_bytes` should be 3,607,615,488 (± nothing — it is
  exact), `layers=bf16:1,fp8_block_128x128:196` (or similar: every decoder linear FP8, lm_head BF16),
  `activation=none`, and **no** `fp8_block_decoded` line. A `fp8_block_decoded` line or ~6.4 GB means the
  provider refused a 3B shape — a bug to fix before the proof.
- If rc≠0, read `t15_serve.txt`; make sure `scripts/lab-serve.sh novanas --stop <run>` ran.

## Left for the proof agent

Fixture (`--act-quant none`), golden c1/c16, `lab-bench --model llama-fp8-block`, eval, soak, row flip,
decisions/perf-log entries — plan Task 15 steps 4–6. The untracked golden/config files are in the old
worktree `agent-adb021480bf19bfbe`.

## Proof agent (2026-09-29, lead rotation 5)

Merged `phase-6a-quantization` (a0ba309). Added `scripts/lab/phase6-novanas-llama-fp8-block.yaml` (the
lab-bench config for `--model llama-fp8-block`; copy of the phase2c Llama config). The Mac-side
`t15_serve.sh` (killed in the ssh outage) is replaced by one novanas-side detached driver that does
the served-bytes check, the GSM8K-200 eval and the bench in one serve, then the vLLM baseline:

- Script: `novanas:/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/t15p/t15_proof_run.sh`
  (Mac copy: session scratchpad `scratchpad/t15p/t15_proof_run.sh`), started 2026-09-29 09:24Z, pid 112710.
  Builds `kbuild/libturbine_hip.so` (nice 19, cores 12-15), then pass A (Turbine, native, GPU 0) and
  pass B (vLLM k3s Job, port 18100), each under port18000.lock → bench.gate → bench.lock, fixtures
  paused. Queued behind the full-GSM8K FP8 KV requeue.
- Log: `…/t15p/t15_proof_run.log`, last line `t15-proof: done rc=<rc> turbine=<rc> vllm=<rc>`, with
  two `SUMMARY` lines (tok/s, TTFT/ITL p50, c1 ITL p50, GSM8K). Outputs in `…/t15p/`:
  `load_events.txt`, `server.log`, `status.json`, `turbine-quality.json`, `eval-compare.txt`,
  `turbine-bench{,-c1}.json`, `vllm-bench{,-c1}.json`, `vllm-quality.json`, `vllm-pod.log`.

How to judge:

1. `load_events.txt`: the `weight_format` event's `weight_bytes` = **3,607,615,488** exactly, every
   decoder linear FP8 (`fp8_block_128x128`), lm_head/embedding BF16, `activation=none`, and **no**
   `fp8_block_decoded` line. ~6.4 GB or a decoded line = the kernel refused a 3B shape: a bug in the
   lead-owned loader/registry → `handoff(<file>)` commit before the proof counts.
2. `eval-compare.txt`: exit 0 against `tests/eval/llama-3.2-3b-instruct/turbine-bf16.json` (0.805)
   with max drop 0.02 (Q9). Copy `turbine-quality.json` to
   `tests/eval/llama-3.2-3b-instruct-fp8-block/turbine.json` and `vllm-quality.json` to `vllm.json`.
3. Bench: c16 tok/s ≥ 1.0 × BF16 (854.7, perf-log phase-start baseline); Turbine ≥ 0.9 × vLLM where
   vLLM serves it. vLLM ran as a k3s Job (the GPU is k3s's pick, maybe GPU 1: its numbers are
   indicative only if the pod landed on GPU 1). No BF16 c1 in this run.
4. Still open after it: the golden fixture (`--act-quant none`, rank 4 in
   `/home/piwi/turbine-ci/fixture.queue`), then `lab-bench --model llama-fp8-block --golden16 --c1`
   (Mac-driven, background), tolerance from the self-spread, labbook, soak (ask the lead), row flip
   as a `handoff(support.rs)` commit, perf-log row.

## Verdict (2026-09-29, proof collector, lead rotation 7)

`t15_proof_run.log`: `t15-proof: done rc=0 turbine=0 vllm=0`. Collected the pass directory by scp
from `novanas:/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/t15p/` into this session's
scratchpad (`t15p_collected/t15p/`). All four pass criteria hold:

| Criterion | Required | Observed | Verdict |
|---|---|---|---|
| `weight_bytes` | 3,607,615,488 | 3,607,615,488 (`load_events.txt` `weight_format` event) | PASS (exact) |
| `fp8_block_decoded` | absent | absent (grepped `load_events.txt`) | PASS |
| c16 tok/s | ≥ 854.7 | 1061.44 | PASS (1.24× BF16 floor, 1.57× this run's own vLLM pass) |
| GSM8K drop vs pair | ≤ 0.02 | baseline (BF16 turbine) 0.805 → candidate (fp8-block turbine) 0.800, drop 0.005 (`eval-compare.txt`: PASS) | PASS |

Other load-event facts: `layers=fp8_block:196`, `activation=none`, `packaging=ct_fp8`, KV `bf16`.
Support row logged at startup: `amd/gfx1201/LlamaForCausalLM/fp8_block/bf16/none` = `experimental`
(WARN, as expected pre-flip).

**BENCH-equivalent line** (not a `lab-bench.sh` run — the driver is `t15_proof_run.sh` — but the
same fields):

```
BENCH t15-proof llama-fp8-block commit=d76d123 gpu=0 tests=skip golden1=SKIP golden16=SKIP \
  tok_s=1061.44 tok_s_c1=114.68 itl_p50_ms=11.46 itl_c1_p50_ms=8.42 ttft_p50_ms=227.24 \
  gsm8k=0.800 (bf16 ref 0.805, drop 0.005) \
  vllm_tok_s=677.79 vllm_tok_s_c1=75.94 vllm_gsm8k=0.805 (vLLM GPU picked by k3s, may not be GPU 0) \
  verdict=PASS
```

**Labbook**: test type `turbine-lab-bench`, set `phase-6a-quantization`.
- Turbine run `2e31910f-20a4-4706-8ae7-e89b5f582a61` (`t15-proof:llama-fp8-block:turbine:d76d123`),
  model `llama-3.2-3b-instruct-fp8-block`, status `pass`.
- vLLM-ROCm run `a86153ab-ff32-49a9-82c3-64fe64434f8c`
  (`t15-proof:llama-fp8-block:vllm-rocm:d76d123`), same model key, status `pass`; added as the
  `vllm-rocm-0-23-gpu0` baseline's member for `model=llama-3.2-3b-instruct-fp8-block` (matching the
  AWQ precedent). Turbine vs this baseline: tok_s ratio 1.566, itl_p50 ratio 0.624, ttft_p50 ratio
  0.794 — target `≥ 75% of vLLM-ROCm 0.23.0 (GPU 0)` reads `on`. `golden_c1`/`golden_c16` left
  unset on both runs (fixture not run yet); `golden_summary` on the Turbine row notes why.

**Open questions / next steps** (unchanged from the previous rotation's list, still open):
1. Golden fixture with `--act-quant none` — rank 4 in `/home/piwi/turbine-ci/fixture.queue`, not
   yet run. Until it lands, this proof's PASS does not cover golden-level numerics, only the eval
   accuracy gate.
2. `lab-bench --model llama-fp8-block --golden16 --c1` once the fixture reference exists.
3. Tolerance bounds from the self-spread method (as OLMoE's was calibrated) — not yet computed for
   this checkpoint; the eval-compare 0.02 drop bound was used instead (Phase 6a per-format default
   for FP8).
4. Soak (`scripts/overload-soak.sh novanas --duration 10m` at minimum) — needs the lead's go-ahead
   per the lab rules.
5. perf-log row for this proof — not yet added; the labbook runs above are the durable record in
   the meantime.

**Row-flip commit**: prepared separately (see below), not merged into this branch's history — the
lead reviews and applies it.

## Rotation 11 (2026-09-30, lead brief r11-fp8block)

Merged `phase-6a-quantization` (tip `ff804c4`). The merge conflicted in `support.rs` on the fp8
Task 14 row landing in the same block as the fp8_block flip; to keep the flip the LAST commit
touching that file, `ef09f16` was reverted (`5695d2a`), the merge then applied cleanly
(`fd16906`), and the flip was re-applied with the same content on top (`7d7e644`).

### 0. lab-serve.sh remote log-stream fix — done

`c648681`: `stop_log_stream` now also `pkill`s the remote `kubectl logs -f job/${JOB}` (the
process an ssh ControlMaster keeps alive after the local client is killed — the cause of the
19-minute hang in the previous rotation's soak) once a stream was actually started, with the
same bracket-trick pattern `gpu_unlock` already uses. New test
`lab_serve_stops_the_remote_log_stream_once` plus an assertion appended to
`lab_serve_dry_run_prints_the_start_sequence`. `scripts/gate.sh` (background, this rotation's
final state): see the tail of this session's log for the exit line; if it hasn't landed yet,
rerun `scripts/gate.sh` before trusting this branch.

### 1. Golden fixture — self-spread still running, not yet committed

`tests/golden/llama-3.2-3b-instruct-fp8-block/reference.jsonl` exists and is verified
(`novanas:/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/llama-3.2-3b-instruct-fp8-block/reference.jsonl`,
16 lines, `rc=0` at 20260930 00:57Z). The self-spread it needs (`--act-quant none`, as the FP8
per-tensor README's method) is running but had not finished as of this handoff:

- Job: `/home/piwi/turbine-ci/fp8block_spread.sh` on novanas, launched detached
  (`setsid nohup`), dequantizes the checkpoint to `/dev/shm` then runs
  `scripts/golden/self_spread.py … --act-quant none` against the reference above, writing
  `.../fixtures/llama-3.2-3b-instruct-fp8-block/spread.json`.
- Queue: inserted `fp8-block-self-spread` (matches `FIXTURE_JOB=r11:fp8-block-self-spread` on
  the job's command line) into `/home/piwi/turbine-ci/fixture.queue` right before the
  `llama-3\.1-8b` line, per the brief (the 8B MXFP4-A16 reference job was already holding
  `fixture.lock` and finishes first; a backup of the queue file before the edit is
  `fixture.queue.bak-r11`). Confirmed by `fixture-order.log`: my job is ranked ahead of every
  other current waiter, so it will run right after the 8B job frees the lock.
- Watch: `/home/piwi/turbine-ci/fp8block_spread.nohup.log` and
  `.../fixtures/llama-3.2-3b-instruct-fp8-block/{dequant.r11.log,spread.r11.log}`
  (the log path baked into the script — see
  `/private/tmp/claude-501/.../scratchpad/fp8block_spread.sh` for the exact script kept in this
  session's scratchpad, not committed). Last line `fp8block-spread: done rc=<rc>`.
- Once `spread.json` exists: build `tests/golden/llama-3.2-3b-instruct-fp8-block/tolerance.json`
  and `README.md` from it exactly as
  `tests/golden/llama-3.2-3b-instruct-fp8/{tolerance.json,README.md}` were (max(spread, BF16
  bounds) — decisions.md "Golden tolerance floor for quantized checkpoints"), copy the verified
  `reference.jsonl` in beside them, commit as
  `tests/golden/llama-3.2-3b-instruct-fp8-block/{reference.jsonl,tolerance.json,README.md}`, and
  run `cargo test -p turbine-bench --test golden quant_fixtures_valid` (the fixture is already
  registered in `QUANT_SLUGS` with the matching repo/revision, so it only needs the three files
  to exist and pass `check_fixture`). Not committed yet in this rotation because the numbers
  aren't known.

### 2. GPU job A: golden1 + golden16 + bench — running, waiting for its go-file

Script `.procoder/handoff/p6a-fp8-t15-bench-r11.sh` (committed `eb85753`, log-redirect-order fix
`5c6df91`), the equivalent of `scripts/lab-bench.sh --model llama-fp8-block --golden16 --c1` as a
native detached script (built from this workspace's `src`, so it doesn't foreground-block a
caller for the whole measurement). Launched on novanas as
`/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/fp8block_bench_launch.sh`
(`setsid nohup`, pid printed in `fp8block_bench_launch.nohup.log`); it builds the kernel library
and the release server/bench/golden, then waits (bounded 24h) for the go-file
`/home/piwi/turbine-ci/gpu-queue/fp8block-bench.go` before taking
`port18000.lock → bench.gate → bench.lock` on GPU 0, pausing CPU fixture jobs for the duration.
Serves `scripts/lab/phase6-novanas-llama-fp8-block.yaml` with
`reliability.circuit.latency_drift_open=100` (the golden-c1 / c1-latency legs serialize one
request at a time, which can look like a latency drift to the default 4.0 threshold) and greps
circuit transitions out of the server log.

- Watch: `.../fp8block-bench/fp8block-bench.log`, last line `fp8block-bench: done rc=<rc>`, and
  the `fp8block-bench: BENCH r11-fp8block llama-fp8-block …` summary line just above it.
  Judge: `golden1`/`golden16` verdicts, `bench.json`'s `output_token_throughput` (compare to the
  854.7 tok/s BF16 floor and the t15-proof pass's 1061.44), `bench-c1.json`'s ITL, and that no
  `circuit:` lines show `CIRCUIT_OPEN`.
- **Sent `ready: fp8block-bench.go` to the lead** — the lead creates the go-file in GPU queue
  order; nothing here starts before it.

### 3. GPU job B: the 10-minute soak — waiter running, waiting for its go-file

Snapshot worktree `.claude/worktrees/r11-soak-fp8block` (`git worktree add --detach` at commit
`5c6df91`, which carries the lab-serve fix), so later commits on `p6a-fp8-t15` don't change the
soak mid-run. Mac-side waiter
`scratchpad/fp8block-soak/wait_and_soak.sh` (not committed, session scratchpad only, following
the previous rotation's `fp8-soak/wait_and_soak.sh` pattern) launched detached (`nohup … &`,
pid in this session): waits for the go-file
`/home/piwi/turbine-ci/gpu-queue/fp8block-soak.go` AND a free `bench.lock` (checked over ssh
every 10 min, bounded 9h), then runs
`scripts/overload-soak.sh novanas --duration 10m --model /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-block`
from the snapshot worktree.

- Watch: `scratchpad/fp8block-soak/wait_and_soak.log`, last line `fp8block-soak: done rc=<rc>`
  naming the verdict directory (`target/soak/novanas-*` under the snapshot worktree).
- **Sent `ready: fp8block-soak.go` to the lead** — same queue-order rule as job A; this should
  run after job A releases `bench.lock` (or whatever the lead's queue order says).

### Left after this rotation

1. The self-spread (§1) — once it lands, build and commit the fixture's `tolerance.json` /
   `README.md`, then run `quant_fixtures_valid`.
2. Judge GPU job A's log once `fp8block-bench.go` is created and the run finishes.
3. Judge GPU job B's log once `fp8block-soak.go` is created and the run finishes; delete the
   snapshot worktree (`git worktree remove .claude/worktrees/r11-soak-fp8block`) once judged.
4. Perf-log row and labbook entries for both runs, once judged (not added yet — the S-11 gate
   needs golden c1/c16 against the real fixture plus the bench and the soak, all three still
   pending at the end of this rotation).
5. The support-matrix row flip (`7d7e644`) stays provisional until 1–3 above close it out; if
   any of them fails, the row needs to go back to `experimental` in a follow-up
   `handoff(support.rs)` commit.

### Rotation 11 update: GPU job A ran already (bench numbers in, golden missing the fixture)

The lead created `fp8block-bench.go` quickly; the job ran and finished:
`fp8block-bench: BENCH r11-fp8block llama-fp8-block commit=unknown gpu=0 rc=1 (...)`. `rc=1`
only because `golden1`/`golden16` failed with "cannot read
tests/golden/llama-3.2-3b-instruct-fp8-block/reference.jsonl" — expected, since §1's fixture
isn't committed yet. The measurement legs succeeded: `bench tok/s 1054.96 ok 200` (matches the
t15-proof pass's 1061.44 within noise) and `c1 itl_ms.p50 8.43 tok_s_c1 112.66 ok 10`. No
`fp8_block_decoded` or `CIRCUIT_OPEN` lines were expected to show and weren't checked yet in this
note — read `.../fp8block-bench/server.log` before trusting the numbers. **Once §1's fixture is
committed, rerun just the golden legs** (`turbine-golden compare --url ... --concurrency 1|16`
against the release binaries already built in this run's workspace) rather than the whole job.

`fp8block-soak.go` was also created; the soak (GPU job B) is running as of this note (job
`turbine-lab-serve-0929224541-301172b6`, snapshot worktree `r11-soak-fp8block`) — not finished,
judge `scratchpad/fp8block-soak/wait_and_soak.log` for its `fp8block-soak: done rc=` line.

## Rotation 12 (2026-09-30, lead brief r12-fp8block-finish): closed out

Merged `phase-6a-quantization` tip `96d3b76` (clean, no conflicts; support.rs untouched upstream,
so the flip `7d7e644` stays the last commit touching it) — merge commit `46374d6`.

1. **Fixture committed** (`ee6bfb1`): `tests/golden/llama-3.2-3b-instruct-fp8-block/{reference.jsonl,tolerance.json,README.md}`.
   The self-spread (queued job from rotation 11) finished clean before this rotation started:
   8 variants, all 16/16 prefix-ok, max likely 0.1829 (bf16-sdpa-full), max tail 0.3107
   (bf16/eager incremental+full) — both below the BF16 Llama floor (0.15/0.55 strict,
   0.25/0.75 batched), so `tolerance.json` is floor-bound: likely 0.19, tail 0.55, batched
   unchanged at 0.25/0.75. `cargo test -p turbine-bench --test golden quant_fixtures_valid`
   passes (run through `scripts/remote-cargo.sh`).
2. **Job A's server.log checked**: 0 occurrences of `fp8_block_decoded` and `CIRCUIT_OPEN` —
   the rotation-11 bench numbers (1054.96 tok/s c16, 112.66 tok/s c1) are clean.
3. **Golden c1+c16 rerun** with the fixture in place: `scripts/lab-bench.sh --model llama-fp8-block
   --label r12-fp8block --golden16` (background) — both PASS (16/16 each; c1 strict bounds, c16
   batched bounds). tok/s 1062.4, ITL p50 11.5 ms, TTFT p50 226 ms; no `fp8_block_decoded` /
   `CIRCUIT_OPEN` in `/tmp/lab-bench-server.last.log` either. Auto-recorded in labbook
   (`turbine-lab-bench`, run `a9b6beb4-e35d-410d-9a96-d6c2e9c482a9`, status pass) including
   `golden_c1`/`golden_c16`.
4. **Soak** (rotation 11's job, judged here): `scratchpad/fp8block-soak/novanas-20260929T224540Z/verdict.json`
   `pass: true`, all 8 checks true (ITL p99 216.4 vs calibration 186.6 ms, GREEN 30 s after
   cool-down, no drops). Recorded in labbook (`turbine-overload-soak`, run
   `d5d92074-83fa-4e23-b1f2-3cb8d1efa3b6`, set `phase-6a-quantization`).
5. **perf-log row** added (Task 15 proof line, `.procoder/perf-log.md`) with the full set of
   numbers: golden PASS/PASS, tok/s 1062.4 (1.24× BF16 floor 854.7), c1 ITL 8.43 ms, GSM8K 0.800
   vs BF16 0.805 (drop 0.005), vLLM-ROCm 677.8 tok/s (1.57×), soak PASS, support row `supported`.
6. **Gate**: `scripts/gate.sh --base 46374d6` → `gate: ok crates=turbine-model,turbine-server
   passed=323 failed=0`.

All four proof legs (numerics/golden, throughput, accuracy, soak) now pass on the merged tip.
The support-matrix flip (`7d7e644`, already in this branch's history) is confirmed, not
provisional. **Ready to merge** — sent to the lead.
