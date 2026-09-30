# Handoff: p6a-fp8-t14 (plan Task 14, the `fp8` proof on Llama-3.2-3B)

Written 2026-09-29 by the T14 builder (rotation 6). Worktree `.claude/worktrees/agent-p6a-fp8-t14`,
branch `p6a-fp8-t14` (from integration 090e09c + d8c8e45). History: `p6a-fp8.md`.

## Done (this branch)

- `tests/golden/llama-3.2-3b-instruct-fp8-dynamic/`: the novanas reference (03:02Z capture, the one
  the spread was measured against; same tokens as the 09-28 capture), `tolerance.json` from the
  self-spread (all eight variants): likely 0.78, tail 5.08, batched the same, `min_prompts_passing` 14
  (decisions "Golden tolerance for activation-quantized checkpoints" and "… floor for quantized
  checkpoints"); README with repo / revision / commands / the spread table. Spread JSON (not
  committed): novanas `/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/llama-3.2-3b-instruct-fp8-dynamic/spread.json`.
- `tests/golden/llama-3.2-3b-instruct-fp8/`: the novanas per-tensor reference, `tolerance.json` still
  the BF16 one (provisional), README says why (below).
- `scripts/lab/phase6-novanas-llama-fp8{,-tensor}.yaml` (copies of phase2c-novanas-llama.yaml).
- `tests/eval/llama-3.2-3b-instruct-fp8-dynamic/gate.json`: vLLM same checkpoint at 0.01 if vLLM
  serves it, else BF16 at 0.02 (Q9); every eval at concurrency 16.
- `.procoder/handoff/t14_proof_run.sh`, `.procoder/handoff/t14_spread.sh`: copies of the detached
  scripts running on novanas.
- `quant_fixtures_valid` passes with both slugs (they were already in `QUANT_SLUGS`).

## Finding: the per-tensor reference predates the fused-max input scales

The committed per-tensor reference was captured 2026-09-28 22:22Z; `quant_reference.py` gave a fused
projection's parts their largest `input_scale` (what the loader and vLLM do) only from 4729281
(09-29 00:28Z), and the engine string of a pre-change capture cannot say which rule produced it. So
`t14_spread.sh` regenerates the reference first and compares (`reference-diff.txt`: SAME /
DIFFERENT); when it differs it replaces the novanas fixture file (old kept as
`reference.pre-t14.jsonl`) and the spread is measured against the new one.

## Running detached on novanas (don't wait; judge when the done lines appear)

1. **Per-tensor spread** (CPU, fixture queue): `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/t14_spread.sh`.
   Log `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/spread/t14-spread.log`, last line
   `t14-spread: done rc=<rc>`. The waiter is `flock …/fixture.lock … FIXTURE_JOB=t14:fp8-tensor-spread
   bash …t14_spread.sh --locked`; it matches no line of `/home/piwi/turbine-ci/fixture.queue`, so
   `fixture-order.sh` ranks it after every listed job (the MXFP4 8B steps) unless the lead adds a
   `t14:fp8-tensor` line. Outputs: `spread/{reference.jsonl,reference-diff.txt,spread.json}`, and
   `…/agent-adb021480bf19bfbe/fixtures/llama-3.2-3b-instruct-fp8/spread.json` (fixtures-r5 step 4 then
   skips it).
2. **GPU proof** (queued on port18000.lock → bench.gate → bench.lock, pass by pass):
   `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/t14_proof_run.sh`, binaries of this branch
   (`/home/piwi/turbine-ci/remote/agent-p6a-fp8-t14/target/release`, kernel lib `…/kbuild`).
   Log `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/run/run.log`, last line `t14_proof_run: done rc=<rc>`,
   one `T14 <pass> tok/s=… itl_p50=… c1_itl_p50=… golden1=… golden16=… gsm8k=…` line per pass.
   Results `…/run/<bf16|fp8|fp8-tensor|vllm>/{golden1.txt,golden16.txt,capture.jsonl,bench.json,
   bench-c1.json,quality.json,status.json,server.log|vllm.log}`; a pass with a `done` file is skipped
   on a rerun (requeue after a crash: start the same script again the same way).
   - bf16: bench c16 (200 req) + c1, GSM8K-200 at c16 (no c16 BF16 GSM8K-200 existed).
   - fp8 (dynamic): golden c1 + c16, bench c16 + c1, GSM8K-200 at c16.
   - fp8-tensor: golden c1 + c16 against the tree's (provisional) tolerance, a c1 `capture.jsonl`
     (`tolerance-used.json`, `reference-used.sha1` beside it), bench c16 + c1.
   - vllm: k3s Job `turbine-lab-vllm-t14-<ts>` (FP8-dynamic, port 18100 under port18100.lock), bench
     c16 + c1, GSM8K-200 at c16; `vllm/status` = SERVED or REFUSED (log `vllm/vllm.log`,
     `vllm/pod.txt`); `vllm/amd-smi.txt` shows which card the pod got (bench only comparable on GPU 0).

## How to judge

- Golden (fp8 dynamic): last line of `run/fp8/golden1.txt` and `golden16.txt` PASS (c16 on the
  batched bounds, here equal to the strict ones). A p05 `missing_top_k=token 9478 at position 23`
  is the reference's near-tie (README), check with `turbine-golden positions --prompt-id p05`.
- Golden (fp8 tensor): once the spread lands, derive `tolerance.json` like the dynamic one (smallest
  two-decimal bounds all variants meet, floored at 0.15 / 0.55 / 0.25 / 0.75, `min_prompts_passing`
  14). If the reference was replaced, copy the new `reference.jsonl` into the tree and re-judge the
  saved capture per prompt: `turbine-golden positions --reference <new ref> --candidate
  run/fp8-tensor/capture.jsonl --prompt-id <id> --tolerance <new tolerance>` (c1 only; the c16 run
  needs a new serve if the reference changed). Otherwise read the golden verdicts against the new
  bounds from the per-prompt maxima in golden1/golden16.txt.
- Throughput (both FP8 variants, vs the same run's bf16 pass): `bench.json`
  `output_token_throughput` ≥ 1.10 × bf16; `bench-c1.json` `itl_ms.p50` ≤ 0.75 × bf16. (Interim
  quick run: per-tensor 965.1 tok/s = 1.13×, ITL 14.0 ms.)
- Accuracy: copy `run/bf16/quality.json` → `tests/eval/llama-3.2-3b-instruct/turbine-bf16-c16.json`,
  `run/fp8/quality.json` → `tests/eval/llama-3.2-3b-instruct-fp8-dynamic/turbine.json`,
  `run/vllm/quality.json` → `…/vllm.json` (if SERVED); run the `command` of `gate.json` (vLLM
  served) or its `bf16_command` after switching the gate to BF16 (REFUSED; record `vllm_refusal`).
  Expect exit 0.

## Collector's verdicts (2026-09-29, rotation 6/7 collector, run/proof commit 3635f68)

The GPU proof run (`t14_proof_run.sh`) is done, `ALLDONE rc: bf16=0 fp8=0 fp8-tensor=0 vllm=0
(vllm status SERVED)`. Numbers are also in `.procoder/perf-log.md` (Phase 6a table) and labbook
(set `phase-6a-quantization`, type `turbine-lab-bench`, external ids
`t14-proof:llama-{bf16,fp8-dynamic,fp8-tensor}:turbine:3635f68` and
`t14-proof:llama-fp8-dynamic:vllm-rocm:3635f68`; the vLLM run is now a member of baseline
`vllm-rocm-0-23-gpu0` matched on `model=llama-3.2-3b-instruct-fp8-dynamic`).

### FP8-dynamic (RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic) — judged in full

- **Golden**: PASS c1 (16/16, strict bounds 0.78/5.08) and PASS c16 (16/16, batched bounds
  0.78/5.08, equal to strict here). Every prompt genuinely passes — no p05 near-tie excuse needed
  (the review's p05 note applies only to the exact `missing_top_k=token 9478 at position 23` miss,
  which does not occur in this run).
- **Throughput**: 976.36 tok/s vs bf16 849.65 = **1.149× — PASS** (bound ≥ 1.10×).
- **ITL**: c1 `itl_ms.p50` 10.362 vs bf16 12.241 = **0.846× — MISS** (bound ≤ 0.75×; fp8-tensor is
  the same, 0.855×, so this is a shared, not dynamic-specific, shortfall — the c1 ITL bound as
  written in this handoff is not met by either variant even though c16 throughput clears its bound
  comfortably).
- **GSM8K-200 (chain-of-thought, concurrency 16)**: turbine fp8-dynamic 0.79 (158/200); turbine
  BF16 same run 0.795 (159/200); vLLM-ROCm 0.23.0 same checkpoint (SERVED) 0.82 (164/200).
  `gate.json`'s own rule makes the vLLM comparison binding once vLLM serves the checkpoint:
  `turbine-golden eval-compare --baseline vllm.json --candidate turbine.json --max-drop 0.01` →
  **drop 0.03, rc=1, MISS**. The BF16-alongside comparison
  (`--baseline turbine-bf16-c16.json --max-drop 0.02`) → drop 0.005, rc=0, PASS, but per
  `gate.json`'s rule text this is recorded alongside, not the binding gate, because vLLM did serve
  the checkpoint. `gate.json.status` now records both numbers and the outcome.
- **vLLM comparison** (throughput/latency, informational): Turbine is 1.50× vLLM's tok/s and
  0.62× (i.e. lower/better) vLLM's c1 ITL; vLLM's own GSM8K on this checkpoint (0.82) is higher
  than both Turbine BF16 (0.795) and Turbine FP8-dynamic (0.79) — this is what makes the accuracy
  gate bind at 0.03 rather than the 0.005 seen against Turbine's own BF16.
- Files: `tests/eval/llama-3.2-3b-instruct/turbine-bf16-c16.json`,
  `tests/eval/llama-3.2-3b-instruct-fp8-dynamic/{turbine,vllm}.json` (committed).

**Overall for FP8-dynamic: golden PASS, throughput PASS, ITL bound MISS, GSM8K MISS against the
binding vLLM gate (PASS against the BF16-alongside bound).** Two of four criteria miss their
stated bound; the lead should decide whether the support-matrix flip proceeds regardless (e.g. if
the ITL bound or the vLLM-binding accuracy rule themselves need revisiting) — not done here.

### FP8 per-tensor (RedHatAI/Llama-3.2-3B-Instruct-FP8) — golden still pending

- **Golden**: not yet judged for real. The proof run's golden c1/c16 (`run/fp8-tensor/golden{1,16}.txt`)
  FAIL against the tree's *provisional* BF16 tolerance (1/16 and 3/16 passing; need 14) — expected,
  since `tests/golden/llama-3.2-3b-instruct-fp8/tolerance.json` is still the BF16 one pending the
  self-spread. Do not read these FAILs as the real verdict.
  - **Spread status now**: `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/spread/t14-spread.log` last
    line as of this handoff: `t14-spread: 2026-09-29 15:59:09 waiting for fixture.lock` (still
    queued behind the MXFP4 8B fixture steps per `fixture.queue`; not started).
  - **Done marker to watch for**: `t14-spread: done rc=<rc>` at the end of that same log. Outputs
    land in `/home/piwi/turbine-ci/scratch/p6a-fp8-t14/spread/{reference.jsonl,reference-diff.txt,spread.json}`.
  - **Re-judge once it lands** (see "How to judge" above, unchanged): derive `tolerance.json` from
    `spread.json` (smallest two-decimal bounds all 8 variants meet, floored at 0.15/0.55/0.25/0.75,
    `min_prompts_passing` 14) the same way the fp8-dynamic one was derived; check
    `spread/reference-diff.txt` for SAME/DIFFERENT — if DIFFERENT, copy the new
    `spread/reference.jsonl` over `tests/golden/llama-3.2-3b-instruct-fp8/reference.jsonl` (keep the
    old one as `reference.pre-t14.jsonl`) before re-judging; then re-judge the saved c1 capture
    without a new serve: `turbine-golden positions --reference <ref> --candidate
    run/fp8-tensor/capture.jsonl --prompt-id <id> --tolerance <new tolerance.json>` per prompt (or
    read the per-prompt maxima straight out of `run/fp8-tensor/golden1.txt` against the new bounds,
    since it already lists `max_abs_logprob_diff_likely`/`_tail` per prompt). The c16 run needs a
    fresh serve only if the reference itself changed (bump `reference-used.sha1` next to the
    capture to confirm).
- **Throughput**: 976.99 tok/s vs bf16 849.65 = **1.149× — PASS** (same margin as fp8-dynamic,
  expected since both ride the same `hipblaslt_fp8` W8A8 GEMM).
- **ITL**: c1 10.469 vs bf16 12.241 = **0.855× — MISS** (bound ≤ 0.75×; same shortfall as
  fp8-dynamic).
- **GSM8K**: not run for this pass (the driver only runs golden + bench for `fp8-tensor`, per the
  "Running detached" section above; no accuracy criterion to report here yet).
- Files in place: `tests/golden/llama-3.2-3b-instruct-fp8/{reference.jsonl,tolerance.json,README.md}`
  (tolerance still provisional, README already says so), capture at
  `run/fp8-tensor/capture.jsonl` on novanas (not copied into the tree — collect it once the spread
  lands and the re-judge needs it, or copy proactively before the scratch directory is pruned).

## Remaining steps (for whichever builder/lead picks this back up)

1. ~~Commit the eval JSONs, perf-log rows, gate.json final form~~ — done, this commit.
2. ~~Labbook upload~~ — done, this commit's companion labbook calls (see above; not a git commit).
3. **Once `t14-spread.sh` prints its `done` line**: derive the real per-tensor tolerance, re-judge
   as described above, commit `tests/golden/llama-3.2-3b-instruct-fp8/{tolerance.json,reference.jsonl,README.md}`
   and a labbook update for the fp8-tensor golden fields, and fold the result into this handoff.
4. Soak (ask the lead/coordinator first, one GPU job at a time):
   `scripts/overload-soak.sh novanas --duration 10m --model /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic`
   — not run by this collector (out of scope per the task: "No soak … the lead decides those").
5. Flip `amd/gfx1201/LlamaForCausalLM/fp8/bf16/none` to `supported` (+ `baseline_rows_present`) as a
   `handoff(crates/turbine-core/src/support.rs)` commit for the lead — **not done here** (out of
   scope: "No … support.rs change: the lead decides those"). Given the ITL-bound and GSM8K-vs-vLLM
   misses above, and the still-pending per-tensor golden, the lead's call on this flip needs to
   weigh those misses explicitly; the per-tensor checkpoint shares the `fp8` row, so its tolerance
   and golden must land before any flip that covers it.
