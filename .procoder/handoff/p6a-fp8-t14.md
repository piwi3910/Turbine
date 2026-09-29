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

## Collector's remaining steps

1. Commit the eval JSONs, the tensor tolerance (and new reference if replaced), perf-log rows
   (`.procoder/perf-log.md`), gate.json final form.
2. Labbook upload (set `phase-6a-quantization`, vLLM FP8-dynamic as baseline where served), a
   `BENCH`-equivalent line per pass from the `T14` lines.
3. Soak (ask the lead/coordinator first, one GPU job at a time):
   `scripts/overload-soak.sh novanas --duration 10m --model /home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic`.
4. Flip `amd/gfx1201/LlamaForCausalLM/fp8/bf16/none` to `supported` (+ `baseline_rows_present`) as a
   `handoff(crates/turbine-core/src/support.rs)` commit for the lead, only when golden, bench, eval
   and soak pass; the per-tensor checkpoint shares the `fp8` row, so its tolerance and golden must be
   in too.
