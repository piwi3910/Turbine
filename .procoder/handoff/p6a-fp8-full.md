# Handoff: p6a-fp8-full (plan Task 14, "Eval, full set" — FP8-dynamic vs vLLM on the full GSM8K)

Written 2026-09-29 by the queuing builder. Worktree `.claude/worktrees/agent-p6a-fp8-full`, branch
`p6a-fp8-full`, fresh from integration `7277c8b`.

## Why

User decision 2026-09-29 ("FP8-dynamic (Task 14): accuracy on full GSM8K against vLLM, c1 ITL as a
perf item"): the Task 14 proof (`.procoder/handoff/p6a-fp8-t14.md`) judged FP8-dynamic accuracy on
GSM8K-**200** at concurrency 16, where it missed the binding gate
(`tests/eval/llama-3.2-3b-instruct-fp8-dynamic/gate.json`, `max_drop` **0.01**): vLLM 0.82 (164/200)
vs Turbine 0.79 (158/200), drop 0.03 > 0.01. This re-judges the same comparison on the full
1,319-item `tests/eval/gsm8k-full.jsonl` (less sampling noise), Turbine vs vLLM-ROCm serving the
*same* checkpoint (`/home/piwi/turbine-models/llama-3.2-3b-instruct-fp8-dynamic`, served name
`RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic`) on gfx1201, **both at concurrency 16** (`eval-compare`
refuses a mixed-concurrency pair by construction). The c1 ITL bound miss from the T14 proof is out
of scope here — it is a separate, later perf item, not re-run by this driver.

## What this driver does

Built from the templates named in the task: the full-GSM8K serve+eval shape of
`/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run_r5.sh` (locks, `run_pass`,
`--tasks tests/eval/gsm8k-full.jsonl`, `~6h` timeout) and the vLLM leg of the T14 proof driver
(`/home/piwi/turbine-ci/scratch/p6a-fp8-t14/t14_proof_run.sh`'s `run_vllm`, k3s Job on port 18100,
same checkpoint/served-name substitution).

- Script (novanas): `/home/piwi/turbine-ci/scratch/p6a-fp8-full/fp8_full_run.sh` (copy committed at
  `.procoder/handoff/fp8_full_run.sh` in this branch).
- Started: `setsid nohup bash fp8_full_run.sh >launch.log 2>&1 </dev/null &` from
  `/home/piwi/turbine-ci/scratch/p6a-fp8-full/` on novanas, pid 681177 (confirmed running, `Ss`
  state, no controlling terminal).
- Builds `libturbine_hip.so` itself first (nice 19, cores 12-15, no lock — CPU work), from
  `/home/piwi/turbine-ci/remote/agent-p6a-fp8-full/kbuild`. The native release `turbine-server` /
  `turbine-bench` (which provides `turbine-golden`) were pre-built by this session via
  `scripts/remote-cargo.sh build --release -p turbine-server -p turbine-bench` into
  `/home/piwi/turbine-ci/remote/agent-p6a-fp8-full/target/release` before the script started.
- Takes `port18000.lock`, then `bench.gate`, then `bench.lock`, **in that order, once, held for the
  whole script** (both legs, not per-pass) — per the task's explicit instruction, this differs from
  the T14 template's per-pass lock/release. CPU fixture jobs are paused (`pkill -STOP … 'scripts/[g]olden/'`)
  for the duration and resumed right after both legs finish, before `eval-compare` runs (which needs
  no GPU/lock).
- **Pass `turbine`**: serves `scripts/lab/phase6-novanas-llama-fp8.yaml` natively
  (`ROCR_VISIBLE_DEVICES=0`, `taskset -c 0-11`, GPU 0), `model.path=llama-3.2-3b-instruct-fp8-dynamic`;
  `turbine-golden eval --url http://127.0.0.1:18000 --concurrency 16 --tasks tests/eval/gsm8k-full.jsonl
  --output json` → `run/turbine.json`; server always killed after (or on a not-ready failure).
- **Pass `vllm`**: k3s Job `turbine-lab-vllm-fp8full-<ts>` from `scripts/lab/novanas-vllm-job.yaml`
  (same checkpoint, port 18100); on ready, the same `turbine-golden eval` against
  `http://127.0.0.1:18100` → `run/vllm.json`; a refusal is recorded (`run/vllm.log`,
  `run/vllm.status=REFUSED`) and the run still completes; the Job is always deleted either way.
- Then, if both JSONs are present and non-empty: `turbine-golden eval-compare --baseline
  run/vllm.json --candidate run/turbine.json --max-drop 0.01` (the `gate.json` value), output
  appended to the log.
- Last log line: `fp8-full: done rc=<rc> turbine=<rc> vllm=<rc>` (`rc` is the OR of the turbine-eval,
  vllm-eval and `eval-compare` exit codes).

## Outputs and where to look

- Log: `/home/piwi/turbine-ci/scratch/p6a-fp8-full/launch.log` (the script's whole stdout/stderr;
  the caller-side `launch.log` from the `setsid nohup … >launch.log` invocation — there is no
  separate internal log file this time, unlike the T14/gsm8k-full-r5 templates).
- Results dir: `/home/piwi/turbine-ci/scratch/p6a-fp8-full/run/`:
  - `turbine.json`, `turbine.err`, `turbine.done`, `turbine-server.log` — the Turbine FP8-dynamic
    full-GSM8K eval report (accuracy, correct/total, concurrency, per-item results) and its server log.
  - `vllm.json`, `vllm.err`, `vllm.done`, `vllm.log`, `vllm.status` (`SERVED`/`REFUSED`),
    `vllm-job.yaml`, `vllm-pod.txt` (only on a not-ready failure) — the vLLM leg.
  - `kbuild.log` — the kernel library build.
- **Do not poll it.** ETA: each full-GSM8K eval is ~6× the GSM8K-200 eval time (the T14 proof's
  GSM8K-200 pass took well under an hour per model at c16; budget a few hours per leg, up to the
  script's own 21600 s / 6 h internal timeout each), plus vLLM Job scheduling/startup. Total: likely
  several hours. `run/turbine.done` / `run/vllm.done` appear as each leg finishes; `launch.log`'s
  last line is the done marker for the whole run.

## How to judge

1. **The gate itself**: `fp8-full: done rc=<rc> turbine=<rc> vllm=<rc>` in `launch.log`, and the
   `eval-compare` line just above it (`baseline accuracy …, candidate accuracy …, max drop 0.0100
   (concurrency 16): PASS|FAIL`). `rc=0` end to end means PASS at the binding gate; a non-zero `rc`
   with `turbine=0 vllm=0` means the accuracy drop itself failed the 0.01 bound (the likely outcome
   given the GSM8K-200 result: vLLM 0.82 vs Turbine 0.79, drop 0.03).
   - If `vllm.status=REFUSED`: `eval-compare` cannot run against vLLM; fall back to `gate.json`'s
     `bf16_command` (against `tests/eval/llama-3.2-3b-instruct/turbine-bf16-c16.json`, `max_drop`
     0.02) or, better, run a full-GSM8K BF16 pass for a like-for-like full-set BF16 baseline (not
     produced by this driver — it only runs the FP8-dynamic and vLLM legs, since the task named only
     those two).
2. **Significance, like the FP8 KV collector's judgment** (`.procoder/handoff/p6a-kv-t24-full.md`,
   "Verdict" section): a literal `eval-compare` FAIL on a 0.01–0.03 drop over 1,319 items can still
   be noise. Load `run/turbine.json` and `run/vllm.json`'s per-item correctness (same task order,
   both JSONL from `tests/eval/gsm8k-full.jsonl`), and report:
   - **lost / gained**: items vLLM got right that Turbine got wrong (lost) vs items Turbine got
     right that vLLM got wrong (gained) — the discordant pairs.
   - **McNemar's test (continuity-corrected)** on the lost/gained counts: `p = P(chi2_1 > (|lost -
     gained| - 1)^2 / (lost + gained))`; a large p (as with OLMoE FP8 KV's 0.23) means the drop is
     not distinguishable from noise despite failing the literal bound.
   - **95% CI of the drop** (e.g. Wilson or a normal approximation on the paired difference) — does
     it include 0?
   - No item flips when its output is unchanged (a sanity check, same as the KV collector's report).
   Report both the literal `eval-compare` verdict and this significance read to the lead before any
   `support.rs` change — that decision is explicitly lead-owned, not this builder's.
3. **Numerics-cause check, only if the drop is both large and significant** (unlike the OLMoE FP8 KV
   case): compare Turbine's wrong answers against vLLM's on the discordant items for a pattern (e.g.
   one class of arithmetic slip, a specific answer-format mismatch in the matcher) before treating it
   as a genuine format-accuracy cost of FP8-dynamic weight quantization.

## Status at handoff time

Script confirmed started and running (pid 681177 on novanas, `launch.log` shows the kernel-library
build in progress as of 2026-09-29 18:00 +04). Not waited on further — per the lead's rules, this
builder ends here rather than polling a multi-hour run. `bench.lock` was free when this run's locks
were requested (the FP8 KV full-GSM8K requeue from rotation 5 had already released it by then), so
this run should not be queued long behind another GPU job unless something new took the lock after
this handoff was written — check `bench.lock`'s holder if the run appears stalled at "waiting for
bench.lock" for an unexpectedly long time.

## Remaining steps (for whoever picks this back up)

1. Wait for `fp8-full: done rc=…` in `launch.log` (hours).
2. Judge per "How to judge" above; compute the flips/McNemar/CI read, not just the literal
   `eval-compare` exit code.
3. Copy `run/turbine.json` and `run/vllm.json` into `tests/eval/llama-3.2-3b-instruct-fp8-dynamic/`
   (e.g. `turbine-full.json`, `vllm-full.json`) and update `gate.json`'s `status` field with the
   full-set result, same style as the GSM8K-200 entry already there.
4. Report the finding to the lead before any `support.rs` change (lead-owned file, per the rotation
   rules) — the flip from `experimental`/current status to `supported` (or not) for
   `amd/gfx1201/LlamaForCausalLM/fp8/bf16/none` depends on this and on the still-open per-tensor
   golden and the c1 ITL bound miss noted in `p6a-fp8-t14.md`.
