# Handoff: p6a-mxfp4 (plan Task 20, MXFP4 proofs)

Written 2026-09-29 ~09:20 by the second Task 20 builder, per the lead's rotation rule (only waiting was
left). Worktree `.claude/worktrees/agent-a4784842b25c93376`, branch `p6a-mxfp4`. Read AGENTS.md, plan
Task 20 and the decisions "W4A4 proof checkpoint" and "Phase 6a gate misses on GSM8K-200" first.

## Done

- f5d84b2: merged `phase-6a-quantization` (d1de280: the W4A4 8B KV recipe repeated for `*k_proj` /
  `*v_proj` is ignored; `tests/eval/gsm8k-full.jsonl`; fixture queue order).
- 14ac7e8 (gate ok, 775 passed): `golden.rs` `QUANT_SLUGS` names `llama-3.1-8b-instruct-mxfp4-a4`
  (AMD Quark W4A4 8B @ 00b0d018) instead of the base 3B; BF16 8B golden fixture committed with a
  **provisional** tolerance (the 3B values; README says so); `tests/eval/llama-3.1-8b-instruct/turbine-bf16.json`
  (0.89) and `tests/eval/llama-3.1-8b-instruct-mxfp4a16/{turbine.json (renamed, 0.835), gate.json}`;
  lab configs `scripts/lab/phase6-novanas-llama8b{,-mxfp4,-mxfp4-a4}.yaml` (the a4 comment now says
  the KV recipe is fp8_e4m3, per the correction).
- Dropped: the base-3B golden fixture with the borrowed Instruct chat template (no reference was ever
  made; superseded by the user's 2026-09-29 decision). No completion-only 3B golden fixture is kept. The
  base-3B side result (GSM8K vs BF16 3B base) is **not run**: `gsm8k-200.jsonl` is chat-only and the base
  checkpoints have no chat template, so it needs a completion-form task set first (open; ask the lead).
  Parked, uncommitted, outside the synced tree: the old 3B README/template and the two 3B lab configs
  (`phase6-novanas-llama-mxfp4-a4.yaml`, `phase6-novanas-llama3b-base.yaml`), and the README-only fixture
  dirs of the two 8B quantized slugs (a README-only dir fails `quant_fixtures_valid`), all in
  `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad/mxfp4/pending-fixtures/`.
  Restore the two 8B READMEs into `tests/golden/<slug>/` together with their `reference.jsonl` and
  `tolerance.json` from fixq. Note: `lab-bench.sh --model llama-mxfp4-a4` still points at the dropped
  3B slug.

## Running (detached; do not start a second copy)

Written 2026-09-29 ~13:25 by the third Task 20 builder. The Mac-side `queue4.sh` died in the 10:50 ssh
outage before any step ran (its `ev3/` holds only "Killed" logs). `phase-6a-quantization` (a0ba309) is
merged (877f394); release `turbine-server`/`turbine-golden` and `kbuild/libturbine_hip.so` rebuilt on novanas.

**queue4n** — one novanas-side detached script (copy of the template `gsm8k_full_run.sh`'s locking):
`/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/q4n/queue4n.sh` (source kept in the lead scratchpad
`mxfp4/queue4n.sh`), log `…/q4n/queue4n.log`, results `…/q4n/<step>.json` (+ `.err`, `.server.log`).
It runs frozen copies of the binaries in `…/q4n/bin/`, so a later remote-cargo sync does not disturb it.
Each step: `flock -x` port18000.lock (vLLM: port18100.lock) → bench.gate → bench.lock, fixture jobs
`pkill -STOP`ped while the GPU runs, native server on GPU 0 (`ROCR_VISIBLE_DEVICES=0`, cores 0-11),
`/ready` wait ≤ 10 min, server always killed, one `<step>: done rc=<rc>` line, last line
`queue4n: ALLDONE rc: …`. Started 13:19 local; queued behind the full-GSM8K FP8 KV requeue (hours).

1. `llama8b-a4` — GSM8K-200, W4A4 8B (`phase6-novanas-llama8b-mxfp4-a4.yaml`) → `q4n/llama8b-a4.json`.
   rc=90 means the server never became ready (tail of `q4n/llama8b-a4.server.log` in the log).
2. `vllm-8b-a4` — vLLM-ROCm k3s Job `turbine-lab-vllm-<q4n/runid>` (the lab-serve template rendered to
   `q4n/vllm-job.yaml`, port 18100, 40 min ready bound, Job always deleted) → `q4n/vllm-8b-a4.json`;
   pod log `q4n/vllm-8b-a4.pod.log`. rc=92 = vLLM never ready: the refusal reason is in the pod log →
   record it in `tests/eval/llama-3.1-8b-instruct-mxfp4-a4/gate.json` like the A16 one.
3. `full-llama8b-mxfp4` then `full-llama8b` — full GSM8K (1,319), MXFP4-A16 8B then BF16 8B, 6 h eval
   timeout each → `q4n/full-llama8b-mxfp4.json`, `q4n/full-llama8b.json`.

Copy the results back with one scp once `ALLDONE` is in the log:
`scp 'piwi@192.168.10.203:/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/q4n/*.json' <dir>`.

**Then (Mac-driven, one at a time, background, only after ALLDONE — they queue on bench.lock otherwise):**
`LABBOOK_SET=phase-6a-quantization scripts/lab-bench.sh --model <m> --label t20 --golden16 --c1` for
`m` = `llama8b`, `llama8b-mxfp4`, `llama8b-mxfp4-a4`. Golden verdicts are informational until the
tolerances are calibrated; the two quantized models have no reference yet, so their golden lines fail.
lab-bench uploads to labbook only when it exits 0 — upload the others by hand (labbook skill).

**Fixtures:** the old `fixq.sh` is gone (`fixq3.log` is stale). The 8B MXFP4 references/spreads now run
inside the lead's `/home/piwi/turbine-ci/scratch/fixtures-r5/fixtures-r5.sh` (log `fixtures-r5.log`;
outputs still in `…/agent-a4784842b25c93376/scratch/fixtures/out/`). At 13:14 its
`mxfp4-llama-3.1-8b-instruct-mxfp4a16-reference` step ended **rc=143 (SIGTERM from outside, 3.5 min into a
24 h timeout)**, so the A16 spread was skipped; the W4A4 reference started next. The A16 reference must be
rerun (lead's pipeline) before the > 0.04 numerics check below can use it.

## How to judge (exact)

- W4A4 GSM8K-200: `turbine-golden eval-compare --baseline tests/eval/llama-3.1-8b-instruct/turbine-bf16.json
--candidate <q4n/llama8b-a4.json> --max-drop 0.04` (4-bit bound; BF16 baseline unless vLLM serves it,
  then vLLM's result is the reference per Q9). Commit as `tests/eval/llama-3.1-8b-instruct-mxfp4-a4/{turbine.json,gate.json[,vllm.json]}`.
- Full GSM8K: commit as `tests/eval/llama-3.1-8b-instruct/turbine-bf16-full.json` and
  `tests/eval/llama-3.1-8b-instruct-mxfp4a16/turbine-full.json`; `eval-compare --max-drop 0.04`.
- **> 0.04 rule (user decision, binding):** if the full-set MXFP4-A16 drop is still > 0.04, do NOT blame the
  format and do NOT change any support status. First look for a Turbine numerics error: once fixq's
  MXFP4-A16 reference exists, `turbine-golden compare` Turbine (MXFP4-A16 server) against it at c1 with the
  BF16 8B spread-based bounds, then `turbine-golden positions --url … --prompt-id <id>` on every failing /
  worst prompt, and the trace tools (`DecoderExecutor::set_trace`, `turbine_model::testing::trace`) if a
  position diverges. Turbine within the reference's own spread = the drop is the format's; outside it = a
  Turbine bug. Report to the lead before ANY support-status change.
- Tolerances (after fixq): per slug from its `spread.json` by the OLMoE method
  (`tests/golden/olmoe-1b-7b-0125-instruct/README.md`); BF16 8B's replaces the provisional values. Then
  rerun `lab-bench --golden16 --c1` for the three 8B models.
- Support rows (`mxfp4`, `mxfp4_a4`) only after the full proof, as a `handoff(support.rs)` commit, and only
  after the lead has the numbers.

## Measured so far

- BF16 8B: GSM8K-200 0.89 (178/200); bench 401.4 tok/s, ITL 31.9 ms, c1 ITL 27.9 ms, golden1 PASS, golden16
  FAIL with the provisional 3B bounds.
- MXFP4-A16 8B: GSM8K-200 0.835 (167/200), drop 0.055 > 0.04 (gate FAILS on 200 items; full set queued).
- W4A4 8B: not yet measured (the earlier attempt failed at startup, fixed by d1de280).

## Open

- Base-3B side result needs a completion-form GSM8K task set (see Done).
