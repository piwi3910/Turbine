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
- **Full GSM8K (queue4n, all rc=0), committed** `tests/eval/llama-3.1-8b-instruct/turbine-bf16-full.json`
  (1133/1319 = 0.8590) and `tests/eval/llama-3.1-8b-instruct-mxfp4a16/turbine-full.json` (1099/1319 =
  0.8332). `turbine-golden eval-compare --baseline …/turbine-bf16-full.json --candidate
  …/turbine-full.json --max-drop 0.04` → `baseline accuracy 0.8590, candidate accuracy 0.8332, max drop
  0.0400: PASS` (drop 0.0258 ≤ 0.04, lead already read this). Per-item flip count (both files carry
  `results[].id/.correct`, ids match 1:1): both correct 1049, both wrong 136, BF16-only-correct 84,
  MXFP4-A16-only-correct 50 → **134 flips total**. McNemar (continuity-corrected, on the 84/50 discordant
  pairs): χ²=8.1269, **p=0.00436** (exact two-sided binomial sign test agrees: p=0.00419); the flip is
  not noise — MXFP4-A16 loses net 34 items (84 vs 50) at 1319 items, consistent with the measured 0.0258
  accuracy drop, and within the 0.04 gate.
- **W4A4 8B, GSM8K-200 (queue4n, rc=0), committed** `tests/eval/llama-3.1-8b-instruct-mxfp4-a4/turbine.json`
  (108/200 = 0.54). vLLM-ROCm serves this checkpoint (unlike the A16 one, which it refuses) — its own
  GSM8K-200 result is committed alongside as `vllm.json` (147/200 = 0.735). Per Q9 the reference here is
  vLLM's number, not BF16: `eval-compare --baseline vllm.json --candidate turbine.json --max-drop 0.04` →
  `baseline accuracy 0.7350, candidate accuracy 0.5400, max drop 0.0400: FAIL` (drop 0.195). Also FAILs
  against the BF16 8B baseline directly (0.89 → 0.54, drop 0.35), so this is not a reference-choice
  artifact — Turbine's W4A4 GSM8K-200 accuracy is far below both. `gate.json` records both comparisons.
  **This is a real accuracy problem in the W4A4 (`mxfp4_a4`) path, well outside any drop bound** — flag
  it to the lead before any `mxfp4_a4` support-status change; it needs the numerics investigation (the
  `> 0.04` rule's diagnostic steps) before anyone concludes it is the format's own accuracy rather than a
  Turbine bug.

## Running (detached; do not start a second copy)

Written 2026-09-29 ~17:25 by the fourth Task 20 builder (collector). All four queue4n GSM8K passes are
done (`ALLDONE rc: a4=0 vllm=0 full-mxfp4=0 full-bf16=0`); its outputs are copied into `tests/eval/` above
and `queue4n` itself is finished — nothing left running from it.

**t20-bench** — one novanas-side detached script (copy of the queue4n/rules-r5.md locking template),
started 13:25 local: `/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/t20-bench.sh` (source kept in
the lead scratchpad's rules-r5.md style; a copy is also under this branch's
`/private/tmp/…/scratchpad/mxfp4/t20-bench.sh` on the Mac that launched it — not committed, novanas-only).
Log `/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/t20/t20-bench.log`, per-model outputs
`t20/<model>-golden1.txt`, `t20/<model>-golden16.txt`, `t20/<model>-bench.json`,
`t20/<model>-bench-c1.json`, `t20/<model>.server.log` for `model` = `llama8b`, `llama8b-mxfp4`,
`llama8b-mxfp4-a4`. It takes `port18000.lock` → `bench.gate` → `bench.lock` **once**, holds them for all
three passes (so the sequence is not interleaved with another GPU job mid-way), runs frozen copies of the
binaries in `t20/bin/` (a later remote-cargo sync does not disturb it), pauses the CPU fixture queue for
the duration, and per model: serves natively on GPU 0 (`ROCR_VISIBLE_DEVICES=0`, cores 0-11), golden
compare at concurrency 1 and 16 against `tests/golden/<slug>/reference.jsonl` (SKIPPED with a `SKIP no
reference` line for `llama8b-mxfp4` and `llama8b-mxfp4-a4` — no reference exists yet, per Done — so those
two golden lines are expected FAILs/SKIPs, informational only), the fixed throughput bench (16 concurrent,
512-word prompts, 256 tokens, 200 requests) and the c1 latency bench (10 requests, 128 tokens,
`--ignore-eos`), then stops its server before the next model. Last log line `t20-bench: done rc=<rc>`
(0 if every pass's server came up; a pass's golden/bench exit codes never fail the run — only
`SERVER NOT READY` sets that pass's rc to 90 and `overall` to 1). At 17:26 it had already finished the
`llama8b` pass (golden1 PASS, golden16 PASS on the real 8B bounds — the provisional-3B-bounds FAIL noted
above was the earlier task's) and was running its throughput bench; `llama8b-mxfp4` and `llama8b-mxfp4-a4`
still to run.

**How to judge t20-bench once `t20-bench: done rc=` is in the log:** tail the log for the three
`<model>: done rc=` lines and the golden PASS/FAIL lines. `t20/<model>-bench.json` /
`-bench-c1.json` are `turbine-bench --output json` reports (`output_token_throughput`, `itl_ms.p50`
fields — c1 ITL is the Phase 6a target metric, e.g. FP8 ≤ 0.75×, INT4/MXFP4 ≤ 0.6× the BF16 c1 ITL from
this same script's `llama8b` pass, so compare within one `t20-bench` run rather than against the
Measured-so-far numbers above, which came from an earlier, differently-loaded run). Golden16 FAILs for
the two quantized models are expected (no reference/tolerance yet — do not treat as a regression); golden
for `llama8b` should PASS both concurrencies. Copy results back with one scp once the log's last line is
`t20-bench: done rc=`:
`scp 'piwi@192.168.10.203:/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/t20/*.json' <dir>` (and the
`*-golden*.txt` / `*.server.log` files if something needs diagnosing). Do not upload to labbook by hand
until told to (`lab-bench.sh` normally does this itself on exit 0; this script is not `lab-bench.sh`, so
nothing is uploaded automatically — ask the lead whether/how to record these in labbook).

## Open

- Base-3B side result needs a completion-form GSM8K task set (see Done).
- **W4A4 (`mxfp4_a4`) GSM8K-200 accuracy (0.54) is far below both vLLM's own result on the same checkpoint
  (0.735) and BF16 (0.89) — a real accuracy problem, not just missing a bound. Needs the `> 0.04` rule's
  numerics investigation (compare Turbine at c1 against a fixq reference once one exists for this slug,
  `turbine-golden positions` on failing prompts, trace tools) before any `mxfp4_a4` support-status
  decision.** Reported to the lead in this rotation's final message.

## State 2026-09-29 ~18:00 (collector, lead rotation)

**T20 t20-bench verdict.** `t20/t20-bench.log` last line `t20-bench: done rc=0`,
`ALLDONE rc: bf16=0 mxfp4=0 mxfp4a4=0` — all three passes' servers came up and completed.

- **llama8b (BF16 8B reference pass):** golden1 PASS 16/16, golden16 PASS 16/16. `llama8b-bench.json`
  output_token_throughput 400.73 tok/s; `llama8b-bench-c1.json` itl_ms.p50 28.0 ms. This is the reference
  the other two passes are judged against (within this same run, per the handoff's judging note — not the
  differently-loaded `t20-llama8b:3b54bf6` row).
- **llama8b-mxfp4 (MXFP4-A16 8B): PASS on the plan's perf targets.** `llama8b-mxfp4-bench.json`
  output_token_throughput 674.23 tok/s = 674.23/400.73 = **1.683x** BF16 (target ≥ 0.9x: PASS).
  `llama8b-mxfp4-bench-c1.json` itl_ms.p50 11.068 ms = 11.068/28.0 = **0.395x** BF16 (target ≤ 0.6x: PASS).
  Golden SKIPPED (no `tests/golden/llama-3.1-8b-instruct-mxfp4a16/reference.jsonl` yet) — informational,
  not a fail. GSM8K accuracy is judged separately (already committed): full 1319-item set drop 0.0258 ≤
  0.04 PASS (`tests/eval/llama-3.1-8b-instruct-mxfp4a16/turbine-full.json` vs `-bf16-full.json`); the
  200-item quick set drop (0.055) is stale/superseded by the full-set result per the prior rotation's note.
- **llama8b-mxfp4-a4 (W4A4 8B): perf numbers informational only (config-read bug, not a clean measurement)**,
  as flagged by the prior rotation. `llama8b-mxfp4-a4-bench.json` output_token_throughput 617.73 tok/s
  (617.73/400.73 = 1.542x, would nominally pass ≥ 0.9x); `-bench-c1.json` itl_ms.p50 11.466 ms
  (11.466/28.0 = 0.410x, would nominally pass ≤ 0.6x) — **do not treat as a pass**: this checkpoint's
  `config.json` is read wrongly (transformers-5 `rope_parameters`, rope_theta silently fell back to 10000),
  a fix is in progress, so these throughput/latency numbers do not reflect the corrected kernel path.
  Golden SKIPPED (no reference yet). GSM8K-200 for this checkpoint is already committed at 0.54 (108/200),
  far below vLLM's own 0.735 on the same checkpoint and BF16's 0.89 — flagged by the prior rotation as a
  likely real accuracy problem, separate from the config bug, needing the ">0.04 rule" numerics
  investigation before any `mxfp4_a4` support-status decision. Not re-investigated this rotation.

Labbook: submitted three rows to set `phase-6a-quantization` (commit 535d0f4, the branch tip these frozen
binaries were built from): `lab-bench:t20-llama8b:535d0f4` (pass, reference row), `lab-bench:t20-llama8b-mxfp4:535d0f4`
(pass, perf gate), `lab-bench:t20-llama8b-mxfp4-a4:535d0f4` (status `info`, config-bug caveat and the GSM8K
gap noted in the run's notes/conclusion).

**No `support.rs` change made** (per the task: don't touch it for `mxfp4` or `mxfp4_a4`).

Remaining, per the prior rotation's Open section: base-3B completion-form GSM8K side result (not started);
W4A4 config-read fix (rope_parameters) plus a re-run of GSM8K-200/full and t20-bench once fixed; W4A4's
0.54 GSM8K-200 accuracy gap needs the numerics investigation even after the config fix, since it may be a
separate real problem.
