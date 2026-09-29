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

**Lead note (10:40):** do not merge `phase-6a-quantization` into this worktree, and do not run gate / remote-cargo /
lab-bench from it, until `queue4.log` says `queue4: done`. queue4 runs this worktree's scripts, and integration's
newer `lab-serve.sh` takes bench.lock itself (f94cbf1 / e06afc9); under queue4's older `bench-lock.sh`, which does not
export `TURBINE_BENCH_LOCK_HELD`, that would wait for its own caller. A remote-cargo sync would also replace
the binaries the evals run. The collector judges from the files below; merge afterwards.

Local `scratchpad/mxfp4/queue4.sh` (nohup, log `scratchpad/mxfp4/queue4.log`, one line per step; step
outputs in `scratchpad/mxfp4/ev3/`; scratchpad = the path above). Every step takes `port18000` (or
`port18100`) and then `bench.lock` exclusively and pauses the fixture queue only once it holds the lock.
At 09:20 it waited on `port18000` behind the other builders' runs (FP8 KV full GSM8K ×4). Steps, in order:

1. W4A4 8B GSM8K-200 (`mxeval2.sh`, lab-serve Job) → `ev3/llama8b-a4.json`, log `ev3/llama8b-a4.log`
   (+ `.json.serve.log`).
2. vLLM-ROCm on the W4A4 8B (`mxvllm2.sh`, port 18100) → `ev3/vllm-8b-a4.json` / `ev3/vllm-8b-a4.log`
   (+ `.json.serve.log`: if vLLM refuses the checkpoint, the reason is there — record it in
   `tests/eval/llama-3.1-8b-instruct-mxfp4-a4/gate.json` like the A16 one).
3. Full GSM8K (1,319) MXFP4-A16 8B → `ev3/full-llama8b-mxfp4.json`, then BF16 8B →
   `ev3/full-llama8b.json` (one after the other, 6 h eval timeout each).
4. `lab-bench.sh --model llama8b | llama8b-mxfp4 | llama8b-mxfp4-a4 --label t20 --golden16 --c1`
   (`LABBOOK_SET=phase-6a-quantization`) → `ev3/bench-<m>.log` (BENCH line), results in the worktree's
   `target/lab-bench/t20-<m>/`. Golden verdicts are informational until the tolerances are calibrated;
   the two quantized models have no reference yet, so their golden lines fail. lab-bench uploads to labbook
   only when it exits 0 — upload the others by hand (labbook skill, set `phase-6a-quantization`).

Remote `fixq.sh` (fixture.lock, ranked third): log
`/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/scratch/fixtures/out/fixq3.log`, outputs in
`…/scratch/fixtures/out/`: `llama-3.1-8b-instruct-mxfp4a16.{reference.jsonl,spread.json}`,
`llama-3.1-8b-instruct-mxfp4-a4.{reference.jsonl,spread.json}` (`--act-quant mxfp4`),
`llama-3.1-8b-instruct.spread.json`; last line `fixq: all done`. It deletes each dequantized /dev/shm
copy after its spread. It runs the scripts of rev 2116221: check that the W4A4 reference step did not
trip on the `*k_proj` / `*v_proj` KV entries (`quant_reference.py` has its own Quark parser).

## How to judge (exact)

- W4A4 GSM8K-200: `turbine-golden eval-compare --baseline tests/eval/llama-3.1-8b-instruct/turbine-bf16.json
  --candidate <ev3/llama8b-a4.json> --max-drop 0.04` (4-bit bound; BF16 baseline unless vLLM serves it,
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
