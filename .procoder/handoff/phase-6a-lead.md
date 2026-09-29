# Handoff: Phase 6a lead (rotation at ~270k tokens, 2026-09-29 ~08:40 +04; updated by the second lead ~08:50)

**Update (second lead, 08:50):** merged p6a-kv-t24 (c1ecafd; one "gate misses" entry, 185ccee's text) and the GSM8K
dataset 781edad (d466c0b); d1de280 Quark K/V `layer_quant_config` mirrors ignored (W4A4 8B startup refusal); review
file updated (386898a); fixture queue ordered by `scripts/lab/fixture-order.sh` + `/home/piwi/turbine-ci/fixture.queue`
(9220129: fp8kv → p6a-int4 → llama-3.1-8b → fp8-block → yarn); gate ok 775 on 386898a. queue3.sh stopped (it paused
the fixtures while waiting for bench.lock); its remaining steps went to the Task 20 builder. Dataset worktree removed.
New builders: Task 18 INT4 `a74d419d213cc35b9` (worktree agent-abee0e2542a317f3c, p6a-int4), Task 20 MXFP4
`a49bfb18f85acff9a` (worktree agent-a4784842b25c93376, p6a-mxfp4), both on non-fixture work first. Asked the YaRN
builder whether it is blocked on the fold (its `timeout 21600 flock` expires ~13:18 while waiting). Items 1, 4, 5
below are done or handed out; a19ca024910301566 finished.

Integration worktree `.claude/worktrees/agent-a4b513efedb95892f`, branch `phase-6a-quantization`, tip 4f45232, clean.
Lab runs from the clean detached worktree `agent-a4b513efedb95892f-lab` (lab scripts rsync uncommitted files).
Per-branch detail: `.procoder/handoff/<branch>.md` in each worktree; this file is the lead's overview.

## Integration state

- Merged: everything up to Tasks 1–13, 16, 17, 19, 21 (host), 22, 23, 25, 26, 27; 2116221 (Quark
  `kv_cache_quant_ignored`, verified by the lead: loader + detect test); 33c5d9a (handoffs, TurboQuant S final);
  4f45232 (decisions: full GSM8K for the gate misses).
- Pending merge: `p6a-kv-t24` (tip 244ce98: ca45bd9 reworded kv_gpu FP8 tests, 8fb9e90 GSM8K-200 evals, 0e877d6
  handoff, 185ccee decisions addendum extending the full-GSM8K rule to OLMoE, 244ce98 handoff). Gate ok 774.
  **185ccee re-adds the "gate misses" entry with OLMoE**; integration already has 4f45232's version — on merge,
  keep ONE entry (185ccee's text supersedes 4f45232's; resolve the conflict by hand).
- Not merged: `p6b-groundwork` (waits for 6b, starts from main after 6a closes).

## Running builders (agent IDs for SendMessage)

| Agent             | Branch                               | Task                                                       | Waiting on                                                                                                                                                  |
| ----------------- | ------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| a4baec4689b995379 | p6a-fp8-t15 (or its worktree branch) | 15 fp8_block kernel + loader FP8 layout, quick tier        | its lab `hip_qgemm` (old queued run → `scratchpad/fp8/labtest_block2.txt`), gate, bench.lock                                                                |
| aa94fe3bfacab12b1 | p6a-yarn-t28                         | 28 YaRN golden tail (p16 1.38, p17-long 1.42)              | fold spread `/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-fold.json` (fixture.lock); may stop with a design question (fold vs cos/sin scaling) |
| a19ca024910301566 | p6a-gsm8k-full (worktree branch)     | full GSM8K fixture + generator                             | its gate; generated `scratchpad/gsm8k-full.jsonl` already, commit not yet on `p6a-gsm8k-full`                                                               |
| a67eec8abc8f117eb | p6a-kv-t24-full                      | full-GSM8K FP8 KV gate, Llama + OLMoE (4 runs, sequential) | the dataset commit, then bench/port locks; ~6× a 200-run each                                                                                               |

Finished (don't reuse): ad2d9c8c7cc380c42 (Task 24 wrap-up).

## Background runs (scratchpad = `/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad`)

- fixture.lock queue on novanas (one at a time, hours each): FP8-dynamic `self_spread` (running; then fp8-block
  reference) via `scratchpad/fp8/fixture_chain2.sh`, outputs `/home/piwi/turbine-ci/remote/agent-adb021480bf19bfbe/fixtures/`;
  MXFP4 8B `fixq.sh` (`/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/scratch/fixtures/out/fixq3.log`);
  YaRN fold (above); INT4 `fixtures6.sh` (`/home/piwi/turbine-ci/scratch/p6a-int4/fixtures6.log`);
  FP8 KV golden `fp8kv_fixtures_t24.sh` (`/home/piwi/turbine-ci/remote/agent-ad2d9c8c7cc380c42/fp8kv_fixtures_t24.log`,
  output `/home/piwi/turbine-ci/golden-work/fp8kv/<slug>/`).
- GPU queue: `scratchpad/mxfp4/queue3.sh` (W4A4 8B eval → vLLM 8B-a4 → lab-bench llama8b / -mxfp4 / -mxfp4-a4);
  `scratchpad/int4/vllm_session.sh … gptq` (vLLM GPTQ, log `scratchpad/int4/vllm-session-gptq.log`);
  the old FP8 builder's `lab-test hip_qgemm` (local pid 72411).
- I killed only one stale local ssh (duplicate fp8-block reference with no remote side).

## Remaining tasks, in order

1. Merge `p6a-kv-t24` (see conflict note). Then merge Task 15 / 28 builders' branches when they report; cherry-pick
   their `handoff(<file>)` commits for lead-owned files; lead restores nothing else by hand.
2. Task 15 lead part: if the builder didn't finish the loader FP8 layout for `fp8_block`, do it (decode fallback per
   unsupported shape, `event="fp8_block_decoded"`). Then the fp8_block proof (new builder).
3. Task 14 proof (new builder, after the FP8 spreads land): tolerances, `lab-bench --model llama-fp8 / llama-fp8-tensor
--golden16 --c1`, vLLM (`scratchpad/fp8/vllm_fp8.sh`), eval, soak (ask coordinator).
4. **Task 18 builder (not started):** gated by `fixtures6.sh` (AWQ spread, GPTQ reference + spread) and the vLLM GPTQ
   session. Brief: `.claude/worktrees/agent-abee0e2542a317f3c/.procoder/handoff/p6a-int4.md`.
5. **Task 20 builder (not started):** gated by `fixq.sh` and `queue3.sh`; must run full GSM8K on MXFP4-A16 8B and
   BF16 8B (decision 4f45232) once `gsm8k-full.jsonl` lands; numerics check before any status change. Brief:
   `.claude/worktrees/agent-a4784842b25c93376/.procoder/handoff/p6a-mxfp4.md` (ec9fce4). Note the misnamed eval file there.
6. Task 24 finish: FP8 KV golden fixtures → tolerance → `lab-bench --model llama-fp8kv / olmoe-fp8kv --golden16`;
   full-GSM8K verdicts (a67eec8abc8f117eb); rows only after.
7. Task 28 per builder outcome.
8. Task 21 two-GPU leg and Task 29's two-GPU tier: **blocked on the PSU**.
9. Task 29 exit; merge into local main; 6b from main with `p6b-groundwork` (YELLOW ladder change, its handoff).

## Open questions

- None open with the coordinator right now. Possible soon: YaRN fold placement (from aa94fe3bfacab12b1); OLMoE FP8 KV
  full-GSM8K result / per-layer scale finding; MXFP4-A16 full-GSM8K result.
- Task 15 test tolerance: the old builder widened `qgemm_fp8_block_matches_cpu` for the dequant path; the new builder
  was told to prefer a BF16-rounded CPU reference with a tight bound — check what it did before merging.

## Rules in force

- PSU: ONE GPU-heavy job at a time (bench.lock, `TURBINE_LAB_ONE_GPU_JOB`); no two-GPU runs; crash → wait bounded, don't debug.
- Disk: cleanup from ~100 GB free (209 GB at 07:56): `git worktree remove` merged clean finished trees →
  `TURBINE_PRUNE_IDLE_HOURS=1 scripts/lab-prune.sh --report` → real run only if it lists finished agents' trees only.
- Rotation: one task per builder; replace at ~300k tokens with a handoff; sonnet for mechanical work.
- Never block in the foreground on a lock or a long remote run (coordinator, 2026-09-29; three builders died to
  the 600 s stream watchdog): start them with `run_in_background` (or detached on the host) and get notified. Put
  this line in every builder brief. Builds and gates no longer take bench.lock (f94cbf1: nice 19, cores 12-15).
- No polling (coordinator, 2026-09-29): a builder whose only remaining work is waiting hours for a queued run writes
  the run's paths and how to judge it into its handoff, messages the lead and ends. The lead checks the novanas logs
  cheaply whenever woken and starts a short-lived collector (sonnet unless numerics) once a run has finished. The lead
  keeps no monitor of its own. Told: a4baec4689b995379, aa94fe3bfacab12b1, a74d419d213cc35b9, a49bfb18f85acff9a
  (the coordinator told a67eec8abc8f117eb).
- Ownership: the lead owns the kernel header, `ffi.rs`, ops, registry, turbine-model config/decoder/loader/weights, core
  support/config, `.procoder/`; builders send `handoff(<file>)` commits. No push; merge into local main at 6a close.
- Design questions go to the coordinator via SendMessage; keep working on anything independent.

## Review file

`.procoder/review-2026-09-29.md` is as the previous lead left it (overnight state, through 2116221). Not yet updated
with: the agent loss and rebuild, the new builders, the Task 24 / 18 / 20 numbers in this file, the GSM8K decisions.
Update it at the next clean point.

## Since 09:00 (second lead)

- Merged p6a-fp8-t15 (16b06ef): `turbine_hip_fp8_block` (W8A16, fused WMMA decode, dequant+hipBLASLt prefill), FP8
  layout in the loader with per-stack decode fallback (`for_kernels` / `resolve_for_providers`), the old widened test
  bound removed. Lead-owned handoffs reviewed (weights/mod.rs, fp8.rs, model.rs) before the merge.
- e06afc9: `lab-serve.sh` takes bench.lock itself for its serve Job's life (holder `runs/serve-locks/<run>.sh` on
  novanas, released on Job deletion), `--gpus 2` refused; `bench-lock.sh` exports `TURBINE_BENCH_LOCK_HELD`.
  Builders' worktrees get it when they merge integration.
- Task 15 proof builder `ae29a5cff88bf0bdb` (worktree agent-a4baec4689b995379), gated on the served-bytes check.

## Detached runs to collect (check cheaply when woken; start a short collector once done)

- Full-GSM8K FP8 KV, 4 passes (a67eec8abc8f117eb, ended): handoff `.procoder/handoff/p6a-kv-t24-full.md` on
  `p6a-kv-t24-full` (027803a). Log `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run.log`, done at
  `ALLDONE rc: …`. Until then no remote-cargo / lab-bench / gate from that worktree and don't remove it. Both
  checkpoints seem to lack k_scale/v_scale (scales 1.0): an OLMoE miss points at e4m3 range/saturation first, then
  Turbine vs the emulated-FP8-KV reference; that collector needs the default model.
- Task 15 served bytes: `scratchpad/t15_weight_bytes.txt` (last line `t15-serve: done rc=… run=…`; local pid 55147);
  the Task 15 proof builder judges it (3,607,615,488 B, no `fp8_block_decoded`).


- 10:30: novanas stopped answering ssh (ping ~300 ms); waited, not debugged. Killed the queued T15 served-bytes chain
  (pid 55147): its old bench-lock.sh would have run the T15 worktree's new self-locking lab-serve.sh and deadlocked
  on its own lock; relaunch it with the new scripts. Worktrees without e06afc9 (p6a-mxfp4 with queue4 running,
  p6a-int4, p6a-yarn-t28, p6a-kv-t24-full) must not merge integration while their old-script queues run.

## Rotation of the second lead (~10:45, ~311k tokens) — START HERE

Integration `phase-6a-quantization` tip = this commit (after 7544183), clean, no push. Scratchpad =
`/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad`.

**Gate:** last green = 386898a (775/0). Not yet green since: T15 merge (16b06ef), e06afc9 (lab-serve lock),
f94cbf1 (builds off bench.lock), docs. Clippy passed on f94cbf1 in 19 s without a lock wait; the test step died
twice to novanas ssh timeouts (the host is up since 07:01, no crash; ssh stalls under load ~10:28 and ~10:40).
A gate on 7544183 was running in the background (`scratchpad/gate-7544183.txt`), likely failing on ssh. First
action: rerun `scripts/gate.sh` (run_in_background) once `ssh novanas true` answers.

**Restarts pending (after the gate is green; fresh agents; every brief says: never block in the foreground on a
lock or a long remote run — run_in_background, get notified; no polling; `git status` first):**
- T28 YaRN (dead aa94fe3bfacab12b1): worktree `agent-aa94fe3bfacab12b1`, branch `p6a-yarn-t28` (c06a1c8), no merge
  state; uncommitted: golden.rs, hf_reference.py, yarn_long_prompt.py, yarn_self_spread.py, yarn16 config and
  fixture dir. Its last known work: CPU-provider A/B, transformers-style attention factor (cos/sin × m before BF16)
  vs the fold, on GPU 0 — the result may be in its scratch/logs; the fold placement is a design question to the
  lead/coordinator before changing lead-owned files. Findings so far: p16 tail same on CPU and HIP (kernels ruled
  out), p17-long HIP tail flatter than reference. The fold spread job may be left to time out (builder's call).
- T15 proof (dead ae29a5cff88bf0bdb): worktree `agent-a4baec4689b995379`, branch `p6a-fp8-t15` (78e51fe, has
  e06afc9 but not f94cbf1 — merge integration first), clean. Brief = the one given to ae29a5cff88bf0bdb (in this
  handoff's history): served-bytes check first (below), then GSM8K-200 vs BF16 3B (FP8 bound 0.02), lab-bench
  `--model llama-fp8-block --golden16 --c1`, vLLM fp8-block baseline, labbook, tolerance once the fp8-block
  reference (fixture queue, `scratchpad/fp8/fixture_chain2.sh`) and a spread exist; soak asks the lead; row flip as
  a handoff(support.rs) commit.

**Detached runs / collectors:**
- T15 served bytes: relaunched ~10:40 as `nohup scripts/bench-lock.sh --name port18000 bash scratchpad/t15_serve.sh`
  from the T15 worktree (new lab-serve takes bench.lock itself). Its lock ssh printed a timeout — check
  `scratchpad/t15_serve_lock.txt`; if bench-lock exited, relaunch the same way. Result: `scratchpad/t15_weight_bytes.txt`,
  last line `t15-serve: done rc=… run=…`; pass = weight_bytes 3,607,615,488, no `fp8_block_decoded`.
- T20 queue4 (dead builder ended deliberately; handoff committed 822d8a6 on `p6a-mxfp4`): local `bash queue4.sh`
  (pid 34597, log `scratchpad/mxfp4/queue4.log`, outputs `scratchpad/mxfp4/ev3/`): W4A4 8B GSM8K-200 → vLLM 8B-a4 →
  full GSM8K MXFP4-A16 then BF16 8B → 3 lab-benches. Done at `queue4: done`. Collector (default model if the full-set
  MXFP4-A16 drop > 0.04: numerics check before any status change) follows `.procoder/handoff/p6a-mxfp4.md` "How to
  judge". Plus fixq (fixture queue, rank 3). The T18 AWQ lab-bench (pid 6025 chain, from a74d419d213cc35b9's
  background call) also waits on port18000.
- Full-GSM8K FP8 KV (4 passes): see "Detached runs to collect" above (ALLDONE line; don't touch that worktree).
- T18 INT4 builder a74d419d213cc35b9: still alive (background lab-bench); no report yet.

**Must not merge integration while their old-script queues run** (their old bench-lock.sh doesn't export
TURBINE_BENCH_LOCK_HELD, the new lab-serve.sh would wait on its own caller, and a remote-cargo sync replaces
binaries mid-run): `p6a-mxfp4` (until `queue4: done`), `p6a-kv-t24-full` (until ALLDONE), `p6a-int4` (until its
AWQ lab-bench and anything else it queued have finished; tell a74d419d213cc35b9). `p6a-yarn-t28`: check
for detached runs before merging.


**Check first on novanas (unverified at rotation, ssh down):** the fixture dispatcher's launching ssh session ended
(exit 1). The dispatcher was started with setsid nohup, so it should still run. Verify with
`pgrep -af '[f]ixture-order.sh'` and `tail /home/piwi/turbine-ci/fixture-order.log`. If it is gone, the waiters it
stopped stay in state T: restart it (`setsid nohup bash /home/piwi/turbine-ci/fixture-order.sh >>…/fixture-order.log
2>&1 </dev/null &`), or `kill -CONT` the `flock …/fixture.lock` waiters.

## Rotation 5 lead (11:45–14:20, ~250k tokens) — START HERE

Integration `phase-6a-quantization` tip = this commit, clean, no push. Scratchpad =
`/private/tmp/claude-501/-Users-pascal-Development-Turbine/7482a1b1-2407-47bb-91f2-d826e25c21af/scratchpad`; running
notes `scratchpad/lead/r5-state.md`; builder rules `scratchpad/lead/rules-r5.md` (add to every brief); briefs `lead/brief-*.md`.

**ssh:** novanas via a shared ControlMaster through dgx-spark (`~/.ssh/config`). Never `-o ControlMaster=no`, `-S none`,
`-F /dev/null`; nothing runs on dgx-spark. Drops happen: bounded background retries. Remote start trap: `pgrep -f name`
inside `ssh host '…'` matches its own command line — start detached jobs from a script FILE on novanas.

**Gate:** ok 777 on 807aabc. On a181274 (merges of T15/T20/T28 handoffs, fixtures, yaml, golden.rs test): 776/1, only
`server_cli sigterm_drains_then_cancels` ("bad chunk size line", 16 s) while novanas carried GSM8K + fixtures; no server
code changed since 807aabc. Rerun of that test ×3: `tasks/bwc7h1ugk.output`. Rerun the whole gate on this tip first.
a0ba309 fixed e06afc9's lab-serve `--gpus 2` dry-run break (+ test).

**Merged this rotation:** a0ba309, p6a-int4 (12449c4), decision 807aabc (tolerance floor = max(spread, BF16 bounds);
native novanas measurement OK with a BENCH-equivalent line + labbook; AWQ/fp8-block soak yes after proof), p6a-fp8-t15
(e1b1c00), p6a-mxfp4 (f6756e4), p6a-yarn-t28 (a181274, plan amendment 72c11b6, plan check COMPLETE), p6a-fp8kv-eager.

**Detached on novanas (collect with a short-lived collector when each log says done; no polling):**
| Run | Log / done marker | Then |
| --- | --- | --- |
| Full GSM8K FP8 KV ×4 | `/home/piwi/turbine-ci/remote/agent-a67eec8abc8f117eb/gsm8k_full_run_r5.log`, `ALLDONE rc:` (started 12:57, holds bench.lock pass by pass; everything GPU queues behind it) | verdicts per decisions "gate misses"; OLMoE miss → numerics check vs the NEW emulated reference |
| Fixtures pass 1 | `/home/piwi/turbine-ci/scratch/fixtures-r5/fixtures-r5.log`, `fixtures-r5: ALLDONE` | pass 2 auto (`rerun-after.sh` → `fixtures-r5b.log`) redoes the MXFP4 8B a16 reference+spread I killed at 13:14 |
| YaRN fold spread (read 1) | `…/fixtures-r5/yarn-fold-r5.log`; out `/home/piwi/turbine-ci/remote/agent-ad603c5a7f228a0ed/yarn-fold.json` | send p16 likely/tail to coordinator |
| YaRN p16 A/B (read 2) | `/home/piwi/turbine-ci/remote/agent-aa94fe3bfacab12b1/yarn-tf-r5.log` `yarn-tf-r5: done rc=`, table `yarn-tf-r5.out` | unf≈0.43 & cpu≈1.49 → fold is the cause; both ≈1.49 → trace op by op. Coordinator then asks the user (Q19). |
| FP8 KV emulation regen | `/home/piwi/turbine-ci/scratch/fp8kv-eager/fp8kv_eager_regen.log` `fp8kv-eager: done rc=` (4–6 h CPU, top of fixture.queue) | new OLMoE FP8 KV reference + 8 spreads, Llama 4 eager rows; then Task 24 tolerance + lab-bench fp8kv golden16 |
| T15 fp8-block proof | `/home/piwi/turbine-ci/remote/agent-a4baec4689b995379/t15p/t15_proof_run.log` `t15-proof: done rc=` | judge per `.procoder/handoff/p6a-fp8-t15.md` (3,607,615,488 B, no fp8_block_decoded, c16 ≥ 854.7, GSM8K drop ≤ 0.02); then fixture (rank 4) → lab-bench --golden16 --c1 → tolerance → labbook → soak → support.rs handoff |
| T18 INT4 proof | `/home/piwi/turbine-ci/scratch/p6a-int4/t18-run/run.log` `t18_int4_run: done rc=` | judge per `p6a-int4.md`; BENCH-equivalent line, labbook (vLLM AWQ baseline), support.rs handoff, AWQ 10-min soak |
| T20 queue4n | `/home/piwi/turbine-ci/remote/agent-a4784842b25c93376/q4n/queue4n.log` `queue4n: ALLDONE rc:` | 3× `lab-bench --label t20 --golden16 --c1` per `p6a-mxfp4.md`; A16 full-set drop > 0.04 → numerics check first. BF16 8B spread needs `out/llama-3.1-8b-instruct.reference.jsonl` (ref8b.sh) — check. |

**Void:** anything judged against the old OLMoE FP8 KV reference/spread (it was bitwise BF16). GSM8K FP8 KV numbers stay valid.

**Builders running:** p6b-groundwork (a06fd6f5ca4a9419f: rebase onto integration + ladder from YELLOW, host-only);
completion-form GSM8K sonnet (a423dc6830dee7f9c, worktree agent-p6a-gsm8k-completion; waiting on its gate; propose its AGENTS.md
line). Merge both when they report (p6b-groundwork stays a separate branch until 6b).

**Remaining after collections:** Task 14 proof (FP8 tensor/dynamic: spreads in fixtures-r5 step 4), Task 21 two-GPU leg and
Task 29 two-GPU tier (blocked on the PSU), Task 29 exit, review file refresh (not yet updated with rotation 5), merge into local main.
