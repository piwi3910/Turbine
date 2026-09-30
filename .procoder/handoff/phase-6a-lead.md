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
code changed since 807aabc. That test alone passed 3/3 (8–10 s) → load flake. Full gate on 8f1bea4+ started at rotation: `scratchpad/gate-r5-final.txt` — read it first.
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

## Rotation 6 lead updates (14:05–) — read with the rotation 5 table above

- Merged: f62fc8a (completion-form GSM8K-200 + AGENTS.md line), ae7fcca (flaky tests `sigterm_drains_then_cancels`,
  `disconnect_releases_kv` robust under load), 090e09c (`turbine-golden eval --concurrency`, report records it, eval-compare
  refuses mixed pairs). 6b docs: bb1aad9 (YELLOW depth = compress only until GREEN; spec S-6, plan T14/T15), review f1c8668.
  p6b-groundwork tip 187c0dc (YELLOW depth policy side; stays separate until 6b; `kv.ladder.*` keys land with 6b Task 2).
- Evals at c16 (lead call, review file): the full-GSM8K driver's and queue4n's `turbine-golden` are now a wrapper adding
  `--concurrency 16` to `eval` (`/home/piwi/turbine-ci/scratch/eval-c16/wrapper.sh`; originals `*.c1`). Llama BF16 full ran at c1 →
  re-run it at c16 after `ALLDONE` before judging Llama FP8 KV (the script's own compare exits 2 on the mix). T15/T18 stay c1.
- Task 14 builder (branch p6a-fp8-t14, brief scratchpad/lead/brief-t14.md): fixtures + FP8-tensor spread (detached, fixture.lock)
  + GPU proof driver (detached, lock chain). Its handoff `.procoder/handoff/p6a-fp8-t14.md` names the done markers.
- Crash policy: decisions "Keep going through crashes (2026-09-29)". Coordinator heartbeat runs r6-check.sh at :17/:47.

## Rotation 6 → 7 handoff (≈15:50) — START HERE next

State notes: `scratchpad/lead/r5-state.md` (lines starting "R6"). Check script: `ssh novanas 'bash -s' < scratchpad/lead/r6-check.sh`
(one ssh; covers the rotation 5 table + T14). Coordinator heartbeat runs it at :17/:47 and wakes the lead.

**Disk (correction 2026-09-29):** novanas kubelet eviction thresholds were lowered to 5 % this morning (configz confirmed):
pods evict at ≈ 45 GB free, not ≈ 134 GB. Lead cleanup trigger stays ≈ 100 GB free (was written as 150 in rotation 5).
Own cleanup allowed: remote `target/` of a removed worktree of ours (`/home/piwi/turbine-ci/remote/agent-<id>/target`).

**Integration `phase-6a-quantization`** tip = this commit (after 8ae7124). Merged in rotation 6: f62fc8a, ae7fcca, 090e09c, f51cc7b
(Task 14 fixtures + drivers). Review file has the lead calls (eval c16, FP8-dynamic weak tail, p05 near-tie, YELLOW-depth reading).

**Detached on novanas, in addition to the rotation 5 table:**
| Run | Done marker | Then |
| --- | --- | --- |
| T14 GPU proof `scratch/p6a-fp8-t14/t14_proof_run.sh` | `t14_proof_run: done rc=` (log in that dir) | judge per `.procoder/handoff/p6a-fp8-t14.md`; p05 excuse only for the exact 9478@23 miss; then labbook, 10-min soak fp8-dynamic, support.rs row flip |
| T14 per-tensor spread `t14_spread.sh` (2nd in fixture.queue) | `t14-spread: done rc=` | tensor `tolerance.json` (same derivation), check `reference-diff.txt`, re-judge saved `capture.jsonl` |
Evals at c16: full-GSM8K driver and queue4n call a wrapper (`/home/piwi/turbine-ci/scratch/eval-c16/wrapper.sh`). Llama BF16 full
ran at c1 → re-run it at c16 after `ALLDONE` before judging Llama FP8 KV. T15/T18 stay c1 (their baselines are c1).

**6b host-only stack (not merged into 6a; rebase onto main after 6a merges, order groundwork → t2 → tq-attn → t3 / t11b):**
- `p6b-groundwork` 187c0dc; `p6b-t2` 40fe8d6 (Task 2 + Task 1 support remainder; registry-driven tier formats, lossy_penalty map);
  `p6b-tq-attn` b1606a9 (cpu::tq_attention, pool page classes); `p6b-stack` df47c83 = t2 + tq-attn (gate --base 187c0dc running →
  `scratchpad/gate-p6b-stack.txt`; check it).
- Builders running: Task 3 on `p6b-t3` (worktree agent-ab443ff21c9b66ddd, brief lead/brief-p6b-t3.md); Task 11 remainder (cpu
  backend TurboQuant L0) on `p6b-t11b` (worktree agent-p6b-tq-attn, brief lead/brief-p6b-t11b.md). Review their handoff(<file>)
  commits (ops, decoder, core are lead-owned). Next: Task 4 (after t3), Task 15 ladder tick + `ladder_under_pinned_pressure`
  (after t3; must fill `demand`, assert stop at GREEN / no tq2 drift), wire `LadderLimits.low_water` from config there.
- Lead calls this rotation: block_formats byte = absolute code (bf16 0, fp8 1, tq4 2, tq2 3; "l0" mapped at the boundary);
  6b plan notes 8ae7124 (progressive gating; fp8 tier experimental at Task 5).

**Remaining for 6a close:** collectors per tables; Task 14 proof; T15/T18/T20/T24/T28 judgements; Task 29 exit on everything
runnable (two-GPU items blocked on the PSU → Unfinished in the review file); merge into local main (no push); start 6b from main.

## Rotation 9 lead (21:50 +04 – 00:20) — START HERE next

State notes: `scratchpad/lead/r5-state.md` (lines "R9"). Check script `scratchpad/lead/r6-check.sh` (one ssh; now also
t28aspread, t28alab, gptqfull, w4a4segv, gpu-queue listing). Builder rules: `lead/rules-r5.md` + `lead/rules-r9-add.md`.

**GPU order is enforced with go-files** `/home/piwi/turbine-ci/gpu-queue/<name>.go` (lead creates, in order). Created:
`w4a4-segv.go` (00:12, running). Next: AWQ soak (Mac-side, builder a81aff2f6d011a14c starts it after `w4a4-segv: done`) →
`gptq-full.go` → `t28a-lab.go` → `w4a4-rerun.go` (after the rope merge; script not written yet — base it on
`scratch/w4a4-numerics/w4a4rope.sh` with the RAW checkpoint and integration binaries; GSM8K-200 c1 vs vLLM 0.735 + bench).

**Integration** `phase-6a-quantization`: merged db6ba1a (p6a-server-flakes), ff051cd (p6a-gptq-numerics) — gate ok 313/0 —,
6c4682c (decision: slow-client timer option A), 1bbbbe7 (p6a-w4a4-numerics: engine-join exit fix; gate ok 314/0 --base ff051cd).

**Pending merges / reviews:**
- `p6a-yarn-t28a` (builder a6caa4f092a670c9c, gate running in its background) → merge first; then `p6a-rope-parameters`
  (021b5f3; config.rs reviewed and ACCEPTED; its gate failures were the load class now fixed) — resolve the config.rs conflict,
  and UN-IGNORE (or delete as duplicate) `crates/turbine-model/tests/config_rope_parameters.rs` from the W4A4 merge.
- `p6b-stack` fast-forwarded to 5d04199 (p6b-t3 reviewed; stays off integration until 6b). `p6b-t11b` builder ab673a1a58605e9eb
  restarted on the 78d3693 wip.

**Answered:** FP8-dynamic full GSM8K drop accepted (A, aa66d8c). GPTQ: full run waits for its go-file;
limit 0.04 vs turbine-bf16-full (gate.json); vLLM refuses the checkpoint, so a miss → coordinator (lead proposed a better
checkpoint, damp 0.01, on both engines).

**FP8 row flip (user decision A, aa66d8c) — NOT done yet, one catch:** the support matrix has a single `fp8` weight column
(`WeightFormatColumn::Fp8`, per-tensor OR per-channel/dynamic), so flipping `gfx1201_quant_row(Fp8)` to `supported` also
covers FP8 per-tensor checkpoints, whose golden is still unjudged (T14 per-tensor spread done → set its tolerance.json,
re-judge the saved `capture.jsonl`, per `.procoder/handoff/p6a-fp8-t14.md`). Plan: judge the per-tensor golden first (small
collector, sonnet), then flip the Fp8 row in support.rs (replace `gfx1201_quant_row(Fp8)` with an explicit `Supported` row + a
comment citing both decisions; update any support test that expects fp8 experimental), gate, commit. If per-tensor fails,
ask the coordinator (split the column, or keep fp8 experimental). Coordinator told of the catch at 00:25.

**Small follow-ups to hand out:** (1) join the TP worker-rank thread at exit like the engine thread (10 s bound, timeout
event, host test; coordinator-approved); (2) remote-cargo core pinning option 1 (tests on 8-11 when bench.lock is free, 12-15
otherwise) — must never overlap a bench that starts mid-gate; find out what `bench.gate` is for first; (3) the review file is
not updated for rotations 8–9.

## Rotation 10 lead (23:45 +04 – 01:15) — START HERE next

State notes: `scratchpad/lead/r5-state.md` (lines "R10"). Check script `scratchpad/lead/r6-check.sh` (add: `tail -3
scratchpad/fp8-soak/wait_and_soak.log` on the Mac). Builder rules: `lead/rules-r5.md` + `lead/rules-r9-add.md`.

**Done this rotation (integration `phase-6a-quantization`):** 5803de9 fp8 weights `supported` on gfx1201 Llama (FP8-dynamic
decision A aa66d8c; per-tensor golden judged from the saved T14 runs against its new self-spread tolerance likely 1.36 /
tail 2.99 → 16/16 c1 and c16; `tests/golden/llama-3.2-3b-instruct-fp8/{tolerance.json,README.md}`), gate ok 788/0.
5319ecc handoff note. 3137c14 merge p6a-int4 (awq_int4 `supported` after the AWQ 10-min soak PASS; support.rs conflict
resolved: fp8 + awq_int4 rows, `baseline_rows_present` admits exactly those two non-BF16 rows), gate ok 788/0.
a5e77bc + 06a880f decisions: GPTQ full GSM8K 0.7369 vs 0.7801 (drop 0.0432 > 0.04; McNemar p = 3.7e-5; overshoot p ≈
0.38) → user B, then C-then-A (see below). Integration gate on 1bbbbe7 was ok 314/0. W4A4 segv repro: new build 8/8
rc=0 stopped=1 (old build 0/8 segv too — the repro never reproduced the crash; the join fix stands on the host test).

**6b (`p6b-stack`, worktree agent-p6b-t2):** fast-forwarded to 45d2e2c (p6b-t11b: TurboQuant L0 on cpu, gate --full 808/0),
then e58f715 (contract §26 TQ names, `docs/extending/kv-format.md` TQ-in-L0 section, plan: block-table `(BlockId, format)`
and class-page addressing moved from Task 11 to Task 17; `docs_extending` ok). Test-bound question accepted (seed mutation
must fail it).

**Running builders:** T28a a6caa4f092a670c9c (gate, then GPU proof on t28a-lab.go); 6b Task 4 adeb53d9bf3bd41d9 (branch
p6b-t4, worktree agent-p6b-t4, gate running); 6b Task 15 a1246e2a457fb8c0b (branch p6b-t15, worktree agent-p6b-t15). Both 6b
builders are from e58f715 and share hierarchy.rs (briefs split the areas) — merge t4 then t15 into p6b-stack, expect a
hierarchy.rs conflict. Briefs `lead/brief-p6b-t4.md`, `lead/brief-p6b-t15.md`.

**GPU queue now:** fp8 10-min soak RUNNING (Mac-side detached waiter pid 30476, `scratchpad/fp8-soak/wait_and_soak.sh`, log
`wait_and_soak.log`, snapshot worktree `.claude/worktrees/lead-soak-fp8` @3137c14, model fp8-dynamic). It touches
`t28a-lab.go` itself when the soak ends, then logs `fp8-soak: done rc=<rc>`. Judge `verdict.json` in the newest
`target/soak/novanas-*` of the snapshot worktree; FAIL → revert the fp8 row to `experimental` (coordinator), gate, report.
Then remove the snapshot worktree. Then: t28a-lab.go (auto) → w4a4-rerun.go (after the rope merge; script not written:
base it on `scratch/w4a4-numerics/w4a4rope.sh`, RAW checkpoint, integration binaries, GSM8K-200 c1 vs vLLM 0.735 + bench)
→ fp8_block jobs (golden fixture, golden16, own 10-min soak; the ef09f16 flip waits for them) → GPTQ runs below.
Old go-files `gptq-full.go`, `w4a4-segv.go` are spent (may be removed).

**GPTQ (user decision C then A, 06a880f), not started — needs a builder (opus for the quantization):**
1. C, early data point only (label it AutoRound; does not decide the row): `kaitchup/Llama-3.2-3B-Instruct-AutoRoundGPTQ-4bit`
   @ `e11f15d2291d8c343a4de84d6bb16ebf7c871dfc` → `hf download … --revision <sha> --local-dir /home/piwi/turbine-models/
   llama-3.2-3b-instruct-autoround-gptq` on novanas (token stays there); full GSM8K c16 with the gptq-full driver
   (`/home/piwi/turbine-ci/remote/agent-p6a-gptq-numerics/gptq-full/`, a go-file of its own); may run earlier if a slot fits.
2. A: llm-compressor GPTQ W4A16 sym g128 damp 0.01 no act-order, ≈ 512 calibration samples, from
   `/home/piwi/turbine-models/llama-3.2-3b-instruct`; Python (uv) at fixture time on novanas; calibration is its own queued
   GPU job; output compressed-tensors pack-quantized (Turbine `ct_pack_int4` → `gptq_int4` row). Full GSM8K c16 on Turbine
   and on vLLM-ROCm if it loads; re-judge against 0.04 vs `tests/eval/llama-3.2-3b-instruct/turbine-bf16-full.json`
   (paired McNemar + CI as in the decision). gptq_int4 stays `experimental` until then. The first full run's output is in
   `scratchpad/gptq-full/`.

**Small follow-ups (no slot yet):** TP worker-rank thread join at exit (10 s bound, timeout event, host test); remote-cargo
core pinning option 1; AGENTS.md prose "Phase 6 quantized formats unsupported on amd" is stale (fix at 6a close or next
AGENTS touch); review file not updated for rotations 8–10. Merge order unchanged: T28a, then p6a-rope-parameters (un-ignore
`crates/turbine-model/tests/config_rope_parameters.rs`), then the formal W4A4 rerun (mxfp4_a4 experimental until then).

## Rotation 11 lead (00:59 – 02:50 +04) — START HERE next

State notes: `scratchpad/lead/r5-state.md` (lines "R11"). Check script `scratchpad/lead/r6-check.sh`. Builder rules:
`lead/rules-r5.md` + `lead/rules-r9-add.md` (PSU rule CHANGED, see below). Briefs this rotation: `lead/brief-r11-*.md`,
`lead/brief-p6b-t4-regress.md`.

**User instructions (2026-09-30, via coordinator), binding until 6a closes:** finish Phase 6a, then PAUSE. 6b is frozen: no
6b builders, no t4/t15 merge, p6b-stack stays at 5050a19 with its notes. At most 2 builders, sonnet for mechanical work.
PSU is NOT a blocker: two-GPU items (Task 21 leg `scripts/lab-cluster.sh --bench-lock tp2-novanas`, Task 29
`lab-test.sh novanas --gpus 2 --features fault-injection --tier full`, anything else deferred for two GPUs) run with
`TURBINE_LAB_ONE_GPU_JOB=0`; if novanas freezes the user reboots, requeue, don't debug. Single-GPU jobs stay one at a time.
Record nothing as deferred-by-PSU. End: merge `phase-6a-quantization` into local `main` (no push), stop all builders and
detached jobs, no novanas job or go-file left, final handoff, report to the coordinator, stop.

**Done this rotation (integration):** 9a94c46 merge p6a-yarn-t28a (Task 28a: ABI v2.10 rope attn_factor, yarn16 tolerance
from the no-fold spread; GPU proof rc=0, golden 17/17 c1/c16, BF16 16/16); merge p6a-rope-parameters; ff804c4 un-ignore
`config_rope_parameters.rs` + plan 28a nit (shim.rs + server startup check). Gate --base c144abf on ff804c4: ok 477/0.
FP8 (per-tensor/dynamic) 10-min soak PASS 8/8 (verdict in `scratchpad/fp8-soak/novanas-20260929T204133Z/`): fp8 row stays
supported. The soak had hung 19 min: lab-serve.sh's `kubectl logs -f` survives under ssh ControlMaster → fix is on
p6a-fp8-t15 (5c6df91 `fix(lab): lab-serve stops its remote log stream`), merges with that branch.

**6b (frozen):** p6b-stack 99dcc4a + 5050a19: user decisions copy_bytes (A), lossy chain rule (builder design, S-3 amended),
ladder step-up only at GREEN (option A, S-6 + AC amended); contract §26 Task 4/15 names; plan Task 15 merge notes (reconcile
Task 15's transfer.rs logical-size observe with copy_bytes; GREEN check; regen `ladder_expected_rungs.json`; assert no
step-up while not GREEN). p6b-t4: ef69456 + bd8bf63 regression fix (commit_progress publish rule on copyless entries) +
3662c44 handoff, gate ok 813/0, kv_sim back to 0.78. p6b-t15: 940b871 + ad006d5 (gate 693/0). Open 6b question for later:
keying recomputed blocks into copyless parent entries (reuse gain, separate Phase 4 behaviour change, not adopted).

**6a results this rotation:**
- W4A4 formal rerun (branch p6a-w4a4-rerun 1f48941, raw checkpoint, integration binaries): GSM8K-200 c1 0.74 vs vLLM 0.735
  → accuracy PASS; c16 631.9 tok/s. Log `scratch/w4a4-rerun/w4a4-rerun.log`; judge the shutdown-segfault line per
  `.procoder/handoff/p6a-w4a4-rerun.md`. Still needed for the mxfp4_a4 flip (spec S-11): an Instruct W4A4 golden
  reference + spread (CPU fixture, check reference-side rope handling — the r8 one was removed for the transformers-5
  rope_parameters bug), golden c1/c16, soak. Needs a builder.
- fp8_block (branch p6a-fp8-t15, builder retired at ~298k; handoff Rotation 11 section): bench job rc=1 only because golden
  could not run (reference.jsonl not yet in the tree): c16 1055 tok/s (200 ok), c1 112.7 tok/s ITL p50 8.43 ms. The golden
  fixture waits on its self-spread (CPU, fixture.queue ranked before `llama-3\.1-8b`); then a golden c1/c16 GPU rerun.
  Soak `fp8block-soak.go` released 22:45Z, Mac waiter `scratchpad/fp8block-soak/wait_and_soak.log`, marker
  `fp8block-soak: done rc=`, snapshot worktree r11-soak-fp8block (remove after). Flip ef09f16 stays the last support.rs
  commit; merge p6a-fp8-t15 after golden + soak pass.
- GPTQ (branch p6a-gptq-numerics; handoff "Rotation 11: C then A"): AutoRound run 1 failed (circuit latency_drift under
  host load) → drivers now `--set reliability.circuit.latency_drift_open=100` with transitions logged (c0253e9, accepted).
  Calibration failed: llmcompressor's fused Triton GPTQ kernel does not compile on ROCm Triton. Fix builder
  a25695ab898be5398 (sonnet, `lead/brief-r11-gptq-calib.md`) → new go-file `gptq-calib2.go`, marker `gptq-calib2: done rc=`.
  Then release `gptq-own.go` (driver pid 1377410 waiting; markers `gptq-own: done rc=`, `gptq-own-vllm: done rc=`).
  Waiter `/home/piwi/turbine-ci/gpu-queue-seq-r11b.sh` (pid 1386054) releases `gptq-autoround2.go` after gptq-own finishes.
  Judge with `scripts/eval/paired_compare.py` vs `turbine-bf16-full.json`; gptq_int4 stays experimental until then.
- CPU fixtures: pass 2 `scratch/fixtures-r5/fixtures-r5c.sh` (YaRN section removed — superseded; pass 1 had re-entered it,
  killed), log `fixtures-r5c.log`, marker `fixtures-r5: ALLDONE`: MXFP4-A16 8B reference started 21:40Z, then its spread →
  then MXFP4-A16 golden c1/c16 GPU job → flip. fixture-order.sh (pid 41159) orders fixture.lock via `fixture.queue`.

**GPU queue now:** fp8block-soak (released) → gptq-calib2 → gptq-own (+vLLM) → gptq-autoround2 (auto) → fp8_block golden
rerun (after its fixture) → MXFP4-A16 golden (after its fixture) → W4A4 golden (after its fixture) → T28a leftovers
(`lab-test --tier quick`, `lab-bench --golden16` llama-yarn16 + llama) → Task 21 two-GPU leg → Task 29 exit (gate --full,
lab-test full + two-GPU tier, lab-bench --golden16 per proof model, soaks per newly supported checkpoint, support matrix,
docs/AGENTS/perf-log, `docs: phase 6a exit`) → merge into local main.

**Small follow-ups (no builder yet):** TP worker-rank thread join (10 s bound, timeout event, host test); INFO
`event="rope_config"` (theta, type, factor) + status field (config.rs, lead-owned); AGENTS.md "Phase 6 quantized formats
unsupported on amd" line stale; review file (rotations 8–11); core pinning in remote-cargo.sh optional.

**Builders running:** a25695ab898be5398 (GPTQ calib fix). fp8_block builder a156d1c78ad466746 finishing its gate/handoff —
retire it (don't resume). One slot free.

## Rotation 12 lead (02:50 – 11:30 +04) — START HERE next

State notes: `scratchpad/lead/r5-state.md` (lines "R12"). Check: `scratchpad/lead/r6-check.sh` + `lead/r12-extra.sh`
(`( cat lead/r6-check.sh; cat lead/r12-extra.sh ) | ssh novanas 'bash -s'`). Builder rules: every brief points to
`lead/brief-r12-common.md` (includes NO local cargo on the Mac — a builder breached it; its target/ and every worktree's
target/debug were deleted). Briefs this rotation: `lead/brief-r12-*.md`.

**Integration** `phase-6a-quantization` tip 96d3b76: 0f3b312 merge p6a-followups (d2033fd TP worker-thread join, 10 s
shared budget, `tp_worker_join_timeout`; e5c28fa INFO `event="rope_config"` + `status.model.rope`, config.rs RopeSummary
reviewed OK; gate 800/0 on base aecb157), 96d3b76 decision MXFP4 p05 (option A).

**Done / verdicts:**
- fp8_block soak PASS 8/8 (`scratchpad/fp8block-soak/novanas-20260929T224540Z/`). fp8_block self-spread (all 8 variants)
  rc=0 after a relaunch (its script had a Mac log path): max likely 0.1829, tail 0.3107.
- r12gpu chain on 0f3b312: lab-test --tier quick PASS; `BENCH r12-yarn16` golden1/16 PASS 845.8 tok/s; `BENCH r12-llama`
  golden1/16 PASS 854.2 tok/s (labbook recorded); **Task 21 two-GPU leg tp2-novanas PASS**. T28a leftovers closed.
- GPTQ own checkpoint (calib3 rc=0 after two ROCm-Triton fixes 1608189 / 2c3fbb7): Turbine 988/1319 = 0.7491, drop 0.0311
  ≤ 0.04 PASS; vLLM same checkpoint 0.7491; AutoRound2 0.7566 (drop 0.0235, data point only).
- Disk: 13 merged worktrees removed, lab-prune 94 GB → novanas ~165–178 GB free.
- MXFP4-A16 8B golden (fixture cb4bd66): 15/16, p05 pos 5 decode-only knife edge (likely 0.3066, tail 6.0156), no
  kernel bug (investigation b62373f/579f746/9a35f21). User decision A: incremental spread. **p05-only result is in:
  transformers' incremental variants do NOT flip p05** (bf16-eager-inc likely 0.0520 tail 0.1985; fp32 sdpa/eager-inc
  0.0407 / 0.4003; no missing). Per the decision: mxfp4 stays experimental, numbers go to the coordinator (sent with this
  handoff). The 16-prompt incremental run continues (`scratch/mxfp4-inc/run.log`, marker `mxfp4-inc: done rc=`,
  `spread-inc.json`) — informational now.

**Builders running (2 = limit):**
- fp8_block completion a3854f9c5d591ea48 (sonnet, `lead/brief-r12-fp8block-finish.md`, worktree agent-a4baec4689b995379,
  p6a-fp8-t15): merge integration, fixture + tolerance, `lab-bench --model llama-fp8-block --golden16`, then "ready to
  merge". Lead merges p6a-fp8-t15 (flip last).
- gptq_int4 proof a2b56fef5216b94f8 (sonnet, `lead/brief-r12-gptq-proof.md`, worktree agent-p6a-gptq-numerics): eval
  files, own-checkpoint golden fixture (8 variants, queue line `gptq-own-fixture` in novanas fixture.queue), lab-bench
  `--model llama-gptq-own --golden16`, vLLM tok/s, soak, flip gptq_int4 last. **Lead amends spec S-11 at merge** (proof
  checkpoint = own llm-compressor checkpoint; told the coordinator).

**Next builder (brief ready, not started — slot limit):** W4A4 8B reference + spread on **cpuhost** (`lead/brief-r12-
w4a4-cpuhost.md`; user-lent TrueNAS box, rules in the brief). The novanas W4A4 job was killed and removed from the queue.

**Remaining 6a after these:** mxfp4 (coordinator answer on p05 numbers); W4A4 fixture → golden → soak → flip mxfp4_a4;
merge p6a-mxfp4-golden (fixture cb4bd66, df8d788, d64f0cc, trace diagnostic) — mxfp4 row not flipped; AGENTS.md stale
Phase 6 line; review file (rotations 8–12; add: tiny_server harness defaults kv.cpu.max_bytes 64 GiB); Task 29 exit:
gate --full, lab-test full, `TURBINE_LAB_ONE_GPU_JOB=0 scripts/lab-test.sh novanas --gpus 2 --features fault-injection
--tier full`, golden16 per proof model, soaks, support matrix, docs; merge into local main; stop everything; final
handoff. Go-files: none pending (spent ones removed). Mac waiter and novanas sequencers r12/r12c ended.
