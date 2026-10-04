# Handoff: p6b-exit — Phase 6b exit (Task 19) run and recorded; four items for the lead/user

Branch `p6b-exit`, from `p6b-stack` `af9b82af`, merged `p6b-smallfix` (`be192407`: tail-tag
expiry `ecf29422`, tq4 bound split `7ad75b8b`) as `29f4de10` when it landed, then the exit docs
commits on top. Exit tiers run in order, each from a committed tree, all on novanas GPU 0 under
the locks; no lab Job, server or lock of this branch is left. Logs under `target/t19/` on the
workstation; lab artifacts under `target/lab-bench/t19-exit*` and `target/soak/`.

## Tier results

| Tier | Command | Result |
| ---- | ------- | ------ |
| a | `scripts/gate.sh --full` | **ok** — `af9b82af` 955 tests, merge `29f4de10` 958 tests, `failed=0` both |
| b | `scripts/lab-test.sh novanas --tier full` | **1,051 passed / 2 failed** (job turbine-lab-test-1004090558) — see finding 2 |
| c | `scripts/lab-test.sh novanas --gpus 2 --features fault-injection --tier full` (`TURBINE_LAB_ONE_GPU_JOB=0`) | **PASS 27/0** (job turbine-lab-test-1004103638) — Phase 0 inventory, Phase 5 two-GPU lists, fault-injection |
| d | `lab-bench.sh --golden16` × 9 models (label `t19-exit`) | **PASS** llama (853.7 tok/s), olmoe (610.3), llama-fp8 (977.3), llama-fp8-block (1059.7), llama-fp8-tensor (976.5), llama-awq (1258.1), llama-gptq-autoround (1265.9); **FAIL llama-fp8kv** (finding 3), **FAIL olmoe-fp8kv 13/16** (the 6a-exit result behind its `experimental` demotion, decision 2026-09-30 B — unchanged) |
| e | `overload-soak.sh novanas --duration 10m --shared-prefix-share 0.5 --set kv.cpu.max_bytes=4GiB --set kv.nvme.max_bytes=16GiB --set kv.ladder.enabled=true --set kv.ladder.l0=true --set kv.ladder.max_format=tq4` | **7/8 checks × 2 runs; `kv_idle` false both times** (finding 4) — ladder actions 984 / 1,666 (criterion `> 0` met), ITL p99 217 ms vs 186 / 183 calibration, GREEN 0 s into cool-down, 0 client-dropped |
| f | `--support-matrix --output json` / `--check-config` / `track-gate.sh` | matrix + `config ok` (row `experimental amd/*/*/bf16/tq4/none`), `zstd` exit 2 naming the key; `track-gate.sh phase-6b-kv-compression` **GATE PASS**; `phase-5p-serving-efficiency` is not an accepted argument (usage exit 2); `phase-7-model-families` GATE FAIL (finding 1) |

Context: the host lost GPU 1 after its morning boot (node allocatable `amd.com/gpu` = 1); the
lead rebooted it on the user's instruction and both GPUs came back. My first two-GPU attempt and
the interim 1-GPU fault-injection fallback died in that reboot; the two-GPU leg was requeued and
passed. Labbook set `phase-6b-kv-compression` holds the seven passing bench runs.

## Findings for the lead (numbers in perf log 6b "Exit runs" and the spec's S-6/S-11 exit lines)

1. **The phase-7 track-gate rule cannot pass after 6b's accepted end state.**
   `scripts/track-gate.sh phase-7-model-families` fails with "no supported amd row with tq4 or
   tq2 KV" — the L0 `tq4` rows stay `experimental` by user decision (2026-10-02 "1 A then C",
   2026-10-04 A). The umbrella's phase-7 start rule (Task 8) needs amending (e.g. count the
   lower-tier `tq4` `supported` state of `TIER_FORMAT_REFUSALS`, or accept the experimental L0
   rows as the closed marker) before track 2 starts. Note the plan's Task 19 line predates the
   5p reorder and names phase-7; the actual order is 6b → 5p (decision "Phase 5p moves after
   Phase 6"), and 5p is not a track-gate argument at all.
2. **Two stale full-tier-only tests in `turbine-kernels/tests/hip_ops.rs`**:
   `implementations_enumerated` (line ~5183 `paged` table: expects 4 `attention_prefill_paged`
   implementations, the library enumerates 7 — Task 12's `ck_tile_fmha_pagedkv_mixed_staged`,
   `turbine_hip_mixed_staged`, `turbine_hip_mixed` are missing; `paged_decode` should be
   re-checked) and `every_implementation_matches_cpu` (no mixed scenario, so the mixed
   implementations "never ran"; `paged_mixed_matches_cpu` already holds the harness to build
   one). `--tier quick` skips both, so only the full tier sees them; they have been red since
   Task 12 landed. Small test-maintenance task for a builder; the kernels themselves are proven
   by `paged_mixed_matches_cpu` / `paged_mixed_classed_matches_cpu` (green in the same run).
3. **`llama-fp8kv` golden fails deterministically on the exit tree**: c1 15/16 (need 14 — the
   token rule holds on every prompt), c16 the same; p10 exceeds the strict likely bound
   0.4274 vs 0.40 (tail 0.8230 vs 2.44) with an identical 32-token prefix, bit-identical across
   two runs. The 6a exit passed this gate at 829.7 tok/s (2026-09-30), so a 6b commit moved the
   fp8-KV paged path's numerics just past the slug's likely bound — not bisected here.
   Options: A) bisect with `turbine-golden positions --prompt-id p10` across the 6b stack;
   B) recalibrate the slug's tolerance with the self-spread method (user call, as in 6a);
   C) demote the row to `experimental` like OLMoE's. The row is `supported` today while its
   gate fails — that inconsistency should not survive the merge.
4. **The ladder soak fails only `kv_idle`, reproducibly** (runs `target/soak/novanas-20261004T{113806,120705}Z`):
   at the final scrape L0 holds 1,556 / 1,291 blocks and L1 134 / 108, where the Task 18 soak on
   tree 94bd16aa with identical flags drained to idle and passed 8/8. The only code change
   between the passing and failing trees is `p6b-smallfix`'s tail-tag expiry (`ecf29422`) —
   prime suspect, not diagnosed. Every load check is true in both runs.

## Also open (carried)

- **S-10 unimplemented** (the exit's one unticked spec criterion besides the soak/golden misses):
  `/turbine/v1/status` has no `quantization.tier_formats` / `quantization.ladder` and `kernels`
  lists `kv_transcode` only when a tier format selects it. Carried from Task 15 (`p6b-t15` open
  point: the summary is `turbine_model::weights::QuantizationSummary`, lead-owned); it is a
  small server-side reporting change plus the pinned test (`status_reports_quantization`
  extension). Not done here — the exit builder was scoped to docs and runs.
- The ladder-on tail/throughput regression keeps its shape with the L0 step (perf log 6b "The
  L0 ladder on the server"); accepted as opt-in with the documented trade (user decision
  2026-10-04 A) — follow-up, not a blocker.
- AGENTS.md's one-GPU PSU rule still stands in the text; today's two-GPU leg ran with the env
  override after the reboot. Whether the rule can be relaxed is the user's call (the PSU itself
  has not been replaced, as far as the tree records).

## Tree state

Docs commits on `p6b-exit` after the merge: the exit docs pass (`a44f3237`), the S-10 note
(`22ec220c`), and the final evidence commit (perf log "Exit runs", spec S-6/S-11 exit lines,
plan Task 19 boxes, this handoff). The plan's status is `complete`; the spec reads COMPLETE
under `procoder spec check`. The two remaining unticked plan boxes are the final `gate.sh` line
(runs green before the commit) and the commit line itself. Nothing pushed; not merged to main —
the lead merges.
