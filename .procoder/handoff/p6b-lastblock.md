# Handoff: p6b-lastblock (decision "6b: OLMoE tq4 — lossless last block in eviction order", 1 A)

Branch `p6b-lastblock` from `p6b-stack` aa42093. Numbers: `.procoder/perf-log.md`, Phase 6b, "Lossless last block scored like its
history". Not pushed.

## Commits

- 028f465 `fix(kv)`: `score_one` prices a lossless last block's retrieval at the bytes its history would be stored at in the tier below
  (`lossier(src, rung(to))`); its copy codec, memory term and the planner's pricing keep its L0-format size. Tests
  `hierarchy::tests::last_block_is_scored_like_its_history` and `kv_sim lossy_multi_turn_green_matches_l0_reuse` (GREEN, L0 64 / 96 / 128
  / 192 × seeds 1 / 2 / 4). Mutation "price it at `c.to_bytes`": both FAIL. No fixture re-pinned. `docs/extending/eviction-policy.md`
  updated. `scripts/gate.sh --base aa42093`: `gate: ok … passed=793` (the first try failed once on the timing test
  `tiny_server iteration_stage_breakdown`, which passed 3 of 3 alone and on the rerun).
- Docs: perf log entry and this handoff.

## Result

Lab A/B on tree 028f465, GPU 0, medians of 3, arms interleaved:

- Llama: `tq4` 0.9061 ≥ `l0` 0.9052. PASS.
- OLMoE: `tq4` 0.8609 < `l0` 0.8627. FAIL. `tq4`'s uncached tokens did not move from the previous A/B (38.1–38.4k); the `l0` runs came out
  higher this time.
- No run was discarded: none left GREEN in its first seconds. All OLMoE `l0` runs (here and in `p6b-olmoe-tq4`'s A/B) go YELLOW on
  `step_time_drift` 24–25 s after `/ready` (1.55–1.63 against 1.5) and stay there, with about 15 `reclaim` events. No OLMoE `tq4` run
  leaves GREEN. Llama `l0` r2 went YELLOW on drift at 33 s, then back to GREEN on `kv_utilization`.
- Not flipped: `tq4` stays `experimental` in `TIER_FORMAT_REFUSALS`. `support::tests`, `support_startup` and the AGENTS.md support line are
  unchanged.

## Open: decision for the lead / user

The kv_sim mechanism is fixed, but the lab OLMoE residue (about 3 blocks of 128 per run) remains. The arms also run under different
pressure states: `l0` gets YELLOW's proactive demotion, triggered by drift, and `tq4` gets only GREEN's on-demand allocation reclaim.
`after_plan` drops a reclaimed L0 copy with no lower-tier copy.

- A) Rerun the OLMoE A/B once the drift builder's step-time-drift fix lands, so both arms run under the same pressure. Then judge the gate
  again.
- B) Investigate GREEN allocation reclaim on OLMoE `tq4`: count `turbine_kv_drops_total{reason="capacity"}` per arm (add it to the
  harness metrics grep), and check whether capacity copy-ahead at GREEN misses reclaim-order victims. kv_sim does not reproduce this
  yet.
- C) Accept the gap and flip. This relaxes the gate and needs the user.

## Rerun commands

The harness is `/home/piwi/turbine-ci/scratch/p6b-lastblock/ab.sh <abs-outdir> <model>:<fmt>:<run>…`, using this branch's remote dir
`remote/agent-p6b-lastblock` (release build and `kbuild` of 028f465). It logs `transitions=` and `drift=` per run. Run it detached on
novanas, port lock first:

`L=/home/piwi/turbine-ci; nohup setsid flock -x $L/port18000.gate flock -x $L/port18000.lock flock -x $L/bench.gate flock -x $L/bench.lock ./ab.sh /home/piwi/turbine-ci/scratch/p6b-lastblock/ab2 olmoe:l0:1 olmoe:tq4:1 … &`

Never edit `ab.sh` while it runs.

If it holds after A or B: flip `tq4` in `crates/turbine-core/src/support.rs` and add the evidence comment. The evidence is golden c1 / c16
PASS (Task 9), the GSM8K-200 shared-prefix medians (Llama 0.775 vs 0.780, McNemar p 0.23–1.0; OLMoE 0.665 vs 0.635, p 0.24), and the
multi-turn medians. Then update `support::tests` (the `check_tier_format` loop over `tq4`/`tq2`), `support_startup` tests
(`kv.nvme.format tq4` Experimental) and the AGENTS.md support line. Golden and GSM8K stay valid: neither change touches numerics, only
which blocks are kept.
