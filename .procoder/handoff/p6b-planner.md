# Handoff: p6b-planner (decision "6b Task 6", point 3, A: why the planner recomputes on the GPU server)

Branch `p6b-planner` from `p6b-stack` fdd3ad3. Numbers: `.procoder/perf-log.md`, Phase 6b, "Planner recompute
investigation". Every commit `scripts/gate.sh --base fdd3ad3` → `gate: ok`.

## Commits

- `c420c09` `feat(kv)`: permanent DEBUG events. `kv_plan` gains `prefill_tps`, `l1_latency_s`, `l1_bandwidth_bps`,
  `lower_copy_bytes`, `matched_lossy`, `cost_recompute_s` / `cost_chosen_s` / `cost_reuse_all_s`; new
  `kv_copy_timed` (path, purpose, bytes, `took_s`, `exact`, `sample_s`, the estimate after it) and `kv_prefill_rate`.
  Enable with `logging.level: info,turbine_kv=debug,turbine_server::kv_orchestrator=debug` (needs a config file:
  `lab-serve.sh --set` refuses `=` in values).
- `69769e2` `fix(kv)`: `TransferBackend::took` returns `Option<CopyTime>` (`Exact` | `Within { at_least, at_most }`).
  `CopyStreamBackend` keeps a `CopyClock` per copy: an I/O-pool stage ends exactly, a copy-stream stage ran at least
  until the last poll that saw it running and at most until the poll that saw it done. `None` is bounded by the
  engine's own polls. Contract §26 updated.
- `d9b5f92` `fix(kv)`: a bounded copy is folded in as the path's unloaded cost (`TransferEngine::seed`, the startup
  calibration, else the fallback) clamped to its bounds.
- this handoff and the perf-log section.

## Root cause (evidence: kv_plan / kv_copy_timed on serve runs 1001033606-2395994b, 1001034410-0eb35462)

L1 → L0 copies end on the copy stream and were timed to the next iteration's poll (25–200 ms at c32). Calibration says
1.4 ms per 14.7 MB block (10.45 GB/s); observed 23–149 ms. The latency estimate reached 63 ms (`l0`) / 97 ms (fp8),
every L1 block priced above recomputing 128 tokens (11–18 ms), and since nothing was promoted any more the estimate was
never sampled again (pre-fix fp8: 8 L1 → L0 copies in the whole run, 154 of 156 L1-hit plans recomputed). Ruled out:
lossy penalty, in-flight caps, L0 pressure, lossless tail, session hints.

## Tests

- `turbine-kv transfer::tests::poll_bounded_copies_do_not_drag_the_estimate`: 25 ms polls of 1.5 ms copies keep the
  calibrated cost and the planner retrieves; copies seen running at 50 ms raise it; fast copies afterwards bring it
  back. Red before each fix (25 ms per block; stuck at 25 ms). Mutations, all FAIL: bound → `at_most`; `at_least`
  dropped; prior = current estimate; `seed` not recorded.
- `turbine-server kv_orchestrator::tests::copies_are_timed_to_their_completion_not_to_the_poll`: L0 → L1 and L1 → L0
  on the fake copy stream, polled 600 ms late, report `Within` with `at_least` < 300 ms (red before: `Exact(600 ms)`);
  the I/O-pool legs stay `Exact`. Mutation `Within` → `Exact(poll)` FAILs.

## Open (needs decisions)

1. **Shared waits summed per block.** Every remaining `recompute_cheaper` plan (35–111 per run) ran with the L1 → L0
   latency estimate above 1 ms after a burst that really waited 33–255 ms behind 250–800 MB of pressure demotions on
   the same FIFO copy stream. The planner charges that wait once per block. Options: A) price a plan's path latency
   once (`latency + Σ bytes / bandwidth`; the copies are pipelined); B) observe a burst of copies started together on
   one path as one sample (span over bytes); C) promotions on their own stream, or ahead of demotions in the transfer
   queue; D) leave it.
2. **No re-sampling once a path prices high.** If no plan promotes, the estimate never moves (seen in v2-fp8 after its
   64-block burst: 33 plans with L1 hits, all recomputed). Options: A) let the estimate decay toward the calibrated
   cost with time since the last sample; B) periodic probe copies; C) leave it (1 would remove most triggers).
3. **SURVIVAL confounds the comparison.** The stressed run trips `exhaustion_horizon` → SURVIVAL (10 s of 503) in fp8
   3 of 3 after the fix (once right after a 64-block promotion burst), 0 of 1 before; `l0` 1 of 3 after, 1 of 1
   before. Ratios are over successful requests, so they are not comparable across runs that trip it. Options: A) a
   less stressed variant (e.g. 24 sessions) for A/B; B) investigate whether promotion bursts should count in the
   exhaustion forecast (turbine-reliability, not this builder's area); C) report medians of 3+ runs.
4. **Device-to-host at 1.6 GB/s** (calibration `l0_to_l1_bytes_per_second`) against 10.45 GB/s host-to-device on GPU
   0: demotions are what the promotions queue behind. Not investigated (per-layer segment copies, or the
   compute-ordering wait).
5. `tp_tiers.rs` (`StaticBackend`) still times its leader's own part at the poll (`p.queued.elapsed()`) and reports it
   `Exact`; only multi-rank tiers are affected.

No serve Job of this branch is left running.
