# Handoff: p6b-planner2 (decision "6b planner follow-ups", 1 A, 2 A, 3 A; tp_tiers timing; shared-prefix eval)

Branch `p6b-planner2` from `p6b-stack` dc71f14. Every code commit `scripts/gate.sh --base dc71f14` → `gate: ok`. Numbers: `.procoder/perf-log.md`,
Phase 6b, "Planner follow-ups re-measured".

## Commits

- `70ea0ce` `fix(kv)`: `planner::plan_cost` charges each path's latency once per plan, shared by the plan's blocks on it (each share × the
  block's `1 + lossy_penalty`), bytes per block. Test `planner::tests::path_latency_is_charged_once_per_plan` (mutation: latency per block,
  FAIL). `lossy_penalty_weighs_retrieval` now puts its 1 ms per block in the rate, not the latency.
- `2df40ad` `fix(kv)`: `TransferEngine` keeps promotions and prefetches in their own queue, started first; background copies (`Demote`,
  `Compress`) on the copy stream (L0 ↔ L1) start only while the stream's projected backlog (copies at their paths' unloaded cost, one after
  another) is below the time since the previous pump (cap `BACKGROUND_WINDOW_MAX` 250 ms, debt note); while no background copy runs,
  promotions leave room under `max_inflight_bytes` for the first queued one. Test `transfer::tests::promotions_go_ahead_of_demotions` (red:
  the promotion started 33rd; mutations no window / no reserve / one FIFO queue, each FAIL). kv_sim `ladder_under_pinned_pressure` fixture
  re-blessed (same nine rung changes, the first two earlier; ladder still recomputes fewer tokens, 131,808 vs 147,328).
- `c206b19` `fix(kv)`: `TransferEngine::estimate` decays latency and time per byte that are slower than the unloaded cost toward it, half-life
  `ESTIMATE_RECOVERY_HALF_LIFE` (5 s, debt note), since the path's last sample; a sample folds onto the decayed value. Test
  `transfer::tests::slow_estimates_recover_without_samples` (mutations: stale prior, 2× half-life, each FAIL).
- `ef244fa` `fix(server)`: `tp_tiers::StaticBackend` takes the leader's own `CopyStreamBackend` timing (`CopyTime`) and combines it with the
  slowest worker's `took_ns`. Test `engine::tp_tiers::tests::leader_copy_is_timed_by_its_backend_not_the_poll` (red: `Exact(300 ms)`;
  mutation: backend timing ignored, FAIL).
- `dbcd614` `fix(bench)`: `turbine-golden eval` runs the first `FILLER_HEAD` (2) items alone before the fillers, so the prefix has a hit (reuse
  evidence) before it is pushed out. Test `eval_fillers_push_the_prefix_to_the_lossy_tier_before_the_rest`.
- `0d2ff49` docs (contract §26, spec S-8), `e801c0e` perf log, this handoff.

## Re-measure (Task 4)

32, 24 and 20 sessions (c = sessions, think 1..4 s) trip SURVIVAL (always `GREEN → SURVIVAL` on `exhaustion_horizon`); 16 sessions / c16 is
clean 6 of 6. Medians of 3: `l0` ratio 0.9059, later-turn TTFT p50 80.4 ms, recompute / retrieve 0 / 27, lossy 0; `fp8_e4m3` 0.9027, 78.4 ms,
2 / 21, 42,624 lossy tokens. At this load the arms are equal: L0 holds most of the live histories.

## Shared-prefix eval (Task 5): not gated, the recipe cannot demote the prefix

Measured: the shared system message renders to 2,944 tokens (23 blocks of 128), not ~47; an item's prompt is ~3,150 tokens. BF16-KV runs
(`kv.cpu.format=l0`, `kv.gpu.max_bytes=4GiB` = 292 blocks, `kv.cpu.max_bytes=4GiB`, c16): 8 fillers (serve 1001095429-021df70e) 0.755,
16 fillers (1001095706-39bc19c8) 0.775; in both the prefix never left L0 (0 demotions, 0 L1 lookups, cached 585,856 of 629,014 prompt tokens,
all from L0; 539 capacity drops of filler blocks). FP8-L1 candidate, same settings, 16 fillers, `--min-lossy-cached-ratio 0.5`
(1001095956-0c96bdb6): 0.790 but lossy 0 of 629,014, guard exit 1, so no usable report (the same all-L0 shape). The accuracy spread between the
three runs (0.755–0.790, all exact KV) is batch-composition noise at c16. No `-sp` report is committed (no pair with real lossy reuse exists);
the JSONs are not kept in the tree. Gate status: not run, not failed — the candidate never read a lossy block.

Cause (code, `directory::evictable` + `hierarchy::demote_to` under `Capacity`): capacity demotion copies only blocks with reuse evidence and
walks leaf-first; the prefix's last block has children in L0 (the two head items' own blocks, no evidence), which are never selected under
`Capacity`, so the prefix never becomes a leaf; allocation then drops the fillers (lowest value) and the prefix stays. Pressure reclaim would
walk past the children, but the eval never leaves GREEN. Options (design question, not decided here):

- A) Capacity demotion may copy an evidence block down without freeing its L0 copy while non-evidence children hold it there ("copy ahead");
  allocation later drops the L0 copy (it has a lower-tier copy). Hierarchy change in `turbine-kv` (recommended: it also helps real shared
  system prompts, which have the same shape).
- B) The eval runner tags the two head items with a session (`prompt_cache_key`), so their blocks have evidence and drain leaf-first.
  Runner-only, but the eval then exercises a session path real one-off traffic does not.
- C) A diagnostics hook to demote a prefix (`POST /turbine/v1/kv/demote`), used only by the gate. New API surface.

## Open

- SURVIVAL at 20–32 sessions is the exhaustion forecast (decision point 3 B, reliability code).
- `BACKGROUND_WINDOW_MAX` and `ESTIMATE_RECOVERY_HALF_LIFE` are constants with debt notes; the D2H slowness (point 4) is another builder's.

No serve Job of this branch is left running.
