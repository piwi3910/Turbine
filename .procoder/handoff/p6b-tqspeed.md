# Handoff: p6b-tqspeed (decision "6b Task 9: TurboQuant lower-tier gate results" 1 A and 3 A)

Branch `p6b-tqspeed` from `p6b-stack` 3349b4a. Numbers: `.procoder/perf-log.md`, Phase 6b, "TurboQuant promotion path".

## Commits

- ae67d24 `feat(kv)`: the DEBUG event `kv_copy_stages` (each finished copy's stages, `name:ran..seen` ms) and `queued_s` in `kv_copy_timed`.
- e636850 `docs(spec,plan)`: the OLMoE multi-turn variant (decision 3 A) in the spec's Lab scripts and commands and its S-4/S-8 lab
  criterion, and in plan Task 9. `procoder spec check` / `plan check` COMPLETE.
- b6c8e11 `perf(kv)`: a lossy L1 → L0 promotion that the device transcode serves copies the pinned L1 slot straight into a device staging
  slot on the copy stream (`L1PinnedTier::locate_len`, `ShardedL1Tier::locate_len`, `CopyStreamBackend::start_l1_decode`), then decodes,
  without going through the I/O pool. Test: `kv_orchestrator::tests::device_transcode_matches_the_host_codec_through_l1_and_l2`
  (mutation "path disabled" fails it).
- Then: perf log, the decisions entry "P6b: TurboQuant transcode — provider evaluation" (remeasured kernel timings), the
  `TIER_FORMAT_REFUSALS` doc comment, the AGENTS.md support line, and this handoff.

## Profile (before the fix)

A tq4 promotion took 272 ms on average: `io_read_coded` 162 ms (p90 858), `h2d_slot` 46 ms, `decode` 38 ms. It queued on the I/O pool
behind host-codec demotion encodes: 59 of 398 demotions found all 32 device staging slots taken, and a promotion held its slot while it
waited. `tq2` had 5 fallbacks, so its I/O read took 2 ms. The decode kernel is 0.04 ms a block.

## Result

- Llama A/B (medians of 3, GPU 0): `tq4` 0.9047 vs `l0` 0.9016, **holds**. Promotions now take 50–63 ms, no plan recomputes, later-turn
  p99 255 ms.
- OLMoE A/B (spec variant): `tq4` 0.8591 vs `l0` 0.8626, **misses by 0.0035**. Every tq4 run is below every l0 run. This is not
  promotion speed (43–45 ms, no recomputes). tq4 caches about 1k fewer tokens per run at the same prompt totals and has more lookup misses
  (247 vs 231). The shortfall equals Task 9's.
- So **no flip**: `tq4` and `tq2` stay `experimental`. Golden and GSM8K evidence is unchanged (no kernel changed; golden prompts never read
  a lossy block). `kv_gpu lossy_tier_reuse_tq4` gives the same numbers as Task 9.
- Gate after each code commit: `gate: ok` (783 passed).

## Open: decision for the lead / user (OLMoE multi-turn, tq4)

- A) Find the ~8 blocks per run that tq4 loses on OLMoE: per-request `kv_plan` and lookup misses, tq4 against l0, looking for
  lossy-lineage misses (S-3: blocks computed over a promoted lossy copy are chained from its lossy key) or demotions that never land.
  Then fix it and rerun the OLMoE A/B. Recommended: the gap is systematic, and on Llama tq4 now beats l0.
- B) Amend the criterion to "within 0.005 of `l0`" (decision 1 C of Task 9) and flip `tq4`.
- C) A per-model row (Llama `supported`, OLMoE `experimental`). `TIER_FORMAT_REFUSALS` has one row per format, so this changes the schema.

## Also seen

- 46 of 428 tq4 demotions still fall back to the host codec when the slots are busy (their I/O p90 is 529 ms). They no longer delay
  promotions. Faster demotions would need a faster encode (0.70 ms a block, the F64 norm loop) or a demotion that waits for a slot instead
  of falling back.
- The Task 13 builder's `prof.sh` (rocprofv3 around turbine-server) hangs after its own SIGTERM and then holds `bench.lock` indefinitely.
  This happened three times. Each time I sent SIGKILL to that already-SIGTERM'd server, and only to that process, after its profile files
  were written. Its script should `kill -9` after its 60 s wait.

Harness (scratch, not committed): native GPU-0 serve per run under `scripts/bench-lock.sh --name port18000 scripts/bench-lock.sh`, then the
bench client on novanas. No server of this branch is left running.
