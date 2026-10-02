# Handoff: p6b-ladderperf (decision "6b Task 16: … ladder-on throughput regression", 2 A), paused

Branch `p6b-ladderperf` from `p6b-stack` 202bd4a. Paused by the lead (user pause) after the root cause and a red test. No lab
run was started, and no Job, server or lock of this branch is left (`kubectl -n turbine-ci get jobs`: none). The fix is written but
not gate-checked, so it is not committed. It is saved as `.procoder/handoff/p6b-ladderperf.patch` (`git apply` on 202bd4a; also `git stash list` in this worktree).

## Root cause (from the Task 16 run data, `target/ladderperf/t16/` = novanas `remote/agent-p6b-t16/runs/`)

When a slab tier's rung changes, new demotions into it stop. The ladder steps L1's rung `l0 → fp8_e4m3` (`fill_high_water`, about 3 s after
YELLOW, in every on-run). From then on, every new demotion into L1 is encoded at `fp8_e4m3` (`copy_codec` → `rung(to)`). That needs an
`fp8_e4m3`-size slot. L1's two 1 GiB slabs hold `l0` slots, and `take_slot` reformats a slab only once it is empty, which never happens
under this load. So `put` ends `Full`, and `on_copy_failed` (Demote) just drops the copy. The L0 block stays, L0 cannot drain, and the
server stays at ORANGE: batch growth 0, prefill budget 0.5, `expensive_queued` admission. That throttle is where the tok/s and TTFT
p99 losses come from. L2 has the same problem for L1 → L2 spills at its `fp8_e4m3` rung.

`make_room` counts free room in bytes (`unit` = rung size), so it sees room and does not spill or evict either. The failures are
invisible: `stats.transfer_errors` only, and a `debug` `kv_transfer_failed`.

Evidence (mt2-on-r1 against mt2-off-r1):

- At the end, L1 holds 137 copies, all `l0`, with 9 `l0` slots free. Its rung is `fp8_e4m3`, and no `fp8_e4m3` copy was ever stored.
- L0 is at 553 of 585 blocks and ORANGE, against 390 of 585 and YELLOW in the off-run.
- ORANGE ran 11:07:37 → 11:08:27 with the reclaim asking for about 1.2 GB each tick, and 1,323 `reclaim` events against 111.
- L0 → L1 demotions were 726 against 1,567.

The better cached ratio and fewer recomputed tokens in r1 and r3 are mostly blocks stuck in L0, not ladder gains: in those runs every
L1/L2 rewrite ended `no_room`, and there were 0 lossy cached tokens.

Other hypotheses, not measured further:

- Lane or copy-engine load is small: 48 rewrites per run in r1 and r3.
- Rewrite bytes are not counted against caps.
- Planner pricing does not explain fewer demotions.
- The ORANGE throttle is the cost, but it is a consequence of this defect.

Red test: `hierarchy::tests::a_demotion_without_a_slot_of_the_rung_stores_at_the_tier_format`. It uses a real `L1PinnedTier`: two
4-block slabs, 5 other `l0` copies, rung set to `fp8_e4m3`. Before the fix it fails with `transfer_errors` 2.

## The fix in the patch (option A below; my call as a defect fix, the lead may prefer otherwise)

- New required `KvTier::free_slots(format, bytes) -> u64`: how many copies the tier can store now without evicting.
  - L1: free slots of that size, plus new slabs below the limit (not under host RED), plus empty other-size slabs at the limit.
  - Sharded L1: the minimum over its shards.
  - L2: free slots of that format that fit, plus new slabs, plus empty other-format slabs at the limit.
  - `MemTier`, L0 and `MirroredTier`: `u64::MAX` (`MirroredTier` delegates).
- `copy_codec` takes the rung only while `rung_has_slot` (free slots > demotions in flight into the tier at that size), else the tier's
  own format. It is never lossier than the rung.
- `rig_tiers` test rig.
- New `LadderReason::RungNoSlot` (`rung_no_slot`), counted when a demotion is stored at the tier's format instead of the rung.

## Exact next steps

1. With the patch, `scripts/remote-cargo.sh test -p turbine-kv --lib` gives 92 passed, 2 failed:
   - `metrics::tests::ladder_families` fails as expected: its reason list needs `rung_no_slot`.
   - The regression test still fails. Its panic message was cut off by `tail` at the pause, so run it alone and diagnose. Look first at:
     - the `demotions == 3` / `used_bytes == 8 * bb` expectations, given the units arithmetic in `make_room` (l0 copies count 2 `fp8` units);
     - whether `demote_to(Pressure)` goes through `copy_codec` before `make_room` sizes the batch.
2. Mutation-check:
   - `copy_codec` ignores `rung_has_slot`;
   - the L1 `free_slots` counts other-size free slots;
   - the in-flight subtraction is dropped.
3. Update the call sites and docs:
   - `turbine-api` `tests/api.rs` `kv_metrics_bounded` (new reason);
   - the `metrics.rs` `LadderReason` doc;
   - the spec S-6 line "new demotions into a tier take its current rung" (add "while the tier has a slot of its size, else its own
     format", plus an edge case);
   - the contract ladder entry.
     Then `scripts/gate.sh`, commit, and delete the remote `target/debug`.
4. Lab A/B, medians of 3 on GPU 0, with the t16 harness (`remote/agent-p6b-t16/h.sh … mt`). Use the default
   `kv.transfer.promotion_copy` (`sdma` on 202bd4a, unless the copy-kernel builder has flipped it), run under the port lock and then
   the bench lock. Expect the ladder arm's demotions, ORANGE share and tok/s to come close to the off arm.
5. Design question for the user (the patch does A only):
   - A) Fall back to the tier format (the patch). Demotions flow, but L1 never actually holds compressed copies while its slabs
     stay non-empty.
   - B) Also free a slab for the new size: spill or evict the least-valuable slab's copies so it reformats. This is slab-by-slab
     work, similar to decision 3 C.
   - C) Smaller L1 slabs, so slabs empty and reformat naturally.

   Recommendation: A now, then measure whether B or C is worth it.
