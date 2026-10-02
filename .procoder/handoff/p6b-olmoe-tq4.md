# Handoff: p6b-olmoe-tq4 (decision "6b: lower-tier tq4 after the promotion fix — OLMoE misses multi-turn by 0.0035", A)

Branch `p6b-olmoe-tq4` from `p6b-stack` 01f2c44, `p6b-stack` 19cc8a4 merged in (lead request). Numbers: `.procoder/perf-log.md`,
Phase 6b, "OLMoE tq4 multi-turn block loss". Builder rotated at the context limit; nothing of this branch runs on novanas.

## Commits

- 018016c `fix(kv)`: a lookup reaches lossy-lineage blocks computed under another request's chain (`directory.rs` lookup: lossy-lineage
  children of the previous block's matching entries are candidates). Tests `hierarchy::tests::lossy_lineage_blocks_survive_an_earlier_switch`
  and `kv_sim lossy_multi_turn_matches_l0_reuse` (the sim driver gains `submit_session`). Mutation "child walk off": both FAIL. Ladder
  fixture re-pinned. `scripts/gate.sh --base 01f2c44`: `gate: ok crates=all passed=906`.
- Merge of `p6b-stack` 19cc8a4 (Task 16's stale `ref_count` and L1 slab fixes). The fixture conflict was resolved by re-pinning: ladder on
  117,056 recomputed tokens, off 147,328.
- Docs: perf log entry and this handoff.

## Findings

1. Lossy-chain misses (fixed). See the perf log. On OLMoE this took the shortfall from 0.0035 to 0.0008.
2. Lossless-tail scoring (open, needs a decision). The tail block is scored at the L0-format bytes it is demoted at (decision B). That is
   about 3.5× its chain's `tq4` retrieval cost. Leaf-first eviction then drains whole histories. In kv_sim at GREEN, `tq4` equals `l0` only
   with `lossless_tail_blocks: 0`, or when `score_one` prices the tail at the tier's rung. The candidate change, not committed:
   in `score_one`, replace `.map_or(self.cfg.block_bytes, |c| c.to_bytes)` with
   `.map_or(self.cfg.block_bytes, |c| self.format_bytes(crate::codec::lossier(c.from, self.rung(to))))`.
   With that change all kv_sim tests pass and the fixture is unchanged. It deviates from the letter of decision B for tail blocks.
3. Task 16's stale `ref_count` plays no part. `ref_count` is read only by the L0 `evictable` check (only for a block with an L0 copy, which
   is synced) and by the ladder (`ladder_victim`, `ladder_sweep`). The A/B runs had the ladder off.
4. Llama: the same two mechanisms apply, but its 585-block L0 holds most histories, so fewer partial demotions happen. In kv_sim the loss
   shrinks to between −32 and 0 tokens at larger L0s. After the fix, the two valid Llama `tq4` runs (0.9063, 0.9067) are at or above every
   `l0` run (median 0.9052).
5. Minor, not lab-relevant: `lossy_entry` creates the promoted entry with `access_count: 0`. Without session hints, a promoted lossy block
   has no reuse evidence, so a YELLOW reclaim leaves it in L0 instead of demoting it. kv_sim without `submit_session` shows `tq4` up to
   −896 tokens. The lab bench sends `--session-hints`.

## Open: decision for the lead / user

OLMoE `tq4` 0.8611 vs `l0` 0.8619 (medians of 3, every `tq4` run below every `l0` run): the gate still fails, so no flip.

- A) Score a lossless-tail block at the tier's rung bytes, as its chain is, for eviction order only. It is still demoted at the L0 format.
  This amends decision B for tail blocks. Then land it test-first: a kv_sim GREEN variant of `lossy_multi_turn_matches_l0_reuse` at
  L0 96, seed 1, fails before (−944) and passes after. Rerun both A/Bs and flip if they hold. Recommended: the tail's value only exists to
  keep its bytes lossless, and pricing it 3.5× above its parents defeats leaf-first ordering.
- B) Leave scoring alone and make leaf-first ordering chain-aware (a parent inherits its leaf's value). This is a broader Phase 4 change.
- C) Accept the 0.0008 residue (within run spread: `l0` runs span 0.0009) and flip now. This relaxes the gate, so it needs the user.

## Next steps

1. Rerun Llama `tq4` r3 (and `l0` r3 if the arms must interleave): `ab.sh <out> llama:tq4:3` on novanas.
   Run it as in `.procoder/perf-log.md`, under `flock -x port18000.gate flock -x port18000.lock flock -x bench.gate flock -x bench.lock`.
   Use that order, port lock first, as `lab-bench.sh` does: the reverse order deadlocks against a waiting lab-bench. The harness is at
   `/home/piwi/turbine-ci/scratch/p6b-olmoe-tq4/ab.sh`, results are in `ab1/`, and the release build and `kbuild` are in this branch's
   remote dir (tree 018016c, without the merge).
2. After the decision: if A, implement it, rebuild (`scripts/remote-cargo.sh build --release -p turbine-server -p turbine-bench` plus the
   kernel cmake as in `lab-bench.sh`), and rerun 3+3 OLMoE and Llama. If it holds, flip `tq4` in `TIER_FORMAT_REFUSALS`
   (`crates/turbine-core/src/support.rs`). Then update `support::tests` (the `check_tier_format` loop expects Experimental), the `tq` case
   in `turbine-server` `support_startup` tests (`kv.nvme.format tq4` Experimental), and the AGENTS.md support line. Golden/GSM8K evidence
   is unaffected: neither change touches the serving path's numerics, only which blocks are reused.
