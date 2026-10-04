# Handoff: p5p-t6 (5p plan Tasks 5 and 6 — config keys, token-granular prefix match)

Branch `p5p-t6` from `p5p-t1t3` f5c42439. Two commits, both gate-clean (whole workspace,
970 tests, remote `target/debug` deleted after each):

- 4e44d0ac `feat(core): phase 5p configuration keys` (plan Task 5)
- 555671c5 `feat(kv): token-granular match of the last prefix block` (plan Task 6)

Host-only: no lab runs, no novanas jobs (the lab A/B lands with Tasks 7–8, recorded in
`.procoder/perf-log.md` under the Phase 5p section).

## The match design (S-4/S-5), five lines

1. **Partial entries**: L0-only `KvBlock`s (`KvBlock::is_partial`: fewer tokens than the
   format's `block_tokens`) keyed like full blocks (`block_key(ns, parent, tail tokens)`),
   chained from the last full block (parent `None` for a sequence shorter than one block).
   `KvDirectory::insert_partial` bounds them at `PARTIAL_PER_PARENT` = 8 per parent and at
   `max_entries`, evicting the least recently used; the evicted entries come back through
   `KvDirectory::evicted_partials()` so the caller frees their L0 slots.
2. **lookup** keeps its signature; after the full-block walk, `compare_tail` compares the
   prompt's remaining tokens with the stored tokens of the children of the last matched key
   (root children by scan — the children index holds keyed parents only), most recently used
   first, at most `PARTIAL_SCAN_LIMIT` = 64, and fills `PrefixMatch.partial: Option<PartialMatch
   { key, block, tokens }>` with the longest common prefix (≥ 1 token). Only children resident
   in an exact page class (`exact_page_class`: the `l0` tag over a `bf16`/`fp8_e4m3` layout, or
   the `bf16` recent-window tag; TurboQuant base and every ladder rung class are lossy) and of
   exact lineage are sources; `PrefixMatch.partial_outcome` records `Reused` / `NoCandidate` /
   `NotResident` (matching child without an L0 copy) / `LossySource` (lossy class or lineage).
3. **Planner**: `reuse_cap_tokens(prompt) = prompt − MIN_RECOMPUTE_TOKENS`; `plan_prefix`
   computes the Phase 4 whole-block plan and then `extend_partial` adds
   `min(B, cap − k·B, covered − k·B)` tail tokens (`KvPlan.partial_tokens`, recompute shrunk)
   from the block at the cutoff — a matched L0-resident exact block when `k < matched`
   (`matched[k] == L0` and served exact), or the directory's partial match when `k == matched`.
4. **Gate of the extension**: it fires only when the cache reaches the prompt's end —
   `partial_tokens > 0` (a candidate matched beyond the full blocks) or
   `covered == prompt_tokens` (the prompt ends exactly on the cached full blocks, e.g. 256 →
   1 + 126). Without it, the plan is the Phase 4 whole-block minimum (verified over 200 random
   cases), which keeps `cutoff_minimises_cost`'s property intact. The `L0Pressure` and
   `TierDegraded` early returns do not extend: under pressure or with a degraded tier the plan
   stays whole-block (conservative; a tail copy there would allocate an extra L0 block).
5. **Metrics/config**: `EvictReason::PartialL0Only` (`partial_l0_only`, pre-registered on
   evictions and drops) and `PartialOutcome` (six outcomes incl. `LossySource` and `Disabled`
   per the 2026-10-04 amendment); the four S-11 keys with validation and
   `SchedulerParams.cache_aware_window` (Task 9 reads it).

## Tests

- `config::tests::phase5p_keys`: defaults (true / 1s / true / 32), inclusive bounds
  (10 ms, 60 s parse; 5 ms, 61 s rejected), `jump_forward_max_tokens` 0/257 and 2048 with
  `max_batch_tokens: 2048` rejected naming the key, `--set` round-trip, `examples/turbine.yaml`
  spells the four keys and loads.
- `directory::tests::token_granular_match` (128-token blocks, the spec's numbers): 3 blocks +
  40-token partial entry, a 25-token match; divergence at token 70 of a cached block → 70
  tokens of that block; shorter prompt vs an L1-only block → `not_resident`; colliding hasher →
  0 tokens and the full-block miss; prompt under one block vs a root-parent entry; the 64-child
  scan limit (a 71st-ranked child is skipped, found again once MRU); the ninth partial entry
  evicts the LRU (freed slot handed back); TurboQuant base class → `lossy_source`, BF16 window
  class → exact source.
- `planner::tests::token_granular_cap`: 129/130/255/256/257 fully cached → exactly
  0+127 / 1+0 / 1+125 / 1+126 / 1+127 (full blocks + partial tokens), recompute ≥ 2; 200 random
  no-partial cases equal the whole-block brute force (cutoff, recompute, cost); an L1-resident
  and a lossy-served source are never split.
- `kv_sim partial_match_falls_back_to_block_reuse`: a prompt diverging inside a cached block
  and a prompt shorter than the cached sequence keep the Phase 4 attach (block-granular
  `cached_prefix_tokens`, whole-block prefill, no violations) — the Task 6 machinery is
  plumbed (`PlanInputs.partial_tokens` fed from `PrefixMatch.partial`) but not yet consumed.
- Mutation-checked (each red, then reverted): the exact-class rule, the scan limit, the
  `reuse_cap_tokens` cap, the per-parent LRU eviction.
- `api kv_metrics_bounded`: `partial_l0_only` added to the documented reason set of the KV
  metric series (the evictions/drops pre-registration renders it at 0 from startup).

## Deviations from the plan text (deliberate, please review)

1. **`cutoff_minimises_cost`'s exact-multiple case changed** (64 fully cached tokens, 4 L0
   blocks): Phase 4 pinned `(reuse 3, recompute 16)`; the S-5 rule makes it
   `(reuse 3, partial 14, recompute 2)`. The plan's "keep every Phase 4 test unchanged and
   passing" and the amended S-4 AC (`256 → 1 + 126`) contradict on exactly this shape; the AC
   (decision 2026-09-28, the spec wins) is implemented and the assertion now pins the 5p
   behaviour with a comment naming the amendment. If the lead prefers the letter of "keep
   unchanged", the alternative is dropping the `covered == prompt_tokens` arm of the gate and
   re-bounding the S-4 AC — not recommended.
2. **The kv_sim partial-reuse test is Task 7's** (as the plan's task split says): a partial
   entry can only exist once `publish_tail` lands, so Task 6's kv_sim test is the Phase 4
   fallback guard above; `kv_sim partial_tail_reuse` (reuse at token granularity, one tail copy
   per reusing request, switch-off equality via `SimReport::plan_digest`) stays with Task 7.
3. **`evicted_partials()`**: the contract's `insert_partial -> Result<(), DirectoryError>`
   cannot report an internally evicted entry; the directory keeps them and the caller drains
   `evicted_partials()` to free the L0 slots. Task 7's `publish_tail` must call it (otherwise
   evicted slots leak); a pool-side assertion there would pin it.

## Notes for Tasks 7–8

- `hierarchy::attach_prefix` already passes `partial_tokens` into `PlanInputs`; it must now
  consume `KvPlan.partial_tokens`: allocate the private tail block in the source's page class,
  plan `tail_copies`, set `cached_tokens = blocks × block_tokens + partial tokens`, and record
  the final `PartialOutcome` (`PlannerCutoff` when the directory matched more than the plan
  reuses, `Disabled` when `kv.partial_prefix_reuse` is false — the lookup itself reports only
  the four comparison outcomes).
- The hierarchy must skip partial entries in demotion / prefetch / promotion / reuse evidence
  and reclaim them with `EvictReason::PartialL0Only`; they are leaves (a filled tail becomes a
  new full block under the full-block key, never a child of the partial entry).
- The split-source eligibility uses `penalty[k].is_none()` ("served exact"). Note a pre-existing
  6b wrinkle: recent-window L0 locations are tagged `bf16`, an unregistered codec name, so
  `format_is_lossy` is true for them and a window block matched by the full-block walk is
  served "lossy" — it is then excluded as a *split* source (conservative), while window-class
  *partial entries* are fine (the directory's tag rule is explicit). If the lead wants window
  blocks usable as split sources too, `exact_page_class`'s `"bf16" => true` arm can be threaded
  into that check.
- Contract §25 still lists the five-variant `PartialOutcome` (no `LossySource`) and `TailCopy`
  field names; the amended plan's six-variant enum is what landed. Lead-owned contract update.

## State

`scripts/gate.sh` → `gate: ok` on both commits (970 tests). Perf log has the Phase 5p row for
this landing. Nothing pushed; no lab work started. Next: Task 7 (`feat(kv,scheduler):
copy-on-write tail block for token-granular prefix reuse`) from this tip.
