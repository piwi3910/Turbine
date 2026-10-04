# Handoff: p6b-t4 (6b plan Task 4 — lossy lineage, opt-out, lossy token counts, planner penalty)

Branch `p6b-t4` from `p6b-stack` e58f715. Host-only; no lab runs, nothing detached.

## Done

`feat(kv): lossy lineage keys, per-request opt-out and lossy token counts` (one commit):

- `identity::lossy_key(key, format, seed)` = BLAKE3("lossy" ‖ key ‖ format ‖ seed LE)[..16];
  `NamespaceKey::seed()` (first 8 bytes LE). The hierarchy uses the _unsalted_ namespace's seed,
  the same one `HostCodec::of` gives the tier codecs, so no salt is needed at promotion time.
- `directory::Lineage { Exact, Lossy { format } }` on `KvBlock.lineage`; `format_is_lossy`.
  `KvDirectory::lookup(.., allow_lossy, seed)`: per block the candidates are the exact key's
  entry, its promoted lossy entries (`lossy_key(k, codec, seed)` for each registered codec) and,
  after a lossy switch, the lossy-chain entry. Exact copies (exact lineage, lossless format) go
  first, fastest tier; else the fastest lossy one when allowed; else `denied` (opt-out) or miss.
  `PrefixMatch.{denied, lossy_from, lossy_chain}`, `keys_for(cut)` gives the keys and lineage a
  request commits under (exact keys unless a block before the cut was lossy). `MatchedBlock.lossy`.
- Chain rule: at the first lossy block s (codec f) the chain key is `lossy_key(k_s, f, seed)`,
  then `H(chain_{i-1}, tokens_i)`; blocks the request computes take those keys and
  `Lineage::Lossy { f }`. Their directory parent is the last _attached entry_ (so a lossless
  tail or an exact entry used after the switch still parents correctly).
- A lossy-format copy of an exact block promoted (or prefetched) into L0 is filed under a new
  entry `lossy_key(k, f, seed)` (lossy lineage, parent = the block's parent), never as an exact
  L0 copy of k (`KvHierarchy.lossy_targets`, ticket → target, bounded by the transfers).
- Opt-out publishes the exact chain: `commit_progress` adds the recomputed L0 copy to an existing
  exact entry whose every copy is lossy (only then — l0-format tiers behave exactly as before).
- Planner: `PlanInputs.{copy_bytes, lossy_penalty, allow_lossy}`; retrieval × (1 + penalty);
  without `allow_lossy` the cap stops at the first lossy block (defence in depth).
  `copy_bytes` (per matched block, its tier format's size) was needed: the transfer EWMA is
  measured per moved (encoded) byte, so pricing a tq4 copy at L0 bytes overestimated it ~4× and
  the planner never retrieved lossy blocks. `HierarchyConfig.{allow_lossy, lossy_penalty}` from
  `kv.lossy_reuse` / `kv.lossy_penalty` (one entry per registered codec).
- `PrefixAttach.lossy_tokens`; `turbine_kv_lossy_cached_tokens_total`,
  `turbine_kv_lossy_denied_total`; DEBUG `kv_lossy_reuse` log. `AttachRequest.allow_lossy:
Option<bool>` (None → `kv.lossy_reuse`). `KvHierarchy::l0_entry(BlockId)`.
- Core: `Usage.lossy_cached_tokens`, `RequestKvPolicy { allow_lossy }`,
  `GenerationRequest.kv_policy: Option<RequestKvPolicy>`. API: `x-turbine-kv-lossy: allow|deny`
  (else 400 `invalid_request`) → `TurbineHeaders.kv_policy`; `usage.prompt_tokens_details
.lossy_cached_tokens` always present (openai.rs expected-JSON tests updated for it). Server:
  `KvOrchestrator::attach(pool, &AttachRequest)`, `ActiveRequest.lossy_cached_tokens`.
- Sim: `KvSimDriver::submit_with(.., allow_lossy)`, `attached(id) -> AttachRecord`.

Tests: `kv_sim lossy_lineage_never_reaches_opted_out` (new), `api kv_metrics_bounded` (lossy
families, `lossy_cached_tokens` in completion and chat, header allow/deny/bad),
`planner::tests::lossy_penalty_weighs_retrieval`, `identity::tests::lossy_keys_never_alias`,
`kv::tests::kv_lossy_header`. Mutation (lookup ignores `allow_lossy`): the kv_sim test fails
(`lossy_denied_total` stays 0; the planner's own cut still keeps lossy blocks away); with the
planner's cut also removed it fails at "an opted-out request got a lossy block".

## Not done / for the lead

- Contract §26 names (lead-owned): `lossy_key`, `Lineage`, `KvDirectory::lookup(.., allow_lossy,
seed)` (seed added), `PlanInputs.copy_bytes` (new), `PrefixMatch.{denied, lossy_from,
lossy_chain}`, `AttachRequest.allow_lossy`, `GenerationRequest.kv_policy`.
- Not in Task 4's file list, left for later tasks: `format` label on `turbine_kv_blocks/bytes`,
  transcode and ladder metric families, `GET /turbine/v1/kv` `lossy` object, the lab
  `kv_gpu lossy_tier_reuse` test (Task 6).
- Task 15 (hierarchy ladder) will conflict textually in hierarchy.rs; my edits are in
  attach_prefix, commit_progress, request_done, on_copy_done/failed, prefetch_keys, new().
  A ladder rewrite of a copy to a lossy rung needs no new lineage handling: lookups read the
  location format; a Task 17 L0 compression must file the compressed block under `lossy_key`
  (like `lossy_entry`).

## OPEN: gate FAIL — `turbine-bench::kv_sim cost_aware_beats_lru` (handed off at ~300k context)

Commit ef69456 (the Task 4 commit) is **not gate-clean**. `scripts/gate.sh --base e58f715`:
`gate: FAIL crates=… passed=811 failed=1` — only `turbine-bench --test kv_sim cost_aware_beats_lru`.
Repro: `scripts/remote-cargo.sh test --release -p turbine-bench --test kv_sim cost_aware_beats_lru -- --nocapture`
(the numbers are simulated, identical in debug and release).

Assertion (benches/turbine-bench/tests/kv_sim.rs:30): `MultiTurn: cost_aware 185.301s vs lru 199.478s (bound 0.9)`.

| MultiTurn simulated_prefill_seconds | cost_aware | lru     | ratio       |
| ----------------------------------- | ---------- | ------- | ----------- |
| base e58f715                        | 195.033    | 251.126 | 0.78 (pass) |
| p6b-t4 ef69456                      | 185.301    | 199.478 | 0.93 (fail) |

Both policies got _better_ (lru much more: 251 → 199), so Task 4 changed hierarchy behaviour
even with `l0`-format tiers (the bench configures no tier format, no lossy anything) — which it
should not: with only lossless copies every lookup, plan and commit should be as before.
I believe it is an unintended change (a bug), not a deliberate consequence of Task 4; do not
move the bound.

Bisect so far:

- Base vs ef69456 measured (table above).
- Reverted the promotion-landing guard in `on_copy_done` (hierarchy.rs:1182-1187,
  `.is_some_and(|b| b.location(TierId::L0).is_none())` → `.is_some()`, old replace-always
  behaviour): **no change** (185.301 / 199.478). Not the cause; reverted back.

Remaining suspects (all in ef69456, each should be a no-op for lossless tiers; check each):

1. `KvDirectory::lookup` rewrite (directory.rs `lookup`): candidate loop, `last_access` set only on
   the chosen entry, pending check now `!contains_key(chain.unwrap_or(key))`, mismatch handling
   when several candidates. Compare `KvStats.lookups` per workload against base first.
2. `attach_prefix`: `r.keys` is now assigned after the promotion loop / `after_plan` (was before
   planning); `register_pending` over `keys_for(cut)`; `copy_bytes` = `format_bytes(location.format)`
   (should equal `block_bytes` for `l0`).
3. `commit_progress`: parent of the first computed block = `r.used.last()` (should equal
   `keys[cut-1]`); the new publish branch (should never fire for `l0` copies).
4. `request_done`: tail/session keys now via `RequestKv::entry(i)` (used[i] else keys[i]).
5. prefetch_keys (`lossy_target`) and the `KvStats`/metrics additions (should be inert).
   Fastest route: dump `KvStats` (lookups, promotions, demotions, evictions) for both commits with
   the bench's `--output json` / the test's stderr and diff; then revert hunks one at a time.

## Also not done from brief-p6b-t4.md

- Gate is not ok (above), so the plan's commit is not final; no other brief item open besides the
  lead-owned contract §26 names listed above. No lab/GPU work was in scope.

## Regression fix (r11 builder, 2026-09-30)

Root cause: `commit_progress`'s P6b S-3 publish rule (hierarchy.rs, the `publish` condition,
~line 980) tested "every copy of the existing exact entry is lossy" with `locations.iter().all(..)`,
which is true for an entry with **no** copy — a parent kept only because it has children
(`forget` keeps it while `child_count > 0`). So with L0-format tiers a recomputed parent was
re-keyed into its copyless entry, where the Phase 4 hierarchy leaves the recomputed block
unkeyed. More reuse for both policies, much more for lru (it evicts parents under their
children more often) — hence 251 → 199 s lru, ratio 0.78 → 0.93.
Fix: `&& !b.locations.is_empty()` — publish only onto an entry holding lossy copies.

kv_sim MultiTurn simulated_prefill_seconds (cost_aware / lru): base e58f715 195.033 / 251.126;
ef69456 185.301 / 199.478; fix 195.033 / 251.126 (identical to base; Mixed and SharedSystem
too).

Test: `hierarchy::tests::commit_leaves_a_copyless_parent_unkeyed` (turbine-kv lib) — fails
without the guard ("filed under a copyless exact entry: [L0 slot 0]"), passes with it.
Mutation re-checked: lookup ignoring `allow_lossy` still fails
`kv_sim lossy_lineage_never_reaches_opted_out`. `copy_bytes` ruled out (equals `block_bytes`
at `l0`).

Open (lead): re-keying a recomputed block into a copyless parent entry would itself be a real
reuse win (lru −21 %, cost_aware −5 % on MultiTurn), a Phase 4 behaviour change of its own;
not done here — it would need its own decision and a re-baselined bound.

Gate: `scripts/gate.sh --base e58f715` → `gate: ok … passed=813 failed=0`.
