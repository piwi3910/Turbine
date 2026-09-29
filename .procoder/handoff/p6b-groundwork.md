# Handoff: p6b-groundwork (Phase 6b host groundwork)

Updated 2026-09-29 (YELLOW depth). Worktree `.claude/worktrees/agent-ab443ff21c9b66ddd`, clean. Rebased onto
`phase-6a-quantization` a181274 (clean rebase, gate ok 341 before any change); merged into 6b when 6b starts
from main (after 6a closes).

## Done (commits on this branch only)

- 59158dd `feat(kv): KV codec registry with l0 and fp8_e4m3` (6b plan Tasks ~1–3 host part).
- 3c8c709 `feat(kv): TurboQuant tq4 and tq2 codecs (CPU reference)` — Gaussian QJL S per (layer, head),
  16-byte-padded records, SplitMix64 seeds, committed codebooks, NMSE bounds tq4 0.07 / tq2 0.75.
- eb6ba59 `feat(kv): compress as a third eviction action with the ladder rule` — `EvictAction::Compress`,
  `LadderContext`, `cost_aware` ladder, `EvictReason::{Compressed, LadderFloor}` (not in metric label sets until 6b Task 15).
- ef5b8d5 `feat(kv): the compression ladder starts at YELLOW, before the tiers are full` (gate ok 341) — the user's amendment
  (decisions.md "User review of the 2026-09-29 overnight decisions", item 3; spec S-6, plan Task 14). Rule in
  `cost_aware::action`: ladder on and pressure not GREEN (= YELLOW or above) → the lowest enabled tier always
  compresses one rung (even with free room), an upper tier only above `kv.ladder.high_water`, an upper tier only
  once every tier below has reached the target rung; at the floor the lowest tier drops only a leaving copy or
  above high water (never with free room), otherwise keeps; GREEN / ladder off / `lru` = Phase 4.
  `policy::tests::ladder_actions` and `registry_conformance::eviction_policies` (new clause: a drop only of a
  leaving copy or above high water) pin it; mutation checks done (old trigger → `ladder_actions` fails; unguarded
  floor drop → both fail).

- `feat(kv): the ladder compresses at YELLOW only until GREEN headroom` — user decision 2026-09-29
  "Compress only until GREEN" (decisions.md "6b ladder: YELLOW depth (2026-09-29)"; spec S-6, plan Task 14).
  `LadderLimits.low_water` (`kv.ladder.low_water`, default 0.85; no config key yet — Task 15 / core config)
  and `LadderContext.demand` (bytes the YELLOW reclaim wants to demote into the tier this tick, as a fraction
  of its capacity; 0 when none). `cost_aware::action` at YELLOW, nothing leaving and fill ≤ high water: the
  lowest tier compresses only while `fill + demand > low_water`, else keeps (Phase 4). GREEN never
  compresses; ORANGE and above, the upper-tier rule and the floor guard as built. `policy::tests::ladder_actions`
  gained the YELLOW keep / compress cases, GREEN after YELLOW, ORANGE with room, and a 50-sweep steady-pressure
  simulation (16 copies, fill re-derived from codec bytes after each action, demand 0.02): YELLOW stops at
  `fp8_e4m3` with GREEN headroom restored, ORANGE walks every copy to `tq2`. `ladder_contract` (conformance)
  gained: no compression at YELLOW of a copy that need not leave while `fill + demand ≤ low_water`.
  Mutation check: removing the low-water stop fails `ladder_actions`. `docs/extending/eviction-policy.md`
  documents `action` and `ladder_contract`.
- Floor guard (at the last rung a copy is dropped only when it must leave or its tier is above high water)
  matches the spec's edge case; accepted as the lead's call 2026-09-29.

All other design details were accepted by the user 2026-09-29 (decisions.md "Phase 6a/6b builder decisions", items 1–11).

## Exact next step (when 6b starts; not before 6a merges into main)

1. Branch 6b from main, merge this branch.
2. Task 15 (hierarchy): `ladder_tick` must fill `LadderContext.demand` (bytes the YELLOW reclaim,
   `apply_reclaim` `DemoteIdle` to the `kv_utilization` YELLOW threshold, would demote into the tier, as a
   fraction of its capacity), refresh `fill` after each rewrite and stop the tick once the policy keeps; which
   copies are offered per tick (oldest, least-reusable; ≤ 32 rewrites, ticks ≥ 50 ms) and the
   rung-for-new-demotions hysteresis (step back up after `deescalate_dwell`) are the hierarchy's job.
   `ladder_under_pinned_pressure` must assert the stop at GREEN and no tq2 drift over a steady-YELLOW
   stretch (plan Task 15 text), and its mutation check drops the `low_water` stop. Add the
   `kv.ladder.low_water` config key (core config: lead-owned).
3. Update 6b plan Task 7 / spec S-4 text for the Gaussian S (accepted) if not yet done.

## Traps

- `turbine-kv` holds its own e4m3 rounding copy (accepted); keep its bit-exact test against the kernels' table.
- Host-only branch: no lab runs needed until the GPU codec tasks.
- Agent harness: background Bash tasks die when the agent's turn ends; start `scripts/gate.sh` detached
  (`nohup bash -c "scripts/gate.sh …; echo rc=\$?" > <scratch>/gate.txt 2>&1 </dev/null &`) and wait on the file.
