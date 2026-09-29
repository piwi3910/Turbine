# Handoff: p6b-groundwork (Phase 6b host groundwork)

Updated 2026-09-29 ~14:30. Worktree `.claude/worktrees/agent-ab443ff21c9b66ddd`, clean. Rebased onto
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

All other design details were accepted by the user 2026-09-29 (decisions.md "Phase 6a/6b builder decisions", items 1–11).

## Exact next step (when 6b starts; not before 6a merges into main)

1. Branch 6b from main, merge this branch.
2. Task 15 (hierarchy): at sustained YELLOW the policy says "compress one rung" for every lowest-tier copy
   offered, so over time the whole lowest tier walks down to `max_format` with free room; which copies are
   offered per tick (oldest, least-reusable; ≤ 32 rewrites, ticks ≥ 50 ms) and the rung-for-new-demotions
   hysteresis (step back up after `deescalate_dwell` below 0.85) are the hierarchy's job — the committed
   pressure trace in `ladder_under_pinned_pressure` must pin how far it goes. Open question for the lead/user if
   the sim shows the whole tier reaching tq2 at mere YELLOW is undesirable.
3. Update 6b plan Task 7 / spec S-4 text for the Gaussian S (accepted) if not yet done.

## Traps

- `turbine-kv` holds its own e4m3 rounding copy (accepted); keep its bit-exact test against the kernels' table.
- Host-only branch: no lab runs needed until the GPU codec tasks.
- Agent harness: background Bash tasks die when the agent's turn ends; start `scripts/gate.sh` detached
  (`nohup bash -c "scripts/gate.sh …; echo rc=\$?" > <scratch>/gate.txt 2>&1 </dev/null &`) and wait on the file.
