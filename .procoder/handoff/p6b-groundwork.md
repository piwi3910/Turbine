# Handoff: p6b-groundwork (Phase 6b host groundwork)

Written 2026-09-29 ~08:00 by the new 6a lead. Worktree `.claude/worktrees/agent-ab443ff21c9b66ddd`, clean.
Branch is 73 commits behind `phase-6a-quantization`; it is merged into 6b when 6b starts from main (after 6a closes).

## Done (commits on this branch only)

- 0a0b20a `feat(kv): KV codec registry with l0 and fp8_e4m3` (6b plan Tasks ~1–3 host part).
- 3d07054 `feat(kv): TurboQuant tq4 and tq2 codecs (CPU reference)` — Gaussian QJL S per (layer, head),
  16-byte-padded records, SplitMix64 seeds, committed codebooks, NMSE bounds tq4 0.07 / tq2 0.75.
- 66dcf2d `feat(kv): compress as a third eviction action with the ladder rule` — `EvictAction::Compress`,
  `LadderContext`, `cost_aware` ladder, `EvictReason::{Compressed, LadderFloor}` (not in metric label sets until 6b Task 15).
- Gate ok (335) at 66dcf2d.

All design details were accepted by the user 2026-09-29 (decisions.md "Phase 6a/6b builder decisions", items 1–11),
**except item 10 (ladder rule), which the user changed: "Start at YELLOW earlier".**

## Exact next step (when 6b starts; not before 6a merges into main)

1. Branch 6b from main, merge this branch.
2. Change the ladder rule in `cost_aware` (66dcf2d) per 6b plan (line ~243) and spec S-6: at YELLOW or above the
   lowest enabled tier compresses one rung even with free room; at any non-GREEN state a tier about to drop or
   above high water acts; still lowest tier first, one rung at a time, GREEN never compresses.
3. Update `policy::tests::ladder_actions` (a YELLOW case with free room compresses one rung; GREEN never) and
   run `cargo test -p turbine-kv` via `scripts/remote-cargo.sh`, then `scripts/gate.sh`.
4. Update 6b plan Task 7 / spec S-4 text for the Gaussian S (accepted) if not yet done.

## Traps

- `turbine-kv` holds its own e4m3 rounding copy (accepted); keep its bit-exact test against the kernels' table.
- Host-only branch: no lab runs needed until the GPU codec tasks.
