# Handoff: p6b-smallfix — tail-tag expiry and the tq4 lab bound split

Branch `p6b-smallfix`, from `p6b-stack` af9b82af. Two commits, both following the two accepted
lead decisions of "6b: stale lossless-tail tags and the tq4 lab bound" (2026-10-03). Host-only:
no lab Job, serve or lock of this branch was created; tree clean, gate run per commit
(`--base p6b-stack`).

## Commits

- `fix(kv)`: **tail tags expire with the latest finished sequence** (decision 1 A).
  `KvHierarchy::request_done` clears the tail set before re-tagging, so `tail` holds only the
  latest finished sequence's last `kv.lossless_tail_blocks` full blocks (bounded at N). The
  expired tail demotes, evicts and ladders like any other block: no permanent raw-L0 handling,
  no permanent L0-sweep/recent-window-conversion skip, no unbounded set. A live sequence keeps
  its S-2/Q13 guarantee — the growing-sequence case is pinned separately. Tests red first:
  - `hierarchy::tests::a_later_finish_expires_the_previous_sequence_tail` — after a second
    sequence finishes, the first one's tail block demotes encoded (`fp8_e4m3`) while the
    latest sequence's tail still demotes at the L0 format; the tail set names only the latest
    sequence's keys. Mutation: dropping the `tail.clear()` fails it and the guard below.
  - `hierarchy::tests::a_growing_sequence_keeps_its_tail_exact_until_the_turn_finishes` —
    turn 2 attaches, commits and releases without finishing; the turn-1 tail keeps its tag and
    demotes raw while the sequence grows; `request_done` re-tags to turn 2. (Green under the
    old code too — it pins the live guarantee, the RED test is the one above.)
  - kv_sim `per_tier_formats` re-pinned: at most one lower-tier copy at `l0` (the latest
    finished sequence's tail; raw copies already stored while tagged stay raw — expiry stops
    new ones, it does not re-encode), at least one expired tail demoted encoded; the fp8/tq4
    capacity factors unchanged. Both ladder ACs
    (`ladder_under_pinned_pressure`, `ladder_l0_under_pinned_pressure`) pass **without a
    fixture re-bless**.
    Docs updated: spec S-2 and its `per_tier_formats` gate wording, `HierarchyConfig` and
    `tail` field docs, `docs/extending/kv-format.md`, `docs/extending/eviction-policy.md`.

- `test(kv_gpu)`: **the tq4 accuracy head bound uses golden's likely/tail split** (decision 2).
  `lossy_tier_reuse_tq4` judged the first 8 answer tokens by a flat 0.25 — golden's _likely_
  bound, which the t9 calibration took from a run whose lossless-tail block was still exact.
  The head bound is now golden's actual rule (the same `turbine-golden compare` comparison):
  a position whose cold logprob is above the `likely_logprob_floor` (−2) is held to 0.25, at
  or below it to the tail bound 0.75; nothing else relaxed (the per-position judgement, the
  90 %-within-0.75 share, and the fp8 arm's tighter 0.3 / 0.5 / 0.9 are unchanged). The split
  lives in `HeadBound::bound_at`, unit-pinned host-side by
  `golden_head_bound_splits_likely_from_tail` (mutation: `>` → `>=` fails it at the −2 case;
  the old code's expiry test also went red on the same mutation run's shape). The all-lossy
  worst case that measured 0.2518 (tq4enc handoff) is inside the split rule exactly as it is
  inside golden at c16.
  `lossy_tier_reuse{,_tq4}` still run with `kv.lossless_tail_blocks=0` — deliberate: with the
  tail at its default, A's own tail block demotes raw and comes back exact, softening the
  all-lossy worst case the bound is held to (the tq4enc handoff's "or leave them at 0"
  option). The tests' comment now records that this is on purpose.

## Verification status

- Gate clean per commit (`scripts/gate.sh --base p6b-stack`: fmt, whole-workspace clippy,
  diff-scoped tests incl. reverse dependencies).
- Fix 1 is fully verified host-side (units + kv_sim + engine tests through the gate).
- Fix 2's `lossy_tier_reuse_tq4` is a lab test: the bound change compiles and is unit-pinned,
  but **it has not run on the GPU on this branch — it is taken by the phase-exit lab pass**
  (expected: the tq4 arm's head judgement now tolerates the cold-unlikely position that
  measured 0.2518, everything else as in the tq4enc verification run).

## Notes for the next builder

- Raw lower-tier copies stored while a block was tagged stay raw; expiry only stops new raw
  demotions (per_tier_formats pins ≤ 1 raw key on its workload). If the phase-exit multi-turn
  ladder runs still end "mostly l0", stale raw copies are not the explanation any more — look
  at the recent window's BF16 class instead.
- The pre-existing `perf-log.md` t18 table is not prettier-clean (column padding); left
  untouched — not this branch's diff.
- p6b-crash's carry-over stands (transient root cause unknown; `l0_keys` BTreeMap suggestion).
