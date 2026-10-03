# Handoff: p6b-tq4enc — the `lossy_tier_reuse{,_tq4}` "not encoded" failure is the lossless tail, not the codec

Branch `p6b-tq4enc` (from `p6b-stack` 7a67569). Two commits, both gate-clean (whole-workspace
gates, 944 tests, remote `target/debug` deleted after each):

- f4d4086 `test(kv_gpu)`: dump the kv document before the lossy-tier byte assert (diagnosis).
- 4848c98 `test(kv_gpu)`: run `lossy_tier_reuse{,_tq4}` with `kv.lossless_tail_blocks=0`.

The lab run that made the diagnosis: job `turbine-lab-test-1003223847-3666bf94`, log kept at
`target/tq4enc/diag-lossy.log` (workstation, this worktree's `target/`). Verification run after
the fix: job `turbine-lab-test-1003225616-2ffbe91a`, log `target/tq4enc/verify-fix.log`
(labbook `p6b-tq4enc:kv_gpu:*`).

## Root cause (established on the lab, not inferred)

`lossy_tier_reuse_tq4` and `lossy_tier_reuse` (`turbine-server --test kv_gpu`) failed their byte
assertion since the t17 quick tier with "L1 holds <0.83 x full size>: not encoded". The dump of
`GET /turbine/v1/kv` at the assertion shows both arms demoted 42 blocks (29 one-shot filler
sequences + A) and L1 held:

- tq4 arm: **28 copies at `l0`** (28 × 14,680,064 = 411,041,792 B) **+ 14 at `tq4`** (14 ×
  4,128,768 = 57,802,752 B) = 468,844,544 — exact.
- fp8 arm: **28 at `l0` + 14 at `fp8_e4m3`** = 513,805,376 (the document's fp8 bucket reads
  102,763,584 = 14 × 7,340,032 + 3,136; the tq4 arm is byte-exact, so this is the document's
  per-format accounting, not a slot size — `used_bytes` is Σ slot sizes; not investigated
  further).

**The tq4 demotion encoding never regressed.** Every non-tail demotion encoded at exactly its
codec's `bytes_per_block`. The 28 raw copies are the lossless tail: every finished sequence
tags its last `kv.lossless_tail_blocks` (default 1) full block(s) in `KvHierarchy.tail`
(`hierarchy.rs` `request_done`), and for a finished one-shot sequence that tag can never leave —
a finished sequence cannot grow out of its tail, so the entry leaves only when the block leaves
the directory ("entries leave with their block"). Spec S-2 (user decision 2026-09-28, Q13) says
exactly this: "the last N full blocks of a sequence at demotion time are demoted at the L0
format". So the tail blocks demoted raw, by contract.

**What changed to make the tests fail now** is the eviction order, not the encoding:
028f465 (`fix(kv): score a lossless last block like its history for eviction order`; user
decision 2026-10-02, "6b: OLMoE tq4 — lossless last block in eviction order", 1 A) prices a tail
block's retrieval like its history's, and its memory term keeps the real L0 size (3.56× a tq4
slot), so under `cost_aware` the stale tails score **lowest** and demote **first**. In the
failing runs the first 42 demotions were the 28 accumulated stale tails + 14 ordinary blocks —
0.83 × full size. At the t9 green run (e88c333) tails scored above their history and demoted
last, so the first 42 demotions were all encoded. The last green kv_gpu before t17 is e88c333
(2026-10-01); 028f465 landed 2026-10-02 inside that window, on the OLMoE tq4 line.

## What landed

`lossy_tier_reuse{,_tq4}` now run with `kv.lossless_tail_blocks=0`. The tests predate the tail
rule (they judge that a capacity demotion into a lossy tier is encoded, reused within the tier
bound, and refused by `x-turbine-kv-lossy: deny`); they never meant to exercise the tail
exemption, and with 29 one-shot fillers the exemption dominates. The tail's own behaviour stays
pinned where it belongs: `document_lists_copies_per_codec` (tail demoted at the L0 format) and
`last_block_is_scored_like_its_history` (its demotion order). The kv-document dump before the
byte assert stays (it is what will diagnose any future failure of these tests).

## Verification after the fix (lab, job turbine-lab-test-1003225616-2ffbe91a)

- `lossy_tier_reuse` (fp8 arm): **PASS end to end** — 42 demotions, L1 holds 42 × 7,340,032
  = 308,290,752 exactly (no raw copies), lossy reuse 384 of 447 prompt tokens, worst first-8
  |Δ logprob| 0.050 within the 0.3 bound, `x-turbine-kv-lossy: deny` bit-equal to cold.
- `lossy_tier_reuse_tq4`: the byte assertion now **passes** — L1 holds 42 × 4,128,768 =
  173,408,256 exactly, so the tq4 demotion encoding is proven exact — and the deny and reuse
  mechanics hold (lossy cached 384, 0.97 of positions within 0.75). The test still fails, one
  step later, on the tq4 tier's **accuracy head bound**: worst first-8 |Δ logprob| 0.2518
  against `TQ4_TIER_BOUNDS` 0.25 (t9, e88c333) — over by 0.002. With the tail exempted, all
  three of A's cached blocks are tq4; at t9 A's lossless tail block was still exact, and the
  t9 measurement (0.242) sat 0.008 under the bound. The 0.25 head bound is this test's proxy
  for golden's batched *likely* bound (0.25 when the cold logprob > −2, else the tail bound
  0.75); the test applies 0.25 flat, which is stricter than golden for any cold-unlikely
  token. Not relaxed here — the tq4 tier's bound is a gate decision.

## Second open question for the lead/user — the tq4 arm's head bound

- **A (recommended): apply the gate's actual rule** — bound each of the first 8 positions by
  0.25 when its cold logprob > −2 (`likely_logprob_floor`), else 0.75, exactly as
  `turbine-golden compare` judges the slug's batched bounds (the comment already claims
  derivation from those gates). Whether the offending token is cold-likely needs one print in
  the test and a lab rerun to confirm it is the unlikely case.
- **B: recalibrate** the head bound for the all-blocks-lossy prefix (e.g. 0.3) with the
  reasoning recorded like t9's was — a user call on a gate-derived number.
- **C: leave the tq4 arm red** until Task 18's gate work recalibrates; the encoding, reuse,
  and deny mechanics it owns are all proven now.

## Open design question for the lead/user — stale tails' standing cost

Not fixed here (it touches user decisions); recorded with options:

- Today, every finished sequence permanently costs one raw-format copy per lower tier: on
  one-shot-heavy workloads up to 1 raw L1 slot (14 MiB) per finished sequence, ~3.56× a tq4
  slot, and — with the ladder (S-6/S-7) — the L0 sweep and the recent-window conversion skip
  tail-tagged blocks **forever**, so those L0 blocks can never ladder-compress. This is likely
  why the mt3 ladder-on runs ended "almost every stored copy still l0". The tail set is also
  unbounded (bounded only by live blocks).
- **A (recommended): tail tags expire** — `tail` holds only the latest finished sequence's
  last-N keys (the field doc in `hierarchy.rs` already says "the latest finished sequence that
  holds them"), i.e. a later finish replaces the set (`tail.clear()` before re-tagging, in
  `request_done`). One-shot sequences' tails demote lossy; interleaved multi-turn sessions lose
  exact tail reuse only across other sessions' turns (lossy reuse covers it, within golden
  bounds). Small change; fix the two lab tests back to the default tail afterwards (or leave
  them at 0 — with expiry their byte assertion holds at any tail value only up to N raw copies).
- **B: keep the contract, live with the cost** — leave the tests at `lossless_tail_blocks=0`,
  accept ≤ 1 raw L1 slot per finished sequence and the permanent L0-sweep skip; document it.
- **C: per-session tails** (each session's latest finished turn keeps its tail until that
  session continues or ends) — keeps multi-session exact tail reuse, needs session-lifetime
  bookkeeping in the hierarchy.

## Notes for the next builder

- The kv_gpu byte assertion (`used < 0.75 × demotions × block_bytes`) reads `turbine_kv_bytes
  {tier=l1,kind=used}` against `turbine_kv_demotions_total{from=l0,to=l1}`; both are sampled
  after the fill loop, so the mix that matters is which blocks demoted, not timing.
- `document_lists_copies_per_codec` (turbine-kv hierarchy tests) asserts the tail demotes at
  the L0 format with `lossless_tail_blocks` at its default — do not "fix" that to match an
  expiry decision without re-pinning it (and `last_block_is_scored_like_its_history`, which
  asserts the tail's L1 memory at the real encoded size and its retrieval like its history's).
- 29 fillers × ~19 blocks ≈ 550 ≈ L0's 585 blocks: the fill loop exits right after L0 first
  overflows, so the demotion burst is ~2 sequences' worth. Anything that shifts L0's effective
  size (e.g. the recent window's BF16 class) shifts these counts.
- Lab logs for this task: `target/tq4enc/` on the workstation.
