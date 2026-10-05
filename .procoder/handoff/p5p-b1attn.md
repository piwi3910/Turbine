# Handoff: Phase 5p — the batch-1 attention own kernel (`p5p-b1attn`)

Branch `p5p-b1attn`, from `p6b-stack` 1c1c5105. Gate-clean commits; the tree is the task's exit
state. Logs under `target/b1attn/`.

## What landed (user decision B, 2026-10-04: the b1 attention own kernel is tasked)

`feat(rocm): the b1 decode attention kernel -- one launch, splits, fused append and combine`
(the commit also carries the tests and the gfx1201 preference; the evaluation entry in
`.procoder/ask/decisions.md` "5p b1 attention: provider evaluation before the own kernel (kernel
reuse rule, 2026-10-05)" landed first, 5866edaf).

- **Reuse evaluation recorded** (the kernel-reuse rule): llama.cpp `fattn-mma-f16` / `fattn-vec`
  not reusable (contiguous ggml K/V, no page table, no ragged decode batch, FATTN_KQ_STRIDE
  padding); CK `fmha_fwd_splitkv` fits the descriptor but is occupancy-bound at batch 1 and its
  multi-split numerics are the rejected "Pre-Phase-5 #3" class; CK `fmha_fwd_pagedkv` measured
  1.2–5× slower for grouped decode; hipBLASLt has no attention primitive. Nothing reusable —
  the own kernel stands.
- **The kernel** (`kernels/rocm/src/paged_attention_b1.hip`, impl `turbine_hip_b1_decode` of
  `attention_decode_paged`): ONE launch per layer where CK's b1 path was three (Turbine append +
  CK splitkv + CK combine).
  - Grid (sequences, KV heads, 8 splits) — batch 1 fills the card with 64 workgroups instead of
    CK's 8; each workgroup attends `ceil(kv_len / 8)` keys of one (sequence, KV head).
  - The k_new/v_new append is fused: the workgroup whose key range holds position kv_len − 1
    writes its KV head's new row before scoring it (exactly one split does; no cross-workgroup
    dependency, the append's launch and boundary disappear).
  - The split combine is fused: each workgroup writes its partial (F32 P·V accumulator, row
    maximum, sum — one 130-float record per head) into a dedicated context workspace
    (`attn_b1_scratch`, grown outside captures, counters always zero between calls), fences,
    counts itself in a per-(sequence, KV head) arrival counter, and the last-arriving workgroup
    combines the splits in ascending split order (deterministic whatever the arrival order),
    writes the output rows and resets the counter. Capture-safe: no host ops, scalars by value,
    device arrays as today; a shape change re-zeroes the counter region before the next eager
    call (another shape's partials may have overlapped it), which never happens under a capture.
  - Numerics, a class of its own (not bitwise CK): F32 scores in ascending-dim chunk order,
    online softmax with a running maximum per wave (waves in wave order, splits in ascending
    order), **P kept in F32 through the P·V product** — unrounded, so the split-local maxima cost
    only F32 rescale roundings, unlike CK's multi-split BF16 P that made "Pre-Phase-5 #3" reject
    splits. Output divided by the sum, rounded to BF16 once.
  - Tier: at most 8 sequences and GQA group ≤ 8 (templates G = 1..=8), grouped heads only
    (OLMoE's 16/16 keeps `ck_tile_fmha_pagedkv` and its bitwise invariance), BF16 flat pages,
    any `block_tokens`. Above the tier CK splitkv stays the default; a request crossing the
    boundary changes its rounding once (the small-m MoE tier pattern).
  - Fixed splits + per-sequence independence ⇒ a row's bits do not depend on its batch
    (within-tier bitwise batch invariance, as the small-m tier).
- **Selection**: first in the library's decode-paged table and first in gfx1201's
  `PAGED_DECODE_ORDER`; the fallback-note test treats it as a non-fallback pick.

## Tests

- `hip_ops::paged_b1_decode_matches_cpu` — decode batches of 1 / 3 / 5 / 8 sequences (a single
  ~8k-token sequence, page-edge 127/128/129, a ragged 8 at ~768), groups 3, 4, 6, 7, 8, 128- and
  16-token pages, against the CPU reference; the pool compared exactly after the (fused) append.
- `hip_ops::paged_decode_splitkv_matches_cpu` — expectations split by the tier (≤ 8 sequences
  b1, above CK splitkv); `paged_prefill_ck_128_matches_cpu` — the 4-sequence decode batch now
  expects b1 at both page sizes.
- `hip_batch_invariance::b1_decode_rows_are_batch_invariant` — a 24/8-head decode row alone =
  bitwise inside 2- and 8-decode batches (first and last position); the 16-decode comparisons
  run against CK splitkv under a recorded bound (0.05 placeholder — tighten to the measured
  max_abs on the first GPU run).
- Mutation checks: (1) the combine skipping the last split fails `paged_b1_decode_matches_cpu`;
  (2) making the split count depend on `num_seqs` fails the invariance test's bitwise cases;
  (3) the append condition without the `kv_len - 1 < k1` bound (the bug found and fixed in
  review) writes several splits' rows and fails the pool-exact comparison.

## Results — GPU runs still PENDING (the blocker)

novanas's GPUs were held the whole task window (13:20 onward, 5 h+) by the user's own
`kuvryn-ai-workloads` pod — both `amd.com/gpu` claims, not touched per the lab rules. Everything
else is done: gate-clean commits, the kernel compiles for gfx1201 (lab build), the reuse
evaluation, the tests, the perf-log section. The GPU work below lands as soon as the claims free,
in this order (each holds the bench lock through the scripts):

1. `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- paged_b1_decode_matches_cpu
paged_decode_splitkv_matches_cpu paged_prefill_ck_128_matches_cpu` — correctness vs CPU.
2. `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_batch_invariance --
b1_decode_rows_are_batch_invariant` — bitwise within the tier; TIGHTEN the 0.05 cross-tier
   placeholder to the printed max_abs + headroom.
3. `scripts/lab-test.sh novanas -- -p turbine-kernels --test hip_ops -- decode_attention_timings`
   — the op-level A/B in one sweep (CK splitkv 19.5 + 2.1 µs/layer recorded vs b1 at b = 1..8;
   re-place `kTurbineB1MaxSeqs` from the lines if the crossover is not between 8 and 9).
4. Mutation checks (snapshot the file, mutate, run, restore): (a) combine skips the last split →
   test 1 must fail; (b) split count derived from `num_seqs` → the invariance bitwise cases must
   fail; (c) the append condition without `kv_len - 1 < k1` (found and fixed in review) → the
   pool-exact comparison must fail.
5. `scripts/lab-bench.sh --quick --c1 --model llama --label b1attn-c1` (golden c1 + c1 tok/s vs
   the S-2 row 77.8 / ITL 12.7 ms) and `scripts/lab-bench.sh --quick --model llama --label
b1attn-c16` (vs 879.1 quick c16; bound 0.97x).
6. `scripts/lab-decode-profile.sh --model llama --label b1attn` + `scripts/decode-profile-report.py
target/decodebw/b1attn-llama --weights 6431025664` — served attention µs/layer vs 21.7 incl.
   merge; effective bandwidth vs the 640 GB/s peak.
7. Record the numbers in the perf-log section above and in decisions ("5p b1 attention" result
   note); then this handoff's Results table.

Target (floor decision B): attention ≤ ~10 µs/layer, Llama c1 toward ≥ 90 tok/s. If the b1
kernel misses the µs target, the `decode_attention_timings` lines say whether the split count
(neither 4 nor 16 was measured — kSplits is one constant), the wave layout, or the append/combine
fusion is the residual; the op bench cycles pools cold, so judge served, not op-level only (the
Task 18 attribution caveat).

| Measurement                                               | Before (CK splitkv) | After (b1) |
| --------------------------------------------------------- | ------------------- | ---------- |
| attention µs/layer, b=1 kv~768 (decode_attention_timings) | 19.5 + 2.1 combine  | TODO       |
| served attention µs/layer (lab-decode-profile, Llama c1)  | 21.7 incl. merge    | TODO       |
| Llama c1 tok/s / ITL p50 (lab-bench --quick)              | 77.8 / 12.7 ms      | TODO       |
| c16 tok/s (bound 0.97×)                                   | 852-879 band        | TODO       |
| golden c1                                                 | PASS                | TODO       |

Perf-log entry: `.procoder/perf-log.md` Phase 5p, "b1 attention: the own kernel lands" (the
decode-BW round's handed-off item). Decisions: the evaluation entry (5866edaf) + a result note
once the numbers are in.

## Open questions for the lead / next builder

- The 16-decode cross-tier bound in the invariance test is a placeholder until the first GPU run
  prints the actual max_abs.
- OLMoE (16/16 heads) decode stays on `ck_tile_fmha_pagedkv`; an equal-heads b1 variant is
  unmeasured headroom (~27.5 µs/layer there) if the grouped result holds up.
- `kTurbineB1MaxSeqs = 8` is the first-guess tier edge; `decode_attention_timings` sweeps
  b = 1..64 for both impls, so the crossover can be re-placed from its lines without a new A/B.
- The inter-kernel gap floor (~3.6 µs/boundary) is untouched by this task; rope-into-append (the
  other handed-off bit-exact item) would remove one more launch + boundary per layer.
