# Handoff: Phase 5p — b1 attention II (`p5p-b1attn2`)

Branch `p5p-b1attn2` from `p6b-stack` 97050b3c. Each commit passed the gate. Nothing pushed. The
task was user decision 2026-10-06 B ("5p b1 attention: result"): iterate the own batch-1 decode
attention kernel toward ~10 µs/layer. The perf log's 5p section "b1 attention II" has the full
table and the evaluated-but-rejected variants.

## What landed

- 37c24412: the P·V loop issues all 32 V loads of a round before it uses the first, so a round
  no longer waits on 32 latencies in a row. It was ~9 µs of the ~18 µs kernel. Bit-identical.
- c47250af: the combining workgroup stages the split maxima and sums in LDS, then loads its 8
  accumulators together. Saves ~0.9 µs. Bit-identical.
- e90631a4: RoPE fused into paged decode attention, **kernel ABI v2.12**:
  - Descriptor: `turbine_attention_paged_desc` gains `rope_positions`, `rope_inv_freq`,
    `rope_rotary_dim` and `rope_attn_factor`; `TURBINE_ABI_MINOR` is now 12.
  - Kernel paths: the b1 kernel rotates q while staging it and the appended K row in the append,
    with the rope kernel's exact operations. Every other path runs the rope kernels in place
    first.
  - Rust side: `PagedAttentionContext.rope: Option<FusedRope>` and
    `AttentionKernel::fuses_decode_rope` (shim: minor ≥ 12; the cpu reference fuses as rope then
    attention). The decoder skips its rope op on decode steps when the kernel fuses it, except
    while tracing.
  - The stub library `TURBINE_STUB_GFX942_V211` keeps reporting 11. The contract
    (`interfaces.md` §9.1) has the v2.12 paragraph.
  - The bits are unchanged.

## Results (novanas GPU 0, bench lock, same day; base 97050b3c)

| Step       | served attn µs/layer | step ms | eff. BW        | c1 tok/s | c16 tok/s | golden c1/c16 |
| ---------- | -------------------- | ------- | -------------- | -------- | --------- | ------------- |
| base       | 16.88 + 2.76 rope    | 12.85   | 501 GB/s 0.78× | 81.2     | 851.2     | PASS/PASS     |
| V loads    | 9.84 + 2.96 rope     | 12.60   | 510 0.80×      | 82.0     | 851.9     | PASS/PASS     |
| combine    | 8.96 + 2.96 rope     | 12.56   | 512 0.80×      | 82.4     | 851.8     | PASS/PASS     |
| rope fused | 10.64 (incl. rope)   | 12.44   | 517 0.81×      | 82.9     | 852.9     | PASS/PASS     |

Tests (lab, green at e90631a4): `hip_ops::{paged_b1_decode_matches_cpu,
paged_decode_splitkv_matches_cpu, paged_prefill_ck_128_matches_cpu,
paged_decode_fused_rope_matches_separate}` and `hip_batch_invariance::{b1_decode_rows_are_batch_invariant,
paged_attention_rows_are_batch_invariant}` (bitwise within each side of the edge, cross-edge bound
7.8125e-3 unchanged). Host: `tiny_model decode_fused_rope_matches_separate_rope` and
`abi_header_neutral header_declares_the_v212_fused_rope_fields`. Mutant killed: an append that
writes the K row unrotated.

## Numerics note (for p5p-t16b, the CPU reference that models b1)

None of the three steps changes b1's math: same P precision (F32), same reduction order, same
split and wave decomposition. The fused RoPE uses the rope kernel's operations, so q and K enter
attention with the same bits as before. The CPU reference's own fused path is literally its rope
op followed by its attention.

## Not landed / evaluated

These were measured in a standalone harness (`target/b1attn2/h/`, in this worktree only).

- **Split and wave shapes**: S4, S16, W16, W4, W2 and W1 do not beat W8 S8 once the two fixes are
  in, and every one of them changes the bits.
- **Merged staging barrier**: no gain.
- **L2 prefetch**: the barrier waits for it; +2.5 µs at ~830 keys.
- **K and V loads together at round top**: spills registers.
- Remaining inside the kernel (cold KV, ~830 keys): K loads ~3.2 µs (DRAM latency), V ~2.2,
  combine ~1.1, the fused rope's f64 trig ~1 µs plus one extra barrier, launch ~1 µs.

## What is next (options for the lead)

The ≥ 90 tok/s c1 mark cannot be reached through attention. Attention plus RoPE is 0.30 ms of a
12.44 ms step; 90 tok/s needs ~11.1 ms. Options:

- A) Close the b1 work here (attention ≤ ~10 µs, 0.81× reached) and move to the other kernel
  boundaries (~1.0 ms of gaps per step). Recommended.
- B) Shave the fused rope's ~1.7 µs: issue the q loads before the trig and drop the extra
  barrier. ≤ ~0.03 ms/step.
- C) Long contexts: a prefetch that does not stall at a barrier, e.g. a next-round K prefetch
  inside the round loop. Helps 8k contexts (−26 µs of round time per call) but not the served
  ~800-token case.

Note on the CK hand-off: `kTurbineB1CkSeqs = 4` dates from the pre-iteration kernel. b1 is now
~2× faster at b = 1, so the crossover may have moved up. Re-timing b = 2–8 with
`decode_attention_timings` is cheap. Any move of the edge changes which rows are bitwise
invariant, so the invariance test edge must move with it.

## Pending check

The coordinator asked for `scripts/lab-test.sh novanas -- -p turbine-model --test tiny_model` once
p5p-t16b (the CPU reference modelling b1's F32 P) is merged. That run is not done yet.
