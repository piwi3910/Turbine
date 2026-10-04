# Handoff: Phase 5p Task 18 — decode bandwidth (2026-10-05)

Branch `p5p-decodebw`, from `p6b-stack` 4e76f676. Tip `aba93eba`. Everything gate-clean; the
tree is the task's exit state.

## What the profile found (the S-19 premise was wrong about the peak)

`scripts/lab-decode-profile.sh` (committed: native serve with `ROCP_TOOL_ATTACH=1`, a c1
`--ignore-eos` load, `rocprofv3 --attach` for 3 s) + `scripts/decode-profile-report.py` (splits
steps at each `logits_reduce_kernel`). Traces under `target/decodebw/prof-base-<model>/`.

- **The R9700's real peak is 640 GB/s** (256-bit, 20 Gbps GDDR6), not the ~900 the S-19 draft
  assumed. Spec S-19 amended (aba93eba); criterion unchanged (recorded factor, golden-safe,
  c16 bound).
- **Llama c1** (13.09 ms span, 11.78 busy, 1.32 gaps; 227 steps): 6.43 GB → 491 GB/s = 0.77×.
  The four GEMVs per layer: gate_up 160 µs (629 GB/s), down 85 µs (592), qkv 56 µs (566),
  o_proj 34 µs (551); LM head 1 245 µs (633). Gaps ~3.6 µs × ~258 boundaries = 10 % of the step
  even inside the graph replay; attention 21.6 µs/layer (CK splitkv + merge, launch-bound);
  rmsnorm is CK (1.8 µs). No cheap Llama landing; nothing landed.
- **OLMoE c1** (8.25 ms span; 361 steps): 2.36 GB → 286 GB/s = 0.45×. Weakest reads on either
  model: small-m MoE gate_up 428 GB/s, down 490 GB/s (`moe_wmma_gemv`, one thread per column),
  moe_scatter 23.9 µs/layer (one 256-thread CTA per token), Turbine rmsnorm 6.7 µs (OLMoE's
  plain n=2048 is below the CK bucket).

## Landed (both bit-exact, one at a time, OLMoE only)

| Landing                                                          | commit   | c1 tok/s  | ITL p50  | c16       | golden c1  |
| ---------------------------------------------------------------- | -------- | --------- | -------- | --------- | ---------- |
| S-2 baseline                                                     | —        | 126.0     | 7.8 ms   | 615.4     | PASS       |
| `kGemvWaves` 4→8 (mapping only; per-column WMMA chain unchanged) | f0d83051 | 129.9     | 7.48     | 623.3     | PASS 15/16 |
| `moe_scatter` one CTA per 256 columns (same add order)           | 9a413bbd | **135.1** | **7.18** | **627.7** | PASS 15/16 |

Effective OLMoE b1 weight-read bandwidth 302 → **329 GB/s (×1.09)** on the ITL baseline
(286 → 297 re-traced on the profile workload); c16 +2.0 % (bound 0.97×). Op-level A/B
(decode_op_timings, b=1, 8 routed rows): moe_experts 295.7 → 272.8 (waves) → 264.5 µs
(scatter). Waves sweep 2/4/8/16: 279.2/295.7/272.8/276.5; `kGemvUnroll=16` 483 µs — rejected
also because it reorders the F32 chain (OLMoE invariance, 2026-09-27 answer (c)).
Tests green per landing: `moe_experts_small_m_matches_cpu`, `moe_experts_grouped_matches_cpu`,
seven `hip_batch_invariance` tests; labbook runs db445c12, fb11a0bf.

**Attribution caveat (after-profile at aba93eba, `target/decodebw/prof-after-olmoe/`):** the
served step 8.25 → 7.95 ms is entirely the scatter's (23.8 → 5.3 µs × 16); the served
`moe_wmma_gemv` time does not move (156.6 → 156.1 µs). The waves landing's reproducible effect
is on the cache-cycled op-timing bench only; serving reads the expert weights warmer than the
op bench cycles them. Keep it (harmless, op-level −8 %, bit-exact), but a K-split GEMV must be
judged served, not op-level only.

## Open options (measured, not landed)

- **K-split small-m GEMV** with a fixed reduction tree: the only way past the
  single-thread-per-column ~430–490 GB/s ceiling; batch-invariant by construction if the tree
  is fixed, but a new numerics class → golden re-check. Biggest OLMoE item (~0.9 ms/step).
- **Routing-chain fusion** (router GEMM + topk + group + positions; the scatter's zero-add):
  ~0.4 ms/step OLMoE, mostly launch/gap time.
- **CK rmsnorm for plain n=2048**: blocked — `rmsnorm_sharded` asserts bitwise equality
  against the plain kernel at 2048 (a Phase 5 tp constraint). Needs a numerics decision first.
- **Llama**: rope+paged_append fusion (~0.2 ms), a batch-1 attention selection (CK splitkv's
  fixed ~25 µs/layer), gap reduction (no lever found: kernarg placement is a no-op, gaps are
  uniform). Llama b1 remains at the profile's recorded state.

## Next

- If the phase exit (plan Task 19) wants the S-19 criterion ticked on OLMoE's factor, the
  evidence is the perf log "Task 18" section + labbook rows above; the after-profile for the
  traced bandwidth is `target/decodebw/prof-after-olmoe/` (run at aba93eba).
- Task 19 items and the 5p tasks owned by other builders are untouched here. No push; branch
  ready to merge into the 5p stack after the lead reviews the Llama "no landing" call.
