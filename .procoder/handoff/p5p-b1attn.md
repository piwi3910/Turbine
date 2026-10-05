# Handoff: Phase 5p — the batch-1 attention own kernel (`p5p-b1attn`)

Branch `p5p-b1attn`, `p6b-stack` a004fd93 merged in (119e0b17). Every commit gate-clean; the tree is
the task's exit state. Logs under `target/b1attn/`, the served trace under
`target/decodebw/b1attn-llama/`. No push.

## What landed

- 5866edaf — the reuse evaluation (decisions "5p b1 attention: provider evaluation"): llama.cpp
  `fattn` not reusable (contiguous ggml K/V, no page table, no ragged decode), CK splitkv fits but is
  occupancy-bound at b1, CK pagedkv slower, hipBLASLt has no attention.
- b1c072ce — impl `turbine_hip_b1_decode` (`kernels/rocm/src/paged_attention_b1.hip`): one launch per
  layer, grid (sequences, KV heads, 8 splits), append fused into the owning split, split combine fused
  into the last-arriving workgroup (ascending split order), P in F32. Grouped heads (group ≤ 8), BF16
  flat pages, any page size; first in gfx1201's decode order.
- ac729455 — fixes from the first GPU run: (1) selection is per op config (the shim probes
  `num_seqs = 1`, the registry binds one index), so a sequence tier in `supports` would have refused
  bound c16 batches; (2) counters moved to their own always-zero buffer (a second graph's partials
  could land on the first graph's counters); (3) the invariance helper sized K/V by q_heads.
- 17cccc2b — the invariance test compares 24 rows (one row let a mutant survive).
- 7d3c8cad — inside the b1 entry, batches of `kTurbineB1CkSeqs` = 4+ sequences run CK splitkv when CK
  serves the page size (b1 loses from b = 8). Invariance: bitwise within each side, ≤ 7.8125e-3
  across the edge (measured 3.906e-3).

## Results (novanas GPU 0, bench lock, same-day baseline a004fd93)

| Measurement                       | CK (base)         | b1 + dispatch (7d3c8cad)    |
| --------------------------------- | ----------------- | --------------------------- |
| op µs/layer b=1 kv~~768 / kv~~8k  | 42.5 / 320.3      | 23.9 / 83.8                 |
| served attention µs/layer (c1)    | 21.7 + 2.9 append | 16.8 (append fused)         |
| step / gaps (ms), served c1 trace | 13.09 / 1.32      | 12.73 / 1.12                |
| effective BW                      | 491 GB/s (0.77×)  | 505 GB/s (0.79×)            |
| c1 tok/s / ITL p50                | 80.0 / 12.29      | 81.2 / 12.10                |
| c16 tok/s                         | 853.2             | 853.0 (b1 everywhere 833.5) |
| golden c1 / c16                   | PASS / PASS       | PASS / PASS                 |

Tests (lab, green): `hip_ops::paged_b1_decode_matches_cpu`, `paged_decode_splitkv_matches_cpu`,
`paged_prefill_ck_128_matches_cpu`, `hip_batch_invariance::b1_decode_rows_are_batch_invariant`,
`paged_attention_rows_are_batch_invariant`. Mutants killed: skip-last-split combine, V-less append,
grid-dependent split count.

## Open (decisions "5p b1 attention: result", for the lead)

The decision-B targets (≤ ~10 µs/layer, ~0.81×, ≥ 90 tok/s) are missed: A) keep as landed and go to
rope-into-append / boundary work (recommended), B) iterate the kernel (split count, rope fused),
C) revert to CK. Note: `implementation()` reports `turbine_hip_b1_decode` for the op config even
where the entry runs CK splitkv (b ≥ 4) — `/turbine/v1/status` cannot show the per-batch split.
`kTurbineB1CkSeqs = 4` is from the 1/4/8/16/64 timing points; b = 2, 3 are interpolated.
