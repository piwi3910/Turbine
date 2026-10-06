# Handoff: Phase 5p — decode-step gaps (`p5p-gaps`)

Branch `p5p-gaps` from `p6b-stack` 8b6f27c4. Each commit passed `scripts/gate.sh`. Nothing pushed. The task:
user decision 2026-10-06 "5p b1 attention II" A — re-time the b1 → CK edge, then attack the Llama b1 decode
step's inter-kernel gaps toward ~0.9× of the 640 GB/s peak. Full numbers: perf log 5p "b1 → CK split-KV edge
re-timed" and "Decode-step boundaries"; the open decision: decisions "5p decode-step gaps: what is left".

## What landed

- e2d0d8c3 — `kTurbineB1CkSeqs` 4 → 10. New perf harness `hip_ops b1_ck_edge_timings` (graph-captured, cold KV,
  RoPE fused; b1 above the edge needs a library built with the edge raised). b1 is not slower than CK through 9
  sequences at ~768–8,192 tokens. `b1_decode_rows_are_batch_invariant` moved with it (b1 side 2/3/9 bitwise, CK
  side 10/16/64 bitwise, cross-edge max |Δ| 3.906e-3 re-measured in lab run 1006094933, bound 7.8125e-3 kept).
  **Coordination:** the tiny_model `card_reference` dispatch constants
  (`crates/turbine-kernels/src/cpu/card_reference.rs`, branch p5p-t16c) are not on this base; whoever merges both
  must set its b1/CK edge to 10.
- 4fa9622c — `logits_reduce` issues 8 loads per lane before using them (`sweep_row`), same per-id steps in the
  same order: bit-identical (the test now prints an FNV digest of all HIP outputs; equal before/after on all six
  cases). Served 117 → 84 µs/step. `decode-profile-report.py --boundaries` prints the per-boundary table.
- docs: perf log sections, decisions entry (open, options A/B/C).

## Results (novanas GPU 0, bench lock, same session)

| Run                  | ITL c1 ms | BW (6.43 GB / ITL) | c1 tok/s | c16 tok/s | golden c1/c16 |
| -------------------- | --------- | ------------------ | -------- | --------- | ------------- |
| base 8b6f27c4        | 11.86     | 542 GB/s (0.847×)  | 82.8     | 852.1     | PASS / PASS   |
| edge 10 e2d0d8c3     | 11.89     | 541 (0.845×)       | 82.6     | 848.8     | PASS / PASS   |
| logits_reduce 4fa962 | 11.81     | 545 (0.851×)       | 83.2     | 855.6     | PASS / PASS   |

OLMoE c1 154.7 → 153.9 tok/s, c16 611.1 → 609.7 (noise), golden c1 PASS.

## The key finding

The traced step (12.44–12.50 ms, "~1.0 ms of gaps", 0.81×) is inflated by rocprofv3: ~1.9 µs per graph-replay
dispatch and ~170 µs per `hipGraphLaunch` (gapbench: 231-node graph, 3.52 µs traced gap vs ~1.6 µs unprofiled;
launch 5.5 vs 179 µs). The real step is the ITL, ~11.8 ms (0.85×): GEMMs 10.75 ms, attention 0.30, other kernels
0.27, ~0.44 ms between kernels (incl. ~0.07 host). 0.9× (11.16 ms) is not reachable bit-exactly; see decisions.

## Not landed / evaluated

- A two-kernel `logits_reduce` prepass (steps 1–2 spread over 32 blocks per row; digest-identical) — no faster
  than the hoisted loads at the served shapes. The remaining nucleus cost is the gather + bitonic sort.
- ROCm runtime env knobs on gapbench (graph batch size, max batch size, device kernargs, skip kernarg copy,
  doorbell, graph queues, active wait): none faster; `ROC_SYSTEM_SCOPE_SIGNAL=0` hangs.

## Scratch (not in the tree)

gapbench source/scripts under the session scratchpad; remote `/home/piwi/turbine-ci/remote/agent-a28047e2f0e0585b8/
{scratch,kbuild-edge,kbuild-lr,src-lr}` (edge-raised and logits variants; safe to delete).
