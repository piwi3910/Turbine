# Handoff: Phase 5p FP8 follow-ups (`p5p-fp8b`)

Branch `p5p-fp8b` from `p6b-stack` 299985a9 (user decision "5p Task 14": 1 A, 2 yes, 3 bisect). Three commits,
gate-clean each, no push. Results in the perf log 5p "S-15 follow-ups"; the open bisect decision in decisions "5p FP8
follow-ups: the per-tensor golden bisect".

## Commits

1. `489fec4b` perf(kernels): FP8-block `down` (k ≥ 8192) at 16 waves — `waves_for` in
   `kernels/rocm/src/qgemm_fp8_block.hip` (+ the eval harness). Gate ok (560).
2. `2a50d2f2` perf(kernels,model): the FP8 activation quantize fused into the W8A8 c1 decode GEMM — **kernel ABI
   v2.13** (`TURBINE_ABI_MINOR 13u`; another builder may take 13 on `p5p-gaps`: whoever merges second renumbers).
   Gate ok (563).
3. docs: the bisect (perf log, decisions entry, this handoff).

## Results (novanas GPU 0, one session; BF16 c1 11.9 ms / 82.7 tok/s)

| c1 quick       | base 299985a9          | after                      | golden                                 |
| -------------- | ---------------------- | -------------------------- | -------------------------------------- |
| FP8-block      | 7.9 ms / 122.4 (0.66×) | **7.1 ms / 135.2 (0.60×)** | c1 PASS, c16 PASS; c16 1059.6 → 1115.1 |
| FP8-dynamic    | 8.1 ms / 121.6 (0.68×) | **7.4 ms / 132.8 (0.62×)** | c1 / c16 PASS; c16 1119.1 → 1129.0     |
| FP8 per-tensor | 7.9 ms / 124.4 (0.66×) | **7.3 ms / 133.6 (0.61×)** | c1 PASS 15/16; c16 FAIL (pre-existing) |

## ABI v2.13 (what changed)

- Header: `turbine_qgemm_desc` gains trailing `float act_static_scale` (struct stays 128 bytes); `a_dtype` BF16 +
  `act_quant` FP8_TENSOR/TOKEN = unquantized activations, quantized in the call exactly as `quantize_act` (bit for
  bit). Contract §9.1 v2.13 paragraph.
- HIP: `turbine_hip_fp8_wmma` `bf16_kernel` — m = 1, prefill = 0 only; each block (4 / 2 / 1 column tiles at 2 / 4 /
  16 waves) quantizes the row into LDS, then the unchanged WMMA loop and epilogue. Wider m was built and measured
  first (any m ≤ 16): it loses from m = 2 (the row quantization repeats per block), so it was cut to m = 1.
- Rust: `QGemmContext.act_static_scale`, `QGemmKernel::quantizes_activations(cfg, rows, prefill)` (shim: minor ≥ 13
  and the bound tier's implementation supports the BF16 probe; cpu-reference: decode calls), `check_qgemm` accepts
  the form; decoder `linear` skips `quantize_act` when the bound kernel quantizes (profiled under the bound config).
- Tests: `hip_qgemm qgemm_fused_act_quant_matches_two_launches` (byte-identical outputs, edge rows, tier binding),
  `cpu::qgemm unquantized_fp8_activations_match_quantize_then_qgemm`, `shim qgemm_unquantized_activations_checked`,
  `abi_header_neutral header_declares_the_v213_qgemm_act_static_scale` (the older header tests now expect 13).
  `qgemm_decode_timings` prints `+q` rows for the fused form. Lab: `hip_qgemm` + `abi_header_neutral` PASS (job
  1006105327-0181a1f1). Served golden c1 of both W8A8 checkpoints is byte-identical to Task 14's two-launch runs.

## Bisect (item 3) — open for the lead

Culprit `b1c072ce` (the b1 decode attention kernel): a numerics-class shift on an edge-tight checkpoint, not a kernel
defect (details and options A/B/C in decisions; recommended A + B). Not fixed here (a fix would change the user-chosen
b1 numerics class).

## Notes

- Measurements ran from git-less exports of each commit (remote dirs `fp8b-base`, `fp8b-t3`, `fp8b-bisect{,2,3}` under
  `/home/piwi/turbine-ci/remote/`; not pruned automatically — delete with `rm -rf` when done). Local copies and every
  `golden1.txt` / `labbook-values.json` under the session scratchpad. lab-bench recorded each run in labbook.
- Not run: `lab-test --tier quick` on the whole tree (only `hip_qgemm` + `abi_header_neutral`); the decode profile
  (`lab-decode-profile.sh`) — per-layer µs come from `qgemm_decode_timings`.
