## P6: FP8 GEMM — provider evaluation (kernel reuse rule)

Date: 2026-09-29 (Phase 6a Task 12). Card: R9700 (gfx1201), GPU 0, ROCm 7.14.1, hipBLASLt 1.4.1
(100401), CK at the pinned `cd9574023093742434e8c992d13b89ab9a6c1cf8` (therock-7.14.1).
Harness: `kernels/rocm/tools/qgemm_eval.cpp` (+ `qgemm_eval_ck.cpp` for the CK candidates),
built with `-DTURBINE_BUILD_QGEMM_EVAL=ON`, run under `scripts/bench-lock.sh` with
`ROCR_VISIBLE_DEVICES=0`. Per candidate, shape and m: hipBLASLt's first heuristic answer and the
best of its first 8 answers, median of 5 rounds × 20 calls, rotating over 2–8 copies of the weight (as many as 512 MiB allows, at most 8); correctness on 6 sampled rows against a host reference with the CPU provider's semantics
(`cpu::qgemm`: e4m3 values × scales, exact products, one rounding to BF16); tolerance one BF16
rounding (2^-8 relative) plus summation order; "row-invariant" = row 0 of the first answer is
bitwise the same at every m.

Shapes (Llama-3.2-3B, fused as the executor runs them): qkv 5120×3072, o 3072×3072,
gate_up 16384×3072, down 3072×8192.

### Candidates

| Candidate | Builds | Correct | Notes |
|---|---|---|---|
| hipBLASLt BF16×BF16→BF16 on the dequantized weight (today's path; W8A16 via dequantize) | yes | — (baseline) | timing baseline |
| hipBLASLt e4m3×e4m3→BF16, scalar A/B scales (`SAB`: FP8_TENSOR × FP8_TENSOR) | yes | yes, max \|Δ\| ≤ 1.9e-2 at outputs ≈ 5–9 (one BF16 ulp) | 8 solutions per shape |
| hipBLASLt e4m3×e4m3→BF16, vector scales `OUTER_VEC_32F` on both (`SABV`: FP8_CHANNEL × FP8_TOKEN) | yes | yes, max \|Δ\| ≤ 2.6e-2 at outputs ≈ 9 | 8 solutions per shape; row-invariant on every shape |
| hipBLASLt mixed scalar × vector (FP8_CHANNEL × FP8_TENSOR, FP8_TENSOR × FP8_TOKEN) | — | — | no solution on gfx1201 (heuristic returns 0 algorithms); served by broadcasting the scalar to a vector and running `SABV` |
| hipBLASLt FP8 with vector scales and F32 D | — | — | no solution on gfx1201 (found by `hip_qgemm`): `hipblaslt_fp8` supports BF16 out only |
| CK `ck_tile` `gemm_quant` RowColQuant (per-token × per-channel), decode tile 16×64×256 and prefill tile 128×128×128, BF16 C | yes (gfx1201, OCP e4m3) | **no**: outputs of magnitude 1e35–1e38 on every shape; m = 1 (and m = 16 with the prefill tile) refused by `IsSupportedArgument` (kPadM = false); CK's own example (`tile_example_gemm_quant -quant_mode=rowcol`) fails its CPU verification on gfx1201 too | reported "times" (e.g. o m=2048 90 µs = 430 TFLOP/s, above the card's FP8 peak) are not real work |
| CK `ck_tile` `gemm_quant` TensorQuant (scalar × scalar), same tiles | yes | **no** (same garbage outputs) | |
| vLLM / aiter ROCm scaled-mm | not built | — | vLLM's FP8 linear on ROCm dispatches to `torch._scaled_mm` (hipBLASLt, i.e. the candidates above) or to aiter, whose kernels are gfx942/gfx950 assembly; no separate gfx12 provider exists |
| Activation quantization: CK `add_rmsnorm2d_rdquant` (fused norm + per-row quant) | not built | — | the v2.9 `quantize_act` descriptor carries no norm inputs, so fusing needs an ABI addition (lead-owned); CK's rdquant also rounds by its own conversion and scale reciprocal, which the bit-exact reference rule excludes |
| Activation quantization: Turbine elementwise kernel (`src/qgemm_quantize.hpp`) | yes | **bit-exact** (codes and scales) for FP8_TOKEN, FP8_GROUP128, FP8_TENSOR at every size, including zero rows (floor), saturation, ties | own kernel: no provider fits the ABI op |

### µs per call, GPU 0 (first heuristic answer / best of 8)

| shape | m | BF16 | FP8 SAB | FP8 SABV |
|---|---|---|---|---|
| qkv | 1 | 64.8 / 54.6 | 48.4 / 26.7 | 66.8 / 35.4 |
| qkv | 16 | 56.5 / 55.2 | 36.7 / 27.6 | 57.9 / 36.2 |
| qkv | 128 | 58.5 / 58.5 | 49.0 / 32.4 | 60.4 / 44.1 |
| qkv | 2048 | 454.7 / 454.7 | 270.1 / 264.5 | 406.1 / 385.3 |
| o | 1 | 51.3 / 39.5 | 27.0 / 18.6 | 56.5 / 23.1 |
| o | 16 | 40.0 / 40.0 | 25.2 / 19.3 | 41.0 / 23.4 |
| o | 128 | 48.4 / 46.8 | 42.6 / 22.3 | 42.3 / 26.1 |
| o | 2048 | 302.1 / 301.3 | 164.3 / 164.3 | 256.1 / 248.5 |
| gate_up | 1 | 168.3 / 162.4 | 97.1 / 90.2 | 124.3 / 117.9 |
| gate_up | 16 | 172.8 / 166.5 | 105.7 / 92.9 | 130.8 / 127.1 |
| gate_up | 128 | 233.8 / 221.4 | 167.6 / 107.1 | 182.1 / 160.3 |
| gate_up | 2048 | 1644.6 / 1596.2 | 851.3 / 851.3 | 1238.5 / 1220.7 |
| down | 1 | 111.4 / 96.5 | 61.7 / 44.1 | 79.5 / 58.2 |
| down | 16 | 112.9 / 97.2 | 66.1 / 44.6 | 83.9 / 59.5 |
| down | 128 | 152.1 / 100.0 | 74.3 / 46.6 | 90.5 / 70.0 |
| down | 2048 | 1005.3 / 1005.3 | 581.9 / 474.6 | 1145.5 / 968.4 |

Activation quantization (Turbine kernel, µs): FP8_TOKEN k=3072: 9.3 / 8.1 / 8.4 / 61.3 at
m = 1 / 16 / 128 / 2048; k=8192: 15.8 / 15.6 / 16.8 / 125.1. FP8_GROUP128 k=3072: 11.9 / 11.4 /
12.1 / 89.0; k=8192: 26.4 / 26.0 / 27.3 / 212.9. FP8_TENSOR k=3072: 6.3 / 6.4 / 6.7 / 53.9;
k=8192: 11.9 / 11.8 / 12.5 / 92.7.

Per decode layer at m = 16 (four GEMMs; the BF16 path runs its tuned table, ≈ best): BF16 358.9 µs;
FP8 SABV 313.6 µs with the first heuristic answers, 246.2 µs with the best, plus ≈ 40 µs of
activation quantization (three k=3072 and one k=8192 token quantizations).

### Pick

- `hipblaslt_fp8` (provider hipblaslt): FP8_TENSOR / FP8_CHANNEL weights × FP8_TENSOR / FP8_TOKEN
  activations through hipBLASLt's FP8 kernels with `SCALAR_32F` (tensor × tensor) or
  `OUTER_VEC_32F` (vector × vector) scale modes; a mixed pairing broadcasts its scalar scale to a
  vector (a device buffer of the context, ≥ 65,536 floats, grown only outside graph capture);
  BF16 out only; k and lda multiples of 16. The only correct provider on gfx1201.
- Activation quantization: own kernel `turbine_hip` (`src/quantize_act.hip`,
  `src/qgemm_quantize.hpp`) — no provider takes the v2.9 `quantize_act` op; CK's fused
  norm+quant would need an ABI change and is not bit-exact with the reference rule.
- Findings to act on: (1) the first heuristic answer is 1.2–2.4× slower than the best of hipBLASLt's
  top 8 at decode sizes (qkv, o, down), so FP8 shapes need pinned solutions (a tuned FP8 table) to
  reach the S-20 targets — without it FP8 decode GEMM time is ≈ BF16's; (2) SABV at m = 2048 is no
  faster than BF16 on down (968 vs 1005 µs) and only 1.2–1.3× on the others (per-token × per-channel
  checkpoints gain in decode, little in prefill); SAB (per-tensor) is 1.7–2.1× faster than BF16
  at m = 2048. (3) CK `gemm_quant` RowCol/Tensor is not usable on gfx1201 at this CK pin.
