// The MXFP4 implementations of the ABI v2.9 ops (qgemm_mxfp4.hip,
// quantize_act_mxfp4.hip), as the implementation table (impl_table.cpp)
// enumerates and runs them, plus the launchers the evaluation harness
// (tools/qgemm_mxfp4_eval.cpp) times without a context. Each run validates its
// descriptor and refuses one it does not support with TURBINE_E_UNSUPPORTED
// and a message naming the implementation.
//
// Provider choice: decision "P6: MXFP4 GEMM — provider evaluation (kernel
// reuse rule)" in .procoder/ask/decisions.md.
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

// turbine_hip_mxfp4: TURBINE_QSCHEME_MXFP4 weights ([n, k/2] E2M1 codes, low
// nibble first; [n, k/32] E8M0 exponents) x BF16 activations (act_quant NONE,
// or MXFP4_EMULATED after turbine_quantize_act), BF16 or F32 out. The codes
// are decoded to BF16 in registers (exact: every E2M1 value, and every E2M1
// value times a power of two, is a BF16 value) and multiplied on the RDNA4
// matrix cores (v_wmma_f32_16x16x16_bf16) with F32 accumulation. k a multiple
// of 64, lda a multiple of 8, a and b 16-byte aligned.
bool qgemm_mxfp4_supports(const turbine_qgemm_desc *d);
int32_t qgemm_mxfp4_run(turbine_ctx *ctx, const turbine_qgemm_desc *d);

// turbine_hip_mxfp4: MXFP4_EMULATED quantize-dequantize of BF16 or F32 rows
// (one wave per 32-column group; Quark's `even` E8M0 rule, E2M1 round to
// nearest even), bit-exact with cpu::quant; x and out may alias.
bool quantize_act_mxfp4_supports(const turbine_quantize_act_desc *d);
int32_t quantize_act_mxfp4_run(turbine_ctx *ctx,
                               const turbine_quantize_act_desc *d);

// The kernel tiles of turbine_hip_mxfp4. Auto, what the run uses, takes Large
// for every prefill call (d->prefill) and picks by m otherwise:
//   Small  -- m <= 16: one 16-row fragment, one column fragment per wave, k
//             split over 8 waves (decode);
//   Medium -- m <= 64: 2 (m <= 32) or 4 row fragments, 2 column waves, k
//             split over 4;
//   Large  -- 128 x 128 block tiles, activations staged in LDS.
// Each tile sums a row in its own order (the k split), so a row's bits depend
// on the tile; within Large they depend on nothing but the row, so prefill
// calls are batch invariant (prefix reuse needs it) and decode calls are not.
enum class Mxfp4Tile { Auto, Small, Medium, Large };

// Enqueues the product of d (already validated: qgemm_mxfp4_supports and
// non-NULL operands) on stream with the given tile.
hipError_t launch_qgemm_mxfp4(hipStream_t stream, const turbine_qgemm_desc *d,
                              Mxfp4Tile tile);

// Enqueues the MXFP4 quantize-dequantize of d (validated) on stream.
hipError_t launch_quantize_mxfp4(hipStream_t stream,
                                 const turbine_quantize_act_desc *d);

} // namespace turbine_hip
