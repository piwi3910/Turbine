// The implementations of the ABI v2.9 ops (qgemm.cpp, qgemm_fp8_wmma.hip,
// quantize_act.hip), as
// the implementation table (impl_table.cpp) enumerates and runs them. Each
// run validates its descriptor and refuses one it does not support with
// TURBINE_E_UNSUPPORTED and a message naming the implementation.
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

// hipblaslt_fp8: FP8_TENSOR / FP8_CHANNEL weights x FP8_TENSOR / FP8_TOKEN
// activations on hipBLASLt.
bool qgemm_fp8_supports(const turbine_qgemm_desc *d);
int32_t qgemm_fp8_run(turbine_ctx *ctx, const turbine_qgemm_desc *d);

// turbine_hip_fp8_wmma (qgemm_fp8_wmma.hip): the same schemes, k a multiple of
// 128; decode steps of up to 16 rows on an own WMMA kernel, every other call
// on qgemm_fp8_run.
bool qgemm_fp8_wmma_supports(const turbine_qgemm_desc *d);
int32_t qgemm_fp8_wmma_run(turbine_ctx *ctx, const turbine_qgemm_desc *d);

// turbine_hip: FP8_TENSOR, FP8_TOKEN and FP8_GROUP128 e4m3 quantization of
// BF16 or F32 rows (qgemm_quantize.hpp).
bool quantize_act_fp8_supports(const turbine_quantize_act_desc *d);
int32_t quantize_act_fp8_run(turbine_ctx *ctx,
                             const turbine_quantize_act_desc *d);

} // namespace turbine_hip
