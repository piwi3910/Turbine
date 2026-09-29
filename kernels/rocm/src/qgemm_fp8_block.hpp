// turbine_hip_fp8_block: the block-scaled FP8 GEMM (qgemm_fp8_block.hip), as
// the implementation table (impl_table.cpp) enumerates and runs it.
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

// FP8_BLOCK weights (128 x 128 blocks) x BF16 activations (act_quant NONE).
bool qgemm_fp8_block_supports(const turbine_qgemm_desc *d);
int32_t qgemm_fp8_block_run(turbine_ctx *ctx, const turbine_qgemm_desc *d);

} // namespace turbine_hip
