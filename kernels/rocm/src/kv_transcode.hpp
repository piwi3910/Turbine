// The implementations of the ABI v2.11 op turbine_kv_transcode
// (kv_transcode.hip), as the implementation table (impl_table.cpp) enumerates
// and runs them. Each run validates its descriptor and refuses one it does not
// support with TURBINE_E_UNSUPPORTED and a message naming the implementation.
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

// turbine_hip_fp8: the FP8 e4m3 codec over BF16 pages, encode and decode.
bool kv_transcode_fp8_supports(const turbine_kv_transcode_desc *d);
int32_t kv_transcode_fp8_run(turbine_ctx *ctx,
                             const turbine_kv_transcode_desc *d);

} // namespace turbine_hip
