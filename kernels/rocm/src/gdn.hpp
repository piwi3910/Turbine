// The implementations of the ABI v2.18 ops turbine_gdn_prefill /
// turbine_gdn_decode (gdn.hip) and turbine_conv1d_prefill /
// turbine_conv1d_update (conv1d.hip), as the implementation table
// (impl_table.cpp) enumerates and runs them. Each run validates its descriptor
// and refuses one it does not support with TURBINE_E_UNSUPPORTED and a message
// naming the implementation.
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

// llama_cpp_gated_delta_net: BF16 q, k, v, out, k_dim == v_dim in {32, 64,
// 128}, v_heads a multiple of k_heads.
bool gdn_supports(const turbine_gdn_desc *d);
int32_t gdn_prefill_run(turbine_ctx *ctx, const turbine_gdn_desc *d);
int32_t gdn_decode_run(turbine_ctx *ctx, const turbine_gdn_desc *d);

// aiter_causal_conv1d_split_qkv (prefill) and llama_cpp_ssm_conv (update):
// BF16, window 4.
bool conv1d_supports(const turbine_conv1d_desc *d);
int32_t conv1d_prefill_run(turbine_ctx *ctx, const turbine_conv1d_desc *d);
int32_t conv1d_update_run(turbine_ctx *ctx, const turbine_conv1d_desc *d);

} // namespace turbine_hip
