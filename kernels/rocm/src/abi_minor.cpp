// turbine_abi_minor: the minor revision of the library it is linked into.
//
// libturbine_hip.so reports TURBINE_ABI_MINOR (18): besides the ABI v2 trios it
// exports the optional add_rmsnorm trio (rmsnorm.cpp), the v2.1 context
// options (context.cpp: the tuned GEMM table switch), logits_reduce
// (logits_reduce.hip), the graph functions (graph.cpp), the v2.3 pinned host
// memory and event functions (memory.cpp), the v2.4 implementation group
// (impl_exports.cpp), the v2.5 copy streams (copy_stream.cpp) and the v2.6
// native stream handle (copy_stream.cpp) with the sharded RMSNorm trios
// (sharded_norm.hip), the v2.7 host-mapped memory and collectives and the v2.8
// device-sequenced collective step (hostmem.hip), and the v2.9 quantized GEMM
// (qgemm.cpp) and activation quantization (quantize_act.hip) trios; the v2.9
// FP8 KV scales of turbine_attention_paged_desc are read only with F8E4M3
// pages, which paged attention refuses until an implementation takes them.
// v2.10 adds no symbol: the rope kernels (elementwise.hip) multiply cos and
// sin by turbine_rope_desc.attn_factor (YaRN's attention factor). v2.11 adds
// the KV transcode trio (kv_transcode.hip) and the host-to-device copy kernel
// (copy_kernel.hip). v2.12 adds no symbol: paged decode attention reads the
// rope_* fields of its descriptor (RoPE fused into the b1 kernel's append;
// every other decode path rotates first with the rope kernels). v2.13 adds no
// symbol: turbine_hip_fp8_wmma takes BF16 activations with an FP8 act_quant on
// small decode calls and quantizes them in the kernel (qgemm_fp8_wmma.hip;
// turbine_qgemm_desc.act_static_scale). v2.14 adds no symbol: vllm_wvsplitk
// takes the SwiGLU input of turbine_gemm_desc (a_silu_mul / a_up_offset) and
// stages silu_mul(gate, up) as its activations (gemm_wvsplitk.hip); hipblaslt
// refuses it. v2.15 adds no symbol: vllm_wvsplitk takes the residual-add
// epilogue and the RMSNorm input of turbine_gemm_desc (c_residual,
// a_norm_weight / a_norm_eps) in decode steps; hipblaslt refuses them.
// v2.18 adds the Gated DeltaNet trios (gdn.hip: gdn_prefill, gdn_decode) and
// the causal conv1d trios (conv1d.hip: conv1d_prefill, conv1d_update); v2.16
// and v2.17 add no symbol of this group.
// libturbine_hip_v23.so, compiled with TURBINE_V23_BUILD, is the same kernels
// without impl_exports.cpp and reports 3, so a caller keeps the library's own
// choice of implementation (the fallback the v2.4 group is optional against)
// and resolves neither the v2.5 copy streams (the tiered KV cache then stays
// on the device) nor the v2.6 group (one device per model) nor the v2.7 group
// (no hostmem collective backend); its rope ignores attn_factor.
#include "turbine_hip.hpp"

extern "C" {

uint32_t turbine_abi_minor(void) {
#ifdef TURBINE_V23_BUILD
  return 3u;
#else
  // The v2.16 expert variants are implemented by turbine_hip_moe_quant (the
  // FP8 block / FP8 tensor / MXFP4 experts, biases, the gpt-oss activation,
  // interleaved gate_up, the shared expert) and TOPK_THEN_SOFTMAX / router_bias
  // by moe_route; BF16 experts with those extras are refused by
  // turbine_moe_experts_supported.
  return TURBINE_ABI_MINOR;
#endif
}

} // extern "C"
