// turbine_abi_minor: the minor revision of the library it is linked into.
//
// libturbine_hip.so reports TURBINE_ABI_MINOR (10): besides the ABI v2 trios it
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
// sin by turbine_rope_desc.attn_factor (YaRN's attention factor).
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
  return TURBINE_ABI_MINOR;
#endif
}

} // extern "C"
