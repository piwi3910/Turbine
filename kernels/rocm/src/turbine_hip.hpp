// Internal definitions shared by the libturbine_hip.so translation units.
//
// Ownership: a turbine_ctx owns its compute stream, its hipBLASLt handle, the
// GEMM workspace, the attention seqstart scratch and the MoE scratch;
// turbine_ctx_destroy releases all of them after draining the stream. Device
// pointers passed in descriptors belong to the caller and are never retained
// beyond the call.
#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

// Only the ABI symbols are exported: every declaration of turbine_kernels.h
// gets default visibility, everything else in the library is hidden
// (-fvisibility=hidden plus a linker version script).
#pragma GCC visibility push(default)
#include "turbine_kernels.h"
#pragma GCC visibility pop

namespace turbine_hip {

// GEMM workspace handed to hipBLASLt
// (HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES).
constexpr size_t kGemmWorkspaceBytes = 32u << 20;
// hipBLASLt algorithm cache bound; the cache is cleared when it fills.
constexpr size_t kGemmAlgoCacheEntries = 1024;

// (m, n, k, lda, ldb, ldc, trans_b, c_dtype)
using GemmKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int32_t, int32_t>;

} // namespace turbine_hip

struct turbine_ctx {
  int device = 0;
  hipStream_t stream = nullptr;
  hipblasLtHandle_t blaslt = nullptr;
  void *workspace = nullptr;
  // Device int32[4]: seqstart_q {0, q_len} then seqstart_k {0, kv_len}.
  int32_t *seqstart = nullptr;
  // Device architecture without target features, e.g. "gfx1201".
  std::string arch;
  // MoE intermediates when the caller passes no (or too small a) workspace;
  // grown on demand, never shrunk.
  void *moe_scratch = nullptr;
  size_t moe_scratch_bytes = 0;
  // hipBLASLt returned a grouped-GEMM solution for this device at creation.
  bool moe_grouped = false;
  std::map<turbine_hip::GemmKey, hipblasLtMatmulAlgo_t> gemm_algos;
  std::mutex error_mutex;
  std::string last_error;
};

namespace turbine_hip {

// Records msg as the context's last error and returns code.
int32_t fail(turbine_ctx *ctx, int32_t code, const std::string &msg);

// Maps a HIP runtime error to TURBINE_E_OUT_OF_MEMORY / TURBINE_E_DEVICE with a
// message starting with hipGetErrorName; returns TURBINE_OK on hipSuccess.
int32_t check_hip(turbine_ctx *ctx, hipError_t err, const char *what);

// Maps a hipBLASLt status to TURBINE_E_LIBRARY (TURBINE_E_OUT_OF_MEMORY for
// HIPBLAS_STATUS_ALLOC_FAILED); returns TURBINE_OK on success.
int32_t check_blaslt(turbine_ctx *ctx, hipblasStatus_t status,
                     const char *what);

// Makes ctx's device current on the calling thread.
int32_t enter(turbine_ctx *ctx);

// hipGetErrorName-style name of a hipBLAS status.
const char *blaslt_status_name(hipblasStatus_t status);

// Turbine HIP kernels (elementwise.hip). Each enqueues on ctx->stream and
// returns a TURBINE_* code; the descriptor was validated by the caller.
int32_t launch_rmsnorm(turbine_ctx *ctx, const turbine_rmsnorm_desc *d);
bool rmsnorm_fallback_supported(const turbine_rmsnorm_desc *d);
int32_t launch_add_rmsnorm(turbine_ctx *ctx, const turbine_add_rmsnorm_desc *d);
bool add_rmsnorm_fallback_supported(const turbine_add_rmsnorm_desc *d);

// Paged-KV kernels (paged_attention.hip); the descriptor was validated by the
// caller (paged_attention.cpp) and has total_q > 0.
// Writes k_new/v_new into their page slots of d->kv_layer.
int32_t launch_paged_append(turbine_ctx *ctx,
                            const turbine_attention_paged_desc *d);
// Ragged causal GQA attention over the pages, head_dim 128, BF16.
int32_t launch_paged_attention(turbine_ctx *ctx,
                               const turbine_attention_paged_desc *d);
// Largest num_q_heads / num_kv_heads the Turbine paged kernel handles.
constexpr int32_t kPagedMaxGroup = 64;

// MoE kernels (moe.hip); descriptors validated by moe.cpp.
// Softmax, top-k and the grouping of rows by expert.
int32_t launch_moe_route(turbine_ctx *ctx, const turbine_moe_route_desc *d);
// Marks every row (token * top_k + slot) as not local (-1) in pos.
int32_t launch_moe_clear_positions(turbine_ctx *ctx, int32_t *pos,
                                   int64_t rows);
// For i in [0, count): row r = sorted_rows[i]; xs[i] = x[r / top_k];
// pos[r] = i.
int32_t launch_moe_gather(turbine_ctx *ctx, const void *x,
                          const int32_t *sorted_rows, int64_t count,
                          int32_t hidden, int32_t top_k, void *xs,
                          int32_t *pos);
// For every token t, over its local rows in ascending position (= ascending
// expert): out[t] = round(out[t] + round(down[pos] * round(w[row]))).
int32_t launch_moe_scatter(turbine_ctx *ctx, const void *down,
                           const int32_t *pos, const float *topk_weights,
                           int32_t num_tokens, int32_t hidden, int32_t top_k,
                           void *out);
// Largest num_experts / top_k the Turbine MoE kernels handle.
constexpr int32_t kMoeMaxExperts = 256;
constexpr int32_t kMoeMaxTopK = 32;
// Most routed rows (num_tokens * top_k) moe_experts runs through the small-m
// kernels (moe_small_m.hip), which read the group sizes on the device; above it
// the per-expert hipBLASLt path needs host_expert_offsets.
constexpr int64_t kMoeSmallMaxRows = 512;
// The small-m path over d's rows: fills pos ([num_tokens * top_k]), act
// ([num_tokens * top_k, inter] BF16) and down ([num_tokens * top_k, hidden]
// BF16), rows indexed by their position in sorted_rows; the caller then
// scatters down into out (launch_moe_scatter). Operands 16-byte aligned,
// hidden and inter multiples of 8.
int32_t launch_moe_small_m(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                           int32_t *pos, void *act, void *down);

// Asks hipBLASLt for a grouped BF16 GEMM solution on ctx's device (moe.cpp);
// false when there is none (ROCm 7.14.1 on gfx1201) or the query fails.
bool probe_grouped_gemm(turbine_ctx *ctx);

} // namespace turbine_hip
