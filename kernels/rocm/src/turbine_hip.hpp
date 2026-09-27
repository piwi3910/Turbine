// Internal definitions shared by the libturbine_hip.so translation units.
//
// Ownership: a turbine_ctx owns its compute stream, its hipBLASLt handle, the
// GEMM workspace, the attention seqstart scratch and the MoE scratch;
// turbine_ctx_destroy releases all of them after draining the stream. Device
// pointers passed in descriptors belong to the caller and are never retained
// beyond the call, except by a captured graph (graph.cpp), which records the
// pointers of its ops until turbine_graph_destroy.
#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
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

// The hipBLASLt algorithm a GEMM shape runs, and whether it came from the
// tuned table (gemm_table.hpp) or the first heuristic answer.
struct GemmChoice {
  hipblasLtMatmulAlgo_t algo;
  bool tuned;
};

struct TunedGemm;

// The card profile a context holds (ABI v2.4 turbine_card_profile, copied by
// turbine_ctx_set_profile). The turbine_<op> entry points read their
// thresholds from it; the library itself names no card.
struct Profile {
  // A build architecture; empty = the device's own.
  std::string arch;
  int32_t wave_size;
  int32_t lds_bytes;
  // Routed rows of the first moe_experts tier.
  int64_t moe_small_max_rows;
  int32_t paged_page_multiple;
};

// The profile of a context until turbine_ctx_set_profile (a caller of ABI
// v2.3 or earlier never sets one): the values of the first card profile of
// crates/turbine-kernels/src/cards/, so such a caller keeps today's choices.
inline const Profile kDefaultProfile{"", 32, 65536, 512, 128};

// The most LDS one workgroup of this library's kernels is built to use (the
// CK FMHA and grouped WMMA MoE tiles assume 64 KiB); a profile describing less
// is refused.
constexpr int32_t kBuiltLdsBytes = 65536;

} // namespace turbine_hip

struct turbine_ctx {
  int device = 0;
  hipStream_t stream = nullptr;
  hipblasLtHandle_t blaslt = nullptr;
  void *workspace = nullptr;
  // Device int32[4]: seqstart_q {0, q_len} then seqstart_k {0, kv_len}.
  int32_t *seqstart = nullptr;
  // Device architecture without target features, e.g. "gfx942".
  std::string arch;
  // Wavefront width of the device (hipDeviceProp_t::warpSize).
  int wave_size = 0;
  // Card profile thresholds (v2.4); kDefaultProfile until
  // turbine_ctx_set_profile.
  turbine_hip::Profile profile = turbine_hip::kDefaultProfile;
  // MoE intermediates when the caller passes no (or too small a) workspace;
  // grown on demand, never shrunk.
  void *moe_scratch = nullptr;
  size_t moe_scratch_bytes = 0;
  // hipBLASLt returned a grouped-GEMM solution for this device at creation.
  bool moe_grouped = false;
  std::map<turbine_hip::GemmKey, turbine_hip::GemmChoice> gemm_algos;
  // TURBINE_OPTION_GEMM_AUTOTUNE: shapes the tuned table (gemm_table.hpp) has
  // a row for run its pinned solution (default); false = the first heuristic
  // answer for every shape.
  bool gemm_table = true;
  // hipBLASLt solution name -> index per output dtype, read once when a table
  // row's pinned index does not carry its name (gemm_table.cpp).
  std::map<int32_t, std::map<std::string, int>> gemm_solutions;
  // Table rows whose fallback was logged (once per row and context).
  std::set<const turbine_hip::TunedGemm *> gemm_table_logged;
  // True between turbine_graph_begin and turbine_graph_end (graph.cpp): the
  // stream is being captured, so copies, syncs and allocations are refused.
  bool capturing = false;
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

// Records that `what` is not allowed while ctx->capturing and returns
// TURBINE_E_ARGUMENT (graph.cpp).
int32_t refuse_while_capturing(turbine_ctx *ctx, const char *what);

// Ends and discards a capture in progress on ctx (no-op otherwise), e.g. before
// the context is destroyed (graph.cpp).
void abandon_capture(turbine_ctx *ctx);

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
// The small-m path over d's rows: fills pos ([num_tokens * top_k]), act
// ([num_tokens * top_k, inter] BF16) and down ([num_tokens * top_k, hidden]
// BF16), rows indexed by their position in sorted_rows; the caller then
// scatters down into out (launch_moe_scatter). Operands 16-byte aligned,
// hidden and inter multiples of 8.
int32_t launch_moe_small_m(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                           int32_t *pos, void *act, void *down);
// pos[sorted_rows[i]] = i for the local positions of d, -1 for the other rows
// (moe_small_m.hip); both device-offset paths start with it.
int32_t launch_moe_positions(turbine_ctx *ctx,
                             const turbine_moe_experts_desc *d, int32_t *pos);
// The grouped WMMA path (moe_grouped.hip) for d's rows (any row count; the
// default takes it above the profile's small-row tier): the same outputs as
// launch_moe_small_m (pos, act and down indexed by position; the caller
// scatters). Needs moe_wmma_shape(d) and
// 16-byte aligned x and expert weights.
int32_t launch_moe_wmma(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                        int32_t *pos, void *act, void *down);
// hidden and inter are multiples of the grouped kernels' depth step (64).
bool moe_wmma_shape(const turbine_moe_experts_desc *d);
// The same WMMA kernels with 16-row tiles (moe_grouped.hip), bitwise equal to
// launch_moe_wmma row by row: the small-m path's kernels when
// moe_wmma_shape(d).
int32_t launch_moe_wmma_small(turbine_ctx *ctx,
                              const turbine_moe_experts_desc *d, int32_t *pos,
                              void *act, void *down);
// The decode kernels of the small-m path (moe_grouped.hip): the same WMMA chain
// as launch_moe_wmma, one wave per 16 columns of an expert, operands straight
// from global memory. Needs moe_gemv_shape(d).
int32_t launch_moe_wmma_gemv(turbine_ctx *ctx,
                             const turbine_moe_experts_desc *d, int32_t *pos,
                             void *act, void *down);
// hidden and inter are multiples of the decode kernels' load step (128).
bool moe_gemv_shape(const turbine_moe_experts_desc *d);

// Asks hipBLASLt for a grouped BF16 GEMM solution on ctx's device (moe.cpp);
// false when there is none (ROCm 7.14.1 on RDNA4) or the query fails.
bool probe_grouped_gemm(turbine_ctx *ctx);

// True when arch (e.g. "gfx942" or "gfx942:sramecc+:xnack-") names one of
// the build architectures (context.cpp).
bool arch_is_built(const char *arch);

// ---- implementations (ABI v2.4, impl_table.cpp) ----
// Each implementation of an op is a supports / run pair. supports takes the
// op's descriptor (NULL allowed, pointer fields ignored, no context); run
// validates the descriptor and its operands like the op's entry point and
// enqueues exactly that implementation, failing with TURBINE_E_UNSUPPORTED
// when it does not support the descriptor.
struct ImplEntry {
  const char *name;
  // Implementation family: "hipblaslt", "ck" or "turbine_hip".
  const char *provider;
  // TURBINE_IMPL_* flags.
  uint32_t flags;
  // When the library chooses (turbine_<op> without an index): whether the
  // context's card profile lets this implementation serve desc (its
  // thresholds: moe_experts' small-row tier, the paged page multiple);
  // nullptr = always.
  bool (*profile_allows)(const Profile &profile, const void *desc);
  bool (*supports)(const void *desc);
  int32_t (*run)(turbine_ctx *ctx, const void *desc);
};
// The implementations of op in library order; nullptr (count 0) for an
// unknown op.
const ImplEntry *impl_entries(int32_t op, int32_t *count);

// The library's own choice for a caller that names no implementation (the
// ABI v2 entry points): the first implementation of op, in library order,
// that supports desc and that profile allows; nullptr when none does.
const ImplEntry *default_entry(const Profile &profile, int32_t op,
                               const void *desc);
// Runs default_entry(ctx->profile, op, desc); when there is none the last
// implementation of op runs, which refuses desc with the op's message.
int32_t run_default(turbine_ctx *ctx, int32_t op, const void *desc);
// The name turbine_<op>_impl reports (context-free, so under
// kDefaultProfile): default_entry's, else the last implementation's.
const char *default_name(int32_t op, const void *desc);
// The TURBINE_IMPL_* flags of default_entry(kDefaultProfile, op, desc), or
// fallback when there is none.
uint32_t default_flags(int32_t op, const void *desc, uint32_t fallback);

// Per-implementation pairs of the ops with more than one implementation.
// rmsnorm.cpp: ck_tile rmsnorm2d (ck true) or the Turbine kernel.
bool rmsnorm_supports(const turbine_rmsnorm_desc *d, bool ck);
int32_t rmsnorm_run(turbine_ctx *ctx, const turbine_rmsnorm_desc *d, bool ck);
bool add_rmsnorm_supports(const turbine_add_rmsnorm_desc *d, bool ck);
int32_t add_rmsnorm_run(turbine_ctx *ctx, const turbine_add_rmsnorm_desc *d,
                        bool ck);
// paged_attention.cpp: CK fmha_fwd_pagedkv (ck true) or the Turbine kernel;
// entry names the op in error messages.
bool paged_supports(const turbine_attention_paged_desc *d, bool ck);
int32_t paged_run(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                  const char *entry, bool ck);
// moe.cpp: the four moe_experts paths.
enum class MoePath { SmallM, Wmma, Grouped, PerExpert };
bool moe_experts_supports(const turbine_moe_experts_desc *d, MoePath path);
int32_t moe_experts_run(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                        MoePath path);

} // namespace turbine_hip
