// Internal definitions shared by the libturbine_hip.so translation units.
//
// Ownership: a turbine_ctx owns its compute stream, its hipBLASLt handle, the
// GEMM workspace and the attention seqstart scratch; turbine_ctx_destroy
// releases all of them after draining the stream. Device pointers passed in
// descriptors belong to the caller and are never retained beyond the call.
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

} // namespace turbine_hip
