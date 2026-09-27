// Identity functions, context lifecycle, context info and error reporting.
#include <hipblaslt/hipblaslt-version.h>
#include <rocm-core/rocm_version.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "turbine_hip.hpp"

#ifndef TURBINE_CK_COMMIT
#error "TURBINE_CK_COMMIT must be defined by the build"
#endif
#ifndef TURBINE_BUILD_ARCHS
#error "TURBINE_BUILD_ARCHS must be defined by the build"
#endif

namespace {

// Message of the most recent failed turbine_ctx_create on this thread.
thread_local std::string g_create_error;

std::once_flag g_banner_once;

int32_t create_fail(int32_t code, const std::string &msg) {
  g_create_error = msg;
  return code;
}

std::string hip_message(hipError_t err, const char *what) {
  std::string msg = hipGetErrorName(err);
  msg += ": ";
  msg += hipGetErrorString(err);
  msg += " (";
  msg += what;
  msg += ")";
  return msg;
}

int32_t hip_code(hipError_t err) {
  return err == hipErrorOutOfMemory ? TURBINE_E_OUT_OF_MEMORY
                                    : TURBINE_E_DEVICE;
}

void log_banner() {
  std::call_once(g_banner_once, [] {
    std::fprintf(stderr,
                 "turbine_hip: libturbine_hip.so built for %s, ROCm %d.%d.%d, "
                 "hipBLASLt %d.%d.%d, CK %s\n",
                 TURBINE_BUILD_ARCHS, ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR,
                 ROCM_VERSION_PATCH, HIPBLASLT_VERSION_MAJOR,
                 HIPBLASLT_VERSION_MINOR, HIPBLASLT_VERSION_PATCH,
                 TURBINE_CK_COMMIT);
  });
}

// Releases whatever a (possibly partially built) context holds.
void release(turbine_ctx *ctx) {
  turbine_hip::abandon_capture(ctx);
  if (ctx->stream != nullptr)
    (void)hipStreamSynchronize(ctx->stream);
  if (ctx->blaslt != nullptr)
    (void)hipblasLtDestroy(ctx->blaslt);
  if (ctx->workspace != nullptr)
    (void)hipFree(ctx->workspace);
  if (ctx->seqstart != nullptr)
    (void)hipFree(ctx->seqstart);
  if (ctx->moe_scratch != nullptr)
    (void)hipFree(ctx->moe_scratch);
  if (ctx->attn_split_scratch != nullptr)
    (void)hipFree(ctx->attn_split_scratch);
  for (void *retired : ctx->attn_split_retired)
    (void)hipFree(retired);
  if (ctx->stream != nullptr)
    (void)hipStreamDestroy(ctx->stream);
  delete ctx;
}

} // namespace

namespace turbine_hip {

int32_t fail(turbine_ctx *ctx, int32_t code, const std::string &msg) {
  if (ctx != nullptr) {
    std::lock_guard<std::mutex> lock(ctx->error_mutex);
    ctx->last_error = msg;
  }
  return code;
}

int32_t check_hip(turbine_ctx *ctx, hipError_t err, const char *what) {
  if (err == hipSuccess)
    return TURBINE_OK;
  // Reset the thread's last-error slot so a later launch check does not
  // report this (non-sticky) failure again.
  (void)hipGetLastError();
  return fail(ctx, hip_code(err), hip_message(err, what));
}

const char *blaslt_status_name(hipblasStatus_t status) {
  switch (status) {
  case HIPBLAS_STATUS_SUCCESS:
    return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:
    return "HIPBLAS_STATUS_INVALID_VALUE";
  case HIPBLAS_STATUS_MAPPING_ERROR:
    return "HIPBLAS_STATUS_MAPPING_ERROR";
  case HIPBLAS_STATUS_EXECUTION_FAILED:
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
  case HIPBLAS_STATUS_NOT_SUPPORTED:
    return "HIPBLAS_STATUS_NOT_SUPPORTED";
  case HIPBLAS_STATUS_ARCH_MISMATCH:
    return "HIPBLAS_STATUS_ARCH_MISMATCH";
  case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
    return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
  case HIPBLAS_STATUS_INVALID_ENUM:
    return "HIPBLAS_STATUS_INVALID_ENUM";
  default:
    return "HIPBLAS_STATUS_UNKNOWN";
  }
}

int32_t check_blaslt(turbine_ctx *ctx, hipblasStatus_t status,
                     const char *what) {
  if (status == HIPBLAS_STATUS_SUCCESS)
    return TURBINE_OK;
  std::string msg = blaslt_status_name(status);
  msg += ": ";
  msg += what;
  return fail(ctx,
              status == HIPBLAS_STATUS_ALLOC_FAILED ? TURBINE_E_OUT_OF_MEMORY
                                                    : TURBINE_E_LIBRARY,
              msg);
}

bool arch_is_built(const char *arch) {
  const std::string full(arch);
  const std::string base = full.substr(0, full.find(':'));
  const std::string archs = TURBINE_BUILD_ARCHS;
  size_t start = 0;
  while (start <= archs.size()) {
    const size_t end = archs.find(',', start);
    const std::string item = archs.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    if (item == base)
      return true;
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return false;
}

int32_t enter(turbine_ctx *ctx) {
  // Clear a stale last error left by a runtime call outside this library, so
  // the launch checks that follow only see errors of this call.
  (void)hipGetLastError();
  return check_hip(ctx, hipSetDevice(ctx->device), "hipSetDevice");
}

} // namespace turbine_hip

extern "C" {

uint32_t turbine_abi_version(void) { return TURBINE_ABI_VERSION; }

const char *turbine_backend_name(void) { return "hip"; }

const char *turbine_build_archs(void) { return TURBINE_BUILD_ARCHS; }

int32_t turbine_ctx_create(int32_t device_ordinal, turbine_ctx **out) {
  if (out == nullptr) {
    return create_fail(TURBINE_E_ARGUMENT, "turbine_ctx_create: out is NULL");
  }
  *out = nullptr;
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) {
    return create_fail(hip_code(err), hip_message(err, "hipGetDeviceCount"));
  }
  if (device_ordinal < 0 || device_ordinal >= count) {
    return create_fail(TURBINE_E_ARGUMENT,
                       "turbine_ctx_create: device ordinal " +
                           std::to_string(device_ordinal) + " out of range (" +
                           std::to_string(count) + " HIP devices visible)");
  }
  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, device_ordinal);
  if (err != hipSuccess) {
    return create_fail(hip_code(err),
                       hip_message(err, "hipGetDeviceProperties"));
  }
  if (!turbine_hip::arch_is_built(props.gcnArchName)) {
    return create_fail(
        TURBINE_E_UNSUPPORTED,
        std::string("turbine_ctx_create: device ") +
            std::to_string(device_ordinal) + " arch " + props.gcnArchName +
            " is not in library build archs " + TURBINE_BUILD_ARCHS);
  }
  err = hipSetDevice(device_ordinal);
  if (err != hipSuccess) {
    return create_fail(hip_code(err), hip_message(err, "hipSetDevice"));
  }

  auto *ctx = new turbine_ctx();
  ctx->device = device_ordinal;
  {
    const std::string full(props.gcnArchName);
    ctx->arch = full.substr(0, full.find(':'));
  }
  ctx->wave_size = props.warpSize;
  err = hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking);
  if (err != hipSuccess) {
    ctx->stream = nullptr;
    release(ctx);
    return create_fail(hip_code(err),
                       hip_message(err, "hipStreamCreateWithFlags"));
  }
  hipblasStatus_t st = hipblasLtCreate(&ctx->blaslt);
  if (st != HIPBLAS_STATUS_SUCCESS) {
    ctx->blaslt = nullptr;
    release(ctx);
    return create_fail(
        st == HIPBLAS_STATUS_ALLOC_FAILED ? TURBINE_E_OUT_OF_MEMORY
                                          : TURBINE_E_LIBRARY,
        std::string(turbine_hip::blaslt_status_name(st)) + ": hipblasLtCreate");
  }
  err = hipMalloc(&ctx->workspace, turbine_hip::kGemmWorkspaceBytes);
  if (err != hipSuccess) {
    ctx->workspace = nullptr;
    release(ctx);
    return create_fail(hip_code(err),
                       hip_message(err, "hipMalloc GEMM workspace"));
  }
  err =
      hipMalloc(reinterpret_cast<void **>(&ctx->seqstart), 4 * sizeof(int32_t));
  if (err != hipSuccess) {
    ctx->seqstart = nullptr;
    release(ctx);
    return create_fail(hip_code(err),
                       hip_message(err, "hipMalloc attention seqstart"));
  }
  log_banner();
  ctx->moe_grouped = turbine_hip::probe_grouped_gemm(ctx);
  *out = ctx;
  return TURBINE_OK;
}

int32_t turbine_ctx_get_info(turbine_ctx *ctx, turbine_ctx_info *out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (out == nullptr) {
    return turbine_hip::fail(ctx, TURBINE_E_ARGUMENT,
                             "turbine_ctx_get_info: out is NULL");
  }
  *out = turbine_ctx_info{};
  out->workspace_bytes = turbine_hip::kGemmWorkspaceBytes;
  // AMD devices have no compute capability.
  out->compute_major = -1;
  out->compute_minor = -1;
  const size_t n = ctx->arch.size() < sizeof(out->device_arch) - 1
                       ? ctx->arch.size()
                       : sizeof(out->device_arch) - 1;
  std::memcpy(out->device_arch, ctx->arch.data(), n);
  out->device_arch[n] = '\0';
  return TURBINE_OK;
}

void turbine_ctx_destroy(turbine_ctx *ctx) {
  if (ctx == nullptr)
    return;
  (void)hipSetDevice(ctx->device);
  release(ctx);
}

// Context options (ABI v2.1). TURBINE_OPTION_GEMM_AUTOTUNE switches the tuned
// GEMM table (gemm_table.hpp) on (1, the default) or off (0: hipBLASLt's first
// heuristic answer for every shape); changing it drops the cached choices.
// TURBINE_OPTION_GEMM_TUNED_SHAPES (read only) counts the cached GEMM shapes
// that run a pinned solution of the table. TURBINE_OPTION_GEMM_PREFILL marks
// the following GEMMs as a prefill step's (the table's invariant rows) or a
// decode step's (its speed rows); choices are cached per mode.
int32_t turbine_ctx_set_option(turbine_ctx *ctx, int32_t option,
                               int64_t value) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  switch (option) {
  case TURBINE_OPTION_GEMM_AUTOTUNE:
    if (value != 0 && value != 1) {
      return turbine_hip::fail(ctx, TURBINE_E_ARGUMENT,
                               "turbine_ctx_set_option: GEMM_AUTOTUNE takes 0 "
                               "or 1, not " +
                                   std::to_string(value));
    }
    if (ctx->gemm_table != (value == 1)) {
      ctx->gemm_table = value == 1;
      ctx->gemm_algos.clear();
    }
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_PREFILL:
    if (value != 0 && value != 1) {
      return turbine_hip::fail(ctx, TURBINE_E_ARGUMENT,
                               "turbine_ctx_set_option: GEMM_PREFILL takes 0 "
                               "or 1, not " +
                                   std::to_string(value));
    }
    ctx->gemm_prefill = value == 1;
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_TUNED_SHAPES:
    return turbine_hip::fail(ctx, TURBINE_E_ARGUMENT,
                             "turbine_ctx_set_option: GEMM_TUNED_SHAPES is "
                             "read only");
  default:
    return turbine_hip::fail(ctx, TURBINE_E_UNSUPPORTED,
                             "turbine_ctx_set_option: unknown option " +
                                 std::to_string(option));
  }
}

int32_t turbine_ctx_get_option(turbine_ctx *ctx, int32_t option, int64_t *out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (out == nullptr) {
    return turbine_hip::fail(ctx, TURBINE_E_ARGUMENT,
                             "turbine_ctx_get_option: out is NULL");
  }
  switch (option) {
  case TURBINE_OPTION_GEMM_AUTOTUNE:
    *out = ctx->gemm_table ? 1 : 0;
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_PREFILL:
    *out = ctx->gemm_prefill ? 1 : 0;
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_TUNED_SHAPES: {
    int64_t tuned = 0;
    for (const auto &entry : ctx->gemm_algos)
      tuned += entry.second.tuned ? 1 : 0;
    *out = tuned;
    return TURBINE_OK;
  }
  default:
    return turbine_hip::fail(ctx, TURBINE_E_UNSUPPORTED,
                             "turbine_ctx_get_option: unknown option " +
                                 std::to_string(option));
  }
}

size_t turbine_last_error(turbine_ctx *ctx, char *buf, size_t len) {
  std::string msg;
  if (ctx == nullptr) {
    msg = g_create_error;
  } else {
    std::lock_guard<std::mutex> lock(ctx->error_mutex);
    msg = ctx->last_error;
  }
  if (buf != nullptr && len > 0) {
    const size_t n = msg.size() < len - 1 ? msg.size() : len - 1;
    std::memcpy(buf, msg.data(), n);
    buf[n] = '\0';
  }
  return msg.size();
}

} // extern "C"
