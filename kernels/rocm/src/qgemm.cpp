// Quantized GEMM (ABI v2.9 turbine_qgemm) on ROCm: `hipblaslt_fp8`, FP8 W8A8
// through hipBLASLt (decision "P6: FP8 GEMM — provider evaluation (kernel
// reuse rule)"): e4m3 weights (FP8_TENSOR / FP8_CHANNEL) times e4m3
// activations (FP8_TENSOR static / FP8_TOKEN dynamic, quantized by
// turbine_quantize_act), F32 accumulation, BF16 out; the scales ride on
// hipBLASLt's scale pointers (qgemm_problem.hpp). Other schemes (block-scaled
// FP8, INT4, MXFP4) have no implementation here yet and are refused as
// unsupported.
//
// The algorithm per shape is hipBLASLt's first heuristic answer, cached per
// exact shape and scale layout for the context's lifetime (as gemm.cpp does
// without a tuned table row).
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "turbine_hip.hpp"

#include "qgemm_impls.hpp"
#include "qgemm_problem.hpp"

using turbine_hip::check_blaslt;
using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

// (m, n, k, lda, ldc, c_dtype, weight scale layout, activation scale layout)
using QGemmKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int32_t, int32_t, int32_t>;

// The per-context state of turbine_qgemm (turbine_ctx::qgemm, created at the
// context's first FP8 GEMM).
struct QGemmCache {
  std::map<QGemmKey, hipblasLtMatmulAlgo_t> algos;
  // Device F32 vector a scalar scale is broadcast into (see qgemm_fp8_run),
  // bcast_cap elements; grown outside graph capture only, freed with the
  // context.
  float *bcast = nullptr;
  int64_t bcast_cap = 0;

  QGemmCache() = default;
  QGemmCache(const QGemmCache &) = delete;
  QGemmCache &operator=(const QGemmCache &) = delete;
  ~QGemmCache() {
    if (bcast != nullptr)
      (void)hipFree(bcast);
  }

  // Enqueues bcast[0..count) = *scalar on ctx's stream.
  int32_t broadcast(turbine_ctx *ctx, const float *scalar, int64_t count);
};

namespace {

__global__ void broadcast_scale(const float *scalar, float *out,
                                int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = *scalar;
}

// Smallest broadcast buffer, so that decode-graph capture (which cannot
// allocate) finds one large enough for every Llama-class width.
constexpr int64_t kMinBroadcast = 65536;

} // namespace

int32_t QGemmCache::broadcast(turbine_ctx *ctx, const float *scalar,
                              int64_t count) {
  if (count > bcast_cap) {
    if (ctx->capturing) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_qgemm: scale broadcast buffer of " +
                      std::to_string(bcast_cap) + " elements cannot grow to " +
                      std::to_string(count) + " while a graph is captured");
    }
    const int64_t cap = std::max(count, kMinBroadcast);
    float *grown = nullptr;
    const int32_t rc = check_hip(
        ctx, hipMalloc(&grown, static_cast<size_t>(cap) * sizeof(float)),
        "hipMalloc (qgemm scale broadcast)");
    if (rc != TURBINE_OK)
      return rc;
    if (bcast != nullptr) {
      // The old buffer may still be read by queued work.
      (void)hipStreamSynchronize(ctx->stream);
      (void)hipFree(bcast);
    }
    bcast = grown;
    bcast_cap = cap;
  }
  constexpr int kThreads = 256;
  const dim3 grid(static_cast<uint32_t>((count + kThreads - 1) / kThreads));
  hipLaunchKernelGGL(broadcast_scale, grid, dim3(kThreads), 0, ctx->stream,
                     scalar, bcast, count);
  return check_hip(ctx, hipGetLastError(), "turbine_qgemm scale broadcast");
}

} // namespace turbine_hip

namespace {

bool fp8_scheme(int32_t scheme) {
  return scheme == TURBINE_QSCHEME_FP8_TENSOR ||
         scheme == TURBINE_QSCHEME_FP8_CHANNEL;
}

bool fp8_act(int32_t act_quant) {
  return act_quant == TURBINE_ACTQ_FP8_TENSOR ||
         act_quant == TURBINE_ACTQ_FP8_TOKEN;
}

bool supported(const turbine_qgemm_desc *d) {
  if (d == nullptr)
    return false;
  if (!fp8_scheme(d->scheme) || !fp8_act(d->act_quant))
    return false;
  if (d->a_dtype != TURBINE_DTYPE_F8E4M3)
    return false;
  // BF16 out only: gfx1201's hipBLASLt has no FP8 kernel with vector scales
  // and an F32 D.
  if (d->c_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->m <= 0 || d->n <= 0 || d->k <= 0)
    return false;
  // hipBLASLt's FP8 kernels read 16-byte vectors along k.
  if (d->k % 16 != 0 || d->lda % 16 != 0)
    return false;
  return d->lda >= d->k && d->ldc >= d->n;
}

std::string describe(const turbine_qgemm_desc *d) {
  return "m=" + std::to_string(d->m) + " n=" + std::to_string(d->n) +
         " k=" + std::to_string(d->k) + " lda=" + std::to_string(d->lda) +
         " scheme=" + std::to_string(d->scheme) +
         " act_quant=" + std::to_string(d->act_quant) +
         " dtypes=" + std::to_string(d->a_dtype) + "," +
         std::to_string(d->c_dtype);
}

} // namespace

extern "C" {

int32_t turbine_qgemm_supported(const turbine_qgemm_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_QGEMM, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_qgemm_impl(const turbine_qgemm_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_QGEMM, d);
}

int32_t turbine_qgemm(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_QGEMM, d);
}

} // extern "C"

namespace turbine_hip {

bool qgemm_fp8_supports(const turbine_qgemm_desc *d) { return supported(d); }

int32_t qgemm_fp8_run(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: descriptor is NULL");
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_qgemm (hipblaslt_fp8): unsupported configuration " +
                    describe(d));
  }
  if (d->a == nullptr || d->b == nullptr || d->c == nullptr ||
      d->a_scales == nullptr || d->b_scales == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_qgemm: NULL operand or scale pointer");
  }
  int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;

  if (!ctx->qgemm)
    ctx->qgemm = std::make_shared<QGemmCache>();
  QGemmCache &cache = *ctx->qgemm;

  // gfx1201's hipBLASLt has FP8 kernels for scalar x scalar and vector x
  // vector scales only: a scalar scale paired with a vector one is broadcast
  // to a vector (n weight scales or m activation scales) first.
  QScale w_scale = weight_qscale(d->scheme);
  QScale a_scale = act_qscale(d->act_quant);
  const float *w_scales = static_cast<const float *>(d->b_scales);
  const float *a_scales = d->a_scales;
  if (w_scale != a_scale) {
    const bool weight = w_scale == QScale::Scalar;
    const int64_t count = weight ? d->n : d->m;
    rc = cache.broadcast(ctx, weight ? w_scales : a_scales, count);
    if (rc != TURBINE_OK)
      return rc;
    (weight ? w_scales : a_scales) = cache.bcast;
    w_scale = a_scale = QScale::Vector;
  }

  const QGemmShape shape{d->m,   d->n,       d->k,    d->lda,
                         d->ldc, d->c_dtype, w_scale, a_scale};
  QGemmProblem problem;
  hipblasStatus_t st = problem.make(shape);
  if (st == HIPBLAS_STATUS_SUCCESS)
    st = problem.set_scales(w_scales, a_scales);
  rc = check_blaslt(
      ctx, st, problem.failed != nullptr ? problem.failed : "turbine_qgemm");
  if (rc != TURBINE_OK)
    return rc;

  auto &algos = cache.algos;
  const QGemmKey key{d->m,
                     d->n,
                     d->k,
                     d->lda,
                     d->ldc,
                     d->c_dtype,
                     static_cast<int32_t>(shape.weight_scale),
                     static_cast<int32_t>(shape.act_scale)};
  auto found = algos.find(key);
  if (found == algos.end()) {
    Preference pref;
    rc = check_blaslt(ctx, hipblasLtMatmulPreferenceCreate(&pref.handle),
                      "hipblasLtMatmulPreferenceCreate");
    if (rc != TURBINE_OK)
      return rc;
    const uint64_t max_ws = kGemmWorkspaceBytes;
    rc =
        check_blaslt(ctx,
                     hipblasLtMatmulPreferenceSetAttribute(
                         pref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &max_ws, sizeof(max_ws)),
                     "hipblasLtMatmulPreferenceSetAttribute");
    if (rc != TURBINE_OK)
      return rc;
    hipblasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    rc = check_blaslt(ctx,
                      hipblasLtMatmulAlgoGetHeuristic(
                          ctx->blaslt, problem.desc.handle,
                          problem.weight.handle, problem.act.handle,
                          problem.out.handle, problem.out.handle, pref.handle,
                          1, &result, &returned),
                      "hipblasLtMatmulAlgoGetHeuristic (FP8)");
    if (rc != TURBINE_OK)
      return rc;
    if (returned < 1 || result.state != HIPBLAS_STATUS_SUCCESS) {
      return fail(ctx, TURBINE_E_UNSUPPORTED,
                  "hipblasLtMatmulAlgoGetHeuristic: no FP8 algorithm for " +
                      describe(d));
    }
    if (algos.size() >= kGemmAlgoCacheEntries)
      algos.clear();
    found = algos.emplace(key, result.algo).first;
  }
  const float beta = 0.0f;
  return check_blaslt(
      ctx,
      hipblasLtMatmul(ctx->blaslt, problem.desc.handle, &d->alpha, d->b,
                      problem.weight.handle, d->a, problem.act.handle, &beta,
                      d->c, problem.out.handle, d->c, problem.out.handle,
                      &found->second, ctx->workspace, kGemmWorkspaceBytes,
                      ctx->stream),
      "hipblasLtMatmul (FP8)");
}

} // namespace turbine_hip
