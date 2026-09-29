// Quantized GEMM (ABI v2.9 turbine_qgemm) on ROCm: `hipblaslt_fp8`, FP8 W8A8
// through hipBLASLt (decision "P6: FP8 GEMM — provider evaluation (kernel
// reuse rule)"): e4m3 weights (FP8_TENSOR / FP8_CHANNEL) times e4m3
// activations (FP8_TENSOR static / FP8_TOKEN dynamic, quantized by
// turbine_quantize_act), F32 accumulation, BF16 out; the scales ride on
// hipBLASLt's scale pointers (qgemm_problem.hpp). Other schemes (block-scaled
// FP8, INT4, MXFP4) have no implementation here yet and are refused as
// unsupported.
//
// The algorithm per shape: in steps that prefill prompt tokens
// (turbine_qgemm_desc::prefill), one solution per shape for every m -- the
// first heuristic answer for m = 2048 -- run with split-K off, so rows do not
// depend on the batch (bit-exact prefix reuse); in decode steps, the pinned
// solution of the tuned FP8 table (qgemm_tuned.hpp) when the context has the
// table on (TURBINE_OPTION_GEMM_AUTOTUNE, as for BF16) and it has a row for the
// call, otherwise hipBLASLt's first heuristic answer for that m. Cached per
// shape, scale layout and step kind for the context's lifetime.
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "turbine_hip.hpp"

#include "qgemm_epilogue.hpp"
#include "qgemm_impls.hpp"
#include "qgemm_problem.hpp"
#include "qgemm_tuned.hpp"

using turbine_hip::check_blaslt;
using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

// (m, n, k, lda, ldc, c_dtype, weight scale layout, activation scale layout,
// prefill step)
using QGemmKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int32_t, int32_t, int32_t, bool>;

// The per-context state of turbine_qgemm (turbine_ctx::qgemm, created at the
// context's first FP8 GEMM).
struct QGemmCache {
  std::map<QGemmKey, hipblasLtMatmulAlgo_t> algos;
  // Tuned rows whose fallback was logged (once per row and context).
  std::set<const QGemmTuned *> logged;
  // Device F32 vector a scalar scale is broadcast into (see qgemm_fp8_run),
  // bcast_cap elements; grown outside graph capture only, freed with the
  // context.
  float *bcast = nullptr;
  int64_t bcast_cap = 0;
  // Prefill with vector scales (prefill_vector_scales): a device 1.0f for the
  // unit scales of the scalar-scale GEMM, and its F32 sums (kPrefillScratch
  // bytes), allocated at the first such call outside graph capture.
  float *unit = nullptr;
  float *sums = nullptr;

  QGemmCache() = default;
  QGemmCache(const QGemmCache &) = delete;
  QGemmCache &operator=(const QGemmCache &) = delete;
  ~QGemmCache() {
    if (bcast != nullptr)
      (void)hipFree(bcast);
    if (unit != nullptr)
      (void)hipFree(unit);
    if (sums != nullptr)
      (void)hipFree(sums);
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

// The m whose first heuristic answer every prefill call of a shape runs.
constexpr int64_t kPrefillCanonicalM = 2048;

// F32 sums of one chunk of rows of a vector-scale prefill.
constexpr size_t kPrefillScratch = 64u << 20;

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

namespace {

// The pinned algorithm of the tuned row for the decode call d, when there is a
// row and this hipBLASLt has its solution and takes the problem with it.
bool pinned_qgemm(turbine_ctx *ctx, QGemmCache &cache, QGemmProblem &problem,
                  const turbine_qgemm_desc *d, const QGemmShape &shape,
                  hipblasLtMatmulAlgo_t *algo) {
  const std::string &arch =
      ctx->profile.arch.empty() ? ctx->arch : ctx->profile.arch;
  const bool vector = shape.weight_scale == QScale::Vector;
  const bool prefill = d->prefill != 0;
  const QGemmTuned *row =
      prefill ? qgemm_tuned_prefill(arch.c_str(), d->n, d->k, vector)
              : qgemm_tuned(arch.c_str(), d->n, d->k, vector, d->m);
  if (row == nullptr || row->solution_index < 0)
    return false;
  const char *miss = nullptr;
  try {
    std::vector<int> index{row->solution_index};
    std::vector<hipblasLtMatmulHeuristicResult_t> found;
    if (hipblaslt_ext::getAlgosFromIndex(ctx->blaslt, index, found) !=
            HIPBLAS_STATUS_SUCCESS ||
        found.empty() ||
        hipblaslt_ext::getSolutionNameFromAlgo(ctx->blaslt, found[0].algo) !=
            row->solution_name) {
      miss = "qgemm_table_unavailable";
    } else {
      const float alpha = 1.0f, beta = 0.0f;
      size_t workspace = 0;
      hipblasStatus_t st;
      if (prefill) {
        hipblaslt_ext::Gemm gemm(ctx->blaslt, problem.desc.handle, &alpha, d->b,
                                 problem.weight.handle, d->a,
                                 problem.act.handle, &beta, d->c,
                                 problem.out.handle, d->c, problem.out.handle);
        hipblaslt_ext::GemmTuning tuning;
        tuning.setSplitK(1);
        st = gemm.isAlgoSupported(found[0].algo, tuning, workspace);
      } else {
        st = hipblaslt_ext::matmulIsAlgoSupported(
            ctx->blaslt, problem.desc.handle, &alpha, problem.weight.handle,
            problem.act.handle, &beta, problem.out.handle, problem.out.handle,
            found[0].algo, workspace);
      }
      if (st != HIPBLAS_STATUS_SUCCESS || workspace > kGemmWorkspaceBytes) {
        miss = "qgemm_table_unsupported";
      } else {
        *algo = found[0].algo;
        return true;
      }
    }
  } catch (const std::exception &) {
    miss = "qgemm_table_unavailable";
  }
  if (cache.logged.insert(row).second) {
    std::fprintf(stderr,
                 "turbine_hip: event=qgemm_table_fallback reason=%s arch=%s "
                 "m=%lld n=%lld k=%lld solution=%s: running hipBLASLt's first "
                 "heuristic answer\n",
                 miss, arch.c_str(), static_cast<long long>(d->m),
                 static_cast<long long>(d->n), static_cast<long long>(d->k),
                 row->solution_name);
  }
  return false;
}

// hipBLASLt's first heuristic answer for problem p (the call d).
int32_t first_heuristic(turbine_ctx *ctx, QGemmProblem &p,
                        const turbine_qgemm_desc *d,
                        hipblasLtMatmulAlgo_t *algo) {
  Preference pref;
  int32_t rc = check_blaslt(ctx, hipblasLtMatmulPreferenceCreate(&pref.handle),
                            "hipblasLtMatmulPreferenceCreate");
  if (rc != TURBINE_OK)
    return rc;
  const uint64_t max_ws = kGemmWorkspaceBytes;
  rc = check_blaslt(ctx,
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
                        ctx->blaslt, p.desc.handle, p.weight.handle,
                        p.act.handle, p.out.handle, p.out.handle, pref.handle,
                        1, &result, &returned),
                    "hipblasLtMatmulAlgoGetHeuristic (FP8)");
  if (rc != TURBINE_OK)
    return rc;
  if (returned < 1 || result.state != HIPBLAS_STATUS_SUCCESS) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "hipblasLtMatmulAlgoGetHeuristic: no FP8 algorithm for " +
                    describe(d));
  }
  *algo = result.algo;
  return TURBINE_OK;
}

// A prefill step's call with a vector scale (FP8_CHANNEL weights or FP8_TOKEN
// activations): gfx1201's hipBLASLt has no OUTER_VEC solution whose rows are
// independent of the call, so the e4m3 sums come from a scalar-scale solution
// with unit scales and F32 out (the tuned table's prefill row, chosen
// row-invariant; else the first heuristic answer for m = 2048), split-K off,
// in chunks of rows that fit kPrefillScratch, and qgemm_epilogue.hpp scales
// them per row and column into BF16. Rows never depend on the call's size or
// on their position in it (bit-exact prefix reuse).
int32_t prefill_vector_scales(turbine_ctx *ctx, QGemmCache &cache,
                              const turbine_qgemm_desc *d) {
  if (cache.sums == nullptr) {
    if (ctx->capturing) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_qgemm: the first vector-scale prefill call cannot "
                  "allocate its buffers while a graph is captured");
    }
    int32_t rc = check_hip(ctx, hipMalloc(&cache.unit, sizeof(float)),
                           "hipMalloc (qgemm unit scale)");
    if (rc != TURBINE_OK)
      return rc;
    rc = check_hip(ctx,
                   hipMemsetD32Async(cache.unit, 0x3f800000u, 1, ctx->stream),
                   "hipMemsetD32Async (qgemm unit scale)");
    if (rc != TURBINE_OK)
      return rc;
    rc = check_hip(ctx, hipMalloc(&cache.sums, kPrefillScratch),
                   "hipMalloc (qgemm prefill sums)");
    if (rc != TURBINE_OK)
      return rc;
  }
  const int64_t chunk = std::max<int64_t>(
      1, static_cast<int64_t>(kPrefillScratch / sizeof(float)) / d->n);
  const bool w_vector = weight_qscale(d->scheme) == QScale::Vector;
  const bool a_vector = act_qscale(d->act_quant) == QScale::Vector;
  const QGemmKey key{0, d->n, d->k, d->lda, d->n, TURBINE_DTYPE_F32,
                     0, 0,    true};
  auto &algos = cache.algos;
  auto found = algos.find(key);
  for (int64_t r0 = 0; r0 < d->m; r0 += chunk) {
    const int64_t rows = std::min(chunk, d->m - r0);
    const QGemmShape shape{rows,           d->n,          d->k,
                           d->lda,         d->n,          TURBINE_DTYPE_F32,
                           QScale::Scalar, QScale::Scalar};
    QGemmProblem problem;
    hipblasStatus_t st = problem.make(shape);
    if (st == HIPBLAS_STATUS_SUCCESS)
      st = problem.set_scales(cache.unit, cache.unit);
    int32_t rc = check_blaslt(
        ctx, st, problem.failed != nullptr ? problem.failed : "turbine_qgemm");
    if (rc != TURBINE_OK)
      return rc;
    const auto *a = static_cast<const uint8_t *>(d->a) + r0 * d->lda;
    if (found == algos.end()) {
      hipblasLtMatmulAlgo_t algo{};
      // The table's prefill row of a vector-scale shape is this path's
      // scalar-scale F32 solution.
      QGemmShape row_shape = shape;
      row_shape.weight_scale = QScale::Vector;
      turbine_qgemm_desc probe = *d;
      probe.m = rows;
      probe.a = a;
      probe.c = cache.sums;
      if (!ctx->gemm_table ||
          !pinned_qgemm(ctx, cache, problem, &probe, row_shape, &algo)) {
        QGemmShape canonical = shape;
        canonical.m = kPrefillCanonicalM;
        QGemmProblem cp;
        st = cp.make(canonical);
        if (st == HIPBLAS_STATUS_SUCCESS)
          st = cp.set_scales(cache.unit, cache.unit);
        rc = check_blaslt(ctx, st,
                          cp.failed != nullptr ? cp.failed : "turbine_qgemm");
        if (rc != TURBINE_OK)
          return rc;
        rc = first_heuristic(ctx, cp, d, &algo);
        if (rc != TURBINE_OK)
          return rc;
        std::fprintf(stderr,
                     "turbine_hip: event=qgemm_prefill_unpinned n=%lld "
                     "k=%lld: no prefill row in the tuned FP8 table; running "
                     "hipBLASLt's scalar-scale F32 answer for m=%lld with "
                     "split-K off, whose rows may depend on their position "
                     "in the call\n",
                     static_cast<long long>(d->n), static_cast<long long>(d->k),
                     static_cast<long long>(kPrefillCanonicalM));
      }
      if (algos.size() >= kGemmAlgoCacheEntries)
        algos.clear();
      found = algos.emplace(key, algo).first;
    }
    try {
      const float alpha = 1.0f, beta = 0.0f;
      hipblaslt_ext::Gemm gemm(ctx->blaslt, problem.desc.handle, &alpha, d->b,
                               problem.weight.handle, a, problem.act.handle,
                               &beta, cache.sums, problem.out.handle,
                               cache.sums, problem.out.handle);
      hipblaslt_ext::GemmTuning tuning;
      tuning.setSplitK(1);
      gemm.setMaxWorkspaceBytes(kGemmWorkspaceBytes);
      st = gemm.initialize(found->second, tuning, ctx->workspace, false,
                           ctx->stream);
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = gemm.run(ctx->stream);
      if (st != HIPBLAS_STATUS_SUCCESS) {
        return fail(ctx, TURBINE_E_LIBRARY,
                    std::string("hipblaslt_ext::Gemm (FP8 prefill sums, "
                                "split-K off): ") +
                        blaslt_status_name(st) + " for " + describe(d));
      }
    } catch (const std::exception &e) {
      return fail(ctx, TURBINE_E_LIBRARY,
                  std::string("hipblaslt_ext::Gemm (FP8 prefill sums): ") +
                      e.what());
    }
    auto *c = static_cast<uint16_t *>(d->c) + r0 * d->ldc;
    rc = check_hip(ctx,
                   launch_qgemm_scale_epilogue(
                       cache.sums, d->n, c, d->ldc,
                       d->a_scales + (a_vector ? r0 : 0), a_vector,
                       static_cast<const float *>(d->b_scales), w_vector,
                       d->alpha, rows, d->n, ctx->stream),
                   "turbine_qgemm scale epilogue");
    if (rc != TURBINE_OK)
      return rc;
  }
  return TURBINE_OK;
}

} // namespace

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

  if (d->prefill != 0 && (weight_qscale(d->scheme) == QScale::Vector ||
                          act_qscale(d->act_quant) == QScale::Vector)) {
    return prefill_vector_scales(ctx, cache, d);
  }

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
  if (d->prefill != 0) {
    // Steps that prefill prompt tokens: one solution per shape whatever m is
    // (the tuned table's prefill row, chosen row-invariant; else hipBLASLt's
    // first answer for a canonical m), run with split-K off, so a row's result
    // does not depend on the call and a prefix-reused prefill of a prompt's
    // suffix computes each row exactly as the whole-prompt prefill does (Phase
    // 4 bit-exact prefix reuse; the BF16 GEMM's invariant rows do the same).
    const QGemmKey key{0,
                       d->n,
                       d->k,
                       d->lda,
                       d->ldc,
                       d->c_dtype,
                       static_cast<int32_t>(shape.weight_scale),
                       static_cast<int32_t>(shape.act_scale),
                       true};
    auto found = algos.find(key);
    if (found == algos.end()) {
      hipblasLtMatmulAlgo_t algo{};
      if (!ctx->gemm_table ||
          !pinned_qgemm(ctx, cache, problem, d, shape, &algo)) {
        QGemmShape canonical = shape;
        canonical.m = kPrefillCanonicalM;
        QGemmProblem cp;
        hipblasStatus_t cst = cp.make(canonical);
        if (cst == HIPBLAS_STATUS_SUCCESS)
          cst = cp.set_scales(w_scales, a_scales);
        rc = check_blaslt(ctx, cst,
                          cp.failed != nullptr ? cp.failed : "turbine_qgemm");
        if (rc != TURBINE_OK)
          return rc;
        rc = first_heuristic(ctx, cp, d, &algo);
        if (rc != TURBINE_OK)
          return rc;
        std::fprintf(stderr,
                     "turbine_hip: event=qgemm_prefill_unpinned n=%lld "
                     "k=%lld: no prefill row in the tuned FP8 table; running "
                     "hipBLASLt's answer for m=%lld with split-K off, whose "
                     "rows may depend on their position in the call\n",
                     static_cast<long long>(d->n), static_cast<long long>(d->k),
                     static_cast<long long>(kPrefillCanonicalM));
      }
      if (algos.size() >= kGemmAlgoCacheEntries)
        algos.clear();
      found = algos.emplace(key, algo).first;
    }
    try {
      const float beta = 0.0f;
      hipblaslt_ext::Gemm gemm(ctx->blaslt, problem.desc.handle, &d->alpha,
                               d->b, problem.weight.handle, d->a,
                               problem.act.handle, &beta, d->c,
                               problem.out.handle, d->c, problem.out.handle);
      hipblaslt_ext::GemmTuning tuning;
      tuning.setSplitK(1);
      gemm.setMaxWorkspaceBytes(kGemmWorkspaceBytes);
      hipblasStatus_t pst = gemm.initialize(found->second, tuning,
                                            ctx->workspace, false, ctx->stream);
      if (pst == HIPBLAS_STATUS_SUCCESS)
        pst = gemm.run(ctx->stream);
      if (pst != HIPBLAS_STATUS_SUCCESS) {
        return fail(ctx, TURBINE_E_LIBRARY,
                    std::string("hipblaslt_ext::Gemm (FP8 prefill, split-K "
                                "off): ") +
                        blaslt_status_name(pst) + " for " + describe(d));
      }
      return TURBINE_OK;
    } catch (const std::exception &e) {
      return fail(ctx, TURBINE_E_LIBRARY,
                  std::string("hipblaslt_ext::Gemm (FP8 prefill): ") +
                      e.what());
    }
  }

  // Decode steps: the tuned table's pinned solution, else the first
  // heuristic answer for this m.
  const QGemmKey key{d->m,
                     d->n,
                     d->k,
                     d->lda,
                     d->ldc,
                     d->c_dtype,
                     static_cast<int32_t>(shape.weight_scale),
                     static_cast<int32_t>(shape.act_scale),
                     false};
  auto found = algos.find(key);
  if (found == algos.end()) {
    hipblasLtMatmulAlgo_t algo{};
    if (!ctx->gemm_table ||
        !pinned_qgemm(ctx, cache, problem, d, shape, &algo)) {
      rc = first_heuristic(ctx, problem, d, &algo);
      if (rc != TURBINE_OK)
        return rc;
    }
    if (algos.size() >= kGemmAlgoCacheEntries)
      algos.clear();
    found = algos.emplace(key, algo).first;
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
