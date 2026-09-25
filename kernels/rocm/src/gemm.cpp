// GEMM through hipBLASLt.
//
// Row-major c[m,n] = alpha * a[m,k] . op(b) + beta * c is computed as the
// column-major product c^T[n,m] = op(b)^T[n,k] . a^T[k,m]: hipBLASLt's A is
// the weight (column-major k x n with ld ldb and HIPBLAS_OP_T when b is [n,k];
// n x k with HIPBLAS_OP_N when b is [k,n]), its B the activation (k x m, ld
// lda, HIPBLAS_OP_N), C and D the output (n x m, ld ldc). F32 accumulation.
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_blaslt;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "hipblaslt";

// Owns one hipBLASLt matrix layout.
struct Layout {
  hipblasLtMatrixLayout_t handle = nullptr;
  ~Layout() {
    if (handle != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(handle);
  }
};

// Owns one hipBLASLt matmul descriptor.
struct MatmulDesc {
  hipblasLtMatmulDesc_t handle = nullptr;
  ~MatmulDesc() {
    if (handle != nullptr)
      (void)hipblasLtMatmulDescDestroy(handle);
  }
};

// Owns one hipBLASLt heuristic preference.
struct Preference {
  hipblasLtMatmulPreference_t handle = nullptr;
  ~Preference() {
    if (handle != nullptr)
      (void)hipblasLtMatmulPreferenceDestroy(handle);
  }
};

bool supported(const turbine_gemm_desc *d) {
  if (d == nullptr)
    return false;
  if (d->a_dtype != TURBINE_DTYPE_BF16 || d->b_dtype != TURBINE_DTYPE_BF16) {
    return false;
  }
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32) {
    return false;
  }
  if (d->trans_b != 0 && d->trans_b != 1)
    return false;
  if (d->m <= 0 || d->n <= 0 || d->k <= 0)
    return false;
  if (d->lda < d->k || d->ldc < d->n)
    return false;
  return d->ldb >= (d->trans_b == 1 ? d->k : d->n);
}

} // namespace

extern "C" {

int32_t turbine_gemm_supported(const turbine_gemm_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_gemm_impl(const turbine_gemm_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_gemm(turbine_ctx *ctx, const turbine_gemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_gemm: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(
        ctx, TURBINE_E_UNSUPPORTED,
        "turbine_gemm: unsupported configuration m=" + std::to_string(d->m) +
            " n=" + std::to_string(d->n) + " k=" + std::to_string(d->k) +
            " trans_b=" + std::to_string(d->trans_b) +
            " dtypes=" + std::to_string(d->a_dtype) + "," +
            std::to_string(d->b_dtype) + "," + std::to_string(d->c_dtype));
  }
  if (d->a == nullptr || d->b == nullptr || d->c == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_gemm: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;

  const hipDataType c_type =
      d->c_dtype == TURBINE_DTYPE_F32 ? HIP_R_32F : HIP_R_16BF;
  const hipblasOperation_t op_weight =
      d->trans_b == 1 ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t op_act = HIPBLAS_OP_N;

  MatmulDesc desc;
  int32_t rc = check_blaslt(
      ctx,
      hipblasLtMatmulDescCreate(&desc.handle, HIPBLAS_COMPUTE_32F, HIP_R_32F),
      "hipblasLtMatmulDescCreate");
  if (rc != TURBINE_OK)
    return rc;
  rc = check_blaslt(
      ctx,
      hipblasLtMatmulDescSetAttribute(desc.handle, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &op_weight, sizeof(op_weight)),
      "hipblasLtMatmulDescSetAttribute TRANSA");
  if (rc != TURBINE_OK)
    return rc;
  rc = check_blaslt(
      ctx,
      hipblasLtMatmulDescSetAttribute(desc.handle, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &op_act, sizeof(op_act)),
      "hipblasLtMatmulDescSetAttribute TRANSB");
  if (rc != TURBINE_OK)
    return rc;

  const uint64_t m = static_cast<uint64_t>(d->m);
  const uint64_t n = static_cast<uint64_t>(d->n);
  const uint64_t k = static_cast<uint64_t>(d->k);
  Layout weight;
  Layout act;
  Layout out;
  rc = check_blaslt(ctx,
                    d->trans_b == 1
                        ? hipblasLtMatrixLayoutCreate(&weight.handle,
                                                      HIP_R_16BF, k, n, d->ldb)
                        : hipblasLtMatrixLayoutCreate(&weight.handle,
                                                      HIP_R_16BF, n, k, d->ldb),
                    "hipblasLtMatrixLayoutCreate weight");
  if (rc != TURBINE_OK)
    return rc;
  rc = check_blaslt(
      ctx, hipblasLtMatrixLayoutCreate(&act.handle, HIP_R_16BF, k, m, d->lda),
      "hipblasLtMatrixLayoutCreate activation");
  if (rc != TURBINE_OK)
    return rc;
  rc = check_blaslt(
      ctx, hipblasLtMatrixLayoutCreate(&out.handle, c_type, n, m, d->ldc),
      "hipblasLtMatrixLayoutCreate output");
  if (rc != TURBINE_OK)
    return rc;

  const turbine_hip::GemmKey key{d->m,   d->n,   d->k,       d->lda,
                                 d->ldb, d->ldc, d->trans_b, d->c_dtype};
  auto found = ctx->gemm_algos.find(key);
  if (found == ctx->gemm_algos.end()) {
    Preference pref;
    rc = check_blaslt(ctx, hipblasLtMatmulPreferenceCreate(&pref.handle),
                      "hipblasLtMatmulPreferenceCreate");
    if (rc != TURBINE_OK)
      return rc;
    const uint64_t max_ws = turbine_hip::kGemmWorkspaceBytes;
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
    rc = check_blaslt(
        ctx,
        hipblasLtMatmulAlgoGetHeuristic(ctx->blaslt, desc.handle, weight.handle,
                                        act.handle, out.handle, out.handle,
                                        pref.handle, 1, &result, &returned),
        "hipblasLtMatmulAlgoGetHeuristic");
    if (rc != TURBINE_OK)
      return rc;
    if (returned < 1 || result.state != HIPBLAS_STATUS_SUCCESS) {
      return fail(ctx, TURBINE_E_UNSUPPORTED,
                  "hipblasLtMatmulAlgoGetHeuristic: no algorithm for m=" +
                      std::to_string(d->m) + " n=" + std::to_string(d->n) +
                      " k=" + std::to_string(d->k));
    }
    if (ctx->gemm_algos.size() >= turbine_hip::kGemmAlgoCacheEntries) {
      ctx->gemm_algos.clear();
    }
    found = ctx->gemm_algos.emplace(key, result.algo).first;
  }

  return check_blaslt(
      ctx,
      hipblasLtMatmul(ctx->blaslt, desc.handle, &d->alpha, d->b, weight.handle,
                      d->a, act.handle, &d->beta, d->c, out.handle, d->c,
                      out.handle, &found->second, ctx->workspace,
                      turbine_hip::kGemmWorkspaceBytes, ctx->stream),
      "hipblasLtMatmul");
}

} // extern "C"
