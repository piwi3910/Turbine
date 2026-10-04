// The hipBLASLt problem of one FP8 W8A8 quantized GEMM (ABI v2.9 turbine_qgemm,
// schemes FP8_TENSOR / FP8_CHANNEL with activations FP8_TENSOR / FP8_TOKEN),
// shared by qgemm.cpp and the provider evaluation harness
// (tools/qgemm_eval.cpp), so the harness times exactly the descriptors the
// library runs.
//
// As gemm_problem.hpp: row-major c[m,n] = alpha * a[m,k] . b[n,k]^T is the
// column-major c^T[n,m] = b^T[n,k] . a^T[k,m], so hipBLASLt's A is the weight
// (e4m3, column-major k x n with ld k, HIPBLAS_OP_T), its B the activation
// (e4m3, k x m with ld lda, HIPBLAS_OP_N), C and D the output (n x m, ld ldc,
// BF16 or F32). The scales ride on the operands: the weight's on A (a scalar,
// HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, for FP8_TENSOR; a vector of n,
// OUTER_VEC_32F, for FP8_CHANNEL), the activation's on B (a scalar for
// FP8_TENSOR, a vector of m for FP8_TOKEN). hipBLASLt computes
// D = alpha * (scale_a[i] * A) . (scale_b[j] * B), F32 accumulation: exactly
// the sum the CPU reference forms (e4m3 x e4m3 products are exact in F32).
#pragma once

#include <hipblaslt/hipblaslt.h>

#include <cstdint>

#include "gemm_problem.hpp"

namespace turbine_hip {

// How one operand's scales are laid out.
enum class QScale : int32_t { Scalar = 0, Vector = 1 };

inline hipblasLtMatmulMatrixScale_t qscale_mode(QScale s) {
  return s == QScale::Vector ? HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                             : HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}

// The shape and scale layout of one FP8 GEMM call.
struct QGemmShape {
  int64_t m, n, k, lda, ldc;
  int32_t c_dtype;
  QScale weight_scale, act_scale;
};

// The weight's scale layout for a TURBINE_QSCHEME_* (FP8 schemes only).
inline QScale weight_qscale(int32_t scheme) {
  return scheme == TURBINE_QSCHEME_FP8_CHANNEL ? QScale::Vector
                                               : QScale::Scalar;
}

// The activation's scale layout for a TURBINE_ACTQ_* (FP8_TENSOR / _TOKEN).
inline QScale act_qscale(int32_t act_quant) {
  return act_quant == TURBINE_ACTQ_FP8_TOKEN ? QScale::Vector : QScale::Scalar;
}

// An FP8 GEMM's descriptor and layouts. make() builds them; set_scales() points
// the descriptor at one call's scale vectors (device pointers, F32). `failed`
// names the call that failed when either returns an error.
struct QGemmProblem {
  MatmulDesc desc;
  Layout weight;
  Layout act;
  Layout out;
  const char *failed = nullptr;

  hipblasStatus_t make(const QGemmShape &s) {
    const hipblasOperation_t op_weight = HIPBLAS_OP_T;
    const hipblasOperation_t op_act = HIPBLAS_OP_N;
    hipblasStatus_t st =
        hipblasLtMatmulDescCreate(&desc.handle, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescCreate", st;
    st = hipblasLtMatmulDescSetAttribute(desc.handle,
                                         HIPBLASLT_MATMUL_DESC_TRANSA,
                                         &op_weight, sizeof(op_weight));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute TRANSA", st;
    st = hipblasLtMatmulDescSetAttribute(
        desc.handle, HIPBLASLT_MATMUL_DESC_TRANSB, &op_act, sizeof(op_act));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute TRANSB", st;
    const hipblasLtMatmulMatrixScale_t a_mode = qscale_mode(s.weight_scale);
    const hipblasLtMatmulMatrixScale_t b_mode = qscale_mode(s.act_scale);
    st = hipblasLtMatmulDescSetAttribute(desc.handle,
                                         HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                         &a_mode, sizeof(a_mode));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute A_SCALE_MODE", st;
    st = hipblasLtMatmulDescSetAttribute(desc.handle,
                                         HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                         &b_mode, sizeof(b_mode));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute B_SCALE_MODE", st;
    const uint64_t m = static_cast<uint64_t>(s.m);
    const uint64_t n = static_cast<uint64_t>(s.n);
    const uint64_t k = static_cast<uint64_t>(s.k);
    st = hipblasLtMatrixLayoutCreate(&weight.handle, HIP_R_8F_E4M3, k, n, k);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate weight", st;
    st = hipblasLtMatrixLayoutCreate(&act.handle, HIP_R_8F_E4M3, k, m, s.lda);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate activation", st;
    st = hipblasLtMatrixLayoutCreate(&out.handle, gemm_out_type(s.c_dtype), n,
                                     m, s.ldc);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate output", st;
    return HIPBLAS_STATUS_SUCCESS;
  }

  // weight_scales: F32 [1] or [n]; act_scales: F32 [1] or [m] (device).
  hipblasStatus_t set_scales(const float *weight_scales,
                             const float *act_scales) {
    const void *a = weight_scales;
    const void *b = act_scales;
    hipblasStatus_t st = hipblasLtMatmulDescSetAttribute(
        desc.handle, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a, sizeof(a));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute A_SCALE_POINTER", st;
    st = hipblasLtMatmulDescSetAttribute(
        desc.handle, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b, sizeof(b));
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatmulDescSetAttribute B_SCALE_POINTER", st;
    return HIPBLAS_STATUS_SUCCESS;
  }
};

} // namespace turbine_hip
