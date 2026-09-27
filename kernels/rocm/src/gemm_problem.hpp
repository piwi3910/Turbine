// The hipBLASLt problem of one Turbine GEMM, shared by gemm.cpp and the GEMM
// tuner (tools/gemm_tune.cpp), so the tuner times exactly the descriptors the
// library runs.
//
// Row-major c[m,n] = alpha * a[m,k] . op(b) + beta * c is computed as the
// column-major product c^T[n,m] = op(b)^T[n,k] . a^T[k,m]: hipBLASLt's A is
// the weight (column-major k x n with ld ldb and HIPBLAS_OP_T when b is [n,k];
// n x k with HIPBLAS_OP_N when b is [k,n]), its B the activation (k x m, ld
// lda, HIPBLAS_OP_N), C and D the output (n x m, ld ldc). BF16 operands, F32
// accumulation, BF16 or F32 output.
#pragma once

#include <hipblaslt/hipblaslt.h>

#include <cstdint>

// Default visibility for the ABI declarations, as turbine_hip.hpp gives them:
// whichever header a translation unit includes first declares the exported
// turbine_* symbols.
#pragma GCC visibility push(default)
#include "turbine_kernels.h"
#pragma GCC visibility pop

namespace turbine_hip {

// Owns one hipBLASLt matrix layout.
struct Layout {
  hipblasLtMatrixLayout_t handle = nullptr;
  Layout() = default;
  Layout(const Layout &) = delete;
  Layout &operator=(const Layout &) = delete;
  ~Layout() {
    if (handle != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(handle);
  }
};

// Owns one hipBLASLt matmul descriptor.
struct MatmulDesc {
  hipblasLtMatmulDesc_t handle = nullptr;
  MatmulDesc() = default;
  MatmulDesc(const MatmulDesc &) = delete;
  MatmulDesc &operator=(const MatmulDesc &) = delete;
  ~MatmulDesc() {
    if (handle != nullptr)
      (void)hipblasLtMatmulDescDestroy(handle);
  }
};

// Owns one hipBLASLt heuristic preference.
struct Preference {
  hipblasLtMatmulPreference_t handle = nullptr;
  Preference() = default;
  Preference(const Preference &) = delete;
  Preference &operator=(const Preference &) = delete;
  ~Preference() {
    if (handle != nullptr)
      (void)hipblasLtMatmulPreferenceDestroy(handle);
  }
};

// The shape of one GEMM call (the fields of turbine_gemm_desc that define the
// problem; c_dtype is TURBINE_DTYPE_BF16 or TURBINE_DTYPE_F32).
struct GemmShape {
  int64_t m, n, k, lda, ldb, ldc;
  int32_t trans_b, c_dtype;
};

inline hipDataType gemm_out_type(int32_t c_dtype) {
  return c_dtype == TURBINE_DTYPE_F32 ? HIP_R_32F : HIP_R_16BF;
}

// hipBLASLt's A operation: the weight, transposed when b is [n, k].
inline hipblasOperation_t gemm_weight_op(int32_t trans_b) {
  return trans_b == 1 ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}

// A GEMM's descriptor and layouts. `failed` names the call that failed when
// make() returns an error.
struct GemmProblem {
  MatmulDesc desc;
  Layout weight;
  Layout act;
  Layout out;
  const char *failed = nullptr;

  hipblasStatus_t make(const GemmShape &s) {
    const hipblasOperation_t op_weight = gemm_weight_op(s.trans_b);
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
    const uint64_t m = static_cast<uint64_t>(s.m);
    const uint64_t n = static_cast<uint64_t>(s.n);
    const uint64_t k = static_cast<uint64_t>(s.k);
    st = s.trans_b == 1 ? hipblasLtMatrixLayoutCreate(&weight.handle,
                                                      HIP_R_16BF, k, n, s.ldb)
                        : hipblasLtMatrixLayoutCreate(&weight.handle,
                                                      HIP_R_16BF, n, k, s.ldb);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate weight", st;
    st = hipblasLtMatrixLayoutCreate(&act.handle, HIP_R_16BF, k, m, s.lda);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate activation", st;
    st = hipblasLtMatrixLayoutCreate(&out.handle, gemm_out_type(s.c_dtype), n,
                                     m, s.ldc);
    if (st != HIPBLAS_STATUS_SUCCESS)
      return failed = "hipblasLtMatrixLayoutCreate output", st;
    return HIPBLAS_STATUS_SUCCESS;
  }
};

} // namespace turbine_hip
