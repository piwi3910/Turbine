// The Composable Kernel candidates of the quantized GEMM provider evaluation
// (qgemm_eval_ck.cpp), callable from tools/qgemm_eval.cpp without CK headers.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

enum class CkQuantMode { RowCol, Tensor, Block128 };

// Enqueues c[m, n] (BF16) = a[m, k] (e4m3) . b[n, k]^T (e4m3) with mode's
// scales on stream. Returns 0 when launched, 1 when CK refuses the problem, 2
// on a launch error.
int ck_qgemm(CkQuantMode mode, bool prefill_tile, const void *a, const void *b,
             void *c, const float *a_scales, const float *b_scales, int64_t m,
             int64_t n, int64_t k, hipStream_t stream);
