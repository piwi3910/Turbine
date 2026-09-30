// The scale epilogue of hipblaslt_fp8's prefill path for vector scales
// (qgemm.cpp), shared with the evaluation harness (tools/qgemm_eval.cpp):
// c[r, j] = BF16(acc[r, j] * w_scale[j] * a_scale[r] * alpha), acc the F32
// e4m3 x e4m3 sums of a scalar-scale hipBLASLt GEMM run with unit scales.
// gfx1201's hipBLASLt has no vector-scale (OUTER_VEC) FP8 solution whose rows
// are independent of the call's size and of their position in it, which
// prefix reuse needs in prefill steps (decision "P6: FP8 GEMM — provider
// evaluation (kernel reuse rule)"); its scalar-scale solutions include such
// ones, so the per-row and per-column scales move into this own kernel.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace turbine_hip {

constexpr int kEpilogueThreads = 256;

// One thread per output element; grid (ceil(n / 256), rows).
__global__ void __launch_bounds__(kEpilogueThreads)
    qgemm_scale_epilogue(const float *acc, int64_t ld_acc, uint16_t *c,
                         int64_t ldc, const float *a_scales, int32_t a_vector,
                         const float *w_scales, int32_t w_vector, float alpha,
                         int64_t n) {
  const int64_t r = blockIdx.y;
  const int64_t j =
      static_cast<int64_t>(blockIdx.x) * kEpilogueThreads + threadIdx.x;
  if (j >= n)
    return;
  float v = acc[r * ld_acc + j] * w_scales[w_vector != 0 ? j : 0];
  v = v * a_scales[a_vector != 0 ? r : 0];
  v = v * alpha;
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(b));
  uint16_t h;
  if ((b & 0x7fffffffu) > 0x7f800000u) {
    h = 0x7fc0; // canonical NaN
  } else {
    h = static_cast<uint16_t>((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
  }
  c[r * ldc + j] = h;
}

inline hipError_t
launch_qgemm_scale_epilogue(const float *acc, int64_t ld_acc, void *c,
                            int64_t ldc, const float *a_scales, bool a_vector,
                            const float *w_scales, bool w_vector, float alpha,
                            int64_t rows, int64_t n, hipStream_t stream) {
  const dim3 grid(
      static_cast<uint32_t>((n + kEpilogueThreads - 1) / kEpilogueThreads),
      static_cast<uint32_t>(rows));
  hipLaunchKernelGGL(qgemm_scale_epilogue, grid, dim3(kEpilogueThreads), 0,
                     stream, acc, ld_acc, static_cast<uint16_t *>(c), ldc,
                     a_scales, a_vector ? 1 : 0, w_scales, w_vector ? 1 : 0,
                     alpha, n);
  return hipGetLastError();
}

} // namespace turbine_hip
