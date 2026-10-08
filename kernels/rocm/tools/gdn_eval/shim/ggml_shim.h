// Turbine gdn_eval shim: the subset of ggml-cuda/common.cuh (+ vendors/hip.h)
// that gated_delta_net.cu and ssm-conv.cu use, for HIP / wave32 gfx12.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <limits>
typedef hipStream_t cudaStream_t;
#define GGML_ASSERT(x)                                                         \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "GGML_ASSERT %s\n", #x);                                 \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define GGML_ABORT(m)                                                          \
  do {                                                                         \
    fprintf(stderr, "GGML_ABORT %s\n", m);                                     \
    abort();                                                                   \
  } while (0)
#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_RESTRICT __restrict__
#define WARP_SIZE 32
#define __shfl_xor_sync(mask, var, laneMask, width)                            \
  __shfl_xor(var, laneMask, width)
static constexpr __device__ __host__ int ggml_cuda_get_physical_warp_size() {
#if defined(__GFX9__) || defined(__GFX8__)
  return 64;
#else
  return 32;
#endif
}
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}
static __device__ __forceinline__ void ggml_cuda_pdl_lc() {}
template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  return x;
}
static const uint3 init_fastdiv_values(uint64_t d_64) {
  GGML_ASSERT(d_64 != 0);
  uint32_t d = (uint32_t)d_64, L = 0;
  while (L < 32 && (uint32_t{1} << L) < d)
    L++;
  uint32_t mp =
      (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
  return make_uint3(mp, L, d);
}
static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 f) {
  return (__umulhi(n, f.x) + n) >> f.y;
}
static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n,
                                                      const uint3 f) {
  return n - fastdiv(n, f) * f.z;
}
__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
  return x / (1.0f + expf(-x));
}
struct ggml_cuda_kernel_launch_params {
  dim3 g, b;
  size_t smem;
  hipStream_t s;
  ggml_cuda_kernel_launch_params(dim3 g_, dim3 b_, size_t m, hipStream_t s_)
      : g(g_), b(b_), smem(m), s(s_) {}
};
template <typename Kern, typename... A>
static void ggml_cuda_kernel_launch(Kern k,
                                    const ggml_cuda_kernel_launch_params &p,
                                    A... a) {
  hipLaunchKernelGGL(k, p.g, p.b, p.smem, p.s, a...);
}
struct ggml_dev_info {
  int warp_size;
};
struct ggml_info_t {
  ggml_dev_info devices[1];
};
static const ggml_info_t &ggml_cuda_info() {
  static ggml_info_t i{{{32}}};
  return i;
}
static int ggml_cuda_get_device() { return 0; }
