// FP8 e4m3 activation quantization kernels (ABI v2.9 turbine_quantize_act,
// modes FP8_TENSOR, FP8_TOKEN and FP8_GROUP128), shared by quantize_act.hip and
// the provider evaluation harness (tools/qgemm_eval.cpp).
//
// Bit-exact with the CPU reference (crates/turbine-kernels/src/cpu/quant.rs):
// the dynamic scale of a row (TOKEN) or of a row's 128-column group (GROUP128)
// is max(amax / 448, 1 / (448 * 512)) in F32 (amax is exact: a max of
// magnitudes), each value is divided by its scale in F32 (HIP's default
// correctly rounded division) and rounded to OCP e4m3fn with ties to even,
// saturated to +-448 (NaN -> 0x7f) by fp8_e4m3_round below, which is the
// reference's function transcribed; no hardware conversion is used, so no
// denormal flushing or rounding mode of the device can change a byte.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

// Default visibility for the ABI declarations (as turbine_hip.hpp gives them).
#pragma GCC visibility push(default)
#include "turbine_kernels.h"
#pragma GCC visibility pop

namespace turbine_hip {

constexpr float kFp8E4m3Max = 448.0f;
constexpr float kFp8MinScale = 1.0f / (448.0f * 512.0f);
// Columns per GROUP128 scale.
constexpr int kActGroup = 128;
// Threads of one quantize block.
constexpr int kQuantThreads = 256;

__host__ __device__ inline uint8_t fp8_e4m3_round(float x) {
  if (x != x)
    return 0x7f;
  uint32_t bits = 0;
  __builtin_memcpy(&bits, &x, sizeof(bits));
  const uint8_t sign = (bits >> 31) != 0 ? 0x80 : 0;
  const float a = x < 0.0f ? -x : x;
  if (a >= kFp8E4m3Max)
    return sign | 0x7e;
  // Below the smallest normal (2^-6): subnormals are multiples of 2^-9; the
  // scaling by 512 is exact and rintf rounds half to even.
  if (a < 0.015625f) {
    const uint8_t m = static_cast<uint8_t>(__builtin_rintf(a * 512.0f));
    return sign | m;
  }
  uint32_t abits = 0;
  __builtin_memcpy(&abits, &a, sizeof(abits));
  int32_t exp = static_cast<int32_t>((abits >> 23) & 0xff) - 127;
  const uint32_t mant = abits & 0x7fffff;
  uint32_t q = mant >> 20;
  const uint32_t rem = mant & 0xfffff;
  const uint32_t half = 0x80000;
  if (rem > half || (rem == half && (q & 1) == 1))
    q += 1;
  if (q == 8) {
    q = 0;
    exp += 1;
  }
  const uint8_t biased = static_cast<uint8_t>(exp + 7);
  return sign | static_cast<uint8_t>(biased << 3) | static_cast<uint8_t>(q);
}

// The value of e4m3fn bits (NaN for 0x7f / 0xff).
__host__ __device__ inline float fp8_e4m3_value(uint8_t b) {
  const float sign = (b & 0x80) != 0 ? -1.0f : 1.0f;
  const int e = (b >> 3) & 0xf;
  const int m = b & 7;
  if (e == 15 && m == 7)
    return __builtin_nanf("");
  const float mag = e == 0 ? static_cast<float>(m) / 8.0f * 0.015625f
                           : (1.0f + static_cast<float>(m) / 8.0f) *
                                 __builtin_ldexpf(1.0f, e - 7);
  return sign * mag;
}

__host__ __device__ inline float dynamic_fp8_scale(float amax) {
  const float s = amax / kFp8E4m3Max;
  return s > kFp8MinScale ? s : kFp8MinScale;
}

// x_bf16: element as BF16 bits (true) or F32.
template <bool Bf16>
__device__ inline float load_act(const void *x, int64_t i) {
  if constexpr (Bf16) {
    const uint32_t bits =
        static_cast<uint32_t>(static_cast<const uint16_t *>(x)[i]) << 16;
    float v;
    __builtin_memcpy(&v, &bits, sizeof(v));
    return v;
  } else {
    return static_cast<const float *>(x)[i];
  }
}

// Block-wide max of v over kQuantThreads threads (all threads get it).
__device__ inline float block_max(float v, float *red) {
  const int t = threadIdx.x;
  red[t] = v;
  __syncthreads();
  for (int s = kQuantThreads / 2; s > 0; s >>= 1) {
    if (t < s)
      red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

// One block per row. Mode: TURBINE_ACTQ_FP8_TENSOR (static_scale),
// TURBINE_ACTQ_FP8_TOKEN or TURBINE_ACTQ_FP8_GROUP128.
template <bool Bf16, int32_t Mode>
__global__ void __launch_bounds__(kQuantThreads)
    quantize_fp8_rows(const void *x, uint8_t *out, float *scales, int64_t cols,
                      int64_t x_stride, int64_t out_stride,
                      float static_scale) {
  __shared__ float red[kQuantThreads];
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x;
  const int64_t xb = row * x_stride;
  uint8_t *o = out + row * out_stride;
  if constexpr (Mode == TURBINE_ACTQ_FP8_TENSOR) {
    if (row == 0 && t == 0)
      scales[0] = static_scale;
    for (int64_t c = t; c < cols; c += kQuantThreads)
      o[c] = fp8_e4m3_round(load_act<Bf16>(x, xb + c) / static_scale);
  } else if constexpr (Mode == TURBINE_ACTQ_FP8_TOKEN) {
    float amax = 0.0f;
    for (int64_t c = t; c < cols; c += kQuantThreads)
      amax = fmaxf(amax, fabsf(load_act<Bf16>(x, xb + c)));
    const float s = dynamic_fp8_scale(block_max(amax, red));
    if (t == 0)
      scales[row] = s;
    for (int64_t c = t; c < cols; c += kQuantThreads)
      o[c] = fp8_e4m3_round(load_act<Bf16>(x, xb + c) / s);
  } else {
    // GROUP128: two groups per pass (kQuantThreads = 2 x 128); each half-block
    // reduces its group through the shared buffer.
    static_assert(kQuantThreads == 2 * kActGroup, "two groups per pass");
    const int64_t groups = (cols + kActGroup - 1) / kActGroup;
    const int half = t / kActGroup;
    const int lane = t % kActGroup;
    for (int64_t g0 = 0; g0 < groups; g0 += 2) {
      const int64_t g = g0 + half;
      const int64_t c = g * kActGroup + lane;
      const bool live = g < groups && c < cols;
      const float v = live ? load_act<Bf16>(x, xb + c) : 0.0f;
      red[t] = fabsf(v);
      __syncthreads();
      for (int s = kActGroup / 2; s > 0; s >>= 1) {
        if (lane < s)
          red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
      }
      const float scale = dynamic_fp8_scale(red[half * kActGroup]);
      __syncthreads();
      if (live) {
        o[c] = fp8_e4m3_round(v / scale);
        if (lane == 0)
          scales[row * groups + g] = scale;
      }
    }
  }
}

// Enqueues the quantization d describes on stream; d is validated (FP8 mode,
// BF16 or F32 x, e4m3 out, rows and cols > 0). Returns the launch error.
inline hipError_t launch_quantize_fp8(const turbine_quantize_act_desc *d,
                                      hipStream_t stream) {
  const dim3 grid(static_cast<uint32_t>(d->rows));
  const dim3 block(kQuantThreads);
  auto *out = static_cast<uint8_t *>(d->out);
  const bool bf16 = d->x_dtype == TURBINE_DTYPE_BF16;
#define TURBINE_QUANT_LAUNCH(B, M)                                             \
  hipLaunchKernelGGL((quantize_fp8_rows<B, M>), grid, block, 0, stream, d->x,  \
                     out, d->scales, d->cols, d->x_stride_row,                 \
                     d->out_stride_row, d->static_scale)
  switch (d->mode) {
  case TURBINE_ACTQ_FP8_TENSOR:
    if (bf16)
      TURBINE_QUANT_LAUNCH(true, TURBINE_ACTQ_FP8_TENSOR);
    else
      TURBINE_QUANT_LAUNCH(false, TURBINE_ACTQ_FP8_TENSOR);
    break;
  case TURBINE_ACTQ_FP8_TOKEN:
    if (bf16)
      TURBINE_QUANT_LAUNCH(true, TURBINE_ACTQ_FP8_TOKEN);
    else
      TURBINE_QUANT_LAUNCH(false, TURBINE_ACTQ_FP8_TOKEN);
    break;
  default:
    if (bf16)
      TURBINE_QUANT_LAUNCH(true, TURBINE_ACTQ_FP8_GROUP128);
    else
      TURBINE_QUANT_LAUNCH(false, TURBINE_ACTQ_FP8_GROUP128);
    break;
  }
#undef TURBINE_QUANT_LAUNCH
  return hipGetLastError();
}

} // namespace turbine_hip
