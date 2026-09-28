// Device code of the INT4 group GEMM (qgemm_int4.hip): the fused WMMA kernel
// and the dequantization kernel, templated on their launch shape so that
// tools/qgemm_int4_eval can sweep the shapes the library fixes.
//
// The fused kernel (turbine_hip_int4_wmma)
// ----------------------------------------
// One wave owns kFragsN 16-column WMMA tiles of the output and one 16-row tile
// of a; the kWaves waves of a block split k into whole groups and add their
// partial sums in LDS in wave order. Nothing is staged: every lane streams its
// weight codes (16 bytes per 64-deep step) and its activation row (64 bytes per
// step) straight into the v_wmma_f32_16x16x16_bf16 operand registers, the next
// group's loads in flight while the current group computes.
//
//   - Codes to BF16 without arithmetic: 0x4300 | q is the BF16 of 128 + q,
//     exactly (q < 16). A group's WMMA chain therefore sums (128 + q) * a, and
//     a second WMMA of the same activations against a tile of ones sums a; the
//     group's contribution is s * (acc - (128 + z) * sum(a)), in F32.
//   - k order: within a 64-deep step, lane half h (lanes 16h .. 16h + 15) owns
//     k [32h, 32h + 32) and slice t of the step takes k 32h + 8t .. + 8 on both
//     operands, so a lane's codes for the whole step are one 16-byte load and
//     its activations four. A WMMA sums over its 16 k in any order the two
//     operands agree on; the order is fixed, so results are reproducible.
//
// Batch invariance: an output element is the same chain in every launch of a
// given shape -- per group a zero accumulator and the group's slices in
// ascending order, groups split over the waves by the same rule for a given k,
// wave partials added in wave order -- and a WMMA result element depends only
// on its row of a, its column of b and its accumulator. So a row's result does
// not depend on m or on the other rows.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "bf16.hpp"

namespace turbine_hip {
namespace int4 {

constexpr int kLanes = 32;
constexpr int kMma = 16;
// k depth of one step: two lane halves of 32.
constexpr int kStep = 64;

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ inline v8f mma(v8s a, v8s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// The eight codes of word w (k offsets 0..7, low nibble first) as BF16 of
// 128 + q: element pair p is bytes p of (w & 0x0f0f0f0f) and of
// ((w >> 4) & 0x0f0f0f0f) in the low and high half, or'ed with 0x4300.
__device__ inline v8s codes(uint32_t w) {
  const uint32_t lo = w & 0x0f0f0f0fu;
  const uint32_t hi = (w >> 4) & 0x0f0f0f0fu;
  u32x4 out;
  out[0] = __builtin_amdgcn_perm(hi, lo, 0x0c040c00u) | 0x43004300u;
  out[1] = __builtin_amdgcn_perm(hi, lo, 0x0c050c01u) | 0x43004300u;
  out[2] = __builtin_amdgcn_perm(hi, lo, 0x0c060c02u) | 0x43004300u;
  out[3] = __builtin_amdgcn_perm(hi, lo, 0x0c070c03u) | 0x43004300u;
  return __builtin_bit_cast(v8s, out);
}

// The fused kernel for groups of kSteps * 64. Grid: (column tiles of
// 16 kFragsN, row tiles of 16).
template <int kFragsN, int kWaves, int kSteps>
__global__ __launch_bounds__(kWaves *kLanes) void wmma_kernel(
    const Bf16 *a, const uint8_t *b, const float *scales, const uint8_t *zeros,
    void *c, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc,
    int32_t c_f32, float alpha) {
  constexpr int kGroup = kSteps * kStep;
  __shared__ float partial[kWaves > 1 ? kWaves - 1 : 1][kFragsN][8][kLanes];
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int wave = static_cast<int>(threadIdx.x) / kLanes;
  const int r = lane % kMma;
  const int h = lane / kMma;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kMma * kFragsN;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kMma;
  const int64_t groups = k / kGroup;
  const int64_t g_begin = groups * wave / kWaves;
  const int64_t g_end = groups * (wave + 1) / kWaves;
  const int64_t row_bytes = k / 2;

  // Rows past m re-read the last row and columns past n the last column; their
  // results are dropped.
  const Bf16 *a_row = a + std::min(m0 + r, m - 1) * lda + 32 * h;
  const uint8_t *b_row[kFragsN];
  int64_t col[kFragsN];
#pragma unroll
  for (int j = 0; j < kFragsN; ++j) {
    col[j] = std::min(n0 + j * kMma + r, n - 1);
    b_row[j] = b + col[j] * row_bytes + 16 * h;
  }

  v8s ones;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    ones[i] = static_cast<short>(0x3f80);

  v8f total[kFragsN];
#pragma unroll
  for (int j = 0; j < kFragsN; ++j)
    total[j] = v8f{};

  u32x4 wb[kSteps][kFragsN];
  v8s av[kSteps][4];
  auto load = [&](int64_t g) {
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int64_t k0 = g * kGroup + st * kStep;
#pragma unroll
      for (int j = 0; j < kFragsN; ++j)
        wb[st][j] = *reinterpret_cast<const u32x4 *>(b_row[j] + k0 / 2);
#pragma unroll
      for (int t = 0; t < 4; ++t)
        av[st][t] = *reinterpret_cast<const v8s *>(a_row + k0 + 8 * t);
    }
  };
  if (g_begin < g_end)
    load(g_begin);
  for (int64_t g = g_begin; g < g_end; ++g) {
    u32x4 cb[kSteps][kFragsN];
    v8s ca[kSteps][4];
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
#pragma unroll
      for (int j = 0; j < kFragsN; ++j)
        cb[st][j] = wb[st][j];
#pragma unroll
      for (int t = 0; t < 4; ++t)
        ca[st][t] = av[st][t];
    }
    if (g + 1 < g_end)
      load(g + 1);
    v8f acc[kFragsN];
#pragma unroll
    for (int j = 0; j < kFragsN; ++j)
      acc[j] = v8f{};
    v8f asum = v8f{};
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        asum = mma(ca[st][t], ones, asum);
#pragma unroll
        for (int j = 0; j < kFragsN; ++j)
          acc[j] = mma(ca[st][t], codes(cb[st][j][t]), acc[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < kFragsN; ++j) {
      const int64_t at = col[j] * groups + g;
      const float s = scales[at];
      const float bias =
          128.0f + (zeros != nullptr ? static_cast<float>(zeros[at]) : 8.0f);
#pragma unroll
      for (int v = 0; v < 8; ++v)
        total[j][v] = fmaf(s, fmaf(-bias, asum[v], acc[j][v]), total[j][v]);
    }
  }

  if (kWaves > 1) {
    if (wave > 0) {
#pragma unroll
      for (int j = 0; j < kFragsN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v)
          partial[wave - 1][j][v][lane] = total[j][v];
    }
    __syncthreads();
    if (wave > 0)
      return;
#pragma unroll
    for (int w = 1; w < kWaves; ++w)
#pragma unroll
      for (int j = 0; j < kFragsN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v)
          total[j][v] += partial[w - 1][j][v][lane];
  }

#pragma unroll
  for (int j = 0; j < kFragsN; ++j) {
    const int64_t cn = n0 + j * kMma + r;
    if (cn >= n)
      continue;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int64_t row = m0 + 8 * h + v;
      if (row >= m)
        continue;
      const float value = alpha * total[j][v];
      if (c_f32)
        static_cast<float *>(c)[row * ldc + cn] = value;
      else
        static_cast<Bf16 *>(c)[row * ldc + cn] = f32_to_bf16(value);
    }
  }
}

// out[row][k] = bf16((q - z) * s) for rows [row0, row0 + rows) of weight b;
// one thread per 32 codes (16 bytes in, 64 bytes out). k is a multiple of 32
// and group a multiple of 32.
__global__ void dequant_kernel(const uint8_t *b, const float *scales,
                               const uint8_t *zeros, int64_t row0, int64_t rows,
                               int64_t k, int32_t group, Bf16 *out) {
  const int64_t chunks_per_row = k / 32;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * chunks_per_row)
    return;
  const int64_t row = i / chunks_per_row;
  const int64_t kk = (i % chunks_per_row) * 32;
  const int64_t src_row = row0 + row;
  const u32x4 w =
      *reinterpret_cast<const u32x4 *>(b + src_row * (k / 2) + kk / 2);
  const int64_t at = src_row * (k / group) + kk / group;
  const float s = scales[at];
  const float z = zeros != nullptr ? static_cast<float>(zeros[at]) : 8.0f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    u32x4 o;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float lo = (static_cast<float>((w[q] >> (8 * p)) & 0xfu) - z) * s;
      const float hi =
          (static_cast<float>((w[q] >> (8 * p + 4)) & 0xfu) - z) * s;
      o[p] = static_cast<uint32_t>(f32_to_bf16(lo).bits) |
             (static_cast<uint32_t>(f32_to_bf16(hi).bits) << 16);
    }
    *reinterpret_cast<u32x4 *>(out + row * k + kk + 8 * q) = o;
  }
}

// Launches wmma_kernel<kFragsN, kWaves, group / 64> for group 64, 128 or 256
// (other groups: hipErrorInvalidValue).
template <int kFragsN, int kWaves>
hipError_t launch_wmma(const void *a, const uint8_t *b, const float *scales,
                       const uint8_t *zeros, void *c, int64_t m, int64_t n,
                       int64_t k, int64_t lda, int64_t ldc, int32_t group,
                       bool c_f32, float alpha, hipStream_t stream) {
  const dim3 grid(
      static_cast<uint32_t>((n + kMma * kFragsN - 1) / (kMma * kFragsN)),
      static_cast<uint32_t>((m + kMma - 1) / kMma));
  const dim3 block(kWaves * kLanes);
  const Bf16 *ab = static_cast<const Bf16 *>(a);
  const int32_t cf = c_f32 ? 1 : 0;
  switch (group) {
  case 64:
    hipLaunchKernelGGL((wmma_kernel<kFragsN, kWaves, 1>), grid, block, 0,
                       stream, ab, b, scales, zeros, c, m, n, k, lda, ldc, cf,
                       alpha);
    break;
  case 128:
    hipLaunchKernelGGL((wmma_kernel<kFragsN, kWaves, 2>), grid, block, 0,
                       stream, ab, b, scales, zeros, c, m, n, k, lda, ldc, cf,
                       alpha);
    break;
  case 256:
    hipLaunchKernelGGL((wmma_kernel<kFragsN, kWaves, 4>), grid, block, 0,
                       stream, ab, b, scales, zeros, c, m, n, k, lda, ldc, cf,
                       alpha);
    break;
  default:
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

} // namespace int4
} // namespace turbine_hip
