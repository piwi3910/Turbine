// Device code of the block-scaled FP8 GEMM turbine_hip_fp8_block
// (qgemm_fp8_block.hip), header-only so the evaluation harness
// (tools/qgemm_eval.cpp --block 1) times the kernels the library runs.
//
// Weights: e4m3 codes, row-major n x k, one F32 scale per 128 x 128 block
// ([ceil(n/128), k/128] row-major); activations BF16 (W8A16: act_quant NONE,
// decision "P6: block-scaled FP8 GEMM — provider evaluation"); out BF16 or F32.
//
// The fused kernel
// ----------------
// The structure of the INT4 fused kernel (qgemm_int4_kernels.hpp): one wave
// owns kFragsN 16-column WMMA tiles of the output and one 16-row tile of a;
// the kWaves waves of a block split k into whole 128-deep blocks and add their
// partial sums in LDS in wave order. Every lane streams its codes (32 bytes per
// 64-deep step) and its activation row (64 bytes per step) straight into the
// v_wmma_f32_16x16x16_bf16 operand registers.
//
//   - Codes to BF16 exactly: v_cvt_pk_f32_fp8 decodes two OCP e4m3 codes to
//     F32 (every e4m3 value, subnormals included, is exact in F32 and in BF16),
//     and the upper halves of the F32 words are the BF16 bits.
//   - Per 128-deep block the WMMA chain sums a * e4m3(q) in F32 (the products
//     are exact); the block's contribution is s_block * sum, added to the
//     output in ascending block order. The CPU reference sums a * (q * s), so
//     the two differ by rounding order only.
//   - k order within a step as in the INT4 kernel: lane half h owns k
//     [32h, 32h + 32), slice t takes k 32h + 8t .. + 8 on both operands.
//
// Batch invariance: an output element is the same chain in every launch of a
// given shape (per block a zero accumulator and its slices in ascending order,
// blocks split over the waves by a rule of k only, wave partials added in wave
// order), and a WMMA result element depends only on its row of a, its column of
// b and its accumulator: a row's result does not depend on m or on the other
// rows.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "bf16.hpp"

namespace turbine_hip {
namespace fp8_block {

constexpr int kLanes = 32;
constexpr int kMma = 16;
// k depth of one step: two lane halves of 32.
constexpr int kStep = 64;
// Weight block edge (block_n = block_k).
constexpr int kBlock = 128;
constexpr int kStepsPerBlock = kBlock / kStep;

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ inline v8f mma(v8s a, v8s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// BF16 bits of two e4m3 codes (bytes 0, 1 of w when hi is false, 2, 3 when
// true) packed low first.
__device__ inline uint32_t pair_bf16(uint32_t w, bool hi) {
  const v2f f =
      hi ? __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w), true)
         : __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w), false);
  return __builtin_amdgcn_perm(__float_as_uint(f[1]), __float_as_uint(f[0]),
                               0x07060302u);
}

// Eight codes (words lo: k 0..3, hi: k 4..7) as BF16.
__device__ inline v8s codes(uint32_t lo, uint32_t hi) {
  u32x4 out;
  out[0] = pair_bf16(lo, false);
  out[1] = pair_bf16(lo, true);
  out[2] = pair_bf16(hi, false);
  out[3] = pair_bf16(hi, true);
  return __builtin_bit_cast(v8s, out);
}

// Grid: (column tiles of 16 kFragsN, row tiles of 16). k is a multiple of
// 128; kFragsN * 16 divides 128, so a wave's columns share one block row of
// scales.
template <int kFragsN, int kWaves>
__global__ __launch_bounds__(kWaves *kLanes) void wmma_kernel(
    const Bf16 *a, const uint8_t *b, const float *scales, void *c, int64_t m,
    int64_t n, int64_t k, int64_t lda, int64_t ldc, int32_t c_f32,
    float alpha) {
  static_assert(kBlock % (kMma * kFragsN) == 0,
                "a wave's columns in one block");
  __shared__ float partial[kWaves > 1 ? kWaves - 1 : 1][kFragsN][8][kLanes];
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int wave = static_cast<int>(threadIdx.x) / kLanes;
  const int r = lane % kMma;
  const int h = lane / kMma;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kMma * kFragsN;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kMma;
  const int64_t blocks = k / kBlock;
  const int64_t g_begin = blocks * wave / kWaves;
  const int64_t g_end = blocks * (wave + 1) / kWaves;
  const float *block_scales = scales + (n0 / kBlock) * blocks;

  // Rows past m re-read the last row and columns past n the last column; their
  // results are dropped.
  const Bf16 *a_row = a + std::min(m0 + r, m - 1) * lda + 32 * h;
  const uint8_t *b_row[kFragsN];
#pragma unroll
  for (int j = 0; j < kFragsN; ++j)
    b_row[j] = b + std::min(n0 + j * kMma + r, n - 1) * k + 32 * h;

  v8f total[kFragsN];
#pragma unroll
  for (int j = 0; j < kFragsN; ++j)
    total[j] = v8f{};

  u32x4 wb[kStepsPerBlock][kFragsN][2];
  v8s av[kStepsPerBlock][4];
  auto load = [&](int64_t g) {
#pragma unroll
    for (int st = 0; st < kStepsPerBlock; ++st) {
      const int64_t k0 = g * kBlock + st * kStep;
#pragma unroll
      for (int j = 0; j < kFragsN; ++j) {
        wb[st][j][0] = *reinterpret_cast<const u32x4 *>(b_row[j] + k0);
        wb[st][j][1] = *reinterpret_cast<const u32x4 *>(b_row[j] + k0 + 16);
      }
#pragma unroll
      for (int t = 0; t < 4; ++t)
        av[st][t] = *reinterpret_cast<const v8s *>(a_row + k0 + 8 * t);
    }
  };
  if (g_begin < g_end)
    load(g_begin);
  for (int64_t g = g_begin; g < g_end; ++g) {
    u32x4 cb[kStepsPerBlock][kFragsN][2];
    v8s ca[kStepsPerBlock][4];
#pragma unroll
    for (int st = 0; st < kStepsPerBlock; ++st) {
#pragma unroll
      for (int j = 0; j < kFragsN; ++j) {
        cb[st][j][0] = wb[st][j][0];
        cb[st][j][1] = wb[st][j][1];
      }
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
#pragma unroll
    for (int st = 0; st < kStepsPerBlock; ++st) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int j = 0; j < kFragsN; ++j) {
          // Slice t: words 2t, 2t + 1 of the lane's 32 codes.
          const uint32_t lo = cb[st][j][t / 2][(t % 2) * 2];
          const uint32_t hi = cb[st][j][t / 2][(t % 2) * 2 + 1];
          acc[j] = mma(ca[st][t], codes(lo, hi), acc[j]);
        }
      }
    }
    const float s = block_scales[g];
#pragma unroll
    for (int j = 0; j < kFragsN; ++j)
#pragma unroll
      for (int v = 0; v < 8; ++v)
        total[j][v] = fmaf(s, acc[j][v], total[j][v]);
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

// out[row][k] = bf16(e4m3(q) * s_block) for rows [row0, row0 + rows) of the
// weight (what the golden reference's dequantized checkpoint holds); one
// thread per 16 codes. k is a multiple of 128.
__global__ void dequant_kernel(const uint8_t *b, const float *scales,
                               int64_t row0, int64_t rows, int64_t k,
                               Bf16 *out) {
  const int64_t chunks_per_row = k / 16;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * chunks_per_row)
    return;
  const int64_t row = i / chunks_per_row;
  const int64_t kk = (i % chunks_per_row) * 16;
  const int64_t src_row = row0 + row;
  const u32x4 w = *reinterpret_cast<const u32x4 *>(b + src_row * k + kk);
  const float s = scales[(src_row / kBlock) * (k / kBlock) + kk / kBlock];
  u32x4 o[2];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const v2f f =
          half != 0
              ? __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[q]), true)
              : __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[q]), false);
      const uint32_t lo = f32_to_bf16(f[0] * s).bits;
      const uint32_t hi = f32_to_bf16(f[1] * s).bits;
      const int e = q * 2 + half; // output word e of 8
      o[e / 4][e % 4] = lo | (hi << 16);
    }
  }
  *reinterpret_cast<u32x4 *>(out + row * k + kk) = o[0];
  *reinterpret_cast<u32x4 *>(out + row * k + kk + 8) = o[1];
}

template <int kFragsN, int kWaves>
hipError_t launch_wmma(const void *a, const uint8_t *b, const float *scales,
                       void *c, int64_t m, int64_t n, int64_t k, int64_t lda,
                       int64_t ldc, bool c_f32, float alpha,
                       hipStream_t stream) {
  const dim3 grid(
      static_cast<uint32_t>((n + kMma * kFragsN - 1) / (kMma * kFragsN)),
      static_cast<uint32_t>((m + kMma - 1) / kMma));
  const dim3 block(kWaves * kLanes);
  hipLaunchKernelGGL((wmma_kernel<kFragsN, kWaves>), grid, block, 0, stream,
                     static_cast<const Bf16 *>(a), b, scales, c, m, n, k, lda,
                     ldc, c_f32 ? 1 : 0, alpha);
  return hipGetLastError();
}

inline hipError_t launch_dequant(const uint8_t *b, const float *scales,
                                 int64_t row0, int64_t rows, int64_t k,
                                 void *out, hipStream_t stream) {
  constexpr int kThreads = 256;
  const int64_t chunks = rows * (k / 16);
  hipLaunchKernelGGL(
      dequant_kernel,
      dim3(static_cast<uint32_t>((chunks + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, b, scales, row0, rows, k,
      static_cast<Bf16 *>(out));
  return hipGetLastError();
}

} // namespace fp8_block
} // namespace turbine_hip
