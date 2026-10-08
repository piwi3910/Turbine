// The grouped prefill kernel of the quantized MoE experts
// (turbine_moe_quant_grouped, P7 Task 16 step 2; moe.cpp run_quant calls it
// from the per-expert loop's place at >= 64 routed rows).
//
// One launch per projection over every active expert, the tile-per-expert
// schedule of moe_grouped.hip's wmma_prefill: a block finds its (expert, row
// tile) from expert_offsets, blocks past the last row tile return, the column
// tiles of a row tile are adjacent in the grid. A block's waves (4 x 2, kFragsM
// x 2 WMMA tiles each) walk k in steps of 64: the threads load the step's
// activation and weight tiles with 16-byte loads, decode the weight tile into
// LDS, then run the WMMAs from LDS.
//
//   FP8 block   LDS holds bf16(e4m3(q) * s_block), the dequantized weight the
//               per-expert loop stages for hipBLASLt (BF16 WMMA, F32 sums);
//   MXFP4       LDS holds e2m1(code) * 2^(e8m0 - 127) as BF16 (exact), BF16
//               WMMA;
//   FP8 tensor  W8A8: LDS holds the e4m3 activation and weight codes,
//               v_wmma_f32_16x16x16_fp8_fp8 (exact products, F32 sums), then
//               ((acc * w_scale) * a_scale) rounded to BF16, as the loop's
//               hipblaslt_fp8 epilogue (qgemm_epilogue.hpp).
// The output is the loop's: a BF16 [rows, ldc] matrix, so the bias, activation
// and scatter kernels of moe_quant.hip and moe.hip run unchanged. A row's value
// is one WMMA chain over ascending k whatever the other rows (and the tile
// shape), so it does not depend on the batch.
#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;

namespace {

constexpr int kTileK = 64;
constexpr int kLds16 = kTileK + 8; // BF16 tile row, 144 bytes
constexpr int kLds8 = kTileK + 16; // FP8 tile row, 80 bytes

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__constant__ const uint16_t kE2m1Bf16[16] = {
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0};

__device__ inline v8f mma16(v8s a, v8s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ inline v8f mma8(v2i a, v2i b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
// Words 2t, 2t + 1 of a lane's 32 codes as one FP8 WMMA operand.
__device__ inline v2i slice(const u32x4 (&w)[2], int t) {
  v2i out;
  out[0] = static_cast<int>(w[t / 2][(t % 2) * 2]);
  out[1] = static_cast<int>(w[t / 2][(t % 2) * 2 + 1]);
  return out;
}

// bf16(e4m3(q) * s) of 16 codes, as the dequant kernel of qgemm_fp8_block.
__device__ inline void decode_block(u32x4 w, float s, u32x4 (&o)[2]) {
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
      const int e = q * 2 + half;
      o[e / 4][e % 4] = lo | (hi << 16);
    }
  }
}

__device__ inline float e8m0_scale(uint32_t e) {
  if (e == 0)
    return __uint_as_float(0x00400000u);
  if (e == 255)
    return __uint_as_float(0x7fc00000u);
  return __uint_as_float(e << 23);
}

// Eight BF16 weights from four code bytes (low nibble = the even element)
// times 2^(e - 127); the exponent add is exact for 2 <= e <= 252.
__device__ inline u32x4 decode_mx(const uint32_t *lut, uint32_t word,
                                  uint32_t e) {
  u32x4 v{lut[word & 0xffu], lut[(word >> 8) & 0xffu],
          lut[(word >> 16) & 0xffu], lut[word >> 24]};
  if (e >= 2u && e <= 252u) {
    const uint32_t delta = static_cast<uint32_t>(static_cast<int32_t>(e) - 127)
                           << 7;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t nz = ((v[i] & 0x7fff7fffu) + 0x7fff7fffu) & 0x80008000u;
      v[i] += (nz >> 15) * delta;
    }
  } else {
    const float scale = e8m0_scale(e);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float lo = __uint_as_float(v[i] << 16) * scale;
      const float hi = __uint_as_float(v[i] & 0xffff0000u) * scale;
      v[i] = (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
    }
  }
  return v;
}

// The (local expert, first position, end position) of row tile `tile`, or
// expert -1 past the last (moe_grouped.hip find_tile).
template <int kThreads, int kTileM>
__device__ void find_tile(const int32_t *offsets, int32_t expert_begin,
                          int32_t num_local, int32_t tile, int32_t *expert,
                          int32_t *begin, int32_t *end) {
  static_assert(turbine_hip::kMoeMaxExperts % kThreads == 0,
                "the tile scan covers the experts in whole passes");
  __shared__ int32_t ends[turbine_hip::kMoeMaxExperts];
  __shared__ int32_t found[3];
  const int t = static_cast<int>(threadIdx.x);
  if (t == 0)
    found[0] = -1;
  for (int32_t e = t; e < turbine_hip::kMoeMaxExperts; e += kThreads) {
    ends[e] = e < num_local ? (offsets[expert_begin + e + 1] -
                               offsets[expert_begin + e] + kTileM - 1) /
                                  kTileM
                            : 0;
  }
  __syncthreads();
  for (int32_t step = 1; step < num_local; step *= 2) {
    int32_t add[turbine_hip::kMoeMaxExperts / kThreads];
#pragma unroll
    for (int i = 0; i < turbine_hip::kMoeMaxExperts / kThreads; ++i) {
      const int32_t e = t + i * kThreads;
      add[i] = e < num_local && e >= step ? ends[e - step] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < turbine_hip::kMoeMaxExperts / kThreads; ++i)
      ends[t + i * kThreads] += add[i];
    __syncthreads();
  }
  for (int32_t e = t; e < num_local; e += kThreads) {
    const int32_t first = e == 0 ? 0 : ends[e - 1];
    if (tile >= first && tile < ends[e]) {
      found[0] = e;
      found[1] = offsets[expert_begin + e] + (tile - first) * kTileM;
      found[2] = offsets[expert_begin + e + 1];
    }
  }
  __syncthreads();
  *expert = found[0];
  *begin = found[1];
  *end = found[2];
}

// A block shape: 4 x 2 waves of FragsM x 2 WMMA tiles.
template <int FragsM> struct Shape {
  static constexpr int kWavesM = 4, kWavesN = 2, kFragsM = FragsM,
                       kFragsN = 2;
  static constexpr int kThreads = 32 * kWavesM * kWavesN;
  static constexpr int kTileM = kWavesM * FragsM * 16;
  static constexpr int kTileN = kWavesN * kFragsN * 16;
};

template <int WD, typename S>
__global__ __launch_bounds__(S::kThreads) void moe_quant_grouped_kernel(
    turbine_hip::MoeQuantGrouped p) {
  constexpr bool kF8 = WD == TURBINE_EXPERT_FP8_TENSOR;
  constexpr int kAChunks = kF8 ? 4 : 8;
  constexpr int kARows = S::kThreads / kAChunks;
  constexpr int kPA = S::kTileM / kARows;
  constexpr int kBRows = S::kThreads / 4;
  constexpr int kPB = S::kTileN / kBRows;
  constexpr int kEA = kF8 ? 1 : 2;
  constexpr int kRowBytes = kF8 ? kLds8 : kLds16 * 2;
  static_assert(kPA * kARows == S::kTileM && kPB * kBRows == S::kTileN, "");
  __shared__ __attribute__((aligned(16))) uint8_t a_s[S::kTileM * kRowBytes];
  __shared__ __attribute__((aligned(16))) uint8_t b_s[S::kTileN * kRowBytes];
  __shared__ uint32_t lut[WD == TURBINE_EXPERT_MXFP4 ? 256 : 1];

  const int32_t n_tiles = p.n_total / S::kTileN;
  const int32_t tile_n =
      static_cast<int32_t>(blockIdx.x % static_cast<unsigned>(n_tiles)) *
      S::kTileN;
  int32_t local, begin, end;
  find_tile<S::kThreads, S::kTileM>(
      p.offsets, p.expert_begin, p.num_local,
      static_cast<int32_t>(blockIdx.x / static_cast<unsigned>(n_tiles)), &local,
      &begin, &end);
  if (local < 0)
    return;
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t % 32;
  const int wave = t / 32;
  if (WD == TURBINE_EXPERT_MXFP4) {
    for (int i = t; i < 256; i += S::kThreads) {
      lut[i] = static_cast<uint32_t>(kE2m1Bf16[i & 15]) |
               (static_cast<uint32_t>(kE2m1Bf16[i >> 4]) << 16);
    }
    __syncthreads();
  }
  const int32_t k = p.k;
  const int a_chunk = t % kAChunks;
  const int a_row0 = t / kAChunks;
  const int b_chunk = t % 4;
  const int b_row0 = t / 4;

  // Rows past the expert's last position re-read its first; their results
  // are dropped.
  const uint8_t *a_g[kPA];
#pragma unroll
  for (int pass = 0; pass < kPA; ++pass) {
    int32_t pos = begin + a_row0 + pass * kARows;
    if (pos >= end)
      pos = begin;
    a_g[pass] = p.a + static_cast<int64_t>(pos - p.first) * p.lda * kEA +
                a_chunk * 16;
  }
  const int ti = tile_n >= p.n[0] ? 1 : 0;
  const int32_t tn = tile_n - (ti != 0 ? p.n[0] : 0);
  const uint8_t *b_base = p.w[ti] + local * p.code_stride[ti];
  const uint8_t *s_base = p.s[ti] + local * p.scale_stride[ti];
  const uint8_t *b_g[kPB];
  const uint8_t *s_g[kPB];
#pragma unroll
  for (int pass = 0; pass < kPB; ++pass) {
    const int64_t r = tn + b_row0 + pass * kBRows;
    if (WD == TURBINE_EXPERT_MXFP4) {
      b_g[pass] = b_base + r * (k / 2) + b_chunk * 8;
      s_g[pass] = s_base + r * (k / 32) + b_chunk / 2;
    } else {
      b_g[pass] = b_base + r * k + b_chunk * 16;
      s_g[pass] = s_base + ((r / 128) * (k / 128)) * 4;
    }
  }

  const int wm = wave / S::kWavesN;
  const int wn = wave % S::kWavesN;
  const int frag_row = lane % 16;
  const int h = lane / 16;
  v8f acc[S::kFragsM][S::kFragsN];
#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i)
#pragma unroll
    for (int j = 0; j < S::kFragsN; ++j)
      acc[i][j] = v8f{};

  u32x4 ra[kPA];
  u32x4 rb[kPB];
  uint32_t rs[kPB];
  auto load = [&](int32_t k0) {
#pragma unroll
    for (int pass = 0; pass < kPA; ++pass)
      ra[pass] = *reinterpret_cast<const u32x4 *>(
          a_g[pass] + static_cast<int64_t>(k0) * kEA);
#pragma unroll
    for (int pass = 0; pass < kPB; ++pass) {
      if (WD == TURBINE_EXPERT_MXFP4) {
        const uint2 c = *reinterpret_cast<const uint2 *>(b_g[pass] + k0 / 2);
        rb[pass] = u32x4{c.x, c.y, 0u, 0u};
        rs[pass] = s_g[pass][k0 / 32];
      } else {
        rb[pass] = *reinterpret_cast<const u32x4 *>(b_g[pass] + k0);
        rs[pass] = WD == TURBINE_EXPERT_FP8_BLOCK
                       ? *reinterpret_cast<const uint32_t *>(s_g[pass] +
                                                             (k0 / 128) * 4)
                       : 0u;
      }
    }
  };
  load(0);
  for (int32_t k0 = 0; k0 < k; k0 += kTileK) {
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < kPA; ++pass) {
      *reinterpret_cast<u32x4 *>(
          &a_s[(a_row0 + pass * kARows) * kRowBytes + a_chunk * 16]) = ra[pass];
    }
#pragma unroll
    for (int pass = 0; pass < kPB; ++pass) {
      const int row = b_row0 + pass * kBRows;
      if (kF8) {
        *reinterpret_cast<u32x4 *>(&b_s[row * kRowBytes + b_chunk * 16]) =
            rb[pass];
      } else {
        u32x4 o[2];
        if (WD == TURBINE_EXPERT_FP8_BLOCK) {
          decode_block(rb[pass], __uint_as_float(rs[pass]), o);
        } else {
          o[0] = decode_mx(lut, rb[pass][0], rs[pass]);
          o[1] = decode_mx(lut, rb[pass][1], rs[pass]);
        }
        *reinterpret_cast<u32x4 *>(&b_s[row * kRowBytes + b_chunk * 32]) = o[0];
        *reinterpret_cast<u32x4 *>(&b_s[row * kRowBytes + b_chunk * 32 + 16]) =
            o[1];
      }
    }
    __syncthreads();
    if (k0 + kTileK < k)
      load(k0 + kTileK);
    if (kF8) {
      u32x4 ar[S::kFragsM][2];
      u32x4 br[S::kFragsN][2];
#pragma unroll
      for (int i = 0; i < S::kFragsM; ++i) {
        const uint8_t *src =
            &a_s[((wm * S::kFragsM + i) * 16 + frag_row) * kRowBytes + 32 * h];
        ar[i][0] = *reinterpret_cast<const u32x4 *>(src);
        ar[i][1] = *reinterpret_cast<const u32x4 *>(src + 16);
      }
#pragma unroll
      for (int j = 0; j < S::kFragsN; ++j) {
        const uint8_t *src =
            &b_s[((wn * S::kFragsN + j) * 16 + frag_row) * kRowBytes + 32 * h];
        br[j][0] = *reinterpret_cast<const u32x4 *>(src);
        br[j][1] = *reinterpret_cast<const u32x4 *>(src + 16);
      }
#pragma unroll
      for (int tt = 0; tt < 4; ++tt)
#pragma unroll
        for (int j = 0; j < S::kFragsN; ++j)
#pragma unroll
          for (int i = 0; i < S::kFragsM; ++i)
            acc[i][j] = mma8(slice(ar[i], tt), slice(br[j], tt), acc[i][j]);
    } else {
#pragma unroll
      for (int kk = 0; kk < kTileK; kk += 16) {
        v8s a[S::kFragsM];
#pragma unroll
        for (int i = 0; i < S::kFragsM; ++i) {
          a[i] = *reinterpret_cast<const v8s *>(
              &a_s[(((wm * S::kFragsM + i) * 16 + frag_row) * kLds16 + kk +
                    h * 8) *
                   2]);
        }
#pragma unroll
        for (int j = 0; j < S::kFragsN; ++j) {
          const v8s b = *reinterpret_cast<const v8s *>(
              &b_s[(((wn * S::kFragsN + j) * 16 + frag_row) * kLds16 + kk +
                    h * 8) *
                   2]);
#pragma unroll
          for (int i = 0; i < S::kFragsM; ++i)
            acc[i][j] = mma16(a[i], b, acc[i][j]);
        }
      }
    }
  }

  const float w_scale = kF8 ? *reinterpret_cast<const float *>(s_base) : 1.0f;
#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i) {
#pragma unroll
    for (int j = 0; j < S::kFragsN; ++j) {
      const int32_t n = tile_n + (wn * S::kFragsN + j) * 16 + frag_row;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int32_t pos = begin + (wm * S::kFragsM + i) * 16 + h * 8 + v;
        if (pos >= end)
          continue;
        float value = acc[i][j][v];
        if (kF8) {
          const float a_scale =
              p.a_scale[p.a_scale_vec != 0 ? pos - p.first : local];
          value = value * w_scale;
          value = value * a_scale;
        }
        static_cast<Bf16 *>(p.out)[static_cast<int64_t>(pos - p.first) * p.ldc +
                                   n] = f32_to_bf16(value);
      }
    }
  }
}

template <int WD, typename S>
int32_t launch(turbine_ctx *ctx, const turbine_hip::MoeQuantGrouped &p,
               int64_t rows) {
  const int64_t row_tiles = (rows + S::kTileM - 1) / S::kTileM + p.num_local;
  const int64_t n_tiles = p.n_total / S::kTileN;
  hipLaunchKernelGGL((moe_quant_grouped_kernel<WD, S>),
                     dim3(static_cast<unsigned>(row_tiles * n_tiles)),
                     dim3(S::kThreads), 0, ctx->stream, p);
  return check_hip(ctx, hipGetLastError(), "moe quant grouped launch");
}

} // namespace

namespace turbine_hip {

int32_t launch_moe_quant_grouped(turbine_ctx *ctx, int32_t weight_dtype,
                                 const MoeQuantGrouped &p, int64_t rows,
                                 bool wide) {
  if (rows == 0)
    return TURBINE_OK;
  switch (weight_dtype) {
  case TURBINE_EXPERT_FP8_BLOCK:
    return wide ? launch<TURBINE_EXPERT_FP8_BLOCK, Shape<2>>(ctx, p, rows)
                : launch<TURBINE_EXPERT_FP8_BLOCK, Shape<1>>(ctx, p, rows);
  case TURBINE_EXPERT_FP8_TENSOR:
    return wide ? launch<TURBINE_EXPERT_FP8_TENSOR, Shape<2>>(ctx, p, rows)
                : launch<TURBINE_EXPERT_FP8_TENSOR, Shape<1>>(ctx, p, rows);
  default:
    return wide ? launch<TURBINE_EXPERT_MXFP4, Shape<2>>(ctx, p, rows)
                : launch<TURBINE_EXPERT_MXFP4, Shape<1>>(ctx, p, rows);
  }
}

} // namespace turbine_hip
