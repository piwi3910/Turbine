// The MXFP4 weight GEMM (ABI v2.9 turbine_qgemm, scheme MXFP4) on ROCm:
// `turbine_hip_mxfp4`, an own kernel per the decision "P6: MXFP4 GEMM —
// provider evaluation (kernel reuse rule)" (CK's BF16 x pk_fp4 BQuant pipeline
// is correct on gfx1201 only with the WMMA warp tile, for m a multiple of 128,
// and ~10x slower than BF16 hipBLASLt; CK's MX GEMM is gfx950-only; llama.cpp's
// MXFP4 kernels quantize the activations to 8 bits; hipBLASLt has no MX
// scaling on gfx1201).
//
//   c[m, n] = alpha * a[m, k] . dequant(b)[n, k]^T, F32 accumulation, one
//   rounding to c's dtype;
//   dequant(b)[r, j] = e2m1(code(r, j)) * 2^(e8m0(r, j / 32) - 127).
//
// Every E2M1 value is a BF16 value, and so is every E2M1 value times a power
// of two inside BF16's range, exactly. The decode kernels (up to 64 rows)
// multiply the activations by the unscaled E2M1 values on the RDNA4 matrix
// cores (v_wmma_f32_16x16x16_bf16, F32 accumulation), one MX group of 32 k
// values (two WMMAs) at a time into a zero accumulator, and add that group sum
// times the group's power-of-two scale (exact) to the running F32 sum: one
// decode per weight, no scaling of it. The prefill kernel, where one decoded
// weight feeds 8 row fragments, decodes the scaled weight (the exponent added
// in the BF16 bits) and chains the WMMAs into one accumulator. Either way the
// result differs from the CPU reference (cpu::qgemm) only by the F32
// summation order.
//
// Layout of the work. The k axis is walked in steps of 64 (two MX groups).
// In a step, lane l of a wave (r = l % 16, h = l / 16) loads the 16 code bytes
// of group h of weight row (column of c) r and the row's two exponents; the
// two half-waves then swap half their codes (a lane exchange with lane l ^ 16),
// so that lane l holds k = 8h .. 8h + 7 and 16 + 8h .. 16 + 8h + 7 of both
// groups: the gfx12 wave32 B operand of the group's two 16-deep WMMA slices
// (lane l holds A[l % 16][8 (l / 16) .. + 8), B[8 (l / 16) .. + 8)[l % 16] and
// C[8 (l / 16) + v][l % 16]). Codes become BF16 through a 256-entry LDS table
// (one byte, two codes, one 32-bit read). The activations of each row
// fragment are read from memory in the same lane layout.
//
// A block holds NW column waves (FN column fragments each) times KS k-split
// wave groups: each group walks a contiguous range of steps and group 0 adds
// the others' partial sums in group order through LDS, so the result is
// deterministic (but depends on the tile, i.e. on m). Rows past m re-read row
// m - 1 and columns past n re-read column n - 1; their results are dropped.
#include <algorithm>
#include <string>

#include "bf16.hpp"
#include "qgemm_mxfp4.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::f32_to_bf16;
using turbine_hip::fail;

namespace {

constexpr int kLanes = 32;
constexpr int kMma = 16;
// k per step: two MX groups of 32.
constexpr int kStep = 64;

// The tiles, from a sweep on the R9700 (GPU 0, the Llama-3.2-3B and
// Llama-3.1-8B shapes; decision "P6: MXFP4 GEMM — provider evaluation (kernel
// reuse rule)"): up to 16 rows one wave per 16 columns with k split over 8
// waves (the weights of 1-16 rows stream at 400-500 GB/s); up to 32 and 64 rows
// 2 and 4 row fragments with k split over 4; above, the LDS kernel with 128 x
// 128 block tiles.
constexpr int kSmallMaxRows = 16;
constexpr int kMediumMaxRows = 64;

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ inline v8f mma(v8s a, v8s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// 2^(e - 127) as F32 (2^-127 is a subnormal; 255 is NaN).
__device__ inline float e8m0_scale(uint32_t e) {
  if (e == 0)
    return __uint_as_float(0x00400000u);
  if (e == 255)
    return __uint_as_float(0x7fc00000u);
  return __uint_as_float(e << 23);
}

// BF16 bits of the 16 E2M1 codes (bit 3 = sign; code 8 is -0).
__constant__ const uint16_t kE2m1Bf16[16] = {
    0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
    0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0};

// Eight BF16 weights from four code bytes (low nibble = the even element),
// through the byte table.
__device__ inline v8s decode8(const uint32_t *lut, uint32_t word) {
  const u32x4 v{lut[word & 0xffu], lut[(word >> 8) & 0xffu],
                lut[(word >> 16) & 0xffu], lut[word >> 24]};
  return __builtin_bit_cast(v8s, v);
}

// Eight BF16 weights times 2^(e - 127) (e the group's E8M0 exponent): the
// table's unscaled values with e - 127 added to the exponent field of every
// nonzero one, which is exact while the result stays a normal number (every
// E2M1 magnitude has exponent field 126 .. 129, so for 2 <= e <= 252); other
// exponents take the F32 product (NaN for 255).
__device__ inline v8s decode8_scaled(const uint32_t *lut, uint32_t word,
                                     uint32_t e) {
  u32x4 v{lut[word & 0xffu], lut[(word >> 8) & 0xffu],
          lut[(word >> 16) & 0xffu], lut[word >> 24]};
  if (e >= 2u && e <= 252u) {
    const uint32_t delta = static_cast<uint32_t>(static_cast<int32_t>(e) - 127)
                           << 7;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Bit 15 / 31 set for each nonzero half (no carry leaves a half).
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
  return __builtin_bit_cast(v8s, v);
}

struct Params {
  const Bf16 *a;
  const uint8_t *b;
  const uint8_t *b_scales;
  void *c;
  int32_t m, n, k;
  int64_t lda, ldc;
  float alpha;
  bool c_f32;
};

// FM row fragments and FN column fragments per wave (all waves of a block
// share the rows), NW column waves, KS k-split groups, U steps loaded per
// batch.
template <int FM, int FN, int NW, int KS, int U>
__global__ __launch_bounds__(kLanes * NW *
                             KS) void mxfp4_gemm_kernel(Params p) {
  __shared__ uint32_t lut[256];
  const int t = static_cast<int>(threadIdx.x);
  for (int i = t; i < 256; i += kLanes * NW * KS) {
    lut[i] = static_cast<uint32_t>(kE2m1Bf16[i & 15]) |
             (static_cast<uint32_t>(kE2m1Bf16[i >> 4]) << 16);
  }
  const int lane = t % kLanes;
  const int wave = t / kLanes;
  const int group = wave / NW;
  const int nw = wave % NW;
  const int r = lane % kMma;
  const int h = lane / kMma;

  const int32_t col0 =
      static_cast<int32_t>(blockIdx.x) * NW * FN * kMma + nw * FN * kMma;
  const int32_t row0 = static_cast<int32_t>(blockIdx.y) * FM * kMma;
  const uint8_t *brow[FN];
  const uint16_t *srow[FN];
#pragma unroll
  for (int j = 0; j < FN; ++j) {
    const int64_t wcol = min(col0 + j * kMma + r, p.n - 1);
    brow[j] = p.b + wcol * (p.k / 2) + h * 16;
    srow[j] =
        reinterpret_cast<const uint16_t *>(p.b_scales + wcol * (p.k / 32));
  }
  const Bf16 *arow[FM];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
    const int64_t row = min(row0 + i * kMma + r, p.m - 1);
    arow[i] = p.a + row * p.lda + h * 8;
  }

  const int32_t steps = p.k / kStep;
  const int32_t s0 = group * steps / KS;
  const int32_t s1 = (group + 1) * steps / KS;

  v8f acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j)
      acc[i][j] = v8f{};
  }
  __syncthreads();

  for (int32_t st = s0; st < s1; st += U) {
    u32x4 bq[U][FN];
    uint32_t se[U][FN];
    v8s aq[U][FM][4];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int32_t s = min(st + u, s1 - 1);
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        bq[u][j] = *reinterpret_cast<const u32x4 *>(brow[j] + s * 32);
        se[u][j] = srow[j][s];
      }
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          aq[u][i][q] =
              *reinterpret_cast<const v8s *>(arow[i] + s * kStep + q * 16);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (st + u >= s1)
        break;
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        // Lane h holds group h; after the swap, words[g][s'] are lane l's
        // codes of group g for slice s' (k = 16 s' + 8 h .. + 8).
        const u32x4 q = bq[u][j];
        const uint32_t send0 = h == 0 ? q[1] : q[0];
        const uint32_t send1 = h == 0 ? q[3] : q[2];
        const uint32_t recv0 = __shfl_xor(send0, kMma, kLanes);
        const uint32_t recv1 = __shfl_xor(send1, kMma, kLanes);
        const uint32_t own0 = h == 0 ? q[0] : q[1];
        const uint32_t own1 = h == 0 ? q[2] : q[3];
        const uint32_t words[2][2] = {
            {h == 0 ? own0 : recv0, h == 0 ? own1 : recv1},
            {h == 0 ? recv0 : own0, h == 0 ? recv1 : own1}};
#pragma unroll
        for (int g = 0; g < 2; ++g) {
          const float scale = e8m0_scale((se[u][j] >> (8 * g)) & 0xffu);
          const v8s b0 = decode8(lut, words[g][0]);
          const v8s b1 = decode8(lut, words[g][1]);
#pragma unroll
          for (int i = 0; i < FM; ++i) {
            v8f sum = mma(aq[u][i][2 * g], b0, v8f{});
            sum = mma(aq[u][i][2 * g + 1], b1, sum);
            acc[i][j] += sum * scale;
          }
        }
      }
    }
  }

  if constexpr (KS > 1) {
    __shared__ float part[(KS - 1) * NW * FM * FN * 8 * kLanes];
    constexpr int kFrags = FM * FN;
    if (group > 0) {
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            part[((((group - 1) * NW + nw) * kFrags + i * FN + j) * 8 + v) *
                     kLanes +
                 lane] = acc[i][j][v];
          }
        }
      }
    }
    __syncthreads();
    if (group > 0)
      return;
    for (int g = 1; g < KS; ++g) {
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            acc[i][j][v] +=
                part[((((g - 1) * NW + nw) * kFrags + i * FN + j) * 8 + v) *
                         kLanes +
                     lane];
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < FN; ++j) {
    const int32_t col = col0 + j * kMma + r;
    if (col >= p.n)
      continue;
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int32_t row = row0 + i * kMma + h * 8 + v;
        if (row >= p.m)
          continue;
        const float value = p.alpha * acc[i][j][v];
        const int64_t at = static_cast<int64_t>(row) * p.ldc + col;
        if (p.c_f32)
          static_cast<float *>(p.c)[at] = value;
        else
          static_cast<Bf16 *>(p.c)[at] = f32_to_bf16(value);
      }
    }
  }
}

// The prefill kernel: the same per-group arithmetic, with the block's
// activation tile (FM * 16 rows x one 64-deep step) staged in LDS, double
// buffered, and shared by the NW column waves (FN fragments each); the weights
// go from memory straight into each wave's registers one step ahead.
template <int FM, int FN, int NW>
__global__ __launch_bounds__(kLanes *NW) void mxfp4_gemm_lds_kernel(Params p) {
  constexpr int kThreads = kLanes * NW;
  constexpr int kTileM = FM * kMma;
  // LDS row: one step of BF16 plus 8 of padding (144 bytes), so the 16 rows a
  // WMMA operand read touches fall on distinct banks.
  constexpr int kRow = kStep + 8;
  constexpr int kChunks = kStep / 8; // 16-byte chunks per tile row
  constexpr int kRowsPerPass = kThreads / kChunks;
  constexpr int kPasses = (kTileM + kRowsPerPass - 1) / kRowsPerPass;
  __shared__ uint32_t lut[256];
  __shared__ __attribute__((aligned(16))) Bf16 a_s[2][kTileM * kRow];
  const int t = static_cast<int>(threadIdx.x);
  for (int i = t; i < 256; i += kThreads) {
    lut[i] = static_cast<uint32_t>(kE2m1Bf16[i & 15]) |
             (static_cast<uint32_t>(kE2m1Bf16[i >> 4]) << 16);
  }
  const int lane = t % kLanes;
  const int nw = t / kLanes;
  const int r = lane % kMma;
  const int h = lane / kMma;

  const int32_t col0 =
      static_cast<int32_t>(blockIdx.x) * NW * FN * kMma + nw * FN * kMma;
  const int32_t row0 = static_cast<int32_t>(blockIdx.y) * kTileM;
  const uint8_t *brow[FN];
  const uint16_t *srow[FN];
#pragma unroll
  for (int j = 0; j < FN; ++j) {
    const int64_t wcol = min(col0 + j * kMma + r, p.n - 1);
    brow[j] = p.b + wcol * (p.k / 2) + h * 16;
    srow[j] =
        reinterpret_cast<const uint16_t *>(p.b_scales + wcol * (p.k / 32));
  }
  // A chunks: tile rows t / kChunks + pass * kRowsPerPass, 16 bytes at column
  // (t % kChunks) * 8 of the step.
  const int chunk = t % kChunks;
  const int lrow0 = t / kChunks;
  const Bf16 *a_g[kPasses];
#pragma unroll
  for (int pass = 0; pass < kPasses; ++pass) {
    const int64_t row =
        min(row0 + min(lrow0 + pass * kRowsPerPass, kTileM - 1), p.m - 1);
    a_g[pass] = p.a + row * p.lda + chunk * 8;
  }
  u32x4 ra[kPasses];
  auto load_a = [&](int32_t s) {
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass)
      ra[pass] = *reinterpret_cast<const u32x4 *>(a_g[pass] + s * kStep);
  };
  auto store_a = [&](int stage) {
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int lr = lrow0 + pass * kRowsPerPass;
      if (lr < kTileM) {
        *reinterpret_cast<u32x4 *>(&a_s[stage][lr * kRow + chunk * 8]) =
            ra[pass];
      }
    }
  };
  u32x4 bq[2][FN];
  uint32_t se[2][FN];
  auto load_b = [&](int buf, int32_t s) {
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      bq[buf][j] = *reinterpret_cast<const u32x4 *>(brow[j] + s * 32);
      se[buf][j] = srow[j][s];
    }
  };

  v8f acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j)
      acc[i][j] = v8f{};
  }
  const int32_t steps = p.k / kStep;
  load_a(0);
  load_b(0, 0);
  store_a(0);
  __syncthreads();
  for (int32_t s0 = 0; s0 < steps; s0 += 2) {
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int32_t s = s0 + st;
      if (s >= steps)
        break;
      const bool more = s + 1 < steps;
      if (more) {
        load_a(s + 1);
        load_b(st ^ 1, s + 1);
      }
      const Bf16 *as = a_s[st];
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const u32x4 q = bq[st][j];
        const uint32_t send0 = h == 0 ? q[1] : q[0];
        const uint32_t send1 = h == 0 ? q[3] : q[2];
        const uint32_t recv0 = __shfl_xor(send0, kMma, kLanes);
        const uint32_t recv1 = __shfl_xor(send1, kMma, kLanes);
        const uint32_t own0 = h == 0 ? q[0] : q[1];
        const uint32_t own1 = h == 0 ? q[2] : q[3];
        const uint32_t words[2][2] = {
            {h == 0 ? own0 : recv0, h == 0 ? own1 : recv1},
            {h == 0 ? recv0 : own0, h == 0 ? recv1 : own1}};
#pragma unroll
        for (int g = 0; g < 2; ++g) {
          // The decoded weights carry their group scale: the WMMAs chain into
          // the accumulator (the decode is shared by the FM row fragments).
          const uint32_t e = (se[st][j] >> (8 * g)) & 0xffu;
          const v8s b0 = decode8_scaled(lut, words[g][0], e);
          const v8s b1 = decode8_scaled(lut, words[g][1], e);
#pragma unroll
          for (int i = 0; i < FM; ++i) {
            const Bf16 *arow = as + (i * kMma + r) * kRow + 32 * g + 8 * h;
            const v8s a0 = *reinterpret_cast<const v8s *>(arow);
            const v8s a1 = *reinterpret_cast<const v8s *>(arow + 16);
            acc[i][j] = mma(a0, b0, acc[i][j]);
            acc[i][j] = mma(a1, b1, acc[i][j]);
          }
        }
      }
      if (more)
        store_a(st ^ 1);
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < FN; ++j) {
    const int32_t col = col0 + j * kMma + r;
    if (col >= p.n)
      continue;
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int32_t row = row0 + i * kMma + h * 8 + v;
        if (row >= p.m)
          continue;
        const float value = p.alpha * acc[i][j][v];
        const int64_t at = static_cast<int64_t>(row) * p.ldc + col;
        if (p.c_f32)
          static_cast<float *>(p.c)[at] = value;
        else
          static_cast<Bf16 *>(p.c)[at] = f32_to_bf16(value);
      }
    }
  }
}

template <int FM, int FN, int NW>
hipError_t launch_lds(hipStream_t stream, const Params &p) {
  constexpr int kCols = NW * FN * kMma;
  const dim3 grid(static_cast<unsigned>((p.n + kCols - 1) / kCols),
                  static_cast<unsigned>((p.m + FM * kMma - 1) / (FM * kMma)));
  hipLaunchKernelGGL((mxfp4_gemm_lds_kernel<FM, FN, NW>), grid,
                     dim3(kLanes * NW), 0, stream, p);
  return hipGetLastError();
}

template <int FM, int FN, int NW, int KS, int U>
hipError_t launch(hipStream_t stream, const Params &p) {
  constexpr int kCols = NW * FN * kMma;
  const dim3 grid(static_cast<unsigned>((p.n + kCols - 1) / kCols),
                  static_cast<unsigned>((p.m + FM * kMma - 1) / (FM * kMma)));
  hipLaunchKernelGGL((mxfp4_gemm_kernel<FM, FN, NW, KS, U>), grid,
                     dim3(kLanes * NW * KS), 0, stream, p);
  return hipGetLastError();
}

bool supported(const turbine_qgemm_desc *d) {
  if (d == nullptr || d->scheme != TURBINE_QSCHEME_MXFP4)
    return false;
  if (d->act_quant != TURBINE_ACTQ_NONE &&
      d->act_quant != TURBINE_ACTQ_MXFP4_EMULATED) {
    return false;
  }
  if (d->a_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32)
    return false;
  if (d->group_size != 32 && d->group_size != 0)
    return false;
  if (d->m < 0 || d->n <= 0 || d->k <= 0 || d->k % kStep != 0)
    return false;
  if (d->m > INT32_MAX || d->n > INT32_MAX || d->k > INT32_MAX)
    return false;
  // 16-byte activation loads: rows start on 16 bytes.
  if (d->lda < d->k || d->lda % 8 != 0 || d->ldc < d->n)
    return false;
  return true;
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace

namespace turbine_hip {

bool qgemm_mxfp4_supports(const turbine_qgemm_desc *d) { return supported(d); }

hipError_t launch_qgemm_mxfp4(hipStream_t stream, const turbine_qgemm_desc *d,
                              Mxfp4Tile tile) {
  const Params p{static_cast<const Bf16 *>(d->a),
                 static_cast<const uint8_t *>(d->b),
                 static_cast<const uint8_t *>(d->b_scales),
                 d->c,
                 static_cast<int32_t>(d->m),
                 static_cast<int32_t>(d->n),
                 static_cast<int32_t>(d->k),
                 d->lda,
                 d->ldc,
                 d->alpha,
                 d->c_dtype == TURBINE_DTYPE_F32};
  if (p.m == 0)
    return hipSuccess;
  if (tile == Mxfp4Tile::Auto) {
    // A prefill step always runs the Large tile: one reduction order per row
    // whatever m is (no k split; a WMMA element depends only on its row of A,
    // its column of B and its accumulator), so a prompt prefilled whole, in
    // chunks or behind a cached prefix gets the same bits (Phase 4 prefix
    // reuse). Decode steps pick the fastest tile for m.
    tile = d->prefill != 0         ? Mxfp4Tile::Large
           : p.m <= kSmallMaxRows  ? Mxfp4Tile::Small
           : p.m <= kMediumMaxRows ? Mxfp4Tile::Medium
                                   : Mxfp4Tile::Large;
  }
  switch (tile) {
  case Mxfp4Tile::Small:
    return launch<1, 1, 1, 8, 4>(stream, p);
  case Mxfp4Tile::Medium:
    return p.m <= 32 ? launch<2, 1, 2, 4, 2>(stream, p)
                     : launch<4, 1, 2, 4, 1>(stream, p);
  default:
    return launch_lds<8, 1, 8>(stream, p);
  }
}

int32_t qgemm_mxfp4_run(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: descriptor is NULL");
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_qgemm (turbine_hip_mxfp4): unsupported configuration "
                "scheme=" +
                    std::to_string(d->scheme) +
                    " act_quant=" + std::to_string(d->act_quant) +
                    " dtypes=" + std::to_string(d->a_dtype) + "," +
                    std::to_string(d->c_dtype) + " m=" + std::to_string(d->m) +
                    " n=" + std::to_string(d->n) + " k=" +
                    std::to_string(d->k) + " lda=" + std::to_string(d->lda));
  }
  if (d->m == 0)
    return TURBINE_OK;
  if (d->a == nullptr || d->b == nullptr || d->b_scales == nullptr ||
      d->c == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: NULL operand");
  }
  if (!aligned16(d->a) || !aligned16(d->b)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_qgemm (turbine_hip_mxfp4): a and b must start on 16 "
                "bytes");
  }
  const int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, launch_qgemm_mxfp4(ctx->stream, d, Mxfp4Tile::Auto),
                   "turbine_qgemm (turbine_hip_mxfp4) launch");
}

} // namespace turbine_hip
