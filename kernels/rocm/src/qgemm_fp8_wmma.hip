// FP8 W8A8 decode GEMM on RDNA4: turbine_hip_fp8_wmma, an own small-m kernel
// for the ABI v2.9 schemes FP8_TENSOR / FP8_CHANNEL weights x FP8_TENSOR /
// FP8_TOKEN activations (the e4m3 codes turbine_quantize_act writes), BF16 out
// (decision "5p Task 14: FP8 c1 decode — provider evaluation (kernel reuse
// rule)": at m = 1 hipBLASLt's best FP8 solutions read the weights at
// 420-480 GB/s against ~600 for the BF16 decode GEMVs; CK's FP8 gemm_quant is
// wrong on gfx1201, llama.cpp has no FP8 type, vLLM's skinny FP8 GEMM is
// MI300 assembly).
//
// Decode steps of at most kMaxRows rows run the kernel below; every other call
// (prefill steps, whose rows must equal the invariant prefill solution's bit
// for bit, and larger decode batches) runs hipblaslt_fp8 (qgemm.cpp)
// unchanged, so the entry serves every call its supports() accepts. The card
// profile puts it first in the small decode row tier only.
//
// The kernel: one wave owns a 16 x 16 output tile (16 rows of a, 16 weight
// rows); the kWaves waves of a block split k into whole 128-deep groups (a rule
// of k only) and add their partial sums in LDS in wave order. Per 64-deep step
// every lane streams 32 weight codes and 32 activation codes straight into the
// v_wmma_f32_16x16x16_fp8_fp8 operand registers (lane half h owns k [32h,
// 32h + 32), slice t takes k 32h + 8t .. + 8 on both operands: any k order
// shared by the two operands sums the same products). e4m3 x e4m3 products are
// exact in F32, so the result differs from hipBLASLt's and from the CPU
// reference by the F32 summation order only; the scales are applied after the
// sum as the prefill epilogue applies them (qgemm_epilogue.hpp): acc * w_scale
// * a_scale * alpha, rounded to BF16 once. A row's result depends only on its
// row of a (rows past m re-read the last row and are dropped), so rows are
// independent of m within the kernel's tier.
//
// Unquantized activations (ABI v2.13, decision "5p Task 14", item 1 A): a
// single-row decode call (m = 1, the c1 step) with a_dtype BF16 and an
// FP8_TENSOR / FP8_TOKEN act_quant runs bf16_kernel below, which quantizes the
// activation row itself, so the decoder's separate turbine_quantize_act launch
// (and its boundary) goes away. Every block first quantizes the row into LDS
// exactly as quantize_act.hip does (qgemm_quantize.hpp: the row's amax -- a
// max, exact in any order -- then max(amax / 448, 1 / (448 * 512)), or the
// static scale; each value divided by its scale in F32 and rounded by
// fp8_e4m3_round), then runs wmma_kernel's loop on the codes in LDS and its
// epilogue with that scale. Codes, scale, summation order and epilogue are
// those of quantize_act + wmma_kernel, so c is bit-identical to the two-launch
// path (hip_qgemm qgemm_fused_act_quant_matches_two_launches). The
// quantization is repeated per block, so a block serves several column tiles
// (tiles_for) and only m = 1 takes this form: at m = 2 the repeated work
// already costs more than the separate launch saves.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "qgemm_impls.hpp"
#include "qgemm_problem.hpp"
#include "qgemm_quantize.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

namespace fp8_wmma {

constexpr int kLanes = 32;
constexpr int kMma = 16;
// k depth of one step (two lane halves of 32) and of one wave-split group.
constexpr int kStep = 64;
constexpr int kGroup = 128;
constexpr int kStepsPerGroup = kGroup / kStep;

typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ inline v8f mma(v2i a, v2i b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// Words 2t, 2t + 1 of a lane's 32 codes (two u32x4) as one WMMA operand.
__device__ inline v2i slice(const u32x4 (&w)[2], int t) {
  v2i out;
  out[0] = static_cast<int>(w[t / 2][(t % 2) * 2]);
  out[1] = static_cast<int>(w[t / 2][(t % 2) * 2 + 1]);
  return out;
}

__device__ inline uint16_t to_bf16(float v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(b));
  if ((b & 0x7fffffffu) > 0x7f800000u)
    return 0x7fc0; // canonical NaN, as qgemm_epilogue.hpp
  return static_cast<uint16_t>((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

// Grid: (column tiles of 16, row tiles of 16). k is a multiple of kGroup.
template <int kWaves>
__global__ __launch_bounds__(kWaves *kLanes) void wmma_kernel(
    const uint8_t *a, const float *a_scales, int32_t a_vector,
    const uint8_t *b, const float *w_scales, int32_t w_vector, uint16_t *c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc, float alpha) {
  __shared__ float partial[kWaves > 1 ? kWaves - 1 : 1][8][kLanes];
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int wave = static_cast<int>(threadIdx.x) / kLanes;
  const int r = lane % kMma;
  const int h = lane / kMma;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kMma;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kMma;
  const int64_t groups = k / kGroup;
  const int64_t g_begin = groups * wave / kWaves;
  const int64_t g_end = groups * (wave + 1) / kWaves;

  const uint8_t *a_row = a + std::min(m0 + r, m - 1) * lda + 32 * h;
  const uint8_t *b_row = b + std::min(n0 + r, n - 1) * k + 32 * h;

  v8f total = v8f{};
  u32x4 wb[kStepsPerGroup][2];
  u32x4 wa[kStepsPerGroup][2];
  auto load = [&](int64_t g) {
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st) {
      const int64_t k0 = g * kGroup + st * kStep;
      wb[st][0] = *reinterpret_cast<const u32x4 *>(b_row + k0);
      wb[st][1] = *reinterpret_cast<const u32x4 *>(b_row + k0 + 16);
      wa[st][0] = *reinterpret_cast<const u32x4 *>(a_row + k0);
      wa[st][1] = *reinterpret_cast<const u32x4 *>(a_row + k0 + 16);
    }
  };
  if (g_begin < g_end)
    load(g_begin);
  for (int64_t g = g_begin; g < g_end; ++g) {
    u32x4 cb[kStepsPerGroup][2];
    u32x4 ca[kStepsPerGroup][2];
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st) {
      cb[st][0] = wb[st][0];
      cb[st][1] = wb[st][1];
      ca[st][0] = wa[st][0];
      ca[st][1] = wa[st][1];
    }
    if (g + 1 < g_end)
      load(g + 1);
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st)
#pragma unroll
      for (int t = 0; t < 4; ++t)
        total = mma(slice(ca[st], t), slice(cb[st], t), total);
  }

  if (kWaves > 1) {
    if (wave > 0) {
#pragma unroll
      for (int v = 0; v < 8; ++v)
        partial[wave - 1][v][lane] = total[v];
    }
    __syncthreads();
    if (wave > 0)
      return;
#pragma unroll
    for (int w = 1; w < kWaves; ++w)
#pragma unroll
      for (int v = 0; v < 8; ++v)
        total[v] += partial[w - 1][v][lane];
  }

  const int64_t cn = n0 + r;
  if (cn >= n)
    return;
  const float ws = w_scales[w_vector != 0 ? cn : 0];
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int64_t row = m0 + 8 * h + v;
    if (row >= m)
      continue;
    float value = total[v] * ws;
    value = value * a_scales[a_vector != 0 ? row : 0];
    value = value * alpha;
    c[row * ldc + cn] = to_bf16(value);
  }
}

// Waves per block by the number of 128-deep groups (hip_qgemm
// qgemm_decode_timings, 1-16 waves at m = 1 and 16): k = 3072 (24 groups) is
// flat from 2 to 16 waves (qkv 28.7-29.1 us, o 18.8-19.2, gate_up 83.4-85.7);
// k = 8192 (64 groups) is not monotonic -- 1/2/4/8/16 waves 54.8 / 69.2 /
// 66.5 / 89.4 / 43.4 us -- and wants 16.
int waves_for(int64_t groups) {
  return groups >= 64 ? 16 : groups >= 32 ? 4 : 2;
}

template <int kWaves>
hipError_t launch(const turbine_qgemm_desc *d, hipStream_t stream) {
  const dim3 grid(static_cast<uint32_t>((d->n + kMma - 1) / kMma),
                  static_cast<uint32_t>((d->m + kMma - 1) / kMma));
  hipLaunchKernelGGL(
      (wmma_kernel<kWaves>), grid, dim3(kWaves * kLanes), 0, stream,
      static_cast<const uint8_t *>(d->a), d->a_scales,
      act_qscale(d->act_quant) == QScale::Vector ? 1 : 0,
      static_cast<const uint8_t *>(d->b), static_cast<const float *>(d->b_scales),
      weight_qscale(d->scheme) == QScale::Vector ? 1 : 0,
      static_cast<uint16_t *>(d->c), d->m, d->n, d->k, d->lda, d->ldc,
      d->alpha);
  return hipGetLastError();
}

// e4m3 code of the BF16 value in bits [16 * half, 16 * half + 16) of word
// w divided by scale (quantize_fp8_rows: load_act<true>(x, i) / scale).
__device__ inline uint32_t code_of(uint32_t w, int half, float scale) {
  const uint32_t bits = half != 0 ? (w & 0xffff0000u) : (w << 16);
  float v;
  __builtin_memcpy(&v, &bits, sizeof(v));
  return fp8_e4m3_round(v / scale);
}

// The unquantized-activation form runs one activation row only: every block
// quantizes the whole row (its 16 output columns need all of k), so the work
// grows with m times the number of blocks; at m = 2 it already loses to the
// two launches (hip_qgemm qgemm_decode_timings).
constexpr int64_t kBf16Rows = 1;
// 8-column chunks of the row each thread holds in registers between the amax
// and the quantization pass (k <= threads * 8 * kChunks).
constexpr int kChunks = 4;

// Column tiles per block of the BF16 form by waves per tile: more tiles share
// one quantization of the row (256 threads at 2 and 4 waves, 512 at 16).
constexpr int tiles_for(int waves) { return waves >= 16 ? 1 : waves >= 4 ? 2 : 4; }

// The BF16 form of wmma_kernel for one row: kTiles column tiles of 16 per
// block, each split over kWaves waves exactly as wmma_kernel splits it (k
// groups by wave, partials added in wave order), on the codes of the row
// quantized into LDS by the whole block first. a: k BF16 values (16-byte
// aligned), k a multiple of kGroup and at most kTiles * kWaves * 32 * 8 *
// kChunks. Grid: (ceil(n / (16 * kTiles)), 1).
template <int kWaves, int kTiles, bool kToken>
__global__ __launch_bounds__(kWaves *kTiles *kLanes) void bf16_kernel(
    const uint16_t *a, float static_scale, const uint8_t *b,
    const float *w_scales, int32_t w_vector, uint16_t *c, int64_t n, int64_t k,
    float alpha) {
  constexpr int kThreads = kWaves * kTiles * kLanes;
  extern __shared__ __attribute__((aligned(16))) uint8_t stage[];
  __shared__ float partial[kTiles][kWaves > 1 ? kWaves - 1 : 1][kLanes];
  __shared__ uint32_t amax_bits;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kLanes;
  const int wave = tid / kLanes;
  const int tile = wave / kWaves;
  const int kw = wave % kWaves;
  const int r = lane % kMma;
  const int h = lane / kMma;
  const int64_t n0 =
      (static_cast<int64_t>(blockIdx.x) * kTiles + tile) * kMma;
  const int64_t groups = k / kGroup;
  const int64_t g_begin = groups * kw / kWaves;
  const int64_t g_end = groups * (kw + 1) / kWaves;

  // The weights of the wave's first group go out first: their latency
  // overlaps the quantization of the row.
  const uint8_t *b_row = b + std::min(n0 + r, n - 1) * k + 32 * h;
  u32x4 wb[kStepsPerGroup][2];
  auto load_b = [&](int64_t g) {
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st) {
      const int64_t k0 = g * kGroup + st * kStep;
      wb[st][0] = *reinterpret_cast<const u32x4 *>(b_row + k0);
      wb[st][1] = *reinterpret_cast<const u32x4 *>(b_row + k0 + 16);
    }
  };
  if (g_begin < g_end)
    load_b(g_begin);

  // 1. The row, held in registers, and its scale: the static one, or the
  // dynamic rule on the row's amax (F32 bits of magnitudes order as their
  // values; a NaN magnitude is skipped, as fmaxf skips it in
  // quantize_fp8_rows).
  u32x4 x[kChunks];
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int64_t i = (static_cast<int64_t>(j) * kThreads + tid) * 8;
    if (i < k)
      x[j] = *reinterpret_cast<const u32x4 *>(a + i);
  }
  float scale = static_scale;
  if constexpr (kToken) {
    if (tid == 0)
      amax_bits = 0u;
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < kChunks; ++j) {
      const int64_t i = (static_cast<int64_t>(j) * kThreads + tid) * 8;
      if (i >= k)
        continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const uint32_t w = x[j][e / 2];
        const uint32_t bits =
            (e % 2 != 0 ? (w & 0xffff0000u) : (w << 16)) & 0x7fffffffu;
        float f;
        __builtin_memcpy(&f, &bits, sizeof(f));
        mx = fmaxf(mx, f);
      }
    }
#pragma unroll
    for (int off = kLanes / 2; off > 0; off /= 2)
      mx = fmaxf(mx, __shfl_xor(mx, off, kLanes));
    __syncthreads();
    if (lane == 0) {
      uint32_t mb;
      __builtin_memcpy(&mb, &mx, sizeof(mb));
      atomicMax(&amax_bits, mb);
    }
    __syncthreads();
    float amax;
    const uint32_t ab = amax_bits;
    __builtin_memcpy(&amax, &ab, sizeof(amax));
    scale = dynamic_fp8_scale(amax);
  }

  // 2. The row's codes into the stage.
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int64_t i = (static_cast<int64_t>(j) * kThreads + tid) * 8;
    if (i >= k)
      continue;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      lo |= code_of(x[j][e / 2], e % 2, scale) << (8 * e);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      hi |= code_of(x[j][2 + e / 2], e % 2, scale) << (8 * e);
    *reinterpret_cast<uint2 *>(stage + i) = make_uint2(lo, hi);
  }
  __syncthreads();

  // 3. wmma_kernel's loop on the staged codes (every lane reads the one row,
  // as wmma_kernel's rows past m re-read the last).
  const uint8_t *a_row = stage + 32 * h;
  v8f acc = v8f{};
  for (int64_t g = g_begin; g < g_end; ++g) {
    u32x4 cb[kStepsPerGroup][2];
    u32x4 ca[kStepsPerGroup][2];
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st) {
      cb[st][0] = wb[st][0];
      cb[st][1] = wb[st][1];
      const int64_t k0 = g * kGroup + st * kStep;
      ca[st][0] = *reinterpret_cast<const u32x4 *>(a_row + k0);
      ca[st][1] = *reinterpret_cast<const u32x4 *>(a_row + k0 + 16);
    }
    if (g + 1 < g_end)
      load_b(g + 1);
#pragma unroll
    for (int st = 0; st < kStepsPerGroup; ++st)
#pragma unroll
      for (int t = 0; t < 4; ++t)
        acc = mma(slice(ca[st], t), slice(cb[st], t), acc);
  }

  // Row 0 of the tile is acc[0] of the lanes of half 0; the tile's waves add
  // their partials in wave order (wmma_kernel's order).
  float total = acc[0];
  if (kWaves > 1) {
    if (kw > 0)
      partial[tile][kw - 1][lane] = total;
    __syncthreads();
    if (kw > 0)
      return;
#pragma unroll
    for (int w = 1; w < kWaves; ++w)
      total += partial[tile][w - 1][lane];
  }
  const int64_t cn = n0 + r;
  if (h != 0 || cn >= n)
    return;
  float value = total * w_scales[w_vector != 0 ? cn : 0];
  value = value * scale;
  value = value * alpha;
  c[cn] = to_bf16(value);
}

// Threads of the BF16 form's block at kWaves waves per tile.
constexpr int bf16_threads(int waves) { return waves * tiles_for(waves) * kLanes; }

template <int kWaves>
hipError_t launch_bf16(const turbine_qgemm_desc *d, hipStream_t stream) {
  constexpr int kTiles = tiles_for(kWaves);
  const int64_t col_tiles = (d->n + kMma - 1) / kMma;
  const dim3 grid(static_cast<uint32_t>((col_tiles + kTiles - 1) / kTiles), 1);
  const dim3 block(bf16_threads(kWaves));
  const size_t lds = static_cast<size_t>(d->k);
  const auto *a = static_cast<const uint16_t *>(d->a);
  const auto *b = static_cast<const uint8_t *>(d->b);
  const auto *ws = static_cast<const float *>(d->b_scales);
  const int32_t w_vector = weight_qscale(d->scheme) == QScale::Vector ? 1 : 0;
  auto *c = static_cast<uint16_t *>(d->c);
  if (d->act_quant == TURBINE_ACTQ_FP8_TOKEN) {
    hipLaunchKernelGGL((bf16_kernel<kWaves, kTiles, true>), grid, block, lds,
                       stream, a, 1.0f, b, ws, w_vector, c, d->n, d->k,
                       d->alpha);
  } else {
    hipLaunchKernelGGL((bf16_kernel<kWaves, kTiles, false>), grid, block, lds,
                       stream, a, d->act_static_scale, b, ws, w_vector, c,
                       d->n, d->k, d->alpha);
  }
  return hipGetLastError();
}

} // namespace fp8_wmma

namespace {

// Decode rows up to which the kernel runs (the timings of hip_qgemm
// qgemm_decode_timings; the card profile's FP8 decode row tier matches it).
constexpr int64_t kMaxRows = 16;

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace

namespace {

// The v2.13 form: BF16 activations with an FP8_TENSOR / FP8_TOKEN act_quant
// (act_static_scale is read only in this form).
bool bf16_form(const turbine_qgemm_desc *d) {
  return d->a_dtype == TURBINE_DTYPE_BF16 &&
         (d->act_quant == TURBINE_ACTQ_FP8_TENSOR ||
          d->act_quant == TURBINE_ACTQ_FP8_TOKEN);
}

bool bf16_supports(const turbine_qgemm_desc *d) {
  if (d->scheme != TURBINE_QSCHEME_FP8_TENSOR &&
      d->scheme != TURBINE_QSCHEME_FP8_CHANNEL)
    return false;
  if (d->c_dtype != TURBINE_DTYPE_BF16 || d->prefill != 0)
    return false;
  if (d->m != fp8_wmma::kBf16Rows || d->n <= 0 || d->k <= 0 ||
      d->k % fp8_wmma::kGroup != 0)
    return false;
  // The row fits the threads' registers (and the LDS stage of k bytes).
  const int waves = fp8_wmma::waves_for(d->k / fp8_wmma::kGroup);
  if (d->k > static_cast<int64_t>(fp8_wmma::bf16_threads(waves)) * 8 *
                 fp8_wmma::kChunks)
    return false;
  if (d->lda < d->k || d->lda % 8 != 0 || d->ldc < d->n)
    return false;
  // A static scale divides every value (as quantize_act requires).
  return d->act_quant != TURBINE_ACTQ_FP8_TENSOR ||
         (d->act_static_scale > 0.0f && d->act_static_scale < INFINITY);
}

} // namespace

bool qgemm_fp8_wmma_supports(const turbine_qgemm_desc *d) {
  if (d != nullptr && bf16_form(d))
    return bf16_supports(d);
  // The calls it hands to hipBLASLt must be ones hipblaslt_fp8 takes, and the
  // kernel needs whole 128-deep groups.
  return qgemm_fp8_supports(d) && d->k % fp8_wmma::kGroup == 0;
}

int32_t qgemm_fp8_wmma_run(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: descriptor is NULL");
  if (!qgemm_fp8_wmma_supports(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_qgemm (turbine_hip_fp8_wmma): unsupported "
                "configuration m=" +
                    std::to_string(d->m) + " n=" + std::to_string(d->n) +
                    " k=" + std::to_string(d->k) +
                    " scheme=" + std::to_string(d->scheme) +
                    " act_quant=" + std::to_string(d->act_quant));
  }
  if (bf16_form(d)) {
    if (d->a == nullptr || d->b == nullptr || d->b_scales == nullptr ||
        d->c == nullptr) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_qgemm: NULL operand or scale pointer");
    }
    if (!aligned16(d->a) || !aligned16(d->b)) {
      return fail(ctx, TURBINE_E_UNSUPPORTED,
                  "turbine_qgemm (turbine_hip_fp8_wmma): BF16 activations "
                  "and weights must be 16-byte aligned");
    }
    const int32_t rc = enter(ctx);
    if (rc != TURBINE_OK)
      return rc;
    hipError_t err;
    switch (fp8_wmma::waves_for(d->k / fp8_wmma::kGroup)) {
    case 16:
      err = fp8_wmma::launch_bf16<16>(d, ctx->stream);
      break;
    case 4:
      err = fp8_wmma::launch_bf16<4>(d, ctx->stream);
      break;
    default:
      err = fp8_wmma::launch_bf16<2>(d, ctx->stream);
      break;
    }
    return check_hip(ctx, err, "qgemm fp8 wmma (BF16 activations) launch");
  }
  if (d->prefill != 0 || d->m > kMaxRows || !aligned16(d->a) ||
      !aligned16(d->b))
    return qgemm_fp8_run(ctx, d);
  if (d->a_scales == nullptr || d->b_scales == nullptr || d->c == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_qgemm: NULL operand or scale pointer");
  }
  const int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  hipError_t err;
  switch (fp8_wmma::waves_for(d->k / fp8_wmma::kGroup)) {
  case 16:
    err = fp8_wmma::launch<16>(d, ctx->stream);
    break;
  case 4:
    err = fp8_wmma::launch<4>(d, ctx->stream);
    break;
  default:
    err = fp8_wmma::launch<2>(d, ctx->stream);
    break;
  }
  return check_hip(ctx, err, "qgemm fp8 wmma kernel launch");
}

} // namespace turbine_hip
