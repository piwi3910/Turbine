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
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "qgemm_impls.hpp"
#include "qgemm_problem.hpp"

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

} // namespace fp8_wmma

namespace {

// Decode rows up to which the kernel runs (the timings of hip_qgemm
// qgemm_decode_timings; the card profile's FP8 decode row tier matches it).
constexpr int64_t kMaxRows = 16;

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace

bool qgemm_fp8_wmma_supports(const turbine_qgemm_desc *d) {
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
