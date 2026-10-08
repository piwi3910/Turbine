// Elementwise kernels of the quantized MoE experts (ABI v2.16, P7 Task 16,
// moe.cpp run_quant): the per-expert static FP8 activation quantization, the
// gated activation with the expert biases (SwiGLU or the gpt-oss clamp, on
// separate or interleaved gate/up columns), the down bias and the shared
// expert's gated combine. The GEMMs between them are the own qgemm
// implementations (decision "P7 quantized MoE expert provider": the per-expert
// loop first).
//
// Numerics follow the cpu-reference moe_experts (crates/turbine-kernels/src/
// cpu/moe.rs): every step rounds to BF16 as the BF16 model's ops do; a
// projection's bias is added to the BF16 qgemm output and rounded again, as
// the BF16 model's `x @ W + b` does, and the reference rounds the same way.
#include "bf16.hpp"
#include "qgemm_quantize.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::round_bf16;

namespace {

constexpr int kBlock = 256;

// The expert in [lo, hi) whose row range of `offsets` holds row r (the
// largest e with offsets[e] <= r).
__device__ inline int32_t expert_of_row(const int32_t *offsets, int32_t lo,
                                        int32_t hi, int32_t r) {
  int32_t a = lo, b = hi - 1;
  while (a < b) {
    const int32_t mid = (a + b + 1) / 2;
    if (offsets[mid] <= r)
      a = mid;
    else
      b = mid - 1;
  }
  return a;
}

// One block per gathered row i: out[i, :] = e4m3(x[i, :] / scales[expert - lo]),
// the arithmetic of turbine_quantize_act's FP8_TENSOR mode with the expert's
// own static scale.
__global__ void __launch_bounds__(kBlock)
    moe_quant_static_kernel(const Bf16 *x, uint8_t *out, int32_t cols,
                            const int32_t *offsets, int32_t first, int32_t lo,
                            int32_t hi, const float *scales) {
  const int64_t i = blockIdx.x;
  const int32_t e = expert_of_row(offsets, lo, hi, first + static_cast<int32_t>(i));
  const float s = scales[e - lo];
  const Bf16 *src = x + i * cols;
  uint8_t *dst = out + i * cols;
  for (int32_t c = threadIdx.x; c < cols; c += blockDim.x)
    dst[c] = turbine_hip::fp8_e4m3_round(bf16_to_f32(src[c]) / s);
}

// One block per gathered row: act[i, j] from the projections' BF16 output gu
// ([rows, 2 * inter]: gate columns then up columns, or interleaved gate/up
// pairs), the optional biases first.
__global__ void __launch_bounds__(kBlock) moe_quant_act_kernel(
    const Bf16 *gu, Bf16 *act, int32_t inter, int32_t interleaved,
    int32_t clamp, float limit, const Bf16 *b_gate, const Bf16 *b_up,
    const int32_t *offsets, int32_t first, int32_t lo, int32_t hi) {
  const int64_t i = blockIdx.x;
  const bool biased = b_gate != nullptr || b_up != nullptr;
  int32_t l = 0;
  if (biased) {
    l = expert_of_row(offsets, lo, hi, first + static_cast<int32_t>(i)) - lo;
  }
  const Bf16 *row = gu + i * 2 * static_cast<int64_t>(inter);
  const int64_t gate_n = interleaved != 0 ? 2 * static_cast<int64_t>(inter)
                                          : static_cast<int64_t>(inter);
  for (int32_t j = threadIdx.x; j < inter; j += blockDim.x) {
    const int32_t gc = interleaved != 0 ? 2 * j : j;
    const int32_t uc = interleaved != 0 ? 2 * j + 1 : inter + j;
    float g = bf16_to_f32(row[gc]);
    float u = bf16_to_f32(row[uc]);
    if (b_gate != nullptr)
      g = round_bf16(g + bf16_to_f32(b_gate[l * gate_n + gc]));
    if (b_up != nullptr) {
      // Interleaved: b_gate carries both columns; b_up is unused.
      u = round_bf16(u + bf16_to_f32(b_up[static_cast<int64_t>(l) * inter + j]));
    } else if (interleaved != 0 && b_gate != nullptr) {
      u = round_bf16(bf16_to_f32(row[uc]) +
                     bf16_to_f32(b_gate[l * gate_n + uc]));
    }
    float o;
    if (clamp != 0) {
      g = fminf(g, limit);
      u = fminf(fmaxf(u, -limit), limit);
      const float sig =
          round_bf16(1.0f / (1.0f + expf(-round_bf16(g * 1.702f))));
      const float glu = round_bf16(g * sig);
      o = round_bf16(round_bf16(u + 1.0f) * glu);
    } else {
      const float silu = g / (1.0f + expf(-g));
      o = round_bf16(round_bf16(silu) * u);
    }
    act[i * inter + j] = f32_to_bf16(o);
  }
}

// y[i, :] = round(y[i, :] + bias[expert(i), :]).
__global__ void __launch_bounds__(kBlock)
    moe_quant_bias_kernel(Bf16 *y, int32_t cols, const Bf16 *bias,
                          const int32_t *offsets, int32_t first, int32_t lo,
                          int32_t hi) {
  const int64_t i = blockIdx.x;
  const int32_t l =
      expert_of_row(offsets, lo, hi, first + static_cast<int32_t>(i)) - lo;
  Bf16 *row = y + i * cols;
  const Bf16 *b = bias + static_cast<int64_t>(l) * cols;
  for (int32_t c = threadIdx.x; c < cols; c += blockDim.x)
    row[c] = f32_to_bf16(bf16_to_f32(row[c]) + bf16_to_f32(b[c]));
}

// One block per token: out[t] = round(out[t] + round(y[t] * scale)), scale =
// round(sigmoid(round(x[t] . gate_w))) or 1 without a gate.
__global__ void __launch_bounds__(kBlock)
    moe_shared_combine_kernel(const Bf16 *x, const Bf16 *y, const Bf16 *gate_w,
                              Bf16 *out, int32_t hidden) {
  __shared__ float red[kBlock];
  const int64_t t = blockIdx.x;
  float scale = 1.0f;
  if (gate_w != nullptr) {
    float dot = 0.0f;
    for (int32_t c = threadIdx.x; c < hidden; c += blockDim.x)
      dot += bf16_to_f32(x[t * hidden + c]) * bf16_to_f32(gate_w[c]);
    red[threadIdx.x] = dot;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s)
        red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    scale = round_bf16(1.0f / (1.0f + expf(-round_bf16(red[0]))));
  }
  for (int32_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    const float o = bf16_to_f32(out[t * hidden + c]);
    const float v = round_bf16(bf16_to_f32(y[t * hidden + c]) * scale);
    out[t * hidden + c] = f32_to_bf16(o + v);
  }
}

} // namespace

namespace turbine_hip {

int32_t launch_moe_quant_static(turbine_ctx *ctx, const void *x, void *out,
                                int64_t rows, int32_t cols,
                                const int32_t *offsets, int32_t first,
                                int32_t lo, int32_t hi, const float *scales) {
  if (rows == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_quant_static_kernel,
                     dim3(static_cast<unsigned>(rows)), dim3(kBlock), 0,
                     ctx->stream, static_cast<const Bf16 *>(x),
                     static_cast<uint8_t *>(out), cols, offsets, first, lo, hi,
                     scales);
  return check_hip(ctx, hipGetLastError(), "moe quant static launch");
}

int32_t launch_moe_quant_act(turbine_ctx *ctx, const void *gu, void *act,
                             int64_t rows, int32_t inter, bool interleaved,
                             bool clamp, float limit, const void *b_gate,
                             const void *b_up, const int32_t *offsets,
                             int32_t first, int32_t lo, int32_t hi) {
  if (rows == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_quant_act_kernel, dim3(static_cast<unsigned>(rows)),
                     dim3(kBlock), 0, ctx->stream,
                     static_cast<const Bf16 *>(gu), static_cast<Bf16 *>(act),
                     inter, interleaved ? 1 : 0, clamp ? 1 : 0, limit,
                     static_cast<const Bf16 *>(b_gate),
                     static_cast<const Bf16 *>(b_up), offsets, first, lo, hi);
  return check_hip(ctx, hipGetLastError(), "moe quant activation launch");
}

int32_t launch_moe_quant_bias(turbine_ctx *ctx, void *y, int64_t rows,
                              int32_t cols, const void *bias,
                              const int32_t *offsets, int32_t first,
                              int32_t lo, int32_t hi) {
  if (rows == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_quant_bias_kernel, dim3(static_cast<unsigned>(rows)),
                     dim3(kBlock), 0, ctx->stream, static_cast<Bf16 *>(y), cols,
                     static_cast<const Bf16 *>(bias), offsets, first, lo, hi);
  return check_hip(ctx, hipGetLastError(), "moe quant bias launch");
}

int32_t launch_moe_shared_combine(turbine_ctx *ctx, const void *x,
                                  const void *y, const void *gate_w, void *out,
                                  int32_t tokens, int32_t hidden) {
  if (tokens == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_shared_combine_kernel,
                     dim3(static_cast<unsigned>(tokens)), dim3(kBlock), 0,
                     ctx->stream, static_cast<const Bf16 *>(x),
                     static_cast<const Bf16 *>(y),
                     static_cast<const Bf16 *>(gate_w), static_cast<Bf16 *>(out),
                     hidden);
  return check_hip(ctx, hipGetLastError(), "moe shared combine launch");
}

} // namespace turbine_hip
