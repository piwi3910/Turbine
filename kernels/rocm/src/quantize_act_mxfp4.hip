// MXFP4 activation emulation (ABI v2.9 turbine_quantize_act, mode
// MXFP4_EMULATED) on ROCm: `turbine_hip_mxfp4`, the quantize-dequantize that
// runs Quark W4A4 checkpoints on RDNA4 (no FP4 matrix instructions): each
// 32-column group of a row gets an E8M0 exponent by Quark's `even` rule, every
// value is divided by it, rounded to the nearest E2M1 value (ties to the even
// code, saturated at 6) and multiplied back. Bit-exact with cpu::quant
// (quantize_dequantize_activations); the group scales are written as F32
// powers of two. An own kernel per the decision "P6: MXFP4 GEMM — provider
// evaluation (kernel reuse rule)": no provider takes the v2.9 quantize_act op
// (CK's MX quantization lives inside its gfx950 MX GEMM pipelines).
//
// One wave per group, one lane per column; x and out may alias (every lane
// reads its value before the group's reduction and writes it after).
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
constexpr int kWaves = 8;
constexpr int kGroup = 32;

// Quark `even` E8M0 exponent of a group whose largest magnitude is amax
// (cpu::quant::mxfp4_scale_even): the mantissa rounded up to the next power of
// two when it is >= 1.75, then floor(log2) - 2, clamped to [-127, 127].
__device__ inline uint32_t scale_even(float amax) {
  if (amax != amax)
    return 255u;
  const uint32_t rounded = (__float_as_uint(amax) + (1u << 21)) & 0xff800000u;
  const int32_t field = static_cast<int32_t>((rounded >> 23) & 0xffu);
  int32_t log2;
  if (field == 255)
    log2 = 32767; // rounded past the largest finite value
  else if (field == 0)
    log2 = -127; // zero or subnormal
  else
    log2 = field - 127;
  const int32_t e = min(max(log2 - 2, -127), 127);
  return static_cast<uint32_t>(e + 127);
}

// 2^(e - 127) as F32 (NaN for 255).
__device__ inline float e8m0_value(uint32_t e) {
  if (e == 0)
    return __uint_as_float(0x00400000u);
  if (e == 255)
    return __uint_as_float(0x7fc00000u);
  return __uint_as_float(e << 23);
}

// The E2M1 value x rounds to (cpu::quant::e2m1_round then e2m1_value): nearest
// of 0, 0.5, 1, 1.5, 2, 3, 4, 6, ties to the even code, saturated at 6, NaN to
// 0; a value that rounds to 0 loses its sign.
__device__ inline float e2m1_qdq(float x) {
  if (x != x)
    return 0.0f;
  const float a = fminf(fabsf(x), 6.0f);
  float q;
  if (a <= 0.25f)
    q = 0.0f;
  else if (a < 0.75f)
    q = 0.5f;
  else if (a <= 1.25f)
    q = 1.0f;
  else if (a < 1.75f)
    q = 1.5f;
  else if (a <= 2.5f)
    q = 2.0f;
  else if (a < 3.5f)
    q = 3.0f;
  else if (a <= 5.0f)
    q = 4.0f;
  else
    q = 6.0f;
  if (q == 0.0f)
    return 0.0f;
  return (__float_as_uint(x) >> 31) ? -q : q;
}

template <bool XF32, bool OutF32>
__global__ __launch_bounds__(kLanes *kWaves) void mxfp4_qdq_kernel(
    const void *x, void *out, float *scales, int64_t rows, int64_t cols,
    int64_t x_stride, int64_t out_stride) {
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int64_t groups_per_row = (cols + kGroup - 1) / kGroup;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * kWaves +
                      static_cast<int64_t>(threadIdx.x) / kLanes;
  if (gid >= rows * groups_per_row)
    return;
  const int64_t row = gid / groups_per_row;
  const int64_t gi = gid % groups_per_row;
  const int64_t col = gi * kGroup + lane;
  const bool valid = col < cols;
  float v = 0.0f;
  if (valid) {
    const int64_t at = row * x_stride + col;
    v = XF32 ? static_cast<const float *>(x)[at]
             : turbine_hip::bf16_to_f32(static_cast<const Bf16 *>(x)[at]);
  }
  // Largest magnitude of the group; a NaN is skipped like Rust's f32::max.
  float amax = fabsf(v);
#pragma unroll
  for (int off = kLanes / 2; off > 0; off /= 2)
    amax = fmaxf(amax, __shfl_xor(amax, off, kLanes));
  const float s = e8m0_value(scale_even(amax));
  const float y = e2m1_qdq(v / s) * s;
  if (valid) {
    const int64_t at = row * out_stride + col;
    if (OutF32)
      static_cast<float *>(out)[at] = y;
    else
      static_cast<Bf16 *>(out)[at] = f32_to_bf16(y);
  }
  if (lane == 0)
    scales[gid] = s;
}

template <bool XF32, bool OutF32>
hipError_t launch(hipStream_t stream, const turbine_quantize_act_desc *d) {
  const int64_t groups = d->rows * ((d->cols + kGroup - 1) / kGroup);
  const int64_t blocks = (groups + kWaves - 1) / kWaves;
  hipLaunchKernelGGL((mxfp4_qdq_kernel<XF32, OutF32>),
                     dim3(static_cast<unsigned>(blocks)), dim3(kLanes * kWaves),
                     0, stream, d->x, d->out, d->scales, d->rows, d->cols,
                     d->x_stride_row, d->out_stride_row);
  return hipGetLastError();
}

bool is_float(int32_t dtype) {
  return dtype == TURBINE_DTYPE_BF16 || dtype == TURBINE_DTYPE_F32;
}

bool supported(const turbine_quantize_act_desc *d) {
  if (d == nullptr || d->mode != TURBINE_ACTQ_MXFP4_EMULATED)
    return false;
  if (!is_float(d->x_dtype) || !is_float(d->out_dtype))
    return false;
  if (d->rows < 0 || d->cols <= 0)
    return false;
  if (d->x_stride_row < d->cols || d->out_stride_row < d->cols)
    return false;
  // One block per 8 groups in a 1-D grid.
  const int64_t groups = d->rows * ((d->cols + kGroup - 1) / kGroup);
  return groups / kWaves < INT32_MAX;
}

} // namespace

namespace turbine_hip {

bool quantize_act_mxfp4_supports(const turbine_quantize_act_desc *d) {
  return supported(d);
}

hipError_t launch_quantize_mxfp4(hipStream_t stream,
                                 const turbine_quantize_act_desc *d) {
  if (d->rows == 0)
    return hipSuccess;
  const bool xf = d->x_dtype == TURBINE_DTYPE_F32;
  const bool of = d->out_dtype == TURBINE_DTYPE_F32;
  if (xf)
    return of ? launch<true, true>(stream, d) : launch<true, false>(stream, d);
  return of ? launch<false, true>(stream, d) : launch<false, false>(stream, d);
}

int32_t quantize_act_mxfp4_run(turbine_ctx *ctx,
                               const turbine_quantize_act_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_quantize_act: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_quantize_act (turbine_hip_mxfp4): unsupported "
                "configuration mode=" +
                    std::to_string(d->mode) +
                    " rows=" + std::to_string(d->rows) +
                    " cols=" + std::to_string(d->cols) +
                    " dtypes=" + std::to_string(d->x_dtype) + "," +
                    std::to_string(d->out_dtype));
  }
  if (d->rows == 0)
    return TURBINE_OK;
  if (d->x == nullptr || d->out == nullptr || d->scales == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_quantize_act: NULL operand");
  }
  const int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, launch_quantize_mxfp4(ctx->stream, d),
                   "turbine_quantize_act (turbine_hip_mxfp4) launch");
}

} // namespace turbine_hip
