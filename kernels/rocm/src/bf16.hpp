// BF16 storage and conversions for the Turbine HIP kernels of ABI v2
// (paged_attention.hip, moe.hip).
//
// BF16 is kept as raw bits. The f32 -> BF16 rounding is spelled out (round to
// nearest, ties to even; NaN stays NaN) so it matches the cpu-reference
// round_to bit for bit instead of depending on how the compiler lowers a
// __bf16 conversion on the target (the same rule as elementwise.hip).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace turbine_hip {

struct Bf16 {
  uint16_t bits;
};

__device__ inline float bf16_to_f32(Bf16 v) {
  return __uint_as_float(static_cast<uint32_t>(v.bits) << 16);
}

__device__ inline Bf16 f32_to_bf16(float v) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return Bf16{static_cast<uint16_t>((u >> 16) | 0x0040u)};
  }
  u += 0x7fffu + ((u >> 16) & 1u);
  return Bf16{static_cast<uint16_t>(u >> 16)};
}

// Rounds v to BF16 and back (the cpu-reference round_to for BF16).
__device__ inline float round_bf16(float v) {
  return bf16_to_f32(f32_to_bf16(v));
}

} // namespace turbine_hip
