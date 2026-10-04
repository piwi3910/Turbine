// FP8 e4m3 (OCP e4m3fn) conversions for the FP8 KV pages (Phase 6a S-13,
// kv.dtype fp8_e4m3), spelled out so they match the cpu-reference
// (crates/turbine-kernels/src/cpu/quant.rs fp8_e4m3_round / fp8_e4m3_value)
// bit for bit instead of depending on how a target lowers an FP8 conversion:
// bias 7, no infinities, NaN = 0x7f / 0xff, largest finite +-448; rounding to
// nearest with ties to even, saturated to +-448.
//
// A page element is written as e4m3(x / scale) and read as
// bf16(e4m3 * scale): attention then runs on BF16 K/V exactly as over BF16
// pages holding those values.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "bf16.hpp"

namespace turbine_hip {

// OCP e4m3fn bits of x: round to nearest, ties to even, saturated to +-448;
// NaN -> 0x7f.
__device__ inline uint8_t f32_to_fp8_e4m3(float x) {
  const uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return 0x7f;
  const uint8_t sign = (u >> 24) & 0x80u;
  const float a = __uint_as_float(u & 0x7fffffffu);
  if (a >= 448.0f)
    return sign | 0x7e;
  // Below the smallest normal (2^-6): multiples of 2^-9 (exact scaling).
  if (a < 0.015625f) {
    const float m = rintf(a * 512.0f); // rintf: ties to even
    return sign | static_cast<uint8_t>(m);
  }
  const uint32_t bits = __float_as_uint(a);
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
  const uint32_t mant = bits & 0x7fffffu;
  uint32_t q = mant >> 20;
  const uint32_t rem = mant & 0xfffffu;
  const uint32_t half = 0x80000u;
  if (rem > half || (rem == half && (q & 1u) == 1u))
    q += 1;
  if (q == 8) {
    q = 0;
    exp += 1;
  }
  return sign | static_cast<uint8_t>((exp + 7) << 3) | static_cast<uint8_t>(q);
}

// The value of OCP e4m3fn bits (NaN for 0x7f / 0xff); exact in f32.
__device__ inline float fp8_e4m3_to_f32(uint8_t b) {
  const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
  const uint32_t e = (b >> 3) & 0xfu;
  const uint32_t m = b & 7u;
  if (e == 15 && m == 7)
    return __uint_as_float(sign | 0x7fc00000u);
  if (e == 0) {
    // m * 2^-9, exact.
    const float mag = static_cast<float>(m) * 0.001953125f;
    return sign != 0 ? -mag : mag;
  }
  return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
}

// A page byte read back into the BF16 activation dtype, as a float.
__device__ inline float fp8_dequant(uint8_t b, float scale) {
  return round_bf16(fp8_e4m3_to_f32(b) * scale);
}

} // namespace turbine_hip
