// TurboQuant device code shared by the KV transcode (kv_transcode_tq.hip) and
// the mixed-format paged attention (paged_attention_mixed.hip): the record
// layout of the codecs tq4 (K and V 4 bits, 144-byte records) and tq2 (K and
// V 2 bits, 80 bytes) of crates/turbine-kv codec/turboquant at head_dim
// 128, and the chunked encode and decode of up to 32 token vectors of one KV
// head by a workgroup of 128 threads. K and V are both TurboQuant_mse (no QJL
// residual; decision "6b Task 9: TurboQuant K quantizer" A).
//
// Bit-exact by transcription of the codec (kv_transcode_tq.hip's header lists
// the steps); every function that computes on values disables floating-point
// contraction in its own body, so a file including this one keeps its own
// contraction setting for the rest of its code.
#pragma once

#include <cstdint>

#include "bf16.hpp"

namespace turbine_hip {
namespace tq {

constexpr int kDim = 128;           // head_dim of the committed codebooks
constexpr int kThreads = 128;       // 4 groups of 32 lanes
constexpr int kGroup = 32;          // lanes holding one vector
constexpr int kPer = kDim / kGroup; // elements of a vector per lane
constexpr int kChunk = 32;          // tokens staged at once
constexpr int kPad = kDim + 1;      // LDS row stride (bank spread)
// Elements of one (layer, KV head) entry of the tables: K signs, V signs.
constexpr int64_t kHeadElems = 2 * kDim;

// f32(sqrt(128)) and f32(1 / sqrt(128)), the codec's constants (each rounded
// once from F64).
constexpr float kSqrtD = 0x1.6a09e6p+3f;
constexpr float kInvSqrtD = 0x1.6a09e6p-4f;

// Field offsets of a record of KB-bit K codes and VB-bit V codes.
template <int KB, int VB> struct Record {
  static constexpr int kKCodes = 0;
  static constexpr int kKNorm = kDim * KB / 8;
  static constexpr int kVCodes = kKNorm + 2;
  static constexpr int kVNorm = kVCodes + kDim * VB / 8;
  static constexpr int kBytes = (kVNorm + 2 + 15) / 16 * 16;
};
using Tq4 = Record<4, 4>;
using Tq2 = Record<2, 2>;
static_assert(Tq4::kBytes == 144 && Tq2::kBytes == 80, "record sizes");

// The unnormalised Walsh-Hadamard transform of the vector 32 lanes hold
// (v[i] is element i * 32 + lane), in the codec's butterfly order.
__device__ inline void fwht(float (&v)[kPer], int lane) {
#pragma clang fp contract(off)
#pragma unroll
  for (int h = 1; h < kGroup; h *= 2) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const float other = __shfl_xor(v[i], h, kGroup);
      v[i] = (lane & h) == 0 ? v[i] + other : other - v[i];
    }
  }
#pragma unroll
  for (int step = 1; step < kPer; step *= 2) {
#pragma unroll
    for (int i = 0; i < kPer; i += 2 * step) {
#pragma unroll
      for (int k = 0; k < step; ++k) {
        const float a = v[i + k];
        const float b = v[i + k + step];
        v[i + k] = a + b;
        v[i + k + step] = a - b;
      }
    }
  }
}

// sqrt of the F64 sum of squares of row[0 .. 128) in order, to F32.
__device__ inline float norm_of(const float *row) {
#pragma clang fp contract(off)
  double acc = 0.0;
  for (int j = 0; j < kDim; ++j) {
    const double v = static_cast<double>(row[j]);
    acc = acc + v * v;
  }
  return static_cast<float>(sqrt(acc));
}

// The midpoint (cb[i] + cb[i + 1]) * 0.5 of an ascending codebook's
// neighbours i, i + 1, as codebook.rs nearest computes it.
__device__ inline float midpoint(const float *cb, int i) {
#pragma clang fp contract(off)
  return (cb[i] + cb[i + 1]) * 0.5f;
}

// norm_of(row) without its 128-step F64 chain wherever that is provably the
// same F32 (bit for bit): the squares, exact in F32 for BF16 inputs (checked
// with a fused multiply-add), are summed in double-F32 (TwoSum, then a
// Fast2Sum renormalisation; relative error <= 128 * 2^-47 = 2^-40), and the
// codec's sequential F64 sum S_seq lies within 2^-46 of the exact sum. r =
// f32(sqrt(S)) of that estimate S is taken when every sum within a relative
// 2^-35 of S rounds to r: when S * (1 - 2^-35) lies above m_lo^2 and S * (1 +
// 2^-35) below m_hi^2, m_lo and m_hi being the midpoints between r and its F32
// neighbours (exact in F64, and so are their squares), the double sqrt of any
// such sum lies strictly between the midpoints, so it rounds to r. Otherwise
// (about 2^-11 of vectors), a square that is not exact, or an element
// outside 2^-40 <= |x| < 2^60 other than 0, the sequential loop. A row of
// zeros gives 0 either way.
__device__ inline float norm_fast(const float *row) {
#pragma clang fp contract(off)
  float hi = 0.0f;
  float lo = 0.0f;
  bool exact = true;
  for (int j = 0; j < kDim; ++j) {
    const float x = row[j];
    const float p = x * x;
    // Exactness is only trusted where neither the square nor its residual
    // can underflow or overflow: 0, or 2^-40 <= |x| < 2^60 (NaN fails).
    const float ax = __builtin_fabsf(x);
    exact = exact && (x == 0.0f || (ax >= 0x1p-40f && ax < 0x1p60f)) &&
            __builtin_fmaf(x, x, -p) == 0.0f;
    // (s, e) = TwoSum(hi, p); (hi, lo) = Fast2Sum(s, lo + e).
    const float s = hi + p;
    const float bb = s - hi;
    const float e = (hi - (s - bb)) + (p - bb);
    const float v = lo + e;
    hi = s + v;
    lo = v - (hi - s);
  }
  const double sum = static_cast<double>(hi) + static_cast<double>(lo);
  if (sum == 0.0 && exact)
    return 0.0f;
  const float r = static_cast<float>(sqrt(sum));
  if (exact && sum > 0.0 && __builtin_isfinite(sum) && r > 0.0f &&
      __builtin_isfinite(r)) {
    constexpr double kWindow = 0x1p-35;
    // r's F32 neighbours (r is positive and finite; above the largest F32
    // comes infinity).
    const uint32_t bits = __builtin_bit_cast(uint32_t, r);
    const float below = __builtin_bit_cast(float, bits - 1u);
    const float above = __builtin_bit_cast(float, bits + 1u);
    const double m_lo = (static_cast<double>(below) + r) * 0.5;
    const double m_hi = (static_cast<double>(r) + above) * 0.5;
    if (__builtin_isfinite(m_hi) && sum * (1.0 - kWindow) > m_lo * m_lo &&
        sum * (1.0 + kWindow) < m_hi * m_hi)
      return r;
  }
  return norm_of(row);
}

// Index of the nearest centroid of an ascending codebook of N entries, given
// its N - 1 midpoints: the number of midpoints x is above (codebook.rs nearest
// scans them in order and stops at the first one x is not above, the same
// count since they ascend), so a value on a midpoint takes the lower code and
// NaN takes 0. A branch-free binary search: log2(N) comparisons.
template <int N> __device__ inline int nearest(const float *mid, float x) {
  int pos = 0;
#pragma unroll
  for (int step = N / 2; step >= 1; step /= 2)
    pos += x > mid[pos + step - 1] ? step : 0;
  return pos;
}

// Byte `byte` of the packed (LSB first) B-bit codes (each code < 2^B, B
// dividing 8): 8 / B codes, the first in the low bits.
template <int B>
__device__ inline uint8_t packed_byte(const uint8_t *codes, int byte) {
  constexpr int kPerByte = 8 / B;
  uint32_t out = 0;
#pragma unroll
  for (int k = 0; k < kPerByte; ++k)
    out |= static_cast<uint32_t>(codes[byte * kPerByte + k]) << (k * B);
  return static_cast<uint8_t>(out);
}

template <int B>
__device__ inline int unpack_code(const uint8_t *packed, int i) {
  int code = 0;
#pragma unroll
  for (int j = 0; j < B; ++j) {
    const int p = i * B + j;
    code |= ((packed[p / 8] >> (p % 8)) & 1) << j;
  }
  return code;
}

// LDS of the chunked encode.
template <int KB, int VB> struct EncodeLds {
  float xs[2][kChunk][kPad];
  uint8_t codes[2][kChunk][kDim];
  float norms[2][kChunk];
  uint16_t norm_bits[2][kChunk];
  float cbk[1 << KB];
  float cbv[1 << VB];
  // The codebooks' midpoints (encode_chunk fills them).
  float midk[1 << KB];
  float midv[1 << VB];
};

// Encodes nt <= kChunk token vectors of one KV head (head_tables: its K signs
// then V signs) that the caller staged in lds.xs[kind][tok][0 .. 128) (kind
// 0 = K, 1 = V), with lds.cbk / lds.cbv loaded, followed by a barrier. Every
// byte of record tok is passed to put(tok, byte, value), in byte order per
// token, the threads of the workgroup sharing the bytes. Ends with a barrier.
// blockDim.x == kThreads, every thread calls it.
template <int KB, int VB, typename Put>
__device__ inline void encode_chunk(EncodeLds<KB, VB> &lds, int nt,
                                    const float *head_tables, Put put) {
#pragma clang fp contract(off)
  using R = Record<KB, VB>;
  const int tid = threadIdx.x;
  const int group = tid / kGroup;
  const int lane = tid % kGroup;
  // Norms: one thread per vector, coordinates in order.
  if (tid < 2 * kChunk && tid % kChunk < nt) {
    const int kind = tid / kChunk;
    const int tok = tid % kChunk;
    const float n = norm_fast(lds.xs[kind][tok]);
    lds.norms[kind][tok] = n;
    lds.norm_bits[kind][tok] = f32_to_bf16(n).bits;
  }
  // The midpoints, by threads the norms leave idle.
  if (tid >= 2 * kChunk && tid < 2 * kChunk + (1 << KB) - 1)
    lds.midk[tid - 2 * kChunk] = midpoint(lds.cbk, tid - 2 * kChunk);
  if (tid >= 3 * kChunk && tid < 3 * kChunk + (1 << VB) - 1)
    lds.midv[tid - 3 * kChunk] = midpoint(lds.cbv, tid - 3 * kChunk);
  __syncthreads();
  // Rotate and quantize: each group of 32 lanes one vector at a time.
  for (int vec = group; vec < 2 * nt; vec += kThreads / kGroup) {
    const int kind = vec / nt;
    const int tok = vec % nt;
    const float *signs = head_tables + kind * kDim;
    float y[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int e = i * kGroup + lane;
      y[i] = lds.xs[kind][tok][e] * signs[e];
    }
    fwht(y, lane);
    const float n = lds.norms[kind][tok];
    const float inv = n > 0.0f ? kSqrtD / n : 0.0f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int e = i * kGroup + lane;
      const float yi = y[i] * kInvSqrtD;
      const int code = kind == 0 ? nearest<1 << KB>(lds.midk, yi * inv)
                                 : nearest<1 << VB>(lds.midv, yi * inv);
      lds.codes[kind][tok][e] = static_cast<uint8_t>(code);
    }
  }
  __syncthreads();
  __syncthreads();
  for (int i = tid; i < nt * R::kBytes; i += kThreads) {
    const int tok = i / R::kBytes;
    const int o = i % R::kBytes;
    uint8_t v = 0;
    if (o < R::kKNorm) {
      v = packed_byte<KB>(lds.codes[0][tok], o);
    } else if (o < R::kVCodes) {
      v = static_cast<uint8_t>(lds.norm_bits[0][tok] >> (8 * (o - R::kKNorm)));
    } else if (o < R::kVNorm) {
      v = packed_byte<VB>(lds.codes[1][tok], o - R::kVCodes);
    } else if (o < R::kVNorm + 2) {
      v = static_cast<uint8_t>(lds.norm_bits[1][tok] >> (8 * (o - R::kVNorm)));
    }
    put(tok, o, v);
  }
  __syncthreads();
}

// LDS of the chunked decode.
template <int KB, int VB> struct DecodeLds {
  uint8_t recs[kChunk][Record<KB, VB>::kBytes];
  float cbk[1 << KB];
  float cbv[1 << VB];
};

// Decodes nt <= kChunk records of one KV head that the caller staged in
// lds.recs (with lds.cbk / lds.cbv), followed by a barrier: element e of
// vector (kind, tok) is passed to put(kind, tok, e, bf16 bits). Ends with a
// barrier. blockDim.x == kThreads, every thread calls it.
template <int KB, int VB, typename Put>
__device__ inline void decode_chunk(DecodeLds<KB, VB> &lds, int nt,
                                    const float *head_tables, Put put) {
#pragma clang fp contract(off)
  using R = Record<KB, VB>;
  const int tid = threadIdx.x;
  const int group = tid / kGroup;
  const int lane = tid % kGroup;
  for (int vec = group; vec < 2 * nt; vec += kThreads / kGroup) {
    const int kind = vec / nt;
    const int tok = vec % nt;
    const uint8_t *rec = lds.recs[tok];
    const int at = kind == 0 ? R::kKNorm : R::kVNorm;
    const uint16_t nb = static_cast<uint16_t>(rec[at] | (rec[at + 1] << 8));
    const float k = bf16_to_f32(Bf16{nb}) / kSqrtD;
    float y[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int e = i * kGroup + lane;
      y[i] = kind == 0 ? lds.cbk[unpack_code<KB>(rec + R::kKCodes, e)] * k
                       : lds.cbv[unpack_code<VB>(rec + R::kVCodes, e)] * k;
    }
    fwht(y, lane);
    const float *signs = head_tables + kind * kDim;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int e = i * kGroup + lane;
      put(kind, tok, e, f32_to_bf16(y[i] * (kInvSqrtD * signs[e])).bits);
    }
  }
  __syncthreads();
}

} // namespace tq
} // namespace turbine_hip
