// TurboQuant device code shared by the KV transcode (kv_transcode_tq.hip) and
// the mixed-format paged attention (paged_attention_mixed.hip): the record
// layout of the codecs tq4 (K 3 + 1 bits, V 4 bits, 144-byte records) and tq2
// (K 1 + 1, V 2, 80 bytes) of crates/turbine-kv codec/turboquant at head_dim
// 128, and the chunked encode and decode of up to 32 token vectors of one KV
// head by a workgroup of 128 threads.
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
// Elements of one (layer, KV head) entry of the tables: K signs, V signs, S.
constexpr int64_t kHeadElems = 2 * kDim + kDim * kDim;

// f32(sqrt(128)), f32(1 / sqrt(128)) and f32(sqrt(pi / 2) / 128), the codec's
// constants (each rounded once from F64).
constexpr float kSqrtD = 0x1.6a09e6p+3f;
constexpr float kInvSqrtD = 0x1.6a09e6p-4f;
constexpr float kQjl = 0x1.40d932p-7f;

// Field offsets of a record of KB-bit K codes and VB-bit V codes.
template <int KB, int VB> struct Record {
  static constexpr int kKCodes = 0;
  static constexpr int kKNorm = kDim * KB / 8;
  static constexpr int kQjl = kKNorm + 2;
  static constexpr int kRNorm = kQjl + kDim / 8;
  static constexpr int kVCodes = kRNorm + 2;
  static constexpr int kVNorm = kVCodes + kDim * VB / 8;
  static constexpr int kBytes = (kVNorm + 2 + 15) / 16 * 16;
};
using Tq4 = Record<3, 4>;
using Tq2 = Record<1, 2>;
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

// Index of the nearest centroid of an ascending codebook of n entries; a
// value on a midpoint takes the lower one (codebook.rs nearest).
__device__ inline int nearest(const float *cb, int n, float x) {
#pragma clang fp contract(off)
  int code = 0;
  for (int i = 0; i + 1 < n; ++i) {
    if (x > (cb[i] + cb[i + 1]) * 0.5f)
      ++code;
    else
      break;
  }
  return code;
}

// Byte `byte` of the packed (LSB first) B-bit codes.
template <int B>
__device__ inline uint8_t packed_byte(const uint8_t *codes, int byte) {
  uint32_t out = 0;
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const int p = byte * 8 + b;
    out |= static_cast<uint32_t>((codes[p / B] >> (p % B)) & 1u) << b;
  }
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
  float xs[2][kChunk][kPad]; // x, then y - y_hat (K)
  uint8_t codes[2][kChunk][kDim];
  float norms[2][kChunk];
  uint16_t norm_bits[2][kChunk];
  uint16_t rnorm_bits[kChunk];
  uint32_t qjl[kChunk][kDim / 32];
  float cbk[1 << KB];
  float cbv[1 << VB];
};

// Encodes nt <= kChunk token vectors of one KV head (head_tables: its K signs,
// V signs and S) that the caller staged in lds.xs[kind][tok][0 .. 128) (kind
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
  const float *qjl_s = head_tables + 2 * kDim;
  // Norms: one thread per vector, coordinates in order.
  if (tid < 2 * kChunk && tid % kChunk < nt) {
    const int kind = tid / kChunk;
    const int tok = tid % kChunk;
    const float n = norm_of(lds.xs[kind][tok]);
    lds.norms[kind][tok] = n;
    lds.norm_bits[kind][tok] = f32_to_bf16(n).bits;
  }
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
    const float k = bf16_to_f32(Bf16{lds.norm_bits[kind][tok]}) / kSqrtD;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int e = i * kGroup + lane;
      const float yi = y[i] * kInvSqrtD;
      const int code = kind == 0 ? nearest(lds.cbk, 1 << KB, yi * inv)
                                 : nearest(lds.cbv, 1 << VB, yi * inv);
      lds.codes[kind][tok][e] = static_cast<uint8_t>(code);
      if (kind == 0)
        lds.xs[0][tok][e] = yi - lds.cbk[code] * k;
    }
  }
  __syncthreads();
  // Residual norms, then the sign bits of S * r: thread i owns row i of S.
  if (tid < nt)
    lds.rnorm_bits[tid] = f32_to_bf16(norm_of(lds.xs[0][tid])).bits;
  {
    float acc[kChunk];
#pragma unroll
    for (int t = 0; t < kChunk; ++t)
      acc[t] = -0.0f;
    const float *row = qjl_s + static_cast<int64_t>(tid) * kDim;
    for (int j = 0; j < kDim; j += 4) {
      const float4 s4 = *reinterpret_cast<const float4 *>(row + j);
      const float s[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int t = 0; t < kChunk; ++t)
          acc[t] = acc[t] + s[q] * lds.xs[0][t][j + q];
      }
    }
    for (int t = 0; t < nt; ++t) {
      const uint64_t mask = __ballot(acc[t] >= 0.0f);
      if (lane == 0) {
        // Rows group * 32 .. + 31 are bits of qjl bytes 4 * group .. + 3.
        const int shift = (threadIdx.x % warpSize) / kGroup * kGroup;
        lds.qjl[t][group] = static_cast<uint32_t>(mask >> shift);
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < nt * R::kBytes; i += kThreads) {
    const int tok = i / R::kBytes;
    const int o = i % R::kBytes;
    uint8_t v = 0;
    if (o < R::kKNorm) {
      v = packed_byte<KB>(lds.codes[0][tok], o);
    } else if (o < R::kQjl) {
      v = static_cast<uint8_t>(lds.norm_bits[0][tok] >> (8 * (o - R::kKNorm)));
    } else if (o < R::kRNorm) {
      const int q = o - R::kQjl;
      v = static_cast<uint8_t>(lds.qjl[tok][q / 4] >> (8 * (q % 4)));
    } else if (o < R::kVCodes) {
      v = static_cast<uint8_t>(lds.rnorm_bits[tok] >> (8 * (o - R::kRNorm)));
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
  float rhat[kChunk][kPad];
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
  const float *qjl_s = head_tables + 2 * kDim;
  // r_hat: thread j owns column j of S; rows in order.
  {
    float acc[kChunk];
#pragma unroll
    for (int t = 0; t < kChunk; ++t)
      acc[t] = 0.0f;
    for (int i = 0; i < kDim; ++i) {
      const float a = qjl_s[static_cast<int64_t>(i) * kDim + tid];
#pragma unroll
      for (int t = 0; t < kChunk; ++t) {
        const float z =
            (lds.recs[t][R::kQjl + i / 8] >> (i % 8)) & 1 ? 1.0f : -1.0f;
        acc[t] = acc[t] + z * a;
      }
    }
    for (int t = 0; t < nt; ++t) {
      const uint16_t rn = static_cast<uint16_t>(
          lds.recs[t][R::kRNorm] | (lds.recs[t][R::kRNorm + 1] << 8));
      const float k = kQjl * bf16_to_f32(Bf16{rn});
      lds.rhat[t][tid] = acc[t] * k;
    }
  }
  __syncthreads();
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
      if (kind == 0) {
        y[i] = lds.cbk[unpack_code<KB>(rec + R::kKCodes, e)] * k;
        y[i] = y[i] + lds.rhat[tok][e];
      } else {
        y[i] = lds.cbv[unpack_code<VB>(rec + R::kVCodes, e)] * k;
      }
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
