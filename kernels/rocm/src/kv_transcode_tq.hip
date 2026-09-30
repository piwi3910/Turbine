// TurboQuant KV transcode (ABI v2.11 turbine_kv_transcode, formats TQ4 and
// TQ2) on ROCm: encode a batch of KV blocks from their BF16 L0 pages into
// TurboQuant records in a device buffer of coded slots, and decode them back
// (Phase 6b S-4, Task 8).
//
// `turbine_hip_tq`: the codecs tq4 (K 3 + 1 bits, V 4 bits, 144-byte records)
// and tq2 (K 1 + 1, V 2, 80 bytes) of crates/turbine-kv codec/turboquant,
// head_dim 128. An own kernel per the decision "P6b: TurboQuant transcode —
// provider evaluation (kernel reuse rule)": no upstream kernel implements the
// codec (vLLM's TurboQuant is Triton / FlyDSL and a different codec, SGLang
// and CK have no TurboQuant), and the one building block reused is the
// wave-shuffle Walsh-Hadamard transform of llama.cpp (ggml-cuda/fwht.cu, MIT,
// at feb9a3d6): 32 lanes hold one 128-vector, 4 elements a lane (element
// i * 32 + lane), butterfly spans 1 .. 16 across lanes, then 32 and 64 across
// registers — the codec's Sylvester order — with the 1 / sqrt(128) scale moved
// after the butterflies, where the codec applies it.
//
// Bit-exact by transcription: every step is the codec's own F32 or F64
// operation in its order, and contraction is off in this file (a fused
// multiply-add would round a dot product differently from the codec):
//   norm       sqrt of the F64 sum of squares in coordinate order, to F32
//              (stored BF16, round to nearest even);
//   rotate     y = fwht(s * x) * f32(1 / sqrt(128));
//   codes      nearest centroid of y * (f32(sqrt(128)) / norm) (0 when the
//              norm is not positive), midpoints (c[i] + c[i+1]) * 0.5 compared
//              with >, so a value on a midpoint takes the lower code;
//   K residual r = y - c[code] * (bf16(norm) / f32(sqrt(128))), its F64 norm
//              in BF16, and the sign bits of S * r (each row an F32 dot
//              product summed in column order; bit set when >= 0);
//   decode     (c[code] * k [+ r_hat]) through fwht, times
//              f32(1 / sqrt(128)) * s, to BF16; r_hat_j = (sum over rows i
//              in order of +-S[i][j]) * (f32(sqrt(pi / 2) / 128) * bf16(rnorm)).
// The signs, S and the codebooks come from d->tq_params, built on the host
// from the codec; nothing is regenerated here.
//
// One workgroup per (KV head, layer, block): 128 threads, the block's tokens
// in chunks of 32, staged in LDS. The page table goes up through the upload
// ring shared with turbine_hip_fp8 (kv_transcode.hpp).
#include "kv_transcode.hpp"

#include <string>

#include "bf16.hpp"

#pragma clang fp contract(off)

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kTqImpl = "turbine_hip_tq";
constexpr int kDim = 128;         // head_dim of the committed codebooks
constexpr int kThreads = 128;     // 4 groups of 32 lanes
constexpr int kGroup = 32;        // lanes holding one vector
constexpr int kPer = kDim / kGroup; // elements of a vector per lane
constexpr int kChunk = 32;        // tokens staged at once
constexpr int kPad = kDim + 1;    // LDS row stride (bank spread)
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

int record_bytes(int32_t format) {
  return format == TURBINE_KVFMT_TQ4 ? Record<3, 4>::kBytes
                                     : Record<1, 2>::kBytes;
}

int64_t tq_slot_bytes(const turbine_kv_transcode_desc *d) {
  return static_cast<int64_t>(d->layers) * d->num_kv_heads * d->block_tokens *
         record_bytes(d->format);
}

bool tq_supported(const turbine_kv_transcode_desc *d) {
  if (d == nullptr)
    return false;
  if (d->format != TURBINE_KVFMT_TQ4 && d->format != TURBINE_KVFMT_TQ2)
    return false;
  if (d->direction != TURBINE_KV_ENCODE && d->direction != TURBINE_KV_DECODE)
    return false;
  // BF16 pages only (FP8 pages, the codec's scaled read, are not built here).
  if (d->page_dtype != TURBINE_DTYPE_BF16 || d->head_dim != kDim)
    return false;
  if (d->num_blocks < 0 || d->layers <= 0 || d->block_tokens <= 0 ||
      d->num_kv_heads <= 0) {
    return false;
  }
  // Grid bounds (x = heads, y = layers, z = blocks) and 64-bit offsets.
  return d->num_kv_heads <= 65535 && d->layers <= 65535 &&
         static_cast<int64_t>(d->block_tokens) * d->num_kv_heads <=
             (int64_t{1} << 24);
}

std::string describe(const turbine_kv_transcode_desc *d) {
  return "format=" + std::to_string(d->format) +
         " direction=" + std::to_string(d->direction) +
         " page_dtype=" + std::to_string(d->page_dtype) +
         " blocks=" + std::to_string(d->num_blocks) +
         " layers=" + std::to_string(d->layers) +
         " block_tokens=" + std::to_string(d->block_tokens) +
         " kv_heads=" + std::to_string(d->num_kv_heads) +
         " head_dim=" + std::to_string(d->head_dim);
}

// The unnormalised Walsh-Hadamard transform of the vector 32 lanes hold
// (v[i] is element i * 32 + lane), in the codec's butterfly order.
__device__ inline void fwht(float (&v)[kPer], int lane) {
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

// The element offset of (kind, token, head, 0) in a page.
__device__ inline int64_t vec_offset(int kind, int token, int head, int tokens,
                                     int heads) {
  return ((static_cast<int64_t>(kind) * tokens + token) * heads + head) * kDim;
}

// Grid: x = KV head, y = layer, z = block. pages[b * layers + l] is layer l of
// block b; the slot of block b is coded + b * slot_bytes.
template <int KB, int VB>
__global__ void __launch_bounds__(kThreads)
    tq_encode_kernel(const uint16_t *const *pages, uint8_t *coded,
                     int64_t slot_bytes, int32_t layers, int32_t tokens,
                     int32_t heads, const float *tables, const float *cb_k,
                     const float *cb_v) {
  using R = Record<KB, VB>;
  __shared__ float xs[2][kChunk][kPad]; // x, then y - y_hat (K)
  __shared__ uint8_t codes[2][kChunk][kDim];
  __shared__ float norms[2][kChunk];
  __shared__ uint16_t norm_bits[2][kChunk];
  __shared__ uint16_t rnorm_bits[kChunk];
  __shared__ uint32_t qjl[kChunk][kDim / 32];
  __shared__ float cbk[1 << KB];
  __shared__ float cbv[1 << VB];

  const int tid = threadIdx.x;
  const int group = tid / kGroup;
  const int lane = tid % kGroup;
  const int head = blockIdx.x;
  const int layer = blockIdx.y;
  const int64_t b = blockIdx.z;
  const uint16_t *page = pages[b * layers + layer];
  const float *head_tables =
      tables + (static_cast<int64_t>(layer) * heads + head) * kHeadElems;
  const float *qjl_s = head_tables + 2 * kDim;
  uint8_t *records =
      coded + b * slot_bytes +
      (static_cast<int64_t>(layer) * heads + head) * tokens * R::kBytes;
  if (tid < (1 << KB))
    cbk[tid] = cb_k[tid];
  if (tid < (1 << VB))
    cbv[tid] = cb_v[tid];

  for (int t0 = 0; t0 < tokens; t0 += kChunk) {
    const int nt = min(kChunk, tokens - t0);
    // Load K and V of the chunk (coalesced along the head dimension).
    for (int i = tid; i < 2 * nt * kDim; i += kThreads) {
      const int kind = i / (nt * kDim);
      const int tok = (i / kDim) % nt;
      const int e = i % kDim;
      xs[kind][tok][e] = turbine_hip::bf16_to_f32(turbine_hip::Bf16{
          page[vec_offset(kind, t0 + tok, head, tokens, heads) + e]});
    }
    __syncthreads();
    // Norms: one thread per vector, coordinates in order.
    if (tid < 2 * kChunk && tid % kChunk < nt) {
      const int kind = tid / kChunk;
      const int tok = tid % kChunk;
      const float n = norm_of(xs[kind][tok]);
      norms[kind][tok] = n;
      norm_bits[kind][tok] = turbine_hip::f32_to_bf16(n).bits;
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
        y[i] = xs[kind][tok][e] * signs[e];
      }
      fwht(y, lane);
      const float n = norms[kind][tok];
      const float inv = n > 0.0f ? kSqrtD / n : 0.0f;
      const float k =
          turbine_hip::bf16_to_f32(turbine_hip::Bf16{norm_bits[kind][tok]}) /
          kSqrtD;
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        const int e = i * kGroup + lane;
        const float yi = y[i] * kInvSqrtD;
        const int code = kind == 0 ? nearest(cbk, 1 << KB, yi * inv)
                                   : nearest(cbv, 1 << VB, yi * inv);
        codes[kind][tok][e] = static_cast<uint8_t>(code);
        if (kind == 0)
          xs[0][tok][e] = yi - cbk[code] * k;
      }
    }
    __syncthreads();
    // Residual norms, then the sign bits of S * r: thread i owns row i of S.
    if (tid < nt) {
      rnorm_bits[tid] =
          turbine_hip::f32_to_bf16(norm_of(xs[0][tid])).bits;
    }
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
            acc[t] = acc[t] + s[q] * xs[0][t][j + q];
        }
      }
      for (int t = 0; t < nt; ++t) {
        const uint64_t mask = __ballot(acc[t] >= 0.0f);
        if (lane == 0) {
          // Rows group * 32 .. + 31 are bits of qjl bytes 4 * group .. + 3.
          const int shift = (threadIdx.x % warpSize) / kGroup * kGroup;
          qjl[t][group] = static_cast<uint32_t>(mask >> shift);
        }
      }
    }
    __syncthreads();
    // The chunk's records are contiguous: assemble them byte by byte.
    for (int i = tid; i < nt * R::kBytes; i += kThreads) {
      const int tok = i / R::kBytes;
      const int o = i % R::kBytes;
      uint8_t v = 0;
      if (o < R::kKNorm) {
        v = packed_byte<KB>(codes[0][tok], o);
      } else if (o < R::kQjl) {
        v = static_cast<uint8_t>(norm_bits[0][tok] >> (8 * (o - R::kKNorm)));
      } else if (o < R::kRNorm) {
        const int q = o - R::kQjl;
        v = static_cast<uint8_t>(qjl[tok][q / 4] >> (8 * (q % 4)));
      } else if (o < R::kVCodes) {
        v = static_cast<uint8_t>(rnorm_bits[tok] >> (8 * (o - R::kRNorm)));
      } else if (o < R::kVNorm) {
        v = packed_byte<VB>(codes[1][tok], o - R::kVCodes);
      } else if (o < R::kVNorm + 2) {
        v = static_cast<uint8_t>(norm_bits[1][tok] >> (8 * (o - R::kVNorm)));
      }
      records[static_cast<int64_t>(t0) * R::kBytes + i] = v;
    }
    __syncthreads();
  }
}

template <int KB, int VB>
__global__ void __launch_bounds__(kThreads)
    tq_decode_kernel(uint16_t *const *pages, const uint8_t *coded,
                     int64_t slot_bytes, int32_t layers, int32_t tokens,
                     int32_t heads, const float *tables, const float *cb_k,
                     const float *cb_v) {
  using R = Record<KB, VB>;
  __shared__ uint8_t recs[kChunk][R::kBytes];
  __shared__ float rhat[kChunk][kPad];
  __shared__ float cbk[1 << KB];
  __shared__ float cbv[1 << VB];

  const int tid = threadIdx.x;
  const int group = tid / kGroup;
  const int lane = tid % kGroup;
  const int head = blockIdx.x;
  const int layer = blockIdx.y;
  const int64_t b = blockIdx.z;
  uint16_t *page = pages[b * layers + layer];
  const float *head_tables =
      tables + (static_cast<int64_t>(layer) * heads + head) * kHeadElems;
  const float *qjl_s = head_tables + 2 * kDim;
  const uint8_t *records =
      coded + b * slot_bytes +
      (static_cast<int64_t>(layer) * heads + head) * tokens * R::kBytes;
  if (tid < (1 << KB))
    cbk[tid] = cb_k[tid];
  if (tid < (1 << VB))
    cbv[tid] = cb_v[tid];

  for (int t0 = 0; t0 < tokens; t0 += kChunk) {
    const int nt = min(kChunk, tokens - t0);
    for (int i = tid; i < nt * R::kBytes; i += kThreads)
      recs[i / R::kBytes][i % R::kBytes] =
          records[static_cast<int64_t>(t0) * R::kBytes + i];
    __syncthreads();
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
              (recs[t][R::kQjl + i / 8] >> (i % 8)) & 1 ? 1.0f : -1.0f;
          acc[t] = acc[t] + z * a;
        }
      }
      for (int t = 0; t < nt; ++t) {
        const uint16_t rn = static_cast<uint16_t>(
            recs[t][R::kRNorm] | (recs[t][R::kRNorm + 1] << 8));
        const float k =
            kQjl * turbine_hip::bf16_to_f32(turbine_hip::Bf16{rn});
        rhat[t][tid] = acc[t] * k;
      }
    }
    __syncthreads();
    for (int vec = group; vec < 2 * nt; vec += kThreads / kGroup) {
      const int kind = vec / nt;
      const int tok = vec % nt;
      const uint8_t *rec = recs[tok];
      const int at = kind == 0 ? R::kKNorm : R::kVNorm;
      const uint16_t nb = static_cast<uint16_t>(rec[at] | (rec[at + 1] << 8));
      const float k = turbine_hip::bf16_to_f32(turbine_hip::Bf16{nb}) / kSqrtD;
      float y[kPer];
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        const int e = i * kGroup + lane;
        if (kind == 0) {
          y[i] = cbk[unpack_code<KB>(rec + R::kKCodes, e)] * k;
          y[i] = y[i] + rhat[tok][e];
        } else {
          y[i] = cbv[unpack_code<VB>(rec + R::kVCodes, e)] * k;
        }
      }
      fwht(y, lane);
      const float *signs = head_tables + kind * kDim;
      uint16_t *dst = page + vec_offset(kind, t0 + tok, head, tokens, heads);
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        const int e = i * kGroup + lane;
        dst[e] = turbine_hip::f32_to_bf16(y[i] * (kInvSqrtD * signs[e])).bits;
      }
    }
    __syncthreads();
  }
}

template <int KB, int VB>
void launch(turbine_ctx *ctx, const turbine_kv_transcode_desc *d, void **table,
            const float *const *codebooks) {
  const dim3 grid(static_cast<uint32_t>(d->num_kv_heads),
                  static_cast<uint32_t>(d->layers),
                  static_cast<uint32_t>(d->num_blocks));
  const float *tables = d->tq_params->tables;
  const float *cb_k = codebooks[KB - 1];
  const float *cb_v = codebooks[VB - 1];
  if (d->direction == TURBINE_KV_ENCODE) {
    hipLaunchKernelGGL((tq_encode_kernel<KB, VB>), grid, dim3(kThreads), 0,
                       ctx->stream,
                       reinterpret_cast<const uint16_t *const *>(table),
                       static_cast<uint8_t *>(d->coded), d->coded_block_bytes,
                       d->layers, d->block_tokens, d->num_kv_heads, tables,
                       cb_k, cb_v);
  } else {
    hipLaunchKernelGGL((tq_decode_kernel<KB, VB>), grid, dim3(kThreads), 0,
                       ctx->stream, reinterpret_cast<uint16_t *const *>(table),
                       static_cast<const uint8_t *>(d->coded),
                       d->coded_block_bytes, d->layers, d->block_tokens,
                       d->num_kv_heads, tables, cb_k, cb_v);
  }
}

} // namespace

namespace turbine_hip {

bool kv_transcode_tq_supports(const turbine_kv_transcode_desc *d) {
  return tq_supported(d);
}

int32_t kv_transcode_tq_run(turbine_ctx *ctx,
                            const turbine_kv_transcode_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: descriptor is NULL");
  }
  if (!tq_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string("turbine_kv_transcode (") + kTqImpl +
                    "): unsupported configuration " + describe(d));
  }
  if (d->num_blocks == 0)
    return TURBINE_OK;
  if (d->pages == nullptr || d->coded == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: NULL page table or coded buffer");
  }
  const turbine_tq_params *tq = d->tq_params;
  if (tq == nullptr || tq->tables == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string("turbine_kv_transcode (") + kTqImpl +
                    "): TurboQuant needs tq_params with its tables");
  }
  for (int bits = 1; bits <= 4; ++bits) {
    if (tq->codebooks[bits - 1] == nullptr) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_kv_transcode: NULL " + std::to_string(bits) +
                      "-bit codebook");
    }
  }
  if (tq->seed != d->seed) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: tq_params are of seed " +
                    std::to_string(tq->seed) + ", the call of seed " +
                    std::to_string(d->seed));
  }
  // The tables' S rows are read as float4.
  if (reinterpret_cast<uintptr_t>(tq->tables) % 16 != 0) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: TurboQuant tables are not 16-byte "
                "aligned");
  }
  if (d->coded_block_bytes != tq_slot_bytes(d)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string("turbine_kv_transcode (") + kTqImpl +
                    "): coded_block_bytes " +
                    std::to_string(d->coded_block_bytes) + " is not the " +
                    std::to_string(tq_slot_bytes(d)) + " bytes of a slot (" +
                    describe(d) + ")");
  }
  for (int64_t i = 0; i < static_cast<int64_t>(d->num_blocks) * d->layers;
       ++i) {
    if (d->pages[i] == nullptr)
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_kv_transcode: NULL page " + std::to_string(i));
  }
  // grid.z is at most 65535 blocks.
  if (d->num_blocks > 65535) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_kv_transcode: more than 65535 blocks in one call");
  }
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_kv_transcode");
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  void **table = nullptr;
  if (int32_t rc = kv_transcode_upload_pages(ctx, d, &table);
      rc != TURBINE_OK)
    return rc;
  if (d->format == TURBINE_KVFMT_TQ4)
    launch<3, 4>(ctx, d, table, tq->codebooks);
  else
    launch<1, 2>(ctx, d, table, tq->codebooks);
  return check_hip(ctx, hipGetLastError(), "turbine_kv_transcode launch");
}

} // namespace turbine_hip
