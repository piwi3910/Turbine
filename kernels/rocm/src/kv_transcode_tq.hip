// TurboQuant KV transcode (ABI v2.11 turbine_kv_transcode, formats TQ4 and
// TQ2) on ROCm: encode a batch of KV blocks from their BF16 L0 pages into
// TurboQuant records in a device buffer of coded slots, and decode them back
// (Phase 6b S-4, Task 8).
//
// `turbine_hip_tq`: the codecs tq4 (K and V 4 bits, 144-byte records) and tq2
// (K and V 2 bits, 80 bytes) of crates/turbine-kv codec/turboquant, head_dim
// 128; K and V are both TurboQuant_mse (decision "6b Task 9: TurboQuant K
// quantizer" A). An own kernel per the decision "P6b: TurboQuant transcode —
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
//   decode     c[code] * (bf16(norm) / f32(sqrt(128))) through fwht, times
//              f32(1 / sqrt(128)) * s, to BF16.
// The signs and the codebooks come from d->tq_params, built on the host from
// the codec; nothing is regenerated here.
//
// One workgroup per (KV head, layer, block): 128 threads, the block's tokens
// in chunks of 32, staged in LDS. The page table goes up through the upload
// ring shared with turbine_hip_fp8 (kv_transcode.hpp).
#include "kv_transcode.hpp"

#include <string>

#include "bf16.hpp"
#include "tq_device.hpp"

#pragma clang fp contract(off)

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kTqImpl = "turbine_hip_tq";
using namespace turbine_hip::tq;

int record_bytes(int32_t format) {
  return format == TURBINE_KVFMT_TQ4 ? Tq4::kBytes : Tq2::kBytes;
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
  __shared__ EncodeLds<KB, VB> lds;

  const int tid = threadIdx.x;
  const int head = blockIdx.x;
  const int layer = blockIdx.y;
  const int64_t b = blockIdx.z;
  const uint16_t *page = pages[b * layers + layer];
  const float *head_tables =
      tables + (static_cast<int64_t>(layer) * heads + head) * kHeadElems;
  uint8_t *records =
      coded + b * slot_bytes +
      (static_cast<int64_t>(layer) * heads + head) * tokens * R::kBytes;
  if (tid < (1 << KB))
    lds.cbk[tid] = cb_k[tid];
  if (tid < (1 << VB))
    lds.cbv[tid] = cb_v[tid];
  const bool vec_loads = (reinterpret_cast<uintptr_t>(page) & 15u) == 0;

  for (int t0 = 0; t0 < tokens; t0 += kChunk) {
    const int nt = min(kChunk, tokens - t0);
    // Load K and V of the chunk (coalesced along the head dimension): 16
    // bytes (8 elements) a thread when the page is 16-byte aligned (every
    // vector starts at a multiple of 256 bytes in it), else one element.
    if (vec_loads) {
      constexpr int kPerLoad = 8;
      for (int i = tid; i < 2 * nt * (kDim / kPerLoad); i += kThreads) {
        const int kind = i / (nt * (kDim / kPerLoad));
        const int tok = (i / (kDim / kPerLoad)) % nt;
        const int e = (i % (kDim / kPerLoad)) * kPerLoad;
        const uint4 w = *reinterpret_cast<const uint4 *>(
            page + vec_offset(kind, t0 + tok, head, tokens, heads) + e);
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          lds.xs[kind][tok][e + 2 * k] = turbine_hip::bf16_to_f32(
              turbine_hip::Bf16{static_cast<uint16_t>(words[k] & 0xffffu)});
          lds.xs[kind][tok][e + 2 * k + 1] = turbine_hip::bf16_to_f32(
              turbine_hip::Bf16{static_cast<uint16_t>(words[k] >> 16)});
        }
      }
    } else {
      for (int i = tid; i < 2 * nt * kDim; i += kThreads) {
        const int kind = i / (nt * kDim);
        const int tok = (i / kDim) % nt;
        const int e = i % kDim;
        lds.xs[kind][tok][e] = turbine_hip::bf16_to_f32(turbine_hip::Bf16{
            page[vec_offset(kind, t0 + tok, head, tokens, heads) + e]});
      }
    }
    __syncthreads();
    // The chunk's records are contiguous.
    encode_chunk<KB, VB>(lds, nt, head_tables,
                         [&](int tok, int o, uint32_t word) {
                           store_word(records +
                                          static_cast<int64_t>(t0 + tok) *
                                              R::kBytes +
                                          o,
                                      word);
                         });
  }
}

template <int KB, int VB>
__global__ void __launch_bounds__(kThreads)
    tq_decode_kernel(uint16_t *const *pages, const uint8_t *coded,
                     int64_t slot_bytes, int32_t layers, int32_t tokens,
                     int32_t heads, const float *tables, const float *cb_k,
                     const float *cb_v) {
  using R = Record<KB, VB>;
  __shared__ DecodeLds<KB, VB> lds;

  const int tid = threadIdx.x;
  const int head = blockIdx.x;
  const int layer = blockIdx.y;
  const int64_t b = blockIdx.z;
  uint16_t *page = pages[b * layers + layer];
  const float *head_tables =
      tables + (static_cast<int64_t>(layer) * heads + head) * kHeadElems;
  const uint8_t *records =
      coded + b * slot_bytes +
      (static_cast<int64_t>(layer) * heads + head) * tokens * R::kBytes;
  if (tid < (1 << KB))
    lds.cbk[tid] = cb_k[tid];
  if (tid < (1 << VB))
    lds.cbv[tid] = cb_v[tid];

  for (int t0 = 0; t0 < tokens; t0 += kChunk) {
    const int nt = min(kChunk, tokens - t0);
    for (int i = tid; i < nt * R::kBytes; i += kThreads)
      lds.recs[i / R::kBytes][i % R::kBytes] =
          records[static_cast<int64_t>(t0) * R::kBytes + i];
    __syncthreads();
    decode_chunk<KB, VB>(
        lds, nt, head_tables, [&](int kind, int tok, int e, uint16_t bits) {
          page[vec_offset(kind, t0 + tok, head, tokens, heads) + e] = bits;
        });
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
    launch<4, 4>(ctx, d, table, tq->codebooks);
  else
    launch<2, 2>(ctx, d, table, tq->codebooks);
  return check_hip(ctx, hipGetLastError(), "turbine_kv_transcode launch");
}

} // namespace turbine_hip
