// Turbine HIP paged-KV kernels: the append of k_new/v_new into their page
// slots (used by every paged attention call) and the ragged paged attention
// used when Composable Kernel has no instance for the page size (CK needs
// block_tokens % 128 == 0; Turbine's default page is 16 tokens).
//
// Pool layout per layer: [num_blocks, 2, block_tokens, num_kv_heads, 128];
// token p of sequence s lives in block block_table[s][p / block_tokens], slot
// p % block_tokens, K half first. Sequence s owns query rows
// [q_indptr[s], q_indptr[s+1]); its new tokens sit at positions
// [kv_lens[s] - q_len, kv_lens[s]), query i at kv_lens[s] - q_len + i.
//
// Numerics follow the cpu-reference paged attention (math::attention): scores
// q.k accumulated in f32 over the head dimension in index order and scaled;
// the exact row maximum over the visible keys (first pass); exp(s - max) in f32
// summed unrounded in key order; P rounded to BF16 for the P.V product,
// accumulated in f32 in key order; the row divided by the sum and rounded to
// BF16 at the end.
//
// FP8 pages (dtype TURBINE_DTYPE_F8E4M3, Phase 6a S-13): one byte per element,
// written as e4m3(x / scale) and read as bf16(e4m3 * scale) with the layer's
// k_scale / v_scale (fp8.hpp), so every kernel below computes on the same BF16
// K/V values BF16 pages would hold. Besides the FP8 variants of the append and
// of the Turbine kernel, this file has the FP8 decode kernel (one workgroup per
// sequence and KV head, every query head of the group at once, the K/V rows
// read once as bytes) and the page dequantization that stages a batch's FP8
// pages as BF16 pages for Composable Kernel's pagedkv prefill.
#include <hip/hip_fp16.h>

#include <type_traits>

#include "bf16.hpp"
#include "fp8.hpp"
#include "paged_fp8.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::f32_to_fp8_e4m3;
using turbine_hip::fp8_dequant;
using turbine_hip::round_bf16;

namespace {

constexpr int kHeadDim = 128;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kThreads = 256;
// Query rows per block: the q heads of one GQA group times a tile of query
// positions.
constexpr int kMaxRows = 64;
// Keys staged in LDS per step.
constexpr int kKeyChunk = 32;
// K/V rows in LDS are padded by one dword so the lanes of a wave reading the
// same column of different keys hit different banks.
constexpr int kKvPitch = kHeadDim + 2;
// Output elements (row, dim) per thread: thread t owns dim t % 128 of rows
// t / 128 + 2 * n.
constexpr int kRowsPerPass = kThreads / kHeadDim;
constexpr int kOutPerThread = kMaxRows / kRowsPerPass;

static_assert(kThreads % kHeadDim == 0, "thread count must cover whole rows");

// ---- append ----

// One block per new token; the sequence is found by binary search in
// q_indptr. Tokens whose page is missing from the block table or outside the
// pool are skipped (the caller validated the table; this keeps a bad table
// from writing outside the pool).
__global__ void paged_append_kernel(const Bf16 *k_new, const Bf16 *v_new,
                                    Bf16 *kv_layer, const int32_t *block_table,
                                    const int32_t *q_indptr,
                                    const int32_t *kv_lens, int32_t num_seqs,
                                    int32_t max_blocks, int32_t num_blocks,
                                    int32_t block_tokens, int32_t kv_heads,
                                    int64_t new_stride) {
  const int32_t row = blockIdx.x;
  // Largest s with q_indptr[s] <= row.
  int32_t lo = 0;
  int32_t hi = num_seqs - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) / 2;
    if (q_indptr[mid] <= row)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int32_t s = lo;
  const int32_t row0 = q_indptr[s];
  const int32_t q_len = q_indptr[s + 1] - row0;
  const int32_t i = row - row0;
  if (i < 0 || i >= q_len)
    return;
  const int32_t pos = kv_lens[s] - q_len + i;
  if (pos < 0)
    return;
  const int32_t page = pos / block_tokens;
  if (page >= max_blocks)
    return;
  const int32_t block =
      block_table[static_cast<int64_t>(s) * max_blocks + page];
  if (block < 0 || block >= num_blocks)
    return;
  const int32_t slot = pos % block_tokens;
  const int64_t token_elems = static_cast<int64_t>(kv_heads) * kHeadDim;
  Bf16 *k_dst =
      kv_layer +
      (static_cast<int64_t>(block) * 2 * block_tokens + slot) * token_elems;
  Bf16 *v_dst = k_dst + static_cast<int64_t>(block_tokens) * token_elems;
  const Bf16 *k_src = k_new + static_cast<int64_t>(row) * new_stride;
  const Bf16 *v_src = v_new + static_cast<int64_t>(row) * new_stride;
  for (int64_t e = threadIdx.x; e < token_elems; e += blockDim.x) {
    k_dst[e] = k_src[e];
    v_dst[e] = v_src[e];
  }
}

// The FP8 append: the same slots, each element written as
// e4m3(x / scale) with K's or V's scale.
__global__ void paged_append_fp8_kernel(
    const Bf16 *k_new, const Bf16 *v_new, uint8_t *kv_layer,
    const int32_t *block_table, const int32_t *q_indptr, const int32_t *kv_lens,
    int32_t num_seqs, int32_t max_blocks, int32_t num_blocks,
    int32_t block_tokens, int32_t kv_heads, int64_t new_stride, float k_scale,
    float v_scale) {
  const int32_t row = blockIdx.x;
  int32_t lo = 0;
  int32_t hi = num_seqs - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) / 2;
    if (q_indptr[mid] <= row)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int32_t s = lo;
  const int32_t row0 = q_indptr[s];
  const int32_t q_len = q_indptr[s + 1] - row0;
  const int32_t i = row - row0;
  if (i < 0 || i >= q_len)
    return;
  const int32_t pos = kv_lens[s] - q_len + i;
  if (pos < 0)
    return;
  const int32_t page = pos / block_tokens;
  if (page >= max_blocks)
    return;
  const int32_t block =
      block_table[static_cast<int64_t>(s) * max_blocks + page];
  if (block < 0 || block >= num_blocks)
    return;
  const int32_t slot = pos % block_tokens;
  const int64_t token_elems = static_cast<int64_t>(kv_heads) * kHeadDim;
  uint8_t *k_dst =
      kv_layer +
      (static_cast<int64_t>(block) * 2 * block_tokens + slot) * token_elems;
  uint8_t *v_dst = k_dst + static_cast<int64_t>(block_tokens) * token_elems;
  const Bf16 *k_src = k_new + static_cast<int64_t>(row) * new_stride;
  const Bf16 *v_src = v_new + static_cast<int64_t>(row) * new_stride;
  for (int64_t e = threadIdx.x; e < token_elems; e += blockDim.x) {
    k_dst[e] = f32_to_fp8_e4m3(bf16_to_f32(k_src[e]) / k_scale);
    v_dst[e] = f32_to_fp8_e4m3(bf16_to_f32(v_src[e]) / v_scale);
  }
}

// How the attention kernels read a page element as BF16.
struct Bf16Pages {
  using Elem = Bf16;
  __device__ static Bf16 k(const Bf16 *p, float) { return *p; }
  __device__ static Bf16 v(const Bf16 *p, float) { return *p; }
};
struct Fp8Pages {
  using Elem = uint8_t;
  __device__ static Bf16 k(const uint8_t *p, float scale) {
    return f32_to_bf16(fp8_dequant(*p, scale));
  }
  __device__ static Bf16 v(const uint8_t *p, float scale) {
    return f32_to_bf16(fp8_dequant(*p, scale));
  }
};

// ---- attention ----

// Grid: (query tiles of the longest sequence, sequences, KV heads). A block
// handles rows r in [0, group * q_tile): q head kv_head * group + r / q_tile,
// query tile * q_tile + r % q_tile of sequence blockIdx.y.
template <typename Pages>
__global__ void __launch_bounds__(kThreads) paged_attention_kernel(
    const Bf16 *q, const typename Pages::Elem *kv_layer, Bf16 *out,
    const int32_t *block_table, const int32_t *q_indptr, const int32_t *kv_lens,
    int32_t max_blocks, int32_t num_blocks, int32_t block_tokens,
    int32_t kv_heads, int32_t group, int32_t q_tile, int64_t q_stride,
    int64_t out_stride, float scale, int32_t causal, float k_scale,
    float v_scale) {
  __shared__ Bf16 q_s[kMaxRows][kHeadDim];
  __shared__ Bf16 k_s[kKeyChunk][kKvPitch];
  __shared__ Bf16 v_s[kKeyChunk][kKvPitch];
  __shared__ float s_s[kMaxRows][kKeyChunk];
  __shared__ float max_s[kMaxRows];
  __shared__ float sum_s[kMaxRows];
  // Element offsets of each staged key's K row (-1: no page).
  __shared__ int64_t koff_s[kKeyChunk];

  const int tid = threadIdx.x;
  const int32_t tile = blockIdx.x;
  const int32_t seq = blockIdx.y;
  const int32_t kv_head = blockIdx.z;
  const int32_t row0 = q_indptr[seq];
  const int32_t q_len = q_indptr[seq + 1] - row0;
  const int32_t kv_len = kv_lens[seq];
  const int32_t first = tile * q_tile;
  // Uniform across the block: return before any barrier.
  if (first >= q_len)
    return;
  const int32_t q_start = kv_len - q_len;
  const int32_t rows = group * q_tile;
  const int32_t last = min(q_len, first + q_tile) - 1;
  const int32_t key_end = causal != 0 ? q_start + last + 1 : kv_len;
  const int64_t token_elems = static_cast<int64_t>(kv_heads) * kHeadDim;
  const int64_t v_half = static_cast<int64_t>(block_tokens) * token_elems;
  const int32_t *table = block_table + static_cast<int64_t>(seq) * max_blocks;

  // Row r's query index, or -1 when r is past the sequence's queries.
  auto query_of = [&](int32_t r) -> int32_t {
    if (r >= rows)
      return -1;
    const int32_t i = first + r % q_tile;
    return i < q_len ? i : -1;
  };
  // Keys row r (query i) may see: [0, limit).
  auto limit_of = [&](int32_t i) -> int32_t {
    return causal != 0 ? q_start + i + 1 : kv_len;
  };

  for (int e = tid; e < kMaxRows * kHeadDim; e += kThreads) {
    const int32_t r = e / kHeadDim;
    const int32_t d = e % kHeadDim;
    const int32_t i = query_of(r);
    Bf16 v{0};
    if (i >= 0) {
      const int32_t h = kv_head * group + r / q_tile;
      v = q[static_cast<int64_t>(row0 + i) * q_stride +
            static_cast<int64_t>(h) * kHeadDim + d];
    }
    q_s[r][d] = v;
  }
  if (tid < kMaxRows) {
    max_s[tid] = kNegInf;
    sum_s[tid] = 0.0f;
  }

  // Stages keys [c0, c0 + kKeyChunk) of K (and V when with_v) into LDS.
  auto stage = [&](int32_t c0, bool with_v) {
    if (tid < kKeyChunk) {
      const int32_t key = c0 + tid;
      int64_t off = -1;
      if (key < key_end) {
        const int32_t page = key / block_tokens;
        const int32_t block = page < max_blocks ? table[page] : -1;
        if (block >= 0 && block < num_blocks) {
          off = (static_cast<int64_t>(block) * 2 * block_tokens +
                 key % block_tokens) *
                    token_elems +
                static_cast<int64_t>(kv_head) * kHeadDim;
        }
      }
      koff_s[tid] = off;
    }
    __syncthreads();
    for (int e = tid; e < kKeyChunk * kHeadDim; e += kThreads) {
      const int j = e / kHeadDim;
      const int d = e % kHeadDim;
      const int64_t off = koff_s[j];
      k_s[j][d] = off >= 0 ? Pages::k(kv_layer + off + d, k_scale) : Bf16{0};
      if (with_v)
        v_s[j][d] = off >= 0 ? Pages::v(kv_layer + off + v_half + d, v_scale)
                             : Bf16{0};
    }
    __syncthreads();
  };

  // Scaled scores of the staged keys into s_s; -inf where not visible.
  auto score = [&](int32_t c0) {
    for (int p = tid; p < kMaxRows * kKeyChunk; p += kThreads) {
      const int32_t r = p / kKeyChunk;
      const int32_t j = p % kKeyChunk;
      const int32_t i = query_of(r);
      const int32_t key = c0 + j;
      float s = kNegInf;
      if (i >= 0 && key < limit_of(i)) {
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d)
          dot += bf16_to_f32(q_s[r][d]) * bf16_to_f32(k_s[j][d]);
        s = dot * scale;
      }
      s_s[r][j] = s;
    }
    __syncthreads();
  };

  // Pass 1: the exact row maxima.
  for (int32_t c0 = 0; c0 < key_end; c0 += kKeyChunk) {
    stage(c0, false);
    score(c0);
    if (tid < kMaxRows) {
      float m = max_s[tid];
      for (int j = 0; j < kKeyChunk; ++j)
        m = fmaxf(m, s_s[tid][j]);
      max_s[tid] = m;
    }
    __syncthreads();
  }

  // Pass 2: exp, sum and P.V in key order.
  float acc[kOutPerThread];
  for (int n = 0; n < kOutPerThread; ++n)
    acc[n] = 0.0f;
  const int dim = tid % kHeadDim;
  const int row_base = tid / kHeadDim;
  for (int32_t c0 = 0; c0 < key_end; c0 += kKeyChunk) {
    stage(c0, true);
    score(c0);
    for (int p = tid; p < kMaxRows * kKeyChunk; p += kThreads) {
      const int32_t r = p / kKeyChunk;
      const int32_t j = p % kKeyChunk;
      const float s = s_s[r][j];
      s_s[r][j] = s == kNegInf ? 0.0f : expf(s - max_s[r]);
    }
    __syncthreads();
    if (tid < kMaxRows) {
      float sum = sum_s[tid];
      for (int j = 0; j < kKeyChunk; ++j)
        sum += s_s[tid][j];
      sum_s[tid] = sum;
    }
    for (int n = 0; n < kOutPerThread; ++n) {
      const int r = row_base + kRowsPerPass * n;
      if (r < rows) {
        float a = acc[n];
        for (int j = 0; j < kKeyChunk; ++j)
          a += round_bf16(s_s[r][j]) * bf16_to_f32(v_s[j][dim]);
        acc[n] = a;
      }
    }
    __syncthreads();
  }

  for (int n = 0; n < kOutPerThread; ++n) {
    const int r = row_base + kRowsPerPass * n;
    const int32_t i = query_of(r);
    if (i < 0)
      continue;
    const int32_t h = kv_head * group + r / q_tile;
    out[static_cast<int64_t>(row0 + i) * out_stride +
        static_cast<int64_t>(h) * kHeadDim + dim] =
        f32_to_bf16(acc[n] / sum_s[r]);
  }
}

// ---- FP8 decode ----

// Lanes of a wave the decode kernel is written for; on a wider wave each half
// works as its own 32 lanes (the shuffles below stay within 32 lanes and each
// half has its own LDS rows).
constexpr int kDecodeLanes = 32;
constexpr int kDecodeWaves = kThreads / kDecodeLanes;
// Head dimensions per lane in the P.V product.
constexpr int kDimsPerLane = kHeadDim / kDecodeLanes;
static_assert(kDimsPerLane == 4, "the P.V loop loads 4 page bytes per lane");

__device__ inline float lane_max(float v) {
#pragma unroll
  for (int m = kDecodeLanes / 2; m > 0; m /= 2)
    v = fmaxf(v, __shfl_xor(v, m, kDecodeLanes));
  return v;
}

__device__ inline float lane_sum(float v) {
#pragma unroll
  for (int m = kDecodeLanes / 2; m > 0; m /= 2)
    v += __shfl_xor(v, m, kDecodeLanes);
  return v;
}

// Orders a wave's LDS writes before its own lanes read them.
__device__ inline void wave_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Two page bytes (bits 0-7 and 8-15 of b2) dequantized: the e4m3 bits
// shifted into an f16 are the value / 256 exactly (subnormals included), so
// value * scale = f16 * (256 * scale) with one rounding, as fp8_dequant; the
// NaN code 0x7f stays NaN. Rounded to BF16 unless UnitScale (every e4m3
// value is exact in BF16).
template <bool UnitScale>
__device__ inline void dequant2(uint32_t b2, float scale256, float &lo,
                                float &hi) {
  const uint32_t h =
      ((b2 & 0x80u) << 8) | ((b2 & 0x7fu) << 7) | ((b2 & 0x8000u) << 16) |
      ((b2 & 0x7f00u) << 15);
  const __half2 v = *reinterpret_cast<const __half2 *>(&h);
  const float2 f = __half22float2(v);
  lo = (b2 & 0x7fu) == 0x7fu ? __builtin_nanf("") : f.x * scale256;
  hi = (b2 & 0x7f00u) == 0x7f00u ? __builtin_nanf("") : f.y * scale256;
  if (!UnitScale) {
    lo = round_bf16(lo);
    hi = round_bf16(hi);
  }
}

// LDS of the decode kernel with G query heads per KV head: the loop's Q, P
// and V offsets, then (after a barrier) the waves' partial outputs.
template <int G> struct DecodeLds {
  struct Loop {
    float q[G][kHeadDim];
    float p[kDecodeWaves][G][kDecodeLanes];
    int64_t voff[kDecodeWaves][kDecodeLanes];
  };
  union {
    Loop loop;
    float o[kDecodeWaves][G][kHeadDim];
  };
  float max[kDecodeWaves][G];
  float sum[kDecodeWaves][G];
};

// Paged decode over FP8 pages: grid (sequences, KV heads), kThreads threads,
// G >= the group's query heads (the unused ones compute on zero queries and
// are not written). Every sequence has one query (row q_indptr[s]) that sees
// all kv_lens[s] keys. The workgroup computes the group's query heads
// together, so each K/V row is read once for all of them: chunks of 32 keys
// go round the waves; in a chunk each lane scores one key for every head (its
// K row read as 8 x 16 bytes), the wave's running maximum moves to the chunk's
// (rescaling its sums and P.V by exp(old - new)), P = exp(s - max) is rounded
// to BF16, and each lane accumulates 4 head dimensions of P.V over the chunk's
// keys (V rows read as 32 x 4 bytes, coalesced). The waves are combined in
// wave order against the overall maximum, the row divided by the sum and
// rounded to BF16 last. As in CK's FMHA (online softmax), P is rounded
// against a running maximum, so rows match the cpu-reference within the
// BF16 tolerance of the other paged kernels, not bit for bit.
template <int G, bool UnitScale>
__global__ void __launch_bounds__(kThreads) paged_decode_fp8_kernel(
    const Bf16 *q, const uint8_t *kv_layer, Bf16 *out,
    const int32_t *block_table, const int32_t *q_indptr, const int32_t *kv_lens,
    int32_t max_blocks, int32_t num_blocks, int32_t block_tokens,
    int32_t kv_heads, int32_t group, int64_t q_stride, int64_t out_stride,
    float scale, float k_scale, float v_scale) {
  __shared__ DecodeLds<G> lds;

  const int tid = threadIdx.x;
  const int lane = tid % kDecodeLanes;
  const int wave = tid / kDecodeLanes;
  const int32_t seq = blockIdx.x;
  const int32_t kv_head = blockIdx.y;
  const int32_t row = q_indptr[seq];
  // Only sequences with one query row (a prefill call hands its decode rows
  // here too). Uniform across the block: return before any barrier.
  if (q_indptr[seq + 1] - row != 1)
    return;
  const int32_t kv_len = kv_lens[seq];
  const int64_t token_elems = static_cast<int64_t>(kv_heads) * kHeadDim;
  const int64_t v_half = static_cast<int64_t>(block_tokens) * token_elems;
  const int32_t *table = block_table + static_cast<int64_t>(seq) * max_blocks;
  const float k256 = 256.0f * k_scale;
  const float v256 = 256.0f * v_scale;

  for (int e = tid; e < G * kHeadDim; e += kThreads) {
    const int g = e / kHeadDim;
    const int d = e % kHeadDim;
    float v = 0.0f;
    if (g < group) {
      const int32_t h = kv_head * group + g;
      v = bf16_to_f32(q[static_cast<int64_t>(row) * q_stride +
                        static_cast<int64_t>(h) * kHeadDim + d]);
    }
    lds.loop.q[g][d] = v;
  }
  __syncthreads();

  float m[G];
  float sum[G];
  float acc[G][kDimsPerLane];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = kNegInf;
    sum[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i)
      acc[g][i] = 0.0f;
  }

  for (int32_t c0 = wave * kDecodeLanes; c0 < kv_len;
       c0 += kDecodeWaves * kDecodeLanes) {
    const int32_t key = c0 + lane;
    int64_t off = -1;
    if (key < kv_len) {
      const int32_t page = key / block_tokens;
      const int32_t block = page < max_blocks ? table[page] : -1;
      if (block >= 0 && block < num_blocks)
        off = (static_cast<int64_t>(block) * 2 * block_tokens +
               key % block_tokens) *
                  token_elems +
              static_cast<int64_t>(kv_head) * kHeadDim;
    }
    float s[G];
#pragma unroll
    for (int g = 0; g < G; ++g)
      s[g] = 0.0f;
    if (off >= 0) {
      const uint4 *row16 = reinterpret_cast<const uint4 *>(kv_layer + off);
#pragma unroll
      for (int c = 0; c < kHeadDim / 16; ++c) {
        const uint4 w = row16[c];
        float k[16];
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
          dequant2<UnitScale>(words[wd] & 0xffffu, k256, k[4 * wd],
                              k[4 * wd + 1]);
          dequant2<UnitScale>(words[wd] >> 16, k256, k[4 * wd + 2],
                              k[4 * wd + 3]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const float4 *qv =
              reinterpret_cast<const float4 *>(&lds.loop.q[g][c * 16]);
          float dot = s[g];
#pragma unroll
          for (int b4 = 0; b4 < 4; ++b4) {
            const float4 qq = qv[b4];
            dot += qq.x * k[4 * b4];
            dot += qq.y * k[4 * b4 + 1];
            dot += qq.z * k[4 * b4 + 2];
            dot += qq.w * k[4 * b4 + 3];
          }
          s[g] = dot;
        }
      }
    }
    float p[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float sg = off >= 0 ? s[g] * scale : kNegInf;
      const float mn = fmaxf(m[g], lane_max(sg));
      // mn is -inf only while the wave has seen no key; nothing to rescale.
      const float factor = mn == kNegInf ? 1.0f : expf(m[g] - mn);
      m[g] = mn;
      sum[g] *= factor;
#pragma unroll
      for (int i = 0; i < kDimsPerLane; ++i)
        acc[g][i] *= factor;
      p[g] = off >= 0 ? expf(sg - mn) : 0.0f;
      sum[g] += p[g];
      lds.loop.p[wave][g][lane] = round_bf16(p[g]);
    }
    lds.loop.voff[wave][lane] = off < 0 ? -1 : off + v_half;
    wave_sync();
    const int32_t keys = min(kDecodeLanes, kv_len - c0);
    for (int j = 0; j < keys; ++j) {
      const int64_t voff = lds.loop.voff[wave][j];
      if (voff < 0)
        continue;
      const uint32_t w4 = *reinterpret_cast<const uint32_t *>(
          kv_layer + voff + lane * kDimsPerLane);
      float v[kDimsPerLane];
      dequant2<UnitScale>(w4 & 0xffffu, v256, v[0], v[1]);
      dequant2<UnitScale>(w4 >> 16, v256, v[2], v[3]);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float pg = lds.loop.p[wave][g][j];
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i)
          acc[g][i] += pg * v[i];
      }
    }
    wave_sync();
  }
  float wave_sums[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    wave_sums[g] = lane_sum(sum[g]);
  // Every wave is done with the loop's LDS before the partial outputs reuse it.
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (lane == 0) {
      lds.max[wave][g] = m[g];
      lds.sum[wave][g] = wave_sums[g];
    }
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i)
      lds.o[wave][g][lane * kDimsPerLane + i] = acc[g][i];
  }
  __syncthreads();
  for (int e = tid; e < group * kHeadDim; e += kThreads) {
    const int g = e / kHeadDim;
    const int d = e % kHeadDim;
    float mx = kNegInf;
    for (int w = 0; w < kDecodeWaves; ++w)
      mx = fmaxf(mx, lds.max[w][g]);
    float o = 0.0f;
    float total = 0.0f;
    for (int w = 0; w < kDecodeWaves; ++w) {
      // A wave that saw no key has max -inf and adds nothing.
      if (lds.max[w][g] == kNegInf)
        continue;
      const float f = expf(lds.max[w][g] - mx);
      o += lds.o[w][g][d] * f;
      total += lds.sum[w][g] * f;
    }
    const int32_t h = kv_head * group + g;
    out[static_cast<int64_t>(row) * out_stride +
        static_cast<int64_t>(h) * kHeadDim + d] = f32_to_bf16(o / total);
  }
}

// ---- FP8 page staging for CK ----

// Grid (pages, sequences of the group): converts page p of sequence g0 + y to
// the BF16 staging block y * pages + p (both halves) and writes that index into
// the staged table [group, pages], and the key count CK sees into
// staged_kv_lens[y]. Pages past a sequence's keys, or missing, are not
// converted (CK never reads them). A sequence with a single query row is left
// to the FP8 decode kernel (launched after CK): its pages are not converted
// and CK sees one (unconverted) key, so CK's row for it is only overwritten.
__global__ void paged_stage_fp8_kernel(
    const uint8_t *kv_layer, Bf16 *staged, int32_t *staged_table,
    int32_t *staged_kv_lens, const int32_t *block_table,
    const int32_t *q_indptr, const int32_t *kv_lens, int32_t g0, int32_t pages,
    int32_t max_blocks, int32_t num_blocks, int32_t block_tokens,
    int64_t token_elems, float k_scale, float v_scale) {
  const int32_t p = blockIdx.x;
  const int32_t y = blockIdx.y;
  const int32_t s = g0 + y;
  const int32_t dst = y * pages + p;
  const bool single = q_indptr[s + 1] - q_indptr[s] == 1;
  if (threadIdx.x == 0) {
    staged_table[dst] = dst;
    if (p == 0)
      staged_kv_lens[y] = single ? 1 : kv_lens[s];
  }
  if (single || static_cast<int64_t>(p) * block_tokens >= kv_lens[s] ||
      p >= max_blocks)
    return;
  const int32_t block = block_table[static_cast<int64_t>(s) * max_blocks + p];
  if (block < 0 || block >= num_blocks)
    return;
  const int64_t half = static_cast<int64_t>(block_tokens) * token_elems;
  const uint32_t *src = reinterpret_cast<const uint32_t *>(
      kv_layer + static_cast<int64_t>(block) * 2 * half);
  Bf16 *out = staged + static_cast<int64_t>(dst) * 2 * half;
  const int64_t words = 2 * half / 4;
  const int64_t k_words = half / 4;
  for (int64_t w = threadIdx.x; w < words; w += blockDim.x) {
    const uint32_t b4 = src[w];
    const float sc = w < k_words ? k_scale : v_scale;
    for (int i = 0; i < 4; ++i)
      out[w * 4 + i] =
          f32_to_bf16(fp8_dequant(static_cast<uint8_t>(b4 >> (8 * i)), sc));
  }
}

} // namespace

namespace turbine_hip {

int32_t launch_paged_append(turbine_ctx *ctx,
                            const turbine_attention_paged_desc *d) {
  if (d->dtype == TURBINE_DTYPE_F8E4M3) {
    hipLaunchKernelGGL(
        paged_append_fp8_kernel, dim3(static_cast<unsigned>(d->total_q)),
        dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(d->k_new),
        static_cast<const Bf16 *>(d->v_new),
        static_cast<uint8_t *>(d->kv_layer), d->block_table, d->q_indptr,
        d->kv_lens, d->num_seqs, d->max_blocks_per_seq, d->num_blocks,
        d->block_tokens, d->num_kv_heads, d->new_stride_token, d->k_scale,
        d->v_scale);
    return check_hip(ctx, hipGetLastError(), "paged FP8 append kernel launch");
  }
  hipLaunchKernelGGL(
      paged_append_kernel, dim3(static_cast<unsigned>(d->total_q)),
      dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(d->k_new),
      static_cast<const Bf16 *>(d->v_new), static_cast<Bf16 *>(d->kv_layer),
      d->block_table, d->q_indptr, d->kv_lens, d->num_seqs,
      d->max_blocks_per_seq, d->num_blocks, d->block_tokens, d->num_kv_heads,
      d->new_stride_token);
  return check_hip(ctx, hipGetLastError(), "paged append kernel launch");
}

int32_t launch_paged_attention(turbine_ctx *ctx,
                               const turbine_attention_paged_desc *d) {
  const int32_t group = d->num_q_heads / d->num_kv_heads;
  int32_t q_tile = kMaxRows / group;
  if (q_tile > d->max_q_len)
    q_tile = d->max_q_len;
  const int32_t tiles = (d->max_q_len + q_tile - 1) / q_tile;
  const dim3 grid(static_cast<unsigned>(tiles),
                  static_cast<unsigned>(d->num_seqs),
                  static_cast<unsigned>(d->num_kv_heads));
  if (d->dtype == TURBINE_DTYPE_F8E4M3) {
    hipLaunchKernelGGL(
        paged_attention_kernel<Fp8Pages>, grid, dim3(kThreads), 0,
        ctx->stream, static_cast<const Bf16 *>(d->q),
        static_cast<const uint8_t *>(d->kv_layer), static_cast<Bf16 *>(d->out),
        d->block_table, d->q_indptr, d->kv_lens, d->max_blocks_per_seq,
        d->num_blocks, d->block_tokens, d->num_kv_heads, group, q_tile,
        d->q_stride_token, d->out_stride_token, d->scale, d->causal,
        d->k_scale, d->v_scale);
  } else {
    hipLaunchKernelGGL(
        paged_attention_kernel<Bf16Pages>, grid, dim3(kThreads), 0,
        ctx->stream, static_cast<const Bf16 *>(d->q),
        static_cast<const Bf16 *>(d->kv_layer), static_cast<Bf16 *>(d->out),
        d->block_table, d->q_indptr, d->kv_lens, d->max_blocks_per_seq,
        d->num_blocks, d->block_tokens, d->num_kv_heads, group, q_tile,
        d->q_stride_token, d->out_stride_token, d->scale, d->causal, 1.0f,
        1.0f);
  }
  return check_hip(ctx, hipGetLastError(), "paged attention kernel launch");
}

int32_t launch_paged_decode_fp8(turbine_ctx *ctx,
                                const turbine_attention_paged_desc *d) {
  const int32_t group = d->num_q_heads / d->num_kv_heads;
  // The kernel's LDS holds kPagedFp8DecodeMaxGroup query heads per KV head: a
  // larger group would index past it (review r13 P1). Callers route such
  // shapes to the Turbine FP8 kernel; this is the last line of defence.
  if (group > kPagedFp8DecodeMaxGroup)
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "paged FP8 decode: GQA group " + std::to_string(group) +
                    " exceeds " + std::to_string(kPagedFp8DecodeMaxGroup));
  const dim3 grid(static_cast<unsigned>(d->num_seqs),
                  static_cast<unsigned>(d->num_kv_heads));
  auto launch = [&](auto kernel) {
    hipLaunchKernelGGL(
        kernel, grid, dim3(kThreads), 0, ctx->stream,
        static_cast<const Bf16 *>(d->q),
        static_cast<const uint8_t *>(d->kv_layer), static_cast<Bf16 *>(d->out),
        d->block_table, d->q_indptr, d->kv_lens, d->max_blocks_per_seq,
        d->num_blocks, d->block_tokens, d->num_kv_heads, group,
        d->q_stride_token, d->out_stride_token, d->scale, d->k_scale,
        d->v_scale);
  };
  // The smallest instantiated head count that holds the group; unit scales
  // skip the BF16 rounding of dequantized values (exact already).
  auto pick = [&](auto unit) {
    constexpr bool U = decltype(unit)::value;
    if (group <= 1)
      launch(paged_decode_fp8_kernel<1, U>);
    else if (group <= 2)
      launch(paged_decode_fp8_kernel<2, U>);
    else if (group <= 3)
      launch(paged_decode_fp8_kernel<3, U>);
    else if (group <= 4)
      launch(paged_decode_fp8_kernel<4, U>);
    else
      launch(paged_decode_fp8_kernel<kPagedFp8DecodeMaxGroup, U>);
  };
  if (d->k_scale == 1.0f && d->v_scale == 1.0f)
    pick(std::true_type{});
  else
    pick(std::false_type{});
  return check_hip(ctx, hipGetLastError(), "paged FP8 decode kernel launch");
}

int32_t launch_paged_stage_fp8(turbine_ctx *ctx,
                               const turbine_attention_paged_desc *d,
                               void *staged, int32_t *staged_table,
                               int32_t *staged_kv_lens, int32_t g0,
                               int32_t count, int32_t pages) {
  hipLaunchKernelGGL(
      paged_stage_fp8_kernel,
      dim3(static_cast<unsigned>(pages), static_cast<unsigned>(count)),
      dim3(kThreads), 0, ctx->stream,
      static_cast<const uint8_t *>(d->kv_layer), static_cast<Bf16 *>(staged),
      staged_table, staged_kv_lens, d->block_table, d->q_indptr, d->kv_lens,
      g0, pages,
      d->max_blocks_per_seq, d->num_blocks, d->block_tokens,
      static_cast<int64_t>(d->num_kv_heads) * kHeadDim, d->k_scale,
      d->v_scale);
  return check_hip(ctx, hipGetLastError(), "paged FP8 staging kernel launch");
}

} // namespace turbine_hip
