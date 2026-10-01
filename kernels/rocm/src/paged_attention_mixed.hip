// Mixed-format paged attention over one layer of the KV block pool (ABI v2.11,
// Phase 6b S-5): the kernels paged_attention.cpp runs for a descriptor whose
// dtype is TURBINE_DTYPE_TQ4 / _TQ2 or whose block_formats table is not NULL.
//
// Pages. Block b's bytes are the first page bytes of its format at kv_layer +
// b * (the page bytes of dtype): BF16 pages [2, block_tokens, kv_heads, 128],
// FP8 pages the same as e4m3 bytes (read as bf16(e4m3 * scale), the 6a
// contract), TurboQuant pages kv_heads * block_tokens records of the codec
// (crates/turbine-kv codec/turboquant; tq_device.hpp), record h * block_tokens
// + t. A block whose format does not fit a page of dtype, or whose id is
// outside the pool, is skipped (the device tables cannot be checked on the
// host).
//
// Append. Each new row is written in its block's format: BF16 as is, FP8 as
// e4m3(x / scale), TurboQuant through tq::encode_chunk (bit-exact with the CPU
// codec, the transcode's code), one workgroup per (KV head, 32 rows).
//
// Attention, "turbine_hip_mixed" (decision "P6b: mixed-format / TurboQuant
// paged attention -- provider evaluation (kernel reuse rule)": an own kernel,
// the evaluation's own_rot prototype; llama.cpp's dequantize-on-load loop is
// the reused pattern). Three kernels:
// - prep (TurboQuant tables present): per query row and KV head, q of each
//   query head of the group rotated once, Rq = H (s_k . q) / sqrt(128), into
//   the scratch;
// - attention: grid (rows, KV heads, splits). A row's keys are cut into at
//   most kMaxSplits splits of a length fixed by its own visible key count (a
//   multiple of kSplitMin), so a row's result does not depend on its batch,
//   and a small batch still fills the device (the evaluation's Llama b1 case:
//   8 workgroups without a split). Every query head of the group in one
//   workgroup, chunks of 32 keys round the waves, each lane one key: BF16 / FP8
//   keys q . k in F32; TurboQuant keys norm / sqrt(d) * <Rq, c[codes]>, the
//   rotated-domain MSE reconstruction (K is TurboQuant_mse like V, decision
//   "6b Task 9: TurboQuant K quantizer" A).
//   Online softmax in F32 (P not rounded); V accumulated in a plain and a
//   rotated accumulator (TurboQuant: norm / sqrt(d) * c[codes]); at the end
//   the rotated one is rotated back once (s_v . H acc / sqrt(d)) and added; the
//   split's unnormalised output, maximum and sum go to the scratch;
// - combine: per row and query head, the splits against their overall
//   maximum, divided by the sum, rounded to BF16.
// It matches cpu::tq_attention (exact attention in F64 over the same KV)
// within the tolerance of hip_ops paged_mixed_matches_cpu.
//
// Staging (the prefill's staged implementations, paged_attention.cpp): the
// batch's pages decoded into BF16 pages of the attention scratch by their
// formats (TurboQuant through tq::decode_chunk, bit-exact with the transcode's
// decode), for an unchanged BF16 kernel to attend over.
#include <algorithm>
#include <string>
#include <type_traits>

#include "bf16.hpp"
#include "fp8.hpp"
#include "paged_fp8.hpp"
#include "paged_mixed.hpp"
#include "tq_device.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::fail;
using turbine_hip::attention_scratch;
using turbine_hip::f32_to_bf16;
using turbine_hip::f32_to_fp8_e4m3;
using turbine_hip::fp8_dequant;
using turbine_hip::round_bf16;
namespace tq = turbine_hip::tq;

namespace {

constexpr int kD = 128;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kFmtBf16 = TURBINE_KVFMT_L0;
constexpr int kFmtFp8 = TURBINE_KVFMT_FP8_E4M3;
constexpr int kFmtTq4 = TURBINE_KVFMT_TQ4;
constexpr int kFmtTq2 = TURBINE_KVFMT_TQ2;

// The attention kernel: 4 waves of 32 lanes.
constexpr int kThreads = 128;
constexpr int kLanes = 32;
constexpr int kWaves = kThreads / kLanes;
constexpr int kDpl = kD / kLanes; // head dims per lane in P.V
// Splits of a row's keys: at most kMaxSplits, each a multiple of kSplitMin
// keys (a split of kSplitMin keys fills the 4 waves once).
constexpr int kSplitMin = kWaves * kLanes;
constexpr int kMaxSplits = 16;
// Scratch bytes the partial outputs of one launch may use; rows beyond are
// launched in further passes.
constexpr int64_t kPartialMaxBytes = int64_t{64} << 20;

// The four codebook pointers of tq_params (1 .. 4 bits), passed to the kernels
// by value (a decode graph captures them with the launch).
struct CodebookPtrs {
  const float *p[4];
};

// 1 / sqrt(128) in F32.
constexpr float kInvSd = 0x1.6a09e6p-4f;

__host__ __device__ inline int64_t page_bytes_of(int fmt, int32_t bt,
                                                 int32_t hkv) {
  const int64_t elems = 2 * static_cast<int64_t>(bt) * hkv * kD;
  switch (fmt) {
  case kFmtBf16:
    return 2 * elems;
  case kFmtFp8:
    return elems;
  case kFmtTq4:
    return static_cast<int64_t>(hkv) * bt * tq::Tq4::kBytes;
  case kFmtTq2:
    return static_cast<int64_t>(hkv) * bt * tq::Tq2::kBytes;
  default:
    return 0;
  }
}

__host__ __device__ inline int format_of_dtype(int32_t dtype) {
  switch (dtype) {
  case TURBINE_DTYPE_BF16:
    return kFmtBf16;
  case TURBINE_DTYPE_F8E4M3:
    return kFmtFp8;
  case TURBINE_DTYPE_TQ4:
    return kFmtTq4;
  case TURBINE_DTYPE_TQ2:
    return kFmtTq2;
  default:
    return -1;
  }
}

// Largest s with q_indptr[s] <= row.
__device__ inline int32_t seq_of_row(const int32_t *q_indptr, int32_t num_seqs,
                                     int32_t row) {
  int32_t lo = 0;
  int32_t hi = num_seqs - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) / 2;
    if (q_indptr[mid] <= row)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// Where a token's page is: the block's byte offset in kv_layer and its format,
// or a negative offset when the block is missing, outside the pool, or of a
// format that does not fit a page of dtype.
struct Where {
  int64_t base;
  int fmt;
};

__device__ inline Where where_of(const int32_t *block_table,
                                 const uint8_t *block_formats, int base_fmt,
                                 int32_t seq, int32_t page, int32_t max_blocks,
                                 int32_t num_blocks, int64_t slot,
                                 int32_t bt, int32_t hkv) {
  if (page >= max_blocks)
    return {-1, 0};
  const int64_t e = static_cast<int64_t>(seq) * max_blocks + page;
  const int32_t block = block_table[e];
  const int fmt = block_formats != nullptr ? block_formats[e] : base_fmt;
  const int64_t bytes = page_bytes_of(fmt, bt, hkv);
  if (block < 0 || block >= num_blocks || bytes == 0 || bytes > slot)
    return {-1, fmt};
  return {static_cast<int64_t>(block) * slot, fmt};
}

// ---- append ----

// BF16 and FP8 rows: one block per new row (the 6a append's addressing).
__global__ void mixed_append_plain_kernel(
    const Bf16 *k_new, const Bf16 *v_new, uint8_t *kv, const int32_t *table,
    const uint8_t *formats, int base_fmt, const int32_t *q_indptr,
    const int32_t *kv_lens, int32_t num_seqs, int32_t max_blocks,
    int32_t num_blocks, int64_t slot, int32_t bt, int32_t hkv,
    int64_t new_stride, float k_scale, float v_scale) {
  const int32_t row = blockIdx.x;
  const int32_t s = seq_of_row(q_indptr, num_seqs, row);
  const int32_t row0 = q_indptr[s];
  const int32_t q_len = q_indptr[s + 1] - row0;
  const int32_t i = row - row0;
  if (i < 0 || i >= q_len)
    return;
  const int32_t pos = kv_lens[s] - q_len + i;
  if (pos < 0)
    return;
  const Where w = where_of(table, formats, base_fmt, s, pos / bt, max_blocks,
                           num_blocks, slot, bt, hkv);
  if (w.base < 0 || (w.fmt != kFmtBf16 && w.fmt != kFmtFp8))
    return;
  const int32_t t = pos % bt;
  const int64_t token_elems = static_cast<int64_t>(hkv) * kD;
  const Bf16 *k_src = k_new + static_cast<int64_t>(row) * new_stride;
  const Bf16 *v_src = v_new + static_cast<int64_t>(row) * new_stride;
  const int64_t k_at = static_cast<int64_t>(t) * token_elems;
  const int64_t v_at = (static_cast<int64_t>(bt) + t) * token_elems;
  if (w.fmt == kFmtBf16) {
    Bf16 *page = reinterpret_cast<Bf16 *>(kv + w.base);
    for (int64_t e = threadIdx.x; e < token_elems; e += blockDim.x) {
      page[k_at + e] = k_src[e];
      page[v_at + e] = v_src[e];
    }
  } else {
    uint8_t *page = kv + w.base;
    for (int64_t e = threadIdx.x; e < token_elems; e += blockDim.x) {
      page[k_at + e] = f32_to_fp8_e4m3(bf16_to_f32(k_src[e]) / k_scale);
      page[v_at + e] = f32_to_fp8_e4m3(bf16_to_f32(v_src[e]) / v_scale);
    }
  }
}

// TurboQuant rows of format FMT (records of KB / VB bits): grid (KV heads,
// chunks of 32 new rows), tq::kThreads threads. Rows of other formats in the
// chunk are encoded as zeros and not written.
template <int KB, int VB, int FMT>
__global__ void __launch_bounds__(tq::kThreads) mixed_append_tq_kernel(
    const Bf16 *k_new, const Bf16 *v_new, uint8_t *kv, const int32_t *table,
    const uint8_t *formats, int base_fmt, const int32_t *q_indptr,
    const int32_t *kv_lens, int32_t num_seqs, int32_t total_q,
    int32_t max_blocks, int32_t num_blocks, int64_t slot, int32_t bt,
    int32_t hkv, int64_t new_stride, const float *tables, const float *cb_k,
    const float *cb_v) {
  using R = tq::Record<KB, VB>;
  __shared__ tq::EncodeLds<KB, VB> lds;
  __shared__ int64_t dst[tq::kChunk];
  __shared__ int any;

  const int tid = threadIdx.x;
  const int32_t head = blockIdx.x;
  const int32_t r0 = blockIdx.y * tq::kChunk;
  const int nt = min(tq::kChunk, total_q - r0);
  if (tid == 0)
    any = 0;
  __syncthreads();
  if (tid < nt) {
    const int32_t row = r0 + tid;
    const int32_t s = seq_of_row(q_indptr, num_seqs, row);
    const int32_t row0 = q_indptr[s];
    const int32_t q_len = q_indptr[s + 1] - row0;
    const int32_t pos = kv_lens[s] - q_len + (row - row0);
    int64_t at = -1;
    if (row - row0 < q_len && pos >= 0) {
      const Where w = where_of(table, formats, base_fmt, s, pos / bt,
                               max_blocks, num_blocks, slot, bt, hkv);
      if (w.base >= 0 && w.fmt == FMT)
        at = w.base +
             (static_cast<int64_t>(head) * bt + pos % bt) * R::kBytes;
    }
    dst[tid] = at;
    if (at >= 0)
      atomicOr(&any, 1);
  }
  __syncthreads();
  if (any == 0)
    return;
  if (tid < (1 << KB))
    lds.cbk[tid] = cb_k[tid];
  if (tid < (1 << VB))
    lds.cbv[tid] = cb_v[tid];
  for (int i = tid; i < 2 * nt * kD; i += tq::kThreads) {
    const int kind = i / (nt * kD);
    const int tok = (i / kD) % nt;
    const int e = i % kD;
    float x = 0.0f;
    if (dst[tok] >= 0) {
      const Bf16 *src = kind == 0 ? k_new : v_new;
      x = bf16_to_f32(src[static_cast<int64_t>(r0 + tok) * new_stride +
                          static_cast<int64_t>(head) * kD + e]);
    }
    lds.xs[kind][tok][e] = x;
  }
  __syncthreads();
  const float *head_tables = tables + static_cast<int64_t>(head) * tq::kHeadElems;
  tq::encode_chunk<KB, VB>(lds, nt, head_tables,
                           [&](int tok, int o, uint8_t v) {
                             if (dst[tok] >= 0)
                               kv[dst[tok] + o] = v;
                           });
}

// ---- attention ----

// The split length of a row with `visible` keys: a multiple of kSplitMin, so
// that at most kMaxSplits splits cover them.
__device__ inline int32_t split_len_of(int32_t visible) {
  int32_t per = (visible + kMaxSplits - 1) / kMaxSplits;
  per = (per + kSplitMin - 1) / kSplitMin * kSplitMin;
  return per > kSplitMin ? per : kSplitMin;
}

// The row a workgroup of a launch over rows [row0, row0 + n) computes: its
// sequence, query row within it and visible keys; false when it is skipped
// (single_only and a sequence of more than one query row).
struct RowInfo {
  int32_t seq;
  int32_t visible;
};

__device__ inline bool row_info(const int32_t *q_indptr, const int32_t *kv_lens,
                                int32_t num_seqs, int32_t row, int32_t causal,
                                int32_t single_only, RowInfo &out) {
  const int32_t s = seq_of_row(q_indptr, num_seqs, row);
  const int32_t row0 = q_indptr[s];
  const int32_t q_len = q_indptr[s + 1] - row0;
  const int32_t i = row - row0;
  if (i < 0 || i >= q_len || (single_only != 0 && q_len != 1))
    return false;
  const int32_t kv_len = kv_lens[s];
  out.seq = s;
  out.visible = causal != 0 ? kv_len - q_len + i + 1 : kv_len;
  return out.visible > 0;
}

// Prep: grid (rows of the launch, KV heads), tq::kThreads threads. Writes
// Rq[g] (g < G) to prep[(r * hkv + h) * G * 128].
template <int G>
__global__ void __launch_bounds__(tq::kThreads) mixed_prep_kernel(
    const Bf16 *q, const int32_t *q_indptr, const int32_t *kv_lens,
    int32_t num_seqs, int32_t row0, int32_t hkv, int32_t group,
    int64_t q_stride, int32_t causal, int32_t single_only,
    const float *tables, float *prep) {
  __shared__ float rq[G][kD];
  const int tid = threadIdx.x;
  const int32_t r = blockIdx.x;
  const int32_t row = row0 + r;
  const int32_t kvh = blockIdx.y;
  RowInfo info;
  if (!row_info(q_indptr, kv_lens, num_seqs, row, causal, single_only, info))
    return;
  const float *tab = tables + static_cast<int64_t>(kvh) * tq::kHeadElems;
  const int grp = tid / tq::kGroup;
  const int lane = tid % tq::kGroup;
  for (int g = grp; g < G; g += tq::kThreads / tq::kGroup) {
    float y[tq::kPer];
#pragma unroll
    for (int i = 0; i < tq::kPer; ++i) {
      const int e = i * tq::kGroup + lane;
      float v = 0.0f;
      if (g < group)
        v = bf16_to_f32(q[static_cast<int64_t>(row) * q_stride +
                          static_cast<int64_t>(kvh * group + g) * kD + e]);
      y[i] = v * tab[e];
    }
    tq::fwht(y, lane);
#pragma unroll
    for (int i = 0; i < tq::kPer; ++i)
      rq[g][i * tq::kGroup + lane] = y[i] * kInvSd;
  }
  __syncthreads();
  float *out = prep + (static_cast<int64_t>(r) * hkv + kvh) * G * kD;
  for (int e = tid; e < G * kD; e += tq::kThreads)
    out[e] = rq[e / kD][e % kD];
}

__device__ inline float lane_max(float v) {
#pragma unroll
  for (int m = kLanes / 2; m > 0; m /= 2)
    v = fmaxf(v, __shfl_xor(v, m, kLanes));
  return v;
}

__device__ inline float lane_sum(float v) {
#pragma unroll
  for (int m = kLanes / 2; m > 0; m /= 2)
    v += __shfl_xor(v, m, kLanes);
  return v;
}

__device__ inline void wave_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <int G> struct AttnLds {
  float q[G][kD];
  float rq[G][kD];
  float cb[4][16];
  float vs[kD];
  union {
    struct {
      float p[kWaves][G][kLanes];
      int64_t voff[kWaves][kLanes];
      float vmul[kWaves][kLanes];
      int32_t vfmt[kWaves][kLanes];
    } loop;
    float o[kWaves][G][kD];    // after it: the waves' partial outputs
  };
  float max[kWaves][G];
  float sum[kWaves][G];
  float oplain[G][kD];
  float orot[G][kD];
};

// The scores of one TurboQuant key (record rec) for G query heads, the softmax
// scale folded into c_mse. The 128 codes go in four steps of 32 (KB
// words each, read as dwords); each step's centroids are looked up once and
// dotted with every head's Rq, so few Rq values are live at once (a full
// unroll over the 128 coordinates hoists them all into registers and spills).
template <int G, int KB>
__device__ inline void tq_score(const uint8_t *rec, const AttnLds<G> &lds,
                                float c_mse, float *s) {
  using R = tq::Record<KB, KB>;
  const auto *rw = reinterpret_cast<const uint32_t *>(rec);
  const float kn = __uint_as_float(
      static_cast<uint32_t>(rec[R::kKNorm] | rec[R::kKNorm + 1] << 8) << 16);
  float mse[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    mse[g] = 0.0f;
#pragma unroll 1
  for (int c = 0; c < kD / 32; ++c) {
    uint32_t w[KB];
#pragma unroll
    for (int k = 0; k < KB; ++k)
      w[k] = rw[c * KB + k];
    float cv[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int bit = j * KB;
      const int wi = bit >> 5;
      const int sh = bit & 31;
      uint32_t code = w[wi] >> sh;
      if (sh + KB > 32)
        code |= w[wi + 1] << (32 - sh);
      cv[j] = lds.cb[KB - 1][code & ((1u << KB) - 1u)];
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < 32; ++j)
        acc += lds.rq[g][32 * c + j] * cv[j];
      mse[g] += acc;
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g)
    s[g] = c_mse * kn * mse[g];
}

// Attention: grid (rows of the launch, KV heads, splits), kThreads threads.
// Writes split z's unnormalised output to part_o[((r * hq + h) * kMaxSplits
// + z) * 128] and its (maximum, sum) to part_ml[...].
template <int G>
__global__ void __launch_bounds__(kThreads) mixed_attn_kernel(
    const Bf16 *q, const uint8_t *kv, const int32_t *table,
    const uint8_t *formats, int base_fmt, const int32_t *q_indptr,
    const int32_t *kv_lens, int32_t num_seqs, int32_t row0, int32_t max_blocks,
    int32_t num_blocks, int64_t slot, int32_t bt, int32_t hkv, int32_t group,
    int64_t q_stride, float scale, float k_scale, float v_scale,
    int32_t causal, int32_t single_only, int32_t has_tq, const float *tables,
    CodebookPtrs codebooks, const float *prep, float *part_o,
    float2 *part_ml) {
  __shared__ AttnLds<G> lds;
  const int tid = threadIdx.x;
  const int lane = tid % kLanes;
  const int wave = tid / kLanes;
  const int32_t r = blockIdx.x;
  const int32_t row = row0 + r;
  const int32_t kvh = blockIdx.y;
  const int32_t z = blockIdx.z;
  RowInfo info;
  if (!row_info(q_indptr, kv_lens, num_seqs, row, causal, single_only, info))
    return;
  const int32_t len = split_len_of(info.visible);
  const int32_t k_begin = z * len;
  if (k_begin >= info.visible)
    return;
  const int32_t k_end = min(info.visible, k_begin + len);
  const int32_t seq = info.seq;
  const int32_t hq = hkv * group;

  for (int e = tid; e < G * kD; e += kThreads) {
    const int g = e / kD;
    const int d = e % kD;
    float v = 0.0f;
    if (g < group)
      v = bf16_to_f32(q[static_cast<int64_t>(row) * q_stride +
                        static_cast<int64_t>(kvh * group + g) * kD + d]);
    lds.q[g][d] = v;
  }
  if (has_tq) {
    const float *pr = prep + (static_cast<int64_t>(r) * hkv + kvh) * G * kD;
    for (int e = tid; e < G * kD; e += kThreads)
      lds.rq[e / kD][e % kD] = pr[e];
    if (tid < 64)
      lds.cb[tid / 16][tid % 16] =
          tid % 16 < (2 << (tid / 16)) ? codebooks.p[tid / 16][tid % 16] : 0.0f;
    const float *tab = tables + static_cast<int64_t>(kvh) * tq::kHeadElems;
    lds.vs[tid] = tab[kD + tid];
  }
  __syncthreads();
  const float c_mse = scale * kInvSd;

  float m[G];
  float sum[G];
  float accp[G][kDpl];
  float accr[G][kDpl];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = kNegInf;
    sum[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDpl; ++i)
      accp[g][i] = accr[g][i] = 0.0f;
  }

  for (int32_t c0 = k_begin + wave * kLanes; c0 < k_end;
       c0 += kWaves * kLanes) {
    const int32_t key = c0 + lane;
    bool valid = key < k_end;
    float s[G];
#pragma unroll
    for (int g = 0; g < G; ++g)
      s[g] = 0.0f;
    int64_t voff = -1;
    float vmul = 1.0f;
    int32_t fmt = kFmtBf16;
    if (valid) {
      const Where w = where_of(table, formats, base_fmt, seq, key / bt,
                               max_blocks, num_blocks, slot, bt, hkv);
      fmt = w.fmt;
      valid = w.base >= 0 && (has_tq || fmt == kFmtBf16 || fmt == kFmtFp8);
      const int32_t t = key % bt;
      if (valid && (fmt == kFmtBf16 || fmt == kFmtFp8)) {
        const int es = fmt == kFmtBf16 ? 2 : 1;
        const int64_t koff =
            w.base + (static_cast<int64_t>(t) * hkv + kvh) * kD * es;
        voff = w.base +
               ((static_cast<int64_t>(bt) + t) * hkv + kvh) * kD * es;
        if (fmt == kFmtBf16) {
          const uint4 *kr = reinterpret_cast<const uint4 *>(kv + koff);
#pragma unroll 2
          for (int c = 0; c < kD / 8; ++c) {
            const uint4 x = kr[c];
            const uint32_t words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int wd = 0; wd < 4; ++wd) {
              const float lo = __uint_as_float(words[wd] << 16);
              const float hi = __uint_as_float(words[wd] & 0xffff0000u);
#pragma unroll
              for (int g = 0; g < G; ++g)
                s[g] += lds.q[g][8 * c + 2 * wd] * lo +
                        lds.q[g][8 * c + 2 * wd + 1] * hi;
            }
          }
        } else {
          vmul = v_scale;
          const uint4 *kr = reinterpret_cast<const uint4 *>(kv + koff);
#pragma unroll 1
          for (int c = 0; c < kD / 16; ++c) {
            const uint4 x = kr[c];
            const uint32_t words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
              for (int b = 0; b < 4; ++b) {
                const float kvv = round_bf16(fp8_dequant(
                    static_cast<uint8_t>(words[wd] >> (8 * b)), k_scale));
#pragma unroll
                for (int g = 0; g < G; ++g)
                  s[g] += lds.q[g][16 * c + 4 * wd + b] * kvv;
              }
            }
          }
        }
#pragma unroll
        for (int g = 0; g < G; ++g)
          s[g] *= scale;
      } else if (valid) {
        if (fmt == kFmtTq4) {
          const uint8_t *rec =
              kv + w.base + (static_cast<int64_t>(kvh) * bt + t) * tq::Tq4::kBytes;
          tq_score<G, 4>(rec, lds, c_mse, s);
          voff = (rec - kv) + tq::Tq4::kVCodes;
          vmul = __uint_as_float(static_cast<uint32_t>(
                                     rec[tq::Tq4::kVNorm] |
                                     rec[tq::Tq4::kVNorm + 1] << 8)
                                 << 16) *
                 kInvSd;
        } else {
          const uint8_t *rec =
              kv + w.base + (static_cast<int64_t>(kvh) * bt + t) * tq::Tq2::kBytes;
          tq_score<G, 2>(rec, lds, c_mse, s);
          voff = (rec - kv) + tq::Tq2::kVCodes;
          vmul = __uint_as_float(static_cast<uint32_t>(
                                     rec[tq::Tq2::kVNorm] |
                                     rec[tq::Tq2::kVNorm + 1] << 8)
                                 << 16) *
                 kInvSd;
        }
      }
    }
    float p[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float sg = valid ? s[g] : kNegInf;
      const float mn = fmaxf(m[g], lane_max(sg));
      const float f = mn == kNegInf ? 1.0f : expf(m[g] - mn);
      m[g] = mn;
      sum[g] *= f;
#pragma unroll
      for (int i = 0; i < kDpl; ++i) {
        accp[g][i] *= f;
        accr[g][i] *= f;
      }
      p[g] = valid ? expf(sg - mn) : 0.0f;
      sum[g] += p[g];
      lds.loop.p[wave][g][lane] = p[g];
    }
    lds.loop.voff[wave][lane] = valid ? voff : -1;
    lds.loop.vmul[wave][lane] = vmul;
    lds.loop.vfmt[wave][lane] = fmt;
    wave_sync();
    const int32_t keys = min(kLanes, k_end - c0);
    for (int j = 0; j < keys; ++j) {
      const int64_t vo = lds.loop.voff[wave][j];
      if (vo < 0)
        continue;
      const float vm = lds.loop.vmul[wave][j];
      const int32_t vf = lds.loop.vfmt[wave][j];
      float v[kDpl];
      if (vf == kFmtBf16) {
        const uint2 x = *reinterpret_cast<const uint2 *>(kv + vo + lane * 8);
        v[0] = __uint_as_float(x.x << 16);
        v[1] = __uint_as_float(x.x & 0xffff0000u);
        v[2] = __uint_as_float(x.y << 16);
        v[3] = __uint_as_float(x.y & 0xffff0000u);
      } else if (vf == kFmtFp8) {
        const uint32_t b4 =
            *reinterpret_cast<const uint32_t *>(kv + vo + lane * 4);
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = round_bf16(
              fp8_dequant(static_cast<uint8_t>(b4 >> (8 * i)), vm));
      } else if (vf == kFmtTq4) {
        const uint32_t c =
            *reinterpret_cast<const uint16_t *>(kv + vo + lane * 2);
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = lds.cb[3][(c >> (4 * i)) & 15] * vm;
      } else {
        const uint32_t c = kv[vo + lane];
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = lds.cb[1][(c >> (2 * i)) & 3] * vm;
      }
      const bool rot = vf == kFmtTq4 || vf == kFmtTq2;
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float pg = lds.loop.p[wave][g][j];
#pragma unroll
        for (int i = 0; i < kDpl; ++i) {
          if (rot)
            accr[g][i] += pg * v[i];
          else
            accp[g][i] += pg * v[i];
        }
      }
    }
    wave_sync();
  }
  // Combine the waves against the split's maximum: the plain accumulators,
  // then the rotated ones, through lds.o.
  float wsum[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    wsum[g] = lane_sum(sum[g]);
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (lane == 0) {
      lds.max[wave][g] = m[g];
      lds.sum[wave][g] = wsum[g];
    }
#pragma unroll
    for (int i = 0; i < kDpl; ++i)
      lds.o[wave][g][lane * kDpl + i] = accp[g][i];
  }
  __syncthreads();
  for (int pass = 0; pass < 2; ++pass) {
    for (int e = tid; e < G * kD; e += kThreads) {
      const int g = e / kD;
      const int d = e % kD;
      float mx = kNegInf;
      for (int w = 0; w < kWaves; ++w)
        mx = fmaxf(mx, lds.max[w][g]);
      float acc = 0.0f;
      for (int w = 0; w < kWaves; ++w) {
        if (lds.max[w][g] == kNegInf)
          continue;
        acc += lds.o[w][g][d] * expf(lds.max[w][g] - mx);
      }
      (pass == 0 ? lds.oplain : lds.orot)[g][d] = acc;
    }
    __syncthreads();
    if (pass == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          lds.o[wave][g][lane * kDpl + i] = accr[g][i];
      __syncthreads();
    }
  }
  // Rotate the TurboQuant part back once: s_v . H acc / sqrt(d).
  const int grp = tid / tq::kGroup;
  const int glane = tid % tq::kGroup;
  for (int g = grp; g < G; g += kThreads / tq::kGroup) {
    float y[tq::kPer];
#pragma unroll
    for (int i = 0; i < tq::kPer; ++i)
      y[i] = lds.orot[g][i * tq::kGroup + glane];
    if (has_tq)
      tq::fwht(y, glane);
#pragma unroll
    for (int i = 0; i < tq::kPer; ++i) {
      const int e = i * tq::kGroup + glane;
      const float rot = has_tq ? y[i] * kInvSd * lds.vs[e] : 0.0f;
      lds.oplain[g][e] += rot;
    }
  }
  __syncthreads();
  for (int e = tid; e < group * kD; e += kThreads) {
    const int g = e / kD;
    const int d = e % kD;
    const int64_t at =
        (static_cast<int64_t>(r) * hq + kvh * group + g) * kMaxSplits + z;
    part_o[at * kD + d] = lds.oplain[g][d];
    if (d == 0) {
      float mx = kNegInf;
      float tot = 0.0f;
      for (int w = 0; w < kWaves; ++w)
        mx = fmaxf(mx, lds.max[w][g]);
      for (int w = 0; w < kWaves; ++w)
        if (lds.max[w][g] != kNegInf)
          tot += lds.sum[w][g] * expf(lds.max[w][g] - mx);
      part_ml[at] = make_float2(mx, tot);
    }
  }
}

// Combine: grid (rows of the launch, query heads), kD threads.
__global__ void __launch_bounds__(kD) mixed_combine_kernel(
    const float *part_o, const float2 *part_ml, Bf16 *out,
    const int32_t *q_indptr, const int32_t *kv_lens, int32_t num_seqs,
    int32_t row0, int32_t hq, int64_t out_stride, int32_t causal,
    int32_t single_only) {
  const int32_t r = blockIdx.x;
  const int32_t row = row0 + r;
  const int32_t h = blockIdx.y;
  const int d = threadIdx.x;
  RowInfo info;
  if (!row_info(q_indptr, kv_lens, num_seqs, row, causal, single_only, info))
    return;
  const int32_t len = split_len_of(info.visible);
  const int32_t splits = (info.visible + len - 1) / len;
  const int64_t at = (static_cast<int64_t>(r) * hq + h) * kMaxSplits;
  float mx = kNegInf;
  for (int z = 0; z < splits; ++z)
    mx = fmaxf(mx, part_ml[at + z].x);
  float o = 0.0f;
  float tot = 0.0f;
  for (int z = 0; z < splits; ++z) {
    const float2 ml = part_ml[at + z];
    if (ml.x == kNegInf)
      continue;
    const float f = expf(ml.x - mx);
    o += part_o[(at + z) * kD + d] * f;
    tot += ml.y * f;
  }
  out[static_cast<int64_t>(row) * out_stride + static_cast<int64_t>(h) * kD +
      d] = f32_to_bf16(tot > 0.0f ? o / tot : 0.0f);
}

// ---- staging ----

template <int KB, int VB> struct StageTq {
  using R = tq::Record<KB, VB>;
  // Decodes this head's block_tokens records at rec0 into the BF16 page out.
  __device__ static void run(tq::DecodeLds<KB, VB> &lds, const uint8_t *rec0,
                             Bf16 *out, int32_t bt, int32_t hkv,
                             int32_t head, const float *head_tables,
                             const CodebookPtrs &codebooks) {
    const int tid = threadIdx.x;
    if (tid < (1 << KB))
      lds.cbk[tid] = codebooks.p[KB - 1][tid];
    if (tid < (1 << VB))
      lds.cbv[tid] = codebooks.p[VB - 1][tid];
    for (int t0 = 0; t0 < bt; t0 += tq::kChunk) {
      const int nt = min(tq::kChunk, bt - t0);
      for (int i = tid; i < nt * R::kBytes; i += tq::kThreads)
        lds.recs[i / R::kBytes][i % R::kBytes] =
            rec0[static_cast<int64_t>(t0) * R::kBytes + i];
      __syncthreads();
      tq::decode_chunk<KB, VB>(
          lds, nt, head_tables, [&](int kind, int tok, int e, uint16_t bits) {
            out[((static_cast<int64_t>(kind) * bt + t0 + tok) * hkv + head) *
                    kD +
                e] = Bf16{bits};
          });
    }
  }
};

// Grid (pages, sequences of the group, KV heads), tq::kThreads threads: page p
// of sequence g0 + y, KV head h, into BF16 staging block y * pages + p.
__global__ void __launch_bounds__(tq::kThreads) mixed_stage_kernel(
    const uint8_t *kv, Bf16 *staged, int32_t *staged_table,
    int32_t *staged_kv_lens, const int32_t *table, const uint8_t *formats,
    int base_fmt, const int32_t *q_indptr, const int32_t *kv_lens, int32_t g0,
    int32_t pages, int32_t max_blocks, int32_t num_blocks, int64_t slot,
    int32_t bt, int32_t hkv, float k_scale, float v_scale,
    const float *tables, CodebookPtrs codebooks) {
  __shared__ union {
    tq::DecodeLds<4, 4> tq4;
    tq::DecodeLds<2, 2> tq2;
  } lds;
  const int32_t p = blockIdx.x;
  const int32_t y = blockIdx.y;
  const int32_t h = blockIdx.z;
  const int32_t s = g0 + y;
  const int32_t dst = y * pages + p;
  const bool single = q_indptr[s + 1] - q_indptr[s] == 1;
  if (threadIdx.x == 0 && h == 0) {
    staged_table[dst] = dst;
    if (p == 0)
      staged_kv_lens[y] = single ? 1 : kv_lens[s];
  }
  if (single || static_cast<int64_t>(p) * bt >= kv_lens[s])
    return;
  const Where w = where_of(table, formats, base_fmt, s, p, max_blocks,
                           num_blocks, slot, bt, hkv);
  if (w.base < 0)
    return;
  Bf16 *out = staged + static_cast<int64_t>(dst) * 2 * bt * hkv * kD;
  const uint8_t *page = kv + w.base;
  if (w.fmt == kFmtBf16 || w.fmt == kFmtFp8) {
    for (int i = threadIdx.x; i < 2 * bt * kD; i += tq::kThreads) {
      const int kind = i / (bt * kD);
      const int t = (i / kD) % bt;
      const int e = i % kD;
      const int64_t at =
          ((static_cast<int64_t>(kind) * bt + t) * hkv + h) * kD + e;
      out[at] = w.fmt == kFmtBf16
                    ? reinterpret_cast<const Bf16 *>(page)[at]
                    : f32_to_bf16(
                          fp8_dequant(page[at], kind == 0 ? k_scale : v_scale));
    }
    return;
  }
  const float *head_tables = tables + static_cast<int64_t>(h) * tq::kHeadElems;
  if (w.fmt == kFmtTq4)
    StageTq<4, 4>::run(lds.tq4,
                       page + static_cast<int64_t>(h) * bt * tq::Tq4::kBytes,
                       out, bt, hkv, h, head_tables, codebooks);
  else
    StageTq<2, 2>::run(lds.tq2,
                       page + static_cast<int64_t>(h) * bt * tq::Tq2::kBytes,
                       out, bt, hkv, h, head_tables, codebooks);
}

int64_t slot_bytes(const turbine_attention_paged_desc *d) {
  return page_bytes_of(format_of_dtype(d->dtype), d->block_tokens,
                       d->num_kv_heads);
}

bool has_tq(const turbine_attention_paged_desc *d) {
  return d->tq_params != nullptr && d->tq_params->tables != nullptr;
}

CodebookPtrs codebooks_of(const turbine_attention_paged_desc *d) {
  CodebookPtrs c{};
  if (has_tq(d))
    for (int i = 0; i < 4; ++i)
      c.p[i] = d->tq_params->codebooks[i];
  return c;
}

const float *tables_of(const turbine_attention_paged_desc *d) {
  return has_tq(d) ? d->tq_params->tables : nullptr;
}

// Calls f(std::integral_constant<int, G>) with the smallest instantiated G
// that holds the GQA group.
template <typename F> void with_group(int32_t group, F f) {
  if (group <= 1)
    f(std::integral_constant<int, 1>{});
  else if (group <= 2)
    f(std::integral_constant<int, 2>{});
  else if (group <= 3)
    f(std::integral_constant<int, 3>{});
  else if (group <= 4)
    f(std::integral_constant<int, 4>{});
  else
    f(std::integral_constant<int, turbine_hip::kPagedMixedMaxGroup>{});
}

int32_t group_bound(int32_t group) {
  return group <= 4 ? group : turbine_hip::kPagedMixedMaxGroup;
}

} // namespace

namespace turbine_hip {

int32_t paged_mixed_format_of(int32_t dtype) { return format_of_dtype(dtype); }

int64_t paged_mixed_page_bytes(int32_t fmt, int32_t block_tokens,
                               int32_t kv_heads) {
  return page_bytes_of(fmt, block_tokens, kv_heads);
}

int32_t launch_paged_append_mixed(turbine_ctx *ctx,
                                  const turbine_attention_paged_desc *d) {
  const int base = format_of_dtype(d->dtype);
  const uint8_t *formats = d->block_formats;
  const int64_t slot = slot_bytes(d);
  auto *kv = static_cast<uint8_t *>(d->kv_layer);
  const auto *k_new = static_cast<const Bf16 *>(d->k_new);
  const auto *v_new = static_cast<const Bf16 *>(d->v_new);
  if (formats != nullptr || base == kFmtBf16 || base == kFmtFp8) {
    hipLaunchKernelGGL(mixed_append_plain_kernel,
                       dim3(static_cast<unsigned>(d->total_q)), dim3(256), 0,
                       ctx->stream, k_new, v_new, kv, d->block_table, formats,
                       base, d->q_indptr, d->kv_lens, d->num_seqs,
                       d->max_blocks_per_seq, d->num_blocks, slot,
                       d->block_tokens, d->num_kv_heads, d->new_stride_token,
                       d->k_scale, d->v_scale);
    if (int32_t rc = check_hip(ctx, hipGetLastError(),
                               "paged mixed append kernel launch");
        rc != TURBINE_OK)
      return rc;
  }
  const CodebookPtrs cb = codebooks_of(d);
  const dim3 grid(static_cast<unsigned>(d->num_kv_heads),
                  static_cast<unsigned>((d->total_q + tq::kChunk - 1) /
                                        tq::kChunk));
  auto encode = [&](auto kernel, int kb, int vb) {
    hipLaunchKernelGGL(kernel, grid, dim3(tq::kThreads), 0, ctx->stream,
                       k_new, v_new, kv, d->block_table, formats, base,
                       d->q_indptr, d->kv_lens, d->num_seqs, d->total_q,
                       d->max_blocks_per_seq, d->num_blocks, slot,
                       d->block_tokens, d->num_kv_heads, d->new_stride_token,
                       tables_of(d), cb.p[kb - 1], cb.p[vb - 1]);
    return check_hip(ctx, hipGetLastError(),
                     "paged TurboQuant append kernel launch");
  };
  if (formats != nullptr || base == kFmtTq4) {
    if (int32_t rc = encode(mixed_append_tq_kernel<4, 4, kFmtTq4>, 4, 4);
        rc != TURBINE_OK)
      return rc;
  }
  if (formats != nullptr || base == kFmtTq2) {
    if (int32_t rc = encode(mixed_append_tq_kernel<2, 2, kFmtTq2>, 2, 2);
        rc != TURBINE_OK)
      return rc;
  }
  return TURBINE_OK;
}

int32_t run_paged_mixed_attention(turbine_ctx *ctx,
                                  const turbine_attention_paged_desc *d,
                                  bool single_only) {
  const int32_t group = d->num_q_heads / d->num_kv_heads;
  if (group > kPagedMixedMaxGroup)
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "paged mixed attention: GQA group " + std::to_string(group) +
                    " exceeds " + std::to_string(kPagedMixedMaxGroup));
  const int32_t g_bound = group_bound(group);
  const bool tq_on = has_tq(d);
  const int32_t hq = d->num_q_heads;
  const int32_t hkv = d->num_kv_heads;
  const int64_t part_row = static_cast<int64_t>(hq) * kMaxSplits *
                           (kD * sizeof(float) + sizeof(float2));
  const int64_t prep_row =
      tq_on ? static_cast<int64_t>(hkv) * g_bound * kD * sizeof(float) : 0;
  const int64_t per_row = part_row + prep_row;
  const int32_t rows = d->total_q;
  const int64_t fit = std::max<int64_t>(1, kPartialMaxBytes / per_row);
  const auto n = static_cast<int32_t>(std::min<int64_t>(rows, fit));
  void *scratch = nullptr;
  if (int32_t rc = attention_scratch(ctx, static_cast<size_t>(n * per_row),
                                     &scratch);
      rc != TURBINE_OK)
    return rc;
  auto *part_o = static_cast<float *>(scratch);
  auto *part_ml = reinterpret_cast<float2 *>(
      static_cast<char *>(scratch) +
      static_cast<int64_t>(n) * hq * kMaxSplits * kD * sizeof(float));
  auto *prep = reinterpret_cast<float *>(static_cast<char *>(scratch) +
                                         static_cast<int64_t>(n) * part_row);
  const int32_t splits = std::max<int32_t>(
      1, std::min<int32_t>(kMaxSplits,
                           (d->max_kv_len + kSplitMin - 1) / kSplitMin));
  const int base = format_of_dtype(d->dtype);
  const int64_t slot = slot_bytes(d);
  const CodebookPtrs cb = codebooks_of(d);
  const int32_t single = single_only ? 1 : 0;
  int32_t rc = TURBINE_OK;
  for (int32_t row0 = 0; row0 < rows && rc == TURBINE_OK; row0 += n) {
    const int32_t cnt = std::min(n, rows - row0);
    with_group(group, [&](auto gc) {
      constexpr int G = decltype(gc)::value;
      if (tq_on) {
        hipLaunchKernelGGL(
            mixed_prep_kernel<G>,
            dim3(static_cast<unsigned>(cnt), static_cast<unsigned>(hkv)),
            dim3(tq::kThreads), 0, ctx->stream,
            static_cast<const Bf16 *>(d->q), d->q_indptr, d->kv_lens,
            d->num_seqs, row0, hkv, group, d->q_stride_token, d->causal,
            single, tables_of(d), prep);
      }
      hipLaunchKernelGGL(
          mixed_attn_kernel<G>,
          dim3(static_cast<unsigned>(cnt), static_cast<unsigned>(hkv),
               static_cast<unsigned>(splits)),
          dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(d->q),
          static_cast<const uint8_t *>(d->kv_layer), d->block_table,
          d->block_formats, base, d->q_indptr, d->kv_lens, d->num_seqs, row0,
          d->max_blocks_per_seq, d->num_blocks, slot, d->block_tokens, hkv,
          group, d->q_stride_token, d->scale, d->k_scale, d->v_scale,
          d->causal, single, tq_on ? 1 : 0, tables_of(d), cb, prep, part_o,
          part_ml);
    });
    hipLaunchKernelGGL(
        mixed_combine_kernel,
        dim3(static_cast<unsigned>(cnt), static_cast<unsigned>(hq)), dim3(kD),
        0, ctx->stream, part_o, part_ml, static_cast<Bf16 *>(d->out),
        d->q_indptr, d->kv_lens, d->num_seqs, row0, hq, d->out_stride_token,
        d->causal, single);
    rc = check_hip(ctx, hipGetLastError(), "paged mixed attention launch");
  }
  return rc;
}

int32_t launch_paged_stage_mixed(turbine_ctx *ctx,
                                 const turbine_attention_paged_desc *d,
                                 void *staged, int32_t *staged_table,
                                 int32_t *staged_kv_lens, int32_t g0,
                                 int32_t count, int32_t pages) {
  hipLaunchKernelGGL(
      mixed_stage_kernel,
      dim3(static_cast<unsigned>(pages), static_cast<unsigned>(count),
           static_cast<unsigned>(d->num_kv_heads)),
      dim3(tq::kThreads), 0, ctx->stream,
      static_cast<const uint8_t *>(d->kv_layer), static_cast<Bf16 *>(staged),
      staged_table, staged_kv_lens, d->block_table, d->block_formats,
      format_of_dtype(d->dtype), d->q_indptr, d->kv_lens, g0, pages,
      d->max_blocks_per_seq, d->num_blocks, slot_bytes(d), d->block_tokens,
      d->num_kv_heads, d->k_scale, d->v_scale, tables_of(d), codebooks_of(d));
  return check_hip(ctx, hipGetLastError(), "paged mixed staging launch");
}

} // namespace turbine_hip
