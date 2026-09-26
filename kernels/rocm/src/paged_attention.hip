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
#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
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

// ---- attention ----

// Grid: (query tiles of the longest sequence, sequences, KV heads). A block
// handles rows r in [0, group * q_tile): q head kv_head * group + r / q_tile,
// query tile * q_tile + r % q_tile of sequence blockIdx.y.
__global__ void __launch_bounds__(kThreads)
    paged_attention_kernel(const Bf16 *q, const Bf16 *kv_layer, Bf16 *out,
                           const int32_t *block_table, const int32_t *q_indptr,
                           const int32_t *kv_lens, int32_t max_blocks,
                           int32_t num_blocks, int32_t block_tokens,
                           int32_t kv_heads, int32_t group, int32_t q_tile,
                           int64_t q_stride, int64_t out_stride, float scale,
                           int32_t causal) {
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
      k_s[j][d] = off >= 0 ? kv_layer[off + d] : Bf16{0};
      if (with_v)
        v_s[j][d] = off >= 0 ? kv_layer[off + v_half + d] : Bf16{0};
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

} // namespace

namespace turbine_hip {

int32_t launch_paged_append(turbine_ctx *ctx,
                            const turbine_attention_paged_desc *d) {
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
  hipLaunchKernelGGL(
      paged_attention_kernel,
      dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(d->num_seqs),
           static_cast<unsigned>(d->num_kv_heads)),
      dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(d->q),
      static_cast<const Bf16 *>(d->kv_layer), static_cast<Bf16 *>(d->out),
      d->block_table, d->q_indptr, d->kv_lens, d->max_blocks_per_seq,
      d->num_blocks, d->block_tokens, d->num_kv_heads, group, q_tile,
      d->q_stride_token, d->out_stride_token, d->scale, d->causal);
  return check_hip(ctx, hipGetLastError(), "paged attention kernel launch");
}

} // namespace turbine_hip
