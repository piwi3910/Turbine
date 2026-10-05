// The Turbine batch-1 paged decode attention (impl "turbine_hip_b1_decode" of
// attention_decode_paged; decision "5p decode-BW floor", user decision B, with
// the provider evaluation of 2026-10-05): Composable Kernel's fmha_fwd_splitkv
// is occupancy-bound at small batches -- one workgroup per (sequence, KV head)
// is 8 workgroups at batch 1 -- and its fixed structure costs three launches
// per layer (the Turbine append, the split-KV kernel, CK's combine) and their
// inter-kernel boundaries. This kernel is ONE launch per layer:
//
// - grid (sequences, KV heads, splits): each workgroup attends its KV slice,
//   so batch 1 fills the card (8 KV heads x kSplits workgroups) instead of 8;
// - the k_new/v_new append is fused: the workgroup whose key range holds the
//   sequence's new token (position kv_lens[s] - 1) writes it before scoring,
//   for its own KV head only, so no other workgroup depends on it;
// - each workgroup writes its partial (per head: the P.V accumulator, the row
//   maximum and the sum) into a context workspace, fences, and counts itself
//   in a per-(sequence, KV head) arrival counter; the last-arriving workgroup
//   combines the splits in ascending split order, writes the output rows and
//   resets the counter to zero for the next call (device memory, so the decode
//   graphs capture the call; the workspace pointer is a pool constant once
//   grown, and the counters are always zero between calls).
//
// Batch invariance: kSplits is a constant and every sequence is independent,
// so a row's bits do not depend on the batch it decodes in (the same
// small-m MoE tier's within-tier property). Kernel selection is per op config
// (the registry binds one implementation index; the batch size is not part of
// the config), so this entry receives every decode batch; paged_attention.cpp
// hands batches of kTurbineB1CkSeqs sequences and more to CK fmha_fwd_splitkv,
// faster there, whenever CK serves the page size.
//
// Numerics (a numerics class of its own; not bitwise CK): scores q.k
// accumulated in F32 over the head dimension in chunks of ascending index
// (one lane per key, the K row read as 16 uint4), an online softmax with a
// running maximum per wave (waves combined in wave order, splits in ascending
// split order against the overall maximum), and P kept in F32 through the
// P.V product -- unrounded, so a split's local maximum costs only the F32
// rescale roundings, unlike CK's multi-split BF16 P ("Pre-Phase-5 #3"). V is
// read exactly (BF16 -> F32). The row is divided by the sum and rounded to
// BF16 at the end.
#include <string>

#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::fail;

namespace {

constexpr int kHeadDim = 128;
constexpr float kNegInf = -__builtin_huge_valf();
constexpr int kThreads = 256;
// Lanes of a wave the kernel is written for (as the FP8 decode kernel).
constexpr int kDecodeLanes = 32;
constexpr int kDecodeWaves = kThreads / kDecodeLanes;
// Head dimensions per lane in the P.V product.
constexpr int kDimsPerLane = kHeadDim / kDecodeLanes;
static_assert(kDimsPerLane == 4, "each lane owns 4 head dimensions");

// The fixed split count (see the file comment): 8 KV-head workgroups become
// 64 at batch 1.
constexpr int32_t kSplits = 8;
// Keys one lane scores per round (its K row: 128 BF16 = 16 uint4).
constexpr int kKWords = kHeadDim * 2 / 16;
static_assert(kKWords == 16, "a K row is 16 uint4 words");

// One lane's BF16 pair (two elements of a uint32 word).
__device__ inline void bf16_pair(uint32_t w, float &lo, float &hi) {
  lo = bf16_to_f32(Bf16{static_cast<uint16_t>(w & 0xffffu)});
  hi = bf16_to_f32(Bf16{static_cast<uint16_t>(w >> 16)});
}

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

// LDS of one workgroup: the loop's Q and P offsets, then (after a barrier) the
// waves' partial outputs.
template <int G> struct B1Lds {
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

// Paged decode over BF16 pages: grid (sequences, KV heads, kSplits). Every
// sequence has one query (row q_indptr[s]) that sees all kv_lens[s] keys.
// Workgroup (s, h, split) attends keys [split * chunk, ...) with
// chunk = ceil(kv_len / kSplits), appending k_new/v_new when its range holds
// key kv_len - 1.
template <int G>
__global__ void __launch_bounds__(kThreads)
    paged_b1_decode_kernel(const Bf16 *q, Bf16 *kv_layer, const Bf16 *k_new,
                           const Bf16 *v_new, Bf16 *out, float *partials,
                           int32_t *counters, const int32_t *block_table,
                           const int32_t *q_indptr, const int32_t *kv_lens,
                           int32_t max_blocks, int32_t num_blocks,
                           int32_t block_tokens, int32_t kv_heads,
                           int64_t q_stride, int64_t new_stride,
                           int64_t out_stride, float scale) {
  __shared__ B1Lds<G> lds;
  __shared__ int32_t is_last;

  const int tid = threadIdx.x;
  const int lane = tid % kDecodeLanes;
  const int wave = tid / kDecodeLanes;
  const int32_t seq = blockIdx.x;
  const int32_t kv_head = blockIdx.y;
  const int32_t split = blockIdx.z;
  const int32_t row = q_indptr[seq];
  const int32_t kv_len = kv_lens[seq];
  const int32_t chunk = (kv_len + kSplits - 1) / kSplits;
  const int32_t k0 = split * chunk;
  const int32_t k1 = min(kv_len, k0 + chunk);
  const int64_t token_elems = static_cast<int64_t>(kv_heads) * kHeadDim;
  const int64_t v_half = static_cast<int64_t>(block_tokens) * token_elems;
  const int32_t *table = block_table + static_cast<int64_t>(seq) * max_blocks;

  // The append, fused: this workgroup's KV head's new row, when its key range
  // holds the sequence's new token (position kv_len - 1) -- exactly one split
  // of the (sequence, KV head) does. No other workgroup reads that key.
  if (k0 <= kv_len - 1 && kv_len - 1 < k1) {
    const int32_t last = kv_len - 1;
    const int32_t page = last / block_tokens;
    if (page < max_blocks) {
      const int32_t block = table[page];
      if (block >= 0 && block < num_blocks) {
        const int32_t slot = last % block_tokens;
        Bf16 *k_dst = kv_layer +
                      (static_cast<int64_t>(block) * 2 * block_tokens + slot) *
                          token_elems +
                      static_cast<int64_t>(kv_head) * kHeadDim;
        Bf16 *v_dst = k_dst + v_half;
        const Bf16 *k_src = k_new + static_cast<int64_t>(row) * new_stride +
                            static_cast<int64_t>(kv_head) * kHeadDim;
        const Bf16 *v_src = v_new + static_cast<int64_t>(row) * new_stride +
                            static_cast<int64_t>(kv_head) * kHeadDim;
        for (int d = tid; d < kHeadDim; d += kThreads) {
          k_dst[d] = k_src[d];
          v_dst[d] = v_src[d];
        }
      }
    }
  }
  __syncthreads();

  for (int e = tid; e < G * kHeadDim; e += kThreads) {
    const int g = e / kHeadDim;
    const int d = e % kHeadDim;
    // The launcher instantiates G == group, so every g is a real head.
    const int32_t h = kv_head * G + g;
    lds.loop.q[g][d] = bf16_to_f32(q[static_cast<int64_t>(row) * q_stride +
                                     static_cast<int64_t>(h) * kHeadDim + d]);
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

  for (int32_t c0 = k0 + wave * kDecodeLanes; c0 < k1;
       c0 += kDecodeWaves * kDecodeLanes) {
    const int32_t key = c0 + lane;
    int64_t off = -1;
    if (key < k1) {
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
      for (int c = 0; c < kKWords; ++c) {
        const uint4 w = row16[c];
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
        float k[8];
#pragma unroll
        for (int wd = 0; wd < 4; ++wd)
          bf16_pair(words[wd], k[2 * wd], k[2 * wd + 1]);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const float4 *qv =
              reinterpret_cast<const float4 *>(&lds.loop.q[g][c * 8]);
          float dot = s[g];
#pragma unroll
          for (int b4 = 0; b4 < 2; ++b4) {
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
      // P stays F32 (the file comment's numerics class).
      lds.loop.p[wave][g][lane] = p[g];
    }
    lds.loop.voff[wave][lane] = off < 0 ? -1 : off + v_half;
    wave_sync();
    const int32_t keys = min(kDecodeLanes, k1 - c0);
    for (int j = 0; j < keys; ++j) {
      const int64_t voff = lds.loop.voff[wave][j];
      if (voff < 0)
        continue;
      const uint2 w4 = *reinterpret_cast<const uint2 *>(kv_layer + voff +
                                                        lane * kDimsPerLane);
      float v[kDimsPerLane];
      bf16_pair(w4.x, v[0], v[1]);
      bf16_pair(w4.y, v[2], v[3]);
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

  // The block's partial: waves combined in wave order against the block
  // maximum, written to the workspace.
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

  float *dst = partials +
               (static_cast<int64_t>(seq) * kv_heads + kv_head) * kSplits * G *
                   (kHeadDim + 2) +
               static_cast<int64_t>(split) * G * (kHeadDim + 2);
  for (int e = tid; e < G * kHeadDim; e += kThreads) {
    const int g = e / kHeadDim;
    const int d = e % kHeadDim;
    float mx = kNegInf;
#pragma unroll
    for (int w = 0; w < kDecodeWaves; ++w)
      mx = fmaxf(mx, lds.max[w][g]);
    float o = 0.0f;
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kDecodeWaves; ++w) {
      // A wave that saw no key has max -inf and adds nothing.
      if (lds.max[w][g] == kNegInf)
        continue;
      const float f = expf(lds.max[w][g] - mx);
      o += lds.o[w][g][d] * f;
      total += lds.sum[w][g] * f;
    }
    dst[g * (kHeadDim + 2) + d] = o;
    if (d == 0) {
      dst[g * (kHeadDim + 2) + kHeadDim] = mx;
      dst[g * (kHeadDim + 2) + kHeadDim + 1] = total;
    }
  }
  // The writes above must be visible to whichever workgroup combines before
  // this one counts itself in.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    is_last =
        atomicAdd(counters + static_cast<int64_t>(seq) * kv_heads + kv_head,
                  1) == kSplits - 1;
  }
  __syncthreads();
  if (!is_last)
    return;

  // The last-arriving workgroup combines the splits in ascending split order
  // (deterministic whatever the arrival order) and resets the counter for the
  // next call.
  __threadfence();
  for (int e = tid; e < G * kHeadDim; e += kThreads) {
    const int g = e / kHeadDim;
    const int d = e % kHeadDim;
    const float *base =
        partials + (static_cast<int64_t>(seq) * kv_heads + kv_head) * kSplits *
                       G * (kHeadDim + 2);
    float mx = kNegInf;
    for (int sp = 0; sp < kSplits; ++sp)
      mx = fmaxf(mx, base[static_cast<int64_t>(sp) * G * (kHeadDim + 2) +
                          g * (kHeadDim + 2) + kHeadDim]);
    float o = 0.0f;
    float total = 0.0f;
    for (int sp = 0; sp < kSplits; ++sp) {
      const float *sb = base + static_cast<int64_t>(sp) * G * (kHeadDim + 2) +
                        g * (kHeadDim + 2);
      // A split with no keys has max -inf and adds nothing.
      if (sb[kHeadDim] == kNegInf)
        continue;
      const float f = expf(sb[kHeadDim] - mx);
      o += sb[d] * f;
      total += sb[kHeadDim + 1] * f;
    }
    const int32_t h = kv_head * G + g;
    out[static_cast<int64_t>(row) * out_stride +
        static_cast<int64_t>(h) * kHeadDim + d] = f32_to_bf16(o / total);
  }
  if (tid == 0)
    counters[static_cast<int64_t>(seq) * kv_heads + kv_head] = 0;
}

} // namespace

namespace turbine_hip {

namespace {
// One launch of the b1 kernel at group G (hipLaunchKernelGGL needs the kernel
// spelled out, so the group goes through a template parameter, not a lambda).
template <int G>
int32_t launch_b1_decode(turbine_ctx *ctx,
                         const turbine_attention_paged_desc *d,
                         const dim3 &grid, float *partials, int32_t *counters) {
  hipLaunchKernelGGL(
      paged_b1_decode_kernel<G>, grid, dim3(kThreads), 0, ctx->stream,
      static_cast<const Bf16 *>(d->q), static_cast<Bf16 *>(d->kv_layer),
      static_cast<const Bf16 *>(d->k_new), static_cast<const Bf16 *>(d->v_new),
      static_cast<Bf16 *>(d->out), partials, counters, d->block_table,
      d->q_indptr, d->kv_lens, d->max_blocks_per_seq, d->num_blocks,
      d->block_tokens, d->num_kv_heads, d->q_stride_token, d->new_stride_token,
      d->out_stride_token, d->scale);
  return check_hip(ctx, hipGetLastError(), "b1 decode kernel launch");
}
} // namespace

// The group's query-head count: the launcher picks the template from it.
int32_t paged_b1_group(const turbine_attention_paged_desc *d) {
  return d->num_q_heads / d->num_kv_heads;
}

bool b1_serves(const turbine_attention_paged_desc *d) {
  // Decode (one query per sequence) of grouped query heads, at most
  // kTurbineB1MaxGroup heads per KV head (the instantiated templates), any
  // number of sequences.
  return d->max_q_len <= 1 && d->total_q <= d->num_seqs &&
         d->num_q_heads > d->num_kv_heads &&
         paged_b1_group(d) <= kTurbineB1MaxGroup;
}

int32_t run_b1_decode(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                      const std::string & /*entry (the launcher names its own
                      launch error)*/) {
  const int32_t group = paged_b1_group(d);
  // Two device buffers, each grown outside a capture only (a decode graph's
  // capture never needs more than the eager run of the same key before it);
  // a replaced buffer is retired, not freed, because a captured graph may
  // still use it:
  // - the partials, one (accumulator, maximum, sum) record per (sequence, KV
  //   head, split, head), written before they are read in every call;
  // - the arrival counters, one per (sequence, KV head), zeroed when the
  //   buffer is created and returned to zero by the kernel at the end of every
  //   call. They live in a buffer of their own so that no other shape's
  //   partials (another decode graph's batch size) can land on them: every
  //   graph and every eager call finds them zero, whatever ran before.
  const int64_t per_sh = static_cast<int64_t>(d->num_seqs) * d->num_kv_heads;
  const int64_t partial_floats = per_sh * kSplits * group * (kHeadDim + 2);
  auto grow = [&](void *&buf, size_t &have, size_t need, bool zero,
                  const char *what) -> int32_t {
    if (have >= need)
      return TURBINE_OK;
    if (ctx->capturing)
      return refuse_while_capturing(ctx, what);
    size_t grown = 1;
    while (grown < need)
      grown <<= 1;
    void *p = nullptr;
    const std::string msg = std::string("hipMalloc ") + what + " " +
                            std::to_string(grown) + " bytes";
    if (int32_t rc = check_hip(ctx, hipMalloc(&p, grown), msg.c_str());
        rc != TURBINE_OK)
      return rc;
    if (zero) {
      if (int32_t rc = check_hip(ctx, hipMemsetAsync(p, 0, grown, ctx->stream),
                                 "b1 attention counter zeroing");
          rc != TURBINE_OK)
        return rc;
    }
    if (buf != nullptr)
      ctx->attn_split_retired.push_back(buf);
    buf = p;
    have = grown;
    return TURBINE_OK;
  };
  if (int32_t rc = grow(ctx->attn_b1_scratch, ctx->attn_b1_scratch_bytes,
                        static_cast<size_t>(partial_floats) * 4, false,
                        "b1 attention partials");
      rc != TURBINE_OK)
    return rc;
  if (int32_t rc =
          grow(ctx->attn_b1_counters, ctx->attn_b1_counters_bytes,
               static_cast<size_t>(per_sh) * 4, true, "b1 attention counters");
      rc != TURBINE_OK)
    return rc;
  auto *partials = static_cast<float *>(ctx->attn_b1_scratch);
  auto *counters = static_cast<int32_t *>(ctx->attn_b1_counters);

  const dim3 grid(static_cast<unsigned>(d->num_seqs),
                  static_cast<unsigned>(d->num_kv_heads),
                  static_cast<unsigned>(kSplits));
  switch (group) {
  case 1:
    return launch_b1_decode<1>(ctx, d, grid, partials, counters);
  case 2:
    return launch_b1_decode<2>(ctx, d, grid, partials, counters);
  case 3:
    return launch_b1_decode<3>(ctx, d, grid, partials, counters);
  case 4:
    return launch_b1_decode<4>(ctx, d, grid, partials, counters);
  case 5:
    return launch_b1_decode<5>(ctx, d, grid, partials, counters);
  case 6:
    return launch_b1_decode<6>(ctx, d, grid, partials, counters);
  case 7:
    return launch_b1_decode<7>(ctx, d, grid, partials, counters);
  default:
    return launch_b1_decode<8>(ctx, d, grid, partials, counters);
  }
}

} // namespace turbine_hip
