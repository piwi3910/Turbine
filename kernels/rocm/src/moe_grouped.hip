// Turbine HIP kernels for moe_experts with many routed rows (impl
// "turbine_hip_moe_wmma", above kMoeSmallMaxRows = num_tokens * top_k): a
// grouped GEMM on the RDNA4 matrix cores (v_wmma_f32_16x16x16_bf16) that reads
// every group size from expert_offsets on the device, so, like the small-m
// path, the caller needs no host copy of the offsets and the step makes no
// device-to-host copy. hipBLASLt has no grouped GEMM for gfx1201 in ROCm
// 7.14.1, and one hipBLASLt call per expert and projection costs ~3.6 ms of
// host time per layer (192 calls for OLMoE) whatever the row count.
//
//   1. positions (moe_small_m.hip): pos[sorted_rows[i]] = i for the local rows,
//      -1 for the others;
//   2. gate-up: one block per (row tile of one expert, column tile of the
//      intermediate) computes gate and up for its tile from the tokens'
//      activations (x[sorted_rows[p] / top_k], gathered on load) and writes
//      act[p] = bf16(round(silu(round(gate))) * round(up));
//   3. down: the same tiling over the hidden columns of the down projection,
//      down[p] = bf16(act[p] . w_down);
//   4. the scatter of moe.hip adds the rows to out in ascending position,
//      which is ascending expert id.
// The grid holds (local rows / tile rows + local experts) row tiles times the
// column tiles, column tiles fastest: a block finds its (expert, row tile) from
// expert_offsets (blocks past the last row tile return at once). The column
// tiles of a row tile run together and share its activation rows in the
// caches, and the row tiles of one expert follow each other closely enough to
// share its weights.
//
// A block's waves (kWavesM x kWavesN) each own kFragsM x kFragsN WMMA tiles of
// the block tile (for gate-up, of both projections). The depth is walked in
// steps of kTileK: the threads load the step's activation and weight tiles
// with coalesced 16-byte loads (8 threads per 128-byte tile row) and store them
// into LDS; the loads run kPrefetch steps ahead of the WMMAs. The gfx12 wave32
// operand layouts: lane l holds A[l % 16][8 (l / 16) .. + 8) (one 16-byte LDS
// read of an activation row), B[8 (l / 16) .. + 8)[l % 16] (of a weight row,
// K-contiguous) and C[8 (l / 16) + v][l % 16] for v < 8.
//
// Numerics follow the cpu-reference moe_experts: F32 accumulation, gate and up
// rounded to BF16, silu rounded, the product rounded, the down projection
// rounded. The summation order is fixed by the kernel (16-wide WMMA steps in
// ascending k), so two runs are bitwise identical.
#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::kMoeMaxExperts;
using turbine_hip::round_bf16;

namespace {

constexpr int kLanes = 32;
// WMMA tile edge and depth.
constexpr int kMma = 16;
// Depth of one pipeline step (one LDS tile).
constexpr int kTileK = 64;
// LDS row of a tile: kTileK BF16 plus 8 of padding (144 bytes), so the 16 rows
// one WMMA operand load reads fall on distinct banks.
constexpr int kLdsRow = kTileK + 8;
// 16-byte chunks per tile row.
constexpr int kChunks = kTileK / 8;

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
// 16 bytes of operands in flight. A clang vector type, not HIP's u32x4 struct:
// arrays of the struct stay in scratch memory, arrays of the vector in VGPRs.
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ inline v8f mma(v8s a, v8s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// The (local expert, first position, end position) of row tile `tile` of
// kTileM rows, or expert -1 when the grid has more row tiles than the experts
// need. Called by every thread of the block.
template <int kThreads, int kTileM>
__device__ void find_tile(const int32_t *offsets, int32_t expert_begin,
                          int32_t num_local, int32_t tile, int32_t *expert,
                          int32_t *begin, int32_t *end) {
  static_assert(kMoeMaxExperts % kThreads == 0,
                "the tile scan covers the experts in whole passes");
  __shared__ int32_t ends[kMoeMaxExperts];
  __shared__ int32_t found[3];
  const int t = static_cast<int>(threadIdx.x);
  if (t == 0)
    found[0] = -1;
  // Row tiles per expert, then an inclusive scan (Hillis-Steele) over them.
  for (int32_t e = t; e < kMoeMaxExperts; e += kThreads) {
    ends[e] = e < num_local ? (offsets[expert_begin + e + 1] -
                               offsets[expert_begin + e] + kTileM - 1) /
                                  kTileM
                            : 0;
  }
  __syncthreads();
  for (int32_t step = 1; step < num_local; step *= 2) {
    int32_t add[kMoeMaxExperts / kThreads];
#pragma unroll
    for (int i = 0; i < kMoeMaxExperts / kThreads; ++i) {
      const int32_t e = t + i * kThreads;
      add[i] = e < num_local && e >= step ? ends[e - step] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kMoeMaxExperts / kThreads; ++i)
      ends[t + i * kThreads] += add[i];
    __syncthreads();
  }
  for (int32_t e = t; e < num_local; e += kThreads) {
    const int32_t first = e == 0 ? 0 : ends[e - 1];
    if (tile >= first && tile < ends[e]) {
      found[0] = e;
      found[1] = offsets[expert_begin + e] + (tile - first) * kTileM;
      found[2] = offsets[expert_begin + e + 1];
    }
  }
  __syncthreads();
  *expert = found[0];
  *begin = found[1];
  *end = found[2];
}

// A block shape: kWavesM x kWavesN waves, each kFragsM x kFragsN WMMA tiles,
// global loads kPrefetch steps ahead.
template <int WavesM, int WavesN, int FragsM, int FragsN, int Prefetch>
struct Shape {
  static constexpr int kWavesM = WavesM;
  static constexpr int kWavesN = WavesN;
  static constexpr int kFragsM = FragsM;
  static constexpr int kFragsN = FragsN;
  static constexpr int kPrefetch = Prefetch;
  static constexpr int kThreads = kLanes * WavesM * WavesN;
  static constexpr int kTileM = WavesM * FragsM * kMma;
  static constexpr int kTileN = WavesN * FragsN * kMma;
  static constexpr int kRowsPerPass = kThreads / kChunks;
  static constexpr int kPassesA = kTileM / kRowsPerPass;
  static constexpr int kPassesB = kTileN / kRowsPerPass;
  static_assert(kPassesA * kRowsPerPass == kTileM &&
                    kPassesB * kRowsPerPass == kTileN,
                "the load passes cover the tiles");
  static_assert(Prefetch >= 1, "the loads run at least one step ahead");
};

// The block shape: 8 waves of 1 x 2 WMMA tiles (a 64 x 64 tile), one step
// loaded ahead. A lab sweep on the R9700 (OLMoE, 128/680/2,048 tokens: waves
// 4x2 and 2x2 with 1x2, 2x2, 2x4 and 4x2 tiles, one or two steps ahead) found
// every shape without spills within 3% of this one (the 2,048-token case runs
// at ~70 TFLOPS, the others are bound by streaming the experts' weights), and
// the larger tiles slower.
using Block = Shape<4, 2, 1, 2, 1>;

// One projection of a (row tile, column tile) of one expert.
//   kGateUp: out[p, n] = bf16(round(silu(round(x_p . w0_n))) * round(x_p .
//            w1_n)) with x_p = in[sorted_rows[p] / top_k], n < inter;
//   else:    out[p, n] = bf16(in[p] . w0_n), n < hidden.
// w0/w1 are [num_local_experts, n_out, depth]; out is [rows, n_out], indexed
// by position.
template <typename S, bool kGateUp>
__global__ __launch_bounds__(S::kThreads) void moe_wmma_kernel(
    const Bf16 *in, const Bf16 *w0, const Bf16 *w1, const int32_t *sorted_rows,
    const int32_t *offsets, int32_t expert_begin, int32_t num_local,
    int32_t top_k, int32_t depth, int32_t n_out, Bf16 *out) {
  constexpr int kPA = S::kPassesA;
  constexpr int kPB = S::kPassesB;
  constexpr int kPF = S::kPrefetch;
  __shared__ __attribute__((aligned(16))) Bf16 a_s[S::kTileM * kLdsRow];
  __shared__ __attribute__((aligned(16))) Bf16 b0_s[S::kTileN * kLdsRow];
  __shared__
      __attribute__((aligned(16))) Bf16 b1_s[kGateUp ? S::kTileN * kLdsRow : 8];
  // Blocks walk the column tiles of one row tile before the next row tile.
  const int32_t n_tiles = (n_out + S::kTileN - 1) / S::kTileN;
  const int32_t tile_n =
      static_cast<int32_t>(blockIdx.x % static_cast<unsigned>(n_tiles)) *
      S::kTileN;
  int32_t local;
  int32_t begin;
  int32_t end;
  find_tile<S::kThreads, S::kTileM>(
      offsets, expert_begin, num_local,
      static_cast<int32_t>(blockIdx.x / static_cast<unsigned>(n_tiles)), &local,
      &begin, &end);
  if (local < 0)
    return;
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t % kLanes;
  const int wave = t / kLanes;

  // Global chunks of this thread: tile rows t / kChunks + pass * kRowsPerPass,
  // 16 bytes at column (t % kChunks) * 8 of the step. Rows past the expert's
  // last position re-read its first, columns past n_out the last; their
  // results are dropped.
  const int chunk = t % kChunks;
  const int row0 = t / kChunks;
  const size_t expert_elems = static_cast<size_t>(n_out) * depth;
  const Bf16 *a_g[kPA];
  const Bf16 *b0_g[kPB];
  const Bf16 *b1_g[kPB];
#pragma unroll
  for (int pass = 0; pass < kPA; ++pass) {
    int32_t p = begin + row0 + pass * S::kRowsPerPass;
    if (p >= end)
      p = begin;
    const int64_t src = kGateUp ? sorted_rows[p] / top_k : p;
    a_g[pass] = in + src * depth + chunk * 8;
  }
#pragma unroll
  for (int pass = 0; pass < kPB; ++pass) {
    const int32_t n = min(tile_n + row0 + pass * S::kRowsPerPass, n_out - 1);
    const size_t w_row = local * expert_elems + static_cast<size_t>(n) * depth;
    b0_g[pass] = w0 + w_row + chunk * 8;
    b1_g[pass] = kGateUp ? w1 + w_row + chunk * 8 : b0_g[pass];
  }

  // Wave tile: rows wm * kFragsM * 16, columns wn * kFragsN * 16.
  const int wm = wave / S::kWavesN;
  const int wn = wave % S::kWavesN;
  const int frag_row = lane % kMma;
  const int frag_k = (lane / kMma) * 8;
  v8f acc0[S::kFragsM][S::kFragsN];
  v8f acc1[S::kFragsM][S::kFragsN];
#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i) {
#pragma unroll
    for (int j = 0; j < S::kFragsN; ++j) {
      acc0[i][j] = v8f{};
      acc1[i][j] = v8f{};
    }
  }

  // kPF steps of chunks in flight: buffer b holds step k0 + b kTileK.
  u32x4 ra[kPF][kPA];
  u32x4 rb0[kPF][kPB];
  u32x4 rb1[kPF][kPB];
#pragma unroll
  for (int b = 0; b < kPF; ++b) {
#pragma unroll
    for (int pass = 0; pass < kPA; ++pass)
      ra[b][pass] = *reinterpret_cast<const u32x4 *>(a_g[pass] + b * kTileK);
#pragma unroll
    for (int pass = 0; pass < kPB; ++pass) {
      rb0[b][pass] = *reinterpret_cast<const u32x4 *>(b0_g[pass] + b * kTileK);
      if (kGateUp)
        rb1[b][pass] =
            *reinterpret_cast<const u32x4 *>(b1_g[pass] + b * kTileK);
    }
  }
  for (int32_t k0 = 0; k0 < depth; k0 += kPF * kTileK) {
#pragma unroll
    for (int b = 0; b < kPF; ++b) {
      // Store buffer b (step k0 + b kTileK) into LDS, refill it with the step
      // kPF ahead, then run the WMMAs of the step in LDS.
      __syncthreads();
#pragma unroll
      for (int pass = 0; pass < kPA; ++pass) {
        const int at = (row0 + pass * S::kRowsPerPass) * kLdsRow + chunk * 8;
        *reinterpret_cast<u32x4 *>(&a_s[at]) = ra[b][pass];
      }
#pragma unroll
      for (int pass = 0; pass < kPB; ++pass) {
        const int at = (row0 + pass * S::kRowsPerPass) * kLdsRow + chunk * 8;
        *reinterpret_cast<u32x4 *>(&b0_s[at]) = rb0[b][pass];
        if (kGateUp)
          *reinterpret_cast<u32x4 *>(&b1_s[at]) = rb1[b][pass];
      }
      __syncthreads();
      const int32_t next = k0 + (b + kPF) * kTileK;
      if (next < depth) {
#pragma unroll
        for (int pass = 0; pass < kPA; ++pass)
          ra[b][pass] = *reinterpret_cast<const u32x4 *>(a_g[pass] + next);
#pragma unroll
        for (int pass = 0; pass < kPB; ++pass) {
          rb0[b][pass] = *reinterpret_cast<const u32x4 *>(b0_g[pass] + next);
          if (kGateUp)
            rb1[b][pass] = *reinterpret_cast<const u32x4 *>(b1_g[pass] + next);
        }
      }
#pragma unroll
      for (int kk = 0; kk < kTileK; kk += kMma) {
        v8s a[S::kFragsM];
#pragma unroll
        for (int i = 0; i < S::kFragsM; ++i) {
          a[i] = *reinterpret_cast<const v8s *>(
              &a_s[((wm * S::kFragsM + i) * kMma + frag_row) * kLdsRow + kk +
                   frag_k]);
        }
#pragma unroll
        for (int j = 0; j < S::kFragsN; ++j) {
          const int at =
              ((wn * S::kFragsN + j) * kMma + frag_row) * kLdsRow + kk + frag_k;
          const v8s b0 = *reinterpret_cast<const v8s *>(&b0_s[at]);
#pragma unroll
          for (int i = 0; i < S::kFragsM; ++i)
            acc0[i][j] = mma(a[i], b0, acc0[i][j]);
          if (kGateUp) {
            const v8s b1 = *reinterpret_cast<const v8s *>(&b1_s[at]);
#pragma unroll
            for (int i = 0; i < S::kFragsM; ++i)
              acc1[i][j] = mma(a[i], b1, acc1[i][j]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i) {
#pragma unroll
    for (int j = 0; j < S::kFragsN; ++j) {
      const int32_t n = tile_n + (wn * S::kFragsN + j) * kMma + frag_row;
      if (n >= n_out)
        continue;
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const int32_t p =
            begin + (wm * S::kFragsM + i) * kMma + (lane / kMma) * 8 + v;
        if (p >= end)
          continue;
        float value = acc0[i][j][v];
        if (kGateUp) {
          const float g = round_bf16(value);
          const float u = round_bf16(acc1[i][j][v]);
          value = round_bf16(g / (1.0f + expf(-g))) * u;
        }
        out[static_cast<size_t>(p) * n_out + n] = f32_to_bf16(value);
      }
    }
  }
}

template <typename S, bool kGateUp>
int32_t launch_proj(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                    const void *in, const void *w0, const void *w1,
                    int32_t depth, int32_t n_out, void *out) {
  const int32_t rows = d->num_tokens * d->top_k;
  const int32_t num_local = d->expert_end - d->expert_begin;
  // Every expert's last row tile may be partial: at most one extra tile each.
  const int64_t row_tiles = (rows + S::kTileM - 1) / S::kTileM + num_local;
  const int64_t n_tiles = (n_out + S::kTileN - 1) / S::kTileN;
  hipLaunchKernelGGL(
      (moe_wmma_kernel<S, kGateUp>),
      dim3(static_cast<unsigned>(row_tiles * n_tiles)), dim3(S::kThreads), 0,
      ctx->stream, static_cast<const Bf16 *>(in), static_cast<const Bf16 *>(w0),
      static_cast<const Bf16 *>(w1), d->sorted_rows, d->expert_offsets,
      d->expert_begin, num_local, d->top_k, depth, n_out,
      static_cast<Bf16 *>(out));
  return check_hip(ctx, hipGetLastError(),
                   kGateUp ? "moe wmma gate-up kernel launch"
                           : "moe wmma down kernel launch");
}

template <typename S>
int32_t launch_shape(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                     void *act, void *down) {
  if (int32_t rc = launch_proj<S, true>(ctx, d, d->x, d->w_gate, d->w_up,
                                        d->hidden, d->inter, act);
      rc != TURBINE_OK) {
    return rc;
  }
  return launch_proj<S, false>(ctx, d, act, d->w_down, nullptr, d->inter,
                               d->hidden, down);
}

} // namespace

namespace turbine_hip {

bool moe_wmma_shape(const turbine_moe_experts_desc *d) {
  constexpr int kStep = Block::kPrefetch * kTileK;
  return d->hidden % kStep == 0 && d->inter % kStep == 0;
}

int32_t launch_moe_wmma(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                        int32_t *pos, void *act, void *down) {
  if (int32_t rc = launch_moe_positions(ctx, d, pos); rc != TURBINE_OK)
    return rc;
  return launch_shape<Block>(ctx, d, act, down);
}

} // namespace turbine_hip
