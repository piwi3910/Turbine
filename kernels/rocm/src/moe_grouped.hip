// Turbine HIP kernels for moe_experts with many routed rows (impls
// "turbine_hip_moe_wmma_prefill", the default above the card profile's
// moe_small_max_rows routed rows, num_tokens * top_k, see "the prefill
// kernels" below, and "turbine_hip_moe_wmma", described here): a
// grouped GEMM on the RDNA4 matrix cores (v_wmma_f32_16x16x16_bf16) that reads
// every group size from expert_offsets on the device, so, like the small-m
// path, the caller needs no host copy of the offsets and the step makes no
// device-to-host copy. hipBLASLt has no grouped GEMM for RDNA4 in ROCm
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
// ascending k), so two runs are bitwise identical, and so is a row computed by
// the small-row blocks below (the small-m tier): the result of a row does not
// depend on the batch it is in.
#include <algorithm>

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

// The small-row blocks (the small-m implementation, launch_moe_small_m, for
// shapes of multiples of 64): 4 waves over one 16-row WMMA tile of the rows and
// 64 columns, so a decode batch with a few rows per expert computes 16 rows per
// tile instead of 64 and spreads each expert's weights over more blocks. Loads
// run two steps ahead where the depth allows it (multiples of 128).
//
// Batch invariance: every output element is the same WMMA chain in every block
// shape -- a zero accumulator, then v_wmma over 16-wide k slices in ascending k
// -- and a WMMA result element depends only on its row of A, its column of B
// and its accumulator. So a row's gate, up, activation and down values are
// bitwise the same whichever shape runs it, i.e. whatever the batch's routed
// row count (hip_batch_invariance, moe_rows_are_batch_invariant). Tiling and
// prefetch depth may change; the chain may not.
using SmallBlock = Shape<1, 4, 1, 1, 2>;
using SmallBlockK64 = Shape<1, 4, 1, 1, 1>;

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

// ---- the prefill kernels (impl "turbine_hip_moe_wmma_prefill") ----
//
// The same per-element WMMA chain as moe_wmma_kernel (a zero accumulator, one
// v_wmma per 16-wide k slice in ascending k, the same lane layout and
// epilogue), so a row's gate, up, activation and down values are bitwise those
// of the other tiers. Built for large routed-row counts, where moe_wmma_kernel
// is bound by the traffic of staging both operands through LDS (every k step
// writes the A and B tiles to LDS and every wave reads its fragments back):
//
//   - A (the gathered activations, shared by every wave of the block) is
//     staged in LDS as before, double-buffered: each k step issues the next
//     step's loads, runs the WMMAs of the current stage, stores the loads into
//     the other stage and synchronizes once;
//   - B (the expert weights, K-contiguous rows, exactly the WMMA B lane
//     layout) goes straight from global memory into each wave's registers, one
//     k step ahead; the waves of a block split the columns (kWavesM = 1), so
//     no weight is read twice by a block and LDS carries A only.
//
// Each wave holds kFragsM row fragments (the block's whole row tile) and
// kFragsN column fragments per projection.
template <int WavesM, int WavesN, int FragsM, int FragsN, int TileK>
struct Stream {
  static constexpr int kWavesM = WavesM;
  static constexpr int kWavesN = WavesN;
  static constexpr int kFragsM = FragsM;
  static constexpr int kFragsN = FragsN;
  static constexpr int kTileK = TileK;
  static constexpr int kThreads = kLanes * WavesM * WavesN;
  static constexpr int kTileM = WavesM * FragsM * kMma;
  static constexpr int kTileN = WavesN * FragsN * kMma;
  static constexpr int kLdsRow = TileK + 8;
  static constexpr int kChunks = TileK / 8;
  static constexpr int kRowsPerPass = kThreads / kChunks;
  static constexpr int kPassesA =
      kTileM >= kRowsPerPass ? kTileM / kRowsPerPass : 1;
  static constexpr int kSlices = TileK / kMma;
  static_assert(kTileM % kRowsPerPass == 0 || kRowsPerPass % kTileM == 0,
                "the A load passes cover the row tile");
};

template <typename S, bool kGateUp>
__global__ __launch_bounds__(S::kThreads) void moe_stream_kernel(
    const Bf16 *in, const Bf16 *w0, const Bf16 *w1, const int32_t *sorted_rows,
    const int32_t *offsets, int32_t expert_begin, int32_t num_local,
    int32_t top_k, int32_t depth, int32_t n_out, Bf16 *out) {
  constexpr int kPA = S::kPassesA;
  constexpr int kA = S::kTileM * S::kLdsRow;
  constexpr int kFN = S::kFragsN;
  constexpr int kSl = S::kSlices;
  __shared__ __attribute__((aligned(16))) Bf16 a_s[2 * kA];
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
  // A chunks: tile rows t / kChunks + pass * kRowsPerPass, 16 bytes at column
  // (t % kChunks) * 8 of the step; rows past the expert's last position re-read
  // its first, and threads past the tile (a tile smaller than one pass) load
  // nothing. Their results are dropped.
  const int chunk = t % S::kChunks;
  const int row0 = t / S::kChunks;
  const bool a_loader = row0 < S::kTileM;
  const Bf16 *a_g[kPA];
#pragma unroll
  for (int pass = 0; pass < kPA; ++pass) {
    int32_t p = begin + (a_loader ? row0 + pass * S::kRowsPerPass : 0);
    if (p >= end)
      p = begin;
    const int64_t src = kGateUp ? sorted_rows[p] / top_k : p;
    a_g[pass] = in + src * depth + chunk * 8;
  }
  // The moe_wmma_kernel lane layout: A[l % 16][8 (l / 16) .. + 8),
  // B[8 (l / 16) .. + 8)[l % 16], C[8 (l / 16) + v][l % 16]. A B fragment is
  // 16 weight rows (output columns) at 16 bytes per lane, read from memory
  // directly; columns past n_out re-read the last one (results dropped).
  const int frag_row = lane % kMma;
  const int frag_k = (lane / kMma) * 8;
  const int wm = wave / S::kWavesN;
  const int wn = wave % S::kWavesN;
  const size_t expert_elems = static_cast<size_t>(n_out) * depth;
  const Bf16 *b0_g[kFN];
  const Bf16 *b1_g[kFN];
#pragma unroll
  for (int j = 0; j < kFN; ++j) {
    const int32_t n = min(tile_n + (wn * kFN + j) * kMma + frag_row, n_out - 1);
    const size_t w_row = local * expert_elems + static_cast<size_t>(n) * depth;
    b0_g[j] = w0 + w_row + frag_k;
    b1_g[j] = kGateUp ? w1 + w_row + frag_k : b0_g[j];
  }
  v8f acc0[S::kFragsM][kFN];
  v8f acc1[kGateUp ? S::kFragsM : 1][kGateUp ? kFN : 1];
#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i) {
#pragma unroll
    for (int j = 0; j < kFN; ++j) {
      acc0[i][j] = v8f{};
      if (kGateUp)
        acc1[i][j] = v8f{};
    }
  }
  u32x4 ra[kPA];
  v8s rb0[2][kSl][kFN];
  v8s rb1[2][kGateUp ? kSl : 1][kGateUp ? kFN : 1];
  auto load_a = [&](int32_t k) {
#pragma unroll
    for (int pass = 0; pass < kPA; ++pass)
      ra[pass] = *reinterpret_cast<const u32x4 *>(a_g[pass] + k);
  };
  auto store_a = [&](int stage) {
    if (!a_loader)
      return;
#pragma unroll
    for (int pass = 0; pass < kPA; ++pass) {
      const int at =
          stage * kA + (row0 + pass * S::kRowsPerPass) * S::kLdsRow + chunk * 8;
      *reinterpret_cast<u32x4 *>(&a_s[at]) = ra[pass];
    }
  };
  auto load_b = [&](int buf, int32_t k) {
#pragma unroll
    for (int s = 0; s < kSl; ++s) {
#pragma unroll
      for (int j = 0; j < kFN; ++j) {
        rb0[buf][s][j] = *reinterpret_cast<const v8s *>(b0_g[j] + k + s * kMma);
        if (kGateUp)
          rb1[buf][s][j] =
              *reinterpret_cast<const v8s *>(b1_g[j] + k + s * kMma);
      }
    }
  };
  load_a(0);
  load_b(0, 0);
  store_a(0);
  __syncthreads();
  // Stage st (LDS buffer and B registers) holds step k; the loads of step
  // k + kTileK go to stage st ^ 1, which every wave finished reading before the
  // previous barrier. depth is a multiple of 2 kTileK (moe_stream_shape).
  for (int32_t k0 = 0; k0 < depth; k0 += 2 * S::kTileK) {
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int32_t next = k0 + (st + 1) * S::kTileK;
      if (next < depth) {
        load_a(next);
        load_b(st ^ 1, next);
      }
      const Bf16 *as = a_s + st * kA;
#pragma unroll
      for (int s = 0; s < kSl; ++s) {
        v8s a[S::kFragsM];
#pragma unroll
        for (int i = 0; i < S::kFragsM; ++i) {
          a[i] = *reinterpret_cast<const v8s *>(
              &as[((wm * S::kFragsM + i) * kMma + frag_row) * S::kLdsRow +
                  s * kMma + frag_k]);
        }
#pragma unroll
        for (int j = 0; j < kFN; ++j) {
#pragma unroll
          for (int i = 0; i < S::kFragsM; ++i)
            acc0[i][j] = mma(a[i], rb0[st][s][j], acc0[i][j]);
          if (kGateUp) {
#pragma unroll
            for (int i = 0; i < S::kFragsM; ++i)
              acc1[i][j] = mma(a[i], rb1[st][s][j], acc1[i][j]);
          }
        }
      }
      if (next < depth)
        store_a(st ^ 1);
      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < S::kFragsM; ++i) {
#pragma unroll
    for (int j = 0; j < kFN; ++j) {
      const int32_t n = tile_n + (wn * kFN + j) * kMma + frag_row;
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
int32_t launch_stream(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                      const void *in, const void *w0, const void *w1,
                      int32_t depth, int32_t n_out, void *out) {
  const int32_t rows = d->num_tokens * d->top_k;
  const int32_t num_local = d->expert_end - d->expert_begin;
  // Every expert's last row tile may be partial: at most one extra tile each.
  const int64_t row_tiles = (rows + S::kTileM - 1) / S::kTileM + num_local;
  const int64_t n_tiles = (n_out + S::kTileN - 1) / S::kTileN;
  hipLaunchKernelGGL(
      (moe_stream_kernel<S, kGateUp>),
      dim3(static_cast<unsigned>(row_tiles * n_tiles)), dim3(S::kThreads), 0,
      ctx->stream, static_cast<const Bf16 *>(in), static_cast<const Bf16 *>(w0),
      static_cast<const Bf16 *>(w1), d->sorted_rows, d->expert_offsets,
      d->expert_begin, num_local, d->top_k, depth, n_out,
      static_cast<Bf16 *>(out));
  return check_hip(ctx, hipGetLastError(),
                   kGateUp ? "moe prefill gate-up kernel launch"
                           : "moe prefill down kernel launch");
}

// The prefill shapes (moe_prefill_timings and a tile sweep on the R9700 with
// OLMoE's real routing; bitwise equal to moe_wmma_kernel at every size):
// gate-up 64 rows x 128 columns of each projection per block, 8 waves of one
// column fragment each, k steps of 128 (the 16-byte weight loads of a lane
// cover 256 contiguous bytes of a weight row per step). Down: the same tile
// while the experts hold few row tiles (the kernel streams weights; deeper k
// steps keep more bytes in flight), 64 x 256 with k steps of 64 once the
// average expert fills kWideTiles row tiles (each A tile then serves twice
// the columns): at 8,192 routed rows the narrow tile takes 0.61 ms against
// 0.68, at 16,384 the wide one 0.78 ms against 0.89.
using PrefillGateUp = Stream<1, 8, 4, 1, 128>;
using PrefillDown = Stream<1, 8, 4, 1, 128>;
using PrefillDownWide = Stream<1, 8, 4, 2, 64>;
constexpr int kWideTiles = 3;

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

// ---- the small-row (decode) WMMA kernels ----
//
// The small-m implementation's kernels for shapes of multiples of 64: the same
// per-element WMMA chain as moe_wmma_kernel (a zero accumulator, then one
// v_wmma per 16-wide k slice in ascending k, operands in the same lane layout,
// the same epilogue), so a row's gate, up, activation and down values are
// bitwise those of the grouped path whichever tier its batch selects. Built for
// decode: a block holds kGemvWaves waves, each owning one 16-column WMMA tile
// of one expert and walking the expert's rows 16 at a time; the operands come
// straight from global memory into the WMMA registers (no LDS, no barriers),
// kGemvUnroll k slices per batch of loads with the next batch in flight. A
// decode step reads every active expert's weights once; the rows are few, so
// the compute the 16-row tile wastes is free.
//
// Shape (moe_decode_tier_timings on the R9700, OLMoE, uniform routing): 8 k
// slices per batch stream the weights at ~550 GB/s from 8 tokens up (4 slices:
// ~480; 16 for the down projection: ~520; non-temporal loads: ~360); 1, 2 or 4
// waves per block time the same. The scalar kernels this replaced reached ~575
// GB/s at 4-16 tokens and fell to ~410 at 64; the 16-row LDS tiles of
// moe_wmma_kernel (SmallBlock) ~455.
constexpr int kGemvWaves = 4;
constexpr int kGemvUnroll = 8;
constexpr int kGemvStep = kGemvUnroll * kMma;

// The local index of the slot-th local expert (ascending) that has rows, or -1
// when fewer experts have rows. Called by every thread of the block.
__device__ int32_t gemv_active_expert(const int32_t *offsets,
                                      int32_t expert_begin, int32_t num_local,
                                      int32_t slot) {
  __shared__ uint8_t has_rows[kMoeMaxExperts];
  __shared__ int32_t found;
  for (int32_t t = static_cast<int32_t>(threadIdx.x); t < num_local;
       t += static_cast<int32_t>(blockDim.x)) {
    has_rows[t] =
        offsets[expert_begin + t] < offsets[expert_begin + t + 1] ? 1 : 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t e = -1;
    int32_t seen = 0;
    for (int32_t x = 0; x < num_local; ++x) {
      if (has_rows[x] != 0) {
        if (seen == slot) {
          e = x;
          break;
        }
        ++seen;
      }
    }
    found = e;
  }
  __syncthreads();
  return found;
}

__device__ inline v8s load8(const Bf16 *p) {
  return *reinterpret_cast<const v8s *>(p);
}

// One projection of one expert's rows over the 16 columns of each wave.
//   kGateUp: out[p, n] = bf16(round(silu(round(x_p . w0_n))) * round(x_p .
//            w1_n)) with x_p = in[sorted_rows[p] / top_k], n < inter;
//   else:    out[p, n] = bf16(in[p] . w0_n), n < hidden.
// Grid: x = column groups of kGemvWaves * 16, y = expert slots. depth is a
// multiple of kGemvStep (moe_gemv_shape).
template <bool kGateUp>
__global__ __launch_bounds__(kGemvWaves *kLanes) void moe_wmma_gemv_kernel(
    const Bf16 *in, const Bf16 *w0, const Bf16 *w1, const int32_t *sorted_rows,
    const int32_t *offsets, int32_t expert_begin, int32_t num_local,
    int32_t top_k, int32_t depth, int32_t n_out, Bf16 *out) {
  const int32_t local = gemv_active_expert(offsets, expert_begin, num_local,
                                           static_cast<int32_t>(blockIdx.y));
  if (local < 0)
    return;
  const int32_t begin = offsets[expert_begin + local];
  const int32_t end = offsets[expert_begin + local + 1];
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int wave = static_cast<int>(threadIdx.x) / kLanes;
  const int32_t n0 =
      (static_cast<int32_t>(blockIdx.x) * kGemvWaves + wave) * kMma;
  if (n0 >= n_out)
    return;
  // The moe_wmma_kernel lane layout: A[l % 16][8 (l / 16) .. + 8),
  // B[8 (l / 16) .. + 8)[l % 16], C[8 (l / 16) + v][l % 16].
  const int frag = lane % kMma;
  const int kh = (lane / kMma) * 8;
  // Columns past n_out re-read the last one; their results are dropped.
  const int32_t n = min(n0 + frag, n_out - 1);
  const size_t expert_elems = static_cast<size_t>(n_out) * depth;
  const Bf16 *b0 =
      w0 + local * expert_elems + static_cast<size_t>(n) * depth + kh;
  const Bf16 *b1 =
      kGateUp ? w1 + local * expert_elems + static_cast<size_t>(n) * depth + kh
              : b0;
  for (int32_t r0 = begin; r0 < end; r0 += kMma) {
    // Rows past the expert's last position re-read its first (moe_wmma_kernel
    // does the same); their results are dropped.
    int32_t p = r0 + frag;
    if (p >= end)
      p = begin;
    const int64_t src = kGateUp ? sorted_rows[p] / top_k : p;
    const Bf16 *a = in + src * depth + kh;
    v8f acc0{};
    v8f acc1{};
    v8s ra[kGemvUnroll];
    v8s rb0[kGemvUnroll];
    v8s rb1[kGemvUnroll];
#pragma unroll
    for (int u = 0; u < kGemvUnroll; ++u) {
      ra[u] = load8(a + u * kMma);
      rb0[u] = load8(b0 + u * kMma);
      if (kGateUp)
        rb1[u] = load8(b1 + u * kMma);
    }
    for (int32_t k = 0; k < depth; k += kGemvStep) {
      v8s ca[kGemvUnroll];
      v8s cb0[kGemvUnroll];
      v8s cb1[kGemvUnroll];
#pragma unroll
      for (int u = 0; u < kGemvUnroll; ++u) {
        ca[u] = ra[u];
        cb0[u] = rb0[u];
        if (kGateUp)
          cb1[u] = rb1[u];
      }
      const int32_t next = k + kGemvStep;
      if (next < depth) {
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) {
          ra[u] = load8(a + next + u * kMma);
          rb0[u] = load8(b0 + next + u * kMma);
          if (kGateUp)
            rb1[u] = load8(b1 + next + u * kMma);
        }
      }
#pragma unroll
      for (int u = 0; u < kGemvUnroll; ++u) {
        acc0 = mma(ca[u], cb0[u], acc0);
        if (kGateUp)
          acc1 = mma(ca[u], cb1[u], acc1);
      }
    }
    const int32_t col = n0 + frag;
    if (col >= n_out)
      continue;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int32_t q = r0 + (lane / kMma) * 8 + v;
      if (q >= end)
        continue;
      float value = acc0[v];
      if (kGateUp) {
        const float g = round_bf16(value);
        const float u = round_bf16(acc1[v]);
        value = round_bf16(g / (1.0f + expf(-g))) * u;
      }
      out[static_cast<size_t>(q) * n_out + col] = f32_to_bf16(value);
    }
  }
}

template <bool kGateUp>
int32_t launch_gemv(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                    const void *in, const void *w0, const void *w1,
                    int32_t depth, int32_t n_out, void *out) {
  const int32_t rows = d->num_tokens * d->top_k;
  const int32_t num_local = d->expert_end - d->expert_begin;
  // At most one expert slot per routed row.
  const unsigned slots = static_cast<unsigned>(std::min(num_local, rows));
  const int32_t cols = kGemvWaves * kMma;
  hipLaunchKernelGGL(
      (moe_wmma_gemv_kernel<kGateUp>),
      dim3(static_cast<unsigned>((n_out + cols - 1) / cols), slots),
      dim3(kGemvWaves * kLanes), 0, ctx->stream, static_cast<const Bf16 *>(in),
      static_cast<const Bf16 *>(w0), static_cast<const Bf16 *>(w1),
      d->sorted_rows, d->expert_offsets, d->expert_begin, num_local, d->top_k,
      depth, n_out, static_cast<Bf16 *>(out));
  return check_hip(ctx, hipGetLastError(),
                   kGateUp ? "moe small wmma gate-up kernel launch"
                           : "moe small wmma down kernel launch");
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

int32_t launch_moe_wmma_small(turbine_ctx *ctx,
                              const turbine_moe_experts_desc *d, int32_t *pos,
                              void *act, void *down) {
  if (int32_t rc = launch_moe_positions(ctx, d, pos); rc != TURBINE_OK)
    return rc;
  constexpr int kDeep = SmallBlock::kPrefetch * kTileK;
  if (d->hidden % kDeep == 0 && d->inter % kDeep == 0)
    return launch_shape<SmallBlock>(ctx, d, act, down);
  return launch_shape<SmallBlockK64>(ctx, d, act, down);
}

bool moe_prefill_shape(const turbine_moe_experts_desc *d) {
  // Each kernel's loop runs two k steps per iteration.
  constexpr int kGate = 2 * PrefillGateUp::kTileK;
  constexpr int kDown =
      2 * std::max(PrefillDown::kTileK, PrefillDownWide::kTileK);
  return d->hidden % kGate == 0 && d->inter % kDown == 0;
}

int32_t launch_moe_wmma_prefill(turbine_ctx *ctx,
                                const turbine_moe_experts_desc *d, int32_t *pos,
                                void *act, void *down) {
  if (int32_t rc = launch_moe_positions(ctx, d, pos); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = launch_stream<PrefillGateUp, true>(
          ctx, d, d->x, d->w_gate, d->w_up, d->hidden, d->inter, act);
      rc != TURBINE_OK) {
    return rc;
  }
  // The average expert's routed rows against the wide tile's row tiles. Only
  // the tiling changes, never an output's WMMA chain.
  const int64_t rows = static_cast<int64_t>(d->num_tokens) * d->top_k;
  if (rows >= static_cast<int64_t>(d->num_experts) * PrefillDownWide::kTileM *
                  kWideTiles) {
    return launch_stream<PrefillDownWide, false>(
        ctx, d, act, d->w_down, nullptr, d->inter, d->hidden, down);
  }
  return launch_stream<PrefillDown, false>(ctx, d, act, d->w_down, nullptr,
                                           d->inter, d->hidden, down);
}

bool moe_gemv_shape(const turbine_moe_experts_desc *d) {
  return d->hidden % kGemvStep == 0 && d->inter % kGemvStep == 0;
}

int32_t launch_moe_wmma_gemv(turbine_ctx *ctx,
                             const turbine_moe_experts_desc *d, int32_t *pos,
                             void *act, void *down) {
  if (int32_t rc = launch_moe_positions(ctx, d, pos); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = launch_gemv<true>(ctx, d, d->x, d->w_gate, d->w_up,
                                     d->hidden, d->inter, act);
      rc != TURBINE_OK) {
    return rc;
  }
  return launch_gemv<false>(ctx, d, act, d->w_down, nullptr, d->inter,
                            d->hidden, down);
}

} // namespace turbine_hip
