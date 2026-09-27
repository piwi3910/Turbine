// Turbine HIP kernels for moe_experts with few routed rows (impl
// "turbine_hip_moe_small_m"; the default takes them up to the card profile's
// moe_small_max_rows routed rows, num_tokens * top_k).
//
// When hidden and inter are multiples of 64 (moe_wmma_shape, e.g. OLMoE) the
// implementation runs WMMA kernels of moe_grouped.hip -- the decode kernels
// (launch_moe_wmma_gemv, one wave per 16 columns, operands straight from
// memory) for multiples of 128, else the grouped kernels with 16-row tiles
// (launch_moe_wmma_small) -- with the same per-element WMMA chain as the tier
// above it, so a row's output does not depend on how many rows share its batch
// (decision 2026-09-27, the OLMoE concurrency-16 golden flip: these scalar
// kernels sum in another order than the WMMA tier, and a decode row that shared
// a batch with a prefill chunk crossed the tier boundary and changed its
// rounding). The scalar kernels below serve the other shapes (multiples of 8);
// for those the tiers are not bitwise equal.
//
// The scalar kernels:
// every group size is read from expert_offsets on the device, so the caller
// needs no host copy of the offsets and the step makes no device-to-host copy.
//
//   1. positions: pos[sorted_rows[i]] = i for the local rows (positions
//      [offsets[expert_begin], offsets[expert_end])), -1 for the others;
//   2. gate-up: one block per (expert with rows, tile of intermediate columns)
//      computes silu(gate(x)) * up(x) for every row of the expert, reading each
//      weight row once per pass of up to kTileRows rows (a decode batch of 64
//      sequences has at most a handful of rows per expert, so each selected
//      expert's weights stream from memory once per step);
//   3. down: the same tiling over the hidden columns of the down projection;
//   4. the scatter of moe.hip adds the rows to out in ascending position, which
//      is ascending expert id.
// The grids hold min(local experts, routed rows) expert slots; block slot j
// finds the j-th local expert with rows from expert_offsets, and slots past the
// last such expert return at once.
//
// Numerics follow the cpu-reference moe_experts: F32 accumulation, gate and up
// rounded to BF16, silu rounded, the product rounded, the down projection
// rounded. Each output element is summed in a fixed order (lane-strided partial
// sums, then a fixed butterfly across 32 lanes), so two runs are bitwise
// identical.
#include "bf16.hpp"
#include "turbine_hip.hpp"

#include <algorithm>

using turbine_hip::Bf16;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::kMoeMaxExperts;
using turbine_hip::round_bf16;

namespace {

// Lanes that share one output column group; a wave32, or half a wave64 (the
// shuffles use width 32).
constexpr int kLanes = 32;
// Column groups per block.
constexpr int kGroups = 8;
constexpr int kThreads = kLanes * kGroups;
// BF16 elements per 16-byte load.
constexpr int kVec = 8;
constexpr int kBlock = 256;
// Output columns per group and rows per pass over the weights. A lab sweep on
// the R9700 (OLMoE shapes, 1-64 tokens; columns 1/2/4 x rows 4/8/16) found the
// choices within a few percent of each other; 4 x 4 was the fastest at 16-64
// tokens.
constexpr int kTileCols = 4;
constexpr int kTileRows = 4;

static_assert(kMoeMaxExperts <= kThreads,
              "the expert lookup uses one thread per local expert");

// The 8 packed BF16 of v as floats, in index order.
struct F8 {
  float v[kVec];
};

__device__ inline F8 unpack(uint4 v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  F8 f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    f.v[2 * q] = __uint_as_float(w[q] << 16);
    f.v[2 * q + 1] = __uint_as_float(w[q] & 0xffff0000u);
  }
  return f;
}

// acc + x[0..8) . w[0..8), summed in index order.
__device__ inline float dot8(const F8 &x, const F8 &w, float acc) {
#pragma unroll
  for (int v = 0; v < kVec; ++v)
    acc += x.v[v] * w.v[v];
  return acc;
}

__device__ inline uint4 load16(const Bf16 *p) {
  return *reinterpret_cast<const uint4 *>(p);
}

// Sum over the 32 lanes of a group in a fixed butterfly order; every lane gets
// the total.
__device__ inline float group_sum(float v) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off /= 2)
    v += __shfl_xor(v, off, kLanes);
  return v;
}

// The local index of the slot-th local expert (ascending) that has rows, or -1
// when fewer experts have rows. Called by every thread of the block.
__device__ int32_t active_expert(const int32_t *offsets, int32_t expert_begin,
                                 int32_t num_local, int32_t slot) {
  __shared__ uint8_t has_rows[kMoeMaxExperts];
  __shared__ int32_t found;
  const int t = static_cast<int>(threadIdx.x);
  if (t < num_local) {
    has_rows[t] =
        offsets[expert_begin + t] < offsets[expert_begin + t + 1] ? 1 : 0;
  }
  __syncthreads();
  if (t == 0) {
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

// One projection over the rows of one local expert.
//   kGateUp: out[p, j] = bf16(round(silu(round(x_p . w0_j))) * round(x_p .
//            w1_j)) with x_p = in[sorted_rows[p] / top_k] (the token's
//            activation), j over inter columns;
//   else:    out[p, j] = bf16(in[p] . w0_j), in = the activations of step 2,
//            j over hidden columns.
// p runs over the expert's positions [offsets[e], offsets[e + 1]); w0/w1 are
// [num_local_experts, n_out, depth]. Each group of 32 lanes owns kCols output
// columns and walks the rows kRows at a time.
template <int kCols, int kRows, bool kGateUp>
__global__ __launch_bounds__(kThreads) void moe_small_proj_kernel(
    const Bf16 *in, const Bf16 *w0, const Bf16 *w1, const int32_t *sorted_rows,
    const int32_t *offsets, int32_t expert_begin, int32_t num_local,
    int32_t top_k, int32_t depth, int32_t n_out, Bf16 *out) {
  static_assert(kRows * kCols <= kLanes,
                "every (row, column) result of a pass has its own writer lane");
  const int32_t local = active_expert(offsets, expert_begin, num_local,
                                      static_cast<int32_t>(blockIdx.y));
  if (local < 0)
    return;
  const int32_t begin = offsets[expert_begin + local];
  const int32_t end = offsets[expert_begin + local + 1];
  const int group = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int32_t col0 =
      (static_cast<int32_t>(blockIdx.x) * kGroups + group) * kCols;
  if (col0 >= n_out)
    return;
  const int32_t ncols = min(kCols, n_out - col0);

  const size_t expert_elems = static_cast<size_t>(n_out) * depth;
  const Bf16 *wa[kCols];
  const Bf16 *wb[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    // Columns past n_out re-read the last one; their results are dropped.
    const size_t row = static_cast<size_t>(col0 + min(c, ncols - 1)) * depth;
    wa[c] = w0 + local * expert_elems + row;
    wb[c] = kGateUp ? w1 + local * expert_elems + row : wa[c];
  }

  const int32_t stride = kLanes * kVec;
  for (int32_t r0 = begin; r0 < end; r0 += kRows) {
    const int32_t nr = min(kRows, end - r0);
    const Bf16 *xr[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int32_t p = r0 + min(i, nr - 1);
      const int64_t src = kGateUp ? sorted_rows[p] / top_k : p;
      xr[i] = in + static_cast<size_t>(src) * depth;
    }
    float acc_a[kRows][kCols];
    float acc_b[kRows][kCols];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        acc_a[i][c] = 0.0f;
        acc_b[i][c] = 0.0f;
      }
    }
    // Each lane walks its 16-byte slices of the weight rows; the next slice's
    // loads are issued before the current one is multiplied, so a group keeps
    // two slices of every weight row in flight.
    int32_t k = lane * kVec;
    uint4 ga[kCols];
    uint4 gb[kCols];
    if (k < depth) {
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        ga[c] = load16(wa[c] + k);
        if (kGateUp)
          gb[c] = load16(wb[c] + k);
      }
    }
    while (k < depth) {
      F8 fa[kCols];
      F8 fb[kCols];
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        fa[c] = unpack(ga[c]);
        if (kGateUp)
          fb[c] = unpack(gb[c]);
      }
      const int32_t next = k + stride;
      if (next < depth) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          ga[c] = load16(wa[c] + next);
          if (kGateUp)
            gb[c] = load16(wb[c] + next);
        }
      }
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        if (i < nr) {
          const F8 xv = unpack(load16(xr[i] + k));
#pragma unroll
          for (int c = 0; c < kCols; ++c) {
            acc_a[i][c] = dot8(xv, fa[c], acc_a[i][c]);
            if (kGateUp)
              acc_b[i][c] = dot8(xv, fb[c], acc_b[i][c]);
          }
        }
      }
      k = next;
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      if (i >= nr)
        break;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const float a = group_sum(acc_a[i][c]);
        float v = a;
        if (kGateUp) {
          const float g = round_bf16(a);
          const float u = round_bf16(group_sum(acc_b[i][c]));
          v = round_bf16(g / (1.0f + expf(-g))) * u;
        }
        if (lane == i * kCols + c && c < ncols) {
          out[static_cast<size_t>(r0 + i) * n_out + col0 + c] = f32_to_bf16(v);
        }
      }
    }
  }
}

// pos[sorted_rows[i]] = i for the local positions, -1 for the others
// (sorted_rows is a permutation of the rows, so each entry is written once).
__global__ void moe_small_positions_kernel(const int32_t *sorted_rows,
                                           const int32_t *offsets,
                                           int32_t expert_begin,
                                           int32_t expert_end, int32_t rows,
                                           int32_t *pos) {
  const int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows)
    return;
  const int32_t first = offsets[expert_begin];
  const int32_t last = offsets[expert_end];
  pos[sorted_rows[i]] = i >= first && i < last ? i : -1;
}

template <int kCols, int kRows>
int32_t launch_proj(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                    const void *act, void *down, unsigned slots) {
  const int32_t num_local = d->expert_end - d->expert_begin;
  const int32_t cols_per_block = kGroups * kCols;
  hipLaunchKernelGGL(
      (moe_small_proj_kernel<kCols, kRows, true>),
      dim3(static_cast<unsigned>((d->inter + cols_per_block - 1) /
                                 cols_per_block),
           slots),
      dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(d->x),
      static_cast<const Bf16 *>(d->w_gate), static_cast<const Bf16 *>(d->w_up),
      d->sorted_rows, d->expert_offsets, d->expert_begin, num_local, d->top_k,
      d->hidden, d->inter, static_cast<Bf16 *>(const_cast<void *>(act)));
  if (int32_t rc = check_hip(ctx, hipGetLastError(),
                             "moe small-m gate-up kernel launch");
      rc != TURBINE_OK) {
    return rc;
  }
  hipLaunchKernelGGL(
      (moe_small_proj_kernel<kCols, kRows, false>),
      dim3(static_cast<unsigned>((d->hidden + cols_per_block - 1) /
                                 cols_per_block),
           slots),
      dim3(kThreads), 0, ctx->stream, static_cast<const Bf16 *>(act),
      static_cast<const Bf16 *>(d->w_down), nullptr, d->sorted_rows,
      d->expert_offsets, d->expert_begin, num_local, d->top_k, d->inter,
      d->hidden, static_cast<Bf16 *>(down));
  return check_hip(ctx, hipGetLastError(), "moe small-m down kernel launch");
}

} // namespace

namespace turbine_hip {

int32_t launch_moe_positions(turbine_ctx *ctx,
                             const turbine_moe_experts_desc *d, int32_t *pos) {
  const int32_t rows = d->num_tokens * d->top_k;
  hipLaunchKernelGGL(moe_small_positions_kernel,
                     dim3(static_cast<unsigned>((rows + kBlock - 1) / kBlock)),
                     dim3(kBlock), 0, ctx->stream, d->sorted_rows,
                     d->expert_offsets, d->expert_begin, d->expert_end, rows,
                     pos);
  return check_hip(ctx, hipGetLastError(), "moe positions kernel launch");
}

int32_t launch_moe_small_m(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                           int32_t *pos, void *act, void *down) {
  // Batch invariance: where the grouped WMMA path serves the shape, the small-m
  // tier runs its kernels with small row tiles, so a row gets the same bits
  // whichever tier its batch's row count selects.
  if (moe_gemv_shape(d))
    return launch_moe_wmma_gemv(ctx, d, pos, act, down);
  if (moe_wmma_shape(d))
    return launch_moe_wmma_small(ctx, d, pos, act, down);
  const int32_t rows = d->num_tokens * d->top_k;
  const int32_t num_local = d->expert_end - d->expert_begin;
  // At most one expert slot per routed row.
  const unsigned slots = static_cast<unsigned>(std::min(num_local, rows));
  if (int32_t rc = launch_moe_positions(ctx, d, pos); rc != TURBINE_OK)
    return rc;
  return launch_proj<kTileCols, kTileRows>(ctx, d, act, down, slots);
}

} // namespace turbine_hip
