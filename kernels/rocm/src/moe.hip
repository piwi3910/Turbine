// Turbine HIP kernels for mixture-of-experts: routing (softmax, top-k,
// grouping rows by expert), the gather of each routed row's activation and
// the weighted scatter-add of the expert outputs. The expert GEMMs run through
// hipBLASLt (moe.cpp).
//
// Numerics follow the cpu-reference moe ops: the router softmax in f32 with a
// sequential sum (over logits rounded to BF16 first with
// TURBINE_MOE_ROUTE_BF16_LOGITS), the top-k set torch.topk selects,
// and out[t] = round(out[t] + round(y * round(w))) applied expert by expert in
// ascending id.
#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::check_hip;
using turbine_hip::f32_to_bf16;
using turbine_hip::kMoeMaxExperts;
using turbine_hip::kMoeMaxTopK;
using turbine_hip::round_bf16;

namespace {

constexpr int kBlock = 256;
constexpr float kNegInf = -__builtin_huge_valf();
// Expert ids staged in LDS per step of the grouping kernel.
constexpr int kSortTile = 2048;

static_assert(kMoeMaxExperts <= kBlock,
              "the grouping kernel uses one thread per expert");

// ---- routing ----

// ---- torch.topk selection ----
//
// PyTorch's CPU torch.topk (largest) fills (value, index) pairs in index order
// and selects with libstdc++ std::nth_element at k - 1 (introselect), or
// std::partial_sort (heap select) when k * 64 <= n, comparing values only (NaN
// first). Among tied values the selected set is whatever those algorithms
// leave in front, so the functions below are line-by-line ports of the
// libstdc++ ones (and of crates/turbine-kernels/src/cpu/topk.rs, which a
// fixture pins to torch.topk itself).

struct TopkElem {
  float v;
  int32_t i;
};

__device__ inline bool topk_greater(const TopkElem &x, const TopkElem &y) {
  return (isnan(x.v) && !isnan(y.v)) || x.v > y.v;
}

__device__ inline void topk_swap(TopkElem *a, int32_t x, int32_t y) {
  const TopkElem t = a[x];
  a[x] = a[y];
  a[y] = t;
}

// std::__move_median_to_first(result, a, b, c)
__device__ void topk_median_to_first(TopkElem *v, int32_t result, int32_t a,
                                     int32_t b, int32_t c) {
  int32_t pick;
  if (topk_greater(v[a], v[b])) {
    if (topk_greater(v[b], v[c]))
      pick = b;
    else if (topk_greater(v[a], v[c]))
      pick = c;
    else
      pick = a;
  } else if (topk_greater(v[a], v[c])) {
    pick = a;
  } else if (topk_greater(v[b], v[c])) {
    pick = c;
  } else {
    pick = b;
  }
  topk_swap(v, result, pick);
}

// std::__unguarded_partition_pivot(first, last)
__device__ int32_t topk_partition_pivot(TopkElem *v, int32_t first,
                                        int32_t last) {
  const int32_t pivot = first;
  topk_median_to_first(v, first, first + 1, first + (last - first) / 2,
                       last - 1);
  first += 1;
  while (true) {
    while (topk_greater(v[first], v[pivot]))
      ++first;
    --last;
    while (topk_greater(v[pivot], v[last]))
      --last;
    if (!(first < last))
      return first;
    topk_swap(v, first, last);
    ++first;
  }
}

// std::__insertion_sort(first, last)
__device__ void topk_insertion_sort(TopkElem *v, int32_t first, int32_t last) {
  if (first == last)
    return;
  for (int32_t i = first + 1; i < last; ++i) {
    const TopkElem val = v[i];
    if (topk_greater(val, v[first])) {
      for (int32_t j = i; j > first; --j)
        v[j] = v[j - 1];
      v[first] = val;
    } else {
      int32_t j = i;
      while (topk_greater(val, v[j - 1])) {
        v[j] = v[j - 1];
        --j;
      }
      v[j] = val;
    }
  }
}

// std::__adjust_heap(first, hole, len, value) with its std::__push_heap.
__device__ void topk_adjust_heap(TopkElem *v, int32_t first, int32_t hole,
                                 int32_t len, TopkElem value) {
  const int32_t top = hole;
  int32_t child = hole;
  while (child < (len - 1) / 2) {
    child = 2 * (child + 1);
    if (topk_greater(v[first + child], v[first + child - 1]))
      --child;
    v[first + hole] = v[first + child];
    hole = child;
  }
  if ((len & 1) == 0 && child == (len - 2) / 2) {
    child = 2 * (child + 1);
    v[first + hole] = v[first + child - 1];
    hole = child - 1;
  }
  int32_t parent = (hole - 1) / 2;
  while (hole > top && topk_greater(v[first + parent], value)) {
    v[first + hole] = v[first + parent];
    hole = parent;
    parent = (hole - 1) / 2;
  }
  v[first + hole] = value;
}

// std::__heap_select(first, middle, last), with std::__make_heap and
// std::__pop_heap.
__device__ void topk_heap_select(TopkElem *v, int32_t first, int32_t middle,
                                 int32_t last) {
  const int32_t len = middle - first;
  if (len >= 2) {
    for (int32_t parent = (len - 2) / 2;; --parent) {
      topk_adjust_heap(v, first, parent, len, v[first + parent]);
      if (parent == 0)
        break;
    }
  }
  for (int32_t i = middle; i < last; ++i) {
    if (topk_greater(v[i], v[first])) {
      const TopkElem value = v[i];
      v[i] = v[first];
      topk_adjust_heap(v, first, 0, len, value);
    }
  }
}

// std::nth_element(v, v + nth, v + n) (std::__introselect).
__device__ void topk_nth_element(TopkElem *v, int32_t nth, int32_t n) {
  int32_t first = 0, last = n;
  if (first == last || nth == last)
    return;
  int32_t depth = 2 * (31 - __clz(n));
  while (last - first > 3) {
    if (depth == 0) {
      topk_heap_select(v, first, nth + 1, last);
      topk_swap(v, first, nth);
      return;
    }
    --depth;
    const int32_t cut = topk_partition_pivot(v, first, last);
    if (cut <= nth)
      first = cut;
    else
      last = cut;
  }
  topk_insertion_sort(v, first, last);
}

// One thread per token: logits rounded to BF16 with
// TURBINE_MOE_ROUTE_BF16_LOGITS, softmax over the experts (max-subtracted,
// sequential sum), the set torch.topk selects, written in descending weight
// with ties to the lower expert id.
__global__ void moe_topk_kernel(const float *logits, int32_t num_tokens,
                                int32_t num_experts, int32_t top_k,
                                int32_t flags, int32_t *topk_ids,
                                float *topk_weights) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= num_tokens)
    return;
  const float *row = logits + t * num_experts;
  const bool bf16_logits = (flags & TURBINE_MOE_ROUTE_BF16_LOGITS) != 0;
  TopkElem v[kMoeMaxExperts];
  float max = kNegInf;
  for (int32_t e = 0; e < num_experts; ++e) {
    v[e].v = bf16_logits ? round_bf16(row[e]) : row[e];
    v[e].i = e;
    max = fmaxf(max, v[e].v);
  }
  float sum = 0.0f;
  for (int32_t e = 0; e < num_experts; ++e)
    sum += expf(v[e].v - max);
  for (int32_t e = 0; e < num_experts; ++e)
    v[e].v = expf(v[e].v - max) / sum;

  if (top_k * 64 <= num_experts)
    topk_heap_select(v, 0, top_k, num_experts);
  else
    topk_nth_element(v, top_k - 1, num_experts);
  // Descending weight, ties to the lower id (insertion sort of the k chosen).
  for (int32_t a = 1; a < top_k; ++a) {
    const TopkElem x = v[a];
    int32_t b = a;
    while (b > 0 && (v[b - 1].v < x.v || (v[b - 1].v == x.v && v[b - 1].i > x.i))) {
      v[b] = v[b - 1];
      --b;
    }
    v[b] = x;
  }
  int32_t *ids = topk_ids + t * top_k;
  float *weights = topk_weights + t * top_k;
  for (int32_t slot = 0; slot < top_k; ++slot) {
    ids[slot] = v[slot].i;
    weights[slot] = v[slot].v;
  }
  if ((flags & TURBINE_MOE_ROUTE_RENORMALIZE) != 0) {
    float chosen_sum = 0.0f;
    for (int32_t slot = 0; slot < top_k; ++slot)
      chosen_sum += weights[slot];
    for (int32_t slot = 0; slot < top_k; ++slot)
      weights[slot] /= chosen_sum;
  }
}

// One block: thread e counts the rows routed to expert e, thread 0 turns the
// counts into offsets, then thread e writes its rows in ascending row order
// (a stable counting sort). The ids are staged through LDS in tiles.
__global__ void moe_group_kernel(const int32_t *topk_ids, int32_t rows,
                                 int32_t num_experts, int32_t *sorted_rows,
                                 int32_t *expert_offsets) {
  __shared__ int32_t tile[kSortTile];
  __shared__ int32_t offsets[kMoeMaxExperts + 1];
  const int e = threadIdx.x;
  const bool owner = e < num_experts;

  int32_t count = 0;
  for (int32_t t0 = 0; t0 < rows; t0 += kSortTile) {
    const int32_t n = min(kSortTile, rows - t0);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      tile[i] = topk_ids[t0 + i];
    __syncthreads();
    if (owner) {
      for (int32_t i = 0; i < n; ++i)
        count += tile[i] == e ? 1 : 0;
    }
    __syncthreads();
  }
  if (owner)
    offsets[e + 1] = count;
  __syncthreads();
  if (threadIdx.x == 0) {
    offsets[0] = 0;
    for (int32_t x = 0; x < num_experts; ++x)
      offsets[x + 1] += offsets[x];
  }
  __syncthreads();
  for (int x = threadIdx.x; x <= num_experts; x += blockDim.x)
    expert_offsets[x] = offsets[x];

  int32_t cursor = owner ? offsets[e] : 0;
  for (int32_t t0 = 0; t0 < rows; t0 += kSortTile) {
    const int32_t n = min(kSortTile, rows - t0);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      tile[i] = topk_ids[t0 + i];
    __syncthreads();
    if (owner) {
      for (int32_t i = 0; i < n; ++i) {
        if (tile[i] == e)
          sorted_rows[cursor++] = t0 + i;
      }
    }
    __syncthreads();
  }
}

// ---- experts ----

__global__ void moe_clear_kernel(int32_t *pos, int64_t rows) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < rows; i += stride) {
    pos[i] = -1;
  }
}

// One block per gathered row i: xs[i] = x[sorted_rows[i] / top_k].
__global__ void moe_gather_kernel(const Bf16 *x, const int32_t *sorted_rows,
                                  int32_t hidden, int32_t top_k, Bf16 *xs,
                                  int32_t *pos) {
  const int64_t i = blockIdx.x;
  const int32_t r = sorted_rows[i];
  const Bf16 *src = x + static_cast<int64_t>(r / top_k) * hidden;
  Bf16 *dst = xs + i * hidden;
  for (int32_t c = threadIdx.x; c < hidden; c += blockDim.x)
    dst[c] = src[c];
  if (threadIdx.x == 0)
    pos[r] = static_cast<int32_t>(i);
}

// One block per token: its local rows sorted by gathered position (ascending
// position is ascending expert id), then each column accumulates them in that
// order with the reference's roundings.
__global__ void moe_scatter_kernel(const Bf16 *down, const int32_t *pos,
                                   const float *topk_weights, int32_t hidden,
                                   int32_t top_k, Bf16 *out) {
  __shared__ int32_t order_pos[kMoeMaxTopK];
  __shared__ float order_w[kMoeMaxTopK];
  __shared__ int32_t order_n;
  const int64_t t = blockIdx.x;
  if (threadIdx.x == 0) {
    int32_t n = 0;
    for (int32_t slot = 0; slot < top_k; ++slot) {
      const int64_t r = t * top_k + slot;
      const int32_t p = pos[r];
      if (p < 0)
        continue;
      // Insertion by position.
      int32_t at = n;
      while (at > 0 && order_pos[at - 1] > p) {
        order_pos[at] = order_pos[at - 1];
        order_w[at] = order_w[at - 1];
        --at;
      }
      order_pos[at] = p;
      order_w[at] = round_bf16(topk_weights[r]);
      ++n;
    }
    order_n = n;
  }
  __syncthreads();
  const int32_t n = order_n;
  if (n == 0)
    return;
  Bf16 *row = out + t * hidden;
  for (int32_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    float o = bf16_to_f32(row[c]);
    for (int32_t k = 0; k < n; ++k) {
      const float y =
          bf16_to_f32(down[static_cast<int64_t>(order_pos[k]) * hidden + c]);
      o = round_bf16(o + round_bf16(y * order_w[k]));
    }
    row[c] = f32_to_bf16(o);
  }
}

unsigned grid_for(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  if (blocks < 1)
    return 1;
  return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

} // namespace

namespace turbine_hip {

int32_t launch_moe_route(turbine_ctx *ctx, const turbine_moe_route_desc *d) {
  if (d->num_tokens > 0) {
    hipLaunchKernelGGL(
        moe_topk_kernel,
        dim3(static_cast<unsigned>(
            (static_cast<int64_t>(d->num_tokens) + kBlock - 1) / kBlock)),
        dim3(kBlock), 0, ctx->stream, d->router_logits, d->num_tokens,
        d->num_experts, d->top_k, d->flags, d->topk_ids, d->topk_weights);
    if (int32_t rc =
            check_hip(ctx, hipGetLastError(), "moe top-k kernel launch");
        rc != TURBINE_OK) {
      return rc;
    }
  }
  const int64_t rows = static_cast<int64_t>(d->num_tokens) * d->top_k;
  hipLaunchKernelGGL(moe_group_kernel, dim3(1), dim3(kBlock), 0, ctx->stream,
                     d->topk_ids, static_cast<int32_t>(rows), d->num_experts,
                     d->sorted_rows, d->expert_offsets);
  return check_hip(ctx, hipGetLastError(), "moe grouping kernel launch");
}

int32_t launch_moe_clear_positions(turbine_ctx *ctx, int32_t *pos,
                                   int64_t rows) {
  if (rows == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_clear_kernel, dim3(grid_for(rows)), dim3(kBlock), 0,
                     ctx->stream, pos, rows);
  return check_hip(ctx, hipGetLastError(), "moe clear kernel launch");
}

int32_t launch_moe_gather(turbine_ctx *ctx, const void *x,
                          const int32_t *sorted_rows, int64_t count,
                          int32_t hidden, int32_t top_k, void *xs,
                          int32_t *pos) {
  if (count == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_gather_kernel, dim3(static_cast<unsigned>(count)),
                     dim3(kBlock), 0, ctx->stream, static_cast<const Bf16 *>(x),
                     sorted_rows, hidden, top_k, static_cast<Bf16 *>(xs), pos);
  return check_hip(ctx, hipGetLastError(), "moe gather kernel launch");
}

int32_t launch_moe_scatter(turbine_ctx *ctx, const void *down,
                           const int32_t *pos, const float *topk_weights,
                           int32_t num_tokens, int32_t hidden, int32_t top_k,
                           void *out) {
  if (num_tokens == 0)
    return TURBINE_OK;
  hipLaunchKernelGGL(moe_scatter_kernel,
                     dim3(static_cast<unsigned>(num_tokens)), dim3(kBlock), 0,
                     ctx->stream, static_cast<const Bf16 *>(down), pos,
                     topk_weights, hidden, top_k, static_cast<Bf16 *>(out));
  return check_hip(ctx, hipGetLastError(), "moe scatter kernel launch");
}

} // namespace turbine_hip
