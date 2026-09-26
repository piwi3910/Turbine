// logits_reduce (kernel ABI v2.1): per F32 logits row, the log-sum-exp, the
// top_n largest raw logits with their ids and, in mode 1, one categorical draw
// by inverse CDF in id order. A Turbine kernel: the reductions follow the
// cpu-reference provider (crates/turbine-kernels/src/cpu/math.rs) and run in a
// fixed order, so results are identical run to run.
//
// One block of 1024 threads (32 waves of 32 lanes) per row. Wave w owns the
// contiguous id range [w * seg, (w + 1) * seg) and its lanes read it in
// coalesced steps of 32 ids, so "id order" is (wave, step, lane) order. The
// row is swept a few times (it stays in L2 / Infinity Cache after the first):
//
//   1. the maximum M (NaN ignored);
//   2. sum exp(v - M) for the lse and, in mode 1, sum exp(v / T - M / T) per
//      wave (f32 exp, f64 sums), the argmax (smallest id holding M) and the
//      first radix-select histogram;
//   3. radix select of the top_n-th largest key, 8 bits per round (at most 3
//      more sweeps), on 32-bit keys ordered like the logits (NaN lowest);
//   4. the collection of the top_n candidates (ties at the threshold by id),
//      sorted by (key descending, id ascending) in shared memory;
//   5. in mode 1, the draw: the wave whose cumulative sum passes u * total,
//      then a scan of that wave's range in id order.
//
// The cpu reference sums f64 exponentials sequentially; here each exponential
// is an f32 expf and the sums are f64 in a fixed tree order, so the lse agrees
// to ~1e-7 relative and the draw differs only where u lies within that
// rounding of a CDF boundary.
#include <cfloat>
#include <climits>
#include <cmath>
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "logits_reduce assumes wave32 (gfx12)"
#endif

namespace {

constexpr const char *kImpl = "turbine_hip";
constexpr int kWave = 32;
constexpr int kWaves = 32;
constexpr int kThreads = kWave * kWaves;
constexpr int kMaxTopN = 64;

// A 32-bit key whose unsigned order is the order of the float (IEEE total
// order: -0 below +0), with every NaN mapped to 0, below everything else.
__device__ inline uint32_t order_key(float v) {
  if (isnan(v))
    return 0u;
  const uint32_t u = __float_as_uint(v);
  const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return k == 0u ? 1u : k;
}

// Sort key of a candidate: key descending, then id ascending.
__device__ inline unsigned long long candidate(uint32_t key, int32_t id) {
  return (static_cast<unsigned long long>(key) << 32) |
         static_cast<unsigned long long>(~static_cast<uint32_t>(id));
}

__device__ inline float wave_max(float v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o, kWave));
  return v;
}

__device__ inline double wave_sum(double v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v += __shfl_xor(v, o, kWave);
  return v;
}

__device__ inline int wave_min(int v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v = min(v, __shfl_xor(v, o, kWave));
  return v;
}

__device__ inline int wave_max_int(int v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v = max(v, __shfl_xor(v, o, kWave));
  return v;
}

// Inclusive prefix sum over the wave's lanes in lane order.
__device__ inline double wave_scan(double v, int lane) {
  for (int o = 1; o < kWave; o <<= 1) {
    const double up = __shfl_up(v, o, kWave);
    if (lane >= o)
      v += up;
  }
  return v;
}

// Lanes below `lane` in a ballot mask.
__device__ inline int lanes_below(unsigned long long mask, int lane) {
  return __popcll(mask & ((1ull << lane) - 1ull));
}

__device__ inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

// Adds the active lanes' bins to the wave's own histogram: one add per
// distinct bin (logits share few exponents, so a step has few distinct top
// bytes), by the lowest lane holding it; no other wave writes `hist`.
__device__ inline void wave_count(unsigned int *hist, uint32_t bin,
                                  bool active, int lane) {
  unsigned long long pending = __ballot(active);
  while (pending != 0ull) {
    const int leader = __ffsll(static_cast<long long>(pending)) - 1;
    const uint32_t b = __shfl(bin, leader, kWave);
    const unsigned long long same = __ballot(active && bin == b);
    if (lane == leader)
      hist[b] += static_cast<unsigned int>(__popcll(same));
    pending &= ~same;
  }
}

__global__ void __launch_bounds__(kThreads)
    logits_reduce_kernel(const float *logits, int64_t vocab, int64_t stride_row,
                         const float *temperature, const float *uniform,
                         const int32_t *mode, int32_t top_n, int32_t *top_ids,
                         float *top_values, float *lse_out, int32_t *sampled,
                         float *sampled_logit) {
  __shared__ float s_max[kWaves];
  __shared__ double s_sum[kWaves];
  __shared__ double s_sum_t[kWaves];
  __shared__ int s_argmax[kWaves];
  __shared__ int s_last[kWaves];
  __shared__ int s_ties[kWaves];
  // Per-wave radix histograms, then their sums.
  __shared__ unsigned int s_wave_hist[kWaves * 256];
  __shared__ unsigned int s_hist[256];
  __shared__ unsigned long long s_sel[kMaxTopN];
  __shared__ int s_count;
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_rem;
  __shared__ bool s_done;
  __shared__ int s_draw_wave;
  __shared__ int s_drawn;

  const int64_t r = blockIdx.x;
  const float *row = logits + r * stride_row;
  const int tid = threadIdx.x;
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const int64_t seg = (vocab + kWaves - 1) / kWaves;
  const int64_t begin = min64(vocab, wave * seg);
  const int64_t end = min64(vocab, begin + seg);
  unsigned int *hist = s_wave_hist + wave * 256;
  for (int b = lane; b < 256; b += kWave)
    hist[b] = 0u;

  // ---- 1. the maximum ----
  float m = -INFINITY;
  for (int64_t j = begin + lane; j < end; j += kWave)
    m = fmaxf(m, row[j]);
  m = wave_max(m);
  if (lane == 0)
    s_max[wave] = m;
  if (tid == 0)
    s_count = 0;
  __syncthreads();
  float big = -INFINITY;
  for (int w = 0; w < kWaves; ++w)
    big = fmaxf(big, s_max[w]);

  const bool draw = mode[r] == 1;
  const float t = temperature[r];
  const float inv_t = 1.0f / t;
  const float big_t = big * inv_t;
  const bool draw_ok = draw && t > 0.0f && isfinite(big_t);

  // ---- 2. sums, argmax, last positive weight, first histogram ----
  double sum = 0.0;
  double sum_t = 0.0;
  int arg = INT_MAX;
  int last = -1;
  for (int64_t base = begin; base < end; base += kWave) {
    const int64_t j = base + lane;
    const bool in = j < end;
    const float v = in ? row[j] : 0.0f;
    wave_count(hist, order_key(v) >> 24, in, lane);
    if (in && !isnan(v)) {
      if (isfinite(big))
        sum += static_cast<double>(expf(v - big));
      if (v == big && big > -INFINITY)
        arg = min(arg, static_cast<int>(j));
      if (draw_ok) {
        const float w = expf(v * inv_t - big_t);
        sum_t += static_cast<double>(w);
        if (w > 0.0f)
          last = static_cast<int>(j);
      }
    }
  }
  sum = wave_sum(sum);
  sum_t = wave_sum(sum_t);
  arg = wave_min(arg);
  last = wave_max_int(last);
  if (lane == 0) {
    s_sum[wave] = sum;
    s_sum_t[wave] = sum_t;
    s_argmax[wave] = arg;
    s_last[wave] = last;
  }
  __syncthreads();

  if (tid == 0) {
    double total = 0.0;
    int a = INT_MAX;
    for (int w = 0; w < kWaves; ++w) {
      total += s_sum[w];
      a = min(a, s_argmax[w]);
    }
    lse_out[r] = isfinite(big) ? big + static_cast<float>(log(total)) : big;
    s_argmax[0] = a == INT_MAX ? 0 : a;
  }

  // ---- 3. radix select of the top_n-th largest key ----
  const int k = static_cast<int>(min64(top_n, vocab));
  if (tid == 0) {
    s_prefix = 0u;
    s_mask = 0u;
    s_rem = k;
    s_done = k == 0;
  }
  __syncthreads();
  for (int round = 0; round < 4; ++round) {
    if (s_done)
      break;
    const int shift = 24 - 8 * round;
    if (round > 0) {
      for (int b = lane; b < 256; b += kWave)
        hist[b] = 0u;
      const uint32_t prefix = s_prefix, mask = s_mask;
      for (int64_t base = begin; base < end; base += kWave) {
        const int64_t j = base + lane;
        const uint32_t key = j < end ? order_key(row[j]) : 0u;
        wave_count(hist, (key >> shift) & 0xffu,
                   j < end && (key & mask) == prefix, lane);
      }
    }
    __syncthreads();
    if (tid < 256) {
      unsigned int c = 0u;
      for (int w = 0; w < kWaves; ++w)
        c += s_wave_hist[w * 256 + tid];
      s_hist[tid] = c;
    }
    __syncthreads();
    if (tid == 0) {
      int rem = s_rem;
      int above = 0;
      int digit = 0;
      for (int b = 255; b >= 0; --b) {
        const int c = static_cast<int>(s_hist[b]);
        if (above + c >= rem) {
          digit = b;
          break;
        }
        above += c;
      }
      rem -= above;
      s_prefix |= static_cast<uint32_t>(digit) << shift;
      s_mask |= 0xffu << shift;
      // Every key with this prefix is taken: no tie to break.
      s_done = static_cast<int>(s_hist[digit]) == rem;
      s_rem = rem;
    }
    __syncthreads();
  }

  // ---- 4. collect: keys above the prefix, then ties by id ----
  const uint32_t prefix = s_prefix, mask = s_mask;
  const int rem = s_rem;
  const bool take_all_ties = s_done;
  int ties = 0;
  if (k > 0) {
    for (int64_t j = begin + lane; j < end; j += kWave) {
      const uint32_t key = order_key(row[j]);
      const uint32_t masked = key & mask;
      if (masked > prefix || (take_all_ties && masked == prefix)) {
        const int slot = atomicAdd(&s_count, 1);
        s_sel[slot] = candidate(key, static_cast<int32_t>(j));
      } else if (masked == prefix) {
        ++ties;
      }
    }
  }
  if (!take_all_ties) {
    // Ties at the threshold (the full key): the rem lowest ids, in id order.
    for (int o = kWave / 2; o > 0; o >>= 1)
      ties += __shfl_xor(ties, o, kWave);
    if (lane == 0)
      s_ties[wave] = ties;
    __syncthreads();
    int rank = 0;
    for (int w = 0; w < wave; ++w)
      rank += s_ties[w];
    for (int64_t base = begin; base < end && rank < rem; base += kWave) {
      const int64_t j = base + lane;
      const bool tie = j < end && order_key(row[j]) == prefix;
      const unsigned long long ballot = __ballot(tie);
      if (tie && rank + lanes_below(ballot, lane) < rem) {
        const int slot = atomicAdd(&s_count, 1);
        s_sel[slot] = candidate(prefix, static_cast<int32_t>(j));
      }
      rank += __popcll(ballot);
    }
  }
  __syncthreads();
  // Bitonic sort of the 64 slots, descending (empty slots sort last).
  if (tid < kMaxTopN && tid >= s_count)
    s_sel[tid] = 0ull;
  __syncthreads();
  for (int size = 2; size <= kMaxTopN; size <<= 1) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      if (tid < kMaxTopN) {
        const int other = tid ^ stride;
        if (other > tid) {
          const bool descending = (tid & size) == 0;
          const unsigned long long a = s_sel[tid], b = s_sel[other];
          if ((a < b) == descending) {
            s_sel[tid] = b;
            s_sel[other] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  if (tid < k) {
    const int32_t id = static_cast<int32_t>(~static_cast<uint32_t>(s_sel[tid]));
    top_ids[r * top_n + tid] = id;
    top_values[r * top_n + tid] = row[id];
  }

  // ---- 5. the categorical draw ----
  if (!draw) {
    if (tid == 0) {
      sampled[r] = -1;
      sampled_logit[r] = __int_as_float(0x7fc00000);
    }
    return;
  }
  if (!draw_ok) {
    if (tid == 0) {
      const int id = s_argmax[0];
      sampled[r] = id;
      sampled_logit[r] = row[id];
    }
    return;
  }
  __shared__ double s_target;
  __shared__ double s_before;
  if (tid == 0) {
    double total = 0.0;
    for (int w = 0; w < kWaves; ++w)
      total += s_sum_t[w];
    const double target = static_cast<double>(uniform[r]) * total;
    double cum = 0.0;
    int found = -1;
    for (int w = 0; w < kWaves; ++w) {
      if (target < cum + s_sum_t[w]) {
        found = w;
        break;
      }
      cum += s_sum_t[w];
    }
    s_target = target;
    s_before = cum;
    s_draw_wave = found;
    // Rounding left the target at the total: the last id with weight.
    int lastw = -1;
    for (int w = 0; w < kWaves; ++w)
      lastw = max(lastw, s_last[w]);
    s_drawn = found < 0 ? lastw : s_last[found];
  }
  __syncthreads();
  if (wave == s_draw_wave) {
    const double target = s_target;
    double cum = s_before;
    for (int64_t base = begin; base < end; base += kWave) {
      const int64_t j = base + lane;
      double w = 0.0;
      if (j < end) {
        const float v = row[j];
        if (!isnan(v))
          w = static_cast<double>(expf(v * inv_t - big_t));
      }
      const double incl = wave_scan(w, lane);
      const bool hit = target < cum + incl;
      const unsigned long long ballot = __ballot(hit);
      if (ballot != 0ull) {
        if (lane == __ffsll(static_cast<long long>(ballot)) - 1)
          s_drawn = static_cast<int>(j);
        break;
      }
      cum += __shfl(incl, kWave - 1, kWave);
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int id = s_drawn;
    sampled[r] = id;
    sampled_logit[r] = row[id];
  }
}

bool supported(const turbine_logits_reduce_desc *d) {
  return d != nullptr && d->rows >= 0 && d->rows <= INT32_MAX &&
         d->vocab >= 1 && d->vocab <= INT32_MAX && d->stride_row >= d->vocab &&
         d->top_n >= 0 && d->top_n <= kMaxTopN && d->top_n <= d->vocab;
}

std::string describe(const turbine_logits_reduce_desc *d) {
  return "rows=" + std::to_string(d->rows) +
         " vocab=" + std::to_string(d->vocab) +
         " stride_row=" + std::to_string(d->stride_row) +
         " top_n=" + std::to_string(d->top_n);
}

} // namespace

extern "C" {

int32_t turbine_logits_reduce_supported(const turbine_logits_reduce_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_logits_reduce_impl(const turbine_logits_reduce_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_logits_reduce(turbine_ctx *ctx,
                              const turbine_logits_reduce_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_logits_reduce: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_logits_reduce: unsupported configuration " +
                    describe(d));
  }
  if (d->rows == 0)
    return TURBINE_OK;
  if (d->logits == nullptr || d->temperature == nullptr ||
      d->uniform == nullptr || d->mode == nullptr || d->lse == nullptr ||
      d->sampled == nullptr || d->sampled_logit == nullptr ||
      (d->top_n > 0 && (d->top_ids == nullptr || d->top_values == nullptr))) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_logits_reduce: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  hipLaunchKernelGGL(logits_reduce_kernel,
                     dim3(static_cast<unsigned>(d->rows)), dim3(kThreads), 0,
                     ctx->stream, d->logits, d->vocab, d->stride_row,
                     d->temperature, d->uniform, d->mode, d->top_n, d->top_ids,
                     d->top_values, d->lse, d->sampled, d->sampled_logit);
  return check_hip(ctx, hipGetLastError(), "logits_reduce kernel launch");
}

} // extern "C"
