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
//   3. radix select of the top_n-th largest key on 32-bit keys ordered like
//      the logits (NaN lowest): round 0's histogram (of the top 8 bits) comes
//      from sweep 2; when the keys at or above round 0's digit number at most
//      kCandCap (the usual case), one sweep gathers them into shared memory and
//      a bitonic sort there yields the top_n exactly (steps 3-4 in one sweep);
//      otherwise 8 bits per round (at most 3 more sweeps), then
//   4. the collection of the top_n candidates (ties at the threshold by id),
//      sorted by (key descending, id ascending) in shared memory. A single
//      candidate (top_n 1, no nucleus) of a row whose maximum is finite and
//      non-zero is the argmax of step 2 (the smallest id holding the maximum:
//      the same id as the key order's), with no further sweep;
//   5. in mode 1, the draw: the wave whose cumulative sum passes u * total,
//      then a scan of that wave's range in id order;
//   6. in mode 1 with top_p < 1, the nucleus draw instead (below).
//
// The cpu reference sums f64 exponentials sequentially; here each exponential
// is an f32 expf and the sums are f64 in a fixed tree order, so the lse agrees
// to ~1e-7 relative and the draw differs only where u lies within that
// rounding of a CDF boundary.
//
// Nucleus (top_p < 1): the host sampler walks every id in descending order
// (ties by id), cuts the shortest prefix holding top_p of the mass and draws u
// times the prefix's mass along it. Here each weight is an integer, the f32
// expf weight (<= 1) times 2^S truncated, with S chosen so vocab * 2^S < 2^64:
// integer sums are exact in any order, so the cut and the draw are found by
// two searches for "the first id, in descending order, whose inclusive prefix
// mass reaches a target" and never depend on a summation order. A row selects
// kNucleusFast (64) sorted candidates in step 4; when their prefix already
// holds top_p of the mass (the usual case: a confident model's nucleus is a
// handful of ids) both searches run over them in shared memory. Otherwise the
// searches are radix selects by mass: 8 bits of the order key per round, a
// 256-bin mass histogram (64-bit shared atomics, exact) of the ids matching
// the digits so far, the digit where the running mass reaches the target,
// then the tied ids of the final key in id order. The first round's
// histogram is shared by both searches. Integer weights differ from the host's
// f64 exponentials by ~1e-7 relative, so the kept prefix and the drawn id
// differ from the cpu reference only where top_p * total or u * kept lies
// within that rounding of a prefix boundary.
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
// Candidates a nucleus row selects and sorts in step 4: a nucleus inside them
// is drawn from shared memory, a larger one by the mass searches.
constexpr int kNucleusFast = kMaxTopN;
// Keys step 3 gathers into shared memory (aliasing the per-wave histograms,
// kWaves * 256 * 4 bytes) instead of running radix rounds 1-3.
constexpr int kCandCap = kWaves * 256 * 4 / 8;

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
__device__ inline void wave_count(unsigned int *hist, uint32_t bin, bool active,
                                  int lane) {
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

__device__ inline int wave_sum_int(int v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v += __shfl_xor(v, o, kWave);
  return v;
}

// The float an order key came from (order_key is a bijection on non-NaN).
__device__ inline float key_value(uint32_t k) {
  return (k & 0x80000000u) ? __uint_as_float(k & 0x7fffffffu)
                           : __uint_as_float(~k);
}

// exp(v / T - max / T) in f32 as the host computes it: v * (1/T) rounded,
// then the difference rounded (no fused multiply-add, so every sweep of the
// row gets the same weight).
__device__ inline float scaled_weight(float v, float inv_t, float big_t) {
  return expf(__fsub_rn(__fmul_rn(v, inv_t), big_t));
}

// w * 2^s truncated, for a finite w in [0, 1] and s < 64: the exact
// product (a float times a power of two) truncated toward zero, computed on
// the bits (equal to static_cast<unsigned long long>(double(w) * 2^s), without
// f64 arithmetic).
__device__ inline unsigned long long scale_trunc(float w, int s) {
  const uint32_t bits = __float_as_uint(w) & 0x7fffffffu;
  const int e = static_cast<int>(bits >> 23);
  const unsigned long long m =
      e == 0 ? (bits & 0x7fffffu) : ((bits & 0x7fffffu) | 0x800000u);
  // w = m * 2^(max(e, 1) - 150).
  const int shift = (e == 0 ? 1 : e) - 150 + s;
  if (shift >= 0)
    return m << shift;
  return shift <= -64 ? 0ull : m >> (-shift);
}

// The integer nucleus weight of a raw logit: scaled_weight (<= 1, NaN 0)
// times 2^S, truncated.
__device__ inline unsigned long long qweight(float v, float inv_t, float big_t,
                                             int scale_bits) {
  if (isnan(v))
    return 0ull;
  return scale_trunc(scaled_weight(v, inv_t, big_t), scale_bits);
}

// Smallest integer mass >= top_p * total (the cut: cum >= top_p * total).
__device__ inline unsigned long long cut_target(float top_p,
                                                unsigned long long total) {
  const double x = static_cast<double>(top_p) * static_cast<double>(total);
  const unsigned long long t = static_cast<unsigned long long>(ceil(x));
  return t < total ? t : total;
}

// Smallest integer mass > u * kept (the draw: u * kept < cum).
__device__ inline unsigned long long draw_target(float u,
                                                 unsigned long long kept) {
  const double x = static_cast<double>(u) * static_cast<double>(kept);
  const unsigned long long t = static_cast<unsigned long long>(floor(x)) + 1ull;
  return t < kept ? t : kept;
}

// Shared state of the nucleus searches.
struct NucleusShared {
  // Round-0 mass histogram (shared by both searches), then later rounds'.
  unsigned long long mass0[256];
  unsigned long long mass[256];
  // Mass of the ids ranked before the current digit prefix.
  unsigned long long above;
  unsigned long long target;
  // The search's answer: its id and inclusive prefix mass.
  unsigned long long found_mass;
  uint32_t prefix, mask;
  int found;
  int need;
  unsigned long long tie_q;
  int ties[kWaves];
};

// Mass (integer weights) per 8-bit digit at `shift` of the keys whose bits
// under `mask` equal `prefix`. Every thread of the block calls it.
__device__ void mass_histogram(const float *row, int64_t begin, int64_t end,
                               int tid, int lane, uint32_t prefix,
                               uint32_t mask, int shift, float inv_t,
                               float big_t, int scale_bits,
                               unsigned long long *hist) {
  if (tid < 256)
    hist[tid] = 0ull;
  __syncthreads();
  for (int64_t j = begin + lane; j < end; j += kWave) {
    const float v = row[j];
    const uint32_t key = order_key(v);
    if ((key & mask) == prefix) {
      const unsigned long long q = qweight(v, inv_t, big_t, scale_bits);
      if (q != 0ull)
        atomicAdd(&hist[(key >> shift) & 0xffu], q);
    }
  }
  __syncthreads();
}

// The first id in (key descending, id ascending) order whose inclusive prefix
// mass is >= target (1 <= target <= the row's mass), into sh.found and
// sh.found_mass; sh.mass0 holds the round-0 histogram. Every thread of the
// block calls it; `fallback` answers if rounding ever left no such id.
__device__ void mass_search(const float *row, int64_t begin, int64_t end,
                            int tid, int wave, int lane, float inv_t,
                            float big_t, int scale_bits,
                            unsigned long long target, int fallback,
                            NucleusShared &sh) {
  if (tid == 0) {
    sh.prefix = 0u;
    sh.mask = 0u;
    sh.above = 0ull;
    sh.target = target;
  }
  __syncthreads();
  for (int round = 0; round < 4; ++round) {
    const int shift = 24 - 8 * round;
    const unsigned long long *h = sh.mass0;
    if (round > 0) {
      mass_histogram(row, begin, end, tid, lane, sh.prefix, sh.mask, shift,
                     inv_t, big_t, scale_bits, sh.mass);
      h = sh.mass;
    }
    if (tid == 0) {
      unsigned long long above = sh.above;
      int digit = 0;
      for (int b = 255; b >= 0; --b) {
        if (above + h[b] >= sh.target) {
          digit = b;
          break;
        }
        above += h[b];
      }
      sh.above = above;
      sh.prefix |= static_cast<uint32_t>(digit) << shift;
      sh.mask |= 0xffu << shift;
    }
    __syncthreads();
  }
  // The ids with the final key all weigh tie_q: the target is reached at
  // the need-th of them in id order.
  const uint32_t key = sh.prefix;
  if (tid == 0) {
    const unsigned long long q =
        qweight(key_value(key), inv_t, big_t, scale_bits);
    const unsigned long long rest = sh.target - sh.above;
    sh.tie_q = q;
    sh.need = q == 0ull ? 1 : static_cast<int>((rest + q - 1ull) / q);
    sh.found = -1;
  }
  int ties = 0;
  for (int64_t j = begin + lane; j < end; j += kWave)
    ties += order_key(row[j]) == key ? 1 : 0;
  ties = wave_sum_int(ties);
  if (lane == 0)
    sh.ties[wave] = ties;
  __syncthreads();
  int rank = 0;
  for (int w = 0; w < wave; ++w)
    rank += sh.ties[w];
  const int want = sh.need - 1;
  if (want >= rank && want < rank + sh.ties[wave]) {
    for (int64_t base = begin; base < end; base += kWave) {
      const int64_t j = base + lane;
      const bool tie = j < end && order_key(row[j]) == key;
      const unsigned long long ballot = __ballot(tie);
      if (tie && rank + lanes_below(ballot, lane) == want)
        sh.found = static_cast<int>(j);
      rank += __popcll(ballot);
      if (rank > want)
        break;
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (sh.found < 0)
      sh.found = fallback;
    sh.found_mass =
        sh.above + static_cast<unsigned long long>(sh.need) * sh.tie_q;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads)
    logits_reduce_kernel(const float *logits, int64_t vocab, int64_t stride_row,
                         const float *temperature, const float *uniform,
                         const int32_t *mode, int32_t top_n, int32_t *top_ids,
                         float *top_values, float *lse_out, int32_t *sampled,
                         float *sampled_logit, const float *top_p) {
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
  // Nucleus rows: the integer mass of the row, the sorted candidates'
  // integer weights, and the search state.
  __shared__ unsigned long long s_qtotal;
  __shared__ unsigned long long s_sel_q[kNucleusFast];
  __shared__ bool s_fast;
  __shared__ NucleusShared s_nuc;

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
  if (tid == 0) {
    s_count = 0;
    s_qtotal = 0ull;
  }
  __syncthreads();
  float big = -INFINITY;
  for (int w = 0; w < kWaves; ++w)
    big = fmaxf(big, s_max[w]);

  const bool draw = mode[r] == 1;
  const float t = temperature[r];
  const float inv_t = 1.0f / t;
  const float big_t = big * inv_t;
  const bool draw_ok = draw && t > 0.0f && isfinite(big_t);
  const float tp = draw && top_p != nullptr ? top_p[r] : 1.0f;
  const bool nucleus = draw_ok && tp < 1.0f;
  // Integer weight scale 2^S with vocab * 2^S < 2^64 (S = 46 at 128,256).
  const int vocab_bits = 64 - __clzll(static_cast<unsigned long long>(vocab));
  const int qbits = 63 - vocab_bits;

  // ---- 2. sums, argmax, last positive weight, first histogram ----
  double sum = 0.0;
  double sum_t = 0.0;
  const bool unit_t = inv_t == 1.0f && big_t == big;
  // Round 0's histogram, unless step 3 needs no radix select (the argmax
  // shortcut below, or no candidates at all).
  const int64_t k_hist = nucleus ? kNucleusFast : top_n;
  const bool need_hist =
      min64(k_hist, vocab) > 1 ||
      (min64(k_hist, vocab) == 1 && !(isfinite(big) && big != 0.0f));
  unsigned long long qsum = 0ull;
  int arg = INT_MAX;
  int last = -1;
  for (int64_t base = begin; base < end; base += kWave) {
    const int64_t j = base + lane;
    const bool in = j < end;
    const float v = in ? row[j] : 0.0f;
    if (need_hist)
      wave_count(hist, order_key(v) >> 24, in, lane);
    if (in && !isnan(v)) {
      if (isfinite(big))
        sum += static_cast<double>(expf(v - big));
      if (v == big && big > -INFINITY)
        arg = min(arg, static_cast<int>(j));
      if (draw_ok) {
        const float w = scaled_weight(v, inv_t, big_t);
        // The id-order draw's mass: at T = 1 every weight equals the lse term
        // (v * 1 is exact), so the lse sum serves (below); a nucleus row draws
        // by integer masses and never reads it.
        if (!nucleus && !unit_t)
          sum_t += static_cast<double>(w);
        if (nucleus)
          qsum += scale_trunc(w, qbits);
        if (w > 0.0f)
          last = static_cast<int>(j);
      }
    }
  }
  sum = wave_sum(sum);
  sum_t = unit_t ? sum : wave_sum(sum_t);
  arg = wave_min(arg);
  last = wave_max_int(last);
  if (lane == 0) {
    s_sum[wave] = sum;
    s_sum_t[wave] = sum_t;
    s_argmax[wave] = arg;
    s_last[wave] = last;
  }
  if (nucleus && qsum != 0ull)
    atomicAdd(&s_qtotal, qsum);
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
  // A nucleus row selects kNucleusFast candidates (it returns top_n of them).
  const int k_sel = nucleus ? static_cast<int>(min64(kNucleusFast, vocab)) : k;
  __shared__ bool s_gather;
  __shared__ bool s_selected;
  __shared__ int s_ncand;
  // A single candidate is the argmax: the smallest id holding the maximum.
  // (Not at a zero maximum: -0 and +0 compare equal but order as keys.)
  const bool argmax_only = k_sel == 1 && isfinite(big) && big != 0.0f;
  if (tid == 0) {
    s_prefix = 0u;
    s_mask = 0u;
    s_rem = k_sel;
    s_done = k_sel == 0 || argmax_only;
    s_gather = false;
    s_selected = argmax_only;
    if (argmax_only) {
      const int a = s_argmax[0];
      s_sel[0] = candidate(order_key(row[a]), static_cast<int32_t>(a));
      s_count = 1;
    }
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
      // Round 0: every key at or above the digit fits in shared memory
      // (above keys taken outright plus the digit's): gather and sort them.
      const int ncand = above + static_cast<int>(s_hist[digit]);
      rem -= above;
      s_prefix |= static_cast<uint32_t>(digit) << shift;
      s_mask |= 0xffu << shift;
      // Every key with this prefix is taken: no tie to break.
      s_done = static_cast<int>(s_hist[digit]) == rem;
      s_rem = rem;
      if (round == 0 && !s_done && ncand <= kCandCap) {
        s_gather = true;
        s_ncand = ncand;
        s_done = true;
      }
    }
    __syncthreads();
  }
  if (s_gather) {
    // ---- 3-4 in one sweep: gather every key >= the round-0 prefix ----
    unsigned long long *cand =
        reinterpret_cast<unsigned long long *>(s_wave_hist);
    const uint32_t floor_key = s_prefix;
    if (tid == 0)
      s_count = 0;
    __syncthreads();
    for (int64_t j = begin + lane; j < end; j += kWave) {
      const uint32_t key = order_key(row[j]);
      if (key >= floor_key) {
        const int slot = atomicAdd(&s_count, 1);
        cand[slot] = candidate(key, static_cast<int32_t>(j));
      }
    }
    __syncthreads();
    const int n = s_count;
    int size = kMaxTopN;
    while (size < n)
      size <<= 1;
    for (int i = n + tid; i < size; i += kThreads)
      cand[i] = 0ull;
    __syncthreads();
    // Bitonic sort, descending: (key descending, id ascending).
    for (int width = 2; width <= size; width <<= 1) {
      for (int stride = width / 2; stride > 0; stride >>= 1) {
        for (int i = tid; i < size; i += kThreads) {
          const int other = i ^ stride;
          if (other > i) {
            const bool descending = (i & width) == 0;
            const unsigned long long a = cand[i], b = cand[other];
            if ((a < b) == descending) {
              cand[i] = b;
              cand[other] = a;
            }
          }
        }
        __syncthreads();
      }
    }
    if (tid < k_sel)
      s_sel[tid] = cand[tid];
    if (tid == 0) {
      s_count = k_sel;
      s_selected = true;
    }
    __syncthreads();
  }

  // ---- 4. collect: keys above the prefix, then ties by id ----
  const uint32_t prefix = s_prefix, mask = s_mask;
  const int rem = s_rem;
  const bool take_all_ties = s_done;
  int ties = 0;
  const bool selected = s_selected;
  if (k_sel > 0 && !selected) {
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
  if (!take_all_ties && !selected) {
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
  if (tid < k_sel) {
    const int32_t id = static_cast<int32_t>(~static_cast<uint32_t>(s_sel[tid]));
    const float v = row[id];
    if (tid < k) {
      top_ids[r * top_n + tid] = id;
      top_values[r * top_n + tid] = v;
    }
    if (nucleus)
      s_sel_q[tid] = qweight(v, inv_t, big_t, qbits);
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
  if (nucleus) {
    // ---- 6. the nucleus draw ----
    const float u = uniform[r];
    __syncthreads();
    if (tid == 0) {
      const unsigned long long cut = cut_target(tp, s_qtotal);
      unsigned long long cum = 0ull;
      int last = -1;
      for (int i = 0; i < k_sel; ++i) {
        cum += s_sel_q[i];
        if (cum >= cut) {
          last = i;
          break;
        }
      }
      s_fast = last >= 0;
      if (last >= 0) {
        const unsigned long long target = draw_target(u, cum);
        unsigned long long c = 0ull;
        int pick = last;
        for (int i = 0; i <= last; ++i) {
          c += s_sel_q[i];
          if (c >= target) {
            pick = i;
            break;
          }
        }
        s_drawn = static_cast<int>(~static_cast<uint32_t>(s_sel[pick]));
      }
    }
    __syncthreads();
    if (!s_fast) {
      // The nucleus reaches past the sorted candidates: the mass searches.
      mass_histogram(row, begin, end, tid, lane, 0u, 0u, 24, inv_t, big_t,
                     qbits, s_nuc.mass0);
      const int fallback = s_argmax[0];
      mass_search(row, begin, end, tid, wave, lane, inv_t, big_t, qbits,
                  cut_target(tp, s_qtotal), fallback, s_nuc);
      const unsigned long long kept = s_nuc.found_mass;
      mass_search(row, begin, end, tid, wave, lane, inv_t, big_t, qbits,
                  draw_target(u, kept), fallback, s_nuc);
      if (tid == 0)
        s_drawn = s_nuc.found;
    }
    __syncthreads();
    if (tid == 0) {
      const int id = s_drawn;
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
          w = static_cast<double>(scaled_weight(v, inv_t, big_t));
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
  hipLaunchKernelGGL(logits_reduce_kernel, dim3(static_cast<unsigned>(d->rows)),
                     dim3(kThreads), 0, ctx->stream, d->logits, d->vocab,
                     d->stride_row, d->temperature, d->uniform, d->mode,
                     d->top_n, d->top_ids, d->top_values, d->lse, d->sampled,
                     d->sampled_logit, d->top_p);
  return check_hip(ctx, hipGetLastError(), "logits_reduce kernel launch");
}

} // extern "C"
