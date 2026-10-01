// turbine_attn_eval: the mixed-format / TurboQuant paged attention provider
// evaluation (Phase 6b Task 10, kernel reuse rule) on the current device.
//
// One layer of a ragged batch over 128-token pages, Llama-3.2-3B (24 query /
// 8 KV heads) and OLMoE-1B-7B (16 / 16) head layouts, head_dim 128. For each
// shape it builds pages in every KV format -- bf16, fp8_e4m3 (per-layer
// scales), tq4 and tq2 (the TurboQuant records of crates/turbine-kv
// codec/turboquant: per (KV head, token) K codes, K norm, QJL signs, residual
// norm, V codes, V norm, 144 / 80 bytes) -- and runs:
//
//   bf16:<impl>      every turbine_attention_{decode,prefill}_paged
//                    implementation of libturbine_hip.so over BF16 pages
//                    (turbine_impl_run by index): the BF16 KV baseline
//   fp8:<impl>       the same over FP8 pages (6a's implementations)
//   tqN:staged_ck    the TurboQuant pages decoded to BF16 pages by the Task 8
//                    transcode (turbine_kv_transcode, turbine_hip_tq, one
//                    layer) and then the first CK BF16 implementation on them:
//                    "dequantize, then an existing kernel" (6a's FP8 prefill
//                    scheme), timed together
//   tqN:own_rot      a prototype own kernel (this file): one workgroup per
//   mixed:own_rot    (query row, KV head), every query head of the group at
//                    once; blocks read by their format tag; TurboQuant keys
//                    scored in the rotated domain (q rotated once and
//                    projected once by the QJL matrix S, per row and head),
//                    TurboQuant values accumulated in the rotated domain and
//                    rotated back once; BF16 / FP8 blocks as 6a's FP8 decode
//                    kernel reads them (online softmax over chunks of 32 keys)
//
// and checks every output against a host reference with cpu::tq_attention's
// semantics (crates/turbine-kernels/src/cpu/tq_attention.rs, DecodeThenAttend:
// each TurboQuant record decoded to F32 exactly as the codec's decode_record,
// FP8 bytes as e4m3 * scale, then exact attention accumulated in F64). The
// staged and ABI candidates write k_new / v_new into the pages first (the
// paged append); their new rows are given as the values the pages hold, so
// the append changes nothing the reference reads (the reference is built from
// the pages read back after the call). The own kernel has no append: it reads
// the new rows from the pages, like Task 12's kernel after its append.
//
// Timing: microseconds per call, median of --rounds rounds of --iters calls,
// rotating over enough copies of the pool (and the TurboQuant tables) to span
// 256 MiB, past the R9700's 64 MiB Infinity Cache.
//
// The other candidates (llama.cpp's HIP flash attention with q4_0 / q8_0 KV,
// timed with its own test-backend-ops; CK's FP8 FMHA; vLLM / aiter ROCm paged
// attention) are judged in the decision "P6b: mixed-format / TurboQuant paged
// attention — provider evaluation (kernel reuse rule)".
//
//   turbine_attn_eval [--models llama,olmoe] [--phases decode,prefill,mixed]
//                     [--iters 20] [--rounds 5] [--check-only 1]
//
// A lab tool (run it on GPU 0 under scripts/bench-lock.sh); not loaded by the
// server. Exit 0 when every candidate matched the reference, 1 otherwise.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "bf16.hpp"
#include "fp8.hpp"
#include "turbine_kernels.h"

namespace {

constexpr int kD = 128;
constexpr int kBlockTokens = 128;
constexpr int kTableElems = 2 * kD + kD * kD;
constexpr size_t kRotateBytes = 256u << 20;
constexpr uint64_t kSeed = 0x6b10;

enum Fmt : uint8_t { kBf16 = 0, kFp8 = 1, kTq4 = 2, kTq2 = 3 };
const char *fmt_name(int f) {
  static const char *n[] = {"bf16", "fp8", "tq4", "tq2"};
  return n[f];
}

// ---- record layout (crates/turbine-kv codec/turboquant/mod.rs) ----

struct Rec {
  int kb, vb, k_norm, qjl, r_norm, v_codes, v_norm, bytes;
};
__host__ __device__ constexpr Rec rec_of(int kb, int vb) {
  const int k_norm = kD * kb / 8;
  const int qjl = k_norm + 2;
  const int r_norm = qjl + kD / 8;
  const int v_codes = r_norm + 2;
  const int v_norm = v_codes + kD * vb / 8;
  return Rec{kb,     vb,      k_norm, qjl,
             r_norm, v_codes, v_norm, (v_norm + 2 + 15) / 16 * 16};
}
constexpr Rec kRecTq4 = rec_of(3, 4);
constexpr Rec kRecTq2 = rec_of(1, 2);
static_assert(kRecTq4.bytes == 144 && kRecTq2.bytes == 80, "record sizes");

size_t page_bytes(int fmt, int hkv) {
  const size_t elems = size_t{2} * kBlockTokens * hkv * kD;
  switch (fmt) {
  case kBf16:
    return 2 * elems;
  case kFp8:
    return elems;
  case kTq4:
    return size_t{1} * hkv * kBlockTokens * kRecTq4.bytes;
  default:
    return size_t{1} * hkv * kBlockTokens * kRecTq2.bytes;
  }
}

// Unit-variance Lloyd-Max codebooks (codebook.rs), [bits - 1][16].
const float kCodebooks[4][16] = {
    {-0.79944444f, 0.79944444f},
    {-1.505193f, -0.45245326f, 0.45245326f, 1.505193f},
    {-2.131471f, -1.3365989f, -0.7533302f, -0.24442486f, 0.24442486f,
     0.7533302f, 1.3365989f, 2.131471f},
    {-2.6888592f, -2.0459254f, -1.6043426f, -1.2477703f, -0.93709695f,
     -0.6536189f, -0.38638088f, -0.12787314f, 0.12787314f, 0.38638088f,
     0.6536189f, 0.93709695f, 1.2477703f, 1.6043426f, 2.0459254f, 2.6888592f}};

// ---- host helpers ----

void check_hip(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
  }
}

bool ok_t(turbine_ctx *ctx, int32_t rc, const char *what) {
  if (rc == TURBINE_OK)
    return true;
  char buf[1024] = {0};
  turbine_last_error(ctx, buf, sizeof(buf));
  std::fprintf(stderr, "%s: %d %s\n", what, rc, buf);
  return false;
}

uint16_t to_bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

float e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const float s = (b & 0x80) ? -1.0f : 1.0f;
  if (e == 15 && m == 7)
    return NAN;
  if (e == 0)
    return s * static_cast<float>(m) * 0.001953125f;
  return s * std::ldexp(1.0f + m / 8.0f, e - 7);
}

// Nearest e4m3fn code (ties to the even code), saturated to +-448.
uint8_t to_e4m3(float x) {
  static std::vector<float> table;
  if (table.empty())
    for (int c = 0; c < 128; ++c)
      table.push_back(c == 127 ? NAN : e4m3_value(static_cast<uint8_t>(c)));
  const float a = std::fabs(x);
  int best = 0;
  float bd = INFINITY;
  for (int c = 0; c < 127; ++c) {
    const float d = std::fabs(table[c] - a);
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      best = c;
    }
  }
  return static_cast<uint8_t>(best | (x < 0 ? 0x80 : 0));
}

template <typename T> void fwht(T *x, int n) {
  for (int h = 1; h < n; h *= 2)
    for (int i = 0; i < n; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const T a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
}

void parallel_for(int64_t n, const std::function<void(int64_t)> &f) {
  const int threads = 8;
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (int64_t i = t; i < n; i += threads)
        f(i);
    });
  for (auto &th : pool)
    th.join();
}

uint32_t get_code(const uint8_t *packed, int bits, int i) {
  uint32_t c = 0;
  for (int j = 0; j < bits; ++j) {
    const int bit = i * bits + j;
    c |= static_cast<uint32_t>((packed[bit / 8] >> (bit % 8)) & 1) << j;
  }
  return c;
}

void put_code(uint8_t *packed, int bits, int i, uint32_t c) {
  for (int j = 0; j < bits; ++j) {
    const int bit = i * bits + j;
    if ((c >> j) & 1)
      packed[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
  }
}

uint16_t rd16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | p[1] << 8);
}
void wr16(uint8_t *p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}

// One KV head's TurboQuant tables: K signs, V signs, QJL S (row-major).
struct HeadTables {
  const float *ks, *vs, *s;
};

// Quantizes x (rotated with signs) to `bits`-bit codes; returns the rotated
// vector minus its reconstruction (the residual) in res.
void tq_quantize(const float *x, const float *signs, int bits, uint8_t *codes,
                 uint8_t *norm_at, double *res) {
  double y[kD];
  double n2 = 0;
  for (int i = 0; i < kD; ++i) {
    y[i] = static_cast<double>(x[i]) * signs[i];
    n2 += static_cast<double>(x[i]) * x[i];
  }
  fwht(y, kD);
  const double sd = std::sqrt(static_cast<double>(kD));
  for (double &v : y)
    v /= sd;
  const uint16_t nb = to_bf16(static_cast<float>(std::sqrt(n2)));
  wr16(norm_at, nb);
  const double n = from_bf16(nb);
  const double inv = n > 0 ? sd / n : 0;
  const float *cb = kCodebooks[bits - 1];
  const int levels = 1 << bits;
  for (int i = 0; i < kD; ++i) {
    const double u = y[i] * inv;
    int c = 0;
    while (c + 1 < levels && u > 0.5 * (cb[c] + cb[c + 1]))
      ++c;
    put_code(codes, bits, i, static_cast<uint32_t>(c));
    if (res)
      res[i] = y[i] - cb[c] * n / sd;
  }
}

void tq_encode(int fmt, const float *k, const float *v, const HeadTables &h,
               uint8_t *rec) {
  const Rec r = fmt == kTq4 ? kRecTq4 : kRecTq2;
  std::memset(rec, 0, r.bytes);
  double res[kD];
  tq_quantize(k, h.ks, r.kb, rec, rec + r.k_norm, res);
  double rn2 = 0;
  for (int i = 0; i < kD; ++i) {
    double p = 0;
    for (int j = 0; j < kD; ++j)
      p += static_cast<double>(h.s[i * kD + j]) * res[j];
    if (p >= 0)
      rec[r.qjl + i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    rn2 += res[i] * res[i];
  }
  wr16(rec + r.r_norm, to_bf16(static_cast<float>(std::sqrt(rn2))));
  tq_quantize(v, h.vs, r.vb, rec + r.v_codes, rec + r.v_norm, nullptr);
}

// cpu::tq_attention's decode_record, step for step, in F32.
void tq_decode(int fmt, const uint8_t *rec, const HeadTables &h, float *k,
               float *v) {
  const Rec r = fmt == kTq4 ? kRecTq4 : kRecTq2;
  const float sd = static_cast<float>(std::sqrt(static_cast<double>(kD)));
  const float *kcb = kCodebooks[r.kb - 1];
  const float *vcb = kCodebooks[r.vb - 1];
  const float kk = from_bf16(rd16(rec + r.k_norm)) / sd;
  float y[kD], res[kD] = {0};
  for (int i = 0; i < kD; ++i)
    y[i] = kcb[get_code(rec, r.kb, i)] * kk;
  for (int i = 0; i < kD; ++i) {
    const float z = (rec[r.qjl + i / 8] >> (i % 8)) & 1 ? 1.0f : -1.0f;
    for (int j = 0; j < kD; ++j)
      res[j] += z * h.s[i * kD + j];
  }
  const float rk = static_cast<float>(std::sqrt(M_PI / 2.0) / kD) *
                   from_bf16(rd16(rec + r.r_norm));
  for (int i = 0; i < kD; ++i)
    y[i] += res[i] * rk;
  const float inv =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(kD)));
  fwht(y, kD);
  for (int i = 0; i < kD; ++i)
    k[i] = y[i] * inv * h.ks[i];
  const float vk = from_bf16(rd16(rec + r.v_norm)) / sd;
  for (int i = 0; i < kD; ++i)
    y[i] = vcb[get_code(rec + r.v_codes, r.vb, i)] * vk;
  fwht(y, kD);
  for (int i = 0; i < kD; ++i)
    v[i] = y[i] * inv * h.vs[i];
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

bool listed(const std::string &list, const char *name) {
  return ("," + list + ",").find("," + std::string(name) + ",") !=
         std::string::npos;
}

} // namespace

// ---- the own prototype kernel ----

namespace own {

using turbine_hip::Bf16;
using turbine_hip::bf16_to_f32;
using turbine_hip::f32_to_bf16;
using turbine_hip::fp8_dequant;

constexpr int kThreads = 256;
constexpr int kLanes = 32;
constexpr int kWaves = kThreads / kLanes;
constexpr int kDpl = kD / kLanes; // head dims per lane in P.V
constexpr float kNegInf = -__builtin_huge_valf();

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

template <int G> struct Lds {
  float q[G][kD];
  float rq[G][kD]; // H (s_k . q) / sqrt(d)
  float sq[G][kD]; // S . rq
  float cb[4][16];
  float vs[kD];
  union {
    struct {
      float p[kWaves][G][kLanes];
      int64_t voff[kWaves][kLanes];
      float vmul[kWaves][kLanes];
      int32_t vfmt[kWaves][kLanes];
    } loop;
    float o[2][kWaves][G][kD]; // plain, rotated partial outputs
  };
  float max[kWaves][G];
  float sum[kWaves][G];
  float oplain[G][kD];
  float orot[G][kD];
  float total[G];
};

__device__ inline uint32_t bits_at(const uint32_t *w, int bit, int n) {
  const int wi = bit >> 5, sh = bit & 31;
  uint32_t v = w[wi] >> sh;
  if (sh + n > 32)
    v |= w[wi + 1] << (32 - sh);
  return v & ((1u << n) - 1u);
}

// The score of one TurboQuant key for G query heads (scale folded in).
template <int G, int KB, int Words>
__device__ inline void tq_score(const uint8_t *rec, const Lds<G> &lds,
                                float c_mse, float c_qjl, float *s) {
  constexpr Rec r = rec_of(KB, KB == 3 ? 4 : 2);
  uint32_t w[Words];
  const uint4 *p16 = reinterpret_cast<const uint4 *>(rec);
#pragma unroll
  for (int i = 0; i < Words / 4; ++i) {
    const uint4 x = p16[i];
    w[4 * i] = x.x;
    w[4 * i + 1] = x.y;
    w[4 * i + 2] = x.z;
    w[4 * i + 3] = x.w;
  }
  const float kn = __uint_as_float(bits_at(w, r.k_norm * 8, 16) << 16);
  const float rn = __uint_as_float(bits_at(w, r.r_norm * 8, 16) << 16);
  float mse[G], qj[G];
#pragma unroll
  for (int g = 0; g < G; ++g)
    mse[g] = qj[g] = 0.0f;
#pragma unroll
  for (int i = 0; i < kD; ++i) {
    const float c = lds.cb[KB - 1][bits_at(w, i * KB, KB)];
    const bool z = bits_at(w, r.qjl * 8 + i, 1) != 0;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      mse[g] += lds.rq[g][i] * c;
      qj[g] += z ? lds.sq[g][i] : -lds.sq[g][i];
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g)
    s[g] = c_mse * kn * mse[g] + c_qjl * rn * qj[g];
}

// Grid (query rows, KV heads), kThreads threads. block_formats null: every
// block is base_fmt. slot_bytes: the pool's page stride.
template <int G>
__global__ void __launch_bounds__(kThreads)
    mixed_attn_kernel(const Bf16 *q, const uint8_t *pool, Bf16 *out,
                      const int32_t *row_seq, const int32_t *block_table,
                      const uint8_t *block_formats, int32_t base_fmt,
                      const int32_t *q_indptr, const int32_t *kv_lens,
                      int32_t max_blocks, int64_t slot_bytes, int32_t kv_heads,
                      int32_t group, int32_t has_tq, const float *tables,
                      const float *codebooks, float scale, float k_scale,
                      float v_scale, int32_t causal) {
  __shared__ Lds<G> lds;
  const int tid = threadIdx.x;
  const int lane = tid % kLanes;
  const int wave = tid / kLanes;
  const int32_t row = blockIdx.x;
  const int32_t kvh = blockIdx.y;
  const int32_t seq = row_seq[row];
  const int32_t q_len = q_indptr[seq + 1] - q_indptr[seq];
  const int32_t kv_len = kv_lens[seq];
  const int32_t visible =
      causal ? kv_len - q_len + (row - q_indptr[seq]) + 1 : kv_len;
  const int32_t bt = kBlockTokens;
  const int64_t q_stride = static_cast<int64_t>(kv_heads) * group * kD;
  const int32_t *table = block_table + static_cast<int64_t>(seq) * max_blocks;
  const uint8_t *fmts =
      block_formats ? block_formats + static_cast<int64_t>(seq) * max_blocks
                    : nullptr;
  const float *tab = tables + static_cast<int64_t>(kvh) * kTableElems;
  const float inv_sd = 0.08838834764831845f; // 1/sqrt(128)

  for (int e = tid; e < G * kD; e += kThreads) {
    const int g = e / kD, d = e % kD;
    float v = 0.0f;
    if (g < group)
      v = bf16_to_f32(q[row * q_stride + (kvh * group + g) * kD + d]);
    lds.q[g][d] = v;
    if (has_tq)
      lds.rq[g][d] = v * tab[d];
  }
  if (tid < 64)
    lds.cb[tid / 16][tid % 16] = codebooks[tid];
  if (has_tq && tid < kD)
    lds.vs[tid] = tab[kD + tid];
  __syncthreads();
  if (has_tq) {
    for (int h = 1; h < kD; h *= 2) {
      for (int p = tid; p < G * (kD / 2); p += kThreads) {
        const int g = p / (kD / 2), j = p % (kD / 2);
        const int i0 = (j / h) * 2 * h + j % h;
        const float a = lds.rq[g][i0], b = lds.rq[g][i0 + h];
        lds.rq[g][i0] = a + b;
        lds.rq[g][i0 + h] = a - b;
      }
      __syncthreads();
    }
    for (int e = tid; e < G * kD; e += kThreads)
      lds.rq[e / kD][e % kD] *= inv_sd;
    __syncthreads();
    // sq[g][i] = S[i] . rq[g]: one wave per row of S.
    const float *S = tab + 2 * kD;
    for (int i = wave; i < kD; i += kWaves) {
      const float4 srow = reinterpret_cast<const float4 *>(S + i * kD)[lane];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float *r = &lds.rq[g][lane * 4];
        float dot =
            srow.x * r[0] + srow.y * r[1] + srow.z * r[2] + srow.w * r[3];
        dot = lane_sum(dot);
        if (lane == 0)
          lds.sq[g][i] = dot;
      }
    }
    __syncthreads();
  }
  const float c_mse = scale * inv_sd;
  const float c_qjl = scale * 1.2533141373155003f / kD; // sqrt(pi/2)/d

  float m[G], sum[G], accp[G][kDpl], accr[G][kDpl];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = kNegInf;
    sum[g] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDpl; ++i)
      accp[g][i] = accr[g][i] = 0.0f;
  }

  for (int32_t c0 = wave * kLanes; c0 < visible; c0 += kWaves * kLanes) {
    const int32_t key = c0 + lane;
    const bool valid = key < visible;
    float s[G];
#pragma unroll
    for (int g = 0; g < G; ++g)
      s[g] = 0.0f;
    int64_t voff = -1;
    float vmul = 1.0f;
    int32_t fmt = 0;
    if (valid) {
      const int32_t page = key / bt;
      const int32_t t = key % bt;
      fmt = fmts ? fmts[page] : base_fmt;
      const int64_t base = static_cast<int64_t>(table[page]) * slot_bytes;
      if (fmt == kBf16 || fmt == kFp8) {
        const int es = fmt == kBf16 ? 2 : 1;
        const int64_t koff =
            base + (static_cast<int64_t>(t) * kv_heads + kvh) * kD * es;
        voff = base + (static_cast<int64_t>(bt + t) * kv_heads + kvh) * kD * es;
        if (fmt == kBf16) {
          const Bf16 *kr = reinterpret_cast<const Bf16 *>(pool + koff);
          for (int d = 0; d < kD; ++d) {
            const float kv = bf16_to_f32(kr[d]);
#pragma unroll
            for (int g = 0; g < G; ++g)
              s[g] += lds.q[g][d] * kv;
          }
        } else {
          vmul = v_scale;
          const uint8_t *kr = pool + koff;
          for (int d = 0; d < kD; ++d) {
            const float kv = fp8_dequant(kr[d], k_scale);
#pragma unroll
            for (int g = 0; g < G; ++g)
              s[g] += lds.q[g][d] * kv;
          }
        }
#pragma unroll
        for (int g = 0; g < G; ++g)
          s[g] *= scale;
      } else {
        const Rec r = fmt == kTq4 ? rec_of(3, 4) : rec_of(1, 2);
        const uint8_t *rec =
            pool + base + (static_cast<int64_t>(kvh) * bt + t) * r.bytes;
        if (fmt == kTq4)
          tq_score<G, 3, 20>(rec, lds, c_mse, c_qjl, s);
        else
          tq_score<G, 1, 12>(rec, lds, c_mse, c_qjl, s);
        voff = (rec - pool) + r.v_codes;
        vmul = __uint_as_float(
                   static_cast<uint32_t>(rec[r.v_norm] | rec[r.v_norm + 1] << 8)
                   << 16) *
               inv_sd;
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
    lds.loop.voff[wave][lane] = voff;
    lds.loop.vmul[wave][lane] = vmul;
    lds.loop.vfmt[wave][lane] = fmt;
    wave_sync();
    const int32_t keys = min(kLanes, visible - c0);
    for (int j = 0; j < keys; ++j) {
      const int64_t vo = lds.loop.voff[wave][j];
      const float vm = lds.loop.vmul[wave][j];
      const int32_t vf = lds.loop.vfmt[wave][j];
      float v[kDpl];
      if (vf == kBf16) {
        const uint2 w = *reinterpret_cast<const uint2 *>(pool + vo + lane * 8);
        v[0] = __uint_as_float(w.x << 16);
        v[1] = __uint_as_float(w.x & 0xffff0000u);
        v[2] = __uint_as_float(w.y << 16);
        v[3] = __uint_as_float(w.y & 0xffff0000u);
      } else if (vf == kFp8) {
        const uint8_t *b = pool + vo + lane * 4;
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = fp8_dequant(b[i], vm);
      } else if (vf == kTq4) {
        const uint32_t c =
            *reinterpret_cast<const uint16_t *>(pool + vo + lane * 2);
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = lds.cb[3][(c >> (4 * i)) & 15] * vm;
      } else {
        const uint32_t c = pool[vo + lane];
#pragma unroll
        for (int i = 0; i < kDpl; ++i)
          v[i] = lds.cb[1][(c >> (2 * i)) & 3] * vm;
      }
      const bool rot = vf >= kTq4;
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
    for (int i = 0; i < kDpl; ++i) {
      lds.o[0][wave][g][lane * kDpl + i] = accp[g][i];
      lds.o[1][wave][g][lane * kDpl + i] = accr[g][i];
    }
  }
  __syncthreads();
  for (int e = tid; e < G * kD; e += kThreads) {
    const int g = e / kD, d = e % kD;
    float mx = kNegInf;
    for (int w = 0; w < kWaves; ++w)
      mx = fmaxf(mx, lds.max[w][g]);
    float op = 0.0f, orr = 0.0f, tot = 0.0f;
    for (int w = 0; w < kWaves; ++w) {
      if (lds.max[w][g] == kNegInf)
        continue;
      const float f = expf(lds.max[w][g] - mx);
      op += lds.o[0][w][g][d] * f;
      orr += lds.o[1][w][g][d] * f;
      tot += lds.sum[w][g] * f;
    }
    lds.oplain[g][d] = op;
    lds.orot[g][d] = orr;
    if (d == 0)
      lds.total[g] = tot;
  }
  __syncthreads();
  if (has_tq) {
    for (int h = 1; h < kD; h *= 2) {
      for (int p = tid; p < G * (kD / 2); p += kThreads) {
        const int g = p / (kD / 2), j = p % (kD / 2);
        const int i0 = (j / h) * 2 * h + j % h;
        const float a = lds.orot[g][i0], b = lds.orot[g][i0 + h];
        lds.orot[g][i0] = a + b;
        lds.orot[g][i0 + h] = a - b;
      }
      __syncthreads();
    }
  }
  for (int e = tid; e < group * kD; e += kThreads) {
    const int g = e / kD, d = e % kD;
    float o = lds.oplain[g][d];
    if (has_tq)
      o += lds.orot[g][d] * inv_sd * lds.vs[d];
    out[row * q_stride + (kvh * group + g) * kD + d] =
        f32_to_bf16(o / lds.total[g]);
  }
}

hipError_t launch(int group, dim3 grid, const void *q, const uint8_t *pool,
                  void *out, const int32_t *row_seq, const int32_t *table,
                  const uint8_t *formats, int32_t base_fmt,
                  const int32_t *q_indptr, const int32_t *kv_lens,
                  int32_t max_blocks, int64_t slot_bytes, int32_t kv_heads,
                  int32_t has_tq, const float *tables, const float *codebooks,
                  float scale, float k_scale, float v_scale, int32_t causal) {
  auto go = [&](auto kernel) {
    hipLaunchKernelGGL(
        kernel, grid, dim3(kThreads), 0, nullptr, static_cast<const Bf16 *>(q),
        pool, static_cast<Bf16 *>(out), row_seq, table, formats, base_fmt,
        q_indptr, kv_lens, max_blocks, slot_bytes, kv_heads, group, has_tq,
        tables, codebooks, scale, k_scale, v_scale, causal);
  };
  if (group <= 1)
    go(mixed_attn_kernel<1>);
  else if (group <= 2)
    go(mixed_attn_kernel<2>);
  else if (group <= 3)
    go(mixed_attn_kernel<3>);
  else if (group <= 4)
    go(mixed_attn_kernel<4>);
  else
    return hipErrorInvalidValue;
  return hipGetLastError();
}

} // namespace own

namespace {

struct Model {
  const char *name;
  int hq, hkv;
};

struct Shape {
  std::string name;
  std::vector<int> q_lens, kv_lens;
  bool decode;
};

// Host data of one case: per (sequence, token, KV head) K and V values, then
// pages in one format (or the mixed pattern), the block table and the
// TurboQuant tables.
struct Case {
  Model model;
  Shape shape;
  int fmt; // -1: mixed (block format = (sequence + block) % 4)
  int num_seqs = 0, total_q = 0, max_q = 0, max_kv = 0, max_blocks = 0,
      num_blocks = 0;
  size_t slot = 0;
  float k_scale = 0.05f, v_scale = 0.04f;
  std::vector<int32_t> q_indptr, kv_lens, table, row_seq;
  std::vector<uint8_t> formats; // per table entry
  std::vector<float> tables;    // [hkv][kTableElems]
  std::vector<uint8_t> pool;    // num_blocks * slot
  std::vector<uint16_t> q, k_new, v_new;
};

HeadTables head(const Case &c, int g) {
  const float *t = c.tables.data() + static_cast<size_t>(g) * kTableElems;
  return {t, t + kD, t + 2 * kD};
}

Case build(const Model &m, const Shape &s, int fmt, std::mt19937 &rng) {
  (void)to_e4m3(0.0f); // builds its table before the threads use it
  Case c{m, s, fmt};
  c.num_seqs = static_cast<int>(s.q_lens.size());
  c.q_indptr.push_back(0);
  for (int i = 0; i < c.num_seqs; ++i) {
    c.q_indptr.push_back(c.q_indptr.back() + s.q_lens[i]);
    c.kv_lens.push_back(s.kv_lens[i]);
    c.max_q = std::max(c.max_q, s.q_lens[i]);
    c.max_kv = std::max(c.max_kv, s.kv_lens[i]);
    c.max_blocks = std::max(c.max_blocks,
                            (s.kv_lens[i] + kBlockTokens - 1) / kBlockTokens);
    for (int r = 0; r < s.q_lens[i]; ++r)
      c.row_seq.push_back(i);
  }
  c.total_q = c.q_indptr.back();
  for (int i = 0; i < c.num_seqs; ++i)
    c.num_blocks += (s.kv_lens[i] + kBlockTokens - 1) / kBlockTokens;
  std::vector<int32_t> perm(c.num_blocks);
  for (int i = 0; i < c.num_blocks; ++i)
    perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  c.table.assign(static_cast<size_t>(c.num_seqs) * c.max_blocks, 0);
  c.formats.assign(c.table.size(), 0);
  std::vector<uint8_t> block_fmt(c.num_blocks, 0);
  int next = 0;
  for (int i = 0; i < c.num_seqs; ++i)
    for (int b = 0; b * kBlockTokens < s.kv_lens[i]; ++b) {
      const size_t e = static_cast<size_t>(i) * c.max_blocks + b;
      c.table[e] = perm[next++];
      c.formats[e] = static_cast<uint8_t>(fmt < 0 ? (i + b) % 4 : fmt);
      block_fmt[c.table[e]] = c.formats[e];
    }
  c.slot = page_bytes(fmt < 0 ? kBf16 : fmt, m.hkv);
  // Tables: random signs and a Gaussian S.
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::bernoulli_distribution coin(0.5);
  c.tables.resize(static_cast<size_t>(m.hkv) * kTableElems);
  for (int g = 0; g < m.hkv; ++g) {
    float *t = c.tables.data() + static_cast<size_t>(g) * kTableElems;
    for (int i = 0; i < 2 * kD; ++i)
      t[i] = coin(rng) ? 1.0f : -1.0f;
    for (int i = 0; i < kD * kD; ++i)
      t[2 * kD + i] = gauss(rng);
  }
  // KV-like values: Gaussian with a few outlier channels per head.
  std::vector<float> chan(static_cast<size_t>(m.hkv) * kD);
  for (auto &v : chan)
    v = (rng() % 32 == 0) ? 6.0f : 1.0f;
  c.pool.assign(static_cast<size_t>(c.num_blocks) * c.slot, 0);
  c.k_new.resize(static_cast<size_t>(c.total_q) * m.hkv * kD);
  c.v_new.resize(c.k_new.size());
  for (int i = 0; i < c.num_seqs; ++i) {
    const int kv = s.kv_lens[i];
    const int first_new = kv - s.q_lens[i];
    std::vector<float> kvv(static_cast<size_t>(kv) * m.hkv * 2 * kD);
    for (size_t e = 0; e < kvv.size(); ++e)
      kvv[e] = from_bf16(
          to_bf16(gauss(rng) * chan[(e / kD / 2) % m.hkv * kD + e % kD]));
    parallel_for(static_cast<int64_t>(kv) * m.hkv, [&](int64_t th) {
      const int t = static_cast<int>(th / m.hkv),
                g = static_cast<int>(th % m.hkv);
      const float *k = &kvv[(static_cast<size_t>(t) * m.hkv + g) * 2 * kD];
      const float *v = k + kD;
      const size_t e = static_cast<size_t>(i) * c.max_blocks + t / kBlockTokens;
      const int f = c.formats[e];
      uint8_t *page = c.pool.data() + static_cast<size_t>(c.table[e]) * c.slot;
      const int tt = t % kBlockTokens;
      float kq[kD], vq[kD]; // the values the page holds (for k_new / v_new)
      if (f == kBf16 || f == kFp8) {
        for (int half = 0; half < 2; ++half) {
          const float *x = half ? v : k;
          float *xq = half ? vq : kq;
          const size_t base =
              ((static_cast<size_t>(half) * kBlockTokens + tt) * m.hkv + g) *
              kD;
          const float sc = half ? c.v_scale : c.k_scale;
          for (int d = 0; d < kD; ++d) {
            if (f == kBf16) {
              const uint16_t b = to_bf16(x[d]);
              wr16(page + 2 * (base + d), b);
              xq[d] = from_bf16(b);
            } else {
              const uint8_t b = to_e4m3(x[d] / sc);
              page[base + d] = b;
              xq[d] = from_bf16(to_bf16(e4m3_value(b) * sc));
            }
          }
        }
      } else {
        const Rec r = f == kTq4 ? kRecTq4 : kRecTq2;
        uint8_t *rec =
            page + (static_cast<size_t>(g) * kBlockTokens + tt) * r.bytes;
        tq_encode(f, k, v, head(c, g), rec);
        tq_decode(f, rec, head(c, g), kq, vq);
      }
      if (t >= first_new) {
        const size_t row = c.q_indptr[i] + (t - first_new);
        for (int d = 0; d < kD; ++d) {
          c.k_new[(row * m.hkv + g) * kD + d] = to_bf16(kq[d]);
          c.v_new[(row * m.hkv + g) * kD + d] = to_bf16(vq[d]);
        }
      }
    });
  }
  c.q.resize(static_cast<size_t>(c.total_q) * m.hq * kD);
  for (auto &v : c.q)
    v = to_bf16(gauss(rng));
  return c;
}

// The reference output rows (DecodeThenAttend, F64) of the checked rows over
// `pool` (the page bytes, read back after a call), [rows, hq, d].
std::vector<double> reference(const Case &c, const std::vector<uint8_t> &pool,
                              const std::vector<int> &rows, bool causal,
                              bool tq_bf16) {
  const Model &m = c.model;
  const int group = m.hq / m.hkv;
  // Decoded K/V per (sequence, KV head, token), F32.
  std::vector<std::vector<float>> kvs(static_cast<size_t>(c.num_seqs) * m.hkv);
  parallel_for(static_cast<int64_t>(c.num_seqs) * m.hkv, [&](int64_t sg) {
    const int s = static_cast<int>(sg / m.hkv),
              g = static_cast<int>(sg % m.hkv);
    const int kv = c.kv_lens[s];
    std::vector<float> &out = kvs[sg];
    out.resize(static_cast<size_t>(kv) * 2 * kD);
    for (int t = 0; t < kv; ++t) {
      const size_t e = static_cast<size_t>(s) * c.max_blocks + t / kBlockTokens;
      const int f = c.formats[e];
      const uint8_t *page =
          pool.data() + static_cast<size_t>(c.table[e]) * c.slot;
      const int tt = t % kBlockTokens;
      float *k = &out[static_cast<size_t>(t) * 2 * kD];
      float *v = k + kD;
      if (f == kBf16 || f == kFp8) {
        for (int half = 0; half < 2; ++half) {
          const size_t base =
              ((static_cast<size_t>(half) * kBlockTokens + tt) * m.hkv + g) *
              kD;
          for (int d = 0; d < kD; ++d)
            (half ? v : k)[d] =
                f == kBf16 ? from_bf16(rd16(page + 2 * (base + d)))
                           : from_bf16(to_bf16(e4m3_value(page[base + d]) *
                                               (half ? c.v_scale : c.k_scale)));
        }
      } else {
        const Rec r = f == kTq4 ? kRecTq4 : kRecTq2;
        tq_decode(f,
                  page + (static_cast<size_t>(g) * kBlockTokens + tt) * r.bytes,
                  head(c, g), k, v);
        if (tq_bf16)
          for (int d = 0; d < 2 * kD; ++d)
            k[d] = from_bf16(to_bf16(k[d]));
      }
    }
  });
  const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
  std::vector<double> ref(rows.size() * m.hq * kD);
  parallel_for(static_cast<int64_t>(rows.size()) * m.hq, [&](int64_t rh) {
    const int ri = static_cast<int>(rh / m.hq), h = static_cast<int>(rh % m.hq);
    const int row = rows[ri];
    const int s = c.row_seq[row];
    const int q_len = c.q_indptr[s + 1] - c.q_indptr[s];
    const int visible = causal
                            ? c.kv_lens[s] - q_len + (row - c.q_indptr[s]) + 1
                            : c.kv_lens[s];
    const std::vector<float> &kv =
        kvs[static_cast<size_t>(s) * m.hkv + h / group];
    const uint16_t *q = &c.q[(static_cast<size_t>(row) * m.hq + h) * kD];
    std::vector<double> sc(visible);
    double mx = -INFINITY;
    for (int t = 0; t < visible; ++t) {
      double dot = 0;
      for (int d = 0; d < kD; ++d)
        dot += static_cast<double>(from_bf16(q[d])) *
               kv[static_cast<size_t>(t) * 2 * kD + d];
      sc[t] = dot * scale;
      mx = std::max(mx, sc[t]);
    }
    double total = 0;
    for (double &x : sc) {
      x = std::exp(x - mx);
      total += x;
    }
    double *o = &ref[(static_cast<size_t>(ri) * m.hq + h) * kD];
    for (int t = 0; t < visible; ++t)
      for (int d = 0; d < kD; ++d)
        o[d] += sc[t] / total * kv[static_cast<size_t>(t) * 2 * kD + kD + d];
  });
  return ref;
}

struct Dev {
  std::vector<void *> pools, tables;
  int copies = 1;
  void *q = nullptr, *k_new = nullptr, *v_new = nullptr, *out = nullptr;
  int32_t *table = nullptr, *q_indptr = nullptr, *kv_lens = nullptr,
          *row_seq = nullptr;
  uint8_t *formats = nullptr;
  float *codebooks = nullptr;
  void *staged = nullptr; // BF16 pages of every block (staged_ck)
};

template <typename T> T *upload(const std::vector<T> &v) {
  void *p = nullptr;
  check_hip(hipMalloc(&p, std::max<size_t>(16, v.size() * sizeof(T))),
            "hipMalloc");
  if (!v.empty())
    check_hip(
        hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
        "upload");
  return static_cast<T *>(p);
}

Dev make_dev(const Case &c, bool staged) {
  Dev d;
  const size_t bytes = c.pool.size() + c.tables.size() * 4;
  d.copies =
      static_cast<int>(std::max<size_t>(2, (kRotateBytes + bytes - 1) / bytes));
  for (int i = 0; i < d.copies; ++i) {
    d.pools.push_back(upload(c.pool));
    d.tables.push_back(upload(c.tables));
  }
  d.q = upload(c.q);
  d.k_new = upload(c.k_new);
  d.v_new = upload(c.v_new);
  check_hip(hipMalloc(&d.out, c.q.size() * 2), "out");
  d.table = upload(c.table);
  d.q_indptr = upload(c.q_indptr);
  d.kv_lens = upload(c.kv_lens);
  d.row_seq = upload(c.row_seq);
  d.formats = upload(c.formats);
  std::vector<float> cb(64, 0.0f);
  for (int b = 0; b < 4; ++b)
    for (int i = 0; i < (2 << b); ++i)
      cb[b * 16 + i] = kCodebooks[b][i];
  d.codebooks = upload(cb);
  if (staged)
    check_hip(hipMalloc(&d.staged, static_cast<size_t>(c.num_blocks) *
                                       page_bytes(kBf16, c.model.hkv)),
              "staged");
  return d;
}

void free_dev(Dev &d) {
  for (void *p : d.pools)
    (void)hipFree(p);
  for (void *p : d.tables)
    (void)hipFree(p);
  for (void *p :
       {d.q, d.k_new, d.v_new, d.out, static_cast<void *>(d.table),
        static_cast<void *>(d.q_indptr), static_cast<void *>(d.kv_lens),
        static_cast<void *>(d.row_seq), static_cast<void *>(d.formats),
        static_cast<void *>(d.codebooks), d.staged})
    if (p)
      (void)hipFree(p);
}

turbine_attention_paged_desc paged_desc(const Case &c, const Dev &d, void *pool,
                                        int dtype) {
  turbine_attention_paged_desc p{};
  p.q = d.q;
  p.k_new = d.k_new;
  p.v_new = d.v_new;
  p.out = d.out;
  p.kv_layer = pool;
  p.block_table = d.table;
  p.q_indptr = d.q_indptr;
  p.kv_lens = d.kv_lens;
  p.num_seqs = c.num_seqs;
  p.total_q = c.total_q;
  p.max_q_len = c.max_q;
  p.max_kv_len = c.max_kv;
  p.max_blocks_per_seq = c.max_blocks;
  p.num_blocks = c.num_blocks;
  p.block_tokens = kBlockTokens;
  p.num_q_heads = c.model.hq;
  p.num_kv_heads = c.model.hkv;
  p.head_dim = kD;
  p.q_stride_token = static_cast<int64_t>(c.model.hq) * kD;
  p.new_stride_token = static_cast<int64_t>(c.model.hkv) * kD;
  p.out_stride_token = p.q_stride_token;
  p.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kD)));
  p.causal = 1;
  p.dtype = dtype;
  p.k_scale = dtype == TURBINE_DTYPE_F8E4M3 ? c.k_scale : 1.0f;
  p.v_scale = dtype == TURBINE_DTYPE_F8E4M3 ? c.v_scale : 1.0f;
  return p;
}

struct Cand {
  std::string name;
  std::function<bool(int)> run; // copy index; false on an ABI error
  bool on_ctx;                  // runs on the context stream
  double kv_bytes;              // bytes of KV the call must read
  bool staged = false;          // reads TurboQuant values decoded to BF16
};

struct Opts {
  int iters = 20, rounds = 5;
  bool check_only = false;
};

int g_failures = 0;

void run_case(turbine_ctx *ctx, const Case &c, const Opts &o, bool prefill_op) {
  const Model &m = c.model;
  const bool staged_ok = c.fmt == kTq4 || c.fmt == kTq2;
  Dev d = make_dev(c, staged_ok);
  const int op = prefill_op ? TURBINE_OP_ATTENTION_PREFILL_PAGED
                            : TURBINE_OP_ATTENTION_DECODE_PAGED;
  std::vector<Cand> cands;
  // KV bytes the call reads (every visible key once per KV head).
  double kv_tokens = 0;
  for (int s = 0; s < c.num_seqs; ++s)
    kv_tokens += c.kv_lens[s];
  const double kv_bytes =
      c.fmt < 0 ? 0
                : kv_tokens * static_cast<double>(page_bytes(c.fmt, m.hkv)) /
                      kBlockTokens;
  if (c.fmt == kBf16 || c.fmt == kFp8) {
    const int dtype =
        c.fmt == kBf16 ? TURBINE_DTYPE_BF16 : TURBINE_DTYPE_F8E4M3;
    for (int i = 0; i < turbine_impl_count(op); ++i) {
      turbine_impl_entry e{};
      turbine_impl_info(op, i, &e);
      turbine_attention_paged_desc p = paged_desc(c, d, d.pools[0], dtype);
      if (turbine_impl_supports(op, i, &p) != 1)
        continue;
      cands.push_back({std::string(fmt_name(c.fmt)) + ":" + e.name,
                       [&, i, dtype](int copy) {
                         turbine_attention_paged_desc pd =
                             paged_desc(c, d, d.pools[copy % d.copies], dtype);
                         return ok_t(ctx, turbine_impl_run(ctx, op, i, &pd),
                                     "impl_run");
                       },
                       true, kv_bytes});
    }
  }
  if (staged_ok) {
    // The first CK BF16 implementation that takes the staged pages.
    turbine_attention_paged_desc probe =
        paged_desc(c, d, d.staged, TURBINE_DTYPE_BF16);
    int ck = -1;
    std::string ck_name;
    for (int i = 0; i < turbine_impl_count(op) && ck < 0; ++i) {
      turbine_impl_entry e{};
      turbine_impl_info(op, i, &e);
      if (std::string(e.provider) == "ck" &&
          turbine_impl_supports(op, i, &probe) == 1) {
        ck = i;
        ck_name = e.name;
      }
    }
    if (ck >= 0) {
      const size_t bf = page_bytes(kBf16, m.hkv);
      cands.push_back(
          {std::string(fmt_name(c.fmt)) + ":staged_ck(" + ck_name + ")",
           [&, ck, bf](int copy) {
             const int k = copy % d.copies;
             std::vector<void *> pages(c.num_blocks);
             for (int b = 0; b < c.num_blocks; ++b)
               pages[b] = static_cast<char *>(d.staged) + b * bf;
             const float *cbs = d.codebooks;
             turbine_tq_params tq{kSeed,
                                  {cbs, cbs + 16, cbs + 32, cbs + 48},
                                  static_cast<const float *>(d.tables[k])};
             turbine_kv_transcode_desc t{};
             t.pages = pages.data();
             t.coded = d.pools[k];
             t.coded_block_bytes = static_cast<int64_t>(c.slot);
             t.seed = kSeed;
             t.num_blocks = c.num_blocks;
             t.layers = 1;
             t.block_tokens = kBlockTokens;
             t.num_kv_heads = m.hkv;
             t.head_dim = kD;
             t.page_dtype = TURBINE_DTYPE_BF16;
             t.format = c.fmt == kTq4 ? TURBINE_KVFMT_TQ4 : TURBINE_KVFMT_TQ2;
             t.direction = TURBINE_KV_DECODE;
             t.tq_params = &tq;
             if (!ok_t(ctx, turbine_kv_transcode(ctx, &t), "kv_transcode"))
               return false;
             turbine_attention_paged_desc pd =
                 paged_desc(c, d, d.staged, TURBINE_DTYPE_BF16);
             return ok_t(ctx, turbine_impl_run(ctx, op, ck, &pd), "impl_run");
           },
           true, kv_bytes, true});
    }
  }
  {
    const bool has_tq = c.fmt != kBf16 && c.fmt != kFp8;
    cands.push_back(
        {std::string(c.fmt < 0 ? "mixed" : fmt_name(c.fmt)) + ":own_rot",
         [&, has_tq](int copy) {
           const int k = copy % d.copies;
           const hipError_t e = own::launch(
               m.hq / m.hkv, dim3(c.total_q, m.hkv), d.q,
               static_cast<const uint8_t *>(d.pools[k]), d.out, d.row_seq,
               d.table, c.fmt < 0 ? d.formats : nullptr, c.fmt < 0 ? 0 : c.fmt,
               d.q_indptr, d.kv_lens, c.max_blocks,
               static_cast<int64_t>(c.slot), m.hkv, has_tq ? 1 : 0,
               static_cast<const float *>(d.tables[k]), d.codebooks,
               static_cast<float>(1.0 / std::sqrt(128.0)), c.k_scale, c.v_scale,
               1);
           if (e != hipSuccess)
             std::fprintf(stderr, "own launch: %s\n", hipGetErrorString(e));
           return e == hipSuccess;
         },
         false, kv_bytes});
  }
  // Rows checked: every row of a decode; a sample of a prefill's.
  std::vector<int> rows;
  for (int r = 0; r < c.total_q; ++r)
    if (c.total_q <= 64 || r % 61 == 0 || r == c.total_q - 1)
      rows.push_back(r);
  const std::string label = std::string(m.name) + " " + c.shape.name + " " +
                            (c.fmt < 0 ? "mixed" : fmt_name(c.fmt));
  for (Cand &cd : cands) {
    auto sync = [&]() {
      if (cd.on_ctx)
        ok_t(ctx, turbine_stream_sync(ctx), "sync");
      check_hip(hipDeviceSynchronize(), "sync");
    };
    check_hip(hipMemset(d.out, 0, c.q.size() * 2), "clear out");
    if (!cd.run(0)) {
      std::printf("%-28s %-44s FAILED to run\n", label.c_str(),
                  cd.name.c_str());
      ++g_failures;
      continue;
    }
    sync();
    std::vector<uint16_t> got(c.q.size());
    check_hip(
        hipMemcpy(got.data(), d.out, got.size() * 2, hipMemcpyDeviceToHost),
        "read out");
    std::vector<uint8_t> pool(c.pool.size());
    check_hip(
        hipMemcpy(pool.data(), d.pools[0], pool.size(), hipMemcpyDeviceToHost),
        "read pool");
    // The staged candidate is judged against the reference over the
    // TurboQuant values rounded to BF16 (what the BF16 staging pages hold);
    // `exact_ratio` is its worst element against the unrounded reference.
    const std::vector<double> ref = reference(c, pool, rows, true, cd.staged);
    const std::vector<double> exact =
        cd.staged ? reference(c, pool, rows, true, false) : ref;
    double exact_ratio = 0;
    // Per element |got - ref| <= 4e-3 + 2^-7 |ref|: the BF16 output rounding
    // (2^-9 relative) and the P / V roundings of the CK and FP8 kernels, with
    // room; `ratio` is the worst element's share of its bound (pass <= 1).
    double max_err = 0, max_ref = 0, ratio = 0;
    for (size_t ri = 0; ri < rows.size(); ++ri)
      for (int e = 0; e < m.hq * kD; ++e) {
        const double r = ref[ri * m.hq * kD + e];
        const double g =
            from_bf16(got[static_cast<size_t>(rows[ri]) * m.hq * kD + e]);
        max_err = std::max(max_err, std::fabs(g - r));
        max_ref = std::max(max_ref, std::fabs(r));
        const double q = std::fabs(g - r) / (4e-3 + std::fabs(r) / 128);
        ratio = std::isfinite(q) ? std::max(ratio, q) : INFINITY;
        const double x = exact[ri * m.hq * kD + e];
        exact_ratio = std::max(exact_ratio,
                               std::fabs(g - x) / (4e-3 + std::fabs(x) / 128));
      }
    const bool pass = ratio <= 1.0;
    if (!pass)
      ++g_failures;
    double us = 0;
    if (!o.check_only) {
      for (int w = 0; w < 3; ++w)
        cd.run(w);
      sync();
      const int iters = prefill_op ? std::max(2, o.iters / 4) : o.iters;
      std::vector<double> per;
      for (int r = 0; r < o.rounds; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
          cd.run(i + 1);
        sync();
        const auto t1 = std::chrono::steady_clock::now();
        per.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
      }
      us = median(per);
    }
    std::printf(
        "%-28s %-44s max|d| %.2e (|ref| %.2f, bound x%.2f) %s  %10.1f us  "
        "%7.1f GB/s%s\n",
        label.c_str(), cd.name.c_str(), max_err, max_ref, ratio,
        pass ? "ok  " : "FAIL", us,
        us > 0 && cd.kv_bytes > 0 ? cd.kv_bytes / us / 1e3 : 0.0,
        cd.staged
            ? ("  (unrounded: x" + std::to_string(exact_ratio) + ")").c_str()
            : "");
    std::fflush(stdout);
  }
  free_dev(d);
}

} // namespace

int main(int argc, char **argv) {
  std::string models = "llama,olmoe", phases = "decode,prefill,mixed";
  Opts o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--models")
      models = argv[i + 1];
    else if (a == "--phases")
      phases = argv[i + 1];
    else if (a == "--iters")
      o.iters = std::atoi(argv[i + 1]);
    else if (a == "--rounds")
      o.rounds = std::atoi(argv[i + 1]);
    else if (a == "--check-only")
      o.check_only = std::atoi(argv[i + 1]) != 0;
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  turbine_ctx *ctx = nullptr;
  if (turbine_ctx_create(0, &ctx) != TURBINE_OK) {
    std::fprintf(stderr, "turbine_ctx_create failed\n");
    return 2;
  }
  hipDeviceProp_t prop{};
  check_hip(hipGetDeviceProperties(&prop, 0), "props");
  std::printf("device %s (%s), abi minor %d\n", prop.name, prop.gcnArchName,
              static_cast<int>(turbine_abi_minor()));
  const Model kModels[] = {{"llama", 24, 8}, {"olmoe", 16, 16}};
  auto rep = [](int n, int v) { return std::vector<int>(n, v); };
  const std::vector<Shape> decode = {
      {"decode b1@768", {1}, {768}, true},
      {"decode b16@768", rep(16, 1), rep(16, 768), true},
      {"decode b16@2k", rep(16, 1), rep(16, 2048), true},
  };
  const std::vector<Shape> prefill = {
      {"prefill 16x512 @1024", rep(16, 512), rep(16, 1536), false},
      {"prefill 1x2048", {2048}, {2048}, false},
  };
  const Shape mixed_decode = {
      "mixed decode b3", {1, 1, 1}, {300, 768, 129}, true};
  const Shape mixed_prefill = {"mixed prefill 2", {64, 7}, {512, 1000}, false};
  std::mt19937 rng(1234);
  for (const Model &m : kModels) {
    if (!listed(models, m.name))
      continue;
    if (listed(phases, "mixed")) {
      run_case(ctx, build(m, mixed_decode, -1, rng), o, false);
      run_case(ctx, build(m, mixed_prefill, -1, rng), o, true);
    }
    if (listed(phases, "decode"))
      for (const Shape &s : decode)
        for (int f : {kBf16, kFp8, kTq4, kTq2})
          run_case(ctx, build(m, s, f, rng), o, false);
    if (listed(phases, "prefill"))
      for (const Shape &s : prefill)
        for (int f : {kBf16, kFp8, kTq4, kTq2})
          run_case(ctx, build(m, s, f, rng), o, true);
  }
  turbine_ctx_destroy(ctx);
  std::printf("attn_eval: %s (%d failures)\n", g_failures ? "FAIL" : "ok",
              g_failures);
  return g_failures ? 1 : 0;
}
