// turbine_qgemm_mxfp4_eval: the MXFP4 GEMM provider evaluation (Phase 6a
// Task 19, kernel reuse rule) on the current device. For the Llama-3.2-3B and
// Llama-3.1-8B linear shapes at the requested m it runs
//
//   bf16            hipBLASLt BF16 x BF16 -> BF16 on the dequantized weight
//                   (the baseline: what the BF16 checkpoint runs; first
//                   heuristic answer and best of the first 8)
//   dequant+bf16    a dequantize-to-BF16 kernel over the whole weight, then
//                   the same hipBLASLt GEMM (the "reuse hipBLASLt" candidate
//                   for prefill; needs an n x k BF16 scratch)
//   mxfp4_small / _medium / _large
//                   turbine_hip_mxfp4 (src/qgemm_mxfp4.hip) with each tile
//
// checks every MXFP4 candidate against a host reference of the CPU provider's
// semantics (crates/turbine-kernels/src/cpu/qgemm.rs: E2M1 x 2^(E8M0 - 127)
// weights, BF16 activations, exact products summed, one rounding to BF16) on
// sampled rows, and times it (median of --rounds rounds of --iters calls,
// rotating over enough weight copies to defeat the 64 MiB infinity cache).
// It also checks turbine_hip_mxfp4's MXFP4_EMULATED quantize-dequantize
// (src/quantize_act_mxfp4.hip) bit-exactly against a host copy of
// cpu::quant's rules and times it.
//
//   turbine_qgemm_mxfp4_eval [--ms 1,4,16,128,2048] [--iters 20] [--rounds 5]
//                            [--shapes 3b.qkv,...]
//
// A lab tool (run it on GPU 0 under scripts/bench-lock.sh); not loaded by the
// server. Exit 0 when every MXFP4 candidate matched the reference and the
// quantize-dequantize was bit-exact, 1 otherwise.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gemm_problem.hpp"
#include "qgemm_mxfp4.hpp"

namespace {

using turbine_hip::GemmProblem;
using turbine_hip::GemmShape;
using turbine_hip::Mxfp4Tile;

constexpr size_t kWorkspaceBytes = 32u << 20;
constexpr int kMaxAlgos = 8;
constexpr size_t kRotateBytes = 512u << 20;

void hip_ok(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "qgemm_mxfp4_eval: %s: %s\n", what,
                 hipGetErrorName(e));
    std::exit(2);
  }
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double unit() {
    return (static_cast<double>(next() >> 11) + 0.5) / 9007199254740992.0;
  }
  float normal(float scale) {
    const double u1 = unit(), u2 = unit();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                              std::cos(6.283185307179586 * u2)) *
           scale;
  }
};

uint16_t bf16_bits(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    return static_cast<uint16_t>((b >> 16) | 0x40u);
  const uint32_t lsb = (b >> 16) & 1;
  return static_cast<uint16_t>((b + 0x7fff + lsb) >> 16);
}

float bf16_value(uint16_t h) {
  const uint32_t b = static_cast<uint32_t>(h) << 16;
  float x;
  std::memcpy(&x, &b, 4);
  return x;
}

// ---- host copies of cpu::quant's rules ----
const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float e2m1_value(uint8_t code) {
  const float v = kE2m1[code & 7];
  return (code & 8) ? -v : v;
}

float e8m0_value(uint8_t e) {
  if (e == 255)
    return std::nanf("");
  return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

uint8_t e2m1_round(float x) {
  if (std::isnan(x))
    return 0;
  const uint8_t sign = std::signbit(x) ? 8 : 0;
  const float a = std::min(std::fabs(x), 6.0f);
  uint8_t best = 0;
  for (uint8_t c = 1; c < 8; ++c) {
    const float d = std::fabs(kE2m1[c] - a);
    const float bd = std::fabs(kE2m1[best] - a);
    if (d < bd || (d == bd && (c & 1) == 0))
      best = c;
  }
  return best == 0 ? 0 : static_cast<uint8_t>(sign | best);
}

uint8_t scale_even(float amax) {
  if (std::isnan(amax))
    return 255;
  uint32_t b;
  std::memcpy(&b, &amax, 4);
  const uint32_t r = (b + (1u << 21)) & 0xff800000u;
  const int field = static_cast<int>((r >> 23) & 0xff);
  const int log2 = field == 255 ? 32767 : field == 0 ? -127 : field - 127;
  const int e = std::clamp(log2 - 2, -127, 127);
  return static_cast<uint8_t>(e + 127);
}

template <typename T> T *device_copy(const std::vector<T> &v) {
  void *p = nullptr;
  hip_ok(hipMalloc(&p, v.size() * sizeof(T)), "hipMalloc");
  hip_ok(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
         "hipMemcpy h2d");
  return static_cast<T *>(p);
}

double median(std::vector<double> v) {
  if (v.empty())
    return std::nan("");
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct LinearShape {
  const char *name;
  int64_t n, k;
};

const LinearShape kShapes[] = {
    {"3b.qkv", 5120, 3072},      {"3b.o", 3072, 3072},
    {"3b.gate_up", 16384, 3072}, {"3b.down", 3072, 8192},
    {"8b.qkv", 6144, 4096},      {"8b.o", 4096, 4096},
    {"8b.gate_up", 28672, 4096}, {"8b.down", 4096, 14336},
};

// Dequantizes b (n x k/2 codes, n x k/32 exponents) into BF16 [n, k].
__global__ void dequant_kernel(const uint8_t *b, const uint8_t *s,
                               uint16_t *out, int64_t n, int64_t k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs = n * k / 2;
  if (i >= pairs)
    return;
  const int64_t row = (2 * i) / k;
  const int64_t col = (2 * i) % k;
  const uint8_t byte = b[i];
  const uint8_t e = s[row * (k / 32) + col / 32];
  const float sc = e == 0 ? __uint_as_float(0x00400000u)
                          : __uint_as_float(static_cast<uint32_t>(e) << 23);
  const float kv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const uint8_t lo = byte & 15, hi = byte >> 4;
  const float v0 = ((lo & 8) ? -kv[lo & 7] : kv[lo & 7]) * sc;
  const float v1 = ((hi & 8) ? -kv[hi & 7] : kv[hi & 7]) * sc;
  out[2 * i] = static_cast<uint16_t>(__float_as_uint(v0) >> 16);
  out[2 * i + 1] = static_cast<uint16_t>(__float_as_uint(v1) >> 16);
}

struct Env {
  hipStream_t stream;
  hipblasLtHandle_t lt;
  void *workspace;
  hipEvent_t e0, e1;
  int iters, rounds;
};

struct ShapeData {
  int64_t n, k, m_max;
  std::vector<uint8_t> codes; // [n, k/2]
  std::vector<uint8_t> exps;  // [n, k/32]
  std::vector<uint16_t> a;    // [m_max, k] BF16
  std::vector<void *> w, s, w_bf16;
  uint16_t *a_dev = nullptr;
  uint16_t *scratch = nullptr; // [n, k] BF16 for dequant+bf16
  void *c = nullptr;
};

// Random weights of unit-variance outputs: codes uniform, exponents around
// 2^-5 (values up to 6 x 2^-4), activations N(0, 1) in BF16.
ShapeData make_shape(const LinearShape &sh, int64_t m_max, Rng &rng) {
  ShapeData d;
  d.n = sh.n;
  d.k = sh.k;
  d.m_max = m_max;
  d.codes.resize(static_cast<size_t>(d.n * d.k / 2));
  for (auto &b : d.codes)
    b = static_cast<uint8_t>(rng.next() & 0xff);
  d.exps.resize(static_cast<size_t>(d.n * d.k / 32));
  const int base = 127 - 3 - static_cast<int>(std::log2(std::sqrt(d.k)));
  for (auto &e : d.exps)
    e = static_cast<uint8_t>(base + static_cast<int>(rng.next() % 3));
  d.a.resize(static_cast<size_t>(m_max * d.k));
  for (auto &v : d.a)
    v = bf16_bits(rng.normal(1.0f));
  const size_t wbytes = d.codes.size() + d.exps.size();
  const size_t copies =
      std::clamp<size_t>(kRotateBytes / std::max<size_t>(wbytes, 1), 2, 8);
  for (size_t i = 0; i < copies; ++i) {
    d.w.push_back(device_copy(d.codes));
    d.s.push_back(device_copy(d.exps));
  }
  d.a_dev = device_copy(d.a);
  // BF16 baseline weights: the dequantized twin, as few copies as fit.
  std::vector<uint16_t> deq(static_cast<size_t>(d.n * d.k));
  for (int64_t r = 0; r < d.n; ++r) {
    for (int64_t c = 0; c < d.k; ++c) {
      const uint8_t byte = d.codes[(r * d.k + c) / 2];
      const uint8_t code = (c % 2 == 0) ? (byte & 15) : (byte >> 4);
      const float v =
          e2m1_value(code) * e8m0_value(d.exps[r * (d.k / 32) + c / 32]);
      deq[r * d.k + c] = bf16_bits(v);
    }
  }
  const size_t bcopies =
      std::clamp<size_t>(kRotateBytes / (deq.size() * 2), 2, 8);
  for (size_t i = 0; i < bcopies; ++i)
    d.w_bf16.push_back(device_copy(deq));
  hip_ok(hipMalloc(reinterpret_cast<void **>(&d.scratch), deq.size() * 2),
         "hipMalloc scratch");
  hip_ok(hipMalloc(&d.c, static_cast<size_t>(m_max * d.n) * 4), "hipMalloc c");
  return d;
}

void free_shape(ShapeData &d) {
  for (void *p : d.w)
    (void)hipFree(p);
  for (void *p : d.s)
    (void)hipFree(p);
  for (void *p : d.w_bf16)
    (void)hipFree(p);
  (void)hipFree(d.a_dev);
  (void)hipFree(d.scratch);
  (void)hipFree(d.c);
}

enum class Cand { Bf16First, Bf16Best, DequantBf16, Small, Medium, Large };

const char *cand_name(Cand c) {
  switch (c) {
  case Cand::Bf16First:
    return "bf16";
  case Cand::Bf16Best:
    return "bf16_best8";
  case Cand::DequantBf16:
    return "dequant+bf16";
  case Cand::Small:
    return "mxfp4_small";
  case Cand::Medium:
    return "mxfp4_medium";
  default:
    return "mxfp4_large";
  }
}

struct Blas {
  GemmProblem prob;
  std::vector<hipblasLtMatmulAlgo_t> algos;
  bool ok = false;
};

void prepare_blas(const Env &env, const ShapeData &d, int64_t m, Blas &b) {
  const GemmShape s{m, d.n, d.k, d.k, d.k, d.n, 1, TURBINE_DTYPE_BF16};
  if (b.prob.make(s) != HIPBLAS_STATUS_SUCCESS)
    return;
  turbine_hip::Preference pref;
  (void)hipblasLtMatmulPreferenceCreate(&pref.handle);
  const uint64_t ws = kWorkspaceBytes;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      pref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  hipblasLtMatmulHeuristicResult_t res[kMaxAlgos];
  int returned = 0;
  if (hipblasLtMatmulAlgoGetHeuristic(
          env.lt, b.prob.desc.handle, b.prob.weight.handle, b.prob.act.handle,
          b.prob.out.handle, b.prob.out.handle, pref.handle, kMaxAlgos, res,
          &returned) != HIPBLAS_STATUS_SUCCESS) {
    return;
  }
  for (int i = 0; i < returned; ++i)
    if (res[i].state == HIPBLAS_STATUS_SUCCESS)
      b.algos.push_back(res[i].algo);
  b.ok = !b.algos.empty();
}

turbine_qgemm_desc mx_desc(const ShapeData &d, int64_t m, size_t w) {
  turbine_qgemm_desc q{};
  q.a = d.a_dev;
  q.b = d.w[w];
  q.b_scales = d.s[w];
  q.c = d.c;
  q.m = m;
  q.n = d.n;
  q.k = d.k;
  q.lda = d.k;
  q.ldc = d.n;
  q.scheme = TURBINE_QSCHEME_MXFP4;
  q.act_quant = TURBINE_ACTQ_NONE;
  q.a_dtype = TURBINE_DTYPE_BF16;
  q.c_dtype = TURBINE_DTYPE_BF16;
  q.group_size = 32;
  q.alpha = 1.0f;
  return q;
}

bool run(const Env &env, Cand cand, ShapeData &d, Blas &blas, int64_t m,
         size_t i, const hipblasLtMatmulAlgo_t *algo) {
  const float alpha = 1.0f, beta = 0.0f;
  switch (cand) {
  case Cand::Bf16First:
  case Cand::Bf16Best: {
    const size_t w = i % d.w_bf16.size();
    return hipblasLtMatmul(env.lt, blas.prob.desc.handle, &alpha, d.w_bf16[w],
                           blas.prob.weight.handle, d.a_dev,
                           blas.prob.act.handle, &beta, d.c,
                           blas.prob.out.handle, d.c, blas.prob.out.handle,
                           algo, env.workspace, kWorkspaceBytes,
                           env.stream) == HIPBLAS_STATUS_SUCCESS;
  }
  case Cand::DequantBf16: {
    const size_t w = i % d.w.size();
    const int64_t pairs = d.n * d.k / 2;
    hipLaunchKernelGGL(dequant_kernel, dim3((pairs + 255) / 256), dim3(256), 0,
                       env.stream, static_cast<const uint8_t *>(d.w[w]),
                       static_cast<const uint8_t *>(d.s[w]), d.scratch, d.n,
                       d.k);
    return hipblasLtMatmul(env.lt, blas.prob.desc.handle, &alpha, d.scratch,
                           blas.prob.weight.handle, d.a_dev,
                           blas.prob.act.handle, &beta, d.c,
                           blas.prob.out.handle, d.c, blas.prob.out.handle,
                           algo, env.workspace, kWorkspaceBytes,
                           env.stream) == HIPBLAS_STATUS_SUCCESS;
  }
  default: {
    const turbine_qgemm_desc q = mx_desc(d, m, i % d.w.size());
    const Mxfp4Tile tile = cand == Cand::Small    ? Mxfp4Tile::Small
                           : cand == Cand::Medium ? Mxfp4Tile::Medium
                                                  : Mxfp4Tile::Large;
    return turbine_hip::launch_qgemm_mxfp4(env.stream, &q, tile) == hipSuccess;
  }
  }
}

double time_cand(const Env &env, Cand cand, ShapeData &d, Blas &blas, int64_t m,
                 const hipblasLtMatmulAlgo_t *algo) {
  for (int i = 0; i < 3; ++i)
    if (!run(env, cand, d, blas, m, i, algo))
      return std::nan("");
  std::vector<double> rounds;
  for (int r = 0; r < env.rounds; ++r) {
    hip_ok(hipEventRecord(env.e0, env.stream), "hipEventRecord");
    for (int i = 0; i < env.iters; ++i)
      (void)run(env, cand, d, blas, m, i, algo);
    hip_ok(hipEventRecord(env.e1, env.stream), "hipEventRecord");
    hip_ok(hipEventSynchronize(env.e1), "hipEventSynchronize");
    float ms = 0;
    hip_ok(hipEventElapsedTime(&ms, env.e0, env.e1), "hipEventElapsedTime");
    rounds.push_back(1000.0 * ms / env.iters);
  }
  return median(rounds);
}

std::vector<int64_t> sample_rows(int64_t m) {
  std::vector<int64_t> rows{0, m / 2, m - 1, m / 3, (2 * m) / 3, m / 7};
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

struct Check {
  double max_abs = 0, ref_max = 0;
  bool ok = true;
};

// A fresh run on weight copy 0 (the reference's weights), its sampled rows vs
// the exact sum of the dequantized products: within one BF16 rounding (2^-8
// relative) plus 1e-6 of the scale.
Check check(const Env &env, Cand cand, ShapeData &d, Blas &blas, int64_t m,
            const hipblasLtMatmulAlgo_t *algo) {
  Check out;
  hip_ok(
      hipMemsetAsync(d.c, 0xff, static_cast<size_t>(m * d.n) * 2, env.stream),
      "hipMemset");
  if (!run(env, cand, d, blas, m, 0, algo)) {
    out.ok = false;
    return out;
  }
  hip_ok(hipStreamSynchronize(env.stream), "sync");
  std::vector<uint16_t> got(static_cast<size_t>(m * d.n));
  hip_ok(hipMemcpy(got.data(), d.c, got.size() * 2, hipMemcpyDeviceToHost),
         "hipMemcpy d2h");
  std::vector<double> wrow(static_cast<size_t>(d.k));
  const auto rows = sample_rows(m);
  for (int64_t j = 0; j < d.n; ++j) {
    for (int64_t c = 0; c < d.k; ++c) {
      const uint8_t byte = d.codes[(j * d.k + c) / 2];
      const uint8_t code = (c % 2 == 0) ? (byte & 15) : (byte >> 4);
      wrow[c] = static_cast<double>(e2m1_value(code)) *
                e8m0_value(d.exps[j * (d.k / 32) + c / 32]);
    }
    for (int64_t r : rows) {
      double acc = 0;
      const uint16_t *a = &d.a[r * d.k];
      for (int64_t c = 0; c < d.k; ++c)
        acc += bf16_value(a[c]) * wrow[c];
      const double g = bf16_value(got[r * d.n + j]);
      const double err = std::fabs(g - acc);
      out.max_abs = std::max(out.max_abs, err);
      out.ref_max = std::max(out.ref_max, std::fabs(acc));
      if (!(err <=
            std::fabs(acc) * (1.0 / 256.0) + 1e-6 * (1 + std::fabs(acc))))
        out.ok = false;
    }
  }
  return out;
}

// ---- activation quantize-dequantize ----
bool eval_qdq(const Env &env, int64_t rows, int64_t cols, Rng &rng) {
  std::vector<float> x(static_cast<size_t>(rows * cols));
  for (auto &v : x)
    v = bf16_value(bf16_bits(rng.normal(1.0f)));
  if (rows >= 3) {
    std::fill(x.begin(), x.begin() + cols, 0.0f); // all-zero groups
    for (int64_t c = 0; c < cols; ++c) {
      // Row 1: exact E2M1 grid points and midpoints at a scale of 2^-3, with
      // a group max of 6 x 2^-3 or 7 x 2^-3 (the `even` round-up).
      const float g = kE2m1[c % 8] * 0.125f;
      const float h = kE2m1[(c % 7) + 1] * 0.125f;
      x[cols + c] = (c % 32 == 0) ? ((c / 32) % 2 ? 0.875f : 0.75f)
                                  : (c % 3 == 0 ? (g + h) / 2 : g);
      if (c % 2)
        x[cols + c] = -x[cols + c];
      x[2 * cols + c] = x[2 * cols + c] * 1e30f; // huge and saturating
    }
  }
  std::vector<uint16_t> xb(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    xb[i] = bf16_bits(x[i]);
  const int64_t gpr = (cols + 31) / 32;
  // Host reference (cpu::quant::quantize_dequantize_activations).
  std::vector<uint16_t> want(x.size());
  std::vector<float> want_s(static_cast<size_t>(rows * gpr));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t g = 0; g < gpr; ++g) {
      float amax = 0;
      const int64_t c0 = g * 32, c1 = std::min(cols, c0 + 32);
      for (int64_t c = c0; c < c1; ++c)
        amax = std::fmax(amax, std::fabs(bf16_value(xb[r * cols + c])));
      const float s = e8m0_value(scale_even(amax));
      for (int64_t c = c0; c < c1; ++c) {
        const float v = bf16_value(xb[r * cols + c]);
        want[r * cols + c] = bf16_bits(e2m1_value(e2m1_round(v / s)) * s);
      }
      want_s[r * gpr + g] = s;
    }
  }
  uint16_t *xd = device_copy(xb);
  uint16_t *od = nullptr;
  float *sd = nullptr;
  hip_ok(hipMalloc(reinterpret_cast<void **>(&od), xb.size() * 2), "hipMalloc");
  hip_ok(hipMalloc(reinterpret_cast<void **>(&sd), want_s.size() * 4),
         "hipMalloc");
  turbine_quantize_act_desc q{};
  q.x = xd;
  q.out = od;
  q.scales = sd;
  q.rows = rows;
  q.cols = cols;
  q.x_stride_row = cols;
  q.out_stride_row = cols;
  q.mode = TURBINE_ACTQ_MXFP4_EMULATED;
  q.static_scale = 1.0f;
  q.x_dtype = TURBINE_DTYPE_BF16;
  q.out_dtype = TURBINE_DTYPE_BF16;
  hip_ok(turbine_hip::launch_quantize_mxfp4(env.stream, &q), "qdq launch");
  hip_ok(hipStreamSynchronize(env.stream), "sync");
  std::vector<uint16_t> got(xb.size());
  std::vector<float> got_s(want_s.size());
  hip_ok(hipMemcpy(got.data(), od, got.size() * 2, hipMemcpyDeviceToHost),
         "d2h");
  hip_ok(hipMemcpy(got_s.data(), sd, got_s.size() * 4, hipMemcpyDeviceToHost),
         "d2h");
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    bad += got[i] != want[i];
  for (size_t i = 0; i < got_s.size(); ++i)
    bad += std::memcmp(&got_s[i], &want_s[i], 4) != 0;
  std::vector<double> rounds;
  for (int r = 0; r < env.rounds; ++r) {
    hip_ok(hipEventRecord(env.e0, env.stream), "hipEventRecord");
    for (int i = 0; i < env.iters; ++i)
      (void)turbine_hip::launch_quantize_mxfp4(env.stream, &q);
    hip_ok(hipEventRecord(env.e1, env.stream), "hipEventRecord");
    hip_ok(hipEventSynchronize(env.e1), "hipEventSynchronize");
    float ms = 0;
    hip_ok(hipEventElapsedTime(&ms, env.e0, env.e1), "hipEventElapsedTime");
    rounds.push_back(1000.0 * ms / env.iters);
  }
  std::printf("qdq rows=%lld cols=%lld us=%.1f %s (%zu mismatches)\n",
              static_cast<long long>(rows), static_cast<long long>(cols),
              median(rounds), bad == 0 ? "bit-exact" : "MISMATCH", bad);
  (void)hipFree(xd);
  (void)hipFree(od);
  (void)hipFree(sd);
  return bad == 0;
}

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    const size_t end = s.find(',', start);
    out.push_back(s.substr(start, end == std::string::npos ? std::string::npos
                                                           : end - start));
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::vector<int64_t> ms{1, 4, 16, 128, 2048};
  std::vector<std::string> shapes;
  Env env{};
  env.iters = 20;
  env.rounds = 5;
  bool check_all = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const char *val = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--ms") {
      ms.clear();
      for (const auto &s : split(val))
        ms.push_back(std::stoll(s));
      ++i;
    } else if (arg == "--iters") {
      env.iters = std::atoi(val);
      ++i;
    } else if (arg == "--rounds") {
      env.rounds = std::atoi(val);
      ++i;
    } else if (arg == "--shapes") {
      shapes = split(val);
      ++i;
    } else if (arg == "--no-check") {
      check_all = false;
    } else {
      std::fprintf(stderr, "usage: turbine_qgemm_mxfp4_eval [--ms a,b] "
                           "[--iters n] [--rounds n] [--shapes a,b] "
                           "[--no-check]\n");
      return 2;
    }
  }
  hip_ok(hipStreamCreate(&env.stream), "hipStreamCreate");
  if (hipblasLtCreate(&env.lt) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "hipblasLtCreate failed\n");
    return 2;
  }
  hip_ok(hipMalloc(&env.workspace, kWorkspaceBytes), "hipMalloc workspace");
  hip_ok(hipEventCreate(&env.e0), "hipEventCreate");
  hip_ok(hipEventCreate(&env.e1), "hipEventCreate");
  hipDeviceProp_t prop;
  hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::printf("device %s (%s), iters %d, rounds %d\n", prop.name,
              prop.gcnArchName, env.iters, env.rounds);

  bool all_ok = true;
  Rng rng{97};
  const int64_t m_max = *std::max_element(ms.begin(), ms.end());
  const Cand cands[] = {Cand::Bf16First, Cand::Bf16Best, Cand::DequantBf16,
                        Cand::Small,     Cand::Medium,   Cand::Large};
  for (const LinearShape &sh : kShapes) {
    if (!shapes.empty() &&
        std::find(shapes.begin(), shapes.end(), sh.name) == shapes.end()) {
      continue;
    }
    ShapeData d = make_shape(sh, m_max, rng);
    for (int64_t m : ms) {
      Blas blas;
      prepare_blas(env, d, m, blas);
      for (Cand cand : cands) {
        double us = std::nan("");
        const hipblasLtMatmulAlgo_t *algo = nullptr;
        if (cand == Cand::Bf16First || cand == Cand::DequantBf16) {
          if (!blas.ok)
            continue;
          algo = &blas.algos[0];
          us = time_cand(env, cand, d, blas, m, algo);
        } else if (cand == Cand::Bf16Best) {
          if (!blas.ok)
            continue;
          for (const auto &a : blas.algos) {
            const double t = time_cand(env, cand, d, blas, m, &a);
            if (std::isnan(us) || t < us) {
              us = t;
              algo = &a;
            }
          }
        } else {
          us = time_cand(env, cand, d, blas, m, nullptr);
        }
        std::string verdict = "-";
        const bool mx = cand != Cand::Bf16First && cand != Cand::Bf16Best;
        if (mx && check_all) {
          const Check c = check(env, cand, d, blas, m, algo);
          char buf[96];
          std::snprintf(buf, sizeof buf, "%s max|d| %.2e (ref max %.2e)",
                        c.ok ? "ok" : "WRONG", c.max_abs, c.ref_max);
          verdict = buf;
          all_ok = all_ok && c.ok;
        }
        const double gb =
            static_cast<double>(d.n * d.k / 2 + d.n * d.k / 32) / 1e3;
        std::printf(
            "%-11s m=%-5lld %-13s us=%9.1f  wGB/s=%7.1f  TFLOPS=%6.1f  %s\n",
            sh.name, static_cast<long long>(m), cand_name(cand), us, gb / us,
            2.0 * m * d.n * d.k / us / 1e6, verdict.c_str());
        std::fflush(stdout);
      }
    }
    free_shape(d);
  }
  for (int64_t cols : {3072, 8192, 4096, 14336}) {
    for (int64_t rows : ms)
      all_ok = eval_qdq(env, rows, cols, rng) && all_ok;
  }
  std::printf("%s\n", all_ok ? "ALL OK" : "SOME CHECKS FAILED");
  return all_ok ? 0 : 1;
}
