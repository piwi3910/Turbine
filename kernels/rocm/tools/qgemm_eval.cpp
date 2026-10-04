// turbine_qgemm_eval: the FP8 GEMM provider evaluation (Phase 6a Task 12,
// kernel reuse rule) on the current device. For the Llama-3.2-3B linear shapes
// at the requested m it runs every hipBLASLt candidate the library could use,
// checks it against a host reference of the CPU provider's semantics
// (crates/turbine-kernels/src/cpu/qgemm.rs: e4m3 values times their scales,
// F32-exact products, the sum rounded once to BF16) and times it:
//
//   bf16         BF16 x BF16 -> BF16, the dequantized weight (today's path,
//                W8A16 through dequantize; the baseline)
//   fp8_sab      e4m3 x e4m3 -> BF16, scalar weight and activation scales
//                (FP8_TENSOR weights, FP8_TENSOR static activations)
//   fp8_sabv     vector x vector (FP8_CHANNEL weights, FP8_TOKEN activations)
//   fp8_wv_as    vector weight scales, scalar activation scale
//   fp8_ws_av    scalar weight scale, vector activation scales
//
// Per candidate and m: the number of algorithms hipBLASLt's heuristic returns
// (up to 8), the first answer's and the best answer's microseconds (median of
// --rounds rounds of --iters calls, rotating over enough weight copies to
// defeat the caches), the largest |GPU - reference| over sampled rows, and
// whether the first answer's row 0 is bitwise the same at every m (row
// invariance: what prefix reuse needs for bit-exact warm prefills).
// It also checks and times the activation quantization kernels
// (src/qgemm_quantize.hpp: FP8_TOKEN, FP8_GROUP128, FP8_TENSOR) bit-exactly
// against the reference's rounding.
//
//   turbine_qgemm_eval [--ms 1,16,128,2048] [--iters 20] [--rounds 5]
//                      [--shapes qkv,o,gate_up,down]
//
// A lab tool (run it on GPU 0 under scripts/bench-lock.sh); not loaded by the
// server. Exit 0 when every candidate that has an algorithm matched the
// reference and every quantization was bit-exact, 1 otherwise.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gemm_problem.hpp"
#include "qgemm_epilogue.hpp"
#include "qgemm_fp8_block_kernels.hpp"
#include "qgemm_problem.hpp"
#include "qgemm_quantize.hpp"
#ifdef TURBINE_QGEMM_EVAL_CK
#include "qgemm_eval_ck.hpp"
#endif

namespace {

using turbine_hip::GemmProblem;
using turbine_hip::GemmShape;
using turbine_hip::QGemmProblem;
using turbine_hip::QGemmShape;
using turbine_hip::QScale;

constexpr size_t kWorkspaceBytes = 32u << 20;
constexpr int kMaxAlgos = 8;
// Weight bytes the rotation covers at least (the R9700 has 64 MiB of
// infinity cache).
constexpr size_t kRotateBytes = 512u << 20;

void hip_ok(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "qgemm_eval: %s: %s\n", what, hipGetErrorName(e));
    std::exit(2);
  }
}

// splitmix64 plus Box-Muller, as the Rust lab tests use.
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
  const uint32_t lsb = (b >> 16) & 1;
  return static_cast<uint16_t>((b + 0x7fff + lsb) >> 16);
}

float bf16_value(uint16_t h) {
  const uint32_t b = static_cast<uint32_t>(h) << 16;
  float x;
  std::memcpy(&x, &b, 4);
  return x;
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
    {"qkv", 5120, 3072},
    {"o", 3072, 3072},
    {"gate_up", 16384, 3072},
    {"down", 3072, 8192},
};

enum class Cand {
  Bf16,
  Sab,
  Sabv,
  WvAs,
  WsAv,
  // Composable Kernel ck_tile gemm_quant (qgemm_eval_ck.cpp), decode and
  // prefill tiles.
  CkRowColDecode,
  CkRowColPrefill,
  CkTensorDecode,
  CkTensorPrefill,
};

bool is_ck(Cand c) { return c >= Cand::CkRowColDecode; }

struct CandInfo {
  Cand cand;
  const char *name;
  QScale w, a;
};

const CandInfo kCands[] = {
    {Cand::Bf16, "bf16", QScale::Scalar, QScale::Scalar},
    {Cand::Sab, "fp8_sab", QScale::Scalar, QScale::Scalar},
    {Cand::Sabv, "fp8_sabv", QScale::Vector, QScale::Vector},
    {Cand::WvAs, "fp8_wv_as", QScale::Vector, QScale::Scalar},
    {Cand::WsAv, "fp8_ws_av", QScale::Scalar, QScale::Vector},
#ifdef TURBINE_QGEMM_EVAL_CK
    {Cand::CkRowColDecode, "ck_rowcol_decode", QScale::Vector, QScale::Vector},
    {Cand::CkRowColPrefill, "ck_rowcol_prefill", QScale::Vector,
     QScale::Vector},
    {Cand::CkTensorDecode, "ck_tensor_decode", QScale::Scalar, QScale::Scalar},
    {Cand::CkTensorPrefill, "ck_tensor_prefill", QScale::Scalar,
     QScale::Scalar},
#endif
};

struct Env {
  hipStream_t stream;
  hipblasLtHandle_t lt;
  void *workspace;
  hipEvent_t e0, e1;
  int iters, rounds;
};

// The data of one linear shape: e4m3 weights with per-row scales (their
// dequantized BF16 twin for the baseline), activations for the largest m as
// BF16 and quantized per token, R device copies of each weight.
struct ShapeData {
  int64_t n, k, m_max;
  std::vector<uint8_t> wq;
  std::vector<float> ws;   // [n]
  std::vector<uint8_t> aq; // [m_max, k] per-token codes
  std::vector<float> as;   // [m_max]
  std::vector<void *> w_fp8, w_bf16;
  uint8_t *a_fp8 = nullptr;
  uint16_t *a_bf16 = nullptr;
  float *ws_dev = nullptr, *as_dev = nullptr;
  void *c = nullptr;
};

ShapeData make_shape(const LinearShape &s, int64_t m_max, Rng &rng) {
  ShapeData d;
  d.n = s.n;
  d.k = s.k;
  d.m_max = m_max;
  d.wq.resize(s.n * s.k);
  d.ws.resize(s.n);
  std::vector<float> row(s.k);
  std::vector<uint16_t> wbf(s.n * s.k);
  for (int64_t r = 0; r < s.n; ++r) {
    float amax = 0;
    for (int64_t c = 0; c < s.k; ++c) {
      row[c] = rng.normal(0.02f);
      amax = std::max(amax, std::fabs(row[c]));
    }
    const float sc = turbine_hip::dynamic_fp8_scale(amax);
    d.ws[r] = sc;
    for (int64_t c = 0; c < s.k; ++c) {
      const uint8_t q = turbine_hip::fp8_e4m3_round(row[c] / sc);
      d.wq[r * s.k + c] = q;
      wbf[r * s.k + c] = bf16_bits(turbine_hip::fp8_e4m3_value(q) * sc);
    }
  }
  d.aq.resize(m_max * s.k);
  d.as.resize(m_max);
  std::vector<uint16_t> abf(m_max * s.k);
  for (int64_t r = 0; r < m_max; ++r) {
    float amax = 0;
    for (int64_t c = 0; c < s.k; ++c) {
      const uint16_t h = bf16_bits(rng.normal(1.0f));
      abf[r * s.k + c] = h;
      amax = std::max(amax, std::fabs(bf16_value(h)));
    }
    const float sc = turbine_hip::dynamic_fp8_scale(amax);
    d.as[r] = sc;
    for (int64_t c = 0; c < s.k; ++c)
      d.aq[r * s.k + c] =
          turbine_hip::fp8_e4m3_round(bf16_value(abf[r * s.k + c]) / sc);
  }
  const size_t w_bytes = static_cast<size_t>(s.n * s.k);
  const size_t copies =
      std::min<size_t>(8, std::max<size_t>(2, kRotateBytes / w_bytes + 1));
  for (size_t i = 0; i < copies; ++i) {
    d.w_fp8.push_back(device_copy(d.wq));
    d.w_bf16.push_back(device_copy(wbf));
  }
  d.a_fp8 = device_copy(d.aq);
  d.a_bf16 = device_copy(abf);
  d.ws_dev = device_copy(d.ws);
  d.as_dev = device_copy(d.as);
  hip_ok(hipMalloc(&d.c, static_cast<size_t>(m_max * s.n) * 4), "hipMalloc c");
  return d;
}

void free_shape(ShapeData &d) {
  for (void *p : d.w_fp8)
    (void)hipFree(p);
  for (void *p : d.w_bf16)
    (void)hipFree(p);
  (void)hipFree(d.a_fp8);
  (void)hipFree(d.a_bf16);
  (void)hipFree(d.ws_dev);
  (void)hipFree(d.as_dev);
  (void)hipFree(d.c);
}

// One candidate's problem at one m: descriptors plus the heuristic's answers.
struct Prepared {
  GemmProblem bf16;
  QGemmProblem fp8;
  std::vector<hipblasLtMatmulAlgo_t> algos;
  hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
  const char *failed = nullptr;
  int64_t m = 0;
};

bool prepare(const Env &env, const CandInfo &ci, ShapeData &d, int64_t m,
             Prepared &p) {
  p.m = m;
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t wl, al, ol;
  if (is_ck(ci.cand)) {
    p.algos.push_back(hipblasLtMatmulAlgo_t{}); // one "algorithm": the instance
    return true;
  }
  if (ci.cand == Cand::Bf16) {
    const GemmShape s{m, d.n, d.k, d.k, d.k, d.n, 1, TURBINE_DTYPE_BF16};
    p.status = p.bf16.make(s);
    p.failed = p.bf16.failed;
    if (p.status != HIPBLAS_STATUS_SUCCESS)
      return false;
    desc = p.bf16.desc.handle;
    wl = p.bf16.weight.handle;
    al = p.bf16.act.handle;
    ol = p.bf16.out.handle;
  } else {
    const QGemmShape s{m, d.n, d.k, d.k, d.n, TURBINE_DTYPE_BF16, ci.w, ci.a};
    p.status = p.fp8.make(s);
    if (p.status == HIPBLAS_STATUS_SUCCESS)
      p.status = p.fp8.set_scales(d.ws_dev, d.as_dev);
    p.failed = p.fp8.failed;
    if (p.status != HIPBLAS_STATUS_SUCCESS)
      return false;
    desc = p.fp8.desc.handle;
    wl = p.fp8.weight.handle;
    al = p.fp8.act.handle;
    ol = p.fp8.out.handle;
  }
  turbine_hip::Preference pref;
  (void)hipblasLtMatmulPreferenceCreate(&pref.handle);
  const uint64_t ws = kWorkspaceBytes;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      pref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  hipblasLtMatmulHeuristicResult_t res[kMaxAlgos];
  int returned = 0;
  p.status = hipblasLtMatmulAlgoGetHeuristic(
      env.lt, desc, wl, al, ol, ol, pref.handle, kMaxAlgos, res, &returned);
  p.failed = "hipblasLtMatmulAlgoGetHeuristic";
  if (p.status != HIPBLAS_STATUS_SUCCESS)
    return false;
  for (int i = 0; i < returned; ++i)
    if (res[i].state == HIPBLAS_STATUS_SUCCESS)
      p.algos.push_back(res[i].algo);
  return !p.algos.empty();
}

hipblasStatus_t run(const Env &env, const CandInfo &ci, ShapeData &d,
                    Prepared &p, const hipblasLtMatmulAlgo_t &algo, size_t w) {
  const float alpha = 1.0f, beta = 0.0f;
#ifdef TURBINE_QGEMM_EVAL_CK
  if (is_ck(ci.cand)) {
    const CkQuantMode mode =
        ci.w == QScale::Vector ? CkQuantMode::RowCol : CkQuantMode::Tensor;
    const bool prefill =
        ci.cand == Cand::CkRowColPrefill || ci.cand == Cand::CkTensorPrefill;
    return ck_qgemm(mode, prefill, d.a_fp8, d.w_fp8[w], d.c, d.as_dev, d.ws_dev,
                    p.m, d.n, d.k, env.stream) == 0
               ? HIPBLAS_STATUS_SUCCESS
               : HIPBLAS_STATUS_NOT_SUPPORTED;
  }
#endif
  if (ci.cand == Cand::Bf16) {
    return hipblasLtMatmul(
        env.lt, p.bf16.desc.handle, &alpha, d.w_bf16[w], p.bf16.weight.handle,
        d.a_bf16, p.bf16.act.handle, &beta, d.c, p.bf16.out.handle, d.c,
        p.bf16.out.handle, &algo, env.workspace, kWorkspaceBytes, env.stream);
  }
  return hipblasLtMatmul(env.lt, p.fp8.desc.handle, &alpha, d.w_fp8[w],
                         p.fp8.weight.handle, d.a_fp8, p.fp8.act.handle, &beta,
                         d.c, p.fp8.out.handle, d.c, p.fp8.out.handle, &algo,
                         env.workspace, kWorkspaceBytes, env.stream);
}

// Median microseconds per call of algo, or NaN when it fails to run.
double time_algo(const Env &env, const CandInfo &ci, ShapeData &d, Prepared &p,
                 const hipblasLtMatmulAlgo_t &algo) {
  const size_t copies = d.w_fp8.size();
  for (int i = 0; i < 3; ++i)
    if (run(env, ci, d, p, algo, i % copies) != HIPBLAS_STATUS_SUCCESS)
      return std::nan("");
  std::vector<double> rounds;
  for (int r = 0; r < env.rounds; ++r) {
    hip_ok(hipEventRecord(env.e0, env.stream), "hipEventRecord");
    for (int i = 0; i < env.iters; ++i)
      (void)run(env, ci, d, p, algo, i % copies);
    hip_ok(hipEventRecord(env.e1, env.stream), "hipEventRecord");
    hip_ok(hipEventSynchronize(env.e1), "hipEventSynchronize");
    float ms = 0;
    hip_ok(hipEventElapsedTime(&ms, env.e0, env.e1), "hipEventElapsedTime");
    rounds.push_back(1000.0 * ms / env.iters);
  }
  return median(rounds);
}

// Rows the correctness check compares (the reference is a host GEMM).
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

// GPU output rows vs the host reference of candidate ci's semantics. BF16
// output: each value within one BF16 rounding (2^-8 relative) of the exact sum
// plus 1e-6 of the row's scale for summation order.
Check check(const CandInfo &ci, const ShapeData &d, int64_t m,
            std::vector<uint16_t> &row0) {
  Check out;
  std::vector<uint16_t> got(static_cast<size_t>(m * d.n));
  hip_ok(hipMemcpy(got.data(), d.c, got.size() * 2, hipMemcpyDeviceToHost),
         "hipMemcpy d2h");
  row0.assign(got.begin(), got.begin() + d.n);
  // The BF16 baseline multiplies unquantized activations: timed, not checked.
  if (ci.cand == Cand::Bf16)
    return out;
  std::vector<double> av(d.k);
  for (int64_t r : sample_rows(m)) {
    for (int64_t c = 0; c < d.k; ++c)
      av[c] = turbine_hip::fp8_e4m3_value(d.aq[r * d.k + c]);
    const double sa = ci.a == QScale::Vector ? d.as[r] : d.as[0];
    for (int64_t j = 0; j < d.n; ++j) {
      double acc = 0;
      const uint8_t *w = &d.wq[j * d.k];
      for (int64_t c = 0; c < d.k; ++c)
        acc += av[c] * turbine_hip::fp8_e4m3_value(w[c]);
      const double sw = ci.w == QScale::Vector ? d.ws[j] : d.ws[0];
      const double ref = acc * sa * sw;
      const double g = bf16_value(got[r * d.n + j]);
      const double err = std::fabs(g - ref);
      out.max_abs = std::max(out.max_abs, err);
      out.ref_max = std::max(out.ref_max, std::fabs(ref));
      if (err > std::fabs(ref) * (1.0 / 256.0) + 1e-6 * (1 + std::fabs(ref)))
        out.ok = false;
    }
  }
  return out;
}

// --- activation quantization ---

bool check_quant(int32_t mode, int64_t rows, int64_t cols, Rng &rng,
                 const Env &env, double *us) {
  std::vector<uint16_t> x(rows * cols);
  for (auto &h : x)
    h = bf16_bits(rng.normal(1.0f));
  // A zero row and a row of huge values: the floor and saturation paths.
  if (rows > 2) {
    std::fill(x.begin(), x.begin() + cols, 0);
    for (int64_t c = 0; c < cols; ++c)
      x[cols + c] = bf16_bits(rng.normal(1e4f));
  }
  const float static_scale = 0.0123f;
  const int64_t groups =
      (cols + turbine_hip::kActGroup - 1) / turbine_hip::kActGroup;
  const size_t n_scales = mode == TURBINE_ACTQ_FP8_TENSOR ? 1
                          : mode == TURBINE_ACTQ_FP8_TOKEN
                              ? rows
                              : static_cast<size_t>(rows * groups);
  // Host reference (the CPU provider's rule).
  std::vector<uint8_t> want(rows * cols);
  std::vector<float> want_s;
  const int64_t g =
      mode == TURBINE_ACTQ_FP8_GROUP128 ? turbine_hip::kActGroup : cols;
  if (mode == TURBINE_ACTQ_FP8_TENSOR)
    want_s.push_back(static_scale);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c0 = 0; c0 < cols; c0 += g) {
      const int64_t c1 = std::min(cols, c0 + g);
      float s = static_scale;
      if (mode != TURBINE_ACTQ_FP8_TENSOR) {
        float amax = 0;
        for (int64_t c = c0; c < c1; ++c)
          amax = std::max(amax, std::fabs(bf16_value(x[r * cols + c])));
        s = turbine_hip::dynamic_fp8_scale(amax);
        want_s.push_back(s);
      }
      for (int64_t c = c0; c < c1; ++c)
        want[r * cols + c] =
            turbine_hip::fp8_e4m3_round(bf16_value(x[r * cols + c]) / s);
    }
  }
  uint16_t *xd = device_copy(x);
  void *od = nullptr;
  float *sd = nullptr;
  hip_ok(hipMalloc(&od, rows * cols), "hipMalloc out");
  hip_ok(hipMalloc(&sd, n_scales * 4), "hipMalloc scales");
  turbine_quantize_act_desc desc{};
  desc.x = xd;
  desc.out = od;
  desc.scales = sd;
  desc.rows = rows;
  desc.cols = cols;
  desc.x_stride_row = cols;
  desc.out_stride_row = cols;
  desc.mode = mode;
  desc.static_scale = static_scale;
  desc.x_dtype = TURBINE_DTYPE_BF16;
  desc.out_dtype = TURBINE_DTYPE_F8E4M3;
  hip_ok(turbine_hip::launch_quantize_fp8(&desc, env.stream), "quantize");
  hip_ok(hipStreamSynchronize(env.stream), "sync");
  std::vector<uint8_t> got(rows * cols);
  std::vector<float> got_s(n_scales);
  hip_ok(hipMemcpy(got.data(), od, got.size(), hipMemcpyDeviceToHost), "d2h");
  hip_ok(hipMemcpy(got_s.data(), sd, n_scales * 4, hipMemcpyDeviceToHost),
         "d2h");
  bool ok = got == want &&
            std::memcmp(got_s.data(), want_s.data(), n_scales * 4) == 0;
  std::vector<double> rounds;
  for (int r = 0; r < env.rounds; ++r) {
    hip_ok(hipEventRecord(env.e0, env.stream), "hipEventRecord");
    for (int i = 0; i < env.iters; ++i)
      (void)turbine_hip::launch_quantize_fp8(&desc, env.stream);
    hip_ok(hipEventRecord(env.e1, env.stream), "hipEventRecord");
    hip_ok(hipEventSynchronize(env.e1), "hipEventSynchronize");
    float ms = 0;
    hip_ok(hipEventElapsedTime(&ms, env.e0, env.e1), "hipEventElapsedTime");
    rounds.push_back(1000.0 * ms / env.iters);
  }
  *us = median(rounds);
  (void)hipFree(xd);
  (void)hipFree(od);
  (void)hipFree(sd);
  return ok;
}

// --tune: for each shape and FP8 scale layout (SAB, SABV) and each decode
// bucket m in {1, 2, 4, 8, 16, 32, 64}, times every solution hipBLASLt lists
// for e4m3 x e4m3 -> BF16 that takes the problem within the workspace, checks
// the fastest against the reference and prints it as a row of the pinned FP8
// table (src/qgemm_tuned.hpp):
//   qgemm_tune n k scale m_max index name best_us heuristic_us
bool tune(Env &env, Rng &rng, const std::vector<std::string> &shapes) {
  bool all_ok = true;
  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  if (hipblaslt_ext::getAllAlgos(
          env.lt, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T,
          HIPBLAS_OP_N, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16BF,
          HIPBLAS_COMPUTE_32F, all) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "qgemm_eval: getAllAlgos (FP8) failed\n");
    return false;
  }
  std::printf("# %zu FP8 solutions listed\n", all.size());
  const int64_t buckets[] = {1, 2, 4, 8, 16, 32, 64};
  for (const LinearShape &s : kShapes) {
    if (std::find(shapes.begin(), shapes.end(), s.name) == shapes.end())
      continue;
    ShapeData d = make_shape(s, 64, rng);
    for (const CandInfo &ci : kCands) {
      if (ci.cand != Cand::Sab && ci.cand != Cand::Sabv)
        continue;
      for (int64_t m : buckets) {
        Prepared p;
        if (!prepare(env, ci, d, m, p)) {
          std::printf("qgemm_tune %s %s m=%lld: no heuristic answer\n", s.name,
                      ci.name, (long long)m);
          all_ok = false;
          continue;
        }
        const double heuristic = time_algo(env, ci, d, p, p.algos[0]);
        double best = heuristic;
        const hipblasLtMatmulAlgo_t *best_algo = &p.algos[0];
        const float alpha = 1.0f, beta = 0.0f;
        for (const auto &r : all) {
          size_t ws = 0;
          if (hipblaslt_ext::matmulIsAlgoSupported(
                  env.lt, p.fp8.desc.handle, &alpha, p.fp8.weight.handle,
                  p.fp8.act.handle, &beta, p.fp8.out.handle, p.fp8.out.handle,
                  const_cast<hipblasLtMatmulAlgo_t &>(r.algo),
                  ws) != HIPBLAS_STATUS_SUCCESS ||
              ws > kWorkspaceBytes) {
            continue;
          }
          const double t = time_algo(env, ci, d, p, r.algo);
          if (!std::isnan(t) && t < best) {
            best = t;
            best_algo = &r.algo;
          }
        }
        (void)run(env, ci, d, p, *best_algo, 0);
        hip_ok(hipStreamSynchronize(env.stream), "sync");
        std::vector<uint16_t> row0;
        const Check c = check(ci, d, m, row0);
        all_ok = all_ok && c.ok;
        hipblasLtMatmulAlgo_t a = *best_algo;
        const std::string name =
            best_algo == &p.algos[0]
                ? std::string("heuristic")
                : hipblaslt_ext::getSolutionNameFromAlgo(env.lt, a);
        const int index =
            best_algo == &p.algos[0] ? -1 : hipblaslt_ext::getIndexFromAlgo(a);
        std::printf("qgemm_tune %lld %lld %s %lld %d %s %.2f %.2f %s\n",
                    (long long)s.n, (long long)s.k,
                    ci.w == QScale::Vector ? "vector" : "scalar", (long long)m,
                    index, name.c_str(), best, heuristic,
                    c.ok ? "ok" : "MISMATCH");
        std::fflush(stdout);
      }
    }
    free_shape(d);
  }
  return all_ok;
}

// --block 1: the block-scaled FP8 candidates (Phase 6a Task 15, 128 x 128
// weight blocks): CK ck_tile ABQuantGrouped (A per row and 128 columns, B per
// block; W8A8) against the host reference, and the W8A16 fallback of decision
// Q3 timed as its parts: an own dequantize-to-BF16 kernel over the whole
// weight, and the BF16 hipBLASLt GEMM on the dequantized weight (also the cost
// of dequantizing once at load), per shape and m.
__global__ void dequant_block_bf16(const uint8_t *w, const float *scales,
                                   uint16_t *out, int64_t n, int64_t k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n * k)
    return;
  const int64_t r = i / k, c = i % k;
  const int64_t kb = (k + 127) / 128;
  const float v =
      turbine_hip::fp8_e4m3_value(w[i]) * scales[(r / 128) * kb + c / 128];
  uint32_t b;
  __builtin_memcpy(&b, &v, 4);
  out[i] = static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

template <typename F> double time_launches(const Env &env, F &&launch) {
  for (int i = 0; i < 3; ++i)
    launch(i);
  std::vector<double> rounds;
  for (int r = 0; r < env.rounds; ++r) {
    hip_ok(hipEventRecord(env.e0, env.stream), "hipEventRecord");
    for (int i = 0; i < env.iters; ++i)
      launch(i);
    hip_ok(hipEventRecord(env.e1, env.stream), "hipEventRecord");
    hip_ok(hipEventSynchronize(env.e1), "hipEventSynchronize");
    float ms = 0;
    hip_ok(hipEventElapsedTime(&ms, env.e0, env.e1), "hipEventElapsedTime");
    rounds.push_back(1000.0 * ms / env.iters);
  }
  return median(rounds);
}

bool eval_block(Env &env, Rng &rng, const std::vector<std::string> &shapes,
                const std::vector<int64_t> &ms) {
  bool all_ok = true;
  const int64_t m_max = ms.back();
  for (const LinearShape &s : kShapes) {
    if (std::find(shapes.begin(), shapes.end(), s.name) == shapes.end())
      continue;
    const int64_t n = s.n, k = s.k, nb = (n + 127) / 128, kb = (k + 127) / 128;
    std::vector<float> w(n * k);
    for (auto &v : w)
      v = rng.normal(0.02f);
    std::vector<float> sw(nb * kb, 0.0f);
    for (int64_t r = 0; r < n; ++r)
      for (int64_t c = 0; c < k; ++c) {
        float &a = sw[(r / 128) * kb + c / 128];
        a = std::max(a, std::fabs(w[r * k + c]));
      }
    for (auto &v : sw)
      v = turbine_hip::dynamic_fp8_scale(v);
    std::vector<uint8_t> wq(n * k);
    std::vector<uint16_t> wbf(n * k);
    for (int64_t r = 0; r < n; ++r)
      for (int64_t c = 0; c < k; ++c) {
        const float sc = sw[(r / 128) * kb + c / 128];
        wq[r * k + c] = turbine_hip::fp8_e4m3_round(w[r * k + c] / sc);
        wbf[r * k + c] =
            bf16_bits(turbine_hip::fp8_e4m3_value(wq[r * k + c]) * sc);
      }
    std::vector<uint8_t> aq(m_max * k);
    std::vector<uint16_t> abf(m_max * k);
    std::vector<float> sa(m_max * kb);
    for (int64_t r = 0; r < m_max; ++r)
      for (int64_t g = 0; g < kb; ++g) {
        float amax = 0;
        const int64_t c1 = std::min(k, g * 128 + 128);
        for (int64_t c = g * 128; c < c1; ++c) {
          abf[r * k + c] = bf16_bits(rng.normal(1.0f));
          amax = std::max(amax, std::fabs(bf16_value(abf[r * k + c])));
        }
        const float sc = turbine_hip::dynamic_fp8_scale(amax);
        sa[r * kb + g] = sc;
        for (int64_t c = g * 128; c < c1; ++c)
          aq[r * k + c] =
              turbine_hip::fp8_e4m3_round(bf16_value(abf[r * k + c]) / sc);
      }
    const size_t copies = std::min<size_t>(
        8, std::max<size_t>(2, kRotateBytes / static_cast<size_t>(n * k) + 1));
    ShapeData view;
    view.n = n;
    view.k = k;
    view.m_max = m_max;
    for (size_t i = 0; i < copies; ++i) {
      view.w_fp8.push_back(device_copy(wq));
      view.w_bf16.push_back(device_copy(wbf));
    }
    view.a_fp8 = device_copy(aq);
    view.a_bf16 = device_copy(abf);
    view.ws_dev = device_copy(sw);
    view.as_dev = device_copy(sa);
    hip_ok(hipMalloc(&view.c, static_cast<size_t>(m_max * n) * 4),
           "hipMalloc c");
    void *deq = nullptr;
    hip_ok(hipMalloc(&deq, static_cast<size_t>(n * k) * 2), "hipMalloc deq");
    const double t_deq = time_launches(env, [&](int i) {
      const int64_t total = n * k;
      hipLaunchKernelGGL(dequant_block_bf16,
                         dim3(static_cast<uint32_t>((total + 255) / 256)),
                         dim3(256), 0, env.stream,
                         static_cast<const uint8_t *>(view.w_fp8[i % copies]),
                         view.ws_dev, static_cast<uint16_t *>(deq), n, k);
    });
    // The dequantize kernel against the host's dequantized weight.
    std::vector<uint16_t> got_deq(n * k);
    hip_ok(hipMemcpy(got_deq.data(), deq, got_deq.size() * 2,
                     hipMemcpyDeviceToHost),
           "d2h");
    const bool deq_ok = got_deq == wbf;
    all_ok = all_ok && deq_ok;
    std::printf("block shape=%s n=%lld k=%lld dequant_bf16_us=%.2f %s\n",
                s.name, (long long)n, (long long)k, t_deq,
                deq_ok ? "bit-exact" : "MISMATCH");
    for (int64_t m : ms) {
      // BF16 GEMM on the dequantized weight (the heuristic's first answer).
      const CandInfo bf16{Cand::Bf16, "bf16", QScale::Scalar, QScale::Scalar};
      Prepared p;
      double t_bf16 = std::nan("");
      if (prepare(env, bf16, view, m, p))
        t_bf16 = time_algo(env, bf16, view, p, p.algos[0]);
      std::printf("block shape=%s m=%lld cand=w8a16_bf16_gemm us=%.2f "
                  "(+ %.2f to dequantize per call)\n",
                  s.name, (long long)m, t_bf16, t_deq);
      // The own fused W8A16 kernel (turbine_hip_fp8_block's decode path),
      // launched as the library does (waves by k / 128).
      if (m <= 64) {
        const int64_t blocks = k / 128;
        auto fused = [&](int i) {
          const auto *w = static_cast<const uint8_t *>(view.w_fp8[i % copies]);
          if (blocks >= 64)
            (void)turbine_hip::fp8_block::launch_wmma<1, 8>(
                view.a_bf16, w, view.ws_dev, view.c, m, n, k, k, n, false, 1.0f,
                env.stream);
          else if (blocks >= 32)
            (void)turbine_hip::fp8_block::launch_wmma<1, 4>(
                view.a_bf16, w, view.ws_dev, view.c, m, n, k, k, n, false, 1.0f,
                env.stream);
          else
            (void)turbine_hip::fp8_block::launch_wmma<1, 2>(
                view.a_bf16, w, view.ws_dev, view.c, m, n, k, k, n, false, 1.0f,
                env.stream);
        };
        const double t_fused = time_launches(env, fused);
        std::printf("block shape=%s m=%lld cand=turbine_hip_fp8_block_fused "
                    "us=%.2f\n",
                    s.name, (long long)m, t_fused);
      }
#ifdef TURBINE_QGEMM_EVAL_CK
      for (const bool prefill : {false, true}) {
        const char *name = prefill ? "ck_abquant_prefill" : "ck_abquant_decode";
        auto launch = [&](int i) {
          return ck_qgemm(CkQuantMode::Block128, prefill, view.a_fp8,
                          view.w_fp8[i % copies], view.c, view.as_dev,
                          view.ws_dev, m, n, k, env.stream);
        };
        if (launch(0) != 0) {
          std::printf("block shape=%s m=%lld cand=%s refused\n", s.name,
                      (long long)m, name);
          continue;
        }
        hip_ok(hipStreamSynchronize(env.stream), "sync");
        std::vector<uint16_t> got(static_cast<size_t>(m * n));
        hip_ok(hipMemcpy(got.data(), view.c, got.size() * 2,
                         hipMemcpyDeviceToHost),
               "d2h");
        double max_abs = 0, ref_max = 0;
        bool ok = true;
        for (int64_t r : sample_rows(m)) {
          for (int64_t j = 0; j < n; ++j) {
            double acc = 0;
            for (int64_t g = 0; g < kb; ++g) {
              double part = 0;
              const int64_t c1 = std::min(k, g * 128 + 128);
              for (int64_t cc = g * 128; cc < c1; ++cc)
                part += static_cast<double>(
                            turbine_hip::fp8_e4m3_value(aq[r * k + cc])) *
                        turbine_hip::fp8_e4m3_value(wq[j * k + cc]);
              acc += part * sa[r * kb + g] * sw[(j / 128) * kb + g];
            }
            const double err = std::fabs(bf16_value(got[r * n + j]) - acc);
            max_abs = std::max(max_abs, err);
            ref_max = std::max(ref_max, std::fabs(acc));
            if (!(err <= std::fabs(acc) / 256.0 + 1e-6 * (1 + std::fabs(acc))))
              ok = false;
          }
        }
        const double t = time_launches(env, [&](int i) { (void)launch(i); });
        all_ok = all_ok && ok;
        std::printf("block shape=%s m=%lld cand=%s us=%.2f max_abs=%.3e "
                    "ref_max=%.3e %s\n",
                    s.name, (long long)m, name, t, max_abs, ref_max,
                    ok ? "ok" : "MISMATCH");
      }
#endif
    }
    free_shape(view);
    (void)hipFree(deq);
  }
  return all_ok;
}

struct ExtRun {
  QGemmProblem p;
  hipblaslt_ext::Gemm *gemm = nullptr;
  ~ExtRun() { delete gemm; }
};

// Builds and initializes the split-K-off call of algo on the rows x k
// activations a (scales as: [rows] or [1]) into c (rows x n). With f32 the
// call is the prefill path of vector scales: scalar scales (as, d.ws_dev read
// as [1]) and F32 out.
bool ext_init(const Env &env, const CandInfo &ci, ShapeData &d,
              hipblasLtMatmulAlgo_t algo, const uint8_t *a, const float *as,
              int64_t rows, void *c, size_t w, bool f32, ExtRun &r) {
  const QGemmShape s{rows,
                     d.n,
                     d.k,
                     d.k,
                     d.n,
                     f32 ? TURBINE_DTYPE_F32 : TURBINE_DTYPE_BF16,
                     f32 ? QScale::Scalar : ci.w,
                     f32 ? QScale::Scalar : ci.a};
  if (r.p.make(s) != HIPBLAS_STATUS_SUCCESS)
    return false;
  if (r.p.set_scales(d.ws_dev, as) != HIPBLAS_STATUS_SUCCESS)
    return false;
  static const float alpha = 1.0f, beta = 0.0f;
  try {
    r.gemm = new hipblaslt_ext::Gemm(
        env.lt, r.p.desc.handle, &alpha, d.w_fp8[w], r.p.weight.handle, a,
        r.p.act.handle, &beta, c, r.p.out.handle, c, r.p.out.handle);
    hipblaslt_ext::GemmTuning tuning;
    tuning.setSplitK(1);
    r.gemm->setMaxWorkspaceBytes(kWorkspaceBytes);
    size_t ws = 0;
    if (r.gemm->isAlgoSupported(algo, tuning, ws) != HIPBLAS_STATUS_SUCCESS ||
        ws > kWorkspaceBytes)
      return false;
    return r.gemm->initialize(algo, tuning, env.workspace, false, env.stream) ==
           HIPBLAS_STATUS_SUCCESS;
  } catch (const std::exception &) {
    return false;
  }
}

// Rows [first, first + rows) of d's activations as their own call (raw
// output bytes): the rows and their scales are copied into fresh (aligned)
// buffers, as a prefill of a prompt's suffix has them.
bool ext_rows(const Env &env, const CandInfo &ci, ShapeData &d,
              hipblasLtMatmulAlgo_t algo, int64_t first, int64_t rows, bool f32,
              std::vector<uint8_t> &out) {
  uint8_t *a = nullptr;
  float *as = nullptr;
  hip_ok(hipMalloc(&a, static_cast<size_t>(rows * d.k)), "hipMalloc a");
  hip_ok(hipMalloc(&as, static_cast<size_t>(rows) * sizeof(float)),
         "hipMalloc as");
  hip_ok(hipMemcpy(a, d.a_fp8 + first * d.k, static_cast<size_t>(rows * d.k),
                   hipMemcpyDeviceToDevice),
         "d2d a");
  const bool vec = !f32 && ci.a == QScale::Vector;
  hip_ok(hipMemcpy(as, d.as_dev + (vec ? first : 0),
                   static_cast<size_t>(vec ? rows : 1) * sizeof(float),
                   hipMemcpyDeviceToDevice),
         "d2d as");
  bool ok = false;
  {
    ExtRun r;
    if (ext_init(env, ci, d, algo, a, as, rows, d.c, 0, f32, r) &&
        r.gemm->run(env.stream) == HIPBLAS_STATUS_SUCCESS) {
      hip_ok(hipStreamSynchronize(env.stream), "sync");
      out.resize(static_cast<size_t>(rows * d.n) * (f32 ? 4 : 2));
      hip_ok(hipMemcpy(out.data(), d.c, out.size(), hipMemcpyDeviceToHost),
             "d2h");
      ok = true;
    }
  }
  (void)hipFree(a);
  (void)hipFree(as);
  return ok;
}

double ext_time(const Env &env, const CandInfo &ci, ShapeData &d,
                hipblasLtMatmulAlgo_t algo, int64_t m, bool f32) {
  std::vector<ExtRun> runs(d.w_fp8.size());
  for (size_t w = 0; w < runs.size(); ++w)
    if (!ext_init(env, ci, d, algo, d.a_fp8, d.as_dev, m, d.c, w, f32, runs[w]))
      return std::nan("");
  return time_launches(env, [&](int i) {
    (void)runs[static_cast<size_t>(i) % runs.size()].gemm->run(env.stream);
  });
}

// --tune-prefill 1: for each shape and FP8 scale layout, the prefill solution
// of the pinned table (a row with m_max 0). Scalar scales: hipBLASLt's
// e4m3 x e4m3 -> BF16 SAB solutions; vector scales: SAB solutions with F32
// out whose sums the own epilogue (src/qgemm_epilogue.hpp) scales per row and
// column (gfx1201 has no row-invariant OUTER_VEC solution). Every solution
// that takes the m = 2048 problem with split-K off, fastest first (m = 2048
// time + 4 x the m = 512 time), the first one whose rows are bitwise
// independent of the call (rows of a 513-row call equal the same rows computed
// as their own calls of 1, 7 and 128 rows from row 0, 213 rows from row 300 and
// 64 rows from row 1, in fresh buffers: what a prefix-reused prefill of a
// prompt's suffix needs) and that matches the reference:
//   qgemm_tune n k scale 0 index name us_2048 us_512 invariant (...)
// (vector rows' times include the epilogue).
bool tune_prefill(Env &env, Rng &rng, const std::vector<std::string> &shapes) {
  bool all_ok = true;
  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  if (hipblaslt_ext::getAllAlgos(
          env.lt, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T,
          HIPBLAS_OP_N, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF, HIP_R_16BF,
          HIPBLAS_COMPUTE_32F, all) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "qgemm_eval: getAllAlgos (FP8) failed\n");
    return false;
  }
  std::vector<hipblasLtMatmulHeuristicResult_t> all_f32;
  (void)hipblaslt_ext::getAllAlgos(
      env.lt, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T,
      HIPBLAS_OP_N, HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_32F, HIP_R_32F,
      HIPBLAS_COMPUTE_32F, all_f32);
  std::printf("# %zu FP8 BF16-out and %zu F32-out solutions listed\n",
              all.size(), all_f32.size());
  const int64_t m_full = 513;
  const std::pair<int64_t, int64_t> parts[] = {
      {0, 1}, {0, 7}, {0, 128}, {300, 213}, {1, 64}};
  for (const LinearShape &s : kShapes) {
    if (std::find(shapes.begin(), shapes.end(), s.name) == shapes.end())
      continue;
    ShapeData d = make_shape(s, 2048, rng);
    // The epilogue's cost for vector scales at m = 2048 and 512.
    double epi[2] = {0, 0};
    {
      void *out = nullptr;
      hip_ok(hipMalloc(&out, static_cast<size_t>(2048 * d.n) * 2),
             "hipMalloc epilogue out");
      const int64_t ms[2] = {2048, 512};
      for (int i = 0; i < 2; ++i) {
        epi[i] = time_launches(env, [&](int) {
          (void)turbine_hip::launch_qgemm_scale_epilogue(
              static_cast<const float *>(d.c), d.n, out, d.n, d.as_dev, true,
              d.ws_dev, true, 1.0f, ms[i], d.n, env.stream);
        });
      }
      (void)hipFree(out);
    }
    for (const CandInfo &ci : kCands) {
      if (ci.cand != Cand::Sab && ci.cand != Cand::Sabv)
        continue;
      const bool f32 = ci.cand == Cand::Sabv;
      const auto &list = f32 ? all_f32 : all;
      struct Timed {
        double us, us2048, us512;
        size_t i;
      };
      std::vector<Timed> timed;
      for (size_t i = 0; i < list.size(); ++i) {
        const double t2048 = ext_time(env, ci, d, list[i].algo, 2048, f32);
        if (std::isnan(t2048))
          continue;
        const double t512 = ext_time(env, ci, d, list[i].algo, 512, f32);
        if (std::isnan(t512))
          continue;
        const double e2048 = f32 ? epi[0] : 0.0, e512 = f32 ? epi[1] : 0.0;
        timed.push_back(
            {t2048 + e2048 + 4 * (t512 + e512), t2048 + e2048, t512 + e512, i});
      }
      std::sort(timed.begin(), timed.end(),
                [](const Timed &a, const Timed &b) { return a.us < b.us; });
      bool found = false;
      int rejected = 0;
      for (const Timed &t : timed) {
        const hipblasLtMatmulAlgo_t algo = list[t.i].algo;
        std::vector<uint8_t> whole;
        if (!ext_rows(env, ci, d, algo, 0, m_full, f32, whole))
          continue;
        const size_t row_bytes = static_cast<size_t>(d.n) * (f32 ? 4 : 2);
        bool invariant = true;
        for (const auto &[first, rows] : parts) {
          std::vector<uint8_t> part;
          if (!ext_rows(env, ci, d, algo, first, rows, f32, part) ||
              !std::equal(part.begin(), part.end(),
                          whole.begin() + first * row_bytes)) {
            invariant = false;
            break;
          }
        }
        if (!invariant) {
          ++rejected;
          continue;
        }
        // Reference: the sums (F32 out, unit-scale semantics: ws[0], as[0])
        // or the scaled BF16 values.
        bool ok = true;
        if (f32) {
          std::vector<double> av(d.k);
          for (int64_t r : sample_rows(m_full)) {
            for (int64_t c = 0; c < d.k; ++c)
              av[c] = turbine_hip::fp8_e4m3_value(d.aq[r * d.k + c]);
            for (int64_t j = 0; j < d.n; ++j) {
              double acc = 0;
              for (int64_t c = 0; c < d.k; ++c)
                acc += av[c] * turbine_hip::fp8_e4m3_value(d.wq[j * d.k + c]);
              const double ref = acc * d.ws[0] * d.as[0];
              float got;
              std::memcpy(&got, &whole[(r * d.n + j) * 4], 4);
              if (std::fabs(got - ref) > 1e-5 * (1 + std::fabs(ref)))
                ok = false;
            }
          }
        } else {
          std::vector<uint16_t> row0;
          (void)ext_rows(env, ci, d, algo, 0, m_full, false, whole);
          ok = check(ci, d, m_full, row0).ok;
        }
        if (!ok) {
          ++rejected;
          continue;
        }
        hipblasLtMatmulAlgo_t a = algo;
        std::printf("qgemm_tune %lld %lld %s 0 %d %s %.2f %.2f invariant "
                    "(%s, %d faster solutions rejected, %zu timed)\n",
                    (long long)s.n, (long long)s.k,
                    ci.w == QScale::Vector ? "vector" : "scalar",
                    hipblaslt_ext::getIndexFromAlgo(a),
                    hipblaslt_ext::getSolutionNameFromAlgo(env.lt, a).c_str(),
                    t.us2048, t.us512, f32 ? "SAB F32 + epilogue" : "SAB BF16",
                    rejected, timed.size());
        std::fflush(stdout);
        found = true;
        break;
      }
      if (!found) {
        std::printf("qgemm_tune %lld %lld %s 0: no row-invariant solution "
                    "(%zu timed)\n",
                    (long long)s.n, (long long)s.k,
                    ci.w == QScale::Vector ? "vector" : "scalar", timed.size());
        all_ok = false;
      }
    }
    free_shape(d);
  }
  return all_ok;
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
  std::vector<int64_t> ms{1, 16, 128, 2048};
  std::vector<std::string> shapes{"qkv", "o", "gate_up", "down"};
  bool tune_mode = false;
  bool block_mode = false;
  bool tune_prefill_mode = false;
  Env env{};
  env.iters = 20;
  env.rounds = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i], value = argv[i + 1];
    if (flag == "--ms") {
      ms.clear();
      for (const auto &v : split(value))
        ms.push_back(std::stoll(v));
    } else if (flag == "--iters") {
      env.iters = std::stoi(value);
    } else if (flag == "--rounds") {
      env.rounds = std::stoi(value);
    } else if (flag == "--shapes") {
      shapes = split(value);
    } else if (flag == "--tune") {
      tune_mode = value == "1";
    } else if (flag == "--block") {
      block_mode = value == "1";
    } else if (flag == "--tune-prefill") {
      tune_prefill_mode = value == "1";
    } else {
      std::fprintf(stderr,
                   "usage: %s [--ms a,b] [--iters n] [--rounds n] "
                   "[--shapes qkv,o,gate_up,down] [--tune 1] [--tune-prefill "
                   "1] [--block 1]\n",
                   argv[0]);
      return 2;
    }
  }
  std::sort(ms.begin(), ms.end());
  hipDeviceProp_t prop{};
  hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  int version = 0;
  hip_ok(hipStreamCreate(&env.stream), "hipStreamCreate");
  if (hipblasLtCreate(&env.lt) != HIPBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "qgemm_eval: hipblasLtCreate failed\n");
    return 2;
  }
  (void)hipblasLtGetVersion(env.lt, &version);
  hip_ok(hipMalloc(&env.workspace, kWorkspaceBytes), "hipMalloc workspace");
  hip_ok(hipEventCreate(&env.e0), "hipEventCreate");
  hip_ok(hipEventCreate(&env.e1), "hipEventCreate");
  std::printf("# device %s (%s), hipBLASLt %d, iters %d x rounds %d\n",
              prop.name, prop.gcnArchName, version, env.iters, env.rounds);
  Rng rng{7};
  if (tune_prefill_mode) {
    const bool ok = tune_prefill(env, rng, shapes);
    std::printf("qgemm_eval: tune-prefill %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
  }
  if (block_mode) {
    const bool ok = eval_block(env, rng, shapes, ms);
    std::printf("qgemm_eval: block %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
  }
  if (tune_mode) {
    const bool ok = tune(env, rng, shapes);
    std::printf("qgemm_eval: tune %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
  }
  bool all_ok = true;
  for (const LinearShape &s : kShapes) {
    if (std::find(shapes.begin(), shapes.end(), s.name) == shapes.end())
      continue;
    ShapeData d = make_shape(s, ms.back(), rng);
    for (const CandInfo &ci : kCands) {
      std::vector<uint16_t> row0_first;
      bool invariant = true;
      for (int64_t m : ms) {
        Prepared p;
        if (!prepare(env, ci, d, m, p)) {
          std::printf("qgemm shape=%s n=%lld k=%lld m=%lld cand=%s algos=0 "
                      "status=%d at=%s\n",
                      s.name, (long long)s.n, (long long)s.k, (long long)m,
                      ci.name, static_cast<int>(p.status),
                      p.failed != nullptr ? p.failed : "-");
          continue;
        }
        const double first = time_algo(env, ci, d, p, p.algos[0]);
        double best = first;
        size_t best_i = 0;
        for (size_t i = 1; i < p.algos.size(); ++i) {
          const double t = time_algo(env, ci, d, p, p.algos[i]);
          if (t < best || std::isnan(best)) {
            best = t;
            best_i = i;
          }
        }
        // Correctness and row 0 from the first answer.
        const hipblasStatus_t st = run(env, ci, d, p, p.algos[0], 0);
        hip_ok(hipStreamSynchronize(env.stream), "sync");
        std::vector<uint16_t> row0;
        const Check c = check(ci, d, m, row0);
        if (row0_first.empty())
          row0_first = row0;
        else if (row0 != row0_first)
          invariant = false;
        const bool ok = st == HIPBLAS_STATUS_SUCCESS && c.ok;
        if (ci.cand != Cand::Bf16)
          all_ok = all_ok && ok;
        std::printf("qgemm shape=%s n=%lld k=%lld m=%lld cand=%s algos=%zu "
                    "first_us=%.2f best_us=%.2f best_idx=%zu max_abs=%.3e "
                    "ref_max=%.3e %s\n",
                    s.name, (long long)s.n, (long long)s.k, (long long)m,
                    ci.name, p.algos.size(), first, best, best_i, c.max_abs,
                    c.ref_max,
                    ci.cand == Cand::Bf16 ? "baseline"
                    : ok                  ? "ok"
                                          : "MISMATCH");
      }
      std::printf("qgemm shape=%s cand=%s row0_invariant=%s\n", s.name, ci.name,
                  invariant ? "yes" : "no");
    }
    free_shape(d);
  }
  const struct {
    int32_t mode;
    const char *name;
  } modes[] = {{TURBINE_ACTQ_FP8_TOKEN, "token"},
               {TURBINE_ACTQ_FP8_GROUP128, "group128"},
               {TURBINE_ACTQ_FP8_TENSOR, "tensor"}};
  for (const auto &mode : modes) {
    for (int64_t cols : {3072, 8192}) {
      for (int64_t m : ms) {
        double us = 0;
        const bool ok = check_quant(mode.mode, m, cols, rng, env, &us);
        all_ok = all_ok && ok;
        std::printf("quant mode=%s cols=%lld m=%lld us=%.2f %s\n", mode.name,
                    (long long)cols, (long long)m, us,
                    ok ? "bit-exact" : "MISMATCH");
      }
    }
  }
  std::printf("qgemm_eval: %s\n", all_ok ? "ok" : "FAIL");
  return all_ok ? 0 : 1;
}
