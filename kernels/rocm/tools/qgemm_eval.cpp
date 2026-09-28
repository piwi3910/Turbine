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
    } else {
      std::fprintf(stderr,
                   "usage: %s [--ms a,b] [--iters n] [--rounds n] "
                   "[--shapes qkv,o,gate_up,down]\n",
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
  bool all_ok = true;
  Rng rng{7};
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
