// turbine_gemm_tune: regenerates a card's tuned GEMM table
// (kernels/rocm/tuning/<arch>/gemm.tsv, read by src/gemm_table.cpp).
//
//   turbine_gemm_tune --shapes kernels/rocm/tuning/gemm_shapes.txt \
//                     --out kernels/rocm/tuning/<arch>/gemm.tsv [--device 0]
//
// Every shape of the shapes file names its mode.
//
// Mode `speed` (tune_shape_speed): per bucket (previous m, m] of the shape's m
// list, the solution with the least total time at the bucket's bounds and
// middle, run as the library runs a speed row (hipblasLtMatmul, split-K as
// hipBLASLt chooses), agreeing with the heuristic's answer and deterministic;
// pinned when it beats the heuristic's per-m answers by more than 2 %, else the
// bucket's row says `heuristic`. Rows may depend on the batch.
//
// Mode `invariant` (tune_shape_invariant): a row's output must never depend on
// how many rows share its batch (batch invariance: the OLMoE c16 golden flip,
// decisions.md). It builds the exact problems libturbine_hip.so runs
// (src/gemm_problem.hpp, dense leading dimensions); candidates are the
// heuristic's solutions (up to 256) at
// every bucket's top m of the shape's m list, each run the way the library runs
// a pinned row: fresh from its solution index, through the hipBLASLt ext API
// with split-K off (by default hipBLASLt splits K for small problems, which
// makes a row's sums depend on m). They are kept when they
//   - support every tested m within the library's GEMM workspace,
//   - agree with the heuristic's first answer at the largest m to 1/64 of the
//     output's largest magnitude,
//   - are deterministic: three runs give bitwise-identical outputs, and
//   - are row-invariant: a heavy-tailed target row's output bits alone (m = 1)
//     equal its bits at positions 0, 1, 15, 16, 17, m/2 and m-1 of batches at
//     every bucket's bounds and middle.
// Invariant solutions giving the target row the same bits sum in the same
// order: they form a class, and the buckets of a shape may use different
// members of ONE class (e.g. a small tile for decode and a large one for
// prefill) while every row keeps its bits at every m. Members are timed with
// weights cycled through >= 512 MB of copies (a forward never finds a layer's
// weights in the caches) and BF16 random data at every bucket's top m; the
// class whose fastest member per bucket has the least weighted mean time
// relative to the heuristic's own per-m answer is pinned (5 interleaved rounds
// for the best 8 classes): --decode-weight (default 0.85, the decode share of
// the served engine time) spread over the buckets up to m = 64, the rest over
// the larger ones. A class more than --max-prefill-loss (default 0.05) slower
// than the heuristic at a bucket of m 1,024 to 2,048 (the served mixed steps)
// is not pinned, so pinning
// never trades prefill time (TTFT) for decode time. The file records the cost
// per bucket (`# cost` lines). A shape with no eligible invariant class gets a
// `heuristic` row (index -1: not row-invariant) and is reported.
//
// Mode `prefix` (the Llama shapes): speed rows over the decode-sized buckets
// (m <= 64), run by decode steps, and one invariant class over every bucket,
// run by the steps that prefill prompt tokens (TURBINE_OPTION_GEMM_PREFILL),
// so a prefix-reused prefill of a suffix reproduces the whole-prompt prefill.
// The class is scored for prefill steps (prefix_weights) and pinned even over
// the prefill guard: prefix reuse is exact only with one.
//
// Output rows: n, k, trans_b, c_dtype, m_max, solution_index, solution_name,
// mode, heuristic_us, tuned_us, shape (tab-separated, `#` header lines name the
// device, ROCm and hipBLASLt). Run it on an idle card: it needs the whole card
// for stable timings (on novanas under scripts/bench-lock.sh, GPU 0).
// TURBINE_TUNE_DEBUG=1 prints, per rejected candidate, the m and position where
// the target row first changed and the first differing byte.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-version.h>
#include <rocm-core/rocm_version.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gemm_problem.hpp"

namespace {

using turbine_hip::GemmProblem;
using turbine_hip::GemmShape;

constexpr size_t kWorkspaceBytes = 32u << 20; // = kGemmWorkspaceBytes
constexpr size_t kCycleBytes = 512u << 20;
// Classes of identical numerics re-timed in the interleaved rounds.
constexpr size_t kRefineClasses = 8;
constexpr int kRounds = 5;
// Largest m of a decode step (one row per sequence: max_seqs 64).
constexpr int64_t kDecodeMaxM = 64;
// --decode-weight default: the share of the served engine time spent in
// decode-only steps at the benchmark workload (profile 2026-09-27: Llama 85 %,
// OLMoE 91 %); the rest is spread over the larger (mixed and prefill) buckets.
double g_decode_weight = 0.85;
// Buckets from this m on are prefill-sized: a pinned class may be at most
// --max-prefill-loss slower than the heuristic's answer there (default 5 %),
// so pinning does not trade the prefill time (TTFT) for decode time.
constexpr int64_t kPrefillGuardM = 1024;
// ... up to this m: the mixed steps of the served configurations
// (scheduler.max_batch_tokens 2,048); larger buckets are scored, not guarded.
constexpr int64_t kPrefillGuardMaxM = 2048;
double g_max_prefill_loss = 0.05;
// Mode prefix: the share of the class score on the decode-sized buckets (a
// prefill step that small is a short prompt or a prefix-reused suffix).
constexpr double kPrefixShortWeight = 0.15;
// Rows besides the target in a solution's class signature (alternately plain
// and heavy-tailed; see the invariance check).
constexpr uint32_t kSignatureRows = 16;

[[noreturn]] void die(const std::string &msg) {
  std::fprintf(stderr, "turbine_gemm_tune: %s\n", msg.c_str());
  std::exit(1);
}

void hip_ok(hipError_t e, const char *what) {
  if (e != hipSuccess)
    die(std::string(what) + ": " + hipGetErrorString(e));
}

void blas_ok(hipblasStatus_t s, const char *what) {
  if (s != HIPBLAS_STATUS_SUCCESS)
    die(std::string(what) + ": hipBLAS status " + std::to_string(s));
}

// `invariant`: rows must not depend on the batch (the pinned class runs with
// split-K off); `speed`: the fastest solution per bucket, split-K allowed;
// `prefix`: both, speed rows for the decode-sized buckets (decode steps) and
// one invariant class for every bucket (steps that prefill prompt tokens).
enum class Mode { Invariant, Speed, Prefix };

struct ShapeSpec {
  std::string name;
  int64_t n, k;
  int32_t trans_b, c_dtype;
  std::vector<int64_t> ms;
  Mode mode;
};

const std::vector<int64_t> kDecodeM = {1, 2, 4, 8, 16, 32, 64};
const std::vector<int64_t> kPrefillM = {128, 256, 512, 1024, 2048, 4096, 8192};

std::vector<ShapeSpec> read_shapes(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    die("cannot read " + path);
  std::vector<ShapeSpec> out;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream cols(line);
    ShapeSpec s;
    std::string dtype, ms, mode;
    if (!(cols >> s.name >> s.n >> s.k >> s.trans_b >> dtype >> ms >> mode))
      die(path + ":" + std::to_string(lineno) + ": expected 7 columns");
    if (mode == "invariant")
      s.mode = Mode::Invariant;
    else if (mode == "speed")
      s.mode = Mode::Speed;
    else if (mode == "prefix")
      s.mode = Mode::Prefix;
    else
      die(path + ":" + std::to_string(lineno) +
          ": mode is invariant, speed or prefix, not " + mode);
    if (dtype == "bf16")
      s.c_dtype = TURBINE_DTYPE_BF16;
    else if (dtype == "f32")
      s.c_dtype = TURBINE_DTYPE_F32;
    else
      die(path + ":" + std::to_string(lineno) + ": c_dtype bf16 or f32");
    std::istringstream items(ms);
    std::string item;
    while (std::getline(items, item, ',')) {
      if (item == "decode")
        s.ms.insert(s.ms.end(), kDecodeM.begin(), kDecodeM.end());
      else if (item == "prefill")
        s.ms.insert(s.ms.end(), kPrefillM.begin(), kPrefillM.end());
      else
        s.ms.push_back(std::stoll(item));
    }
    std::sort(s.ms.begin(), s.ms.end());
    s.ms.erase(std::unique(s.ms.begin(), s.ms.end()), s.ms.end());
    out.push_back(s);
  }
  return out;
}

// BF16 bits of uniform values in about [-1, 1) * scale (xorshift).
void fill_bf16(std::vector<uint16_t> &v, uint32_t seed, float scale) {
  uint32_t x = seed * 2654435761u + 1;
  for (auto &h : v) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const float f = (static_cast<float>(x >> 8) / 8388608.0f - 1.0f) * scale;
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    h = static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  }
}

// Heavy-tailed BF16 values: fill_bf16, then every 97th element 30x larger.
// Real hidden states have outlier channels; their large partial sums cancel,
// so two summation orders round to different outputs far more often than on
// uniform data (a BF16 output hides most F32 order differences otherwise, and
// the invariance and class checks would be nearly blind).
void fill_heavy_bf16(std::vector<uint16_t> &v, uint32_t seed) {
  fill_bf16(v, seed, 1.0f);
  for (size_t i = 0; i < v.size(); i += 97) {
    uint32_t bits = static_cast<uint32_t>(v[i]) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    f *= 30.0f;
    std::memcpy(&bits, &f, 4);
    v[i] = static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  }
}

__device__ float load_out(const void *p, size_t i, int f32) {
  if (f32)
    return static_cast<const float *>(p)[i];
  return __uint_as_float(
      static_cast<uint32_t>(static_cast<const uint16_t *>(p)[i]) << 16);
}

// out[0] = max |a - b|, out[1] = max |b| as float bits (non-negative floats
// order like their bits); out[2] = count of bitwise-different elements.
__global__ void compare_kernel(const void *a, const void *b, size_t n, int f32,
                               uint32_t *out) {
  float diff = 0, mag = 0;
  uint32_t neq = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float x = load_out(a, i, f32), y = load_out(b, i, f32);
    diff = fmaxf(diff, fabsf(x - y));
    mag = fmaxf(mag, fabsf(y));
    neq += __float_as_uint(x) != __float_as_uint(y) ? 1u : 0u;
  }
  atomicMax(&out[0], __float_as_uint(diff));
  atomicMax(&out[1], __float_as_uint(mag));
  atomicAdd(&out[2], neq);
}

struct Comparison {
  float max_diff, max_ref;
  uint32_t differing;
};

// a against the reference b, elements of the output dtype.
Comparison compare(const void *a, const void *b, size_t n, int32_t c_dtype,
                   uint32_t *scratch, hipStream_t stream) {
  hip_ok(hipMemsetAsync(scratch, 0, 3 * sizeof(uint32_t), stream), "memset");
  compare_kernel<<<1024, 256, 0, stream>>>(
      a, b, n, c_dtype == TURBINE_DTYPE_F32 ? 1 : 0, scratch);
  hip_ok(hipGetLastError(), "compare_kernel");
  uint32_t h[3];
  hip_ok(hipMemcpyAsync(h, scratch, sizeof(h), hipMemcpyDeviceToHost, stream),
         "read comparison");
  hip_ok(hipStreamSynchronize(stream), "sync");
  Comparison c{};
  std::memcpy(&c.max_diff, &h[0], 4);
  std::memcpy(&c.max_ref, &h[1], 4);
  c.differing = h[2];
  return c;
}

struct Bench {
  hipblasLtHandle_t lt;
  hipStream_t stream;
  void *workspace;
  hipEvent_t e0, e1;
};

struct Operands {
  std::vector<void *> weights;
  void *act = nullptr;
  void *out = nullptr;
  void *out2 = nullptr;
  void *ref = nullptr;
  uint32_t *scratch = nullptr;
  size_t out_bytes = 0;
};

hipblasStatus_t run(Bench &b, GemmProblem &p, const hipblasLtMatmulAlgo_t &algo,
                    const void *w, const void *a, void *c) {
  const float alpha = 1.0f, beta = 0.0f;
  return hipblasLtMatmul(b.lt, p.desc.handle, &alpha, w, p.weight.handle, a,
                         p.act.handle, &beta, c, p.out.handle, c, p.out.handle,
                         &algo, b.workspace, kWorkspaceBytes, b.stream);
}

// Mean microseconds per call over iters calls, weights cycled.
double time_us(Bench &b, GemmProblem &p, const hipblasLtMatmulAlgo_t &algo,
               Operands &o, int iters) {
  hip_ok(hipEventRecord(b.e0, b.stream), "hipEventRecord");
  for (int i = 0; i < iters; ++i)
    blas_ok(run(b, p, algo, o.weights[i % o.weights.size()], o.act, o.out),
            "hipblasLtMatmul");
  hip_ok(hipEventRecord(b.e1, b.stream), "hipEventRecord");
  hip_ok(hipEventSynchronize(b.e1), "hipEventSynchronize");
  float ms = 0;
  hip_ok(hipEventElapsedTime(&ms, b.e0, b.e1), "hipEventElapsedTime");
  return 1000.0 * ms / iters;
}

// A candidate the way the library runs a pinned row (src/gemm_table.cpp
// pinned_gemm): through the hipBLASLt ext API with split-K off, so its sums do
// not depend on m. Checks the problem within the workspace; with run, also
// enqueues it.
hipblasStatus_t pinned(Bench &b, GemmProblem &p, hipblasLtMatmulAlgo_t &algo,
                       const void *w, const void *a, void *c, bool run) {
  const float alpha = 1.0f, beta = 0.0f;
  hipblaslt_ext::Gemm g(b.lt, p.desc.handle, &alpha, w, p.weight.handle, a,
                        p.act.handle, &beta, c, p.out.handle, c, p.out.handle);
  hipblaslt_ext::GemmTuning tuning;
  tuning.setSplitK(1);
  g.setMaxWorkspaceBytes(kWorkspaceBytes);
  size_t ws = 0;
  hipblasStatus_t st = g.isAlgoSupported(algo, tuning, ws);
  if (st != HIPBLAS_STATUS_SUCCESS || ws > kWorkspaceBytes)
    return st != HIPBLAS_STATUS_SUCCESS ? st : HIPBLAS_STATUS_NOT_SUPPORTED;
  if (!run)
    return HIPBLAS_STATUS_SUCCESS;
  st = g.initialize(algo, tuning, b.workspace, false, b.stream);
  return st != HIPBLAS_STATUS_SUCCESS ? st : g.run(b.stream);
}

bool pinned_supports(Bench &b, GemmProblem &p, hipblasLtMatmulAlgo_t &algo,
                     Operands &o) {
  return pinned(b, p, algo, o.weights[0], o.act, o.out, false) ==
         HIPBLAS_STATUS_SUCCESS;
}

// Mean microseconds per call of a pinned candidate over iters calls, weights
// cycled; the ext GEMMs are initialized before the timed loop, so it times the
// kernels, not the host.
double pinned_time_us(Bench &b, GemmProblem &p, hipblasLtMatmulAlgo_t algo,
                      Operands &o, int iters) {
  const float alpha = 1.0f, beta = 0.0f;
  hipblaslt_ext::GemmTuning tuning;
  tuning.setSplitK(1);
  std::vector<std::unique_ptr<hipblaslt_ext::Gemm>> gs;
  const size_t n = std::min<size_t>(o.weights.size(), iters);
  for (size_t i = 0; i < n; ++i) {
    gs.push_back(std::make_unique<hipblaslt_ext::Gemm>(
        b.lt, p.desc.handle, &alpha, o.weights[i], p.weight.handle, o.act,
        p.act.handle, &beta, o.out, p.out.handle, o.out, p.out.handle));
    gs.back()->setMaxWorkspaceBytes(kWorkspaceBytes);
    blas_ok(gs.back()->initialize(algo, tuning, b.workspace, false, b.stream),
            "pinned initialize");
  }
  hip_ok(hipEventRecord(b.e0, b.stream), "hipEventRecord");
  for (int i = 0; i < iters; ++i)
    blas_ok(gs[i % n]->run(b.stream), "pinned run");
  hip_ok(hipEventRecord(b.e1, b.stream), "hipEventRecord");
  hip_ok(hipEventSynchronize(b.e1), "hipEventSynchronize");
  float ms = 0;
  hip_ok(hipEventElapsedTime(&ms, b.e0, b.e1), "hipEventElapsedTime");
  return 1000.0 * ms / iters;
}

struct Row {
  const ShapeSpec *spec;
  int64_t m;
  // -1 and "heuristic": no solution passed; hipBLASLt's first answer per call.
  int index;
  std::string name;
  double heuristic_us, tuned_us;
  // An invariant row (split-K off, row-invariant), else a speed row.
  bool invariant;
};

// One timing point of a shape: a bucket's top m, its problem, the heuristic's
// first answer there (what the library runs without a table) and the fastest
// correct solution at that m, invariant or not (the cost of invariance).
struct Point {
  int64_t m;
  std::unique_ptr<GemmProblem> p;
  hipblasLtMatmulAlgo_t heuristic;
  std::vector<double> heuristic_rounds;
  double best_any_us = 0;
};

struct Candidate {
  int index;
  std::string name;
  hipblasLtMatmulAlgo_t algo;
  // Per point: the algorithm checked against that point's problem.
  std::vector<hipblasLtMatmulAlgo_t> at;
  // FNV-1a of the target row's output bits (equal at every tested m).
  uint64_t sig = 0;
  std::vector<double> quick;
  std::vector<std::vector<double>> rounds;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

hipblasLtMatmulAlgo_t heuristic_answer(Bench &b, GemmProblem &p,
                                       hipblasLtMatmulPreference_t pref) {
  hipblasLtMatmulHeuristicResult_t first{};
  int returned = 0;
  blas_ok(hipblasLtMatmulAlgoGetHeuristic(
              b.lt, p.desc.handle, p.weight.handle, p.act.handle, p.out.handle,
              p.out.handle, pref, 1, &first, &returned),
          "heuristic");
  if (returned < 1)
    die("no heuristic answer");
  return first.algo;
}

std::unique_ptr<GemmProblem> problem(const ShapeSpec &spec, int64_t m) {
  auto p = std::make_unique<GemmProblem>();
  if (p->make(GemmShape{m, spec.n, spec.k, spec.k, spec.k, spec.n, spec.trans_b,
                        spec.c_dtype}) != HIPBLAS_STATUS_SUCCESS)
    die(std::string("problem: ") + p->failed);
  return p;
}

bool supports(Bench &b, GemmProblem &p, hipblasLtMatmulAlgo_t &algo) {
  size_t ws = 0;
  const float alpha = 1.0f, beta = 0.0f;
  return hipblaslt_ext::matmulIsAlgoSupported(
             b.lt, p.desc.handle, &alpha, p.weight.handle, p.act.handle, &beta,
             p.out.handle, p.out.handle, algo, ws) == HIPBLAS_STATUS_SUCCESS &&
         ws <= kWorkspaceBytes;
}

// Positions of the target row in an m-row batch: first, second, around the
// 16-row tile edge, middle, last (hip_batch_invariance's positions).
std::vector<int64_t> positions(int64_t m) {
  std::vector<int64_t> v;
  for (int64_t p : {int64_t{0}, int64_t{1}, int64_t{15}, int64_t{16},
                    int64_t{17}, m / 2, m - 1})
    if (p < m)
      v.push_back(p);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

// The shape's one pinned solution: among the solutions that support every m of
// the shape, agree with the heuristic, are deterministic and are row-invariant
// (the target row's output bits are the same alone, m = 1, and at every
// position of batches of every tested m), the one with the least weighted mean
// time relative to the heuristic's own per-m answers over the shape's buckets.
// One solution for every m, so a row's result never depends on its batch.
// Weight of bucket point i of points (ascending m) in a shape's score: the
// decode weight spread evenly over the buckets up to kDecodeMaxM, the rest over
// the larger ones.
// Mode prefix: the decode-sized buckets (short prompts, prefix-reused
// suffixes) share kPrefixShortWeight evenly; the larger ones share the rest in
// proportion to their tokens, capped at kPrefillGuardMaxM (the served mixed
// steps; a larger step runs at the capped bucket's rate), so the score is the
// class's relative prefill time over a token-weighted mix of step sizes.
template <typename P> std::vector<double> prefix_weights(const P &points) {
  double decode = 0, tokens = 0;
  for (const auto &pt : points) {
    if (pt.m <= kDecodeMaxM)
      decode += 1;
    else
      tokens += static_cast<double>(std::min(pt.m, kPrefillGuardMaxM));
  }
  const double wd = tokens == 0 ? 1.0 : decode == 0 ? 0.0 : kPrefixShortWeight;
  std::vector<double> w;
  for (const auto &pt : points)
    w.push_back(pt.m <= kDecodeMaxM ? wd / decode
                                    : (1.0 - wd) *
                                          static_cast<double>(std::min(
                                              pt.m, kPrefillGuardMaxM)) /
                                          tokens);
  return w;
}

template <typename P>
std::vector<double> bucket_weights(const P &points, double decode_weight) {
  size_t decode = 0;
  for (const auto &pt : points)
    decode += pt.m <= kDecodeMaxM ? 1 : 0;
  const size_t other = points.size() - decode;
  const double wd = other == 0 ? 1.0 : decode == 0 ? 0.0 : decode_weight;
  std::vector<double> w;
  for (const auto &pt : points)
    w.push_back(pt.m <= kDecodeMaxM ? wd / static_cast<double>(decode)
                                    : (1.0 - wd) / static_cast<double>(other));
  return w;
}

// prefix: the class serves only the steps that prefill prompt tokens (decode
// steps run the shape's speed rows): its decode-sized buckets (short prompts
// and prefix-reused suffixes) weigh kPrefixShortWeight in the score, and the
// prefill guard does not apply: the shape must get a class (prefix reuse is
// exact only with one), so the best one is pinned and its cost recorded.
std::vector<Row> tune_shape_invariant(Bench &b, const ShapeSpec &spec,
                                      bool prefix) {
  const int64_t m_top = spec.ms.back();
  const size_t elem = spec.c_dtype == TURBINE_DTYPE_F32 ? 4 : 2;
  Operands o;
  const size_t w_elems = static_cast<size_t>(spec.n) * spec.k;
  const size_t copies =
      std::clamp<size_t>(kCycleBytes / (w_elems * 2) + 1, 2, 64);
  std::vector<uint16_t> host(w_elems);
  fill_bf16(host, static_cast<uint32_t>(spec.n * 31 + spec.k),
            1.0f / std::sqrt(static_cast<float>(spec.k)));
  for (size_t i = 0; i < copies; ++i) {
    void *w = nullptr;
    hip_ok(hipMalloc(&w, w_elems * 2), "hipMalloc weights");
    hip_ok(hipMemcpy(w, host.data(), w_elems * 2, hipMemcpyHostToDevice),
           "upload weights");
    o.weights.push_back(w);
  }
  std::vector<uint16_t> act(static_cast<size_t>(m_top) * spec.k);
  fill_bf16(act, static_cast<uint32_t>(spec.n * 7 + 3), 1.0f);
  hip_ok(hipMalloc(&o.act, act.size() * 2), "hipMalloc activations");
  hip_ok(hipMemcpy(o.act, act.data(), act.size() * 2, hipMemcpyHostToDevice),
         "upload activations");
  // The target row (distinct data) and a scratch copy of each row it
  // overwrites.
  std::vector<uint16_t> target(spec.k), target2(spec.k);
  fill_heavy_bf16(target, 977u);
  fill_heavy_bf16(target2, 1913u);
  o.out_bytes = static_cast<size_t>(m_top) * spec.n * elem;
  hip_ok(hipMalloc(&o.out, o.out_bytes), "hipMalloc output");
  hip_ok(hipMalloc(&o.out2, o.out_bytes), "hipMalloc output");
  hip_ok(hipMalloc(&o.ref, o.out_bytes), "hipMalloc reference");
  hip_ok(hipMalloc(&o.scratch, 3 * sizeof(uint32_t)), "hipMalloc scratch");

  turbine_hip::Preference pref;
  blas_ok(hipblasLtMatmulPreferenceCreate(&pref.handle), "preference");
  const uint64_t max_ws = kWorkspaceBytes;
  blas_ok(hipblasLtMatmulPreferenceSetAttribute(
              pref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
              sizeof(max_ws)),
          "preference workspace");

  // Timing points: every bucket's top m. Invariance m values: every bucket's
  // bounds and middle.
  std::vector<Point> points;
  std::vector<int64_t> inv_ms;
  int64_t prev = 0;
  for (int64_t m_hi : spec.ms) {
    Point pt{m_hi, problem(spec, m_hi)};
    pt.heuristic = heuristic_answer(b, *pt.p, pref.handle);
    points.push_back(std::move(pt));
    const int64_t lo = prev + 1;
    inv_ms.push_back(m_hi);
    if (lo < m_hi)
      inv_ms.push_back(lo);
    if (m_hi - lo >= 4)
      inv_ms.push_back((lo + m_hi) / 2);
    prev = m_hi;
  }
  std::sort(inv_ms.begin(), inv_ms.end(), std::greater<>());
  inv_ms.erase(std::unique(inv_ms.begin(), inv_ms.end()), inv_ms.end());
  std::vector<std::unique_ptr<GemmProblem>> inv_p;
  for (int64_t m : inv_ms)
    inv_p.push_back(problem(spec, m));

  // Candidates: the heuristic's lists at every point, deduplicated by name,
  // supported at every point and every invariance m.
  std::vector<Candidate> cands;
  for (auto &pt : points) {
    std::vector<hipblasLtMatmulHeuristicResult_t> all(256);
    int returned = 0;
    blas_ok(hipblasLtMatmulAlgoGetHeuristic(
                b.lt, pt.p->desc.handle, pt.p->weight.handle, pt.p->act.handle,
                pt.p->out.handle, pt.p->out.handle, pref.handle, 256,
                all.data(), &returned),
            "heuristic list");
    all.resize(returned);
    for (auto &r : all) {
      if (r.state != HIPBLAS_STATUS_SUCCESS)
        continue;
      std::string name = hipblaslt_ext::getSolutionNameFromAlgo(b.lt, r.algo);
      if (std::any_of(cands.begin(), cands.end(),
                      [&](const Candidate &c) { return c.name == name; }))
        continue;
      // The algorithm exactly as the library obtains a pinned row's
      // (gemm_table.cpp): fresh from its solution index, then checked per
      // problem. A heuristic list's entry can carry settings the heuristic
      // chose for that list's m, which the library never sees.
      std::vector<int> index{hipblaslt_ext::getIndexFromAlgo(r.algo)};
      std::vector<hipblasLtMatmulHeuristicResult_t> fresh;
      if (hipblaslt_ext::getAlgosFromIndex(b.lt, index, fresh) !=
              HIPBLAS_STATUS_SUCCESS ||
          fresh.empty() ||
          hipblaslt_ext::getSolutionNameFromAlgo(b.lt, fresh[0].algo) != name)
        continue;
      cands.push_back(Candidate{index[0], name, fresh[0].algo});
    }
  }
  const size_t offered = cands.size();
  int unsupported = 0, wrong = 0, nondet = 0, variant = 0;
  {
    std::vector<Candidate> kept;
    for (auto &c : cands) {
      bool ok = true;
      for (auto &p : inv_p) {
        hipblasLtMatmulAlgo_t a = c.algo;
        ok = ok && pinned_supports(b, *p, a, o);
      }
      for (auto &pt : points) {
        hipblasLtMatmulAlgo_t a = c.algo;
        ok = ok && pinned_supports(b, *pt.p, a, o);
        c.at.push_back(a);
      }
      if (ok)
        kept.push_back(std::move(c));
      else
        ++unsupported;
    }
    cands = std::move(kept);
  }

  // Correctness against the heuristic and determinism at the top m.
  {
    Point &top = points.back();
    const size_t out_elems = static_cast<size_t>(top.m) * spec.n;
    blas_ok(run(b, *top.p, top.heuristic, o.weights[0], o.act, o.ref),
            "reference");
    const float tol = std::max(
        compare(o.ref, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                .max_ref /
            64.0f,
        1e-6f);
    std::vector<Candidate> kept;
    for (auto &c : cands) {
      hipblasLtMatmulAlgo_t a = c.at.back();
      hip_ok(hipMemsetAsync(o.out, 0, o.out_bytes, b.stream), "memset");
      if (pinned(b, *top.p, a, o.weights[0], o.act, o.out, true) !=
              HIPBLAS_STATUS_SUCCESS ||
          compare(o.out, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                  .max_diff > tol) {
        ++wrong;
        continue;
      }
      bool same = true;
      for (int rep = 0; rep < 2 && same; ++rep) {
        hip_ok(hipMemsetAsync(o.out2, 0xff, o.out_bytes, b.stream), "memset");
        blas_ok(pinned(b, *top.p, a, o.weights[0], o.act, o.out2, true),
                "rerun");
        same =
            compare(o.out2, o.out, out_elems, spec.c_dtype, o.scratch, b.stream)
                .differing == 0;
      }
      if (!same) {
        ++nondet;
        continue;
      }
      kept.push_back(std::move(c));
    }
    cands = std::move(kept);
  }

  // Row invariance: the target row's output bits alone (m = 1) must equal its
  // bits at every position of every invariance m (largest first: the likeliest
  // to differ).
  {
    const size_t row_bytes = static_cast<size_t>(spec.n) * elem;
    std::vector<char> alone(row_bytes), got(row_bytes);
    std::vector<uint16_t> saved(spec.k);
    auto target_run = [&](GemmProblem &p, hipblasLtMatmulAlgo_t a, int64_t pos,
                          std::vector<char> &row) {
      char *arow = static_cast<char *>(o.act) + pos * spec.k * 2;
      hip_ok(hipMemcpy(saved.data(), arow, spec.k * 2, hipMemcpyDeviceToHost),
             "save row");
      hip_ok(hipMemcpy(arow, target.data(), spec.k * 2, hipMemcpyHostToDevice),
             "target row");
      hip_ok(hipDeviceSynchronize(), "sync before the invariance run");
      blas_ok(pinned(b, p, a, o.weights[0], o.act, o.out, true),
              "invariance run");
      hip_ok(hipStreamSynchronize(b.stream), "sync after the invariance run");
      hip_ok(hipMemcpy(row.data(), static_cast<char *>(o.out) + pos * row_bytes,
                       row_bytes, hipMemcpyDeviceToHost),
             "read row");
      hip_ok(hipMemcpy(arow, saved.data(), spec.k * 2, hipMemcpyHostToDevice),
             "restore row");
    };
    std::unique_ptr<GemmProblem> one = problem(spec, 1);
    std::vector<Candidate> kept;
    for (auto &c : cands) {
      hipblasLtMatmulAlgo_t a1 = c.algo;
      if (!pinned_supports(b, *one, a1, o)) {
        ++unsupported;
        continue;
      }
      target_run(*one, a1, 0, alone);
      // The class signature: the target row's bits and kSignatureRows more
      // rows' (heavy-tailed and plain, distinct seeds), each alone. Two
      // summation orders agree on a few rows by chance often enough that a
      // two-row signature put solutions of different orders into one class
      // (the tp 2 shapes, P5 Task 29: members of one "class" gave a row other
      // bits at other m).
      c.sig = 1469598103934665603ull;
      for (char ch : alone)
        c.sig = (c.sig ^ static_cast<uint8_t>(ch)) * 1099511628211ull;
      for (uint32_t s = 0; s < kSignatureRows; ++s) {
        std::vector<char> extra(row_bytes);
        // s = 0: target2 as filled above (seed 1913, the historical second
        // row); it is refilled after the loop.
        if (s > 0 && s % 2 == 1)
          fill_bf16(target2, 4099u + s, 1.0f);
        else if (s > 0)
          fill_heavy_bf16(target2, 4099u + s);
        std::swap(target, target2);
        target_run(*one, a1, 0, extra);
        std::swap(target, target2);
        for (char ch : extra)
          c.sig = (c.sig ^ static_cast<uint8_t>(ch)) * 1099511628211ull;
      }
      fill_heavy_bf16(target2, 1913u);
      bool invariant = true;
      for (size_t i = 0; i < inv_ms.size() && invariant; ++i) {
        hipblasLtMatmulAlgo_t a = c.algo;
        (void)pinned_supports(b, *inv_p[i], a, o);
        for (int64_t pos : positions(inv_ms[i])) {
          target_run(*inv_p[i], a, pos, got);
          if (std::memcmp(alone.data(), got.data(), row_bytes) != 0) {
            if (std::getenv("TURBINE_TUNE_DEBUG") != nullptr) {
              size_t diff = 0, first = 0;
              for (size_t e = 0; e < row_bytes; ++e)
                if (alone[e] != got[e] && diff++ == 0)
                  first = e;
              std::fprintf(stderr,
                           "debug: first differing byte %zu: %02x vs %02x\n",
                           first, static_cast<uint8_t>(alone[first]),
                           static_cast<uint8_t>(got[first]));
              std::fprintf(stderr,
                           "debug: %s variant at m=%lld pos=%lld (%zu of %zu "
                           "bytes)\n",
                           c.name.substr(0, 70).c_str(),
                           static_cast<long long>(inv_ms[i]),
                           static_cast<long long>(pos), diff, row_bytes);
            }
            invariant = false;
            break;
          }
        }
      }
      if (invariant)
        kept.push_back(std::move(c));
      else
        ++variant;
    }
    cands = std::move(kept);
  }

  // Timing: the heuristic's own answer at every point, and the best solution
  // there regardless of invariance (quick pass over the full candidate list is
  // too slow; the heuristic's list at each point is timed briefly instead).
  for (auto &pt : points) {
    (void)time_us(b, *pt.p, pt.heuristic, o, 2);
    const double h = time_us(b, *pt.p, pt.heuristic, o, 5);
    const int iters = std::clamp(static_cast<int>(1000.0 / h), 3, 50);
    std::vector<hipblasLtMatmulHeuristicResult_t> all(64);
    int returned = 0;
    blas_ok(hipblasLtMatmulAlgoGetHeuristic(
                b.lt, pt.p->desc.handle, pt.p->weight.handle, pt.p->act.handle,
                pt.p->out.handle, pt.p->out.handle, pref.handle, 64, all.data(),
                &returned),
            "heuristic list");
    pt.best_any_us = time_us(b, *pt.p, pt.heuristic, o, iters);
    for (int i = 0; i < returned; ++i) {
      if (all[i].state != HIPBLAS_STATUS_SUCCESS ||
          !supports(b, *pt.p, all[i].algo))
        continue;
      pt.best_any_us =
          std::min(pt.best_any_us, time_us(b, *pt.p, all[i].algo, o, iters));
    }
  }
  // Solutions whose target row has the same bits (the same signature) at every
  // tested m sum every output in the same order: they form one class, and a
  // shape may use any member of one class per bucket without a row's result
  // depending on its batch. Quick per-bucket times of every invariant
  // candidate, then per class the fastest member per bucket.
  const std::vector<double> weight =
      prefix ? prefix_weights(points) : bucket_weights(points, g_decode_weight);
  std::vector<double> h_quick(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    Point &pt = points[i];
    (void)time_us(b, *pt.p, pt.heuristic, o, 2);
    h_quick[i] = time_us(b, *pt.p, pt.heuristic, o, 5);
  }
  for (auto &c : cands) {
    c.quick.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      Point &pt = points[i];
      const int iters =
          std::clamp(static_cast<int>(1000.0 / h_quick[i]), 3, 50);
      (void)pinned_time_us(b, *pt.p, c.at[i], o, 1);
      c.quick[i] = pinned_time_us(b, *pt.p, c.at[i], o, iters);
    }
  }
  struct Class {
    uint64_t sig;
    // Per bucket: the index into cands of the member run there.
    std::vector<size_t> member;
    double score = 0;
  };
  std::vector<Class> classes;
  for (size_t j = 0; j < cands.size(); ++j) {
    auto it = std::find_if(classes.begin(), classes.end(), [&](const Class &k) {
      return k.sig == cands[j].sig;
    });
    if (it == classes.end()) {
      classes.push_back(
          Class{cands[j].sig, std::vector<size_t>(points.size(), j)});
      continue;
    }
    for (size_t i = 0; i < points.size(); ++i)
      if (cands[j].quick[i] < cands[it->member[i]].quick[i])
        it->member[i] = j;
  }
  // A class over the prefill guard at any prefill-sized bucket is scored
  // after every class within it.
  auto guarded = [&](const Class &k, auto time_of) {
    if (prefix)
      return true;
    for (size_t i = 0; i < points.size(); ++i)
      if (points[i].m >= kPrefillGuardM && points[i].m <= kPrefillGuardMaxM &&
          time_of(k, i) > 1.0 + g_max_prefill_loss)
        return false;
    return true;
  };
  auto quick_ratio = [&](const Class &k, size_t i) {
    return cands[k.member[i]].quick[i] / h_quick[i];
  };
  for (auto &k : classes) {
    k.score = 0;
    for (size_t i = 0; i < points.size(); ++i)
      k.score += weight[i] * quick_ratio(k, i);
    // Quick timings are noisy: a small margin before the refined check.
    if (!guarded(k, [&](const Class &c, size_t i) {
          return quick_ratio(c, i) - 0.03;
        }))
      k.score += 1000.0;
  }
  std::sort(classes.begin(), classes.end(),
            [](const Class &x, const Class &y) { return x.score < y.score; });
  for (size_t ci = 0; ci < classes.size() && ci < 8; ++ci) {
    double dec = 0, pre = 0, nd = 0, np = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      const double r = cands[classes[ci].member[i]].quick[i] / h_quick[i];
      if (points[i].m <= kDecodeMaxM) {
        dec += r;
        nd += 1;
      } else {
        pre += r;
        np += 1;
      }
    }
    std::fprintf(stderr,
                 "turbine_gemm_tune: %-14s class %zu: decode x%.3f prefill "
                 "x%.3f (quick, vs heuristic)\n",
                 spec.name.c_str(), ci, nd > 0 ? dec / nd : 0.0,
                 np > 0 ? pre / np : 0.0);
  }
  if (classes.size() > kRefineClasses)
    classes.resize(kRefineClasses);

  // Refinement: 5 interleaved rounds per bucket of the heuristic's answer and
  // every member the best classes run there; the class with the least
  // weighted mean relative time is pinned.
  std::vector<std::vector<size_t>> timed(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    for (const auto &k : classes)
      if (std::find(timed[i].begin(), timed[i].end(), k.member[i]) ==
          timed[i].end())
        timed[i].push_back(k.member[i]);
  }
  for (auto &c : cands)
    c.rounds.assign(points.size(), {});
  for (int round = 0; round < kRounds; ++round) {
    for (size_t i = 0; i < points.size(); ++i) {
      Point &pt = points[i];
      const int iters =
          std::clamp(static_cast<int>(3000.0 / h_quick[i]), 5, 200);
      pt.heuristic_rounds.push_back(time_us(b, *pt.p, pt.heuristic, o, iters));
      for (size_t j : timed[i])
        cands[j].rounds[i].push_back(
            pinned_time_us(b, *pt.p, cands[j].at[i], o, iters));
    }
  }
  auto refined_ratio = [&](const Class &k, size_t i) {
    return median(cands[k.member[i]].rounds[i]) /
           median(points[i].heuristic_rounds);
  };
  // Every refined class's per-bucket time relative to the heuristic's (the
  // record of what the pinned class costs against the alternatives).
  for (size_t ci = 0; ci < classes.size(); ++ci) {
    std::string line;
    double score = 0;
    for (size_t i = 0; i < points.size(); ++i) {
      char cell[32];
      std::snprintf(cell, sizeof(cell), " %lld:%.3f",
                    static_cast<long long>(points[i].m),
                    refined_ratio(classes[ci], i));
      line += cell;
      score += weight[i] * refined_ratio(classes[ci], i);
    }
    std::fprintf(stderr,
                 "turbine_gemm_tune: %-14s refined class %zu score %.3f:%s\n",
                 spec.name.c_str(), ci, score, line.c_str());
  }
  const Class *best = nullptr;
  bool guard_failed = false;
  for (auto &k : classes) {
    k.score = 0;
    for (size_t i = 0; i < points.size(); ++i)
      k.score += weight[i] * refined_ratio(k, i);
    if (!guarded(k, refined_ratio)) {
      guard_failed = true;
      continue;
    }
    if (best == nullptr || k.score < best->score)
      best = &k;
  }
  if (best == nullptr && guard_failed)
    std::fprintf(
        stderr,
        "turbine_gemm_tune: %-14s every row-invariant class is more "
        "than %.0f%% slower than the heuristic at m in [%lld, %lld]: the "
        "shape keeps the heuristic (not row-invariant)\n",
        spec.name.c_str(), 100.0 * g_max_prefill_loss,
        static_cast<long long>(kPrefillGuardM),
        static_cast<long long>(kPrefillGuardMaxM));

  size_t members = 0;
  if (best != nullptr) {
    std::vector<size_t> distinct = best->member;
    std::sort(distinct.begin(), distinct.end());
    members = std::unique(distinct.begin(), distinct.end()) - distinct.begin();
  }
  std::fprintf(
      stderr,
      "turbine_gemm_tune: %-14s candidates=%zu (rejected: %d "
      "unsupported, %d wrong, %d nondeterministic, %d not "
      "row-invariant); %s\n",
      spec.name.c_str(), offered, unsupported, wrong, nondet, variant,
      best != nullptr
          ? ("pinned class of " + std::to_string(members) + " solution(s)")
                .c_str()
          : "NONE invariant: heuristic kept, not row-invariant");
  std::vector<Row> rows;
  for (size_t i = 0; i < points.size(); ++i) {
    const Point &pt = points[i];
    const double h = median(pt.heuristic_rounds);
    const Candidate *c = best != nullptr ? &cands[best->member[i]] : nullptr;
    const double t = c != nullptr ? median(c->rounds[i]) : h;
    std::fprintf(stderr,
                 "turbine_gemm_tune: %-14s m=%-5lld heuristic %8.1f us  "
                 "pinned %8.1f us (%+6.1f%%)  fastest any %8.1f us  %s\n",
                 spec.name.c_str(), static_cast<long long>(pt.m), h, t,
                 100.0 * (t - h) / h, pt.best_any_us,
                 c != nullptr ? c->name.substr(0, 60).c_str() : "heuristic");
    rows.push_back(Row{&spec, pt.m, c != nullptr ? c->index : -1,
                       c != nullptr ? c->name : std::string("heuristic"), h, t,
                       true});
  }

  for (void *w : o.weights)
    (void)hipFree(w);
  (void)hipFree(o.act);
  (void)hipFree(o.out);
  (void)hipFree(o.out2);
  (void)hipFree(o.ref);
  (void)hipFree(o.scratch);
  return rows;
}

// Speed mode (a shape of a model that does not need batch invariance): per
// bucket (previous m, m], the solution with the least total time over the
// bucket's sample m values (its bounds and middle), run the way the library
// runs a speed row (hipblasLtMatmul, hipBLASLt's own split-K), pinned when it
// beats the heuristic's own per-m answers by more than 2 %; otherwise the
// bucket's row says `heuristic`. Candidates must agree with the heuristic and
// be deterministic; they need not be row-invariant.
std::vector<Row> tune_shape_speed(Bench &b, const ShapeSpec &spec) {
  constexpr double kMinGain = 0.02;
  constexpr size_t kRefineSpeed = 8;
  const int64_t m_top = spec.ms.back();
  const size_t elem = spec.c_dtype == TURBINE_DTYPE_F32 ? 4 : 2;
  Operands o;
  const size_t w_elems = static_cast<size_t>(spec.n) * spec.k;
  const size_t copies =
      std::clamp<size_t>(kCycleBytes / (w_elems * 2) + 1, 2, 64);
  std::vector<uint16_t> host(w_elems);
  fill_bf16(host, static_cast<uint32_t>(spec.n * 31 + spec.k),
            1.0f / std::sqrt(static_cast<float>(spec.k)));
  for (size_t i = 0; i < copies; ++i) {
    void *w = nullptr;
    hip_ok(hipMalloc(&w, w_elems * 2), "hipMalloc weights");
    hip_ok(hipMemcpy(w, host.data(), w_elems * 2, hipMemcpyHostToDevice),
           "upload weights");
    o.weights.push_back(w);
  }
  std::vector<uint16_t> act(static_cast<size_t>(m_top) * spec.k);
  fill_heavy_bf16(act, static_cast<uint32_t>(spec.n * 7 + 3));
  hip_ok(hipMalloc(&o.act, act.size() * 2), "hipMalloc activations");
  hip_ok(hipMemcpy(o.act, act.data(), act.size() * 2, hipMemcpyHostToDevice),
         "upload activations");
  o.out_bytes = static_cast<size_t>(m_top) * spec.n * elem;
  hip_ok(hipMalloc(&o.out, o.out_bytes), "hipMalloc output");
  hip_ok(hipMalloc(&o.out2, o.out_bytes), "hipMalloc output");
  hip_ok(hipMalloc(&o.ref, o.out_bytes), "hipMalloc reference");
  hip_ok(hipMalloc(&o.scratch, 3 * sizeof(uint32_t)), "hipMalloc scratch");
  turbine_hip::Preference pref;
  blas_ok(hipblasLtMatmulPreferenceCreate(&pref.handle), "preference");
  const uint64_t max_ws = kWorkspaceBytes;
  blas_ok(hipblasLtMatmulPreferenceSetAttribute(
              pref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
              sizeof(max_ws)),
          "preference workspace");

  struct SpeedCand {
    int index;
    std::string name;
    std::vector<hipblasLtMatmulAlgo_t> at; // per sample, checked
    double quick = 0;
    std::vector<std::vector<double>> rounds;
  };
  std::vector<Row> rows;
  int64_t prev = 0;
  for (const int64_t m_hi : spec.ms) {
    const int64_t lo = prev + 1;
    prev = m_hi;
    std::vector<int64_t> ms{m_hi};
    if (lo < m_hi)
      ms.push_back(lo);
    if (m_hi - lo >= 4)
      ms.push_back((lo + m_hi) / 2);
    std::vector<std::unique_ptr<GemmProblem>> ps;
    std::vector<hipblasLtMatmulAlgo_t> heur;
    for (int64_t m : ms) {
      ps.push_back(problem(spec, m));
      heur.push_back(heuristic_answer(b, *ps.back(), pref.handle));
    }
    // Candidates: the heuristic's list at the top m, fresh from their
    // solution index (as the library obtains a pinned row's), checked at every
    // sample.
    std::vector<hipblasLtMatmulHeuristicResult_t> all(256);
    int returned = 0;
    blas_ok(hipblasLtMatmulAlgoGetHeuristic(
                b.lt, ps[0]->desc.handle, ps[0]->weight.handle,
                ps[0]->act.handle, ps[0]->out.handle, ps[0]->out.handle,
                pref.handle, 256, all.data(), &returned),
            "heuristic list");
    all.resize(returned);
    std::vector<SpeedCand> cands;
    for (auto &r : all) {
      if (r.state != HIPBLAS_STATUS_SUCCESS)
        continue;
      const std::string name =
          hipblaslt_ext::getSolutionNameFromAlgo(b.lt, r.algo);
      if (std::any_of(cands.begin(), cands.end(),
                      [&](const SpeedCand &c) { return c.name == name; }))
        continue;
      std::vector<int> index{hipblaslt_ext::getIndexFromAlgo(r.algo)};
      std::vector<hipblasLtMatmulHeuristicResult_t> fresh;
      if (hipblaslt_ext::getAlgosFromIndex(b.lt, index, fresh) !=
              HIPBLAS_STATUS_SUCCESS ||
          fresh.empty())
        continue;
      SpeedCand c{index[0], name};
      bool ok = true;
      for (auto &p : ps) {
        hipblasLtMatmulAlgo_t a = fresh[0].algo;
        ok = ok && supports(b, *p, a);
        c.at.push_back(a);
      }
      if (ok)
        cands.push_back(std::move(c));
    }
    // Correctness and determinism at the top m.
    int wrong = 0, nondet = 0;
    {
      const size_t out_elems = static_cast<size_t>(m_hi) * spec.n;
      const size_t bytes = out_elems * elem;
      blas_ok(run(b, *ps[0], heur[0], o.weights[0], o.act, o.ref), "reference");
      const float tol = std::max(
          compare(o.ref, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                  .max_ref /
              64.0f,
          1e-6f);
      std::vector<SpeedCand> kept;
      for (auto &c : cands) {
        hip_ok(hipMemsetAsync(o.out, 0, bytes, b.stream), "memset");
        if (run(b, *ps[0], c.at[0], o.weights[0], o.act, o.out) !=
                HIPBLAS_STATUS_SUCCESS ||
            compare(o.out, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                    .max_diff > tol) {
          ++wrong;
          continue;
        }
        bool same = true;
        for (int rep = 0; rep < 2 && same; ++rep) {
          hip_ok(hipMemsetAsync(o.out2, 0xff, bytes, b.stream), "memset");
          blas_ok(run(b, *ps[0], c.at[0], o.weights[0], o.act, o.out2),
                  "rerun");
          same = compare(o.out2, o.out, out_elems, spec.c_dtype, o.scratch,
                         b.stream)
                     .differing == 0;
        }
        if (!same) {
          ++nondet;
          continue;
        }
        kept.push_back(std::move(c));
      }
      cands = std::move(kept);
    }
    // Quick pass at the top m, then interleaved rounds at every sample.
    (void)time_us(b, *ps[0], heur[0], o, 3);
    const double h_est = time_us(b, *ps[0], heur[0], o, 5);
    const int quick_iters = std::clamp(static_cast<int>(1000.0 / h_est), 3, 50);
    for (auto &c : cands) {
      (void)time_us(b, *ps[0], c.at[0], o, 1);
      c.quick = time_us(b, *ps[0], c.at[0], o, quick_iters);
    }
    std::sort(cands.begin(), cands.end(),
              [](const SpeedCand &x, const SpeedCand &y) {
                return x.quick < y.quick;
              });
    if (cands.size() > kRefineSpeed)
      cands.resize(kRefineSpeed);
    std::vector<std::vector<double>> h_rounds(ps.size());
    for (auto &c : cands)
      c.rounds.resize(ps.size());
    for (int round = 0; round < kRounds; ++round) {
      for (size_t si = 0; si < ps.size(); ++si) {
        const double est = time_us(b, *ps[si], heur[si], o, 2);
        const int iters = std::clamp(static_cast<int>(3000.0 / est), 5, 200);
        h_rounds[si].push_back(time_us(b, *ps[si], heur[si], o, iters));
        for (auto &c : cands)
          c.rounds[si].push_back(time_us(b, *ps[si], c.at[si], o, iters));
      }
    }
    double base = 0;
    for (auto &r : h_rounds)
      base += median(r);
    const SpeedCand *best = nullptr;
    double best_total = 0;
    for (const auto &c : cands) {
      double total = 0;
      for (const auto &r : c.rounds)
        total += median(r);
      if (best == nullptr || total < best_total) {
        best = &c;
        best_total = total;
      }
    }
    const bool pin = best != nullptr && best_total < (1.0 - kMinGain) * base;
    const double h_top = median(h_rounds[0]);
    const double t_top = pin ? median(best->rounds[0]) : h_top;
    std::fprintf(stderr,
                 "turbine_gemm_tune: %-14s speed m=(%lld,%lld] candidates=%zu "
                 "(rejected: %d wrong, %d nondeterministic) heuristic %.1f us "
                 "-> %.1f us (%+.1f%%) at m=%lld %s\n",
                 spec.name.c_str(), static_cast<long long>(lo - 1),
                 static_cast<long long>(m_hi), cands.size(), wrong, nondet,
                 h_top, t_top, 100.0 * (t_top - h_top) / h_top,
                 static_cast<long long>(m_hi), pin ? "" : "(heuristic kept)");
    rows.push_back(Row{&spec, m_hi, pin ? best->index : -1,
                       pin ? best->name : std::string("heuristic"), h_top,
                       t_top, false});
  }
  for (void *w : o.weights)
    (void)hipFree(w);
  (void)hipFree(o.act);
  (void)hipFree(o.out);
  (void)hipFree(o.out2);
  (void)hipFree(o.ref);
  (void)hipFree(o.scratch);
  return rows;
}

} // namespace

int main(int argc, char **argv) {
  std::string shapes_path, out_path;
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--shapes" && i + 1 < argc)
      shapes_path = argv[++i];
    else if (a == "--out" && i + 1 < argc)
      out_path = argv[++i];
    else if (a == "--device" && i + 1 < argc)
      device = std::atoi(argv[++i]);
    else if (a == "--decode-weight" && i + 1 < argc)
      g_decode_weight = std::atof(argv[++i]);
    else if (a == "--max-prefill-loss" && i + 1 < argc)
      g_max_prefill_loss = std::atof(argv[++i]);
    else
      die("usage: turbine_gemm_tune --shapes <gemm_shapes.txt> --out "
          "<tuning/<arch>/gemm.tsv> [--device <n>] [--decode-weight <0..1>] "
          "[--max-prefill-loss <fraction>]");
  }
  if (shapes_path.empty() || out_path.empty())
    die("--shapes and --out are required");
  const std::vector<ShapeSpec> shapes = read_shapes(shapes_path);

  hip_ok(hipSetDevice(device), "hipSetDevice");
  hipDeviceProp_t props{};
  hip_ok(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
  const std::string full(props.gcnArchName);
  const std::string arch = full.substr(0, full.find(':'));

  Bench b{};
  blas_ok(hipblasLtCreate(&b.lt), "hipblasLtCreate");
  hip_ok(hipStreamCreate(&b.stream), "hipStreamCreate");
  hip_ok(hipMalloc(&b.workspace, kWorkspaceBytes), "hipMalloc workspace");
  hip_ok(hipEventCreate(&b.e0), "hipEventCreate");
  hip_ok(hipEventCreate(&b.e1), "hipEventCreate");

  std::vector<Row> rows;
  for (const auto &s : shapes) {
    std::vector<Row> shape_rows;
    switch (s.mode) {
    case Mode::Invariant:
      shape_rows = tune_shape_invariant(b, s, false);
      break;
    case Mode::Speed:
      shape_rows = tune_shape_speed(b, s);
      break;
    case Mode::Prefix: {
      // Decode steps have at most kDecodeMaxM rows (one per sequence).
      ShapeSpec decode = s;
      decode.ms.erase(std::remove_if(decode.ms.begin(), decode.ms.end(),
                                     [](int64_t m) { return m > kDecodeMaxM; }),
                      decode.ms.end());
      shape_rows = tune_shape_speed(b, decode);
      for (Row &r : shape_rows)
        r.spec = &s;
      std::vector<Row> inv = tune_shape_invariant(b, s, true);
      shape_rows.insert(shape_rows.end(), inv.begin(), inv.end());
      break;
    }
    }
    rows.insert(rows.end(), shape_rows.begin(), shape_rows.end());
  }

  std::ofstream out(out_path);
  if (!out)
    die("cannot write " + out_path);
  char date[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(date, sizeof(date), "%Y-%m-%d", std::gmtime(&now));
  out << "# Tuned GEMM table for " << arch << " (" << props.name
      << "), written by turbine_gemm_tune on " << date << "\n"
      << "# ROCm " << ROCM_VERSION_MAJOR << "." << ROCM_VERSION_MINOR << "."
      << ROCM_VERSION_PATCH << ", hipBLASLt " << HIPBLASLT_VERSION_MAJOR << "."
      << HIPBLASLT_VERSION_MINOR << "." << HIPBLASLT_VERSION_PATCH
      << "; shapes " << shapes_path << "\n"
      << "# A row serves m in (previous row's m_max, m_max] of its shape; "
         "regenerate with docs/extending/card-family.md (Tuned GEMM table).\n"
      << "# mode invariant: one numerics class per shape, row-invariant at "
         "every m, run with split-K off; mode speed: the fastest solution per "
         "bucket, split-K allowed. A shape with both: decode steps run its "
         "speed "
         "rows, prefill steps its invariant rows.\n"
      << "# Cost per bucket top m, microseconds, heuristic's per-m answer -> "
         "pinned:\n";
  for (const Row &r : rows) {
    char line[160];
    std::snprintf(line, sizeof(line),
                  "# cost %-14s %-9s m=%-5lld %9.1f -> %9.1f (%+.1f%%)\n",
                  r.spec->name.c_str(), r.invariant ? "invariant" : "speed",
                  static_cast<long long>(r.m), r.heuristic_us, r.tuned_us,
                  r.heuristic_us > 0
                      ? 100.0 * (r.tuned_us - r.heuristic_us) / r.heuristic_us
                      : 0.0);
    out << line;
  }
  out << "# n\tk\ttrans_b\tc_dtype\tm_max\tsolution_index\tsolution_name\t"
         "mode\theuristic_us\ttuned_us\tshape\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row &r = rows[i];
    // Merge into the next bucket of the same shape when it pins the same
    // solution: that row then serves this bucket too.
    if (i + 1 < rows.size() && rows[i + 1].spec == r.spec &&
        rows[i + 1].invariant == r.invariant && rows[i + 1].name == r.name)
      continue;
    char us[64];
    std::snprintf(us, sizeof(us), "%.1f\t%.1f", r.heuristic_us, r.tuned_us);
    out << r.spec->n << "\t" << r.spec->k << "\t" << r.spec->trans_b << "\t"
        << (r.spec->c_dtype == TURBINE_DTYPE_F32 ? "f32" : "bf16") << "\t"
        << r.m << "\t" << r.index << "\t" << r.name << "\t"
        << (r.invariant ? "invariant" : "speed") << "\t" << us << "\t"
        << r.spec->name << "\n";
  }
  std::fprintf(stderr, "turbine_gemm_tune: wrote %s\n", out_path.c_str());
  return 0;
}
