// turbine_gemm_tune: regenerates a card's tuned GEMM table
// (kernels/rocm/tuning/<arch>/gemm.tsv, read by src/gemm_table.cpp).
//
//   turbine_gemm_tune --shapes kernels/rocm/tuning/gemm_shapes.txt \
//                     --out kernels/rocm/tuning/<arch>/gemm.tsv [--device 0]
//
// For every shape of the shapes file and every bucket (previous m, m] of its
// m list, it builds the exact hipBLASLt problems libturbine_hip.so runs
// (src/gemm_problem.hpp, dense leading dimensions) at the bucket's sample m
// values (its bounds and middle), takes as candidates up to 256 heuristic
// solutions at the top m plus each sample's own first answer, and keeps those
// that
//   - support every sample within the library's GEMM workspace,
//   - agree with the heuristic's first answer (what the library runs without a
//     table) to 1/64 of the output's largest magnitude, and
//   - are deterministic: three runs give bitwise-identical outputs (solutions
//     that split K with atomics would make a decode graph replay or a golden
//     run irreproducible).
// Each survivor is timed with its weights cycled through >= 512 MB of copies
// (a forward never finds a layer's weights in the caches), BF16 random data;
// the 8 fastest at the top m are re-timed at every sample in 5 interleaved
// rounds, the heuristic's own per-m answer alongside. The candidate with the
// least total of medians is pinned only when it beats the heuristic's total by
// more than 2 %; otherwise the bucket's row says `heuristic` (index -1): the
// library then asks hipBLASLt per call, as without a table. Consecutive
// buckets of a shape with the same outcome are merged.
//
// Output rows: n, k, trans_b, c_dtype, m_max, solution_index, solution_name,
// heuristic_us, tuned_us (tab-separated, `#` header lines name the device,
// ROCm and hipBLASLt). Run it on an idle card: it needs the whole card for
// stable timings (on novanas under scripts/bench-lock.sh, GPU 0).
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
constexpr int kRefine = 8;
constexpr int kRounds = 5;
constexpr double kMinGain = 0.02;

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

struct ShapeSpec {
  std::string name;
  int64_t n, k;
  int32_t trans_b, c_dtype;
  std::vector<int64_t> ms;
  // Buckets up to this m keep the heuristic without being tuned (`keep:<m>`).
  int64_t keep_to = 0;
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
    std::string dtype, ms;
    if (!(cols >> s.name >> s.n >> s.k >> s.trans_b >> dtype >> ms))
      die(path + ":" + std::to_string(lineno) + ": expected 6 columns");
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
      else if (item.rfind("keep:", 0) == 0)
        s.keep_to = std::stoll(item.substr(5));
      else
        s.ms.push_back(std::stoll(item));
    }
    if (s.keep_to > 0)
      s.ms.push_back(s.keep_to);
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

struct Row {
  const ShapeSpec *spec;
  int64_t m;
  // -1 and "heuristic": the bucket keeps hipBLASLt's first answer per call.
  int index;
  std::string name;
  double heuristic_us, tuned_us;
};

// One m the bucket is measured at: its problem and the heuristic's first
// answer there (what the library runs without a table).
struct Sample {
  int64_t m;
  std::unique_ptr<GemmProblem> p;
  hipblasLtMatmulAlgo_t heuristic;
  std::string heuristic_name;
  std::vector<double> heuristic_rounds;
};

struct Candidate {
  int index;
  std::string name;
  // The algorithm checked against each sample's problem (matmulIsAlgoSupported
  // records the problem in the algorithm), in sample order.
  std::vector<hipblasLtMatmulAlgo_t> algos;
  double quick_us = 0;
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

// The rows of one shape: per bucket (previous m, m], the solution with the
// least total time over the bucket's sample m values (its bounds and middle),
// pinned when that total beats the heuristic's own per-m answers by more than
// kMinGain; otherwise the bucket keeps the heuristic.
std::vector<Row> tune_shape(Bench &b, const ShapeSpec &spec) {
  const int64_t m_top = spec.ms.back();
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
  const size_t elem = spec.c_dtype == TURBINE_DTYPE_F32 ? 4 : 2;
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

  std::vector<Row> rows;
  int64_t prev = 0;
  for (const int64_t m_hi : spec.ms) {
    const int64_t lo = prev + 1;
    prev = m_hi;
    if (m_hi <= spec.keep_to) {
      // Not tuned: one heuristic row covers every bucket up to keep_to.
      if (m_hi == spec.keep_to) {
        std::fprintf(stderr,
                     "turbine_gemm_tune: %-14s m=(0,%lld] kept on the "
                     "heuristic (keep:%lld)\n",
                     spec.name.c_str(), static_cast<long long>(m_hi),
                     static_cast<long long>(m_hi));
        rows.push_back(Row{&spec, m_hi, -1, "heuristic", 0.0, 0.0});
      }
      continue;
    }
    std::vector<int64_t> ms{m_hi};
    if (lo < m_hi)
      ms.push_back(lo);
    if (m_hi - lo >= 4)
      ms.push_back((lo + m_hi) / 2);
    std::vector<Sample> samples;
    for (int64_t m : ms) {
      Sample s{m, std::make_unique<GemmProblem>()};
      if (s.p->make(GemmShape{m, spec.n, spec.k, spec.k, spec.k, spec.n,
                              spec.trans_b, spec.c_dtype}) !=
          HIPBLAS_STATUS_SUCCESS)
        die(std::string("problem: ") + s.p->failed);
      s.heuristic = heuristic_answer(b, *s.p, pref.handle);
      s.heuristic_name =
          hipblaslt_ext::getSolutionNameFromAlgo(b.lt, s.heuristic);
      samples.push_back(std::move(s));
    }

    // Candidates: the heuristic's list at the bucket's top m, plus each
    // sample's own first answer.
    std::vector<hipblasLtMatmulHeuristicResult_t> all(256);
    int returned = 0;
    blas_ok(hipblasLtMatmulAlgoGetHeuristic(
                b.lt, samples[0].p->desc.handle, samples[0].p->weight.handle,
                samples[0].p->act.handle, samples[0].p->out.handle,
                samples[0].p->out.handle, pref.handle, 256, all.data(),
                &returned),
            "heuristic list");
    all.resize(returned);
    std::vector<hipblasLtMatmulAlgo_t> pool;
    for (auto &r : all)
      if (r.state == HIPBLAS_STATUS_SUCCESS)
        pool.push_back(r.algo);
    for (auto &s : samples)
      pool.push_back(s.heuristic);
    std::vector<Candidate> cands;
    for (auto &a : pool) {
      std::string name = hipblaslt_ext::getSolutionNameFromAlgo(b.lt, a);
      if (std::any_of(cands.begin(), cands.end(),
                      [&](const Candidate &c) { return c.name == name; }))
        continue;
      Candidate c{hipblaslt_ext::getIndexFromAlgo(a), name};
      bool supported = true;
      for (auto &s : samples) {
        hipblasLtMatmulAlgo_t copy = a;
        size_t ws = 0;
        const float alpha = 1.0f, beta = 0.0f;
        if (hipblaslt_ext::matmulIsAlgoSupported(
                b.lt, s.p->desc.handle, &alpha, s.p->weight.handle,
                s.p->act.handle, &beta, s.p->out.handle, s.p->out.handle, copy,
                ws) != HIPBLAS_STATUS_SUCCESS ||
            ws > kWorkspaceBytes) {
          supported = false;
          break;
        }
        c.algos.push_back(copy);
      }
      if (supported)
        cands.push_back(std::move(c));
    }
    const size_t offered = cands.size();

    // Correctness against each sample's heuristic answer and determinism
    // (three bitwise-identical runs), at every sample.
    int wrong = 0, nondet = 0;
    for (size_t si = 0; si < samples.size(); ++si) {
      Sample &s = samples[si];
      const size_t out_elems = static_cast<size_t>(s.m) * spec.n;
      const size_t bytes = out_elems * elem;
      blas_ok(run(b, *s.p, s.heuristic, o.weights[0], o.act, o.ref),
              "reference");
      const float tol = std::max(
          compare(o.ref, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                  .max_ref /
              64.0f,
          1e-6f);
      std::vector<Candidate> kept;
      for (auto &c : cands) {
        hip_ok(hipMemsetAsync(o.out, 0, bytes, b.stream), "memset");
        if (run(b, *s.p, c.algos[si], o.weights[0], o.act, o.out) !=
                HIPBLAS_STATUS_SUCCESS ||
            compare(o.out, o.ref, out_elems, spec.c_dtype, o.scratch, b.stream)
                    .max_diff > tol) {
          ++wrong;
          continue;
        }
        bool same = true;
        for (int rep = 0; rep < 2 && same; ++rep) {
          hip_ok(hipMemsetAsync(o.out2, 0xff, bytes, b.stream), "memset");
          blas_ok(run(b, *s.p, c.algos[si], o.weights[0], o.act, o.out2),
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

    // Quick pass at the bucket's top m: about 1 ms of work per candidate.
    Sample &top = samples[0];
    (void)time_us(b, *top.p, top.heuristic, o, 3);
    const double h_est = time_us(b, *top.p, top.heuristic, o, 5);
    const int quick_iters = std::clamp(static_cast<int>(1000.0 / h_est), 3, 50);
    for (auto &c : cands) {
      (void)time_us(b, *top.p, c.algos[0], o, 1);
      c.quick_us = time_us(b, *top.p, c.algos[0], o, quick_iters);
    }
    std::sort(cands.begin(), cands.end(),
              [](const Candidate &x, const Candidate &y) {
                return x.quick_us < y.quick_us;
              });
    if (cands.size() > kRefine)
      cands.resize(kRefine);

    // Refinement: interleaved rounds of about 3 ms per candidate and sample,
    // the heuristic's own answer timed alongside at each sample.
    for (auto &c : cands)
      c.rounds.resize(samples.size());
    for (int round = 0; round < kRounds; ++round) {
      for (size_t si = 0; si < samples.size(); ++si) {
        Sample &s = samples[si];
        const double est = time_us(b, *s.p, s.heuristic, o, 2);
        const int iters = std::clamp(static_cast<int>(3000.0 / est), 5, 200);
        s.heuristic_rounds.push_back(time_us(b, *s.p, s.heuristic, o, iters));
        for (auto &c : cands)
          c.rounds[si].push_back(time_us(b, *s.p, c.algos[si], o, iters));
      }
    }
    double base = 0;
    for (auto &s : samples)
      base += median(s.heuristic_rounds);
    const Candidate *best = nullptr;
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
    std::string sample_list;
    for (auto &s : samples)
      sample_list += (sample_list.empty() ? "" : ",") + std::to_string(s.m);
    std::fprintf(
        stderr,
        "turbine_gemm_tune: %-14s m=(%lld,%lld] at m=%s candidates=%zu "
        "(rejected: %d wrong, %d nondeterministic) heuristic %.1f us "
        "-> %.1f us (%+.1f%%) %s\n",
        spec.name.c_str(), static_cast<long long>(lo - 1),
        static_cast<long long>(m_hi), sample_list.c_str(), offered, wrong,
        nondet, base, pin ? best_total : base,
        pin ? 100.0 * (best_total - base) / base : 0.0,
        pin ? best->name.c_str() : "(heuristic kept)");
    const double h_top = median(samples[0].heuristic_rounds);
    rows.push_back(Row{&spec, m_hi, pin ? best->index : -1,
                       pin ? best->name : std::string("heuristic"), h_top,
                       pin ? median(best->rounds[0]) : h_top});
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
    else
      die("usage: turbine_gemm_tune --shapes <gemm_shapes.txt> --out "
          "<tuning/<arch>/gemm.tsv> [--device <n>]");
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
    std::vector<Row> shape_rows = tune_shape(b, s);
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
      << "# n\tk\ttrans_b\tc_dtype\tm_max\tsolution_index\tsolution_name\t"
         "heuristic_us\ttuned_us\tshape\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row &r = rows[i];
    // Merge into the next bucket of the same shape when it pins the same
    // solution: that row then serves this bucket too.
    if (i + 1 < rows.size() && rows[i + 1].spec == r.spec &&
        rows[i + 1].name == r.name)
      continue;
    char us[64];
    std::snprintf(us, sizeof(us), "%.1f\t%.1f", r.heuristic_us, r.tuned_us);
    out << r.spec->n << "\t" << r.spec->k << "\t" << r.spec->trans_b << "\t"
        << (r.spec->c_dtype == TURBINE_DTYPE_F32 ? "f32" : "bf16") << "\t"
        << r.m << "\t" << r.index << "\t" << r.name << "\t" << us << "\t"
        << r.spec->name << "\n";
  }
  std::fprintf(stderr, "turbine_gemm_tune: wrote %s\n", out_path.c_str());
  return 0;
}
