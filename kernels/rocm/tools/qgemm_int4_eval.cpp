// turbine_qgemm_int4_eval: the INT4 group GEMM provider evaluation (Phase 6a
// Task 16, kernel reuse rule) on the current device. For the Llama-3.2-3B
// linear shapes (qkv 5120 x 3072, o 3072 x 3072, gate_up 16384 x 3072, down
// 3072 x 8192), group 128, both INT4 schemes (ZP: AWQ zero points, SYM: GPTQ
// implicit 8) and each requested m it runs, through libturbine_hip.so's ABI:
//
//   bf16            turbine_gemm (hipBLASLt, tuned table) on the weight
//                   dequantized to BF16: the BF16 model's cost, the baseline
//   <impl>          every turbine_qgemm implementation of the library that
//                   supports the descriptor (turbine_impl_run by index)
//
// checks every INT4 implementation against a host reference of the CPU
// provider's semantics (crates/turbine-kernels/src/cpu/qgemm.rs: the weight
// (q - z) * s in F32, the activations as stored, an exact sum rounded once to
// the output) over sampled rows -- the BF16 baseline and the dequant path
// against the same sum over the BF16-rounded weight, which they multiply -- and
// times it: microseconds per call (median of
// --rounds rounds of --iters calls, rotating over enough weight copies to
// defeat the 64 MiB cache). It also reports whether row 0 of each INT4
// implementation is bitwise the same at every m (row invariance).
//
// The external candidates (Composable Kernel's ck_tile BQuant, llama.cpp's
// mul_mat_vec_q / mmq, vLLM's ROCm kernels) are evaluated with their own tools;
// see the decision "P6: INT4 group GEMM — provider evaluation (kernel reuse
// rule)".
//
//   turbine_qgemm_int4_eval [--ms 1,4,16,128,2048] [--iters 20] [--rounds 5]
//                           [--shapes qkv,o,gate_up,down] [--schemes zp,sym]
//                           [--sweep 1]
//
// A lab tool (run it on GPU 0 under scripts/bench-lock.sh); not loaded by the
// server. Exit 0 when every INT4 implementation matched the reference, 1
// otherwise.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "qgemm_int4_kernels.hpp"
#include "turbine_kernels.h"

namespace {

struct Shape {
  const char *name;
  int64_t n, k;
};

const Shape kShapes[] = {{"qkv", 5120, 3072},
                         {"o", 3072, 3072},
                         {"gate_up", 16384, 3072},
                         {"down", 3072, 8192}};
constexpr int32_t kGroup = 128;
// Weight copies rotate until they span this many bytes (past the 64 MiB
// Infinity Cache of the R9700).
constexpr size_t kRotateBytes = 256u << 20;

void check_hip(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
  }
}

void check_t(turbine_ctx *ctx, int32_t rc, const char *what) {
  if (rc != TURBINE_OK) {
    char buf[1024] = {0};
    turbine_last_error(ctx, buf, sizeof(buf));
    std::fprintf(stderr, "%s: %d %s\n", what, rc, buf);
    std::exit(2);
  }
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

std::vector<int64_t> parse_list(const char *s) {
  std::vector<int64_t> out;
  for (const char *p = s; *p;) {
    out.push_back(std::strtoll(p, nullptr, 10));
    p = std::strchr(p, ',');
    if (!p)
      break;
    ++p;
  }
  return out;
}

bool listed(const std::string &list, const char *name) {
  return ("," + list + ",").find("," + std::string(name) + ",") !=
         std::string::npos;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// A launch shape of the fused kernel (--sweep 1).
using LaunchFn = hipError_t (*)(const void *, const uint8_t *, const float *,
                                const uint8_t *, void *, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int32_t, bool, float,
                                hipStream_t);
struct Variant {
  std::string name;
  LaunchFn launch;
};

template <int kFragsN, int kWaves> Variant variant() {
  return {"wmma_f" + std::to_string(kFragsN) + "_w" + std::to_string(kWaves),
          &turbine_hip::int4::launch_wmma<kFragsN, kWaves>};
}

const std::vector<Variant> &variants() {
  static const std::vector<Variant> v = {
      variant<1, 1>(), variant<1, 2>(), variant<1, 4>(), variant<1, 8>(),
      variant<2, 1>(), variant<2, 2>(), variant<2, 4>(), variant<2, 8>(),
      variant<4, 1>(), variant<4, 2>(), variant<4, 4>()};
  return v;
}

// One weight's host data and its rotating device copies.
struct Weight {
  int64_t n, k, groups;
  bool zp;
  std::vector<uint8_t> codes; // [n, k/2]
  std::vector<float> scales;  // [n, groups]
  std::vector<uint8_t> zeros; // [n, groups]
  std::vector<uint16_t> deq;  // [n, k] bf16((q - z) s), the baseline weight
  int copies = 0;
  std::vector<void *> d_codes, d_scales, d_zeros, d_deq;

  float value(int64_t row, int64_t col) const {
    const uint8_t byte = codes[row * (k / 2) + col / 2];
    const float q = static_cast<float>(col % 2 == 0 ? byte & 15 : byte >> 4);
    const int64_t at = row * groups + col / kGroup;
    const float z = zp ? static_cast<float>(zeros[at]) : 8.0f;
    return (q - z) * scales[at];
  }
};

Weight make_weight(int64_t n, int64_t k, bool zp, std::mt19937 &rng) {
  Weight w{n, k, k / kGroup, zp, {}, {}, {}, {}};
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> scale(0.002f, 0.02f);
  w.codes.resize(n * k / 2);
  for (auto &c : w.codes)
    c = static_cast<uint8_t>(byte(rng));
  w.scales.resize(n * w.groups);
  for (auto &s : w.scales)
    s = scale(rng);
  w.zeros.resize(n * w.groups);
  for (auto &z : w.zeros)
    z = static_cast<uint8_t>(zp ? nib(rng) : 8);
  w.deq.resize(n * k);
  for (int64_t r = 0; r < n; ++r)
    for (int64_t c = 0; c < k; ++c)
      w.deq[r * k + c] = to_bf16(w.value(r, c));
  const size_t int4_bytes =
      w.codes.size() + w.scales.size() * 4 + w.zeros.size();
  w.copies = static_cast<int>(
      std::max<size_t>(2, (kRotateBytes + int4_bytes - 1) / int4_bytes));
  for (int i = 0; i < w.copies; ++i) {
    void *p;
    check_hip(hipMalloc(&p, w.codes.size()), "hipMalloc codes");
    check_hip(
        hipMemcpy(p, w.codes.data(), w.codes.size(), hipMemcpyHostToDevice),
        "copy codes");
    w.d_codes.push_back(p);
    check_hip(hipMalloc(&p, w.scales.size() * 4), "hipMalloc scales");
    check_hip(hipMemcpy(p, w.scales.data(), w.scales.size() * 4,
                        hipMemcpyHostToDevice),
              "copy scales");
    w.d_scales.push_back(p);
    check_hip(hipMalloc(&p, w.zeros.size()), "hipMalloc zeros");
    check_hip(
        hipMemcpy(p, w.zeros.data(), w.zeros.size(), hipMemcpyHostToDevice),
        "copy zeros");
    w.d_zeros.push_back(p);
  }
  const size_t bf16_bytes = w.deq.size() * 2;
  const int bf16_copies = static_cast<int>(
      std::max<size_t>(2, (kRotateBytes + bf16_bytes - 1) / bf16_bytes));
  for (int i = 0; i < bf16_copies; ++i) {
    void *p;
    check_hip(hipMalloc(&p, bf16_bytes), "hipMalloc deq");
    check_hip(hipMemcpy(p, w.deq.data(), bf16_bytes, hipMemcpyHostToDevice),
              "copy deq");
    w.d_deq.push_back(p);
  }
  return w;
}

void free_weight(Weight &w) {
  for (auto *v : {&w.d_codes, &w.d_scales, &w.d_zeros, &w.d_deq})
    for (void *p : *v)
      (void)hipFree(p);
}

// Host reference of sampled rows: exact sum (double) of a (BF16) times the F32
// weight, rounded once to BF16.
std::vector<int64_t> sample_rows(int64_t m) {
  std::vector<int64_t> rows;
  const int64_t want = std::min<int64_t>(m, 8);
  for (int64_t i = 0; i < want; ++i)
    rows.push_back(want == 1 ? 0 : i * (m - 1) / (want - 1));
  return rows;
}

struct Result {
  double us = 0;
  double max_err = 0; // max |got - ref| over sampled rows
  double ref_rms = 0; // rms of the reference over sampled rows
  std::vector<uint16_t> row0;
};

} // namespace

int main(int argc, char **argv) {
  std::vector<int64_t> ms = {1, 4, 16, 128, 2048};
  int iters = 20;
  int rounds = 5;
  std::string shapes = "qkv,o,gate_up,down";
  std::string schemes = "zp,sym";
  bool sweep = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--ms")
      ms = parse_list(argv[i + 1]);
    else if (arg == "--iters")
      iters = std::atoi(argv[i + 1]);
    else if (arg == "--rounds")
      rounds = std::atoi(argv[i + 1]);
    else if (arg == "--shapes")
      shapes = argv[i + 1];
    else if (arg == "--schemes")
      schemes = argv[i + 1];
    else if (arg == "--sweep")
      sweep = std::atoi(argv[i + 1]) != 0;
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  turbine_ctx *ctx = nullptr;
  if (turbine_ctx_create(0, &ctx) != TURBINE_OK) {
    std::fprintf(stderr, "turbine_ctx_create failed\n");
    return 2;
  }
  const int32_t count = turbine_impl_count(TURBINE_OP_QGEMM);
  std::vector<std::string> impl_names;
  for (int32_t i = 0; i < count; ++i) {
    turbine_impl_entry e{};
    check_t(ctx, turbine_impl_info(TURBINE_OP_QGEMM, i, &e), "impl_info");
    impl_names.push_back(e.name);
  }
  const int64_t max_m = *std::max_element(ms.begin(), ms.end());
  std::mt19937 rng(20260929);
  bool ok = true;
  std::printf("shape scheme m candidate us weight_GB/s max_err ref_rms "
              "err_ratio row0_invariant\n");
  for (const Shape &shape : kShapes) {
    if (!listed(shapes, shape.name))
      continue;
    const int64_t k = shape.k;
    std::vector<uint16_t> a_host(max_m * k);
    std::uniform_real_distribution<float> av(-1.0f, 1.0f);
    for (auto &v : a_host)
      v = to_bf16(av(rng));
    void *d_a;
    check_hip(hipMalloc(&d_a, a_host.size() * 2), "hipMalloc a");
    check_hip(
        hipMemcpy(d_a, a_host.data(), a_host.size() * 2, hipMemcpyHostToDevice),
        "copy a");
    void *d_c;
    check_hip(hipMalloc(&d_c, max_m * shape.n * 2), "hipMalloc c");
    for (const char *scheme_name : {"zp", "sym"}) {
      if (!listed(schemes, scheme_name))
        continue;
      const bool zp = std::strcmp(scheme_name, "zp") == 0;
      Weight w = make_weight(shape.n, k, zp, rng);
      std::map<std::string, std::vector<uint16_t>> row0_by_name;
      std::map<std::string, bool> invariant;
      for (int64_t m : ms) {
        const std::vector<int64_t> rows = sample_rows(m);
        // ref: the exact product; ref_bf16: the product with the BF16-rounded
        // dequantized weight (what the BF16 baseline and the dequant path
        // compute).
        std::vector<double> ref(rows.size() * shape.n);
        std::vector<double> ref_bf16(rows.size() * shape.n);
        double sq = 0;
        for (size_t ri = 0; ri < rows.size(); ++ri) {
          for (int64_t col = 0; col < shape.n; ++col) {
            double acc = 0;
            double acc16 = 0;
            for (int64_t kk = 0; kk < k; ++kk) {
              const double a =
                  static_cast<double>(from_bf16(a_host[rows[ri] * k + kk]));
              acc += a * w.value(col, kk);
              acc16 += a * from_bf16(w.deq[col * k + kk]);
            }
            ref[ri * shape.n + col] = acc;
            ref_bf16[ri * shape.n + col] = acc16;
            sq += acc * acc;
          }
        }
        const double ref_rms = std::sqrt(sq / ref.size());
        // Candidates: the BF16 baseline, the library's qgemm implementations
        // (turbine_impl_run on the context stream) and, with --sweep, the fused
        // kernel's launch shapes (default stream).
        struct Cand {
          std::string name;
          std::function<int32_t(int)> run; // copy index -> TURBINE_* code
          bool on_ctx;
        };
        std::vector<Cand> cands;
        turbine_gemm_desc gd{};
        gd.a = d_a;
        gd.c = d_c;
        gd.m = m;
        gd.n = shape.n;
        gd.k = k;
        gd.lda = k;
        gd.ldb = k;
        gd.ldc = shape.n;
        gd.trans_b = 1;
        gd.a_dtype = gd.b_dtype = gd.c_dtype = TURBINE_DTYPE_BF16;
        gd.alpha = 1.0f;
        cands.push_back({"bf16",
                         [&, gd](int copy) mutable {
                           gd.b = w.d_deq[copy % w.d_deq.size()];
                           return turbine_gemm(ctx, &gd);
                         },
                         true});
        turbine_qgemm_desc qd{};
        qd.a = d_a;
        qd.c = d_c;
        qd.m = m;
        qd.n = shape.n;
        qd.k = k;
        qd.lda = k;
        qd.ldc = shape.n;
        qd.scheme =
            zp ? TURBINE_QSCHEME_INT4_GROUP_ZP : TURBINE_QSCHEME_INT4_GROUP_SYM;
        qd.act_quant = TURBINE_ACTQ_NONE;
        qd.a_dtype = TURBINE_DTYPE_BF16;
        qd.c_dtype = TURBINE_DTYPE_BF16;
        qd.group_size = kGroup;
        qd.alpha = 1.0f;
        for (int32_t cand = 0; cand < count; ++cand) {
          if (turbine_impl_supports(TURBINE_OP_QGEMM, cand, &qd) != 1)
            continue;
          cands.push_back(
              {impl_names[static_cast<size_t>(cand)],
               [&, qd, cand](int copy) mutable {
                 const int i = copy % w.copies;
                 qd.b = w.d_codes[i];
                 qd.b_scales = w.d_scales[i];
                 qd.b_zeros =
                     zp ? static_cast<const uint8_t *>(w.d_zeros[i]) : nullptr;
                 return turbine_impl_run(ctx, TURBINE_OP_QGEMM, cand, &qd);
               },
               true});
        }
        if (sweep) {
          for (const Variant &v : variants()) {
            cands.push_back(
                {v.name,
                 [&, v](int copy) {
                   const int i = copy % w.copies;
                   const hipError_t e =
                       v.launch(d_a, static_cast<const uint8_t *>(w.d_codes[i]),
                                static_cast<const float *>(w.d_scales[i]),
                                zp ? static_cast<const uint8_t *>(w.d_zeros[i])
                                   : nullptr,
                                d_c, m, shape.n, k, k, shape.n, kGroup, false,
                                1.0f, nullptr);
                   return e == hipSuccess ? TURBINE_OK : TURBINE_E_DEVICE;
                 },
                 false});
          }
        }
        for (size_t ci = 0; ci < cands.size(); ++ci) {
          Cand &cd = cands[ci];
          const char *name = cd.name.c_str();
          auto sync = [&]() {
            if (cd.on_ctx)
              check_t(ctx, turbine_stream_sync(ctx), "sync");
            else
              check_hip(hipDeviceSynchronize(), "sync");
          };
          // correctness (one call on copy 0)
          check_t(ctx, cd.run(0), name);
          sync();
          std::vector<uint16_t> got(m * shape.n);
          check_hip(
              hipMemcpy(got.data(), d_c, got.size() * 2, hipMemcpyDeviceToHost),
              "copy c");
          // |got - ref| against one BF16 output step (2^-8 |ref|) plus the BF16
          // rounding of the dequantized weight summed over k (2^-7 of the
          // reference rms, several standard deviations).
          const bool rounded =
              ci == 0 || cd.name.find("dequant") != std::string::npos;
          double max_err = 0;
          double ratio = 0;
          for (size_t ri = 0; ri < rows.size(); ++ri)
            for (int64_t col = 0; col < shape.n; ++col) {
              const double r = rounded ? ref_bf16[ri * shape.n + col]
                                       : ref[ri * shape.n + col];
              const double e =
                  std::fabs(from_bf16(got[rows[ri] * shape.n + col]) - r);
              max_err = std::max(max_err, e);
              ratio = std::max(
                  ratio, e / (std::fabs(r) / 256 + ref_rms / 128 + 1e-30));
            }
          std::vector<uint16_t> row0(got.begin(), got.begin() + shape.n);
          auto &seen = row0_by_name[cd.name];
          bool &inv = invariant.try_emplace(cd.name, true).first->second;
          if (seen.empty())
            seen = row0;
          else if (seen != row0)
            inv = false;
          // timing
          int copy = 1;
          for (int i = 0; i < 3; ++i)
            check_t(ctx, cd.run(copy++), name);
          sync();
          std::vector<double> per_round;
          for (int r = 0; r < rounds; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i)
              check_t(ctx, cd.run(copy++), name);
            sync();
            const auto t1 = std::chrono::steady_clock::now();
            per_round.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count() /
                iters);
          }
          const bool pass = ratio <= 1.0;
          if (ci > 0 && !pass)
            ok = false;
          const double us = median(per_round);
          const double bytes =
              ci == 0
                  ? static_cast<double>(w.deq.size() * 2)
                  : static_cast<double>(w.codes.size() + w.scales.size() * 4 +
                                        (zp ? w.zeros.size() : 0));
          std::printf("%s %s %lld %s %.1f %.0f %.4g %.4g %.3g%s %s\n",
                      shape.name, scheme_name, static_cast<long long>(m), name,
                      us, bytes / (us * 1e3), max_err, ref_rms, ratio,
                      pass ? "" : " FAIL", inv ? "yes" : "NO");
          std::fflush(stdout);
        }
      }
      free_weight(w);
    }
    (void)hipFree(d_a);
    (void)hipFree(d_c);
  }
  turbine_ctx_destroy(ctx);
  return ok ? 0 : 1;
}
