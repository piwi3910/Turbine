// P7 Task 14: quantized MoE expert provider evaluation (gfx1201).
//
// For each expert shape and token count: random top-k routing, the routed rows
// pre-gathered by expert, then the two expert GEMMs (gate_up [2I x H], down
// [H x I]) over every active expert. Candidates:
//   own_loop:<impl>  every turbine_qgemm implementation of libturbine_hip.so
//                    that accepts the format, called once per active expert and
//                    projection (the "own kernel extended to grouped form"
//                    baseline as it runs today: host expert offsets, one launch
//                    per expert); W8A8 rows include the activation quantize.
//   bf16_moe_fused   turbine_moe_experts on BF16 weights of the same shape (the
//                    full fused op incl. SiLU and scatter: what a BF16
//                    checkpoint of that shape costs; not a quantized
//                    candidate).
//   hipblaslt probes hipblaslt_ext::GroupedGemm heuristic for FP8xFP8 and
//                    FP8 x BF16 (reported; timed when a solution exists).
// Correctness: sampled rows of 3 experts per projection against a host
// reference over the exact dequantized weights (and, for W8A8 static, the
// e4m3-quantized activations), bound |d| <= |ref|/256 + rms/128 (the 6a qgemm
// rule: one BF16 output rounding plus the BF16 weight rounding over k).
// For gpt-oss also: the error a q8_1 (W4A8, llama.cpp's MoE arithmetic)
// activation quantization adds against the W4A16 reference, same bound.
// Timing: 5 rounds x 10 runs (50), median of the round means, weight copies
// rotated past the 64 MiB infinity cache.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "turbine_kernels.h"

namespace {

void check_hip(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
  }
}

std::string last_err(turbine_ctx *ctx) {
  char buf[1024] = {0};
  turbine_last_error(ctx, buf, sizeof(buf));
  return buf;
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
float e4m3_value(uint8_t c) {
  const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  float v =
      e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
  return s ? -v : v;
}
// nearest e4m3 code, ties to even, saturating at 448
uint8_t e4m3_round(float x) {
  static std::vector<std::pair<float, uint8_t>> table = [] {
    std::vector<std::pair<float, uint8_t>> t;
    for (int c = 0; c < 127; ++c) // 0x7f is NaN
      t.push_back(
          {e4m3_value(static_cast<uint8_t>(c)), static_cast<uint8_t>(c)});
    return t;
  }();
  const float a = std::min(std::fabs(x), 448.0f);
  auto it = std::lower_bound(
      table.begin(), table.end(), std::make_pair(a, uint8_t(0)),
      [](auto &l, auto &r) { return l.first < r.first; });
  uint8_t code;
  if (it == table.end())
    code = 126;
  else if (it == table.begin() || it->first == a)
    code = it->second;
  else {
    auto lo = it - 1;
    const float dl = a - lo->first, dh = it->first - a;
    code = dl < dh   ? lo->second
           : dh < dl ? it->second
                     : ((lo->second & 1) ? it->second : lo->second);
  }
  return static_cast<uint8_t>(code | (x < 0 ? 0x80 : 0));
}
float e2m1_value(uint8_t c) {
  static const float v[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  return (c & 8) ? -v[c & 7] : v[c & 7];
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

enum class Fmt { Fp8Block, Fp8Tensor, Mxfp4 };

struct Shape {
  const char *name;
  int experts, topk, hidden, inter;
  Fmt fmt;
};
const Shape kShapes[] = {{"qwen3_30b_a3b", 128, 8, 2048, 768, Fmt::Fp8Block},
                         {"mixtral_8x7b", 8, 2, 4096, 14336, Fmt::Fp8Tensor},
                         {"gpt_oss_20b", 32, 4, 2880, 2880, Fmt::Mxfp4}};

// One projection's weights for every expert: n x k each.
struct Proj {
  int64_t n, k;
  Fmt fmt;
  int copies = 1;
  std::vector<uint8_t> codes;   // expert-major
  std::vector<float> scales;    // fp8: block or tensor scales per expert
  std::vector<uint8_t> e8m0;    // mxfp4 scales per expert
  std::vector<void *> d_codes;  // per copy
  std::vector<void *> d_scales; // per copy
  int64_t code_bytes() const { return fmt == Fmt::Mxfp4 ? n * k / 2 : n * k; }
  int64_t scale_count() const {
    if (fmt == Fmt::Fp8Block)
      return ((n + 127) / 128) * ((k + 127) / 128);
    if (fmt == Fmt::Fp8Tensor)
      return 1;
    return n * k / 32;
  }
  int64_t scale_bytes() const {
    return fmt == Fmt::Mxfp4 ? scale_count() : scale_count() * 4;
  }
  double value(int e, int64_t row, int64_t col) const {
    if (fmt == Fmt::Mxfp4) {
      const uint8_t byte = codes[e * code_bytes() + (row * k + col) / 2];
      const uint8_t q = (col & 1) ? (byte >> 4) : (byte & 15);
      const int ex = e8m0[e * scale_count() + row * (k / 32) + col / 32];
      return e2m1_value(q) * std::ldexp(1.0, ex - 127);
    }
    const double w = e4m3_value(codes[e * code_bytes() + row * k + col]);
    if (fmt == Fmt::Fp8Tensor)
      return w * scales[e];
    const int64_t kb = (k + 127) / 128;
    return w * scales[e * scale_count() + (row / 128) * kb + col / 128];
  }
};

uint64_t g_lcg = 0x9e3779b97f4a7c15ull;
inline uint32_t fast_rand() {
  g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<uint32_t>(g_lcg >> 33);
}

void make_proj(Proj &p, int experts, int64_t rotate_bytes) {
  const int64_t cb = p.code_bytes(), sc = p.scale_count();
  p.codes.resize(static_cast<size_t>(experts) * cb);
  for (auto &c : p.codes) {
    if (p.fmt == Fmt::Mxfp4)
      c = static_cast<uint8_t>(fast_rand());
    else // sign | magnitude code < 0x58 (|w| <= 15): no NaN, no saturation
      c = static_cast<uint8_t>((fast_rand() & 0x80) | (fast_rand() % 0x58));
  }
  if (p.fmt == Fmt::Mxfp4) {
    p.e8m0.resize(static_cast<size_t>(experts) * sc);
    for (auto &s : p.e8m0)
      s = static_cast<uint8_t>(120 + fast_rand() % 6);
  } else {
    p.scales.resize(static_cast<size_t>(experts) * sc);
    for (auto &s : p.scales)
      s = (0.5f + (fast_rand() % 1000) / 1000.0f) * 1e-2f;
  }
  const int64_t layer = experts * (cb + p.scale_bytes());
  p.copies = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(4, (rotate_bytes + layer - 1) / layer)));
  for (int c = 0; c < p.copies; ++c) {
    void *dc, *ds;
    check_hip(hipMalloc(&dc, p.codes.size()), "malloc codes");
    check_hip(
        hipMemcpy(dc, p.codes.data(), p.codes.size(), hipMemcpyHostToDevice),
        "copy codes");
    const size_t sb = static_cast<size_t>(experts) * p.scale_bytes();
    check_hip(hipMalloc(&ds, sb), "malloc scales");
    check_hip(hipMemcpy(ds,
                        p.fmt == Fmt::Mxfp4
                            ? static_cast<const void *>(p.e8m0.data())
                            : static_cast<const void *>(p.scales.data()),
                        sb, hipMemcpyHostToDevice),
              "copy scales");
    p.d_codes.push_back(dc);
    p.d_scales.push_back(ds);
  }
}

void free_proj(Proj &p) {
  for (void *d : p.d_codes)
    hipFree(d);
  for (void *d : p.d_scales)
    hipFree(d);
  p.d_codes.clear();
  p.d_scales.clear();
}

struct Routing {
  std::vector<int32_t> offsets; // [E+1] rows per expert, gathered order
  std::vector<int32_t> sorted_rows;
  std::vector<float> weights;
  std::vector<int32_t> active;
};

Routing route(int tokens, int experts, int topk, std::mt19937 &rng) {
  std::vector<std::vector<int32_t>> by_e(experts);
  std::vector<int> ids(experts);
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < experts; ++e)
      ids[e] = e;
    for (int s = 0; s < topk; ++s) {
      std::uniform_int_distribution<int> d(s, experts - 1);
      std::swap(ids[s], ids[d(rng)]);
      by_e[ids[s]].push_back(t * topk + s);
    }
  }
  Routing r;
  r.offsets.push_back(0);
  for (int e = 0; e < experts; ++e) {
    for (int32_t row : by_e[e])
      r.sorted_rows.push_back(row);
    r.offsets.push_back(static_cast<int32_t>(r.sorted_rows.size()));
    if (!by_e[e].empty())
      r.active.push_back(e);
  }
  r.weights.assign(static_cast<size_t>(tokens) * topk, 1.0f / topk);
  return r;
}

// q8_1 as llama.cpp's MoE path quantizes activations: blocks of 32, d =
// amax/127 (F16), q = round(x/d).
std::vector<double> q8_1(const std::vector<double> &x) {
  std::vector<double> out(x.size());
  for (size_t b = 0; b < x.size(); b += 32) {
    double amax = 0;
    for (size_t i = b; i < b + 32 && i < x.size(); ++i)
      amax = std::max(amax, std::fabs(x[i]));
    const float d = static_cast<float>(amax / 127.0);
    const double dh = static_cast<double>(static_cast<_Float16>(d));
    for (size_t i = b; i < b + 32 && i < x.size(); ++i)
      out[i] = d == 0 ? 0 : std::nearbyint(x[i] / d) * dh;
  }
  return out;
}

bool listed(const std::string &list, const std::string &name) {
  return list.empty() ||
         ("," + list + ",").find("," + name + ",") != std::string::npos;
}

// hipBLASLt grouped-GEMM heuristic probe for (a_type weight, b_type act).
int probe_grouped(hipblasLtHandle_t h, hipDataType wt, hipDataType at,
                  const std::vector<int64_t> &rows, int64_t n_out,
                  int64_t depth, void *dummy, const float *one) {
  try {
    hipblaslt_ext::GroupedGemm gemm(h, HIPBLAS_OP_T, HIPBLAS_OP_N, wt, at,
                                    HIP_R_16BF, HIP_R_16BF,
                                    HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m, n, k, b;
    for (int64_t r : rows) {
      m.push_back(n_out);
      n.push_back(r);
      k.push_back(depth);
      b.push_back(1);
    }
    std::vector<hipblaslt_ext::GemmEpilogue> ep(rows.size());
    std::vector<hipblaslt_ext::GemmInputs> in(rows.size());
    for (auto &i : in) {
      i.setA(dummy);
      i.setB(dummy);
      i.setC(dummy);
      i.setD(dummy);
      i.setAlpha(one);
      i.setBeta(one);
      if (wt == HIP_R_8F_E4M3)
        i.setScaleA(dummy);
      if (at == HIP_R_8F_E4M3)
        i.setScaleB(dummy);
    }
    if (gemm.setProblem(m, n, k, b, ep, in) != HIPBLAS_STATUS_SUCCESS)
      return -1;
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(64u << 20);
    std::vector<hipblasLtMatmulHeuristicResult_t> res;
    if (gemm.algoGetHeuristic(8, pref, res) != HIPBLAS_STATUS_SUCCESS)
      return -2;
    return static_cast<int>(res.size());
  } catch (...) {
    return -3;
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string shapes_arg, cands_arg;
  std::vector<int> token_counts = {1, 16, 256, 4096};
  int rounds = 5, iters = 10;
  bool check_only = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--shapes")
      shapes_arg = argv[i + 1];
    else if (a == "--cands")
      cands_arg = argv[i + 1];
    else if (a == "--tokens") {
      token_counts.clear();
      for (const char *p = argv[i + 1]; p && *p;) {
        token_counts.push_back(std::atoi(p));
        p = std::strchr(p, ',');
        if (p)
          ++p;
      }
    } else if (a == "--check-only")
      check_only = std::atoi(argv[i + 1]) != 0;
  }
  turbine_ctx *ctx = nullptr;
  if (turbine_ctx_create(0, &ctx) != TURBINE_OK) {
    std::fprintf(stderr, "turbine_ctx_create failed\n");
    return 2;
  }
  std::printf("# libturbine_hip abi minor %u\n", turbine_abi_minor());
  const int32_t nq = turbine_impl_count(TURBINE_OP_QGEMM);
  std::vector<std::string> qnames;
  for (int32_t i = 0; i < nq; ++i) {
    turbine_impl_entry e{};
    turbine_impl_info(TURBINE_OP_QGEMM, i, &e);
    qnames.push_back(e.name);
  }
  hipblasLtHandle_t blaslt;
  hipblasLtCreate(&blaslt);
  std::mt19937 rng(20261008);
  std::printf(
      "shape fmt tokens rows active proj candidate us max_ratio pass note\n");
  for (const Shape &sh : kShapes) {
    if (!listed(shapes_arg, sh.name))
      continue;
    const int64_t H = sh.hidden, I = sh.inter;
    Proj gu{2 * I, H, sh.fmt}, dn{H, I, sh.fmt};
    make_proj(gu, sh.experts, 512ll << 20);
    make_proj(dn, sh.experts, 512ll << 20);
    const float act_scale =
        0.05f; // Mixtral static input scale (x in [-1,1] -> |q| <= 20)
    const int max_t =
        *std::max_element(token_counts.begin(), token_counts.end());
    const int64_t max_rows = static_cast<int64_t>(max_t) * sh.topk;
    // gathered activations (BF16) for gate_up [rows, H] and down [rows, I]
    std::vector<uint16_t> x_gu(max_rows * H), x_dn(max_rows * I);
    std::uniform_real_distribution<float> av(-1.0f, 1.0f);
    for (auto &v : x_gu)
      v = to_bf16(av(rng));
    for (auto &v : x_dn)
      v = to_bf16(av(rng));
    void *d_xgu, *d_xdn, *d_q8gu, *d_q8dn, *d_ogu, *d_odn;
    float *d_sgu, *d_sdn;
    check_hip(hipMalloc(&d_xgu, x_gu.size() * 2), "m");
    check_hip(hipMalloc(&d_xdn, x_dn.size() * 2), "m");
    check_hip(hipMalloc(&d_q8gu, x_gu.size()), "m");
    check_hip(hipMalloc(&d_q8dn, x_dn.size()), "m");
    check_hip(hipMalloc(&d_ogu, max_rows * 2 * I * 2), "m");
    check_hip(hipMalloc(&d_odn, max_rows * H * 2), "m");
    check_hip(hipMalloc(&d_sgu, sizeof(float) * max_rows * (H / 128 + 1)), "m");
    check_hip(hipMalloc(&d_sdn, sizeof(float) * max_rows * (I / 128 + 1)), "m");
    check_hip(
        hipMemcpy(d_xgu, x_gu.data(), x_gu.size() * 2, hipMemcpyHostToDevice),
        "c");
    check_hip(
        hipMemcpy(d_xdn, x_dn.data(), x_dn.size() * 2, hipMemcpyHostToDevice),
        "c");
    const char *fmt_name = sh.fmt == Fmt::Fp8Block    ? "fp8_block"
                           : sh.fmt == Fmt::Fp8Tensor ? "fp8_tensor"
                                                      : "mxfp4";
    for (int T : token_counts) {
      Routing r = route(T, sh.experts, sh.topk, rng);
      const int64_t rows = static_cast<int64_t>(T) * sh.topk;
      // host references for 3 sampled experts x up to 2 rows per projection
      std::vector<int> sample_e = {
          r.active.front(), r.active[r.active.size() / 2], r.active.back()};
      sample_e.erase(std::unique(sample_e.begin(), sample_e.end()),
                     sample_e.end());
      struct Ref {
        int e;
        int64_t row; // gathered row index
        std::vector<double> gu, dn, gu_q81, dn_q81;
      };
      std::vector<Ref> refs;
      double sq_gu = 0, sq_dn = 0;
      size_t cnt_gu = 0, cnt_dn = 0;
      for (int e : sample_e) {
        const int32_t b = r.offsets[e], end = r.offsets[e + 1];
        for (int32_t gr = b; gr < end && gr < b + 2; ++gr) {
          Ref rf{e, gr, {}, {}, {}, {}};
          auto act = [&](const std::vector<uint16_t> &x, int64_t k) {
            std::vector<double> a(k);
            for (int64_t c = 0; c < k; ++c) {
              double v = from_bf16(x[gr * k + c]);
              if (sh.fmt ==
                  Fmt::Fp8Tensor) // static W8A8: the checkpoint's arithmetic
                v = e4m3_value(e4m3_round(static_cast<float>(v / act_scale))) *
                    act_scale;
              a[c] = v;
            }
            return a;
          };
          const std::vector<double> agu = act(x_gu, H), adn = act(x_dn, I);
          const std::vector<double> qgu = q8_1(agu), qdn = q8_1(adn);
          for (int64_t col = 0; col < gu.n; ++col) {
            double s = 0, s8 = 0;
            for (int64_t c = 0; c < H; ++c) {
              const double w = gu.value(e, col, c);
              s += agu[c] * w;
              s8 += qgu[c] * w;
            }
            rf.gu.push_back(s);
            rf.gu_q81.push_back(s8);
            sq_gu += s * s;
            ++cnt_gu;
          }
          for (int64_t col = 0; col < dn.n; ++col) {
            double s = 0, s8 = 0;
            for (int64_t c = 0; c < I; ++c) {
              const double w = dn.value(e, col, c);
              s += adn[c] * w;
              s8 += qdn[c] * w;
            }
            rf.dn.push_back(s);
            rf.dn_q81.push_back(s8);
            sq_dn += s * s;
            ++cnt_dn;
          }
          refs.push_back(std::move(rf));
        }
      }
      const double rms_gu = std::sqrt(sq_gu / cnt_gu),
                   rms_dn = std::sqrt(sq_dn / cnt_dn);
      auto ratio_of = [&](bool gate_up, const std::vector<uint16_t> *got,
                          bool q81) {
        double worst = 0;
        for (const Ref &rf : refs) {
          const std::vector<double> &ref = gate_up ? rf.gu : rf.dn;
          const std::vector<double> &alt = gate_up ? rf.gu_q81 : rf.dn_q81;
          const int64_t n = gate_up ? 2 * I : H;
          const double rms = gate_up ? rms_gu : rms_dn;
          for (int64_t c = 0; c < n; ++c) {
            const double g = q81 ? alt[c] : from_bf16((*got)[rf.row * n + c]);
            worst = std::max(worst, std::fabs(g - ref[c]) /
                                        (std::fabs(ref[c]) / 256 + rms / 128));
          }
        }
        return worst;
      };
      if (sh.fmt == Fmt::Mxfp4)
        std::printf("%s %s %d %lld %zu both q8_1_emulated - %.3g %s W4A8 "
                    "(llama.cpp MoE arithmetic) vs W4A16 "
                    "reference, gate_up %.3g down %.3g\n",
                    sh.name, fmt_name, T, (long long)rows, r.active.size(),
                    std::max(ratio_of(true, nullptr, true),
                             ratio_of(false, nullptr, true)),
                    std::max(ratio_of(true, nullptr, true),
                             ratio_of(false, nullptr, true)) <= 1
                        ? "pass"
                        : "FAIL",
                    ratio_of(true, nullptr, true),
                    ratio_of(false, nullptr, true));
      struct Cand {
        std::string name;
        std::function<int32_t(int)> run; // copy index
        std::string note;
      };
      std::vector<Cand> cands;
      // own_loop candidates
      for (int32_t qi = 0; qi < nq; ++qi) {
        for (int fused = 0; fused < 2; ++fused) {
          turbine_qgemm_desc q{};
          q.m = std::max<int64_t>(1, r.offsets[r.active.front() + 1] -
                                         r.offsets[r.active.front()]);
          q.c_dtype = TURBINE_DTYPE_BF16;
          q.alpha = 1.0f;
          q.prefill = T >= 256 ? 1 : 0;
          if (sh.fmt == Fmt::Fp8Block) {
            if (fused)
              continue;
            q.scheme = TURBINE_QSCHEME_FP8_BLOCK;
            q.block_n = q.block_k = 128;
            q.act_quant = TURBINE_ACTQ_NONE;
            q.a_dtype = TURBINE_DTYPE_BF16;
          } else if (sh.fmt == Fmt::Mxfp4) {
            if (fused)
              continue;
            q.scheme = TURBINE_QSCHEME_MXFP4;
            q.group_size = 32;
            q.act_quant = TURBINE_ACTQ_NONE;
            q.a_dtype = TURBINE_DTYPE_BF16;
          } else {
            q.scheme = TURBINE_QSCHEME_FP8_TENSOR;
            q.act_quant = TURBINE_ACTQ_FP8_TENSOR;
            q.a_dtype = fused ? TURBINE_DTYPE_BF16 : TURBINE_DTYPE_F8E4M3;
            q.act_static_scale = act_scale;
          }
          // every active expert's call must be accepted
          bool ok = true;
          for (int e : r.active) {
            const int64_t m = r.offsets[e + 1] - r.offsets[e];
            turbine_qgemm_desc a = q, b = q;
            a.m = b.m = m;
            a.n = 2 * I, a.k = H, a.lda = H, a.ldc = 2 * I;
            b.n = H, b.k = I, b.lda = I, b.ldc = H;
            if (turbine_impl_supports(TURBINE_OP_QGEMM, qi, &a) != 1 ||
                turbine_impl_supports(TURBINE_OP_QGEMM, qi, &b) != 1) {
              ok = false;
              break;
            }
          }
          const std::string name =
              "own_loop:" + qnames[qi] +
              (sh.fmt == Fmt::Fp8Tensor ? (fused ? "+fusedq" : "+quant_act")
                                        : "");
          if (!ok) {
            std::printf("%s %s %d %lld %zu both %s - - refused not accepted "
                        "for every expert call\n",
                        sh.name, fmt_name, T, (long long)rows, r.active.size(),
                        name.c_str());
            continue;
          }
          const bool w8a8 = sh.fmt == Fmt::Fp8Tensor;
          cands.push_back(
              {name,
               [&, q, qi, fused, w8a8, rows](int copy) mutable -> int32_t {
                 const int ci = copy % gu.copies, cd = copy % dn.copies;
                 const void *agu = d_xgu, *adn = d_xdn;
                 if (w8a8 && !fused) {
                   for (int pj = 0; pj < 2; ++pj) {
                     turbine_quantize_act_desc qa{};
                     qa.x = pj ? d_xdn : d_xgu;
                     qa.out = pj ? d_q8dn : d_q8gu;
                     qa.scales = pj ? d_sdn : d_sgu;
                     qa.rows = rows;
                     qa.cols = pj ? I : H;
                     qa.x_stride_row = qa.out_stride_row = qa.cols;
                     qa.mode = TURBINE_ACTQ_FP8_TENSOR;
                     qa.static_scale = act_scale;
                     qa.x_dtype = TURBINE_DTYPE_BF16;
                     qa.out_dtype = TURBINE_DTYPE_F8E4M3;
                     const int32_t rc = turbine_quantize_act(ctx, &qa);
                     if (rc != TURBINE_OK)
                       return rc;
                   }
                   agu = d_q8gu;
                   adn = d_q8dn;
                 }
                 const size_t esz = (w8a8 && !fused) ? 1 : 2;
                 for (int e : r.active) {
                   const int64_t b = r.offsets[e], m = r.offsets[e + 1] - b;
                   turbine_qgemm_desc a = q;
                   a.m = m;
                   a.n = 2 * I, a.k = H, a.lda = H, a.ldc = 2 * I;
                   a.a = static_cast<const char *>(agu) + b * H * esz;
                   a.a_scales = (w8a8 && !fused) ? d_sgu : nullptr;
                   a.b = static_cast<const char *>(gu.d_codes[ci]) +
                         e * gu.code_bytes();
                   a.b_scales = static_cast<const char *>(gu.d_scales[ci]) +
                                e * gu.scale_bytes();
                   a.c = static_cast<char *>(d_ogu) + b * 2 * I * 2;
                   int32_t rc = turbine_impl_run(ctx, TURBINE_OP_QGEMM, qi, &a);
                   if (rc != TURBINE_OK)
                     return rc;
                   turbine_qgemm_desc d = q;
                   d.m = m;
                   d.n = H, d.k = I, d.lda = I, d.ldc = H;
                   d.a = static_cast<const char *>(adn) + b * I * esz;
                   d.a_scales = (w8a8 && !fused) ? d_sdn : nullptr;
                   d.b = static_cast<const char *>(dn.d_codes[cd]) +
                         e * dn.code_bytes();
                   d.b_scales = static_cast<const char *>(dn.d_scales[cd]) +
                                e * dn.scale_bytes();
                   d.c = static_cast<char *>(d_odn) + b * H * 2;
                   rc = turbine_impl_run(ctx, TURBINE_OP_QGEMM, qi, &d);
                   if (rc != TURBINE_OK)
                     return rc;
                 }
                 return TURBINE_OK;
               },
               ""});
        }
      }
      for (Cand &cd : cands) {
        if (!listed(cands_arg, cd.name))
          continue;
        int32_t rc = cd.run(0);
        if (rc == TURBINE_OK)
          rc = turbine_stream_sync(ctx);
        if (rc != TURBINE_OK) {
          std::printf("%s %s %d %lld %zu both %s - - error %s\n", sh.name,
                      fmt_name, T, (long long)rows, r.active.size(),
                      cd.name.c_str(), last_err(ctx).c_str());
          continue;
        }
        std::vector<uint16_t> ggu(rows * 2 * I), gdn(rows * H);
        check_hip(
            hipMemcpy(ggu.data(), d_ogu, ggu.size() * 2, hipMemcpyDeviceToHost),
            "o");
        check_hip(
            hipMemcpy(gdn.data(), d_odn, gdn.size() * 2, hipMemcpyDeviceToHost),
            "o");
        const double rg = ratio_of(true, &ggu, false),
                     rd = ratio_of(false, &gdn, false);
        double us = -1;
        if (!check_only) {
          int copy = 1;
          for (int i = 0; i < 3; ++i)
            cd.run(copy++);
          turbine_stream_sync(ctx);
          std::vector<double> per;
          for (int rd2 = 0; rd2 < rounds; ++rd2) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i)
              cd.run(copy++);
            turbine_stream_sync(ctx);
            per.push_back(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count() /
                          iters);
          }
          us = median(per);
        }
        std::printf(
            "%s %s %d %lld %zu both %s %.1f %.3g %s gate_up %.3g down %.3g\n",
            sh.name, fmt_name, T, (long long)rows, r.active.size(),
            cd.name.c_str(), us, std::max(rg, rd),
            std::max(rg, rd) <= 1 ? "pass" : "FAIL", rg, rd);
        std::fflush(stdout);
      }
      // hipBLASLt grouped probes
      if (listed(cands_arg, "hipblaslt_grouped")) {
        std::vector<int64_t> grows;
        for (int e : r.active)
          grows.push_back(r.offsets[e + 1] - r.offsets[e]);
        static float one = 1.0f;
        const int f8f8 = probe_grouped(blaslt, HIP_R_8F_E4M3, HIP_R_8F_E4M3,
                                       grows, 2 * I, H, d_ogu, &one);
        const int f8bf = probe_grouped(blaslt, HIP_R_8F_E4M3, HIP_R_16BF, grows,
                                       2 * I, H, d_ogu, &one);
        const int bfbf = probe_grouped(blaslt, HIP_R_16BF, HIP_R_16BF, grows,
                                       2 * I, H, d_ogu, &one);
        std::printf("%s %s %d %lld %zu gate_up hipblaslt_grouped_probe - - - "
                    "solutions: fp8xfp8 %d, fp8xbf16 %d, "
                    "bf16xbf16 %d (negative: setProblem/heuristic error)\n",
                    sh.name, fmt_name, T, (long long)rows, r.active.size(),
                    f8f8, f8bf, bfbf);
      }
      // BF16 fused MoE of the same shape (reference cost)
      if (listed(cands_arg, "bf16_moe_fused") && !check_only) {
        static void *wg = nullptr, *wu = nullptr, *wd = nullptr, *ws = nullptr;
        static const Shape *wsh = nullptr;
        const size_t ew = static_cast<size_t>(sh.experts) * I * H * 2;
        if (wsh != &sh) {
          if (wg) {
            hipFree(wg);
            hipFree(wu);
            hipFree(wd);
          }
          check_hip(hipMalloc(&wg, ew), "m");
          check_hip(hipMalloc(&wu, ew), "m");
          check_hip(hipMalloc(&wd, ew), "m");
          check_hip(hipMemset(wg, 0x3c, ew), "s"); // small BF16 values
          check_hip(hipMemset(wu, 0x3c, ew), "s");
          check_hip(hipMemset(wd, 0x3c, ew), "s");
          if (!ws)
            check_hip(hipMalloc(&ws, 1ull << 30), "m");
          wsh = &sh;
        }
        int32_t *d_rows, *d_off;
        float *d_w;
        check_hip(hipMalloc(&d_rows, r.sorted_rows.size() * 4), "m");
        check_hip(hipMalloc(&d_off, r.offsets.size() * 4), "m");
        check_hip(hipMalloc(&d_w, r.weights.size() * 4), "m");
        hipMemcpy(d_rows, r.sorted_rows.data(), r.sorted_rows.size() * 4,
                  hipMemcpyHostToDevice);
        hipMemcpy(d_off, r.offsets.data(), r.offsets.size() * 4,
                  hipMemcpyHostToDevice);
        hipMemcpy(d_w, r.weights.data(), r.weights.size() * 4,
                  hipMemcpyHostToDevice);
        turbine_moe_experts_desc m{};
        m.x = d_xgu;
        m.w_gate = wg;
        m.w_up = wu;
        m.w_down = wd;
        m.sorted_rows = d_rows;
        m.expert_offsets = d_off;
        m.topk_weights = d_w;
        m.host_expert_offsets = r.offsets.data();
        m.out = d_odn;
        m.workspace = ws;
        m.workspace_bytes = 1ull << 30;
        m.num_tokens = T;
        m.hidden = static_cast<int32_t>(H);
        m.inter = static_cast<int32_t>(I);
        m.top_k = sh.topk;
        m.num_experts = sh.experts;
        m.expert_begin = 0;
        m.expert_end = sh.experts;
        m.dtype = TURBINE_DTYPE_BF16;
        int32_t rc = turbine_moe_experts(ctx, &m);
        if (rc == TURBINE_OK)
          rc = turbine_stream_sync(ctx);
        if (rc != TURBINE_OK) {
          std::printf("%s %s %d %lld %zu fused bf16_moe_fused - - error %s\n",
                      sh.name, fmt_name, T, (long long)rows, r.active.size(),
                      last_err(ctx).c_str());
        } else {
          std::vector<double> per;
          for (int rd2 = 0; rd2 < rounds; ++rd2) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i)
              turbine_moe_experts(ctx, &m);
            turbine_stream_sync(ctx);
            per.push_back(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count() /
                          iters);
          }
          std::printf("%s %s %d %lld %zu fused bf16_moe_fused[%s] %.1f - - "
                      "BF16 weights (one copy; full op incl. "
                      "SiLU+scatter)\n",
                      sh.name, fmt_name, T, (long long)rows, r.active.size(),
                      turbine_moe_experts_impl(&m), median(per));
        }
        hipFree(d_rows);
        hipFree(d_off);
        hipFree(d_w);
      }
      std::fflush(stdout);
    }
    free_proj(gu);
    free_proj(dn);
    for (void *p : {d_xgu, d_xdn, d_q8gu, d_q8dn, d_ogu, d_odn})
      hipFree(p);
    hipFree(d_sgu);
    hipFree(d_sdn);
  }
  hipblasLtDestroy(blaslt);
  turbine_ctx_destroy(ctx);
  return 0;
}
