// Mixture-of-experts ops (ABI v2): moe_route and moe_experts.
//
// moe_route runs the Turbine routing kernels (moe.hip). moe_experts computes
// out[t] += sum_k w_k * down(silu(gate(x_t)) * up(x_t)) over the local experts
// [expert_begin, expert_end):
//   1. gather: the rows routed to local experts are contiguous in sorted_rows
//      (grouped by ascending expert), so one kernel copies their activations
//      into xs in that order and records each row's position;
//   2. gate and up GEMMs per expert on its slice of xs (BF16 out, F32
//      accumulate), SiLU * up in place over all rows (turbine_silu_mul), the
//      down GEMM per expert;
//   3. scatter: every token adds its experts' outputs in ascending expert
//      order (moe.hip), so the result does not depend on the GEMM batching.
// hipBLASLt ships no grouped-GEMM solution for gfx1201 in ROCm 7.14.1, so step
// 2 issues one hipblasLtMatmul per expert and projection (impl
// "hipblaslt_per_expert"). When a context finds a grouped solution at creation
// (probe_grouped_gemm), each projection runs as one hipblaslt_ext::GroupedGemm
// over the experts instead (impl "hipblaslt_grouped"), falling back to the
// per-expert calls for a batch the grouped heuristic rejects.
//
// Intermediates live in the caller's workspace when it is large enough,
// otherwise in a context-owned scratch buffer grown on demand.
#include <hipblaslt/hipblaslt-ext.hpp>

#include <atomic>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "turbine_hip.hpp"

using turbine_hip::check_blaslt;
using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImplRoute = "turbine_hip";
constexpr const char *kImplPerExpert = "hipblaslt_per_expert";
constexpr const char *kImplGrouped = "hipblaslt_grouped";
constexpr size_t kAlign = 256;

// Set once any context of this process found a grouped-GEMM solution; read by
// the context-free turbine_moe_experts_impl.
std::atomic<bool> g_grouped_available{false};

size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

bool route_supported(const turbine_moe_route_desc *d) {
  if (d == nullptr)
    return false;
  if (d->num_experts < 1 || d->num_experts > turbine_hip::kMoeMaxExperts)
    return false;
  if (d->top_k < 1 || d->top_k > d->num_experts ||
      d->top_k > turbine_hip::kMoeMaxTopK) {
    return false;
  }
  if (d->renormalize != 0 && d->renormalize != 1)
    return false;
  return d->num_tokens >= 0 &&
         static_cast<int64_t>(d->num_tokens) * d->top_k <= INT32_MAX;
}

bool experts_supported(const turbine_moe_experts_desc *d) {
  if (d == nullptr || d->dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->hidden < 1 || d->inter < 1 || d->num_tokens < 0)
    return false;
  if (d->top_k < 1 || d->top_k > d->num_experts ||
      d->top_k > turbine_hip::kMoeMaxTopK) {
    return false;
  }
  if (d->expert_begin < 0 || d->expert_begin >= d->expert_end ||
      d->expert_end > d->num_experts) {
    return false;
  }
  return static_cast<int64_t>(d->num_tokens) * d->top_k <= INT32_MAX;
}

std::string describe(const turbine_moe_experts_desc *d) {
  return "tokens=" + std::to_string(d->num_tokens) +
         " hidden=" + std::to_string(d->hidden) +
         " inter=" + std::to_string(d->inter) +
         " top_k=" + std::to_string(d->top_k) +
         " experts=" + std::to_string(d->num_experts) + " local=[" +
         std::to_string(d->expert_begin) + "," + std::to_string(d->expert_end) +
         ")" + " dtype=" + std::to_string(d->dtype);
}

// One projection of one expert: out[rows, n_out] = act[rows, depth] .
// weight[n_out, depth]^T, all row-major BF16.
struct Projection {
  const void *act;
  const void *weight;
  void *out;
  int64_t rows;
  int64_t n_out;
  int64_t depth;
};

int32_t run_one(turbine_ctx *ctx, const Projection &p) {
  turbine_gemm_desc g{};
  g.a = p.act;
  g.b = p.weight;
  g.c = p.out;
  g.m = p.rows;
  g.n = p.n_out;
  g.k = p.depth;
  g.lda = p.depth;
  g.ldb = p.depth;
  g.ldc = p.n_out;
  g.trans_b = 1;
  g.a_dtype = TURBINE_DTYPE_BF16;
  g.b_dtype = TURBINE_DTYPE_BF16;
  g.c_dtype = TURBINE_DTYPE_BF16;
  g.alpha = 1.0f;
  g.beta = 0.0f;
  return turbine_gemm(ctx, &g);
}

const float kOne = 1.0f;
const float kZero = 0.0f;

// Runs the projections as one grouped GEMM. In hipBLASLt's column-major view
// each is D[n_out, rows] = W^T[n_out, depth] . A[depth, rows] with packed
// leading dimensions (opA = T on the weight, opB = N on the activations).
// Returns TURBINE_E_UNSUPPORTED (without recording an error) when the
// heuristic has no solution for this batch.
int32_t run_grouped(turbine_ctx *ctx, const std::vector<Projection> &ps) {
  try {
    hipblaslt_ext::GroupedGemm gemm(ctx->blaslt, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                    HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                                    HIP_R_16BF, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m;
    std::vector<int64_t> n;
    std::vector<int64_t> k;
    std::vector<int64_t> batch;
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(ps.size());
    std::vector<hipblaslt_ext::GemmInputs> inputs(ps.size());
    for (size_t i = 0; i < ps.size(); ++i) {
      m.push_back(ps[i].n_out);
      n.push_back(ps[i].rows);
      k.push_back(ps[i].depth);
      batch.push_back(1);
      inputs[i].setA(ps[i].weight);
      inputs[i].setB(ps[i].act);
      inputs[i].setC(ps[i].out);
      inputs[i].setD(ps[i].out);
      inputs[i].setAlpha(&kOne);
      inputs[i].setBeta(&kZero);
    }
    int32_t rc =
        check_blaslt(ctx, gemm.setProblem(m, n, k, batch, epilogue, inputs),
                     "GroupedGemm::setProblem");
    if (rc != TURBINE_OK)
      return rc;
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(turbine_hip::kGemmWorkspaceBytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    if (gemm.algoGetHeuristic(1, pref, results) != HIPBLAS_STATUS_SUCCESS ||
        results.empty()) {
      return TURBINE_E_UNSUPPORTED;
    }
    rc = check_blaslt(ctx,
                      gemm.initialize(results[0].algo, ctx->workspace,
                                      /*useUserArgs=*/false, ctx->stream),
                      "GroupedGemm::initialize");
    if (rc != TURBINE_OK)
      return rc;
    return check_blaslt(ctx, gemm.run(ctx->stream), "GroupedGemm::run");
  } catch (const std::exception &e) {
    return fail(ctx, TURBINE_E_LIBRARY,
                std::string("hipblaslt_ext::GroupedGemm: ") + e.what());
  }
}

int32_t run_projections(turbine_ctx *ctx, const std::vector<Projection> &ps) {
  if (ps.empty())
    return TURBINE_OK;
  if (ctx->moe_grouped && ps.size() > 1) {
    const int32_t rc = run_grouped(ctx, ps);
    if (rc != TURBINE_E_UNSUPPORTED)
      return rc;
  }
  for (const Projection &p : ps) {
    if (int32_t rc = run_one(ctx, p); rc != TURBINE_OK)
      return rc;
  }
  return TURBINE_OK;
}

// Returns device memory of at least bytes: the caller's workspace when it is
// large enough, else the context's scratch (grown after draining the stream,
// which may still use the old buffer).
int32_t scratch(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                size_t bytes, char **out) {
  if (d->workspace != nullptr && d->workspace_bytes >= bytes) {
    *out = static_cast<char *>(d->workspace);
    return TURBINE_OK;
  }
  if (ctx->moe_scratch_bytes < bytes) {
    if (ctx->moe_scratch != nullptr) {
      if (int32_t rc = check_hip(ctx, hipStreamSynchronize(ctx->stream),
                                 "hipStreamSynchronize (MoE scratch)");
          rc != TURBINE_OK) {
        return rc;
      }
      (void)hipFree(ctx->moe_scratch);
      ctx->moe_scratch = nullptr;
      ctx->moe_scratch_bytes = 0;
    }
    const std::string what =
        "hipMalloc MoE scratch " + std::to_string(bytes) + " bytes";
    if (int32_t rc =
            check_hip(ctx, hipMalloc(&ctx->moe_scratch, bytes), what.c_str());
        rc != TURBINE_OK) {
      ctx->moe_scratch = nullptr;
      return rc;
    }
    ctx->moe_scratch_bytes = bytes;
  }
  *out = static_cast<char *>(ctx->moe_scratch);
  return TURBINE_OK;
}

int32_t run_experts(turbine_ctx *ctx, const turbine_moe_experts_desc *d) {
  const int32_t *host = d->host_expert_offsets;
  const int64_t all_rows = static_cast<int64_t>(d->num_tokens) * d->top_k;
  if (host[0] != 0 || host[d->num_experts] != all_rows) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: host_expert_offsets must run from 0 to " +
                    std::to_string(all_rows));
  }
  for (int32_t e = 0; e < d->num_experts; ++e) {
    if (host[e + 1] < host[e]) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_moe_experts: host_expert_offsets decrease at "
                  "expert " +
                      std::to_string(e));
    }
  }
  const int64_t first = host[d->expert_begin];
  const int64_t rows = host[d->expert_end] - first;
  if (rows == 0)
    return TURBINE_OK;

  const size_t h = static_cast<size_t>(d->hidden);
  const size_t inter = static_cast<size_t>(d->inter);
  const size_t n = static_cast<size_t>(rows);
  const size_t pos_bytes = align_up(static_cast<size_t>(all_rows) * 4);
  const size_t xs_bytes = align_up(n * h * 2);
  const size_t mid_bytes = align_up(n * inter * 2);
  const size_t down_bytes = align_up(n * h * 2);
  char *ws = nullptr;
  if (int32_t rc = scratch(
          ctx, d, pos_bytes + xs_bytes + 2 * mid_bytes + down_bytes, &ws);
      rc != TURBINE_OK) {
    return rc;
  }
  auto *pos = reinterpret_cast<int32_t *>(ws);
  char *xs = ws + pos_bytes;
  char *gate = xs + xs_bytes;
  char *up = gate + mid_bytes;
  char *down = up + mid_bytes;

  int32_t rc = turbine_hip::launch_moe_clear_positions(ctx, pos, all_rows);
  if (rc != TURBINE_OK)
    return rc;
  rc = turbine_hip::launch_moe_gather(ctx, d->x, d->sorted_rows + first, rows,
                                      d->hidden, d->top_k, xs, pos);
  if (rc != TURBINE_OK)
    return rc;

  // Per-expert slices of the gathered rows and of the weights.
  const auto *w_gate = static_cast<const char *>(d->w_gate);
  const auto *w_up = static_cast<const char *>(d->w_up);
  const auto *w_down = static_cast<const char *>(d->w_down);
  // Every expert's gate, up and down matrix holds inter * hidden BF16s.
  const size_t expert_weight_bytes = inter * h * 2;
  std::vector<Projection> gate_ps;
  std::vector<Projection> up_ps;
  std::vector<Projection> down_ps;
  for (int32_t e = d->expert_begin; e < d->expert_end; ++e) {
    const int64_t count = host[e + 1] - host[e];
    if (count == 0)
      continue;
    const size_t off = static_cast<size_t>(host[e] - first);
    const size_t local = static_cast<size_t>(e - d->expert_begin);
    const void *x_e = xs + off * h * 2;
    char *gate_e = gate + off * inter * 2;
    char *up_e = up + off * inter * 2;
    gate_ps.push_back({x_e, w_gate + local * expert_weight_bytes, gate_e, count,
                       d->inter, d->hidden});
    up_ps.push_back({x_e, w_up + local * expert_weight_bytes, up_e, count,
                     d->inter, d->hidden});
    down_ps.push_back({gate_e, w_down + local * expert_weight_bytes,
                       down + off * h * 2, count, d->hidden, d->inter});
  }
  if (rc = run_projections(ctx, gate_ps); rc != TURBINE_OK)
    return rc;
  if (rc = run_projections(ctx, up_ps); rc != TURBINE_OK)
    return rc;

  // silu(gate) * up, written over gate (each element reads then writes the
  // same index).
  turbine_silu_mul_desc act{};
  act.gate = gate;
  act.up = up;
  act.out = gate;
  act.rows = rows;
  act.cols = d->inter;
  act.gate_stride_row = d->inter;
  act.up_stride_row = d->inter;
  act.out_stride_row = d->inter;
  act.dtype = TURBINE_DTYPE_BF16;
  if (rc = turbine_silu_mul(ctx, &act); rc != TURBINE_OK)
    return rc;

  if (rc = run_projections(ctx, down_ps); rc != TURBINE_OK)
    return rc;
  return turbine_hip::launch_moe_scatter(ctx, down, pos, d->topk_weights,
                                         d->num_tokens, d->hidden, d->top_k,
                                         d->out);
}

} // namespace

namespace turbine_hip {

bool probe_grouped_gemm(turbine_ctx *ctx) {
  try {
    // A representative two-expert problem (1024 x 16 x 2048 each); the
    // pointers are never dereferenced by the heuristic query.
    hipblaslt_ext::GroupedGemm gemm(ctx->blaslt, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                    HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                                    HIP_R_16BF, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m{1024, 1024};
    std::vector<int64_t> n{16, 16};
    std::vector<int64_t> k{2048, 2048};
    std::vector<int64_t> batch{1, 1};
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(2);
    std::vector<hipblaslt_ext::GemmInputs> inputs(2);
    for (auto &in : inputs) {
      in.setA(ctx->workspace);
      in.setB(ctx->workspace);
      in.setC(ctx->workspace);
      in.setD(ctx->workspace);
      in.setAlpha(&kOne);
      in.setBeta(&kZero);
    }
    if (gemm.setProblem(m, n, k, batch, epilogue, inputs) !=
        HIPBLAS_STATUS_SUCCESS) {
      return false;
    }
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(kGemmWorkspaceBytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    const bool found =
        gemm.algoGetHeuristic(1, pref, results) == HIPBLAS_STATUS_SUCCESS &&
        !results.empty();
    if (found)
      g_grouped_available.store(true);
    return found;
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "turbine_hip: grouped GEMM probe failed (%s); MoE experts "
                 "use per-expert hipBLASLt GEMMs\n",
                 e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "turbine_hip: grouped GEMM probe failed; MoE experts "
                         "use per-expert hipBLASLt GEMMs\n");
    return false;
  }
}

} // namespace turbine_hip

extern "C" {

int32_t turbine_moe_route_supported(const turbine_moe_route_desc *d) {
  return route_supported(d) ? 1 : 0;
}

const char *turbine_moe_route_impl(const turbine_moe_route_desc *d) {
  (void)d;
  return kImplRoute;
}

int32_t turbine_moe_route(turbine_ctx *ctx, const turbine_moe_route_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_route: descriptor is NULL");
  }
  if (!route_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_moe_route: unsupported configuration tokens=" +
                    std::to_string(d->num_tokens) +
                    " experts=" + std::to_string(d->num_experts) +
                    " top_k=" + std::to_string(d->top_k) +
                    " renormalize=" + std::to_string(d->renormalize));
  }
  if (d->expert_offsets == nullptr ||
      (d->num_tokens > 0 &&
       (d->router_logits == nullptr || d->topk_ids == nullptr ||
        d->topk_weights == nullptr || d->sorted_rows == nullptr))) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_moe_route: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return turbine_hip::launch_moe_route(ctx, d);
}

int32_t turbine_moe_experts_supported(const turbine_moe_experts_desc *d) {
  return experts_supported(d) ? 1 : 0;
}

const char *turbine_moe_experts_impl(const turbine_moe_experts_desc *d) {
  (void)d;
  return g_grouped_available.load() ? kImplGrouped : kImplPerExpert;
}

int32_t turbine_moe_experts(turbine_ctx *ctx,
                            const turbine_moe_experts_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: descriptor is NULL");
  }
  if (!experts_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_moe_experts: unsupported configuration " +
                    describe(d));
  }
  if (d->host_expert_offsets == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: host_expert_offsets is NULL");
  }
  if (d->num_tokens == 0)
    return TURBINE_OK;
  if (d->x == nullptr || d->w_gate == nullptr || d->w_up == nullptr ||
      d->w_down == nullptr || d->sorted_rows == nullptr ||
      d->topk_weights == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_moe_experts: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return run_experts(ctx, d);
}

} // extern "C"
