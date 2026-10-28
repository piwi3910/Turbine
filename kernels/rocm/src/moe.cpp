// Mixture-of-experts ops (ABI v2): moe_route and moe_experts.
//
// moe_route runs the Turbine routing kernels (moe.hip). moe_experts computes
// out[t] += sum_k w_k * down(silu(gate(x_t)) * up(x_t)) over the local experts
// [expert_begin, expert_end).
//
// Six implementations (impl_table.cpp; the sixth, "turbine_hip_moe_quant",
// runs the v2.16 quantized experts, see run_quant below; the caller picks one
// with turbine_impl_run, or turbine_moe_experts takes the first that supports
// the descriptor within the context's card profile): the Turbine small-m
// kernels (moe_small_m.hip, impl "turbine_hip_moe_small_m"; the default uses
// them up to the profile's moe_small_max_rows routed rows, num_tokens * top_k;
// for hidden and inter multiples of 64 they are WMMA kernels with the grouped
// tier's per-element chain, bitwise equal to the next tiers row by row), the
// Turbine prefill WMMA kernels (moe_grouped.hip, impl
// "turbine_hip_moe_wmma_prefill"; hidden and inter multiples of 256; the
// default's choice above the small-m tier), the Turbine grouped WMMA kernels
// (moe_grouped.hip, impl "turbine_hip_moe_wmma"; hidden and inter multiples of
// 64), all three reading the group sizes from expert_offsets on the device
// (host_expert_offsets may be NULL, and
// turbine_moe_experts_needs_host_offsets says so), and the hipBLASLt paths,
// which read host_expert_offsets:
//   1. gather: the rows routed to local experts are contiguous in sorted_rows
//      (grouped by ascending expert), so one kernel copies their activations
//      into xs in that order and records each row's position;
//   2. gate and up GEMMs per expert on its slice of xs (BF16 out, F32
//      accumulate), SiLU * up in place over all rows (turbine_silu_mul), the
//      down GEMM per expert;
//   3. scatter: every token adds its experts' outputs in ascending expert
//      order (moe.hip), so the result does not depend on the GEMM batching.
// "hipblaslt_per_expert" issues one hipblasLtMatmul per expert and projection
// in step 2 (hipBLASLt ships no grouped-GEMM solution for RDNA4 in ROCm
// 7.14.1). When a context found a grouped solution at creation
// (probe_grouped_gemm), "hipblaslt_grouped" runs each projection as one
// hipblaslt_ext::GroupedGemm over the experts instead, falling back to the
// per-expert calls for a batch the grouped heuristic rejects.
//
// Intermediates live in the caller's workspace when it is large enough,
// otherwise in a context-owned scratch buffer grown on demand.
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
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
constexpr const char *kImplSmallM = "turbine_hip_moe_small_m";
constexpr const char *kImplWmmaPrefill = "turbine_hip_moe_wmma_prefill";
constexpr const char *kImplWmma = "turbine_hip_moe_wmma";
constexpr const char *kImplQuant = "turbine_hip_moe_quant";
constexpr const char *kImplQuantLoop = "turbine_hip_moe_quant_loop";
constexpr const char *kImplQuantGrouped = "turbine_hip_moe_quant_grouped";
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
  if ((d->flags &
       ~(TURBINE_MOE_ROUTE_RENORMALIZE | TURBINE_MOE_ROUTE_BF16_LOGITS |
         TURBINE_ROUTE_TOPK_THEN_SOFTMAX)) != 0)
    return false;
  return d->num_tokens >= 0 &&
         static_cast<int64_t>(d->num_tokens) * d->top_k <= INT32_MAX;
}

bool quantized(const turbine_moe_experts_desc *d) {
  return d->weight_dtype != TURBINE_EXPERT_BF16;
}

// The pre-v2.16 form: BF16 weights, SwiGLU, no biases or shared expert.
bool plain_bf16(const turbine_moe_experts_desc *d) {
  return d->weight_dtype == TURBINE_EXPERT_BF16 &&
         d->activation == TURBINE_EXPERT_SWIGLU &&
         d->gate_up_interleaved == 0 && d->w_gate_scale == nullptr &&
         d->w_up_scale == nullptr && d->w_down_scale == nullptr &&
         d->act_static_scale == nullptr &&
         d->act_static_scale_down == nullptr && d->b_gate == nullptr &&
         d->b_up == nullptr && d->b_down == nullptr &&
         d->shared_w_gate == nullptr && d->shared_w_up == nullptr &&
         d->shared_w_down == nullptr && d->shared_gate_w == nullptr &&
         d->shared_inter == 0;
}

size_t weight_code_bytes(int32_t weight_dtype, int64_t n, int64_t k) {
  return static_cast<size_t>(weight_dtype == TURBINE_EXPERT_MXFP4 ? n * k / 2
                                                                  : n * k);
}

size_t weight_scale_bytes(int32_t weight_dtype, int64_t n, int64_t k) {
  switch (weight_dtype) {
  case TURBINE_EXPERT_FP8_BLOCK:
    return static_cast<size_t>(((n + 127) / 128) * ((k + 127) / 128)) * 4;
  case TURBINE_EXPERT_FP8_TENSOR:
    return 4;
  default:
    return static_cast<size_t>(n * k / 32);
  }
}

// One expert projection as a qgemm call (the own implementations of v2.9):
// c[m, n] = a[m, k] . dequant(b)[n, k]^T. Always a "prefill" call so that a
// row's output does not depend on how many rows its expert got (the batch
// invariance of the loop); the W8A8 activation form is set by the caller.
turbine_qgemm_desc qgemm_for(const turbine_moe_experts_desc *d, int64_t m,
                             int64_t n, int64_t k) {
  turbine_qgemm_desc q{};
  q.m = m;
  q.n = n;
  q.k = k;
  q.lda = k;
  q.ldc = n;
  q.c_dtype = TURBINE_DTYPE_BF16;
  q.alpha = 1.0f;
  q.prefill = 1;
  q.a_dtype = TURBINE_DTYPE_BF16;
  q.act_quant = TURBINE_ACTQ_NONE;
  switch (d->weight_dtype) {
  case TURBINE_EXPERT_FP8_BLOCK:
    q.scheme = TURBINE_QSCHEME_FP8_BLOCK;
    q.block_n = q.block_k = 128;
    break;
  case TURBINE_EXPERT_FP8_TENSOR:
    q.scheme = TURBINE_QSCHEME_FP8_TENSOR;
    q.act_quant = TURBINE_ACTQ_FP8_TENSOR;
    q.a_dtype = TURBINE_DTYPE_F8E4M3;
    break;
  default:
    q.scheme = TURBINE_QSCHEME_MXFP4;
    q.group_size = 32;
    break;
  }
  return q;
}

// The v2.16 quantized experts: the structure of the descriptor and whether the
// qgemm implementations take the three projection shapes.
bool quant_supported(const turbine_moe_experts_desc *d) {
  const int32_t wd = d->weight_dtype;
  if (wd != TURBINE_EXPERT_FP8_BLOCK && wd != TURBINE_EXPERT_FP8_TENSOR &&
      wd != TURBINE_EXPERT_MXFP4) {
    return false;
  }
  if (d->activation != TURBINE_EXPERT_SWIGLU &&
      d->activation != TURBINE_EXPERT_GPTOSS_CLAMP) {
    return false;
  }
  if (d->activation == TURBINE_EXPERT_GPTOSS_CLAMP &&
      !(d->clamp_limit > 0.0f && d->clamp_limit < INFINITY)) {
    return false;
  }
  if (d->gate_up_interleaved != 0 && d->gate_up_interleaved != 1)
    return false;
  const bool interleaved = d->gate_up_interleaved != 0;
  // Support is decided by the configuration alone (the Rust shim probes with
  // NULL pointers); the operands are checked by run_quant. An FP8 tensor
  // expert with BF16 activations (no act_static_scale) has no qgemm
  // implementation and is refused there.
  if (d->shared_inter < 0)
    return false;
  const int64_t gate_n = interleaved ? 2 * static_cast<int64_t>(d->inter)
                                     : static_cast<int64_t>(d->inter);
  const turbine_qgemm_desc gu = qgemm_for(d, 1, gate_n, d->hidden);
  const turbine_qgemm_desc dn = qgemm_for(d, 1, d->hidden, d->inter);
  return turbine_qgemm_supported(&gu) == 1 && turbine_qgemm_supported(&dn) == 1;
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
  if (static_cast<int64_t>(d->num_tokens) * d->top_k > INT32_MAX)
    return false;
  // BF16 experts run only in the pre-v2.16 form; the v2.16 variants are the
  // quantized experts of turbine_hip_moe_quant.
  return quantized(d) ? quant_supported(d) : plain_bf16(d);
}

bool aligned16(const void *p) {
  return reinterpret_cast<uintptr_t>(p) % 16 == 0;
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

// grouped: try one grouped GEMM per projection first (hipblaslt_grouped), else
// one GEMM per expert (hipblaslt_per_expert).
int32_t run_projections(turbine_ctx *ctx, const std::vector<Projection> &ps,
                        bool grouped) {
  if (ps.empty())
    return TURBINE_OK;
  if (grouped && ps.size() > 1) {
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
    if (ctx->capturing) {
      return turbine_hip::refuse_while_capturing(
          ctx, "turbine_moe_experts (growing the MoE scratch)");
    }
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

int32_t run_experts(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                    bool grouped) {
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
  if (rc = run_projections(ctx, gate_ps, grouped); rc != TURBINE_OK)
    return rc;
  if (rc = run_projections(ctx, up_ps, grouped); rc != TURBINE_OK)
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

  if (rc = run_projections(ctx, down_ps, grouped); rc != TURBINE_OK)
    return rc;
  return turbine_hip::launch_moe_scatter(ctx, down, pos, d->topk_weights,
                                         d->num_tokens, d->hidden, d->top_k,
                                         d->out);
}

// The device-offset paths (small-m, prefill WMMA, grouped WMMA): positions,
// fused gate-up with SiLU, down, then the ordered scatter; no host data.
int32_t run_device_offsets(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                           turbine_hip::MoePath path) {
  using turbine_hip::MoePath;
  if (!aligned16(d->x) || !aligned16(d->w_gate) || !aligned16(d->w_up) ||
      !aligned16(d->w_down)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string("turbine_moe_experts: implementation ") +
                    turbine_hip::moe_path_name(path) +
                    " needs 16-byte aligned x and expert weights");
  }
  const size_t rows = static_cast<size_t>(d->num_tokens) * d->top_k;
  const size_t pos_bytes = align_up(rows * 4);
  const size_t act_bytes = align_up(rows * static_cast<size_t>(d->inter) * 2);
  const size_t down_bytes = align_up(rows * static_cast<size_t>(d->hidden) * 2);
  char *ws = nullptr;
  if (int32_t rc = scratch(ctx, d, pos_bytes + act_bytes + down_bytes, &ws);
      rc != TURBINE_OK) {
    return rc;
  }
  auto *pos = reinterpret_cast<int32_t *>(ws);
  char *act = ws + pos_bytes;
  char *down = act + act_bytes;
  int32_t rc = TURBINE_OK;
  switch (path) {
  case MoePath::SmallM:
    rc = turbine_hip::launch_moe_small_m(ctx, d, pos, act, down);
    break;
  case MoePath::WmmaPrefill:
    rc = turbine_hip::launch_moe_wmma_prefill(ctx, d, pos, act, down);
    break;
  default:
    rc = turbine_hip::launch_moe_wmma(ctx, d, pos, act, down);
    break;
  }
  if (rc != TURBINE_OK)
    return rc;
  return turbine_hip::launch_moe_scatter(ctx, down, pos, d->topk_weights,
                                         d->num_tokens, d->hidden, d->top_k,
                                         d->out);
}

// The quantized experts (ABI v2.16, impl "turbine_hip_moe_quant"): the own
// per-expert loop of decision "P7 quantized MoE expert provider".
//   1. gather the routed rows' activations (as run_experts), W8A8: quantize
//      them with each expert's static scale (moe_quant.hip);
//   2. per active expert one qgemm for gate_up (interleaved) or one each for
//      gate and up, written into the two halves of a [rows, 2 * inter] buffer;
//   3. one launch for the biases and the gated activation over every row;
//   4. W8A8: quantize the activation (static per expert, or dynamic per
//      token), per active expert one qgemm for down, the down bias;
//   5. the ordered scatter (as every path), then the shared expert.
// Every qgemm is a prefill call (qgemm_for), so a row's output does not depend
// on the number of rows routed to its expert.
int32_t run_shared_expert(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                          char *sg, char *su, char *sdown) {
  const int64_t tokens = d->num_tokens;
  const int64_t si = d->shared_inter;
  int32_t rc =
      run_one(ctx, {d->x, d->shared_w_gate, sg, tokens, si, d->hidden});
  if (rc != TURBINE_OK)
    return rc;
  rc = run_one(ctx, {d->x, d->shared_w_up, su, tokens, si, d->hidden});
  if (rc != TURBINE_OK)
    return rc;
  turbine_silu_mul_desc act{};
  act.gate = sg;
  act.up = su;
  act.out = sg;
  act.rows = tokens;
  act.cols = si;
  act.gate_stride_row = si;
  act.up_stride_row = si;
  act.out_stride_row = si;
  act.dtype = TURBINE_DTYPE_BF16;
  if (rc = turbine_silu_mul(ctx, &act); rc != TURBINE_OK)
    return rc;
  rc = run_one(ctx, {sg, d->shared_w_down, sdown, tokens, d->hidden, si});
  if (rc != TURBINE_OK)
    return rc;
  return turbine_hip::launch_moe_shared_combine(
      ctx, d->x, sdown, d->shared_gate_w, d->out, d->num_tokens, d->hidden);
}

// Which of the two projection paths run_quant takes: Auto = the per-expert
// loop below kGroupedMinRows routed rows and for the formats the grouped
// kernel does not win on, the grouped kernel otherwise; Loop / Grouped force
// one (impls turbine_hip_moe_quant_loop / _grouped).
enum class QuantMode { Auto, Loop, Grouped };

bool is_quant_path(turbine_hip::MoePath path) {
  using turbine_hip::MoePath;
  return path == MoePath::Quant || path == MoePath::QuantLoop ||
         path == MoePath::QuantGrouped;
}

QuantMode quant_mode(turbine_hip::MoePath path) {
  using turbine_hip::MoePath;
  return path == MoePath::QuantLoop      ? QuantMode::Loop
         : path == MoePath::QuantGrouped ? QuantMode::Grouped
                                         : QuantMode::Auto;
}

// Routed rows from which Auto runs the grouped kernel.
constexpr int64_t kGroupedMinRows = 8;

// The formats Auto runs grouped, measured over the full op on the Task 14
// shapes at 256 and 2048 tokens (perf-log "P7 grouped quantized experts"):
// FP8 block (Qwen3, 128 small experts) 1445 / 3595 us vs the loop's 7873 /
// 9585, MXFP4 (gpt-oss) 1585 / 6364 vs 3551 / 7381. FP8 tensor (Mixtral, 8
// large experts, W8A8) stays on the loop: the loop's per-expert hipBLASLt FP8
// GEMMs run the big matrices faster than the fp8_fp8 WMMA chain (4667 / 8083
// us vs the grouped kernel's 4785 / 12856), and 8 launches save little.
bool grouped_wins(int32_t weight_dtype) {
  switch (weight_dtype) {
  case TURBINE_EXPERT_FP8_BLOCK:
    return true;
  case TURBINE_EXPERT_MXFP4:
    return true;
  default: // TURBINE_EXPERT_FP8_TENSOR: the loop.
    return false;
  }
}

// What the grouped kernel needs of the shape: 64-wide column tiles over
// gate_up and down, 64-deep k steps.
bool grouped_shape(const turbine_moe_experts_desc *d) {
  if (d->hidden % 64 != 0 || d->inter % 64 != 0)
    return false;
  return d->weight_dtype != TURBINE_EXPERT_FP8_BLOCK ||
         (d->hidden % 128 == 0 && d->inter % 128 == 0);
}

int32_t run_quant(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                  QuantMode mode) {
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
  const bool interleaved = d->gate_up_interleaved != 0;
  const bool w8a8 = d->weight_dtype == TURBINE_EXPERT_FP8_TENSOR;
  const bool static_down = w8a8 && d->act_static_scale_down != nullptr;
  const int32_t wd = d->weight_dtype;
  const int64_t h = d->hidden;
  const int64_t inter = d->inter;
  const int64_t gate_n = interleaved ? 2 * inter : inter;
  const size_t r = static_cast<size_t>(rows);
  const size_t t = static_cast<size_t>(d->num_tokens);
  const size_t si = static_cast<size_t>(d->shared_inter);

  const size_t pos_bytes = align_up(static_cast<size_t>(all_rows) * 4);
  const size_t xs_bytes = align_up(r * h * 2);
  const size_t xq_bytes = w8a8 ? align_up(r * h) : 0;
  const size_t gu_bytes = align_up(r * 2 * inter * 2);
  const size_t act_bytes = align_up(r * inter * 2);
  const size_t actq_bytes = w8a8 ? align_up(r * inter) : 0;
  const size_t asc_bytes = (w8a8 && !static_down) ? align_up(r * 4) : 0;
  const size_t down_bytes = align_up(r * h * 2);
  const size_t sg_bytes = align_up(t * si * 2);
  const size_t sdown_bytes = si != 0 ? align_up(t * h * 2) : 0;
  char *ws = nullptr;
  if (int32_t rc = scratch(ctx, d,
                           pos_bytes + xs_bytes + xq_bytes + gu_bytes +
                               act_bytes + actq_bytes + asc_bytes + down_bytes +
                               2 * sg_bytes + sdown_bytes,
                           &ws);
      rc != TURBINE_OK) {
    return rc;
  }
  auto *pos = reinterpret_cast<int32_t *>(ws);
  char *xs = ws + pos_bytes;
  char *xq = xs + xs_bytes;
  char *gu = xq + xq_bytes;
  char *act = gu + gu_bytes;
  char *actq = act + act_bytes;
  auto *asc = reinterpret_cast<float *>(actq + actq_bytes);
  char *down = reinterpret_cast<char *>(asc) + asc_bytes;
  char *sg = down + down_bytes;
  char *su = sg + sg_bytes;
  char *sdown = su + sg_bytes;

  if (d->w_gate_scale == nullptr || d->w_down_scale == nullptr ||
      (!interleaved && d->w_up_scale == nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: a quantized expert needs its weight "
                "scales");
  }
  if (wd == TURBINE_EXPERT_FP8_TENSOR && d->act_static_scale == nullptr) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_moe_experts: FP8 tensor experts run W8A8 only "
                "(act_static_scale is NULL)");
  }
  if (wd != TURBINE_EXPERT_FP8_TENSOR &&
      (d->act_static_scale != nullptr || d->act_static_scale_down != nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: act_static_scale is for FP8 tensor "
                "experts only");
  }
  if (d->act_static_scale == nullptr && d->act_static_scale_down != nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: act_static_scale_down without "
                "act_static_scale");
  }
  if ((d->shared_inter > 0) != (d->shared_w_gate != nullptr) ||
      (d->shared_inter > 0) != (d->shared_w_up != nullptr) ||
      (d->shared_inter > 0) != (d->shared_w_down != nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: shared_inter and the shared weights "
                "must be given together");
  }
  int32_t rc = TURBINE_OK;
  if (rows > 0) {
    if (d->expert_offsets == nullptr) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_moe_experts: expert_offsets is NULL");
    }
    if (rc = turbine_hip::launch_moe_clear_positions(ctx, pos, all_rows);
        rc != TURBINE_OK) {
      return rc;
    }
    if (rc = turbine_hip::launch_moe_gather(ctx, d->x, d->sorted_rows + first,
                                            rows, d->hidden, d->top_k, xs, pos);
        rc != TURBINE_OK) {
      return rc;
    }
    const int32_t first32 = static_cast<int32_t>(first);
    const int32_t lo = d->expert_begin;
    const int32_t hi = d->expert_end;
    if (w8a8) {
      if (rc = turbine_hip::launch_moe_quant_static(
              ctx, xs, xq, rows, d->hidden, d->expert_offsets, first32, lo, hi,
              d->act_static_scale);
          rc != TURBINE_OK) {
        return rc;
      }
    }
    const auto *w_gate = static_cast<const char *>(d->w_gate);
    const auto *w_up = static_cast<const char *>(d->w_up);
    const auto *w_down = static_cast<const char *>(d->w_down);
    const auto *s_gate = static_cast<const char *>(d->w_gate_scale);
    const auto *s_up = static_cast<const char *>(d->w_up_scale);
    const auto *s_down = static_cast<const char *>(d->w_down_scale);
    const size_t es_in = w8a8 ? 1 : 2;
    const size_t gate_code = weight_code_bytes(wd, gate_n, h);
    const size_t gate_scale = weight_scale_bytes(wd, gate_n, h);
    const size_t up_code = weight_code_bytes(wd, inter, h);
    const size_t up_scale = weight_scale_bytes(wd, inter, h);

    bool grouped = mode == QuantMode::Grouped ||
                   (mode == QuantMode::Auto && rows >= kGroupedMinRows &&
                    grouped_wins(wd) && grouped_shape(d));
    if (grouped &&
        (!grouped_shape(d) || !aligned16(d->w_gate) || !aligned16(d->w_down) ||
         (!interleaved && !aligned16(d->w_up)))) {
      if (mode == QuantMode::Grouped) {
        return fail(ctx, TURBINE_E_UNSUPPORTED,
                    "turbine_moe_experts: the grouped quantized experts need "
                    "hidden and inter multiples of 64 (128 for FP8 block) and "
                    "16-byte aligned weights");
      }
      grouped = false;
    }
    // Grouped W8A8 tiles read 64 activation bytes per row: a is the e4m3
    // buffer of the gathered rows (quantized above).
    int32_t active = 0;
    for (int32_t e = lo; e < hi; ++e)
      active += host[e + 1] > host[e] ? 1 : 0;
    // 128-row tiles once every active expert averages ~96 rows; below that the
    // 64-row tiles spread each expert's weights over more blocks.
    const bool wide =
        rows >= 96 * static_cast<int64_t>(active < 1 ? 1 : active);
    if (grouped) {
      turbine_hip::MoeQuantGrouped g{};
      g.a = reinterpret_cast<const uint8_t *>(w8a8 ? xq : xs);
      g.lda = h;
      g.w[0] = reinterpret_cast<const uint8_t *>(w_gate);
      g.s[0] = reinterpret_cast<const uint8_t *>(s_gate);
      g.n[0] = static_cast<int32_t>(gate_n);
      g.code_stride[0] = static_cast<int64_t>(gate_code);
      g.scale_stride[0] = static_cast<int64_t>(gate_scale);
      g.n_total = static_cast<int32_t>(gate_n);
      if (!interleaved) {
        g.w[1] = reinterpret_cast<const uint8_t *>(w_up);
        g.s[1] = reinterpret_cast<const uint8_t *>(s_up);
        g.n[1] = static_cast<int32_t>(inter);
        g.code_stride[1] = static_cast<int64_t>(up_code);
        g.scale_stride[1] = static_cast<int64_t>(up_scale);
        g.n_total += static_cast<int32_t>(inter);
      }
      g.a_scale = d->act_static_scale;
      g.a_scale_vec = 0;
      g.offsets = d->expert_offsets;
      g.expert_begin = lo;
      g.num_local = hi - lo;
      g.first = first32;
      g.k = static_cast<int32_t>(h);
      g.ldc = 2 * inter;
      g.out = gu;
      if (rc = turbine_hip::launch_moe_quant_grouped(ctx, wd, g, rows, wide);
          rc != TURBINE_OK) {
        return rc;
      }
    }
    for (int32_t e = lo; e < hi && !grouped; ++e) {
      const int64_t count = host[e + 1] - host[e];
      if (count == 0)
        continue;
      const size_t off = static_cast<size_t>(host[e] - first);
      const size_t l = static_cast<size_t>(e - lo);
      turbine_qgemm_desc q = qgemm_for(d, count, gate_n, h);
      q.a = (w8a8 ? xq : xs) + off * static_cast<size_t>(h) * es_in;
      q.a_scales = w8a8 ? d->act_static_scale + l : nullptr;
      q.b = w_gate + l * gate_code;
      q.b_scales = s_gate + l * gate_scale;
      q.c = gu + off * static_cast<size_t>(2 * inter) * 2;
      q.ldc = 2 * inter;
      if (rc = turbine_qgemm(ctx, &q); rc != TURBINE_OK)
        return rc;
      if (!interleaved) {
        turbine_qgemm_desc u = qgemm_for(d, count, inter, h);
        u.a = q.a;
        u.a_scales = q.a_scales;
        u.b = w_up + l * up_code;
        u.b_scales = s_up + l * up_scale;
        u.c = static_cast<char *>(q.c) + inter * 2;
        u.ldc = 2 * inter;
        if (rc = turbine_qgemm(ctx, &u); rc != TURBINE_OK)
          return rc;
      }
    }
    if (rc = turbine_hip::launch_moe_quant_act(
            ctx, gu, act, rows, d->inter, interleaved,
            d->activation == TURBINE_EXPERT_GPTOSS_CLAMP, d->clamp_limit,
            d->b_gate, interleaved ? nullptr : d->b_up, d->expert_offsets,
            first32, lo, hi);
        rc != TURBINE_OK) {
      return rc;
    }
    if (w8a8) {
      if (static_down) {
        rc = turbine_hip::launch_moe_quant_static(
            ctx, act, actq, rows, d->inter, d->expert_offsets, first32, lo, hi,
            d->act_static_scale_down);
      } else {
        turbine_quantize_act_desc qa{};
        qa.x = act;
        qa.out = actq;
        qa.scales = asc;
        qa.rows = rows;
        qa.cols = inter;
        qa.x_stride_row = qa.out_stride_row = inter;
        qa.mode = TURBINE_ACTQ_FP8_TOKEN;
        qa.x_dtype = TURBINE_DTYPE_BF16;
        qa.out_dtype = TURBINE_DTYPE_F8E4M3;
        rc = turbine_quantize_act(ctx, &qa);
      }
      if (rc != TURBINE_OK)
        return rc;
    }
    const size_t down_code = weight_code_bytes(wd, h, inter);
    const size_t down_scale = weight_scale_bytes(wd, h, inter);
    if (grouped) {
      turbine_hip::MoeQuantGrouped g{};
      g.a = reinterpret_cast<const uint8_t *>(w8a8 ? actq : act);
      g.lda = inter;
      g.w[0] = reinterpret_cast<const uint8_t *>(w_down);
      g.s[0] = reinterpret_cast<const uint8_t *>(s_down);
      g.n[0] = static_cast<int32_t>(h);
      g.code_stride[0] = static_cast<int64_t>(down_code);
      g.scale_stride[0] = static_cast<int64_t>(down_scale);
      g.n_total = static_cast<int32_t>(h);
      g.a_scale = static_down ? d->act_static_scale_down : asc;
      g.a_scale_vec = static_down ? 0 : 1;
      g.offsets = d->expert_offsets;
      g.expert_begin = lo;
      g.num_local = hi - lo;
      g.first = first32;
      g.k = static_cast<int32_t>(inter);
      g.ldc = h;
      g.out = down;
      if (rc = turbine_hip::launch_moe_quant_grouped(ctx, wd, g, rows, wide);
          rc != TURBINE_OK) {
        return rc;
      }
    }
    for (int32_t e = lo; e < hi && !grouped; ++e) {
      const int64_t count = host[e + 1] - host[e];
      if (count == 0)
        continue;
      const size_t off = static_cast<size_t>(host[e] - first);
      const size_t l = static_cast<size_t>(e - lo);
      turbine_qgemm_desc q = qgemm_for(d, count, h, inter);
      q.a = (w8a8 ? actq : act) + off * static_cast<size_t>(inter) * es_in;
      if (w8a8) {
        q.act_quant =
            static_down ? TURBINE_ACTQ_FP8_TENSOR : TURBINE_ACTQ_FP8_TOKEN;
        q.a_scales = static_down ? d->act_static_scale_down + l : asc + off;
      }
      q.b = w_down + l * down_code;
      q.b_scales = s_down + l * down_scale;
      q.c = down + off * static_cast<size_t>(h) * 2;
      if (rc = turbine_qgemm(ctx, &q); rc != TURBINE_OK)
        return rc;
    }
    if (d->b_down != nullptr) {
      if (rc = turbine_hip::launch_moe_quant_bias(ctx, down, rows, d->hidden,
                                                  d->b_down, d->expert_offsets,
                                                  first32, lo, hi);
          rc != TURBINE_OK) {
        return rc;
      }
    }
    if (rc = turbine_hip::launch_moe_scatter(ctx, down, pos, d->topk_weights,
                                             d->num_tokens, d->hidden, d->top_k,
                                             d->out);
        rc != TURBINE_OK) {
      return rc;
    }
  }
  if (d->shared_inter > 0)
    return run_shared_expert(ctx, d, sg, su, sdown);
  return TURBINE_OK;
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

const char *moe_path_name(MoePath path) {
  switch (path) {
  case MoePath::SmallM:
    return kImplSmallM;
  case MoePath::WmmaPrefill:
    return kImplWmmaPrefill;
  case MoePath::Wmma:
    return kImplWmma;
  case MoePath::Grouped:
    return kImplGrouped;
  case MoePath::PerExpert:
    return kImplPerExpert;
  case MoePath::Quant:
    return kImplQuant;
  case MoePath::QuantLoop:
    return kImplQuantLoop;
  case MoePath::QuantGrouped:
    return kImplQuantGrouped;
  }
  return kImplPerExpert;
}

bool moe_experts_supports(const turbine_moe_experts_desc *d, MoePath path) {
  if (!experts_supported(d))
    return false;
  // The quantized experts have their own path; the BF16 paths take only the
  // pre-v2.16 form (experts_supported admits nothing else of BF16).
  if (path == MoePath::Quant || path == MoePath::QuantLoop)
    return quantized(d);
  if (path == MoePath::QuantGrouped)
    return quantized(d) && grouped_shape(d);
  if (quantized(d))
    return false;
  switch (path) {
  case MoePath::SmallM:
    return d->hidden % 8 == 0 && d->inter % 8 == 0;
  case MoePath::WmmaPrefill:
    return moe_prefill_shape(d);
  case MoePath::Wmma:
    return moe_wmma_shape(d);
  case MoePath::Grouped:
    return g_grouped_available.load();
  case MoePath::PerExpert:
    return true;
  case MoePath::Quant:
  case MoePath::QuantLoop:
  case MoePath::QuantGrouped:
    return false;
  }
  return false;
}

int32_t moe_experts_run(turbine_ctx *ctx, const turbine_moe_experts_desc *d,
                        MoePath path) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: descriptor is NULL");
  }
  if (!moe_experts_supports(d, path)) {
    return fail(
        ctx, TURBINE_E_UNSUPPORTED,
        "turbine_moe_experts: unsupported configuration " + describe(d) +
            (experts_supported(d) ? std::string(" for ") + moe_path_name(path)
                                  : std::string()));
  }
  const bool device_offsets = path == MoePath::SmallM ||
                              path == MoePath::WmmaPrefill ||
                              path == MoePath::Wmma;
  // Zero tokens read nothing: like expert_offsets below, the host copy is only
  // required when there are rows to group.
  if (!device_offsets && d->num_tokens > 0 &&
      d->host_expert_offsets == nullptr) {
    return fail(
        ctx, TURBINE_E_ARGUMENT,
        "turbine_moe_experts: host_expert_offsets is NULL for " +
            std::to_string(static_cast<int64_t>(d->num_tokens) * d->top_k) +
            " routed rows and hidden=" + std::to_string(d->hidden) +
            " inter=" + std::to_string(d->inter) + " (implementation " +
            moe_path_name(path) + " reads the group sizes on the host)");
  }
  if (d->num_tokens == 0)
    return TURBINE_OK;
  const bool w_up_unused = is_quant_path(path) && d->gate_up_interleaved != 0;
  if (d->x == nullptr || d->w_gate == nullptr ||
      (d->w_up == nullptr && !w_up_unused) || d->w_down == nullptr ||
      d->sorted_rows == nullptr || d->topk_weights == nullptr ||
      d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_moe_experts: NULL operand");
  }
  if (is_quant_path(path) && d->expert_offsets == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: expert_offsets is NULL");
  }
  if (device_offsets && d->expert_offsets == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_moe_experts: expert_offsets is NULL");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (device_offsets)
    return run_device_offsets(ctx, d, path);
  if (is_quant_path(path))
    return run_quant(ctx, d, quant_mode(path));
  return run_experts(ctx, d, path == MoePath::Grouped);
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
                    " flags=" + std::to_string(d->flags));
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
  return turbine_hip::default_name(TURBINE_OP_MOE_EXPERTS, d);
}

int32_t
turbine_moe_experts_needs_host_offsets(const turbine_moe_experts_desc *d) {
  return (turbine_hip::default_flags(TURBINE_OP_MOE_EXPERTS, d,
                                     TURBINE_IMPL_NEEDS_HOST_OFFSETS) &
          TURBINE_IMPL_NEEDS_HOST_OFFSETS) != 0
             ? 1
             : 0;
}

int32_t turbine_moe_experts(turbine_ctx *ctx,
                            const turbine_moe_experts_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_MOE_EXPERTS, d);
}

} // extern "C"
