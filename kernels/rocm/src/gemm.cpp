// GEMM through hipBLASLt.
//
// The problem (layouts, transposes, F32 accumulation) is gemm_problem.hpp's.
// The algorithm per shape: the tuned table's pinned solution (run with split-K
// off through the hipBLASLt ext API, gemm_table.hpp) when the context
// has the table on (TURBINE_OPTION_GEMM_AUTOTUNE, default) and the table has a
// row for the shape on this card (gemm_table.hpp); otherwise, or when the
// pinned solution is not available in this hipBLASLt or rejects the problem
// (logged once per row with its reason code), hipBLASLt's first heuristic
// answer. A shape with rows of both kinds runs its invariant rows in steps
// that prefill prompt tokens (TURBINE_OPTION_GEMM_PREFILL) and its speed rows
// in decode steps. The choice is cached per exact shape and step kind for the
// context's lifetime.
#include <cstdio>

#include "gemm_problem.hpp"
#include "gemm_table.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_blaslt;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "hipblaslt";

bool supported(const turbine_gemm_desc *d) {
  if (d == nullptr)
    return false;
  if (d->a_dtype != TURBINE_DTYPE_BF16 || d->b_dtype != TURBINE_DTYPE_BF16) {
    return false;
  }
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32) {
    return false;
  }
  if (d->trans_b != 0 && d->trans_b != 1)
    return false;
  if (d->m <= 0 || d->n <= 0 || d->k <= 0)
    return false;
  if (d->lda < d->k || d->ldc < d->n)
    return false;
  return d->ldb >= (d->trans_b == 1 ? d->k : d->n);
}

} // namespace

extern "C" {

int32_t turbine_gemm_supported(const turbine_gemm_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_gemm_impl(const turbine_gemm_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_gemm(turbine_ctx *ctx, const turbine_gemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_gemm: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(
        ctx, TURBINE_E_UNSUPPORTED,
        "turbine_gemm: unsupported configuration m=" + std::to_string(d->m) +
            " n=" + std::to_string(d->n) + " k=" + std::to_string(d->k) +
            " trans_b=" + std::to_string(d->trans_b) +
            " dtypes=" + std::to_string(d->a_dtype) + "," +
            std::to_string(d->b_dtype) + "," + std::to_string(d->c_dtype));
  }
  if (d->a == nullptr || d->b == nullptr || d->c == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_gemm: NULL operand");
  }
  int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;

  const turbine_hip::GemmShape shape{d->m,   d->n,   d->k,       d->lda,
                                     d->ldb, d->ldc, d->trans_b, d->c_dtype};
  turbine_hip::GemmProblem problem;
  const hipblasStatus_t made = problem.make(shape);
  rc = check_blaslt(
      ctx, made, problem.failed != nullptr ? problem.failed : "turbine_gemm");
  if (rc != TURBINE_OK)
    return rc;

  const turbine_hip::GemmKey key{d->m,       d->n,       d->k,
                                 d->lda,     d->ldb,     d->ldc,
                                 d->trans_b, d->c_dtype, ctx->gemm_prefill};
  auto found = ctx->gemm_algos.find(key);
  if (found == ctx->gemm_algos.end()) {
    turbine_hip::GemmChoice choice{};
    bool chosen = false;
    const std::string &arch =
        ctx->profile.arch.empty() ? ctx->arch : ctx->profile.arch;
    const turbine_hip::TunedGemm *row =
        ctx->gemm_table
            ? turbine_hip::tuned_gemm(arch, shape, ctx->gemm_prefill)
            : nullptr;
    if (row != nullptr && row->solution_index >= 0) {
      turbine_hip::TunedGemmMiss miss{};
      chosen = turbine_hip::resolve_tuned_gemm(ctx, *row, problem, d,
                                               &choice.algo, &miss);
      choice.tuned = chosen;
      choice.split_k_off = chosen && row->invariant;
      if (!chosen && ctx->gemm_table_logged.insert(row).second) {
        std::fprintf(stderr,
                     "turbine_hip: event=gemm_table_fallback reason=%s "
                     "arch=%s m=%lld n=%lld k=%lld c_dtype=%d solution=%s: "
                     "running hipBLASLt's first heuristic answer\n",
                     turbine_hip::tuned_gemm_miss_code(miss), arch.c_str(),
                     static_cast<long long>(d->m), static_cast<long long>(d->n),
                     static_cast<long long>(d->k), d->c_dtype,
                     row->solution_name);
      }
    }
    if (!chosen) {
      turbine_hip::Preference pref;
      rc = check_blaslt(ctx, hipblasLtMatmulPreferenceCreate(&pref.handle),
                        "hipblasLtMatmulPreferenceCreate");
      if (rc != TURBINE_OK)
        return rc;
      const uint64_t max_ws = turbine_hip::kGemmWorkspaceBytes;
      rc = check_blaslt(ctx,
                        hipblasLtMatmulPreferenceSetAttribute(
                            pref.handle,
                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                            sizeof(max_ws)),
                        "hipblasLtMatmulPreferenceSetAttribute");
      if (rc != TURBINE_OK)
        return rc;
      hipblasLtMatmulHeuristicResult_t result{};
      int returned = 0;
      rc = check_blaslt(ctx,
                        hipblasLtMatmulAlgoGetHeuristic(
                            ctx->blaslt, problem.desc.handle,
                            problem.weight.handle, problem.act.handle,
                            problem.out.handle, problem.out.handle, pref.handle,
                            1, &result, &returned),
                        "hipblasLtMatmulAlgoGetHeuristic");
      if (rc != TURBINE_OK)
        return rc;
      if (returned < 1 || result.state != HIPBLAS_STATUS_SUCCESS) {
        return fail(ctx, TURBINE_E_UNSUPPORTED,
                    "hipblasLtMatmulAlgoGetHeuristic: no algorithm for m=" +
                        std::to_string(d->m) + " n=" + std::to_string(d->n) +
                        " k=" + std::to_string(d->k));
      }
      choice = turbine_hip::GemmChoice{result.algo, false, false};
    }
    if (ctx->gemm_algos.size() >= turbine_hip::kGemmAlgoCacheEntries) {
      ctx->gemm_algos.clear();
    }
    found = ctx->gemm_algos.emplace(key, choice).first;
  }

  if (found->second.split_k_off) {
    return check_blaslt(
        ctx,
        turbine_hip::pinned_gemm(ctx, problem, d, &found->second.algo, true),
        "hipblaslt_ext::Gemm (pinned, split-K off)");
  }
  return check_blaslt(
      ctx,
      hipblasLtMatmul(ctx->blaslt, problem.desc.handle, &d->alpha, d->b,
                      problem.weight.handle, d->a, problem.act.handle, &d->beta,
                      d->c, problem.out.handle, d->c, problem.out.handle,
                      &found->second.algo, ctx->workspace,
                      turbine_hip::kGemmWorkspaceBytes, ctx->stream),
      "hipblasLtMatmul");
}

} // extern "C"
