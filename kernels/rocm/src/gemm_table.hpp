// The tuned GEMM algorithm table: per card architecture, the hipBLASLt
// solution measured fastest for each GEMM shape Turbine runs. It is card data,
// not code: kernels/rocm/tuning/<arch>/gemm.tsv, written by the GEMM tuner
// (tools/gemm_tune.cpp) and compiled into the library by
// cmake/gemm_table.py for the architectures of the build.
//
// A row pins one solution by its stable identity, the hipBLASLt solution name
// (the kernel's full parameter string), plus the solution index it had when it
// was tuned (a fast path: the index is checked against the name before use).
// It serves m in (previous row's m_max, m_max] of its (n, k, trans_b, c_dtype);
// an m above every row of the shape takes the largest row. A row with
// solution_index -1 (name "heuristic") pins nothing: no solution beat
// hipBLASLt's own per-call answer over that bucket, so the library asks the
// heuristic, as for a shape without rows.
#pragma once

#include <cstdint>
#include <string>

#include "gemm_problem.hpp"

struct turbine_ctx;

namespace turbine_hip {

struct TunedGemm {
  const char *arch;
  int64_t n, k;
  int32_t trans_b, c_dtype;
  int64_t m_max;
  int32_t solution_index;
  const char *solution_name;
};

// The table row serving s on arch, or nullptr when the table has none.
const TunedGemm *tuned_gemm(const std::string &arch, const GemmShape &s);

// Why a table row could not be used; the log's reason code.
enum class TunedGemmMiss {
  // The solution name is not in this hipBLASLt's solution list.
  Unavailable,
  // The solution rejects this problem (e.g. a leading dimension it cannot
  // take) or needs more workspace than the context has.
  Unsupported,
};

const char *tuned_gemm_miss_code(TunedGemmMiss miss);

// Resolves row t for problem p (alpha, beta the call's scalars) on ctx's
// hipBLASLt handle into *algo. On failure returns false and sets *miss.
bool resolve_tuned_gemm(turbine_ctx *ctx, const TunedGemm &t, GemmProblem &p,
                        const float *alpha, const float *beta,
                        hipblasLtMatmulAlgo_t *algo, TunedGemmMiss *miss);

} // namespace turbine_hip
