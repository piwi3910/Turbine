// The Composable Kernel candidates of the quantized GEMM provider evaluation
// (tools/qgemm_eval.cpp; Phase 6a Tasks 12 and 15): ck_tile gemm_quant from the
// pinned CK checkout, instantiated for the build's GPU with the tile configs of
// CK's own example (example/ck_tile/38_block_scale_gemm), e4m3 A (activations,
// row-major m x k) and B (weights, column-major k x n = row-major n x k), F32
// accumulation, BF16 C:
//
//   RowColQuant    per-row A scales [m] x per-column B scales [n]
//   TensorQuant    scalar A scale x scalar B scale
//   ABQuantGrouped A scales per row and 128 columns [m, k/128] x B scales per
//                  128 x 128 block ([n/128, k/128] row-major)
//
// Each in a decode tile (GemmConfigQuantDecode: 16 x 64 x 256) and a prefill
// tile (GemmConfigQuantPrefill / GemmConfigABQuantPrefill: 128 x 128 x 128).
// Compiled only with -DTURBINE_QGEMM_EVAL_CK and CK's include directories.
#include "qgemm_eval_ck.hpp"

#include "run_gemm_quant_example.inc"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using Fp8 = ck_tile::fp8_t;
using TypeConfig = GemmQuantTypeConfig<Fp8, Fp8, ck_tile::bf16_t, float>;
using PerElement = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 1>>;
using Group128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using Block128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

template <typename Config, typename AQ, typename BQ, ck_tile::QuantType Mode>
int run(const ck_tile::QuantGemmHostArgs &args, hipStream_t stream) {
  try {
    (void)gemm_calc_quant<Config, TypeConfig, Row, Row, Col, Col, Row, AQ, BQ,
                          Mode, ck_tile::element_wise::PassThrough>(
        args, ck_tile::stream_config{stream, false, 0, 0, 1, true, false, 1});
  } catch (const std::exception &) {
    return 1; // IsSupportedArgument refused the problem
  }
  return hipGetLastError() == hipSuccess ? 0 : 2;
}

} // namespace

int ck_qgemm(CkQuantMode mode, bool prefill_tile, const void *a, const void *b,
             void *c, const float *a_scales, const float *b_scales, int64_t m,
             int64_t n, int64_t k, hipStream_t stream) {
  ck_tile::QuantGemmHostArgs args;
  args.a_ptr = a;
  args.b_ptr = b;
  args.c_ptr = c;
  args.aq_ptr = a_scales;
  args.bq_ptr = b_scales;
  args.k_batch = 1;
  args.M = static_cast<ck_tile::index_t>(m);
  args.N = static_cast<ck_tile::index_t>(n);
  args.K = static_cast<ck_tile::index_t>(k);
  args.stride_A = static_cast<ck_tile::index_t>(k);
  args.stride_B = static_cast<ck_tile::index_t>(k);
  args.stride_C = static_cast<ck_tile::index_t>(n);
  switch (mode) {
  case CkQuantMode::RowCol:
    args.QK_A = 1;
    args.QK_B = 1;
    args.stride_AQ = 1;
    args.stride_BQ = 1;
    return prefill_tile
               ? run<GemmConfigQuantPrefill<Fp8>, PerElement, PerElement,
                     ck_tile::QuantType::RowColQuant>(args, stream)
               : run<GemmConfigQuantDecode<Fp8>, PerElement, PerElement,
                     ck_tile::QuantType::RowColQuant>(args, stream);
  case CkQuantMode::Tensor:
    args.QK_A = 1;
    args.QK_B = 1;
    args.stride_AQ = 1;
    args.stride_BQ = 1;
    return prefill_tile
               ? run<GemmConfigQuantPrefill<Fp8>, PerElement, PerElement,
                     ck_tile::QuantType::TensorQuant>(args, stream)
               : run<GemmConfigQuantDecode<Fp8>, PerElement, PerElement,
                     ck_tile::QuantType::TensorQuant>(args, stream);
  case CkQuantMode::Block128: {
    const ck_tile::index_t kq = static_cast<ck_tile::index_t>((k + 127) / 128);
    args.QK_A = kq;
    args.QK_B = kq;
    args.stride_AQ = kq;
    args.stride_BQ = kq;
    return prefill_tile
               ? run<GemmConfigABQuantPrefill<Fp8, false>, Group128, Block128,
                     ck_tile::QuantType::ABQuantGrouped>(args, stream)
               : run<GemmConfigQuantDecode<Fp8>, Group128, Block128,
                     ck_tile::QuantType::ABQuantGrouped>(args, stream);
  }
  }
  return 1;
}
