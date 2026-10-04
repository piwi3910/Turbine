// Block-scaled FP8 GEMM on RDNA4: turbine_hip_fp8_block, the ABI v2.9 scheme
// FP8_BLOCK (128 x 128 blocks) with BF16 activations (W8A16, act_quant NONE),
// an own kernel (decision "P6: block-scaled FP8 GEMM — provider evaluation
// (kernel reuse rule)": no provider has a correct block-scaled FP8 GEMM on
// gfx1201; the user chose an own kernel over serving the weights decoded to
// BF16). The weights stay e4m3 on the device.
//
// Decode steps with at most kFusedMaxRows rows run the fused WMMA kernel of
// qgemm_fp8_block_kernels.hpp (codes decoded in registers). Prefill steps and
// larger decode batches dequantize the weight to BF16 (bf16(e4m3(q) * s), as
// the golden reference's dequantized checkpoint holds it) into a per-context
// staging buffer and run the hipBLASLt BF16 GEMM on it (turbine_gemm, trans_b,
// the layer's own (n, k): the tuned GEMM table's shape), with
// TURBINE_OPTION_GEMM_PREFILL raised in prefill steps so the table's
// invariant class (split-K off, one solution for every m) runs: a row's
// result does not depend on how many rows share the call (bit-exact prefix
// reuse). The fused kernel is row-invariant by construction.
#include <algorithm>
#include <memory>
#include <string>

#include "qgemm_fp8_block.hpp"
#include "qgemm_fp8_block_kernels.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

namespace {

constexpr int64_t kBlock = fp8_block::kBlock;
// Rows up to which a decode step runs the fused kernel (the harness's
// --block 1 timings: faster than dequantize + BF16 GEMM up to here).
constexpr int64_t kFusedMaxRows = 64;
// Largest BF16 weight staged whole (a Llama-3.1-8B gate_up is 224 MiB); a
// larger one is staged and multiplied in column chunks of whole blocks.
constexpr int64_t kDequantMaxBytes = 256 << 20;
// The fused kernel's launch shape: one 16-column tile per wave, the waves of a
// block splitting k into whole 128-deep blocks (about eight per wave).
constexpr int kFragsN = 1;

int waves_for(int64_t blocks) {
  return blocks >= 64 ? 8 : blocks >= 32 ? 4 : 2;
}

bool supported(const turbine_qgemm_desc *d) {
  if (d == nullptr || d->scheme != TURBINE_QSCHEME_FP8_BLOCK ||
      d->act_quant != TURBINE_ACTQ_NONE || d->a_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->block_n != kBlock || d->block_k != kBlock)
    return false;
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32)
    return false;
  if (d->m <= 0 || d->n <= 0 || d->k <= 0 || d->k % kBlock != 0)
    return false;
  return d->lda >= d->k && d->lda % 8 == 0 && d->ldc >= d->n;
}

std::string describe(const turbine_qgemm_desc *d) {
  return "m=" + std::to_string(d->m) + " n=" + std::to_string(d->n) +
         " k=" + std::to_string(d->k) + " lda=" + std::to_string(d->lda) +
         " ldc=" + std::to_string(d->ldc) +
         " scheme=" + std::to_string(d->scheme) +
         " act_quant=" + std::to_string(d->act_quant) +
         " a_dtype=" + std::to_string(d->a_dtype) +
         " c_dtype=" + std::to_string(d->c_dtype) +
         " block=" + std::to_string(d->block_n) + "x" +
         std::to_string(d->block_k);
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

hipError_t launch_fused(const turbine_qgemm_desc *d, hipStream_t stream) {
  const auto *b = static_cast<const uint8_t *>(d->b);
  const auto *s = static_cast<const float *>(d->b_scales);
  const bool f32 = d->c_dtype == TURBINE_DTYPE_F32;
  switch (waves_for(d->k / kBlock)) {
  case 8:
    return fp8_block::launch_wmma<kFragsN, 8>(d->a, b, s, d->c, d->m, d->n,
                                              d->k, d->lda, d->ldc, f32,
                                              d->alpha, stream);
  case 4:
    return fp8_block::launch_wmma<kFragsN, 4>(d->a, b, s, d->c, d->m, d->n,
                                              d->k, d->lda, d->ldc, f32,
                                              d->alpha, stream);
  default:
    return fp8_block::launch_wmma<kFragsN, 2>(d->a, b, s, d->c, d->m, d->n,
                                              d->k, d->lda, d->ldc, f32,
                                              d->alpha, stream);
  }
}

} // namespace

// The dequant path's BF16 staging buffer (turbine_ctx::qgemm_fp8_block), grown
// outside graph capture only and freed with the context.
struct Fp8BlockScratch {
  void *buf = nullptr;
  size_t bytes = 0;

  Fp8BlockScratch() = default;
  Fp8BlockScratch(const Fp8BlockScratch &) = delete;
  Fp8BlockScratch &operator=(const Fp8BlockScratch &) = delete;
  ~Fp8BlockScratch() {
    if (buf != nullptr)
      (void)hipFree(buf);
  }
};

namespace {

int32_t run_dequant(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  const int64_t chunk_rows = std::max<int64_t>(
      kBlock, kDequantMaxBytes / (2 * d->k) / kBlock * kBlock);
  const int64_t rows = std::min(chunk_rows, d->n);
  const size_t bytes = static_cast<size_t>(rows * d->k * 2);
  if (!ctx->qgemm_fp8_block)
    ctx->qgemm_fp8_block = std::make_shared<Fp8BlockScratch>();
  Fp8BlockScratch &scratch = *ctx->qgemm_fp8_block;
  if (scratch.bytes < bytes) {
    if (ctx->capturing) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_qgemm (turbine_hip_fp8_block): staging buffer of " +
                      std::to_string(scratch.bytes) + " bytes cannot grow to " +
                      std::to_string(bytes) + " while a graph is captured");
    }
    void *grown = nullptr;
    const int32_t rc = check_hip(ctx, hipMalloc(&grown, bytes),
                                 "hipMalloc (qgemm fp8 block staging)");
    if (rc != TURBINE_OK)
      return rc;
    if (scratch.buf != nullptr) {
      // Queued work may still read the old buffer.
      (void)hipStreamSynchronize(ctx->stream);
      (void)hipFree(scratch.buf);
    }
    scratch.buf = grown;
    scratch.bytes = bytes;
  }
  const auto *b = static_cast<const uint8_t *>(d->b);
  const auto *s = static_cast<const float *>(d->b_scales);
  const int64_t c_elem = d->c_dtype == TURBINE_DTYPE_F32 ? 4 : 2;
  // The step kind the caller set for plain GEMMs, restored after ours.
  const bool step_prefill = ctx->gemm_prefill;
  ctx->gemm_prefill = d->prefill != 0;
  int32_t result = TURBINE_OK;
  for (int64_t row0 = 0; row0 < d->n && result == TURBINE_OK; row0 += rows) {
    const int64_t count = std::min(rows, d->n - row0);
    result = check_hip(ctx,
                       fp8_block::launch_dequant(b, s, row0, count, d->k,
                                                 scratch.buf, ctx->stream),
                       "qgemm fp8 block dequant kernel launch");
    if (result != TURBINE_OK)
      break;
    turbine_gemm_desc gd{};
    gd.a = d->a;
    gd.b = scratch.buf;
    gd.c = static_cast<char *>(d->c) + row0 * c_elem;
    gd.m = d->m;
    gd.n = count;
    gd.k = d->k;
    gd.lda = d->lda;
    gd.ldb = d->k;
    gd.ldc = d->ldc;
    gd.trans_b = 1;
    gd.a_dtype = TURBINE_DTYPE_BF16;
    gd.b_dtype = TURBINE_DTYPE_BF16;
    gd.c_dtype = d->c_dtype;
    gd.alpha = d->alpha;
    gd.beta = 0.0f;
    result = turbine_gemm(ctx, &gd);
  }
  ctx->gemm_prefill = step_prefill;
  return result;
}

} // namespace

bool qgemm_fp8_block_supports(const turbine_qgemm_desc *d) {
  return supported(d);
}

int32_t qgemm_fp8_block_run(turbine_ctx *ctx, const turbine_qgemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: descriptor is NULL");
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_qgemm (turbine_hip_fp8_block): unsupported "
                "configuration " +
                    describe(d));
  }
  if (d->a == nullptr || d->b == nullptr || d->c == nullptr ||
      d->b_scales == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_qgemm: NULL operand or scale pointer");
  }
  if (!aligned16(d->a) || !aligned16(d->b)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_qgemm (turbine_hip_fp8_block): a and b must be "
                "16-byte aligned");
  }
  const int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  if (d->prefill != 0 || d->m > kFusedMaxRows)
    return run_dequant(ctx, d);
  return check_hip(ctx, launch_fused(d, ctx->stream),
                   "qgemm fp8 block wmma kernel launch");
}

} // namespace turbine_hip
