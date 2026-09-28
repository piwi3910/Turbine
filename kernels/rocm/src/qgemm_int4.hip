// INT4 group-quantized GEMM on RDNA4 (qgemm_int4.hpp): the ABI v2.9 schemes
// INT4_GROUP_ZP (AWQ, asymmetric GPTQ) and INT4_GROUP_SYM (symmetric GPTQ,
// compressed-tensors pack-quantized) with BF16 activations (W4A16), chosen by
// the decision "P6: INT4 group GEMM — provider evaluation (kernel reuse
// rule)".
//
// turbine_hip_int4_wmma: the fused kernel of qgemm_int4_kernels.hpp (codes
// streamed into the WMMA registers; decode and small batches). Groups of 64,
// 128 or 256.
//
// turbine_hip_int4_dequant: weight rows are dequantized in chunks of at most
// kDequantChunkBytes of BF16 into a per-context staging buffer, each chunk
// followed by the hipBLASLt BF16 GEMM of its output columns (turbine_gemm,
// trans_b); for prefill-sized m, where the BF16 GEMM's efficiency outweighs
// the extra pass over the weight. The weight is rounded to BF16 as the golden
// reference's dequantized checkpoint is (bf16((q - z) * s)). Groups of a
// multiple of 32.
#include <algorithm>
#include <memory>
#include <string>

#include "qgemm_int4.hpp"
#include "qgemm_int4_kernels.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

namespace {

// The fused kernel's launch shape (tools/qgemm_int4_eval --sweep on the
// R9700, Llama-3.2-3B shapes): one 16-column tile per wave; the waves of a
// block split k so that each keeps about eight groups (k 3072 at group 128: 2
// waves; k 8192: 8). The split depends on k and the group only, so a row's
// result does not depend on m.
constexpr int kFragsN = 1;

int waves_for(int64_t groups) {
  return groups >= 64 ? 8 : groups >= 32 ? 4 : 2;
}

// Largest chunk of dequantized weight rows the dequant path stages at once.
constexpr int64_t kDequantChunkBytes = 32 << 20;

bool int4_scheme(int32_t scheme) {
  return scheme == TURBINE_QSCHEME_INT4_GROUP_ZP ||
         scheme == TURBINE_QSCHEME_INT4_GROUP_SYM;
}

// What both paths need of a descriptor (pointers aside).
bool common(const turbine_qgemm_desc *d) {
  if (d == nullptr || !int4_scheme(d->scheme) ||
      d->act_quant != TURBINE_ACTQ_NONE || d->a_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32)
    return false;
  if (d->m <= 0 || d->n <= 0 || d->k <= 0 || d->lda < d->k || d->ldc < d->n)
    return false;
  return d->group_size > 0 && d->k % d->group_size == 0;
}

Int4Gemm gemm_of(const turbine_qgemm_desc *d) {
  return Int4Gemm{d->a,
                  static_cast<const uint8_t *>(d->b),
                  static_cast<const float *>(d->b_scales),
                  d->scheme == TURBINE_QSCHEME_INT4_GROUP_ZP ? d->b_zeros
                                                             : nullptr,
                  d->c,
                  d->m,
                  d->n,
                  d->k,
                  d->lda,
                  d->ldc,
                  d->group_size,
                  d->c_dtype == TURBINE_DTYPE_F32,
                  d->alpha};
}

std::string describe(const turbine_qgemm_desc *d) {
  return "m=" + std::to_string(d->m) + " n=" + std::to_string(d->n) +
         " k=" + std::to_string(d->k) + " lda=" + std::to_string(d->lda) +
         " ldc=" + std::to_string(d->ldc) +
         " scheme=" + std::to_string(d->scheme) +
         " act_quant=" + std::to_string(d->act_quant) +
         " a_dtype=" + std::to_string(d->a_dtype) +
         " c_dtype=" + std::to_string(d->c_dtype) +
         " group=" + std::to_string(d->group_size);
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace

// The dequant path's BF16 staging buffer (turbine_ctx::qgemm_int4), grown
// outside graph capture only and freed with the context.
struct Int4Scratch {
  void *buf = nullptr;
  size_t bytes = 0;

  Int4Scratch() = default;
  Int4Scratch(const Int4Scratch &) = delete;
  Int4Scratch &operator=(const Int4Scratch &) = delete;
  ~Int4Scratch() {
    if (buf != nullptr)
      (void)hipFree(buf);
  }
};

bool int4_wmma_shape(const Int4Gemm &g) {
  return g.m > 0 && g.n > 0 && g.k > 0 &&
         (g.group == 64 || g.group == 128 || g.group == 256) &&
         g.k % g.group == 0 && g.lda % 8 == 0 && g.lda >= g.k && g.ldc >= g.n;
}

bool int4_dequant_shape(const Int4Gemm &g) {
  return g.m > 0 && g.n > 0 && g.k > 0 && g.group > 0 && g.group % 32 == 0 &&
         g.k % g.group == 0 && g.lda >= g.k && g.ldc >= g.n;
}

hipError_t launch_int4_wmma(const Int4Gemm &g, hipStream_t stream) {
  switch (waves_for(g.k / g.group)) {
  case 8:
    return int4::launch_wmma<kFragsN, 8>(g.a, g.b, g.scales, g.zeros, g.c, g.m,
                                         g.n, g.k, g.lda, g.ldc, g.group,
                                         g.c_f32, g.alpha, stream);
  case 4:
    return int4::launch_wmma<kFragsN, 4>(g.a, g.b, g.scales, g.zeros, g.c, g.m,
                                         g.n, g.k, g.lda, g.ldc, g.group,
                                         g.c_f32, g.alpha, stream);
  default:
    return int4::launch_wmma<kFragsN, 2>(g.a, g.b, g.scales, g.zeros, g.c, g.m,
                                         g.n, g.k, g.lda, g.ldc, g.group,
                                         g.c_f32, g.alpha, stream);
  }
}

hipError_t launch_int4_dequant(const Int4Gemm &g, int64_t row0, int64_t rows,
                               void *out, hipStream_t stream) {
  constexpr int kThreads = 256;
  const int64_t chunks = rows * (g.k / 32);
  hipLaunchKernelGGL(
      int4::dequant_kernel,
      dim3(static_cast<uint32_t>((chunks + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, stream, g.b, g.scales, g.zeros, row0, rows, g.k,
      g.group, static_cast<Bf16 *>(out));
  return hipGetLastError();
}

const char *qgemm_int4_name(Int4Path path) {
  return path == Int4Path::Wmma ? "turbine_hip_int4_wmma"
                                : "turbine_hip_int4_dequant";
}

bool qgemm_int4_supports(const turbine_qgemm_desc *d, Int4Path path) {
  if (!common(d))
    return false;
  const Int4Gemm g = gemm_of(d);
  return path == Int4Path::Wmma ? int4_wmma_shape(g) : int4_dequant_shape(g);
}

namespace {

int32_t run_dequant(turbine_ctx *ctx, const Int4Gemm &g) {
  const int64_t chunk_rows =
      std::max<int64_t>(16, kDequantChunkBytes / (2 * g.k) / 16 * 16);
  const int64_t rows = std::min(chunk_rows, g.n);
  const size_t bytes = static_cast<size_t>(rows * g.k * 2);
  if (!ctx->qgemm_int4)
    ctx->qgemm_int4 = std::make_shared<Int4Scratch>();
  Int4Scratch &scratch = *ctx->qgemm_int4;
  if (scratch.bytes < bytes) {
    if (ctx->capturing) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_qgemm (turbine_hip_int4_dequant): staging buffer "
                  "of " +
                      std::to_string(scratch.bytes) + " bytes cannot grow to " +
                      std::to_string(bytes) + " while a graph is captured");
    }
    void *grown = nullptr;
    const int32_t rc = check_hip(ctx, hipMalloc(&grown, bytes),
                                 "hipMalloc (qgemm int4 staging)");
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
  const int64_t c_elem = g.c_f32 ? 4 : 2;
  for (int64_t row0 = 0; row0 < g.n; row0 += rows) {
    const int64_t count = std::min(rows, g.n - row0);
    int32_t rc = check_hip(
        ctx, launch_int4_dequant(g, row0, count, scratch.buf, ctx->stream),
        "qgemm int4 dequant kernel launch");
    if (rc != TURBINE_OK)
      return rc;
    turbine_gemm_desc gd{};
    gd.a = g.a;
    gd.b = scratch.buf;
    gd.c = static_cast<char *>(g.c) + row0 * c_elem;
    gd.m = g.m;
    gd.n = count;
    gd.k = g.k;
    gd.lda = g.lda;
    gd.ldb = g.k;
    gd.ldc = g.ldc;
    gd.trans_b = 1;
    gd.a_dtype = TURBINE_DTYPE_BF16;
    gd.b_dtype = TURBINE_DTYPE_BF16;
    gd.c_dtype = g.c_f32 ? TURBINE_DTYPE_F32 : TURBINE_DTYPE_BF16;
    gd.alpha = g.alpha;
    gd.beta = 0.0f;
    rc = turbine_gemm(ctx, &gd);
    if (rc != TURBINE_OK)
      return rc;
  }
  return TURBINE_OK;
}

} // namespace

int32_t qgemm_int4_run(turbine_ctx *ctx, const turbine_qgemm_desc *d,
                       Int4Path path) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_qgemm: descriptor is NULL");
  if (!qgemm_int4_supports(d, path)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string("turbine_qgemm (") + qgemm_int4_name(path) +
                    "): unsupported configuration " + describe(d));
  }
  if (d->a == nullptr || d->b == nullptr || d->c == nullptr ||
      d->b_scales == nullptr ||
      (d->scheme == TURBINE_QSCHEME_INT4_GROUP_ZP && d->b_zeros == nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_qgemm: NULL operand, scale or zero-point pointer");
  }
  if (!aligned16(d->a) || !aligned16(d->b)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string("turbine_qgemm (") + qgemm_int4_name(path) +
                    "): a and b must be 16-byte aligned");
  }
  int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  const Int4Gemm g = gemm_of(d);
  if (path == Int4Path::Dequant)
    return run_dequant(ctx, g);
  return check_hip(ctx, launch_int4_wmma(g, ctx->stream),
                   "qgemm int4 wmma kernel launch");
}

} // namespace turbine_hip
