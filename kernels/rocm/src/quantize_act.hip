// Activation quantization (ABI v2.9 turbine_quantize_act) on ROCm:
// `turbine_hip`, Turbine's own FP8 e4m3 kernels (qgemm_quantize.hpp) for the
// modes FP8_TENSOR (static), FP8_TOKEN and FP8_GROUP128 (dynamic), bit-exact
// with the CPU reference. An own kernel per the decision "P6: FP8 GEMM —
// provider evaluation (kernel reuse rule)": the one provider candidate, CK's
// fused add_rmsnorm2d_rdquant, takes the norm's inputs, which the v2.9
// quantize_act descriptor does not carry, and rounds by its own conversion.
// MXFP4_EMULATED has no implementation here yet.
#include "turbine_hip.hpp"

#include "qgemm_impls.hpp"
#include "qgemm_quantize.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

bool supported(const turbine_quantize_act_desc *d) {
  if (d == nullptr)
    return false;
  if (d->mode != TURBINE_ACTQ_FP8_TENSOR && d->mode != TURBINE_ACTQ_FP8_TOKEN &&
      d->mode != TURBINE_ACTQ_FP8_GROUP128) {
    return false;
  }
  if (d->x_dtype != TURBINE_DTYPE_BF16 && d->x_dtype != TURBINE_DTYPE_F32)
    return false;
  if (d->out_dtype != TURBINE_DTYPE_F8E4M3)
    return false;
  if (d->rows <= 0 || d->cols <= 0 || d->rows > INT32_MAX)
    return false;
  if (d->x_stride_row < d->cols || d->out_stride_row < d->cols)
    return false;
  // A static scale divides every value: it must be a positive finite number.
  if (d->mode == TURBINE_ACTQ_FP8_TENSOR &&
      !(d->static_scale > 0.0f && d->static_scale < INFINITY)) {
    return false;
  }
  return true;
}

} // namespace

extern "C" {

int32_t turbine_quantize_act_supported(const turbine_quantize_act_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_QUANTIZE_ACT, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_quantize_act_impl(const turbine_quantize_act_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_QUANTIZE_ACT, d);
}

int32_t turbine_quantize_act(turbine_ctx *ctx,
                             const turbine_quantize_act_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_QUANTIZE_ACT, d);
}

} // extern "C"

namespace turbine_hip {

bool quantize_act_fp8_supports(const turbine_quantize_act_desc *d) {
  return supported(d);
}

int32_t quantize_act_fp8_run(turbine_ctx *ctx,
                             const turbine_quantize_act_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_quantize_act: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_quantize_act (turbine_hip): unsupported configuration "
                "mode=" +
                    std::to_string(d->mode) +
                    " rows=" + std::to_string(d->rows) +
                    " cols=" + std::to_string(d->cols) +
                    " dtypes=" + std::to_string(d->x_dtype) + "," +
                    std::to_string(d->out_dtype));
  }
  if (d->x == nullptr || d->out == nullptr || d->scales == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_quantize_act: NULL operand");
  }
  const int32_t rc = enter(ctx);
  if (rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, launch_quantize_fp8(d, ctx->stream),
                   "turbine_quantize_act launch");
}

} // namespace turbine_hip
