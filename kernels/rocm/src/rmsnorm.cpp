// RMSNorm: ck_tile rmsnorm2d_fwd for the hand-instantiated BF16 bucket, the
// Turbine HIP kernel (elementwise.hip) for every other dimension and dtype.
//
// The CK instance is the one CK's rmsnorm2d generator emits for BF16 -> BF16,
// 2048 < n <= 3072, n % 8 == 0, no fused add or quant, with the T5-like
// ("model sensitive") pipeline: x * inv_rms is rounded to BF16 before the
// gamma multiply, which is Hugging Face LlamaRMSNorm's rounding and the
// cpu-reference numerics.
#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/rmsnorm2d.hpp>

#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImplCk = "ck_tile_rmsnorm2d";
constexpr const char *kImplTurbine = "turbine_hip";

// The instantiated bucket: 2048 < n <= 3072, n % 8 == 0.
constexpr int64_t kBucketMinExclusive = 2048;
constexpr int64_t kBucketMax = 3072;
constexpr int64_t kVector = 8;

// Tile: Repeat_M 1, Repeat_N 2, ThreadPerBlock_M 1, ThreadPerBlock_N 256,
// Vector_N 8 (Block_N = 4096 >= n, padded).
using BlockTile = ck_tile::sequence<1, 2 * 256 * kVector>;
using ThreadPerBlock = ck_tile::sequence<1, 256>;
using Vector = ck_tile::sequence<1, kVector>;
using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;

using PipelineTraits = ck_tile::Rmsnorm2dFwdTraits<
    /*kPadN=*/true, /*kSaveInvRms=*/false, /*kSaveUnquant=*/false,
    /*kTwoPass=*/false, ck_tile::Rmsnorm2dFusedAddEnum::NO_ADD,
    ck_tile::Rmsnorm2dFusedQuantEnum::NO_SWEEP,
    ck_tile::Rmsnorm2dSensitiveEnum::T5_MODEL_LIKE>;

using Problem = ck_tile::Rmsnorm2dFwdPipelineProblem<
    ck_tile::bf16_t, // X
    ck_tile::bf16_t, // gamma
    float,           // compute
    ck_tile::bf16_t, // Y
    ck_tile::bf16_t, // inv rms (not stored)
    ck_tile::bf16_t, // unquantised Y (not stored)
    float,           // smooth scale (unused)
    float,           // Y scale (unused)
    Shape, PipelineTraits>;

using Pipeline = ck_tile::Rmsnorm2dFwdPipelineModelSensitiveT5Pass<Problem>;
using Epilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
    float, ck_tile::bf16_t, false, true, false>>;
using Kernel = ck_tile::Rmsnorm2dFwd<Pipeline, Epilogue>;

bool ck_bucket(const turbine_rmsnorm_desc *d) {
  return d->dtype == TURBINE_DTYPE_BF16 && d->dim > kBucketMinExclusive &&
         d->dim <= kBucketMax && d->dim % kVector == 0 &&
         d->x_stride_row % kVector == 0 && d->out_stride_row % kVector == 0 &&
         d->rows <= INT32_MAX && d->x_stride_row <= INT32_MAX &&
         d->out_stride_row <= INT32_MAX;
}

bool valid(const turbine_rmsnorm_desc *d) {
  return d != nullptr && d->rows >= 1 && d->dim >= 1 &&
         d->x_stride_row >= d->dim && d->out_stride_row >= d->dim;
}

bool supported(const turbine_rmsnorm_desc *d) {
  if (!valid(d))
    return false;
  return ck_bucket(d) || turbine_hip::rmsnorm_fallback_supported(d);
}

int32_t launch_ck(turbine_ctx *ctx, const turbine_rmsnorm_desc *d) {
  ck_tile::Rmsnorm2dFwdHostArgs host{};
  host.p_x = d->x;
  host.p_x_residual = nullptr;
  host.p_sm_scale = nullptr;
  host.p_gamma = d->weight;
  host.p_y = d->out;
  host.p_y_residual = nullptr;
  host.p_y_scale = nullptr;
  host.p_invRms = nullptr;
  host.p_y_unquant = nullptr;
  host.epsilon = d->eps;
  host.m = static_cast<ck_tile::index_t>(d->rows);
  host.n = static_cast<ck_tile::index_t>(d->dim);
  host.x_stride = static_cast<ck_tile::index_t>(d->x_stride_row);
  host.xr_stride = static_cast<ck_tile::index_t>(d->x_stride_row);
  host.y_stride = static_cast<ck_tile::index_t>(d->out_stride_row);
  host.yr_stride = static_cast<ck_tile::index_t>(d->out_stride_row);

  const dim3 grids = Kernel::GridSize(host);
  const dim3 blocks = Kernel::BlockSize();
  const auto kargs = Kernel::MakeKargs(host);
  const ck_tile::stream_config stream{ctx->stream};
  (void)ck_tile::launch_kernel(
      stream, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
  return check_hip(ctx, hipGetLastError(), "rmsnorm2d_fwd launch");
}

} // namespace

extern "C" {

int32_t turbine_rmsnorm_supported(const turbine_rmsnorm_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_rmsnorm_impl(const turbine_rmsnorm_desc *d) {
  if (valid(d) && ck_bucket(d))
    return kImplCk;
  return kImplTurbine;
}

int32_t turbine_rmsnorm(turbine_ctx *ctx, const turbine_rmsnorm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rmsnorm: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_rmsnorm: unsupported configuration rows=" +
                    std::to_string(d->rows) + " dim=" + std::to_string(d->dim) +
                    " dtype=" + std::to_string(d->dtype));
  }
  if (d->x == nullptr || d->weight == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rmsnorm: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (ck_bucket(d))
    return launch_ck(ctx, d);
  return turbine_hip::launch_rmsnorm(ctx, d);
}

} // extern "C"
