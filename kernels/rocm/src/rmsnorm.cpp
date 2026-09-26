// RMSNorm and the fused residual add + RMSNorm (ABI v2.1): ck_tile
// rmsnorm2d_fwd for the hand-instantiated BF16 buckets, the Turbine HIP kernels
// (elementwise.hip) for every other dimension, stride and dtype.
//
// The CK instances are the ones CK's rmsnorm2d generator emits for BF16 ->
// BF16, n % 8 == 0, no quant, with the T5-like ("model sensitive") pipeline:
// x * inv_rms is rounded to BF16 before the gamma multiply, which is Hugging
// Face LlamaRMSNorm's rounding and the cpu-reference numerics.
//   rmsnorm:      2048 < n <= 3072, no fused add.
//   add_rmsnorm:  1024 < n <= 2048 and 2048 < n <= 3072, fused add
//                 PRE_ADD_STORE: residual + x is rounded to BF16, stored back
//                 into the residual and normalised from that rounded value
//                 (the pipeline re-reads the stored BF16), so the result is
//                 add followed by rmsnorm, as the ABI requires.
#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/rmsnorm2d.hpp>

#include <string>

#include "turbine_hip.hpp"

using turbine_hip::add_rmsnorm_fallback_supported;
using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;
using turbine_hip::launch_add_rmsnorm;
using turbine_hip::launch_rmsnorm;
using turbine_hip::rmsnorm_fallback_supported;

namespace {

constexpr const char *kImplCk = "ck_tile_rmsnorm2d";
constexpr const char *kImplTurbine = "turbine_hip";

constexpr int64_t kVector = 8;
constexpr int64_t kThreads = 256;

// One CK rmsnorm2d instance: a block of 256 threads handles one row of up to
// RepeatN * 256 * 8 elements (padded), with the given fused add.
template <int RepeatN, ck_tile::Rmsnorm2dFusedAddEnum Add> struct CkRmsnorm {
  using BlockTile = ck_tile::sequence<1, RepeatN * kThreads * kVector>;
  using ThreadPerBlock = ck_tile::sequence<1, kThreads>;
  using Vector = ck_tile::sequence<1, kVector>;
  using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;

  using PipelineTraits = ck_tile::Rmsnorm2dFwdTraits<
      /*kPadN=*/true, /*kSaveInvRms=*/false, /*kSaveUnquant=*/false,
      /*kTwoPass=*/false, Add, ck_tile::Rmsnorm2dFusedQuantEnum::NO_SWEEP,
      ck_tile::Rmsnorm2dSensitiveEnum::T5_MODEL_LIKE>;

  using Problem = ck_tile::Rmsnorm2dFwdPipelineProblem<
      ck_tile::bf16_t, // X (and the residual)
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

  static int32_t launch(turbine_ctx *ctx,
                        const ck_tile::Rmsnorm2dFwdHostArgs &host,
                        const char *what) {
    const dim3 grids = Kernel::GridSize(host);
    const dim3 blocks = Kernel::BlockSize();
    const auto kargs = Kernel::MakeKargs(host);
    const ck_tile::stream_config stream{ctx->stream};
    (void)ck_tile::launch_kernel(
        stream, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    return check_hip(ctx, hipGetLastError(), what);
  }
};

using CkNorm3072 = CkRmsnorm<2, ck_tile::Rmsnorm2dFusedAddEnum::NO_ADD>;
using CkAddNorm2048 =
    CkRmsnorm<1, ck_tile::Rmsnorm2dFusedAddEnum::PRE_ADD_STORE>;
using CkAddNorm3072 =
    CkRmsnorm<2, ck_tile::Rmsnorm2dFusedAddEnum::PRE_ADD_STORE>;

// A BF16 row stride CK's vector loads accept (and that fits index_t).
bool ck_stride(int64_t stride) {
  return stride % kVector == 0 && stride <= INT32_MAX;
}

// ---- rmsnorm ----

// The instantiated bucket: 2048 < n <= 3072, n % 8 == 0.
bool ck_bucket(const turbine_rmsnorm_desc *d) {
  return d->dtype == TURBINE_DTYPE_BF16 && d->dim > 2048 && d->dim <= 3072 &&
         d->dim % kVector == 0 && ck_stride(d->x_stride_row) &&
         ck_stride(d->out_stride_row) && d->rows <= INT32_MAX;
}

bool valid(const turbine_rmsnorm_desc *d) {
  return d != nullptr && d->rows >= 1 && d->dim >= 1 &&
         d->x_stride_row >= d->dim && d->out_stride_row >= d->dim;
}

bool supported(const turbine_rmsnorm_desc *d) {
  if (!valid(d))
    return false;
  return ck_bucket(d) || rmsnorm_fallback_supported(d);
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
  return CkNorm3072::launch(ctx, host, "rmsnorm2d_fwd launch");
}

// ---- add_rmsnorm ----

// The instantiated buckets: 1024 < n <= 3072, n % 8 == 0 (the 2048-wide tile
// up to 2048, the 4096-wide padded tile above).
bool add_ck_bucket(const turbine_add_rmsnorm_desc *d) {
  return d->dtype == TURBINE_DTYPE_BF16 && d->dim > 1024 && d->dim <= 3072 &&
         d->dim % kVector == 0 && ck_stride(d->residual_stride_row) &&
         ck_stride(d->x_stride_row) && ck_stride(d->out_stride_row) &&
         d->rows <= INT32_MAX;
}

bool add_valid(const turbine_add_rmsnorm_desc *d) {
  return d != nullptr && d->rows >= 1 && d->dim >= 1 &&
         d->residual_stride_row >= d->dim && d->x_stride_row >= d->dim &&
         d->out_stride_row >= d->dim;
}

bool add_supported(const turbine_add_rmsnorm_desc *d) {
  if (!add_valid(d))
    return false;
  return add_ck_bucket(d) || add_rmsnorm_fallback_supported(d);
}

int32_t launch_add_ck(turbine_ctx *ctx, const turbine_add_rmsnorm_desc *d) {
  ck_tile::Rmsnorm2dFwdHostArgs host{};
  host.p_x = d->x;
  host.p_x_residual = d->residual;
  host.p_sm_scale = nullptr;
  host.p_gamma = d->weight;
  host.p_y = d->out;
  // In place: each row's residual is loaded whole before the sum is stored.
  host.p_y_residual = d->residual;
  host.p_y_scale = nullptr;
  host.p_invRms = nullptr;
  host.p_y_unquant = nullptr;
  host.epsilon = d->eps;
  host.m = static_cast<ck_tile::index_t>(d->rows);
  host.n = static_cast<ck_tile::index_t>(d->dim);
  host.x_stride = static_cast<ck_tile::index_t>(d->x_stride_row);
  host.xr_stride = static_cast<ck_tile::index_t>(d->residual_stride_row);
  host.y_stride = static_cast<ck_tile::index_t>(d->out_stride_row);
  host.yr_stride = static_cast<ck_tile::index_t>(d->residual_stride_row);
  if (d->dim <= 2048)
    return CkAddNorm2048::launch(ctx, host, "rmsnorm2d_fwd (add) launch");
  return CkAddNorm3072::launch(ctx, host, "rmsnorm2d_fwd (add) launch");
}

std::string describe(const turbine_add_rmsnorm_desc *d) {
  return "rows=" + std::to_string(d->rows) + " dim=" + std::to_string(d->dim) +
         " dtype=" + std::to_string(d->dtype) +
         " strides=" + std::to_string(d->residual_stride_row) + "/" +
         std::to_string(d->x_stride_row) + "/" +
         std::to_string(d->out_stride_row);
}

} // namespace

namespace turbine_hip {

bool rmsnorm_supports(const turbine_rmsnorm_desc *d, bool ck) {
  if (!valid(d))
    return false;
  return ck ? ck_bucket(d) : rmsnorm_fallback_supported(d);
}

int32_t rmsnorm_run(turbine_ctx *ctx, const turbine_rmsnorm_desc *d, bool ck) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rmsnorm: descriptor is NULL");
  }
  if (!rmsnorm_supports(d, ck)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_rmsnorm: unsupported configuration rows=" +
                    std::to_string(d->rows) + " dim=" + std::to_string(d->dim) +
                    " dtype=" + std::to_string(d->dtype) +
                    (supported(d)
                         ? std::string(" for ") + (ck ? kImplCk : kImplTurbine)
                         : std::string()));
  }
  if (d->x == nullptr || d->weight == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rmsnorm: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (ck)
    return launch_ck(ctx, d);
  return launch_rmsnorm(ctx, d);
}

bool add_rmsnorm_supports(const turbine_add_rmsnorm_desc *d, bool ck) {
  if (!add_valid(d))
    return false;
  return ck ? add_ck_bucket(d) : add_rmsnorm_fallback_supported(d);
}

int32_t add_rmsnorm_run(turbine_ctx *ctx, const turbine_add_rmsnorm_desc *d,
                        bool ck) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_add_rmsnorm: descriptor is NULL");
  }
  if (!add_rmsnorm_supports(d, ck)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_add_rmsnorm: unsupported configuration " +
                    describe(d) +
                    (add_supported(d)
                         ? std::string(" for ") + (ck ? kImplCk : kImplTurbine)
                         : std::string()));
  }
  if (d->residual == nullptr || d->x == nullptr || d->weight == nullptr ||
      d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_add_rmsnorm: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (ck)
    return launch_add_ck(ctx, d);
  return launch_add_rmsnorm(ctx, d);
}

} // namespace turbine_hip

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
  return turbine_hip::rmsnorm_run(ctx, d, valid(d) && ck_bucket(d));
}

int32_t turbine_add_rmsnorm_supported(const turbine_add_rmsnorm_desc *d) {
  return add_supported(d) ? 1 : 0;
}

const char *turbine_add_rmsnorm_impl(const turbine_add_rmsnorm_desc *d) {
  if (add_valid(d) && add_ck_bucket(d))
    return kImplCk;
  return kImplTurbine;
}

int32_t turbine_add_rmsnorm(turbine_ctx *ctx,
                            const turbine_add_rmsnorm_desc *d) {
  return turbine_hip::add_rmsnorm_run(ctx, d, add_valid(d) && add_ck_bucket(d));
}

} // extern "C"
