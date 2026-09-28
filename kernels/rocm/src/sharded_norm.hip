// ABI v2.6 (Phase 5, tensor parallelism): the sharded RMSNorm of a rank's
// contiguous slice of a normalised dimension (OLMoE's QK-norm over all heads,
// split by heads). row_sumsq writes the F32 sum of squares of each row of the
// slice; the caller all-reduces the sums across ranks; rmsnorm_sharded scales
// the slice with the full sum.
//
// No ROCm library covers it (decision "P5 T6: sharded RMSNorm — provider
// evaluation"): ck_tile rmsnorm2d reduces and scales in one pipeline over the
// rows it is given and takes no external sum of squares.
//
// Numerics are the rmsnorm fallback kernel's (elementwise.hip) and the
// cpu-reference's: F32 accumulation, x * inv rounded to the dtype before the
// weight multiply. One 256-thread block per row; each thread sums a strided
// subset of the row, then a fixed tree over the block's partial sums: the
// order depends only on the row length, so a row's result never depends on
// the number of rows in the call or on the other rows (batch invariance), and
// row_sumsq followed by rmsnorm_sharded over the whole row is bitwise the
// Turbine rmsnorm kernel.
#include <hip/hip_fp16.h>

#include <string>

#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "turbine_hip";
constexpr int kBlock = 256;

// ---- dtype access: load to f32, store with round-to-nearest-even ----

template <typename T> struct Access;
template <> struct Access<float> {
  __device__ static float load(float v) { return v; }
  __device__ static float store(float v) { return v; }
};
template <> struct Access<__half> {
  __device__ static float load(__half v) { return __half2float(v); }
  __device__ static __half store(float v) { return __float2half(v); }
};
template <> struct Access<turbine_hip::Bf16> {
  __device__ static float load(turbine_hip::Bf16 v) {
    return turbine_hip::bf16_to_f32(v);
  }
  __device__ static turbine_hip::Bf16 store(float v) {
    return turbine_hip::f32_to_bf16(v);
  }
};

// Rounds v to T and back (the cpu-reference round_to).
template <typename T> __device__ inline float round_to(float v) {
  return Access<T>::load(Access<T>::store(v));
}

// The block's sum of squares of row[0, dim), in the rmsnorm fallback's order:
// thread t sums elements t, t + 256, ...; then a halving tree.
template <typename T>
__device__ float block_sumsq(const T *row, int64_t dim, float *partial) {
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float v = Access<T>::load(row[j]);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset /= 2) {
    if (static_cast<int>(threadIdx.x) < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return partial[0];
}

template <typename T>
__global__
__launch_bounds__(kBlock) void row_sumsq_kernel(const T *x, float *sumsq,
                                                int64_t dim, int64_t x_stride) {
  __shared__ float partial[kBlock];
  const int64_t r = blockIdx.x;
  const float total = block_sumsq<T>(x + r * x_stride, dim, partial);
  if (threadIdx.x == 0) {
    sumsq[r] = total;
  }
}

template <typename T>
__global__ __launch_bounds__(kBlock) void rmsnorm_sharded_kernel(
    const T *x, const T *weight, const float *sumsq, T *out, int64_t dim,
    int64_t full_dim, int64_t x_stride, int64_t out_stride, float eps) {
  const int64_t r = blockIdx.x;
  const T *x_row = x + r * x_stride;
  T *o_row = out + r * out_stride;
  const float inv = 1.0f / sqrtf(sumsq[r] / static_cast<float>(full_dim) + eps);
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float normed = round_to<T>(Access<T>::load(x_row[j]) * inv);
    o_row[j] = Access<T>::store(normed * Access<T>::load(weight[j]));
  }
}

template <typename T> struct RowSumsqLaunch {
  static void run(hipStream_t s, const turbine_row_sumsq_desc *d) {
    hipLaunchKernelGGL(
        row_sumsq_kernel<T>, dim3(static_cast<unsigned>(d->rows)), dim3(kBlock),
        0, s, static_cast<const T *>(d->x), d->sumsq, d->dim, d->x_stride_row);
  }
};

template <typename T> struct RmsnormShardedLaunch {
  static void run(hipStream_t s, const turbine_rmsnorm_sharded_desc *d) {
    hipLaunchKernelGGL(
        rmsnorm_sharded_kernel<T>, dim3(static_cast<unsigned>(d->rows)),
        dim3(kBlock), 0, s, static_cast<const T *>(d->x),
        static_cast<const T *>(d->weight), d->sumsq, static_cast<T *>(d->out),
        d->dim, d->full_dim, d->x_stride_row, d->out_stride_row, d->eps);
  }
};

// Launches Launch<T> for the descriptor's float dtype.
template <template <typename> class Launch, typename D>
void dispatch(int32_t dtype, hipStream_t s, const D *d) {
  switch (dtype) {
  case TURBINE_DTYPE_BF16:
    Launch<turbine_hip::Bf16>::run(s, d);
    break;
  case TURBINE_DTYPE_F16:
    Launch<__half>::run(s, d);
    break;
  default:
    Launch<float>::run(s, d);
    break;
  }
}

bool float_dtype(int32_t dtype) {
  return dtype == TURBINE_DTYPE_BF16 || dtype == TURBINE_DTYPE_F16 ||
         dtype == TURBINE_DTYPE_F32;
}

// One block per row: the grid's x dimension holds up to 2^31 - 1 blocks.
bool rows_fit(int64_t rows) { return rows >= 1 && rows <= INT32_MAX; }

bool sumsq_supported(const turbine_row_sumsq_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && rows_fit(d->rows) &&
         d->dim >= 1 && d->x_stride_row >= d->dim;
}

bool sharded_supported(const turbine_rmsnorm_sharded_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && rows_fit(d->rows) &&
         d->dim >= 1 && d->full_dim >= d->dim && d->x_stride_row >= d->dim &&
         d->out_stride_row >= d->dim;
}

std::string describe(int64_t rows, int64_t dim, int32_t dtype) {
  return "rows=" + std::to_string(rows) + " dim=" + std::to_string(dim) +
         " dtype=" + std::to_string(dtype);
}

} // namespace

extern "C" {

int32_t turbine_row_sumsq_supported(const turbine_row_sumsq_desc *d) {
  return sumsq_supported(d) ? 1 : 0;
}

const char *turbine_row_sumsq_impl(const turbine_row_sumsq_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_row_sumsq(turbine_ctx *ctx, const turbine_row_sumsq_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_row_sumsq: descriptor is NULL");
  }
  if (!sumsq_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_row_sumsq: unsupported configuration " +
                    describe(d->rows, d->dim, d->dtype) +
                    " x_stride=" + std::to_string(d->x_stride_row));
  }
  if (d->x == nullptr || d->sumsq == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_row_sumsq: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<RowSumsqLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "row_sumsq kernel launch");
}

int32_t
turbine_rmsnorm_sharded_supported(const turbine_rmsnorm_sharded_desc *d) {
  return sharded_supported(d) ? 1 : 0;
}

const char *
turbine_rmsnorm_sharded_impl(const turbine_rmsnorm_sharded_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_rmsnorm_sharded(turbine_ctx *ctx,
                                const turbine_rmsnorm_sharded_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_rmsnorm_sharded: descriptor is NULL");
  }
  if (!sharded_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_rmsnorm_sharded: unsupported configuration " +
                    describe(d->rows, d->dim, d->dtype) +
                    " full_dim=" + std::to_string(d->full_dim) +
                    " strides=" + std::to_string(d->x_stride_row) + "/" +
                    std::to_string(d->out_stride_row));
  }
  if (d->x == nullptr || d->weight == nullptr || d->sumsq == nullptr ||
      d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_rmsnorm_sharded: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<RmsnormShardedLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "rmsnorm_sharded kernel launch");
}

} // extern "C"
