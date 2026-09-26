// Turbine HIP kernels: embedding gather, half-split RoPE, SiLU-and-multiply,
// residual add and the RMSNorm and add+RMSNorm fallbacks. No ROCm library ships these as
// standalone kernels (spec S-7). Numerics follow the cpu-reference provider:
// f32 compute, rounding to the activation dtype at the points where the
// Hugging Face BF16 forward materialises a tensor in that dtype.
#include <hip/hip_fp16.h>

#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "turbine_hip";
constexpr int kBlock = 256;

// ---- dtype access: load to f32, store with round-to-nearest-even ----

// BF16 storage as raw bits. The f32 -> BF16 rounding is spelled out (round to
// nearest, ties to even; NaN stays NaN) so it matches the cpu-reference
// round_to bit for bit instead of depending on how the compiler lowers a
// __bf16 conversion on the target.
struct Bf16 {
  uint16_t bits;
};

template <typename T> __device__ inline float to_f32(T v);
template <> __device__ inline float to_f32<float>(float v) { return v; }
template <> __device__ inline float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ inline float to_f32<Bf16>(Bf16 v) {
  return __uint_as_float(static_cast<uint32_t>(v.bits) << 16);
}

template <typename T> __device__ inline T from_f32(float v);
template <> __device__ inline float from_f32<float>(float v) { return v; }
template <> __device__ inline __half from_f32<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline Bf16 from_f32<Bf16>(float v) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return Bf16{static_cast<uint16_t>((u >> 16) | 0x0040u)};
  }
  u += 0x7fffu + ((u >> 16) & 1u);
  return Bf16{static_cast<uint16_t>(u >> 16)};
}

// Rounds v to T and back (the cpu-reference round_to).
template <typename T> __device__ inline float round_to(float v) {
  return to_f32<T>(from_f32<T>(v));
}

bool float_dtype(int32_t dtype) {
  return dtype == TURBINE_DTYPE_BF16 || dtype == TURBINE_DTYPE_F16 ||
         dtype == TURBINE_DTYPE_F32;
}

// Launches kernel<T> for the descriptor's float dtype.
template <template <typename> class Launch, typename... Args>
void dispatch(int32_t dtype, Args... args) {
  switch (dtype) {
  case TURBINE_DTYPE_BF16:
    Launch<Bf16>::run(args...);
    break;
  case TURBINE_DTYPE_F16:
    Launch<__half>::run(args...);
    break;
  default:
    Launch<float>::run(args...);
    break;
  }
}

unsigned grid_for(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 65535 ? blocks : 65535);
}

// ---- embedding ----

template <typename T>
__global__ void embedding_kernel(const int32_t *ids, const T *table, T *out,
                                 int64_t hidden, int64_t vocab_offset,
                                 int64_t vocab_rows, int64_t out_stride_row) {
  const int64_t t = blockIdx.x;
  const int64_t row = static_cast<int64_t>(ids[t]) - vocab_offset;
  T *dst = out + t * out_stride_row;
  if (row >= 0 && row < vocab_rows) {
    const T *src = table + row * hidden;
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x)
      dst[j] = src[j];
  } else {
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
      dst[j] = from_f32<T>(0.0f);
    }
  }
}

template <typename T> struct EmbeddingLaunch {
  static void run(hipStream_t s, const turbine_embedding_desc *d) {
    hipLaunchKernelGGL(embedding_kernel<T>,
                       dim3(static_cast<unsigned>(d->num_tokens)), dim3(kBlock),
                       0, s, d->ids, static_cast<const T *>(d->table),
                       static_cast<T *>(d->out), d->hidden, d->vocab_offset,
                       d->vocab_rows, d->out_stride_row);
  }
};

bool embedding_supported(const turbine_embedding_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->num_tokens >= 0 &&
         d->num_tokens <= INT32_MAX && d->hidden >= 1 && d->vocab_rows >= 0 &&
         d->out_stride_row >= d->hidden;
}

// ---- add ----

template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    out[i] = from_f32<T>(to_f32<T>(a[i]) + to_f32<T>(b[i]));
  }
}

template <typename T> struct AddLaunch {
  static void run(hipStream_t s, const turbine_add_desc *d) {
    hipLaunchKernelGGL(add_kernel<T>, dim3(grid_for(d->n)), dim3(kBlock), 0, s,
                       static_cast<const T *>(d->a),
                       static_cast<const T *>(d->b), static_cast<T *>(d->out),
                       d->n);
  }
};

bool add_supported(const turbine_add_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->n >= 0;
}

// ---- silu_mul ----

// Grid: (rows, column chunks of kSiluCols): a decode batch of a few rows
// still spreads over the whole device.
constexpr int64_t kSiluCols = 4 * kBlock;

unsigned silu_chunks(int64_t cols) {
  const int64_t chunks = (cols + kSiluCols - 1) / kSiluCols;
  return static_cast<unsigned>(chunks < 65535 ? chunks : 65535);
}

template <typename T>
__global__ void silu_mul_kernel(const T *gate, const T *up, T *out,
                                int64_t cols, int64_t gate_stride,
                                int64_t up_stride, int64_t out_stride) {
  const int64_t r = blockIdx.x;
  const T *g_row = gate + r * gate_stride;
  const T *u_row = up + r * up_stride;
  T *o_row = out + r * out_stride;
  const int64_t step = static_cast<int64_t>(gridDim.y) * blockDim.x;
  for (int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       j < cols; j += step) {
    const float g = to_f32<T>(g_row[j]);
    const float silu = g / (1.0f + expf(-g));
    o_row[j] = from_f32<T>(round_to<T>(silu) * to_f32<T>(u_row[j]));
  }
}

template <typename T> struct SiluMulLaunch {
  static void run(hipStream_t s, const turbine_silu_mul_desc *d) {
    hipLaunchKernelGGL(
        silu_mul_kernel<T>,
        dim3(static_cast<unsigned>(d->rows), silu_chunks(d->cols)),
        dim3(kBlock), 0, s, static_cast<const T *>(d->gate),
        static_cast<const T *>(d->up), static_cast<T *>(d->out), d->cols,
        d->gate_stride_row, d->up_stride_row, d->out_stride_row);
  }
};

bool silu_mul_supported(const turbine_silu_mul_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->rows >= 0 &&
         d->rows <= INT32_MAX && d->cols >= 1 &&
         d->gate_stride_row >= d->cols && d->up_stride_row >= d->cols &&
         d->out_stride_row >= d->cols;
}

// ---- rope (half-split, HF rotate_half) ----

// Rotary pairs whose cos/sin a block stages in LDS (half of rotary_dim); a
// wider rotation evaluates them per element.
constexpr int32_t kRopeMaxHalf = 512;

// cos/sin of the f32 angle evaluated in f64 and rounded to f32: the correctly
// rounded f32 value a host libm returns, which the device's f32 cosf/sinf (a
// few ulp) can miss by one BF16 rounding step. Then rounded to T.
template <typename T> __device__ inline float rope_cos(float f) {
  return round_to<T>(static_cast<float>(cos(static_cast<double>(f))));
}
template <typename T> __device__ inline float rope_sin(float f) {
  return round_to<T>(static_cast<float>(sin(static_cast<double>(f))));
}

// Heads (q heads, then k heads) one block rotates: enough pairs to fill it.
int32_t rope_heads_per_block(int32_t half) {
  const int32_t h = kBlock / half;
  return h > 0 ? h : 1;
}

// Grid: (tokens, groups of heads_per_block heads). Each block evaluates the
// token's cos/sin once per rotary pair (in LDS) for all of its heads.
template <typename T>
__global__ void rope_kernel(T *q, T *k, const int32_t *positions,
                            const float *inv_freq, int32_t num_q_heads,
                            int32_t num_kv_heads, int32_t head_dim,
                            int32_t rotary_dim, int64_t q_stride,
                            int64_t k_stride, int32_t heads_per_block) {
  __shared__ float cos_s[kRopeMaxHalf];
  __shared__ float sin_s[kRopeMaxHalf];
  const int64_t t = blockIdx.x;
  const int32_t half = rotary_dim / 2;
  const int32_t heads = num_q_heads + num_kv_heads;
  const int32_t h0 = static_cast<int32_t>(blockIdx.y) * heads_per_block;
  const int32_t h1 = min(heads, h0 + heads_per_block);
  const float pos = static_cast<float>(positions[t]);
  const bool staged = half <= kRopeMaxHalf;
  if (staged) {
    for (int32_t i = threadIdx.x; i < half; i += blockDim.x) {
      const float f = pos * inv_freq[i];
      cos_s[i] = rope_cos<T>(f);
      sin_s[i] = rope_sin<T>(f);
    }
    __syncthreads();
  }
  for (int32_t w = threadIdx.x; w < (h1 - h0) * half; w += blockDim.x) {
    const int32_t h = h0 + w / half;
    const int32_t i = w % half;
    T *x = h < num_q_heads
               ? q + t * q_stride + static_cast<int64_t>(h) * head_dim
               : k + t * k_stride +
                     static_cast<int64_t>(h - num_q_heads) * head_dim;
    float c;
    float s;
    if (staged) {
      c = cos_s[i];
      s = sin_s[i];
    } else {
      const float f = pos * inv_freq[i];
      c = rope_cos<T>(f);
      s = rope_sin<T>(f);
    }
    const float x1 = to_f32<T>(x[i]);
    const float x2 = to_f32<T>(x[i + half]);
    x[i] = from_f32<T>(round_to<T>(x1 * c) + round_to<T>(-x2 * s));
    x[i + half] = from_f32<T>(round_to<T>(x2 * c) + round_to<T>(x1 * s));
  }
}

template <typename T> struct RopeLaunch {
  static void run(hipStream_t s, const turbine_rope_desc *d) {
    const int32_t per_block = rope_heads_per_block(d->rotary_dim / 2);
    const int32_t heads = d->num_q_heads + d->num_kv_heads;
    const int32_t groups = (heads + per_block - 1) / per_block;
    hipLaunchKernelGGL(
        rope_kernel<T>,
        dim3(static_cast<unsigned>(d->num_tokens),
             static_cast<unsigned>(groups)),
        dim3(kBlock), 0, s, static_cast<T *>(d->q), static_cast<T *>(d->k),
        d->positions, d->inv_freq, d->num_q_heads, d->num_kv_heads, d->head_dim,
        d->rotary_dim, d->q_stride_token, d->k_stride_token, per_block);
  }
};

bool rope_supported(const turbine_rope_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->style == 0 &&
         d->num_tokens >= 0 && d->num_tokens <= INT32_MAX &&
         d->num_q_heads >= 1 && d->num_kv_heads >= 0 && d->head_dim >= 2 &&
         d->rotary_dim >= 2 && d->rotary_dim % 2 == 0 &&
         d->rotary_dim <= d->head_dim &&
         // grid.y: one block per group of heads (at most one head per block).
         static_cast<int64_t>(d->num_q_heads) + d->num_kv_heads <= 65535 &&
         d->q_stride_token >=
             static_cast<int64_t>(d->num_q_heads) * d->head_dim &&
         d->k_stride_token >=
             static_cast<int64_t>(d->num_kv_heads) * d->head_dim;
}

// ---- rmsnorm fallback ----

template <typename T>
__global__ void rmsnorm_kernel(const T *x, const T *weight, T *out, int64_t dim,
                               int64_t x_stride, int64_t out_stride,
                               float eps) {
  __shared__ float partial[kBlock];
  const int64_t r = blockIdx.x;
  const T *x_row = x + r * x_stride;
  T *o_row = out + r * out_stride;
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float v = to_f32<T>(x_row[j]);
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
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(dim) + eps);
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float normed = round_to<T>(to_f32<T>(x_row[j]) * inv);
    o_row[j] = from_f32<T>(normed * to_f32<T>(weight[j]));
  }
}

template <typename T> struct RmsnormLaunch {
  static void run(hipStream_t s, const turbine_rmsnorm_desc *d) {
    hipLaunchKernelGGL(rmsnorm_kernel<T>, dim3(static_cast<unsigned>(d->rows)),
                       dim3(kBlock), 0, s, static_cast<const T *>(d->x),
                       static_cast<const T *>(d->weight),
                       static_cast<T *>(d->out), d->dim, d->x_stride_row,
                       d->out_stride_row, d->eps);
  }
};

// ---- add_rmsnorm fallback (ABI v2.1) ----

// residual = round(residual + x) in place, then out = rmsnorm(residual): the
// rmsnorm_kernel above on the rounded sum, each thread re-reading only the
// elements it stored itself.
template <typename T>
__global__ void add_rmsnorm_kernel(T *residual, const T *x, const T *weight,
                                   T *out, int64_t dim, int64_t residual_stride,
                                   int64_t x_stride, int64_t out_stride,
                                   float eps) {
  __shared__ float partial[kBlock];
  const int64_t r = blockIdx.x;
  T *r_row = residual + r * residual_stride;
  const T *x_row = x + r * x_stride;
  T *o_row = out + r * out_stride;
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float v = round_to<T>(to_f32<T>(r_row[j]) + to_f32<T>(x_row[j]));
    r_row[j] = from_f32<T>(v);
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
  const float inv = 1.0f / sqrtf(partial[0] / static_cast<float>(dim) + eps);
  for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
    const float normed = round_to<T>(to_f32<T>(r_row[j]) * inv);
    o_row[j] = from_f32<T>(normed * to_f32<T>(weight[j]));
  }
}

template <typename T> struct AddRmsnormLaunch {
  static void run(hipStream_t s, const turbine_add_rmsnorm_desc *d) {
    hipLaunchKernelGGL(add_rmsnorm_kernel<T>,
                       dim3(static_cast<unsigned>(d->rows)), dim3(kBlock), 0, s,
                       static_cast<T *>(d->residual),
                       static_cast<const T *>(d->x),
                       static_cast<const T *>(d->weight),
                       static_cast<T *>(d->out), d->dim, d->residual_stride_row,
                       d->x_stride_row, d->out_stride_row, d->eps);
  }
};

std::string unsupported(const char *op, int32_t dtype) {
  return std::string(op) +
         ": unsupported configuration (dtype=" + std::to_string(dtype) + ")";
}

} // namespace

namespace turbine_hip {

bool rmsnorm_fallback_supported(const turbine_rmsnorm_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->rows >= 1 &&
         d->rows <= INT32_MAX && d->dim >= 1 && d->x_stride_row >= d->dim &&
         d->out_stride_row >= d->dim;
}

int32_t launch_rmsnorm(turbine_ctx *ctx, const turbine_rmsnorm_desc *d) {
  dispatch<RmsnormLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "rmsnorm kernel launch");
}

bool add_rmsnorm_fallback_supported(const turbine_add_rmsnorm_desc *d) {
  return d != nullptr && float_dtype(d->dtype) && d->rows >= 1 &&
         d->rows <= INT32_MAX && d->dim >= 1 &&
         d->residual_stride_row >= d->dim && d->x_stride_row >= d->dim &&
         d->out_stride_row >= d->dim;
}

int32_t launch_add_rmsnorm(turbine_ctx *ctx,
                           const turbine_add_rmsnorm_desc *d) {
  dispatch<AddRmsnormLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "add_rmsnorm kernel launch");
}

} // namespace turbine_hip

extern "C" {

// ---- embedding ----

int32_t turbine_embedding_supported(const turbine_embedding_desc *d) {
  return embedding_supported(d) ? 1 : 0;
}

const char *turbine_embedding_impl(const turbine_embedding_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_embedding(turbine_ctx *ctx, const turbine_embedding_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_embedding: descriptor is NULL");
  }
  if (!embedding_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                unsupported("turbine_embedding", d->dtype));
  }
  if (d->num_tokens == 0)
    return TURBINE_OK;
  if (d->ids == nullptr || d->table == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_embedding: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<EmbeddingLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "embedding kernel launch");
}

// ---- add ----

int32_t turbine_add_supported(const turbine_add_desc *d) {
  return add_supported(d) ? 1 : 0;
}

const char *turbine_add_impl(const turbine_add_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_add(turbine_ctx *ctx, const turbine_add_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_add: descriptor is NULL");
  }
  if (!add_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                unsupported("turbine_add", d->dtype));
  }
  if (d->n == 0)
    return TURBINE_OK;
  if (d->a == nullptr || d->b == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_add: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<AddLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "add kernel launch");
}

// ---- silu_mul ----

int32_t turbine_silu_mul_supported(const turbine_silu_mul_desc *d) {
  return silu_mul_supported(d) ? 1 : 0;
}

const char *turbine_silu_mul_impl(const turbine_silu_mul_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_silu_mul(turbine_ctx *ctx, const turbine_silu_mul_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_silu_mul: descriptor is NULL");
  }
  if (!silu_mul_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                unsupported("turbine_silu_mul", d->dtype));
  }
  if (d->rows == 0)
    return TURBINE_OK;
  if (d->gate == nullptr || d->up == nullptr || d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_silu_mul: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<SiluMulLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "silu_mul kernel launch");
}

// ---- rope ----

int32_t turbine_rope_supported(const turbine_rope_desc *d) {
  return rope_supported(d) ? 1 : 0;
}

const char *turbine_rope_impl(const turbine_rope_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_rope(turbine_ctx *ctx, const turbine_rope_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rope: descriptor is NULL");
  }
  if (!rope_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                unsupported("turbine_rope", d->dtype));
  }
  if (d->num_tokens == 0)
    return TURBINE_OK;
  if (d->q == nullptr || (d->num_kv_heads > 0 && d->k == nullptr) ||
      d->positions == nullptr || d->inv_freq == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_rope: NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  dispatch<RopeLaunch>(d->dtype, ctx->stream, d);
  return check_hip(ctx, hipGetLastError(), "rope kernel launch");
}

} // extern "C"
