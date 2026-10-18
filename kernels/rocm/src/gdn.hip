// Gated DeltaNet (ABI v2.18 turbine_gdn_prefill / turbine_gdn_decode) on ROCm
// (Phase 7 Task 30): the delta-rule recurrence of Qwen3.5's linear-attention
// layers, one persistent warp per (sequence, value head, state column).
//
// Third-party code: the kernel below is llama.cpp's gated_delta_net_cuda
// (ggml/src/ggml-cuda/gated_delta_net.cu) at commit
// bd4eeaa047006cb1fe71999fbd11134b5836e167 (github.com/ggml-org/llama.cpp),
// MIT, Copyright (c) 2023-2026 The ggml authors; the license text is
// kernels/rocm/third_party/LICENSES/llama.cpp-LICENSE. Taken: the per-column
// warp mapping (a warp owns one state column S[col][:], k contiguous, the
// rows spread over its lanes, warp-shuffle reductions for S^T k and S^T q),
// the fused update S = g S + k delta with the read S^T q in the same pass, and
// the transposed state layout [v_dim, k_dim]. Changed (Turbine, 2026-10-08;
// decision "P7 Gated DeltaNet provider", B): q, k, v and the output are BF16
// with their own token row strides (a conv1d output feeds the kernel in
// place); q and k are L2-normalised in F32 per token (transformers'
// use_qk_l2norm_in_kernel, eps 1e-6) and q is scaled by 1 / sqrt(k_dim) before
// the read instead of the output after; value head h reads key head
// h / (v_heads / k_heads) (transformers' repeat_interleave, llama.cpp reads
// h % k_heads); the sequences are delimited by cu_seqlens (one token each for
// decode), each reads and writes its own state slot through state_slots; the
// state after every B tokens of a sequence is also written to boundary_states
// (Task 32's snapshots); the ggml dispatch, KDA variant, keep-intermediate
// states and fast-division helpers are dropped; wave32 only (a warp is 32
// lanes).
//
// Every (sequence, head, column) is computed by one warp in a fixed order, so
// a result does not depend on the other sequences of the call, and a prefill
// split into two calls with the state carried equals the whole prefill.
#include <string>

#include "bf16.hpp"
#include "gdn.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {
namespace {

constexpr const char *kImpl = "llama_cpp_gated_delta_net";
constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr float kL2Eps = 1e-6f;

struct GdnArgs {
  const uint16_t *q;
  const uint16_t *k;
  const uint16_t *v;
  const float *g;
  const float *beta;
  float *state;
  const int32_t *slots;
  const int32_t *cu; // null: one token per sequence
  uint16_t *out;
  float *boundary;
  int64_t q_stride, k_stride, v_stride, out_stride;
  int32_t boundary_tokens;
  int32_t v_heads;
  int32_t per_key; // v_heads / k_heads
};

__device__ inline float warp_sum(float x) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1)
    x += __shfl_xor(x, off, kWarp);
  return x;
}

__device__ inline float load_bf16(const uint16_t *p) {
  return bf16_to_f32(Bf16{*p});
}

// ---- llama.cpp gated_delta_net_cuda, adapted (see the header) ----
template <int S_v>
__global__ void __launch_bounds__(kWarp *kWarps, 2) gdn_kernel(GdnArgs a) {
  constexpr int rows_per_lane = S_v / kWarp;
  const int h = blockIdx.x;
  const int seq = blockIdx.y;
  const int lane = threadIdx.x;
  const int col = blockIdx.z * kWarps + threadIdx.y;
  const int kh = h / a.per_key;

  const int tok0 = a.cu != nullptr ? a.cu[seq] : seq;
  const int tok1 = a.cu != nullptr ? a.cu[seq + 1] : seq + 1;
  const int64_t slot = a.slots[seq];
  float *st = a.state + ((slot * a.v_heads + h) * S_v + col) * S_v;

  int row0 = 0;
  if (a.boundary != nullptr) {
    for (int s = 0; s < seq; ++s)
      row0 += (a.cu[s + 1] - a.cu[s]) / a.boundary_tokens;
  }

  const float q_scale = 1.0f / sqrtf(static_cast<float>(S_v));
  float s_shard[rows_per_lane];
#pragma unroll
  for (int r = 0; r < rows_per_lane; r++)
    s_shard[r] = st[r * kWarp + lane];

  for (int t = tok0; t < tok1; t++) {
    const uint16_t *q_t = a.q + t * a.q_stride + kh * S_v;
    const uint16_t *k_t = a.k + t * a.k_stride + kh * S_v;
    const uint16_t *v_t = a.v + t * a.v_stride + h * S_v;
    const float beta_val = a.beta[t * a.v_heads + h];
    const float g_val = expf(a.g[t * a.v_heads + h]);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float ksq = 0.0f, qsq = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      const int i = r * kWarp + lane;
      k_reg[r] = load_bf16(k_t + i);
      q_reg[r] = load_bf16(q_t + i);
      ksq += k_reg[r] * k_reg[r];
      qsq += q_reg[r] * q_reg[r];
    }
    const float k_norm = 1.0f / sqrtf(warp_sum(ksq) + kL2Eps);
    const float q_norm = 1.0f / sqrtf(warp_sum(qsq) + kL2Eps) * q_scale;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      k_reg[r] *= k_norm;
      q_reg[r] *= q_norm;
    }

    // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++)
      kv_shard += s_shard[r] * k_reg[r];
    const float kv_col = warp_sum(kv_shard);

    // delta[col] = (v[col] - g * kv[col]) * beta
    const float delta_col = (load_bf16(v_t + col) - g_val * kv_col) * beta_val;

    // fused: S[i][col] = g * S[i][col] + k[i] * delta[col];
    // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
      attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = warp_sum(attn_partial);
    if (lane == 0)
      a.out[t * a.out_stride + h * S_v + col] = f32_to_bf16(attn_col).bits;

    const int done = t - tok0 + 1;
    if (a.boundary != nullptr && done % a.boundary_tokens == 0) {
      const int64_t row = row0 + done / a.boundary_tokens - 1;
      float *dst = a.boundary + ((row * a.v_heads + h) * S_v + col) * S_v;
#pragma unroll
      for (int r = 0; r < rows_per_lane; r++)
        dst[r * kWarp + lane] = s_shard[r];
    }
  }

#pragma unroll
  for (int r = 0; r < rows_per_lane; r++)
    st[r * kWarp + lane] = s_shard[r];
}
// ---- end of llama.cpp ----

std::string describe(const turbine_gdn_desc *d) {
  return "k_heads=" + std::to_string(d->k_heads) +
         " v_heads=" + std::to_string(d->v_heads) +
         " k_dim=" + std::to_string(d->k_dim) +
         " v_dim=" + std::to_string(d->v_dim) +
         " dtype=" + std::to_string(d->dtype);
}

int32_t run(turbine_ctx *ctx, const turbine_gdn_desc *d, bool prefill) {
  const char *what = prefill ? "turbine_gdn_prefill" : "turbine_gdn_decode";
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(what) + ": descriptor is NULL");
  if (!gdn_supports(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + " (" + kImpl +
                    "): unsupported configuration " + describe(d));
  }
  if (d->num_tokens <= 0 || d->num_seqs <= 0)
    return TURBINE_OK;
  if (d->q == nullptr || d->k == nullptr || d->v == nullptr ||
      d->g == nullptr || d->beta == nullptr || d->state == nullptr ||
      d->state_slots == nullptr || d->out == nullptr ||
      (prefill && d->cu_seqlens == nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(what) + ": NULL pointer");
  }
  if (d->boundary_states != nullptr &&
      (!prefill || d->boundary_tokens <= 0)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) +
                    ": boundary_states needs a prefill and boundary_tokens > 0");
  }
  if (d->q_stride < static_cast<int64_t>(d->k_heads) * d->k_dim ||
      d->k_stride < static_cast<int64_t>(d->k_heads) * d->k_dim ||
      d->v_stride < static_cast<int64_t>(d->v_heads) * d->v_dim ||
      d->out_stride < static_cast<int64_t>(d->v_heads) * d->v_dim) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": a token row stride is below the row width");
  }
  if (d->num_seqs > 65535 || (d->v_dim / kWarps) > 65535) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + ": more than 65535 sequences in one call");
  }
  if (!prefill && d->num_tokens != d->num_seqs) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": decode takes one token per sequence");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  GdnArgs a{static_cast<const uint16_t *>(d->q),
            static_cast<const uint16_t *>(d->k),
            static_cast<const uint16_t *>(d->v),
            d->g,
            d->beta,
            d->state,
            d->state_slots,
            prefill ? d->cu_seqlens : nullptr,
            static_cast<uint16_t *>(d->out),
            d->boundary_states,
            d->q_stride,
            d->k_stride,
            d->v_stride,
            d->out_stride,
            d->boundary_tokens,
            d->v_heads,
            d->v_heads / d->k_heads};
  const dim3 grid(static_cast<uint32_t>(d->v_heads),
                  static_cast<uint32_t>(d->num_seqs),
                  static_cast<uint32_t>(d->v_dim / kWarps));
  const dim3 block(kWarp, kWarps);
  switch (d->v_dim) {
  case 32:
    hipLaunchKernelGGL(gdn_kernel<32>, grid, block, 0, ctx->stream, a);
    break;
  case 64:
    hipLaunchKernelGGL(gdn_kernel<64>, grid, block, 0, ctx->stream, a);
    break;
  default:
    hipLaunchKernelGGL(gdn_kernel<128>, grid, block, 0, ctx->stream, a);
    break;
  }
  return check_hip(ctx, hipGetLastError(), what);
}

} // namespace

bool gdn_supports(const turbine_gdn_desc *d) {
  return d != nullptr && d->dtype == TURBINE_DTYPE_BF16 &&
         d->k_dim == d->v_dim &&
         (d->k_dim == 32 || d->k_dim == 64 || d->k_dim == 128) &&
         d->k_heads > 0 && d->v_heads > 0 && d->v_heads % d->k_heads == 0;
}

int32_t gdn_prefill_run(turbine_ctx *ctx, const turbine_gdn_desc *d) {
  return run(ctx, d, true);
}

int32_t gdn_decode_run(turbine_ctx *ctx, const turbine_gdn_desc *d) {
  return run(ctx, d, false);
}

} // namespace turbine_hip

extern "C" {

int32_t turbine_gdn_prefill_supported(const turbine_gdn_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_GDN_PREFILL, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_gdn_prefill_impl(const turbine_gdn_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_GDN_PREFILL, d);
}

int32_t turbine_gdn_prefill(turbine_ctx *ctx, const turbine_gdn_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_GDN_PREFILL, d);
}

int32_t turbine_gdn_decode_supported(const turbine_gdn_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_GDN_DECODE, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_gdn_decode_impl(const turbine_gdn_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_GDN_DECODE, d);
}

int32_t turbine_gdn_decode(turbine_ctx *ctx, const turbine_gdn_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_GDN_DECODE, d);
}

} // extern "C"
