// Causal conv1d, window 4, with SiLU (ABI v2.18 turbine_conv1d_prefill /
// turbine_conv1d_update) on ROCm (Phase 7 Task 30): the depthwise convolution
// in front of Qwen3.5's Gated DeltaNet, over the packed q | k | v channels.
//
// Third-party code, prefill: conv1d_split_qkv_channel_last_t of AITER's
// csrc/kernels/causal_conv1d_fwd_split_qkv.cu at commit
// 4bb520baca1d39ad6cee99e0261fe19d1778ec14 (github.com/ROCm/aiter), MIT,
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.; the license text is
// kernels/rocm/third_party/LICENSES/aiter-LICENSE. Taken: one thread per
// channel (adjacent channels per wave), a chunk of TM tokens per block along
// grid.x, the rolling window of four inputs in registers, the SiLU written
// with the exp2 / rcp intrinsics. Changed (Turbine, 2026-10-08; decision "P7
// Gated DeltaNet provider", B): the sequences are delimited by cu_seqlens and
// the (sequence, chunk) of a block is found by scanning it (no batch_ptr /
// token_chunk_offset_ptr); the state is always read (the caller zeroes a new
// sequence's slot: no has_initial_state) and the inputs before a sequence read
// from it by position; the output is one packed [tokens, channels] row (AITER
// splits it into q, k and v tensors; the Gated DeltaNet reads the three
// column ranges in place); the F32 to BF16 conversion rounds to nearest even
// (AITER truncates); the sum is accumulated left to right with FP contraction
// off, in a function shared with the update kernel, so prefill and update are
// bit-equal; the BF16 tail after every boundary_tokens tokens of a sequence
// is written to boundary_conv_states (Task 32's snapshots).
//
// Update: one token per sequence, a thread per (sequence, channel), the
// window in registers, structured like llama.cpp's ssm_conv kernel
// (ggml/src/ggml-cuda/ssm-conv.cu, MIT, bd4eeaa047006cb1fe71999fbd11134b5836e167;
// license kernels/rocm/third_party/LICENSES/llama.cpp-LICENSE) but reading the
// three previous inputs from the sequence's conv_state slot instead of a
// concatenated input, and writing the shifted window back.
#include <string>

#include "bf16.hpp"
#include "gdn.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {
namespace {

constexpr const char *kPrefillImpl = "aiter_causal_conv1d_split_qkv";
constexpr const char *kUpdateImpl = "llama_cpp_ssm_conv";
constexpr int kKw = 4;      // window
constexpr int kNt = 256;    // threads per block
constexpr int kTm = 16;     // tokens per chunk

struct ConvArgs {
  const uint16_t *x;
  const uint16_t *w;
  const uint16_t *bias; // may be null
  uint16_t *state;
  const int32_t *slots;
  const int32_t *cu; // null: one token per sequence
  uint16_t *out;
  uint16_t *boundary;
  int64_t x_stride, out_stride;
  int32_t boundary_tokens;
  int32_t num_seqs;
  int32_t channels;
  int32_t silu;
};

// bias + w0 x0 + w1 x1 + w2 x2 + w3 x3 in F32, left to right, then SiLU:
// shared by both kernels so they agree to the bit.
#pragma clang fp contract(off)
__device__ inline uint16_t conv_step(float bias, const float (&w)[kKw], float x0,
                                     float x1, float x2, float x3, bool silu) {
  float acc = bias + w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * x3;
  if (silu) {
    const float e = __builtin_amdgcn_exp2f(acc * (-1.4426950408889634f));
    acc *= __builtin_amdgcn_rcpf(1.f + e);
  }
  return f32_to_bf16(acc).bits;
}
#pragma clang fp contract(on)

__device__ inline float ld(const uint16_t *p) { return bf16_to_f32(Bf16{*p}); }

// ---- AITER conv1d_split_qkv_channel_last_t, adapted (see the header) ----
__global__ __launch_bounds__(kNt) void conv1d_prefill_kernel(ConvArgs a) {
  // (sequence, chunk) of this block: the pid-th chunk of the concatenated
  // schedule of ceil(len / kTm) chunks per sequence.
  int seq = 0, chunk = static_cast<int>(blockIdx.x);
  for (;; ++seq) {
    if (seq >= a.num_seqs)
      return;
    const int len = a.cu[seq + 1] - a.cu[seq];
    const int n = (len + kTm - 1) / kTm;
    if (chunk < n)
      break;
    chunk -= n;
  }
  const int gfeat = static_cast<int>(blockIdx.y) * kNt + threadIdx.x;
  if (gfeat >= a.channels)
    return;
  const int seq_start = a.cu[seq];
  const int seqlen = a.cu[seq + 1] - seq_start;
  const int tok_start = chunk * kTm;
  uint16_t *state = a.state + (static_cast<int64_t>(a.slots[seq]) * a.channels + gfeat) * 3;

  float w[kKw];
#pragma unroll
  for (int i = 0; i < kKw; ++i)
    w[i] = ld(a.w + static_cast<int64_t>(gfeat) * kKw + i);
  const float bias = a.bias != nullptr ? ld(a.bias + gfeat) : 0.f;

  const uint16_t *xc = a.x + gfeat;
  // The input at position p of the sequence; before its start, the stored tail
  // (state[j] is position j - 3).
  float win[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int p = tok_start - 3 + i;
    win[i] = p >= 0 ? ld(xc + (seq_start + p) * a.x_stride) : ld(state + p + 3);
  }
  float x0 = win[0], x1 = win[1], x2 = win[2];

  const int count = min(kTm, seqlen - tok_start);
#pragma unroll
  for (int i = 0; i < kTm; ++i) {
    if (i < count) {
      const int64_t t = seq_start + tok_start + i;
      const float x3 = ld(xc + t * a.x_stride);
      a.out[t * a.out_stride + gfeat] =
          conv_step(bias, w, x0, x1, x2, x3, a.silu != 0);
      x0 = x1;
      x1 = x2;
      x2 = x3;
    }
  }

  if (a.boundary != nullptr) {
    int row0 = 0;
    for (int s = 0; s < seq; ++s)
      row0 += (a.cu[s + 1] - a.cu[s]) / a.boundary_tokens;
    for (int bt = (tok_start / a.boundary_tokens + 1) * a.boundary_tokens;
         bt <= tok_start + count; bt += a.boundary_tokens) {
      uint16_t *dst =
          a.boundary +
          ((static_cast<int64_t>(row0) + bt / a.boundary_tokens - 1) * a.channels + gfeat) * 3;
      for (int i = 0; i < 3; ++i)
        dst[i] = xc[(seq_start + bt - 3 + i) * a.x_stride];
    }
  }

  // Exactly one chunk stores the new tail of each sequence; no other chunk reads
  // the stored one (only positions before the start read it).
  if (chunk == 0) {
    uint16_t tail[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int p = seqlen - 3 + j;
      tail[j] = p >= 0 ? xc[(seq_start + p) * a.x_stride] : state[p + 3];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j)
      state[j] = tail[j];
  }
}
// ---- end of AITER ----

// ---- llama.cpp ssm_conv structure, adapted (see the header) ----
__global__ __launch_bounds__(kNt) void conv1d_update_kernel(ConvArgs a) {
  const int c = static_cast<int>(blockIdx.x) * kNt + threadIdx.x;
  const int seq = blockIdx.y;
  if (c >= a.channels)
    return;
  uint16_t *state = a.state + (static_cast<int64_t>(a.slots[seq]) * a.channels + c) * 3;
  float w[kKw];
#pragma unroll
  for (int i = 0; i < kKw; ++i)
    w[i] = ld(a.w + static_cast<int64_t>(c) * kKw + i);
  const float bias = a.bias != nullptr ? ld(a.bias + c) : 0.f;
  const uint16_t s0 = state[0], s1 = state[1], s2 = state[2];
  const uint16_t xin = a.x[seq * a.x_stride + c];
  a.out[seq * a.out_stride + c] =
      conv_step(bias, w, bf16_to_f32(Bf16{s0}), bf16_to_f32(Bf16{s1}),
                bf16_to_f32(Bf16{s2}), bf16_to_f32(Bf16{xin}), a.silu != 0);
  state[0] = s1;
  state[1] = s2;
  state[2] = xin;
}
// ---- end of llama.cpp ----

std::string describe(const turbine_conv1d_desc *d) {
  return "channels=" + std::to_string(d->channels) +
         " kernel=" + std::to_string(d->kernel) +
         " dtype=" + std::to_string(d->dtype);
}

int32_t run(turbine_ctx *ctx, const turbine_conv1d_desc *d, bool prefill) {
  const char *what = prefill ? "turbine_conv1d_prefill" : "turbine_conv1d_update";
  const char *impl = prefill ? kPrefillImpl : kUpdateImpl;
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(what) + ": descriptor is NULL");
  if (!conv1d_supports(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + " (" + impl + "): unsupported configuration " +
                    describe(d));
  }
  if (d->activation != 0 && d->activation != 1) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + ": activation " + std::to_string(d->activation));
  }
  if (d->num_tokens <= 0 || d->num_seqs <= 0)
    return TURBINE_OK;
  if (d->x == nullptr || d->w == nullptr || d->conv_state == nullptr ||
      d->state_slots == nullptr || d->out == nullptr ||
      (prefill && d->cu_seqlens == nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(what) + ": NULL pointer");
  }
  if (d->boundary_conv_states != nullptr &&
      (!prefill || d->boundary_tokens < kKw - 1)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) +
                    ": boundary_conv_states needs a prefill and boundary_tokens >= 3");
  }
  if (d->x_stride < d->channels || d->out_stride < d->channels) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": a token row stride is below the channel count");
  }
  if (!prefill && d->num_tokens != d->num_seqs) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": update takes one token per sequence");
  }
  if (d->num_seqs > 65535) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + ": more than 65535 sequences in one call");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  ConvArgs a{static_cast<const uint16_t *>(d->x),
             static_cast<const uint16_t *>(d->w),
             static_cast<const uint16_t *>(d->bias),
             static_cast<uint16_t *>(d->conv_state),
             d->state_slots,
             prefill ? d->cu_seqlens : nullptr,
             static_cast<uint16_t *>(d->out),
             static_cast<uint16_t *>(d->boundary_conv_states),
             d->x_stride,
             d->out_stride,
             d->boundary_tokens,
             d->num_seqs,
             d->channels,
             d->activation};
  const uint32_t feature_blocks = static_cast<uint32_t>((d->channels + kNt - 1) / kNt);
  if (prefill) {
    // At most ceil(tokens / kTm) + sequences chunks; surplus blocks return at once.
    const int64_t programs = (static_cast<int64_t>(d->num_tokens) + kTm - 1) / kTm + d->num_seqs;
    hipLaunchKernelGGL(conv1d_prefill_kernel, dim3(static_cast<uint32_t>(programs), feature_blocks),
                       dim3(kNt), 0, ctx->stream, a);
  } else {
    hipLaunchKernelGGL(conv1d_update_kernel,
                       dim3(feature_blocks, static_cast<uint32_t>(d->num_seqs)),
                       dim3(kNt), 0, ctx->stream, a);
  }
  return check_hip(ctx, hipGetLastError(), what);
}

} // namespace

bool conv1d_supports(const turbine_conv1d_desc *d) {
  return d != nullptr && d->dtype == TURBINE_DTYPE_BF16 && d->kernel == kKw &&
         d->channels > 0;
}

int32_t conv1d_prefill_run(turbine_ctx *ctx, const turbine_conv1d_desc *d) {
  return run(ctx, d, true);
}

int32_t conv1d_update_run(turbine_ctx *ctx, const turbine_conv1d_desc *d) {
  return run(ctx, d, false);
}

} // namespace turbine_hip

extern "C" {

int32_t turbine_conv1d_prefill_supported(const turbine_conv1d_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_CONV1D_PREFILL, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_conv1d_prefill_impl(const turbine_conv1d_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_CONV1D_PREFILL, d);
}

int32_t turbine_conv1d_prefill(turbine_ctx *ctx, const turbine_conv1d_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_CONV1D_PREFILL, d);
}

int32_t turbine_conv1d_update_supported(const turbine_conv1d_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_CONV1D_UPDATE, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_conv1d_update_impl(const turbine_conv1d_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_CONV1D_UPDATE, d);
}

int32_t turbine_conv1d_update(turbine_ctx *ctx, const turbine_conv1d_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_CONV1D_UPDATE, d);
}

} // extern "C"
