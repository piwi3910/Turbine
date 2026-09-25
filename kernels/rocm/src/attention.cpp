// Prefill and decode attention through Composable Kernel ck_tile FMHA forward.
//
// One sequence runs as a group-mode batch of one: seqstart_q = {0, q_len},
// seqstart_k = {0, q_start + q_len}. Causal attention uses the bottom-right
// aligned mask (window left -1, right 0), so query i (absolute position
// q_start + i) attends keys 0..=q_start + i. GQA is native: query head h reads
// KV head h / (num_q_heads / num_kv_heads).
#include <string>

#include "fmha_fwd.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "ck_tile_fmha_fwd";
constexpr int32_t kHeadDim = 128;

// Writes seqstart_q {0, q_len} and seqstart_k {0, kv_len} into the context's
// device scratch in stream order, so no host buffer outlives the call.
__global__ void write_seqstart(int32_t *seqstart, int32_t q_len,
                               int32_t kv_len) {
  seqstart[0] = 0;
  seqstart[1] = q_len;
  seqstart[2] = 0;
  seqstart[3] = kv_len;
}

bool supported(const turbine_attention_desc *d) {
  if (d == nullptr)
    return false;
  if (d->dtype != TURBINE_DTYPE_BF16 || d->head_dim != kHeadDim)
    return false;
  if (d->causal != 0 && d->causal != 1)
    return false;
  if (d->q_len < 1 || d->q_start < 0)
    return false;
  if (d->num_q_heads < 1 || d->num_kv_heads < 1)
    return false;
  if (d->num_q_heads % d->num_kv_heads != 0)
    return false;
  // ck_tile index_t is 32-bit: every stride and the key count must fit.
  const int64_t kv_len = static_cast<int64_t>(d->q_start) + d->q_len;
  if (kv_len > INT32_MAX)
    return false;
  if (d->q_stride_token < static_cast<int64_t>(d->num_q_heads) * kHeadDim ||
      d->out_stride_token < static_cast<int64_t>(d->num_q_heads) * kHeadDim ||
      d->kv_stride_token < static_cast<int64_t>(d->num_kv_heads) * kHeadDim) {
    return false;
  }
  return d->q_stride_token <= INT32_MAX && d->out_stride_token <= INT32_MAX &&
         d->kv_stride_token <= INT32_MAX;
}

std::string describe(const turbine_attention_desc *d) {
  return "q_len=" + std::to_string(d->q_len) +
         " q_start=" + std::to_string(d->q_start) +
         " heads=" + std::to_string(d->num_q_heads) + "/" +
         std::to_string(d->num_kv_heads) +
         " head_dim=" + std::to_string(d->head_dim) +
         " dtype=" + std::to_string(d->dtype) +
         " causal=" + std::to_string(d->causal);
}

int32_t run(turbine_ctx *ctx, const turbine_attention_desc *d,
            const char *entry) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(entry) + ": descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(entry) + ": unsupported configuration " +
                    describe(d));
  }
  if (d->q == nullptr || d->k_cache == nullptr || d->v_cache == nullptr ||
      d->out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(entry) + ": NULL operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;

  const int32_t kv_len = d->q_start + d->q_len;
  hipLaunchKernelGGL(write_seqstart, dim3(1), dim3(1), 0, ctx->stream,
                     ctx->seqstart, d->q_len, kv_len);
  if (int32_t rc = check_hip(ctx, hipGetLastError(), "write_seqstart launch");
      rc != TURBINE_OK) {
    return rc;
  }

  fmha_fwd_traits traits{};
  traits.hdim_q = kHeadDim;
  traits.hdim_v = kHeadDim;
  traits.data_type = "bf16";
  traits.is_group_mode = true;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type =
      d->causal == 1 ? mask_enum::mask_bottom_right : mask_enum::no_mask;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.has_dropout = false;
  traits.qscale_type = quant_scale_enum::no_scale;
  traits.skip_min_seqlen_q = false;
  traits.has_sink = false;

  fmha_fwd_args args{};
  args.q_ptr = d->q;
  args.k_ptr = d->k_cache;
  args.v_ptr = d->v_cache;
  args.o_ptr = d->out;
  args.seqstart_q_ptr = ctx->seqstart;
  args.seqstart_k_ptr = ctx->seqstart + 2;
  args.seqlen_q = d->q_len;
  args.seqlen_k = kv_len;
  args.batch = 1;
  args.max_seqlen_q = d->q_len;
  args.hdim_q = kHeadDim;
  args.hdim_v = kHeadDim;
  args.nhead_q = d->num_q_heads;
  args.nhead_k = d->num_kv_heads;
  args.scale_s = d->scale;
  args.logits_soft_cap = 0.0f;
  args.stride_q = static_cast<ck_tile::index_t>(d->q_stride_token);
  args.stride_k = static_cast<ck_tile::index_t>(d->kv_stride_token);
  args.stride_v = static_cast<ck_tile::index_t>(d->kv_stride_token);
  args.stride_o = static_cast<ck_tile::index_t>(d->out_stride_token);
  args.nhead_stride_q = kHeadDim;
  args.nhead_stride_k = kHeadDim;
  args.nhead_stride_v = kHeadDim;
  args.nhead_stride_o = kHeadDim;
  if (d->causal == 1) {
    args.window_size_left = -1;
    args.window_size_right = 0;
  } else {
    args.window_size_left = -1;
    args.window_size_right = -1;
  }
  args.sink_size = 0;
  args.mask_type = static_cast<ck_tile::index_t>(traits.mask_type);
  args.min_seqlen_q = 0;
  args.p_drop = 0.0f;
  args.s_randval = false;
  args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

  const ck_tile::stream_config stream{ctx->stream};
  const float result = fmha_fwd(traits, args, stream);
  if (result < 0.0f) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(entry) + ": fmha_fwd has no ck_tile instance for " +
                    describe(d));
  }
  return check_hip(ctx, hipGetLastError(), "fmha_fwd launch");
}

} // namespace

extern "C" {

int32_t turbine_attention_prefill(turbine_ctx *ctx,
                                  const turbine_attention_prefill_desc *d) {
  return run(ctx, d, "turbine_attention_prefill");
}

int32_t
turbine_attention_prefill_supported(const turbine_attention_prefill_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *
turbine_attention_prefill_impl(const turbine_attention_prefill_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_attention_decode(turbine_ctx *ctx,
                                 const turbine_attention_decode_desc *d) {
  return run(ctx, d, "turbine_attention_decode");
}

int32_t
turbine_attention_decode_supported(const turbine_attention_decode_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *
turbine_attention_decode_impl(const turbine_attention_decode_desc *d) {
  (void)d;
  return kImpl;
}

} // extern "C"
