// Paged decode attention on Composable Kernel ck_tile fmha_fwd_splitkv (impl
// "ck_tile_fmha_splitkv" of attention_decode_paged; paged_attention.cpp
// validates the descriptor and appends k_new/v_new first).
//
// Group mode over the same page table as fmha_fwd_pagedkv: seqstart_q =
// q_indptr (one query per sequence), seqlen_k = kv_lens, page stride 2 *
// block_tokens * Hkv * 128 elements, v_ptr = k_ptr + block_tokens * Hkv * 128.
// A single query sees every key, so the causal mask is no mask; with no mask,
// no bias and max_seqlen_q == 1, CK's split-KV kernel merges the query heads
// of one KV head into the rows of one tile (kMergeNumHeadGroupsSeqLenQ): one
// workgroup per (sequence, KV head) reads each K/V page once for the whole
// group, instead of once per query head. The kernel writes F32 outputs and
// their log-sum-exp into the context's split scratch and CK's combine kernel
// rounds them into out.
//
// One split (decision "Pre-Phase-5 #3", provisional): with the head merge one
// workgroup per (sequence, KV head) already fills the R9700 from 16 sequences
// up, and 1 split is the fastest count there at every context (hip_ops
// decode_attention_timings, sweep of 1-32 splits). More splits pay only below
// that (batch 1: 40 -> 24 us at ~768 tokens, 297 -> 72 us at ~8,192) but
// round P to BF16 against each split's own running maximum, which leaves the
// cpu-reference numerics model of CK attention (tiny Llama: 3.8e-6 -> 0.10
// max |logit diff|). With one split a row's result is also independent of the
// batch it decodes in.
//
// The scratch depends on num_seqs and the heads only, so a decode graph's
// capture never needs more than the eager run of the same key before it; it
// grows outside a capture only, and a replaced buffer stays allocated until
// the context is destroyed (a captured graph may still read it).
#include <string>

#include "fmha_fwd.hpp"
#include "paged_fp8.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::fail;

namespace {

constexpr int32_t kHeadDim = 128;
// The split count (see the header comment).
constexpr int32_t kSplits = 1;

} // namespace

namespace turbine_hip {

// Grows ctx's attention scratch (the split-KV accumulators, and the staged FP8
// pages of paged_attention.cpp, which never run under a capture) to at least
// bytes, rounded up to a power of two (never while capturing).
int32_t attention_scratch(turbine_ctx *ctx, size_t bytes, void **out) {
  if (ctx->attn_split_scratch_bytes < bytes) {
    if (ctx->capturing) {
      return turbine_hip::refuse_while_capturing(
          ctx, "turbine_attention_decode_paged (growing the split-KV scratch)");
    }
    size_t grown_bytes = 1;
    while (grown_bytes < bytes)
      grown_bytes <<= 1;
    void *grown = nullptr;
    const std::string what =
        "hipMalloc split-KV scratch " + std::to_string(grown_bytes) + " bytes";
    if (int32_t rc =
            check_hip(ctx, hipMalloc(&grown, grown_bytes), what.c_str());
        rc != TURBINE_OK) {
      return rc;
    }
    if (ctx->attn_split_scratch != nullptr)
      ctx->attn_split_retired.push_back(ctx->attn_split_scratch);
    ctx->attn_split_scratch = grown;
    ctx->attn_split_scratch_bytes = grown_bytes;
  }
  *out = ctx->attn_split_scratch;
  return TURBINE_OK;
}

} // namespace turbine_hip

namespace {

int32_t split_scratch(turbine_ctx *ctx, size_t bytes, void **out) {
  return turbine_hip::attention_scratch(ctx, bytes, out);
}

} // namespace

namespace turbine_hip {

bool ck_splitkv_serves(const turbine_attention_paged_desc *d) {
  // Decode (one query per sequence) of grouped query heads: the head merge
  // is what this path adds over fmha_fwd_pagedkv, and it engages only when
  // there are more query heads than KV heads.
  return d->max_q_len <= 1 && d->total_q <= d->num_seqs &&
         d->num_q_heads > d->num_kv_heads;
}

int32_t run_ck_splitkv(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                       const std::string &entry) {
  const int32_t splits = kSplits;
  const int64_t rows = static_cast<int64_t>(d->total_q);
  const int64_t lse_elems =
      static_cast<int64_t>(d->num_q_heads) * splits * rows;
  const int64_t o_elems = lse_elems * kHeadDim;
  void *scratch = nullptr;
  if (int32_t rc = split_scratch(
          ctx, static_cast<size_t>(lse_elems + o_elems) * sizeof(float),
          &scratch);
      rc != TURBINE_OK) {
    return rc;
  }
  float *o_acc = static_cast<float *>(scratch);
  float *lse_acc = o_acc + o_elems;

  fmha_fwd_splitkv_traits traits{};
  traits.hdim_q = kHeadDim;
  traits.hdim_v = kHeadDim;
  traits.data_type = "bf16";
  traits.is_group_mode = true;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask_enum::no_mask;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.do_fp8_static_quant = false;
  traits.has_sink = false;

  const int64_t token_elems = static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
  const auto page_stride =
      static_cast<ck_tile::index_t>(2 * d->block_tokens * token_elems);
  const auto *pool = static_cast<const uint16_t *>(d->kv_layer);

  fmha_fwd_splitkv_args args{};
  args.q_ptr = d->q;
  args.k_ptr = pool;
  args.v_ptr = pool + static_cast<int64_t>(d->block_tokens) * token_elems;
  args.bias_ptr = nullptr;
  args.lse_acc_ptr = lse_acc;
  args.o_acc_ptr = o_acc;
  args.lse_ptr = nullptr;
  args.o_ptr = d->out;
  args.block_table_ptr = const_cast<int32_t *>(d->block_table);
  args.batch_stride_block_table = d->max_blocks_per_seq;
  args.page_block_size = d->block_tokens;
  args.is_gappy = false;
  args.cache_batch_idx = nullptr;
  // As for fmha_fwd_pagedkv: seqstart_k is read only for an offset the paged
  // path never uses; q_indptr is a valid [num_seqs + 1] array for it.
  args.seqstart_q_ptr = d->q_indptr;
  args.seqstart_k_ptr = d->q_indptr;
  args.seqlen_k_ptr = d->kv_lens;
  args.sink_ptr = nullptr;
  args.seqlen_q = 1;
  args.seqlen_k = d->max_kv_len;
  args.batch = d->num_seqs;
  args.max_seqlen_q = 1;
  args.hdim_q = kHeadDim;
  args.hdim_v = kHeadDim;
  args.nhead_q = d->num_q_heads;
  args.nhead_k = d->num_kv_heads;
  args.num_splits = splits;
  args.scale_s = d->scale;
  args.scale_p = 1.0f;
  args.scale_o = 1.0f;
  args.logits_soft_cap = 0.0f;
  args.stride_q = static_cast<ck_tile::index_t>(d->q_stride_token);
  args.stride_k = static_cast<ck_tile::index_t>(token_elems);
  args.stride_v = static_cast<ck_tile::index_t>(token_elems);
  args.stride_bias = 0;
  args.stride_o_acc = kHeadDim;
  args.stride_o = static_cast<ck_tile::index_t>(d->out_stride_token);
  args.nhead_stride_q = kHeadDim;
  args.nhead_stride_k = kHeadDim;
  args.nhead_stride_v = kHeadDim;
  args.nhead_stride_bias = 0;
  args.nhead_stride_lse = 0;
  // Accumulators [nhead_q, splits, total_q, (128)].
  args.nhead_stride_lse_acc = static_cast<ck_tile::index_t>(splits * rows);
  args.nhead_stride_o_acc =
      static_cast<ck_tile::index_t>(splits * rows * kHeadDim);
  args.nhead_stride_o = kHeadDim;
  args.batch_stride_q = 0;
  args.batch_stride_k = page_stride;
  args.batch_stride_v = page_stride;
  args.batch_stride_bias = 0;
  args.batch_stride_lse = 0;
  args.batch_stride_lse_acc = 0;
  args.batch_stride_o_acc = 0;
  args.batch_stride_o = 0;
  args.split_stride_lse_acc = static_cast<ck_tile::index_t>(rows);
  args.split_stride_o_acc = static_cast<ck_tile::index_t>(rows * kHeadDim);
  args.window_size_left = -1;
  args.window_size_right = -1;
  args.sink_size = 0;
  args.mask_type = static_cast<ck_tile::index_t>(mask_enum::no_mask);

  const ck_tile::stream_config stream{ctx->stream};
  const float result = fmha_fwd_splitkv(traits, args, stream);
  if (result < 0.0f) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                entry + ": fmha_fwd_splitkv has no ck_tile instance for seqs=" +
                    std::to_string(d->num_seqs) +
                    " heads=" + std::to_string(d->num_q_heads) + "/" +
                    std::to_string(d->num_kv_heads) +
                    " block_tokens=" + std::to_string(d->block_tokens) +
                    " splits=" + std::to_string(splits));
  }
  return check_hip(ctx, hipGetLastError(), "fmha_fwd_splitkv launch");
}

} // namespace turbine_hip
