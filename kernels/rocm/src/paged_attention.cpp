// Paged attention over one layer of the KV block pool (ABI v2):
// attention_prefill_paged and attention_decode_paged.
//
// Every call first appends k_new/v_new into their page slots with the Turbine
// append kernel (paged_attention.hip), then attends with one of the
// implementations (impl_table.cpp; the caller picks one with
// turbine_impl_run, or the entry points take the first that supports the
// descriptor, CK only for pages of a multiple of the context's card-profile
// paged_page_multiple):
// - Composable Kernel ck_tile fmha_fwd_pagedkv in group mode (impl
//   "ck_tile_fmha_pagedkv") -- seqstart_q = q_indptr, seqlen_k = kv_lens, the
//   block table as CK's page table with page stride 2 * block_tokens * Hkv *
//   128 elements, v_ptr = k_ptr + block_tokens * Hkv * 128, bottom-right causal
//   mask (query i of a sequence with q_len queries and kv_len keys sees keys
//   0..=kv_len - q_len + i). The pagedkv instances of the pinned CK commit
//   serve only page sizes that are a multiple of kCkPagedkvPage, which its
//   supports reports;
// - Composable Kernel ck_tile fmha_fwd_splitkv in group mode (impl
//   "ck_tile_fmha_splitkv", decode only; paged_attention_splitkv.cpp): the
//   same page table, one query per sequence, the query heads of a KV head
//   merged into one tile (grouped query heads only), one split, rounded by
//   CK's split-KV combine kernel;
// - the Turbine b1 decode kernel (impl "turbine_hip_b1_decode", decode only;
//   paged_attention_b1.hip): grouped query heads (group <= 8), any batch,
//   BF16 flat pages; grid (sequences, KV heads, splits) fills the
//   card at batch 1, the append and the split combine are fused into the same
//   launch, and P stays F32 (a numerics class of its own, not bitwise CK);
// - the Turbine HIP paged kernel (impl "turbine_hip"), any page size.
// Decode is the same computation with one query per sequence and runs the same
// paths.
//
// FP8 pages (dtype TURBINE_DTYPE_F8E4M3, kv.dtype fp8_e4m3, Phase 6a S-13):
// the append writes e4m3(x / scale) with the descriptor's k_scale / v_scale
// (read at ABI minor >= 9; a caller only sends dtype F8E4M3 from there), and
// attention reads bf16(e4m3 * scale). The BF16 implementations refuse FP8
// pages; the FP8 ones (decision "P6: FP8 paged attention -- provider
// evaluation") are:
// - "ck_tile_fmha_pagedkv_fp8_staged" (prefill, pages of a multiple of 128):
//   the batch's pages dequantized into a BF16 staging pool in the context's
//   attention scratch, then CK fmha_fwd_pagedkv over it with a staged block
//   table, in groups of sequences whose staged pages fit kStagedMaxBytes; a
//   call where one sequence alone does not fit runs the Turbine FP8 kernel.
//   Sequences with a single query row (decode rows riding along a prefill)
//   are not staged: the FP8 decode kernel computes them after CK, as it does
//   in a decode call, so a decode row's result does not depend on the batch;
// - "turbine_hip_fp8_decode" (decode, any page size): paged_attention.hip's
//   FP8 decode kernel, the query heads of a KV head together, pages read as
//   bytes;
// - "turbine_hip_fp8" (any page size): the Turbine kernel reading FP8 pages.
//
// Mixed-format pages (ABI v2.11, Phase 6b S-5): a descriptor whose dtype is
// TURBINE_DTYPE_TQ4 / _TQ2 or whose block_formats is not NULL. Only the
// mixed implementations take it (the ones above refuse it), and they append
// each row in its block's format (paged_attention_mixed.hip). Decision "P6b:
// mixed-format / TurboQuant paged attention -- provider evaluation":
// - "turbine_hip_mixed" (decode, and prefill of any page size): the own
//   kernel, TurboQuant blocks read in the rotated domain without materialising
//   K or V;
// - "ck_tile_fmha_pagedkv_mixed_staged" (prefill, pages of a multiple of 128;
//   user decision "6b Task 10: TurboQuant prefill attention", A): the batch's
//   blocks decoded by their formats into a BF16 staging pool, then CK
//   fmha_fwd_pagedkv over it, as the FP8 staged prefill; decode rows riding
//   along run turbine_hip_mixed after, so their result does not depend on
//   the batch; a sequence alone over kStagedMaxBytes runs turbine_hip_mixed;
// - "turbine_hip_mixed_staged" (prefill, any page size): the same staging with
//   the Turbine BF16 kernel attending.
//
// Sliding window and sink logits (ABI v2.17, Phase 7 S-8) and head_dim 64 /
// 256 (decision "P7 attention sinks/window instances"): only the CK paths over
// BF16 flat pages of a 128 multiple take them -- fmha_fwd_pagedkv (d128 only at
// the pinned CK commit; its masked and _sink instances) and fmha_fwd_splitkv
// (d64 / d128 / d256, masked and _sink instances; for d64 and d256 it serves
// prefill as well as decode, as the pinned commit has no gfx12 pagedkv
// instance of those head dims). The window is CK's bottom-right local mask
// (window_size_left = window_left, window_size_right = 0); the sinks are CK's
// sink_ptr, the running max and a sum of 1 before the first key. The Turbine
// kernels (b1, FP8, mixed, the 16-token kernel) stay head_dim 128 without
// window or sinks and refuse the rest.
//
// The device arrays (block_table, q_indptr, kv_lens) are trusted: the host
// cannot read them without a synchronisation. The Turbine kernels skip pages
// outside the pool instead of faulting.
#include <string>

#include <algorithm>
#include <cmath>

#include "fmha_fwd.hpp"
#include "paged_fp8.hpp"
#include "paged_mixed.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImplCk = "ck_tile_fmha_pagedkv";
constexpr const char *kImplCkSplitkv = "ck_tile_fmha_splitkv";
constexpr const char *kImplTurbine = "turbine_hip";
constexpr const char *kImplTurbineB1 = "turbine_hip_b1_decode";
constexpr const char *kImplCkFp8Staged = "ck_tile_fmha_pagedkv_fp8_staged";
constexpr const char *kImplTurbineFp8Decode = "turbine_hip_fp8_decode";
constexpr const char *kImplTurbineFp8 = "turbine_hip_fp8";
constexpr const char *kImplCkMixedStaged = "ck_tile_fmha_pagedkv_mixed_staged";
constexpr const char *kImplTurbineMixedStaged = "turbine_hip_mixed_staged";
constexpr const char *kImplTurbineMixed = "turbine_hip_mixed";
// Page sizes the mixed-format kernels take are multiples of this.
constexpr int32_t kMixedPageMultiple = 16;
// Bytes of BF16 staged pages (plus their table) one CK call of the staged FP8
// prefill may use: sequences are staged in groups that fit.
constexpr int64_t kStagedMaxBytes = int64_t{256} << 20;
// The head dim of the Turbine kernels and of the CK pagedkv instances.
constexpr int32_t kHeadDim = 128;
// Page sizes the CK pagedkv instances serve are multiples of this: a
// property of the instances (a capability supports reports), not a tuning
// threshold.
constexpr int32_t kCkPagedkvPage = 128;
// Grid y (sequences) and z (KV heads) limits of the Turbine kernel.
constexpr int32_t kMaxGridYZ = 65535;

// A descriptor only the mixed-format implementations take: TurboQuant pages,
// a block_formats table, or per-class page addressing (ABI v2.11; the classed
// resolution lives in the mixed kernels).
bool mixed(const turbine_attention_paged_desc *d) {
  return d->dtype == TURBINE_DTYPE_TQ4 || d->dtype == TURBINE_DTYPE_TQ2 ||
         d->block_formats != nullptr || d->page_classes != nullptr;
}

// Head dim 128 without a window or sinks: what every path takes. The rest
// (ABI v2.17) only the CK paths over BF16 flat pages of a 128 multiple.
bool plain(const turbine_attention_paged_desc *d) {
  return d->head_dim == kHeadDim && d->window_left < 0 && d->sinks == nullptr;
}

bool supported(const turbine_attention_paged_desc *d) {
  if (d == nullptr)
    return false;
  if ((d->dtype != TURBINE_DTYPE_BF16 && d->dtype != TURBINE_DTYPE_F8E4M3 &&
       d->dtype != TURBINE_DTYPE_TQ4 && d->dtype != TURBINE_DTYPE_TQ2) ||
      (d->head_dim != 64 && d->head_dim != kHeadDim && d->head_dim != 256)) {
    return false;
  }
  if (d->window_left < -1 || (d->window_left >= 0 && d->causal != 1))
    return false;
  if (!plain(d) && (d->dtype != TURBINE_DTYPE_BF16 || mixed(d) ||
                    d->block_tokens % kCkPagedkvPage != 0)) {
    return false;
  }
  if (mixed(d) &&
      (d->block_tokens % kMixedPageMultiple != 0 || d->num_q_heads < 1 ||
       d->num_kv_heads < 1 ||
       d->num_q_heads / d->num_kv_heads > turbine_hip::kPagedMixedMaxGroup)) {
    return false;
  }
  if (d->causal != 0 && d->causal != 1)
    return false;
  if (d->num_q_heads < 1 || d->num_kv_heads < 1 ||
      d->num_kv_heads > kMaxGridYZ) {
    return false;
  }
  if (d->num_q_heads % d->num_kv_heads != 0 ||
      d->num_q_heads / d->num_kv_heads > turbine_hip::kPagedMaxGroup) {
    return false;
  }
  if (d->block_tokens < 1 || d->num_blocks < 0 || d->max_blocks_per_seq < 0)
    return false;
  if (d->num_seqs < 0 || d->num_seqs > kMaxGridYZ || d->total_q < 0 ||
      d->max_q_len < 0 || d->max_kv_len < 0) {
    return false;
  }
  // v2.12 fused RoPE: half-split over an even rotary width within the head.
  if (d->rope_inv_freq != nullptr &&
      (d->rope_rotary_dim < 2 || d->rope_rotary_dim % 2 != 0 ||
       d->rope_rotary_dim > d->head_dim)) {
    return false;
  }
  return d->q_stride_token >=
             static_cast<int64_t>(d->num_q_heads) * d->head_dim &&
         d->out_stride_token >=
             static_cast<int64_t>(d->num_q_heads) * d->head_dim &&
         d->new_stride_token >=
             static_cast<int64_t>(d->num_kv_heads) * d->head_dim;
}

// What the CK instances can serve: a page size they are built for and every
// stride within ck_tile's 32-bit index_t.
bool ck_serves(const turbine_attention_paged_desc *d) {
  if (d->block_tokens % kCkPagedkvPage != 0)
    return false;
  const int64_t page_stride =
      2 * static_cast<int64_t>(d->block_tokens) * d->num_kv_heads * d->head_dim;
  return page_stride <= INT32_MAX && d->q_stride_token <= INT32_MAX &&
         d->out_stride_token <= INT32_MAX;
}

std::string describe(const turbine_attention_paged_desc *d) {
  return "seqs=" + std::to_string(d->num_seqs) +
         " total_q=" + std::to_string(d->total_q) +
         " max_q_len=" + std::to_string(d->max_q_len) +
         " max_kv_len=" + std::to_string(d->max_kv_len) +
         " block_tokens=" + std::to_string(d->block_tokens) +
         " heads=" + std::to_string(d->num_q_heads) + "/" +
         std::to_string(d->num_kv_heads) +
         " head_dim=" + std::to_string(d->head_dim) +
         " dtype=" + std::to_string(d->dtype) +
         " causal=" + std::to_string(d->causal) +
         " window_left=" + std::to_string(d->window_left) +
         " sinks=" + (d->sinks != nullptr ? "1" : "0");
}

// The pages CK reads: the pool (BF16) or a staged copy of FP8 pages, with its
// block table, for sequences [g0, g0 + count).
struct CkPages {
  const uint16_t *pool;
  const int32_t *table;
  int32_t table_stride;
  // The group's key counts ([count], group-relative).
  const int32_t *kv_lens;
  int32_t g0;
  int32_t count;
};

int32_t run_ck_pages(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                     const std::string &entry, const CkPages &pages);

int32_t run_ck(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
               const std::string &entry) {
  return run_ck_pages(ctx, d, entry,
                      CkPages{static_cast<const uint16_t *>(d->kv_layer),
                              d->block_table, d->max_blocks_per_seq, d->kv_lens,
                              0, d->num_seqs});
}

int32_t run_ck_pages(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                     const std::string &entry, const CkPages &pages) {
  fmha_fwd_pagedkv_traits traits{};
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
  traits.use_pagedkv = true;
  traits.do_fp8_static_quant = false;
  traits.skip_min_seqlen_q = false;
  traits.has_sink = d->sinks != nullptr;

  const int64_t token_elems = static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
  const auto page_stride =
      static_cast<ck_tile::index_t>(2 * d->block_tokens * token_elems);
  // BF16 elements, addressed as their 16-bit storage.
  const uint16_t *pool = pages.pool;

  fmha_fwd_pagedkv_args args{};
  args.q_ptr = d->q;
  args.k_ptr = pool;
  args.v_ptr = pool + static_cast<int64_t>(d->block_tokens) * token_elems;
  args.bias_ptr = nullptr;
  args.lse_ptr = nullptr;
  args.o_ptr = d->out;
  args.block_table_ptr = const_cast<int32_t *>(pages.table);
  args.batch_stride_block_table = pages.table_stride;
  args.page_block_size = d->block_tokens;
  args.is_gappy = false;
  args.cache_batch_idx = nullptr;
  // Group mode with a page table: query rows from seqstart_q, key counts from
  // seqlen_k. The kernel reads seqstart_k[b] but, with pages, only for an
  // offset it never uses; q_indptr is a valid [num_seqs + 1] array for it.
  // A group [g0, g0 + count) reads its slice of the arrays: q_indptr holds
  // absolute rows, so the group's queries stay where they are.
  args.seqstart_q_ptr = d->q_indptr + pages.g0;
  args.seqstart_k_ptr = d->q_indptr + pages.g0;
  args.seqlen_k_ptr = pages.kv_lens;
  args.sink_ptr = d->sinks;
  args.seqlen_q = d->max_q_len;
  args.seqlen_k = d->max_kv_len;
  args.batch = pages.count;
  args.max_seqlen_q = d->max_q_len;
  args.hdim_q = kHeadDim;
  args.hdim_v = kHeadDim;
  args.nhead_q = d->num_q_heads;
  args.nhead_k = d->num_kv_heads;
  args.scale_s = d->scale;
  args.scale_p = 1.0f;
  args.scale_o = 1.0f;
  args.logits_soft_cap = 0.0f;
  args.stride_q = static_cast<ck_tile::index_t>(d->q_stride_token);
  args.stride_k = static_cast<ck_tile::index_t>(token_elems);
  args.stride_v = static_cast<ck_tile::index_t>(token_elems);
  args.stride_bias = 0;
  args.stride_o = static_cast<ck_tile::index_t>(d->out_stride_token);
  args.nhead_stride_q = kHeadDim;
  args.nhead_stride_k = kHeadDim;
  args.nhead_stride_v = kHeadDim;
  args.nhead_stride_bias = 0;
  args.nhead_stride_lse = 0;
  args.nhead_stride_o = kHeadDim;
  args.batch_stride_q = 0;
  args.batch_stride_k = page_stride;
  args.batch_stride_v = page_stride;
  args.batch_stride_bias = 0;
  args.batch_stride_lse = 0;
  args.batch_stride_o = 0;
  // The bottom-right causal mask, local with a window (ABI v2.17).
  args.window_size_left = d->window_left;
  args.window_size_right = d->causal == 1 ? 0 : -1;
  // CK's sink_size is StreamingLLM's kept-first-tokens, not the sink logits.
  args.sink_size = 0;
  args.mask_type = static_cast<ck_tile::index_t>(traits.mask_type);
  args.min_seqlen_q = 0;

  const ck_tile::stream_config stream{ctx->stream};
  const float result = fmha_fwd_pagedkv(traits, args, stream);
  if (result < 0.0f) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                entry + ": fmha_fwd_pagedkv has no ck_tile instance for " +
                    describe(d));
  }
  return check_hip(ctx, hipGetLastError(), "fmha_fwd_pagedkv launch");
}

// What path serves beyond the common checks of supported().
bool path_serves(const turbine_attention_paged_desc *d,
                 turbine_hip::PagedPath path) {
  const bool fp8 = d->dtype == TURBINE_DTYPE_F8E4M3;
  // Head dims other than 128, a window and sinks: the CK paths only.
  if (!plain(d) && path != turbine_hip::PagedPath::CkPagedkv &&
      path != turbine_hip::PagedPath::CkSplitkv) {
    return false;
  }
  if (mixed(d)) {
    switch (path) {
    case turbine_hip::PagedPath::CkPagedkvMixedStaged:
      return ck_serves(d);
    case turbine_hip::PagedPath::TurbineMixedStaged:
    case turbine_hip::PagedPath::TurbineMixed:
      return true;
    default:
      return false;
    }
  }
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkvMixedStaged:
  case turbine_hip::PagedPath::TurbineMixedStaged:
  case turbine_hip::PagedPath::TurbineMixed:
    return false;
  case turbine_hip::PagedPath::CkPagedkv:
    // The pinned commit's gfx12 pagedkv instances are head_dim 128 only.
    return !fp8 && d->head_dim == kHeadDim && ck_serves(d);
  case turbine_hip::PagedPath::CkSplitkv:
    return !fp8 && ck_serves(d) && turbine_hip::ck_splitkv_serves(d);
  case turbine_hip::PagedPath::TurbineB1:
    return !fp8 && turbine_hip::b1_serves(d);
  case turbine_hip::PagedPath::Turbine:
    return !fp8;
  case turbine_hip::PagedPath::CkPagedkvFp8Staged:
    // Its single-query rows run the FP8 decode kernel, whose LDS holds at most
    // kPagedFp8DecodeMaxGroup query heads per KV head: larger groups go to
    // TurbineFp8 (review r13 P1).
    return fp8 && ck_serves(d) &&
           d->num_q_heads / d->num_kv_heads <=
               turbine_hip::kPagedFp8DecodeMaxGroup;
  case turbine_hip::PagedPath::TurbineFp8Decode:
    return fp8 && d->max_q_len <= 1 && d->total_q <= d->num_seqs &&
           d->num_q_heads / d->num_kv_heads <=
               turbine_hip::kPagedFp8DecodeMaxGroup;
  case turbine_hip::PagedPath::TurbineFp8:
    return fp8;
  }
  return false;
}

const char *path_name(turbine_hip::PagedPath path) {
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkv:
    return kImplCk;
  case turbine_hip::PagedPath::CkSplitkv:
    return kImplCkSplitkv;
  case turbine_hip::PagedPath::TurbineB1:
    return kImplTurbineB1;
  case turbine_hip::PagedPath::CkPagedkvFp8Staged:
    return kImplCkFp8Staged;
  case turbine_hip::PagedPath::TurbineFp8Decode:
    return kImplTurbineFp8Decode;
  case turbine_hip::PagedPath::TurbineFp8:
    return kImplTurbineFp8;
  case turbine_hip::PagedPath::CkPagedkvMixedStaged:
    return kImplCkMixedStaged;
  case turbine_hip::PagedPath::TurbineMixedStaged:
    return kImplTurbineMixedStaged;
  case turbine_hip::PagedPath::TurbineMixed:
    return kImplTurbineMixed;
  case turbine_hip::PagedPath::Turbine:
    break;
  }
  return kImplTurbine;
}

// The staged FP8 prefill: groups of sequences whose pages, dequantized to BF16,
// fit kStagedMaxBytes, each staged into the attention scratch and run through
// CK; the Turbine FP8 kernel when one sequence alone does not fit.
int32_t run_ck_fp8_staged(turbine_ctx *ctx,
                          const turbine_attention_paged_desc *d,
                          const std::string &entry) {
  const int64_t token_elems = static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
  const int64_t block_bytes =
      2 * static_cast<int64_t>(d->block_tokens) * token_elems * 2;
  const int64_t pages =
      (static_cast<int64_t>(d->max_kv_len) + d->block_tokens - 1) /
      d->block_tokens;
  if (pages < 1)
    return TURBINE_OK;
  const int64_t per_seq = pages * (block_bytes + 4) + 4;
  if (per_seq > kStagedMaxBytes || pages > d->max_blocks_per_seq)
    return turbine_hip::launch_paged_attention(ctx, d);
  const int64_t group =
      std::min<int64_t>(kStagedMaxBytes / per_seq, d->num_seqs);
  const int64_t pool_bytes = group * pages * block_bytes;
  void *scratch = nullptr;
  if (int32_t rc = turbine_hip::attention_scratch(
          ctx, static_cast<size_t>(pool_bytes + group * (pages + 1) * 4),
          &scratch);
      rc != TURBINE_OK) {
    return rc;
  }
  auto *staged = static_cast<uint16_t *>(scratch);
  auto *table =
      reinterpret_cast<int32_t *>(static_cast<char *>(scratch) + pool_bytes);
  int32_t *staged_kv_lens = table + group * pages;
  for (int64_t g0 = 0; g0 < d->num_seqs; g0 += group) {
    const auto count =
        static_cast<int32_t>(std::min<int64_t>(group, d->num_seqs - g0));
    if (int32_t rc = turbine_hip::launch_paged_stage_fp8(
            ctx, d, staged, table, staged_kv_lens, static_cast<int32_t>(g0),
            count, static_cast<int32_t>(pages));
        rc != TURBINE_OK) {
      return rc;
    }
    if (int32_t rc = run_ck_pages(
            ctx, d, entry,
            CkPages{staged, table, static_cast<int32_t>(pages), staged_kv_lens,
                    static_cast<int32_t>(g0), count});
        rc != TURBINE_OK) {
      return rc;
    }
  }
  // The single-query rows CK only saw one unconverted key for (after every
  // group, so they overwrite CK's rows).
  return turbine_hip::launch_paged_decode_fp8(ctx, d);
}

// Whether a batch may hold a sequence of exactly one query row, judged on the
// host from the descriptor alone (q_indptr is device memory): not when every
// sequence must have max_q_len > 1 rows -- a single sequence, or total_q equal
// to num_seqs * max_q_len.
bool may_have_single_query_row(const turbine_attention_paged_desc *d) {
  if (d->max_q_len <= 1)
    return true;
  return !(d->num_seqs == 1 ||
           static_cast<int64_t>(d->total_q) ==
               static_cast<int64_t>(d->num_seqs) * d->max_q_len);
}

// The staged mixed-format prefill: groups of sequences whose pages, decoded to
// BF16, fit kStagedMaxBytes, each staged into the attention scratch and
// attended by CK (use_ck) or the Turbine BF16 kernel; turbine_hip_mixed when
// one sequence alone does not fit, and for the single-query rows after.
int32_t run_mixed_staged(turbine_ctx *ctx,
                         const turbine_attention_paged_desc *d,
                         const std::string &entry, bool use_ck) {
  const int64_t token_elems = static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
  const int64_t block_bytes =
      2 * static_cast<int64_t>(d->block_tokens) * token_elems * 2;
  const int64_t pages =
      (static_cast<int64_t>(d->max_kv_len) + d->block_tokens - 1) /
      d->block_tokens;
  if (pages < 1)
    return TURBINE_OK;
  const int64_t per_seq = pages * (block_bytes + 4) + 4;
  if (per_seq > kStagedMaxBytes || pages > d->max_blocks_per_seq)
    return turbine_hip::run_paged_mixed_attention(ctx, d, false);
  const int64_t group =
      std::min<int64_t>(kStagedMaxBytes / per_seq, d->num_seqs);
  const int64_t pool_bytes = group * pages * block_bytes;
  void *scratch = nullptr;
  if (int32_t rc = turbine_hip::attention_scratch(
          ctx, static_cast<size_t>(pool_bytes + group * (pages + 1) * 4),
          &scratch);
      rc != TURBINE_OK) {
    return rc;
  }
  auto *staged = static_cast<uint16_t *>(scratch);
  auto *table =
      reinterpret_cast<int32_t *>(static_cast<char *>(scratch) + pool_bytes);
  int32_t *staged_kv_lens = table + group * pages;
  for (int64_t g0 = 0; g0 < d->num_seqs; g0 += group) {
    const auto count =
        static_cast<int32_t>(std::min<int64_t>(group, d->num_seqs - g0));
    if (int32_t rc = turbine_hip::launch_paged_stage_mixed(
            ctx, d, staged, table, staged_kv_lens, static_cast<int32_t>(g0),
            count, static_cast<int32_t>(pages));
        rc != TURBINE_OK) {
      return rc;
    }
    int32_t rc = TURBINE_OK;
    if (use_ck) {
      rc = run_ck_pages(ctx, d, entry,
                        CkPages{staged, table, static_cast<int32_t>(pages),
                                staged_kv_lens, static_cast<int32_t>(g0),
                                count});
    } else {
      // The Turbine BF16 kernel over the group's staged pages.
      turbine_attention_paged_desc sd = *d;
      sd.dtype = TURBINE_DTYPE_BF16;
      sd.kv_layer = staged;
      sd.block_table = table;
      sd.q_indptr = d->q_indptr + g0;
      sd.kv_lens = staged_kv_lens;
      sd.num_seqs = count;
      sd.max_blocks_per_seq = static_cast<int32_t>(pages);
      sd.num_blocks = static_cast<int32_t>(count * pages);
      sd.block_formats = nullptr;
      sd.tq_params = nullptr;
      rc = turbine_hip::launch_paged_attention(ctx, &sd);
    }
    if (rc != TURBINE_OK)
      return rc;
  }
  // The single-query rows the BF16 kernel only saw one unconverted key for
  // (after every group, so they overwrite its rows). Skipped when the batch
  // cannot hold one (decision "6b Task 13", 3 C): its grid spans every query
  // row, so on a prefill-only batch it cost ~0.2 ms a layer for nothing.
  if (!may_have_single_query_row(d))
    return TURBINE_OK;
  return turbine_hip::run_paged_mixed_attention(ctx, d, true);
}

// The v2.11 operands of a mixed-format call: the TurboQuant tables (whenever
// a block can be TurboQuant) and the 16-byte alignment the kernels' vector
// loads need.
std::string mixed_operand_error(const turbine_attention_paged_desc *d) {
  const turbine_tq_params *tq = d->tq_params;
  if (tq == nullptr || tq->tables == nullptr)
    return "TurboQuant pages or a block_formats table need tq_params with the "
           "layer's tables";
  for (int bits = 1; bits <= 4; ++bits)
    if (tq->codebooks[bits - 1] == nullptr)
      return "NULL " + std::to_string(bits) + "-bit TurboQuant codebook";
  if (reinterpret_cast<uintptr_t>(tq->tables) % 16 != 0)
    return "TurboQuant tables are not 16-byte aligned";
  if (reinterpret_cast<uintptr_t>(d->kv_layer) % 16 != 0)
    return "kv_layer is not 16-byte aligned";
  return {};
}

int32_t run(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
            const char *entry_name, turbine_hip::PagedPath path) {
  const std::string entry(entry_name);
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, entry + ": descriptor is NULL");
  if (!supported(d) || !path_serves(d, path)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                entry + ": unsupported configuration " + describe(d) +
                    (supported(d) ? std::string(" for ") + path_name(path)
                                  : std::string()));
  }
  if (d->num_seqs == 0 || d->total_q == 0)
    return TURBINE_OK;
  if (d->max_q_len < 1 || d->max_blocks_per_seq < 1 || d->num_blocks < 1) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                entry +
                    ": a batch with queries needs max_q_len, "
                    "max_blocks_per_seq and num_blocks >= 1 (" +
                    describe(d) + ")");
  }
  if (d->q == nullptr || d->k_new == nullptr || d->v_new == nullptr ||
      d->out == nullptr || d->kv_layer == nullptr ||
      d->block_table == nullptr || d->q_indptr == nullptr ||
      d->kv_lens == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, entry + ": NULL operand");
  }
  if (mixed(d)) {
    if (const std::string why = mixed_operand_error(d); !why.empty())
      return fail(ctx, TURBINE_E_ARGUMENT, entry + ": " + why);
  }
  if ((d->dtype == TURBINE_DTYPE_F8E4M3 || d->block_formats != nullptr) &&
      !(std::isfinite(d->k_scale) && d->k_scale > 0.0f &&
        std::isfinite(d->v_scale) && d->v_scale > 0.0f)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                entry +
                    ": FP8 pages need finite positive k_scale / v_scale, "
                    "got " +
                    std::to_string(d->k_scale) + " / " +
                    std::to_string(d->v_scale));
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  // The b1 entry is selected per op config, so it receives every decode batch
  // size; from kTurbineB1CkSeqs sequences on, CK fmha_fwd_splitkv is as fast
  // or faster (hip_ops b1_ck_edge_timings on the R9700, RoPE fused, in a
  // graph, cold KV, us per layer b1 / CK: b=1 13.1 / 41.6 at ~768 tokens;
  // b=9 261.2 / 266.9 at ~4,096 and 502.5 / 507.9 at ~8,192; b=10 569.2 /
  // 563.7 at ~8,192; b=64 535 / 403 at ~768), so the entry hands those
  // batches to it when CK serves the page size. A request
  // whose batch crosses the edge changes its rounding once (the two are
  // different numerics classes); within each side rows are batch-invariant.
  if (path == turbine_hip::PagedPath::TurbineB1 &&
      d->num_seqs >= turbine_hip::kTurbineB1CkSeqs && ck_serves(d) &&
      turbine_hip::ck_splitkv_serves(d)) {
    path = turbine_hip::PagedPath::CkSplitkv;
  }
  // v2.12 fused RoPE: the b1 kernel rotates q and the appended k row itself;
  // every other path rotates q and k_new in place first, with the rope
  // kernels (the same operations as a separate turbine_rope call).
  if (d->rope_inv_freq != nullptr && d->rope_positions == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, entry + ": NULL rope_positions");
  }
  if (d->rope_inv_freq != nullptr &&
      path != turbine_hip::PagedPath::TurbineB1) {
    turbine_rope_desc rd{};
    rd.q = const_cast<void *>(d->q);
    rd.k = const_cast<void *>(d->k_new);
    rd.positions = d->rope_positions;
    rd.inv_freq = d->rope_inv_freq;
    rd.num_tokens = d->total_q;
    rd.num_q_heads = d->num_q_heads;
    rd.num_kv_heads = d->num_kv_heads;
    rd.head_dim = d->head_dim;
    rd.rotary_dim = d->rope_rotary_dim;
    rd.q_stride_token = d->q_stride_token;
    rd.k_stride_token = d->new_stride_token;
    rd.style = 0;
    rd.dtype = TURBINE_DTYPE_BF16;
    rd.attn_factor = d->rope_attn_factor;
    if (int32_t rc = turbine_rope(ctx, &rd); rc != TURBINE_OK)
      return rc;
  }
  // The b1 decode kernel appends inside the kernel; every other path takes
  // the common append first.
  if (int32_t rc =
          path == turbine_hip::PagedPath::TurbineB1
              ? TURBINE_OK
              : (mixed(d) ? turbine_hip::launch_paged_append_mixed(ctx, d)
                          : turbine_hip::launch_paged_append(ctx, d));
      rc != TURBINE_OK) {
    return rc;
  }
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkvMixedStaged:
    return run_mixed_staged(ctx, d, entry, true);
  case turbine_hip::PagedPath::TurbineMixedStaged:
    return run_mixed_staged(ctx, d, entry, false);
  case turbine_hip::PagedPath::TurbineMixed:
    return turbine_hip::run_paged_mixed_attention(ctx, d, false);
  case turbine_hip::PagedPath::CkPagedkv:
    return run_ck(ctx, d, entry);
  case turbine_hip::PagedPath::CkSplitkv:
    return turbine_hip::run_ck_splitkv(ctx, d, entry);
  case turbine_hip::PagedPath::TurbineB1:
    return turbine_hip::run_b1_decode(ctx, d, entry);
  case turbine_hip::PagedPath::CkPagedkvFp8Staged:
    return run_ck_fp8_staged(ctx, d, entry);
  case turbine_hip::PagedPath::TurbineFp8Decode:
    return turbine_hip::launch_paged_decode_fp8(ctx, d);
  case turbine_hip::PagedPath::Turbine:
  case turbine_hip::PagedPath::TurbineFp8:
    break;
  }
  return turbine_hip::launch_paged_attention(ctx, d);
}

} // namespace

namespace turbine_hip {

bool paged_supports(const turbine_attention_paged_desc *d, PagedPath path) {
  return supported(d) && path_serves(d, path);
}

int32_t paged_run(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
                  const char *entry, PagedPath path) {
  return run(ctx, d, entry, path);
}

} // namespace turbine_hip

extern "C" {

int32_t
turbine_attention_prefill_paged(turbine_ctx *ctx,
                                const turbine_attention_prefill_paged_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_ATTENTION_PREFILL_PAGED, d);
}

int32_t turbine_attention_prefill_paged_supported(
    const turbine_attention_prefill_paged_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_attention_prefill_paged_impl(
    const turbine_attention_prefill_paged_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_ATTENTION_PREFILL_PAGED, d);
}

int32_t
turbine_attention_decode_paged(turbine_ctx *ctx,
                               const turbine_attention_decode_paged_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_ATTENTION_DECODE_PAGED, d);
}

int32_t turbine_attention_decode_paged_supported(
    const turbine_attention_decode_paged_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_attention_decode_paged_impl(
    const turbine_attention_decode_paged_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_ATTENTION_DECODE_PAGED, d);
}

} // extern "C"
