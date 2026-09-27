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
// - the Turbine HIP paged kernel (impl "turbine_hip"), any page size.
// Decode is the same computation with one query per sequence and runs the same
// paths.
//
// The device arrays (block_table, q_indptr, kv_lens) are trusted: the host
// cannot read them without a synchronisation. The Turbine kernels skip pages
// outside the pool instead of faulting.
#include <string>

#include "fmha_fwd.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImplCk = "ck_tile_fmha_pagedkv";
constexpr const char *kImplCkSplitkv = "ck_tile_fmha_splitkv";
constexpr const char *kImplTurbine = "turbine_hip";
constexpr int32_t kHeadDim = 128;
// Page sizes the CK pagedkv instances serve are multiples of this: a
// property of the instances (a capability supports reports), not a tuning
// threshold.
constexpr int32_t kCkPagedkvPage = 128;
// Grid y (sequences) and z (KV heads) limits of the Turbine kernel.
constexpr int32_t kMaxGridYZ = 65535;

bool supported(const turbine_attention_paged_desc *d) {
  if (d == nullptr)
    return false;
  if (d->dtype != TURBINE_DTYPE_BF16 || d->head_dim != kHeadDim)
    return false;
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
  return d->q_stride_token >= static_cast<int64_t>(d->num_q_heads) * kHeadDim &&
         d->out_stride_token >=
             static_cast<int64_t>(d->num_q_heads) * kHeadDim &&
         d->new_stride_token >=
             static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
}

// What the CK instances can serve: a page size they are built for and every
// stride within ck_tile's 32-bit index_t.
bool ck_serves(const turbine_attention_paged_desc *d) {
  if (d->block_tokens % kCkPagedkvPage != 0)
    return false;
  const int64_t page_stride =
      2 * static_cast<int64_t>(d->block_tokens) * d->num_kv_heads * kHeadDim;
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
         " causal=" + std::to_string(d->causal);
}

int32_t run_ck(turbine_ctx *ctx, const turbine_attention_paged_desc *d,
               const std::string &entry) {
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
  traits.has_sink = false;

  const int64_t token_elems = static_cast<int64_t>(d->num_kv_heads) * kHeadDim;
  const auto page_stride =
      static_cast<ck_tile::index_t>(2 * d->block_tokens * token_elems);
  // BF16 elements, addressed as their 16-bit storage.
  const auto *pool = static_cast<const uint16_t *>(d->kv_layer);

  fmha_fwd_pagedkv_args args{};
  args.q_ptr = d->q;
  args.k_ptr = pool;
  args.v_ptr = pool + static_cast<int64_t>(d->block_tokens) * token_elems;
  args.bias_ptr = nullptr;
  args.lse_ptr = nullptr;
  args.o_ptr = d->out;
  args.block_table_ptr = const_cast<int32_t *>(d->block_table);
  args.batch_stride_block_table = d->max_blocks_per_seq;
  args.page_block_size = d->block_tokens;
  args.is_gappy = false;
  args.cache_batch_idx = nullptr;
  // Group mode with a page table: query rows from seqstart_q, key counts from
  // seqlen_k. The kernel reads seqstart_k[b] but, with pages, only for an
  // offset it never uses; q_indptr is a valid [num_seqs + 1] array for it.
  args.seqstart_q_ptr = d->q_indptr;
  args.seqstart_k_ptr = d->q_indptr;
  args.seqlen_k_ptr = d->kv_lens;
  args.sink_ptr = nullptr;
  args.seqlen_q = d->max_q_len;
  args.seqlen_k = d->max_kv_len;
  args.batch = d->num_seqs;
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
  args.window_size_left = -1;
  args.window_size_right = d->causal == 1 ? 0 : -1;
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
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkv:
    return ck_serves(d);
  case turbine_hip::PagedPath::CkSplitkv:
    return ck_serves(d) && turbine_hip::ck_splitkv_serves(d);
  case turbine_hip::PagedPath::Turbine:
    return true;
  }
  return false;
}

const char *path_name(turbine_hip::PagedPath path) {
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkv:
    return kImplCk;
  case turbine_hip::PagedPath::CkSplitkv:
    return kImplCkSplitkv;
  case turbine_hip::PagedPath::Turbine:
    break;
  }
  return kImplTurbine;
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
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = turbine_hip::launch_paged_append(ctx, d); rc != TURBINE_OK)
    return rc;
  switch (path) {
  case turbine_hip::PagedPath::CkPagedkv:
    return run_ck(ctx, d, entry);
  case turbine_hip::PagedPath::CkSplitkv:
    return turbine_hip::run_ck_splitkv(ctx, d, entry);
  case turbine_hip::PagedPath::Turbine:
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
