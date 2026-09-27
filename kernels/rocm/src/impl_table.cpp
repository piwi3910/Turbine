// The implementations of every op (ABI v2.4), in library order: the table
// turbine_impl_count / _info / _supports / _run enumerate and run
// (impl_exports.cpp). Index order is the library order.
//
//   gemm                         0 hipblaslt [hipblaslt]
//   attention_prefill / _decode  0 ck_tile_fmha_fwd [ck]
//   rmsnorm, add_rmsnorm         0 ck_tile_rmsnorm2d [ck] (the instantiated
//                                  BF16 buckets), 1 turbine_hip [turbine_hip]
//   attention_prefill_paged      0 ck_tile_fmha_pagedkv [ck] (pages of a
//                                  multiple of 128 tokens), 1 turbine_hip
//   attention_decode_paged       0 ck_tile_fmha_splitkv [ck] (grouped query
//                                  heads, pages of a multiple of 128 tokens),
//                                  1 ck_tile_fmha_pagedkv [ck], 2 turbine_hip
//   copy_blocks                  0 hip_memcpy_d2d [turbine_hip]
//   moe_experts                  0 turbine_hip_moe_small_m (hidden and inter
//                                  multiples of 8; the first row tier),
//                                  1 turbine_hip_moe_wmma (multiples of 64),
//                                  2 hipblaslt_grouped (only when hipBLASLt
//                                  has a grouped solution), 3
//                                  hipblaslt_per_expert; 2 and 3 read
//                                  host_expert_offsets
//   rope, silu_mul, embedding,   0 turbine_hip [turbine_hip]
//   add, moe_route, logits_reduce
#include <string>

#include "turbine_hip.hpp"

namespace turbine_hip {
namespace {

constexpr const char *kHipblaslt = "hipblaslt";
constexpr const char *kCk = "ck";
constexpr const char *kTurbine = "turbine_hip";

// An implementation that is the whole op: its entry point trio.
template <typename D, int32_t (*Supported)(const D *),
          int32_t (*Run)(turbine_ctx *, const D *)>
struct Whole {
  static bool supports(const void *d) {
    return Supported(static_cast<const D *>(d)) == 1;
  }
  static int32_t run(turbine_ctx *ctx, const void *d) {
    return Run(ctx, static_cast<const D *>(d));
  }
};

template <typename D, int32_t (*Supported)(const D *),
          int32_t (*Run)(turbine_ctx *, const D *)>
constexpr ImplEntry whole(const char *name, const char *provider) {
  return {name,
          provider,
          0,
          nullptr,
          &Whole<D, Supported, Run>::supports,
          &Whole<D, Supported, Run>::run};
}

template <bool Ck> struct Norm {
  static bool supports(const void *d) {
    return rmsnorm_supports(static_cast<const turbine_rmsnorm_desc *>(d), Ck);
  }
  static int32_t run(turbine_ctx *ctx, const void *d) {
    return rmsnorm_run(ctx, static_cast<const turbine_rmsnorm_desc *>(d), Ck);
  }
};

template <bool Ck> struct AddNorm {
  static bool supports(const void *d) {
    return add_rmsnorm_supports(
        static_cast<const turbine_add_rmsnorm_desc *>(d), Ck);
  }
  static int32_t run(turbine_ctx *ctx, const void *d) {
    return add_rmsnorm_run(
        ctx, static_cast<const turbine_add_rmsnorm_desc *>(d), Ck);
  }
};

constexpr const char kPrefillPaged[] = "turbine_attention_prefill_paged";
constexpr const char kDecodePaged[] = "turbine_attention_decode_paged";

template <const char *Entry, PagedPath Path> struct Paged {
  static bool supports(const void *d) {
    return paged_supports(static_cast<const turbine_attention_paged_desc *>(d),
                          Path);
  }
  static int32_t run(turbine_ctx *ctx, const void *d) {
    return paged_run(ctx, static_cast<const turbine_attention_paged_desc *>(d),
                     Entry, Path);
  }
};

template <MoePath Path> struct Experts {
  static bool supports(const void *d) {
    return moe_experts_supports(
        static_cast<const turbine_moe_experts_desc *>(d), Path);
  }
  static int32_t run(turbine_ctx *ctx, const void *d) {
    return moe_experts_run(
        ctx, static_cast<const turbine_moe_experts_desc *>(d), Path);
  }
};

// The profile's page multiple: the default runs CK paged attention only for
// pages of a multiple of it (supports already requires the CK instance's own).
bool page_multiple_allows(const Profile &profile, const void *d) {
  const auto *desc = static_cast<const turbine_attention_paged_desc *>(d);
  return profile.paged_page_multiple > 0 &&
         desc->block_tokens % profile.paged_page_multiple == 0;
}

// The profile's first moe_experts row tier: the default runs the small-m
// kernels up to moe_small_max_rows routed rows.
bool small_rows_allows(const Profile &profile, const void *d) {
  const auto *desc = static_cast<const turbine_moe_experts_desc *>(d);
  return static_cast<int64_t>(desc->num_tokens) * desc->top_k <=
         profile.moe_small_max_rows;
}

template <typename S>
constexpr ImplEntry
entry(const char *name, const char *provider, uint32_t flags = 0,
      bool (*profile_allows)(const Profile &, const void *) = nullptr) {
  return {name, provider, flags, profile_allows, &S::supports, &S::run};
}

const ImplEntry kGemm[] = {
    whole<turbine_gemm_desc, turbine_gemm_supported, turbine_gemm>("hipblaslt",
                                                                   kHipblaslt),
};
const ImplEntry kAttentionPrefill[] = {
    whole<turbine_attention_prefill_desc, turbine_attention_prefill_supported,
          turbine_attention_prefill>("ck_tile_fmha_fwd", kCk),
};
const ImplEntry kAttentionDecode[] = {
    whole<turbine_attention_decode_desc, turbine_attention_decode_supported,
          turbine_attention_decode>("ck_tile_fmha_fwd", kCk),
};
const ImplEntry kRmsnorm[] = {
    entry<Norm<true>>("ck_tile_rmsnorm2d", kCk),
    entry<Norm<false>>("turbine_hip", kTurbine),
};
const ImplEntry kRope[] = {
    whole<turbine_rope_desc, turbine_rope_supported, turbine_rope>(
        "turbine_hip", kTurbine),
};
const ImplEntry kSiluMul[] = {
    whole<turbine_silu_mul_desc, turbine_silu_mul_supported, turbine_silu_mul>(
        "turbine_hip", kTurbine),
};
const ImplEntry kEmbedding[] = {
    whole<turbine_embedding_desc, turbine_embedding_supported,
          turbine_embedding>("turbine_hip", kTurbine),
};
const ImplEntry kAdd[] = {
    whole<turbine_add_desc, turbine_add_supported, turbine_add>("turbine_hip",
                                                                kTurbine),
};
const ImplEntry kPrefillPagedImpls[] = {
    entry<Paged<kPrefillPaged, PagedPath::CkPagedkv>>(
        "ck_tile_fmha_pagedkv", kCk, 0, page_multiple_allows),
    entry<Paged<kPrefillPaged, PagedPath::Turbine>>("turbine_hip", kTurbine),
};
const ImplEntry kDecodePagedImpls[] = {
    entry<Paged<kDecodePaged, PagedPath::CkSplitkv>>(
        "ck_tile_fmha_splitkv", kCk, 0, page_multiple_allows),
    entry<Paged<kDecodePaged, PagedPath::CkPagedkv>>(
        "ck_tile_fmha_pagedkv", kCk, 0, page_multiple_allows),
    entry<Paged<kDecodePaged, PagedPath::Turbine>>("turbine_hip", kTurbine),
};
const ImplEntry kCopyBlocks[] = {
    whole<turbine_copy_blocks_desc, turbine_copy_blocks_supported,
          turbine_copy_blocks>("hip_memcpy_d2d", kTurbine),
};
const ImplEntry kMoeRoute[] = {
    whole<turbine_moe_route_desc, turbine_moe_route_supported,
          turbine_moe_route>("turbine_hip", kTurbine),
};
const ImplEntry kMoeExperts[] = {
    entry<Experts<MoePath::SmallM>>("turbine_hip_moe_small_m", kTurbine, 0,
                                    small_rows_allows),
    entry<Experts<MoePath::Wmma>>("turbine_hip_moe_wmma", kTurbine),
    entry<Experts<MoePath::Grouped>>("hipblaslt_grouped", kHipblaslt,
                                     TURBINE_IMPL_NEEDS_HOST_OFFSETS),
    entry<Experts<MoePath::PerExpert>>("hipblaslt_per_expert", kHipblaslt,
                                       TURBINE_IMPL_NEEDS_HOST_OFFSETS),
};
const ImplEntry kAddRmsnorm[] = {
    entry<AddNorm<true>>("ck_tile_rmsnorm2d", kCk),
    entry<AddNorm<false>>("turbine_hip", kTurbine),
};
const ImplEntry kLogitsReduce[] = {
    whole<turbine_logits_reduce_desc, turbine_logits_reduce_supported,
          turbine_logits_reduce>("turbine_hip", kTurbine),
};

struct OpImpls {
  const ImplEntry *entries;
  int32_t count;
};

template <size_t N> constexpr OpImpls of(const ImplEntry (&entries)[N]) {
  return {entries, static_cast<int32_t>(N)};
}

// Indexed by TURBINE_OP_*.
const OpImpls kOps[] = {
    of(kGemm),
    of(kAttentionPrefill),
    of(kAttentionDecode),
    of(kRmsnorm),
    of(kRope),
    of(kSiluMul),
    of(kEmbedding),
    of(kAdd),
    of(kPrefillPagedImpls),
    of(kDecodePagedImpls),
    of(kCopyBlocks),
    of(kMoeRoute),
    of(kMoeExperts),
    of(kAddRmsnorm),
    of(kLogitsReduce),
};
static_assert(sizeof(kOps) / sizeof(kOps[0]) == TURBINE_OP_LOGITS_REDUCE + 1,
              "one implementation list per TURBINE_OP_* code");

} // namespace

const ImplEntry *impl_entries(int32_t op, int32_t *count) {
  if (op < 0 || op > TURBINE_OP_LOGITS_REDUCE) {
    *count = 0;
    return nullptr;
  }
  *count = kOps[op].count;
  return kOps[op].entries;
}

const ImplEntry *default_entry(const Profile &profile, int32_t op,
                               const void *desc) {
  int32_t count = 0;
  const ImplEntry *entries = impl_entries(op, &count);
  if (desc == nullptr)
    return nullptr;
  for (int32_t i = 0; i < count; ++i) {
    const ImplEntry &e = entries[i];
    if (e.supports(desc) &&
        (e.profile_allows == nullptr || e.profile_allows(profile, desc))) {
      return &e;
    }
  }
  return nullptr;
}

int32_t run_default(turbine_ctx *ctx, int32_t op, const void *desc) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  int32_t count = 0;
  const ImplEntry *entries = impl_entries(op, &count);
  if (entries == nullptr)
    return fail(ctx, TURBINE_E_ARGUMENT, "unknown op " + std::to_string(op));
  const ImplEntry *e = default_entry(ctx->profile, op, desc);
  // No implementation takes desc: the last one runs and refuses it with the
  // op's message (a NULL descriptor or an unsupported configuration).
  return (e != nullptr ? e : &entries[count - 1])->run(ctx, desc);
}

const char *default_name(int32_t op, const void *desc) {
  int32_t count = 0;
  const ImplEntry *entries = impl_entries(op, &count);
  if (entries == nullptr)
    return "";
  const ImplEntry *e = default_entry(kDefaultProfile, op, desc);
  return (e != nullptr ? e : &entries[count - 1])->name;
}

uint32_t default_flags(int32_t op, const void *desc, uint32_t fallback) {
  const ImplEntry *e = default_entry(kDefaultProfile, op, desc);
  return e != nullptr ? e->flags : fallback;
}

} // namespace turbine_hip
