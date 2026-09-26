// ABI v2.4 exports: implementation enumeration, explicit runs and the card
// profile, over the table of impl_table.cpp. Linked into libturbine_hip.so
// only; libturbine_hip_v23.so leaves this file out, so it has no v2.4 group
// (abi_minor.cpp reports 3 there).
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::fail;
using turbine_hip::ImplEntry;

namespace {

// Implementation index of op, or nullptr for an unknown op or index.
const ImplEntry *find(int32_t op, int32_t index) {
  int32_t count = 0;
  const ImplEntry *entries = turbine_hip::impl_entries(op, &count);
  if (entries == nullptr || index < 0 || index >= count)
    return nullptr;
  return &entries[index];
}

} // namespace

extern "C" {

int32_t turbine_impl_count(int32_t op) {
  int32_t count = 0;
  if (turbine_hip::impl_entries(op, &count) == nullptr)
    return TURBINE_E_ARGUMENT;
  return count;
}

int32_t turbine_impl_info(int32_t op, int32_t index, turbine_impl_entry *out) {
  const ImplEntry *e = find(op, index);
  if (e == nullptr || out == nullptr)
    return TURBINE_E_ARGUMENT;
  out->name = e->name;
  out->provider = e->provider;
  out->flags = e->flags;
  return TURBINE_OK;
}

int32_t turbine_impl_supports(int32_t op, int32_t index, const void *desc) {
  const ImplEntry *e = find(op, index);
  if (e == nullptr || desc == nullptr)
    return TURBINE_E_ARGUMENT;
  return e->supports(desc) ? 1 : 0;
}

int32_t turbine_impl_run(turbine_ctx *ctx, int32_t op, int32_t index,
                         const void *desc) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  const ImplEntry *e = find(op, index);
  if (e == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_impl_run: no implementation " + std::to_string(index) +
                    " of op " + std::to_string(op));
  }
  // The implementation's run refuses a descriptor it does not support
  // (TURBINE_E_UNSUPPORTED) with the op's message naming the implementation.
  return e->run(ctx, desc);
}

int32_t turbine_ctx_set_profile(turbine_ctx *ctx,
                                const turbine_card_profile *p) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (p == nullptr || p->struct_bytes < sizeof(turbine_card_profile) ||
      p->arch == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_ctx_set_profile: profile is NULL, has no arch or is "
                "smaller than turbine_card_profile");
  }
  if (!turbine_hip::arch_is_built(p->arch)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string("turbine_ctx_set_profile: arch ") + p->arch +
                    " is not in library build archs " + TURBINE_BUILD_ARCHS);
  }
  if (p->wave_size != ctx->wave_size) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_ctx_set_profile: wave size " +
                    std::to_string(p->wave_size) + ", the device runs " +
                    std::to_string(ctx->wave_size));
  }
  if (p->lds_bytes < turbine_hip::kBuiltLdsBytes) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_ctx_set_profile: " + std::to_string(p->lds_bytes) +
                    " LDS bytes, the kernels use up to " +
                    std::to_string(turbine_hip::kBuiltLdsBytes));
  }
  if (p->moe_small_max_rows < 0 || p->paged_page_multiple < 1) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_ctx_set_profile: moe_small_max_rows must be >= 0 and "
                "paged_page_multiple >= 1");
  }
  ctx->profile =
      turbine_hip::Profile{p->arch, p->wave_size, p->lds_bytes,
                           p->moe_small_max_rows, p->paged_page_multiple};
  return TURBINE_OK;
}

} // extern "C"
