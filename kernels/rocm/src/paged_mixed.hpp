// Mixed-format paged attention (ABI v2.11, Phase 6b S-5): the kernels of
// paged_attention_mixed.hip that paged_attention.cpp launches for a descriptor
// whose dtype is TURBINE_DTYPE_TQ4 / _TQ2 or whose block_formats is not NULL.
// The descriptors were validated by paged_attention.cpp (head_dim 128, a GQA
// group of at most kPagedMixedMaxGroup, block_tokens a multiple of 16, the
// TurboQuant tables present and 16-byte aligned, finite positive FP8 scales).
#pragma once

#include <cstddef>
#include <cstdint>

// turbine_hip.hpp includes the ABI header with default visibility; it must
// come first (the header guard keeps a later include from changing that).
#include "turbine_hip.hpp"

namespace turbine_hip {

// Largest num_q_heads / num_kv_heads the mixed-format kernels handle.
constexpr int32_t kPagedMixedMaxGroup = 8;

// The TURBINE_KVFMT_* code of pages of dtype (-1: not a KV page dtype).
int32_t paged_mixed_format_of(int32_t dtype);

// Bytes of one layer of one page of format fmt (TURBINE_KVFMT_*), 0 for an
// unknown code.
int64_t paged_mixed_page_bytes(int32_t fmt, int32_t block_tokens,
                               int32_t kv_heads);

// Appends k_new / v_new into their pages, each row in its block's format
// (TurboQuant records through the codec's encode, bit-exact with the CPU
// codec; FP8 as e4m3(x / scale); BF16 as is).
int32_t launch_paged_append_mixed(turbine_ctx *ctx,
                                  const turbine_attention_paged_desc *d);

// The own mixed-format attention ("turbine_hip_mixed"): every query row (or,
// with single_only, the rows of sequences with one query row) over its
// visible keys, each block read by its format, TurboQuant blocks scored and
// accumulated in the rotated domain. Splits each row's keys into at most 16
// splits of a length fixed by the row's own key count (batch-invariant),
// combined in a second kernel. Uses the context's attention scratch, grown
// outside a capture only.
int32_t run_paged_mixed_attention(turbine_ctx *ctx,
                                  const turbine_attention_paged_desc *d,
                                  bool single_only);

// Decodes the pages of sequences [g0, g0 + count) into BF16 blocks of staged
// ([count * pages] blocks, the BF16 pool layout) by their formats, writes the
// staged block table [count, pages] into staged_table and the key counts into
// staged_kv_lens [count]; a sequence with one query row is not staged and
// gets one key (its row is computed by run_paged_mixed_attention after).
int32_t launch_paged_stage_mixed(turbine_ctx *ctx,
                                 const turbine_attention_paged_desc *d,
                                 void *staged, int32_t *staged_table,
                                 int32_t *staged_kv_lens, int32_t g0,
                                 int32_t count, int32_t pages);

} // namespace turbine_hip
