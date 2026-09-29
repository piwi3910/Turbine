// FP8 KV pages (Phase 6a S-13): the kernels of paged_attention.hip that
// paged_attention.cpp launches besides the append and the Turbine kernel
// (turbine_hip.hpp), and their limits. The descriptors were validated by
// paged_attention.cpp (dtype TURBINE_DTYPE_F8E4M3, head_dim 128, total_q > 0,
// finite positive k_scale / v_scale).
#pragma once

#include <cstddef>
#include <cstdint>

// turbine_hip.hpp includes the ABI header with default visibility; it must
// come first (the header guard keeps a later include from changing that).
#include "turbine_hip.hpp"

namespace turbine_hip {

// Largest num_q_heads / num_kv_heads the FP8 decode kernel handles.
constexpr int32_t kPagedFp8DecodeMaxGroup = 8;

// Decode over FP8 pages, every query head of a KV head in one workgroup: the
// sequences with exactly one query row (every one in a decode call; the
// decode rows of a staged prefill call); the others are skipped.
int32_t launch_paged_decode_fp8(turbine_ctx *ctx,
                                const turbine_attention_paged_desc *d);

// Dequantizes the pages of sequences [g0, g0 + count) into BF16 blocks of
// staged ([count * pages] blocks, the pool's block layout) and writes the
// staged block table [count, pages] (row-major) into staged_table and the key
// counts CK sees into staged_kv_lens [count]; a sequence with one query row is
// not staged (the decode kernel computes it) and gets one key.
int32_t launch_paged_stage_fp8(turbine_ctx *ctx,
                               const turbine_attention_paged_desc *d,
                               void *staged, int32_t *staged_table,
                               int32_t *staged_kv_lens, int32_t g0,
                               int32_t count, int32_t pages);

// Grows ctx's attention scratch (paged_attention_splitkv.cpp) to at least
// bytes; refused while capturing.
int32_t attention_scratch(turbine_ctx *ctx, size_t bytes, void **out);

} // namespace turbine_hip
