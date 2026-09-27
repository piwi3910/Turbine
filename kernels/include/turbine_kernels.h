/* Turbine vendor-neutral kernel C ABI, version 2.4 (contract section 9).
 *
 * Every backend shim library implements this header and is loaded by
 * turbine-kernels at run time. No vendor type, identifier or name appears here,
 * so every backend shares one ABI.
 *
 * Return codes. Every function except the identity functions and
 * turbine_ctx_destroy returns int32_t: TURBINE_OK (0) on success, a negative
 * TURBINE_E_* code on failure. turbine_<op>_supported returns 1 (supported) or
 * 0 (not supported), negative on an internal error.
 *
 * Pointers. All pointers in descriptors are device pointers unless the field
 * comment says "host". turbine_<op>_supported and turbine_<op>_impl ignore
 * pointer fields (they may be NULL) and never need a context.
 *
 * Streams. All work is enqueued on the context's compute stream;
 * turbine_stream_sync is the only blocking call (besides the capture
 * boundaries of the v2.1 graph functions). Host buffers passed to
 * turbine_memcpy_h2d / turbine_memcpy_d2h must stay valid until the next
 * turbine_stream_sync (for v2.3 pinned host memory: until an event recorded
 * after the copy has completed).
 *
 * Versions. TURBINE_ABI_VERSION is the major version and must match exactly.
 * A minor revision only adds optional symbols or descriptor flag bits: a
 * library without the symbols (minor 0, no turbine_abi_minor) still loads, and
 * the caller falls back; a library of an earlier minor reports a descriptor
 * with a flag bit it does not know unsupported.
 *
 * Layout. Row-major everywhere; strides are in elements; "leading dimension"
 * is the row stride in elements.
 *
 * Symbols. Only turbine_* symbols are exported.
 *
 * Ownership. Every device pointer is allocated by the library (turbine_malloc)
 * and freed exactly once by its owner (turbine_free). The library never retains
 * a caller pointer beyond the call, except that a captured graph (v2.1)
 * records the device pointers of the ops captured into it. The context owns
 * its streams, library handles and workspace and releases them in
 * turbine_ctx_destroy.
 *
 * Errors. turbine_last_error(ctx, buf, len) copies the NUL-terminated message
 * of the most recent failure on ctx (with ctx == NULL: of the most recent
 * failed turbine_ctx_create on the calling thread) into the host buffer buf and
 * returns the full message length excluding the NUL; with buf == NULL or
 * len == 0 it writes nothing and only returns the length. Device runtime
 * messages start with the runtime's own error name. */
#ifndef TURBINE_KERNELS_H
#define TURBINE_KERNELS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TURBINE_ABI_VERSION 2u

/* ---- status codes ---- */
#define TURBINE_OK 0
/* invalid argument */
#define TURBINE_E_ARGUMENT (-1)
/* unsupported configuration */
#define TURBINE_E_UNSUPPORTED (-2)
/* device (or pinned) out of memory */
#define TURBINE_E_OUT_OF_MEMORY (-3)
/* device runtime error */
#define TURBINE_E_DEVICE (-4)
/* provider library error */
#define TURBINE_E_LIBRARY (-5)

/* ---- dtype codes (= Rust DType::abi_code); 16..63 reserved for quantized
 * formats ---- */
#define TURBINE_DTYPE_BF16 0
#define TURBINE_DTYPE_F16 1
#define TURBINE_DTYPE_F32 2
#define TURBINE_DTYPE_I32 3
#define TURBINE_DTYPE_I64 4

typedef struct turbine_ctx turbine_ctx;

/* ======== v1: identity, context, memory ======== */
uint32_t turbine_abi_version(void);
/* Equals the execution.backend value the library serves. */
const char *turbine_backend_name(void);
/* Comma-separated device architectures the library was compiled for. */
const char *turbine_build_archs(void);
/* device_ordinal is the device's vendor_index in the Turbine inventory. */
int32_t turbine_ctx_create(int32_t device_ordinal, turbine_ctx **out);
void turbine_ctx_destroy(turbine_ctx *ctx);
int32_t turbine_malloc(turbine_ctx *ctx, size_t bytes, void **out);
int32_t turbine_free(turbine_ctx *ctx, void *ptr);
int32_t turbine_memcpy_h2d(turbine_ctx *ctx, void *dst_device,
                           const void *src_host, size_t bytes);
int32_t turbine_memcpy_d2h(turbine_ctx *ctx, void *dst_host,
                           const void *src_device, size_t bytes);
int32_t turbine_stream_sync(turbine_ctx *ctx);
/* free_bytes and total_bytes are host outputs. */
int32_t turbine_mem_info(turbine_ctx *ctx, size_t *free_bytes,
                         size_t *total_bytes);
size_t turbine_last_error(turbine_ctx *ctx, char *buf, size_t len);

/* ======== v1: op descriptors ======== */

/* c[m,n] = alpha * a[m,k] . op(b) + beta * c; BF16 in, BF16 or F32 out, F32
 * accumulate. */
typedef struct turbine_gemm_desc {
  const void *a;
  const void *b;
  void *c;
  int64_t m, n, k;
  int64_t lda, ldb, ldc;
  /* 1: b is [n,k] (HF Linear weight), 0: b is [k,n] */
  int32_t trans_b;
  int32_t a_dtype, b_dtype, c_dtype;
  float alpha, beta;
} turbine_gemm_desc;

/* Contiguous per-sequence KV. Query i sits at absolute position q_start + i;
 * decode has q_len = 1. */
typedef struct turbine_attention_desc {
  /* [q_len, num_q_heads, head_dim] */
  const void *q;
  /* [kv_capacity, num_kv_heads, head_dim], rows [0, q_start + q_len) valid */
  const void *k_cache;
  const void *v_cache;
  /* [q_len, num_q_heads, head_dim] */
  void *out;
  int32_t q_len, q_start;
  int32_t num_q_heads, num_kv_heads, head_dim;
  int64_t q_stride_token, kv_stride_token, out_stride_token;
  /* 1/sqrt(head_dim) */
  float scale;
  int32_t causal, dtype;
} turbine_attention_desc;
typedef turbine_attention_desc turbine_attention_prefill_desc;
typedef turbine_attention_desc turbine_attention_decode_desc;

typedef struct turbine_rmsnorm_desc {
  const void *x;
  const void *weight;
  void *out;
  int64_t rows, dim, x_stride_row, out_stride_row;
  float eps;
  int32_t dtype;
} turbine_rmsnorm_desc;

/* In place on q and k. */
typedef struct turbine_rope_desc {
  void *q;
  void *k;
  /* [num_tokens] */
  const int32_t *positions;
  /* [rotary_dim/2], computed on the host in FP64 (llama3 scaling or standard),
   * stored F32 */
  const float *inv_freq;
  int32_t num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim;
  int64_t q_stride_token, k_stride_token;
  /* 0 = half-split (HF rotate_half) */
  int32_t style;
  int32_t dtype;
} turbine_rope_desc;

/* out = silu(gate) * up */
typedef struct turbine_silu_mul_desc {
  const void *gate;
  const void *up;
  void *out;
  int64_t rows, cols, gate_stride_row, up_stride_row, out_stride_row;
  int32_t dtype;
} turbine_silu_mul_desc;

/* out[t] = table[ids[t] - vocab_offset], or 0 when that row is outside
 * [0, vocab_rows). */
typedef struct turbine_embedding_desc {
  const int32_t *ids;
  const void *table;
  void *out;
  int64_t num_tokens, hidden, vocab_offset, vocab_rows, out_stride_row;
  int32_t dtype;
} turbine_embedding_desc;

/* out = a + b over n contiguous elements */
typedef struct turbine_add_desc {
  const void *a;
  const void *b;
  void *out;
  int64_t n;
  int32_t dtype;
} turbine_add_desc;

/* One trio per op, <op> in gemm, attention_prefill, attention_decode, rmsnorm,
 * rope, silu_mul, embedding, add: turbine_<op> enqueues the op;
 * turbine_<op>_supported reports whether the descriptor's shapes and dtypes
 * can run; turbine_<op>_impl names the implementation that would run them (a
 * static string owned by the library, logged in the kernel selection record).
 */
int32_t turbine_gemm(turbine_ctx *ctx, const turbine_gemm_desc *d);
int32_t turbine_gemm_supported(const turbine_gemm_desc *d);
const char *turbine_gemm_impl(const turbine_gemm_desc *d);

int32_t turbine_attention_prefill(turbine_ctx *ctx,
                                  const turbine_attention_prefill_desc *d);
int32_t
turbine_attention_prefill_supported(const turbine_attention_prefill_desc *d);
const char *
turbine_attention_prefill_impl(const turbine_attention_prefill_desc *d);

int32_t turbine_attention_decode(turbine_ctx *ctx,
                                 const turbine_attention_decode_desc *d);
int32_t
turbine_attention_decode_supported(const turbine_attention_decode_desc *d);
const char *
turbine_attention_decode_impl(const turbine_attention_decode_desc *d);

int32_t turbine_rmsnorm(turbine_ctx *ctx, const turbine_rmsnorm_desc *d);
int32_t turbine_rmsnorm_supported(const turbine_rmsnorm_desc *d);
const char *turbine_rmsnorm_impl(const turbine_rmsnorm_desc *d);

int32_t turbine_rope(turbine_ctx *ctx, const turbine_rope_desc *d);
int32_t turbine_rope_supported(const turbine_rope_desc *d);
const char *turbine_rope_impl(const turbine_rope_desc *d);

int32_t turbine_silu_mul(turbine_ctx *ctx, const turbine_silu_mul_desc *d);
int32_t turbine_silu_mul_supported(const turbine_silu_mul_desc *d);
const char *turbine_silu_mul_impl(const turbine_silu_mul_desc *d);

int32_t turbine_embedding(turbine_ctx *ctx, const turbine_embedding_desc *d);
int32_t turbine_embedding_supported(const turbine_embedding_desc *d);
const char *turbine_embedding_impl(const turbine_embedding_desc *d);

int32_t turbine_add(turbine_ctx *ctx, const turbine_add_desc *d);
int32_t turbine_add_supported(const turbine_add_desc *d);
const char *turbine_add_impl(const turbine_add_desc *d);

/* ======== v2: context info, paged attention, block copy, MoE ======== */

/* Fixed properties of a context; out is a host struct. */
typedef struct turbine_ctx_info {
  /* fixed at context creation */
  uint64_t workspace_bytes;
  /* -1 when not applicable */
  int32_t compute_major, compute_minor;
  /* NUL-terminated device architecture name */
  char device_arch[32];
} turbine_ctx_info;
int32_t turbine_ctx_get_info(turbine_ctx *ctx, turbine_ctx_info *out);

/* Ragged batch over one layer of the paged KV pool: appends k_new/v_new into
 * their page slots, then attends. Sequence s owns query rows
 * [q_indptr[s], q_indptr[s+1]) and, after the append, kv_lens[s] tokens; its
 * new tokens sit at positions [kv_lens[s] - q_len, kv_lens[s]). Token p lives
 * in block block_table[s][p / block_tokens] at slot p % block_tokens. */
typedef struct turbine_attention_paged_desc {
  /* [total_q, num_q_heads, head_dim] */
  const void *q;
  /* [total_q, num_kv_heads, head_dim] */
  const void *k_new;
  const void *v_new;
  /* [total_q, num_q_heads, head_dim] */
  void *out;
  /* this layer's pool [num_blocks, 2, block_tokens, num_kv_heads, head_dim] */
  void *kv_layer;
  /* [num_seqs, max_blocks_per_seq] */
  const int32_t *block_table;
  /* [num_seqs + 1] */
  const int32_t *q_indptr;
  /* [num_seqs], after the append */
  const int32_t *kv_lens;
  int32_t num_seqs, total_q, max_q_len, max_kv_len, max_blocks_per_seq,
      num_blocks, block_tokens;
  int32_t num_q_heads, num_kv_heads, head_dim;
  int64_t q_stride_token, new_stride_token, out_stride_token;
  float scale;
  int32_t causal, dtype;
} turbine_attention_paged_desc;
typedef turbine_attention_paged_desc turbine_attention_prefill_paged_desc;
typedef turbine_attention_paged_desc turbine_attention_decode_paged_desc;

/* Forks blocks (n > 1) across all layers: for each i, block src_blocks[i] is
 * copied to dst_blocks[i] in every layer. Layer l's block b is the block_bytes
 * bytes at pool + l * layer_stride_bytes + b * block_bytes. */
typedef struct turbine_copy_blocks_desc {
  void *pool;
  /* block_bytes = per-layer block size */
  int64_t layer_stride_bytes, block_bytes;
  int32_t num_layers;
  /* host arrays [count] */
  const int32_t *src_blocks;
  const int32_t *dst_blocks;
  int32_t count;
} turbine_copy_blocks_desc;

/* turbine_moe_route_desc flags. RENORMALIZE divides the selected weights by
 * their sum (without it they stay the softmax weights). BF16_LOGITS (v2.2)
 * rounds each logit to BF16 (round to nearest even) before the softmax. */
#define TURBINE_MOE_ROUTE_RENORMALIZE 1
#define TURBINE_MOE_ROUTE_BF16_LOGITS 2

/* Softmax in F32, then the top_k experts PyTorch's CPU torch.topk selects:
 * libstdc++ std::nth_element (introselect) at top_k - 1 over (weight, id) in
 * id order, or std::partial_sort (heap select) when top_k * 64 <= num_experts,
 * comparing weights only (NaN first), so a tie at the top_k-th place does not
 * always keep the lower id; then the permutation. */
typedef struct turbine_moe_route_desc {
  /* [num_tokens, num_experts] F32 */
  const float *router_logits;
  /* flags: TURBINE_MOE_ROUTE_* bits (the field was named renormalize before
   * v2.2; its values 0 and 1 keep their meaning) */
  int32_t num_tokens, num_experts, top_k, flags;
  /* [num_tokens, top_k], per token in descending weight (ties: lower id) */
  int32_t *topk_ids;
  float *topk_weights;
  /* [num_tokens * top_k]: rows (token * top_k + slot) grouped by expert */
  int32_t *sorted_rows;
  /* [num_experts + 1] */
  int32_t *expert_offsets;
} turbine_moe_route_desc;

/* out += sum_k w_k * down(silu(gate(x)) * up(x)) over the local experts,
 * accumulated in ascending expert order. */
typedef struct turbine_moe_experts_desc {
  /* [num_tokens, hidden] */
  const void *x;
  /* [num_local_experts, inter, hidden] */
  const void *w_gate;
  const void *w_up;
  /* [num_local_experts, hidden, inter] */
  const void *w_down;
  const int32_t *sorted_rows;
  const int32_t *expert_offsets;
  const float *topk_weights;
  /* host copy [num_experts + 1], for group sizes */
  const int32_t *host_expert_offsets;
  /* [num_tokens, hidden], accumulated */
  void *out;
  void *workspace;
  size_t workspace_bytes;
  int32_t num_tokens, hidden, inter, top_k, num_experts;
  /* local expert range [expert_begin, expert_end) */
  int32_t expert_begin, expert_end;
  int32_t dtype;
} turbine_moe_experts_desc;

/* v2 trios: attention_prefill_paged, attention_decode_paged, copy_blocks,
 * moe_route, moe_experts. */
int32_t
turbine_attention_prefill_paged(turbine_ctx *ctx,
                                const turbine_attention_prefill_paged_desc *d);
int32_t turbine_attention_prefill_paged_supported(
    const turbine_attention_prefill_paged_desc *d);
const char *turbine_attention_prefill_paged_impl(
    const turbine_attention_prefill_paged_desc *d);

int32_t
turbine_attention_decode_paged(turbine_ctx *ctx,
                               const turbine_attention_decode_paged_desc *d);
int32_t turbine_attention_decode_paged_supported(
    const turbine_attention_decode_paged_desc *d);
const char *turbine_attention_decode_paged_impl(
    const turbine_attention_decode_paged_desc *d);

int32_t turbine_copy_blocks(turbine_ctx *ctx,
                            const turbine_copy_blocks_desc *d);
int32_t turbine_copy_blocks_supported(const turbine_copy_blocks_desc *d);
const char *turbine_copy_blocks_impl(const turbine_copy_blocks_desc *d);

int32_t turbine_moe_route(turbine_ctx *ctx, const turbine_moe_route_desc *d);
int32_t turbine_moe_route_supported(const turbine_moe_route_desc *d);
const char *turbine_moe_route_impl(const turbine_moe_route_desc *d);

int32_t turbine_moe_experts(turbine_ctx *ctx,
                            const turbine_moe_experts_desc *d);
int32_t turbine_moe_experts_supported(const turbine_moe_experts_desc *d);
const char *turbine_moe_experts_impl(const turbine_moe_experts_desc *d);

/* Optional (a library may omit the symbol; the caller then always passes
 * host_expert_offsets): 1 when turbine_moe_experts reads host_expert_offsets
 * for a descriptor of this shape and num_tokens, 0 when it reads the group
 * sizes on the device only and host_expert_offsets may be NULL, so the caller
 * needs no device-to-host copy after turbine_moe_route. Like _supported it
 * ignores pointer fields and needs no context. */
int32_t
turbine_moe_experts_needs_host_offsets(const turbine_moe_experts_desc *d);

/* ======== v2.1 (additive, optional): minor version, context options, fused
 * ops, graphs ========
 * Every symbol below is optional: turbine-kernels resolves them only when
 * turbine_abi_minor() >= 1, and a library without them runs the ABI v2 paths
 * (add then rmsnorm, whole logits rows, eager launches). */
/* v2.2 adds TURBINE_MOE_ROUTE_BF16_LOGITS (a flag bit, no new symbol); v2.3
 * adds pinned host memory and events; v2.4 implementation enumeration and the
 * card profile; v2.5 copy streams and asynchronous copies (all below). */
#define TURBINE_ABI_MINOR 5u
uint32_t turbine_abi_minor(void);

/* Context options (int64 values). Unknown options return
 * TURBINE_E_UNSUPPORTED. */
/* 1 = GEMM shapes the library's tuned algorithm table for the device's card
 * has a row for run that row's pinned algorithm (default 1), 0 = the first
 * heuristic answer for every shape */
#define TURBINE_OPTION_GEMM_AUTOTUNE 1
/* read only: number of GEMM shapes run so far on this context that use a
 * pinned algorithm of the tuned table */
#define TURBINE_OPTION_GEMM_TUNED_SHAPES 2
/* 1 = the GEMMs that follow belong to a step that prefills prompt tokens: a
 * shape whose tuned table has row-invariant rows runs them, so a row's bits do
 * not depend on how many rows share the call (a prefix-reused prefill of a
 * prompt's suffix gives the rows of the whole-prompt prefill); 0 (default) =
 * a decode step: the table's speed rows where the shape has them. Additive in
 * ABI v2.5 (a library without it answers TURBINE_E_UNSUPPORTED). */
#define TURBINE_OPTION_GEMM_PREFILL 3
int32_t turbine_ctx_set_option(turbine_ctx *ctx, int32_t option, int64_t value);
/* out is a host int64_t. */
int32_t turbine_ctx_get_option(turbine_ctx *ctx, int32_t option, int64_t *out);

/* Per row r: residual[r] = round(residual[r] + x[r]) to dtype, in place; then
 * out[r] = rmsnorm(residual[r]) * weight, equal to add followed by rmsnorm on
 * the rounded sum. */
typedef struct turbine_add_rmsnorm_desc {
  /* [rows, dim], updated in place */
  void *residual;
  /* [rows, dim] */
  const void *x;
  /* [dim] */
  const void *weight;
  /* [rows, dim] */
  void *out;
  int64_t rows, dim, residual_stride_row, x_stride_row, out_stride_row;
  float eps;
  int32_t dtype;
} turbine_add_rmsnorm_desc;

/* Per logits row r (F32 [rows, vocab], row stride stride_row): lse[r] = the
 * log-sum-exp of the raw logits (NaN ignored); top_ids[r] / top_values[r] =
 * the top_n (<= 64) largest raw logits and their ids, descending, ties to the
 * lower id, NaN last; when mode[r] = 1, sampled[r] is one categorical draw at
 * temperature[r] with the uniform[r] (argmax when temperature[r] <= 0 or no
 * logit is finite) and sampled_logit[r] its raw logit; mode[r] = 0 leaves
 * sampled[r] = -1 and sampled_logit[r] = NaN. The draw's weights are
 * w = exp(logit / temperature[r] - max) (NaN weighs 0):
 *   - top_p[r] >= 1 (or top_p NULL): the smallest id whose cumulative sum of w
 *     in id order exceeds uniform[r] * total;
 *   - top_p[r] < 1 (nucleus): over the ids in descending logit order (ties to
 *     the lower id), keep the shortest prefix whose sum of w reaches
 *     top_p[r] * total (at least one id), then take the first id of that
 *     prefix whose cumulative sum exceeds uniform[r] * the prefix's sum.
 * All buffers are device buffers; the per-row arrays are [rows], top_ids and
 * top_values [rows, top_n]. */
typedef struct turbine_logits_reduce_desc {
  const float *logits;
  int64_t rows, vocab, stride_row;
  const float *temperature;
  /* in [0, 1) */
  const float *uniform;
  /* 0 = reduce only, 1 = also draw a categorical sample */
  const int32_t *mode;
  int32_t top_n;
  int32_t *top_ids;
  float *top_values;
  float *lse;
  int32_t *sampled;
  float *sampled_logit;
  /* [rows] in (0, 1], or NULL for 1 on every row: the draw's nucleus mass */
  const float *top_p;
} turbine_logits_reduce_desc;

/* v2.1 trios: add_rmsnorm, logits_reduce. */
int32_t turbine_add_rmsnorm(turbine_ctx *ctx,
                            const turbine_add_rmsnorm_desc *d);
int32_t turbine_add_rmsnorm_supported(const turbine_add_rmsnorm_desc *d);
const char *turbine_add_rmsnorm_impl(const turbine_add_rmsnorm_desc *d);

int32_t turbine_logits_reduce(turbine_ctx *ctx,
                              const turbine_logits_reduce_desc *d);
int32_t turbine_logits_reduce_supported(const turbine_logits_reduce_desc *d);
const char *turbine_logits_reduce_impl(const turbine_logits_reduce_desc *d);

/* Graphs: turbine_graph_begin starts capturing the compute stream;
 * turbine_graph_end stops and instantiates the captured work into *out;
 * turbine_graph_launch enqueues it on the compute stream;
 * turbine_graph_destroy releases it. Between begin and end only op calls are
 * allowed: turbine_memcpy_*, turbine_stream_sync, turbine_malloc and
 * turbine_free return TURBINE_E_ARGUMENT while capturing. A failed capture
 * leaves the context usable. A graph records the device pointers its ops were
 * captured with: the caller keeps those buffers alive and destroys the graph
 * before freeing them. */
typedef struct turbine_graph turbine_graph;
int32_t turbine_graph_begin(turbine_ctx *ctx);
int32_t turbine_graph_end(turbine_ctx *ctx, turbine_graph **out);
int32_t turbine_graph_launch(turbine_ctx *ctx, turbine_graph *g);
int32_t turbine_graph_destroy(turbine_ctx *ctx, turbine_graph *g);

/* ======== v2.3 (additive, optional): pinned host memory and events ========
 * The compute-stream subset of the Phase 4 memory functions (v2.5 adds the
 * rest). Resolved only when turbine_abi_minor() >= 3 and all six symbols
 * exist; a library without them keeps the ABI v2 copies, which
 * turbine-kernels follows with turbine_stream_sync.
 *
 * turbine_host_alloc_pinned returns bytes of page-locked host memory (a host
 * pointer, never NULL on success): turbine_memcpy_h2d / turbine_memcpy_d2h
 * with it as the host side are enqueued on the compute stream and return
 * without waiting for earlier work or for the copy. The caller keeps the bytes
 * unchanged (h2d) or unread (d2h) until an event recorded after the copy has
 * completed, and frees the memory with turbine_host_free_pinned only after
 * that.
 *
 * An event marks a point of a stream: turbine_event_record captures the work
 * enqueued on stream s so far (s NULL = the compute stream, the only stream
 * before v2.5; re-recording moves the mark); turbine_event_synchronize blocks
 * the calling thread until that work has completed, not later work; an event
 * never recorded is complete. */
typedef struct turbine_stream turbine_stream;
typedef struct turbine_event turbine_event;
int32_t turbine_host_alloc_pinned(turbine_ctx *ctx, size_t bytes, void **out);
int32_t turbine_host_free_pinned(turbine_ctx *ctx, void *ptr);
int32_t turbine_event_create(turbine_ctx *ctx, turbine_event **out);
int32_t turbine_event_destroy(turbine_ctx *ctx, turbine_event *e);
int32_t turbine_event_record(turbine_ctx *ctx, turbine_event *e,
                             turbine_stream *s);
int32_t turbine_event_synchronize(turbine_ctx *ctx, turbine_event *e);

/* ======== v2.4 (additive, optional): implementation enumeration and card
 * profile ========
 * Resolved only when turbine_abi_minor() >= 4 and all five functions exist; a
 * library without them keeps choosing the implementation of every
 * turbine_<op> call itself.
 *
 * Each op has one or more implementations, indexed 0..turbine_impl_count(op)
 * in the library's order. The caller asks which of them supports a descriptor
 * (turbine_impl_supports: like turbine_<op>_supported, it ignores pointer
 * fields and needs no context), picks one, and runs exactly that one with
 * turbine_impl_run. The turbine_<op> entry points stay: they run the first
 * implementation, in library order, that supports the descriptor, within the
 * row tiers of the context's card profile (moe_experts: the implementations
 * marked for small row counts serve up to moe_small_max_rows routed rows).
 * Op codes follow the order of the op trios in this header. */
#define TURBINE_OP_GEMM 0
#define TURBINE_OP_ATTENTION_PREFILL 1
#define TURBINE_OP_ATTENTION_DECODE 2
#define TURBINE_OP_RMSNORM 3
#define TURBINE_OP_ROPE 4
#define TURBINE_OP_SILU_MUL 5
#define TURBINE_OP_EMBEDDING 6
#define TURBINE_OP_ADD 7
#define TURBINE_OP_ATTENTION_PREFILL_PAGED 8
#define TURBINE_OP_ATTENTION_DECODE_PAGED 9
#define TURBINE_OP_COPY_BLOCKS 10
#define TURBINE_OP_MOE_ROUTE 11
#define TURBINE_OP_MOE_EXPERTS 12
#define TURBINE_OP_ADD_RMSNORM 13
#define TURBINE_OP_LOGITS_REDUCE 14
/* the implementation reads host_expert_offsets (moe_experts) */
#define TURBINE_IMPL_NEEDS_HOST_OFFSETS 1u
typedef struct turbine_impl_entry {
  /* static storage, e.g. the names turbine_<op>_impl returns */
  const char *name;
  /* implementation family, static storage */
  const char *provider;
  /* TURBINE_IMPL_* */
  uint32_t flags;
} turbine_impl_entry;
typedef struct turbine_card_profile {
  /* sizeof(turbine_card_profile) */
  uint32_t struct_bytes;
  /* host string; one of turbine_build_archs() */
  const char *arch;
  int32_t wave_size;
  int32_t lds_bytes;
  /* routed rows of the first moe_experts tier */
  int64_t moe_small_max_rows;
  int32_t paged_page_multiple;
} turbine_card_profile;
/* Number of implementations of op (>= 1), or TURBINE_E_ARGUMENT for an unknown
 * op. */
int32_t turbine_impl_count(int32_t op);
/* out is a host struct; TURBINE_E_ARGUMENT for an unknown op or index. */
int32_t turbine_impl_info(int32_t op, int32_t index, turbine_impl_entry *out);
/* 1 supported, 0 not, < 0 error; desc is the op's descriptor; pointer fields
 * ignored; no context. */
int32_t turbine_impl_supports(int32_t op, int32_t index, const void *desc);
/* Runs implementation index of op on ctx's compute stream;
 * TURBINE_E_UNSUPPORTED when it does not support desc. */
int32_t turbine_impl_run(turbine_ctx *ctx, int32_t op, int32_t index,
                         const void *desc);
/* Copies *p into the context (the defaults of turbine_<op> read it);
 * TURBINE_E_UNSUPPORTED when wave_size differs from the compiled wave size,
 * lds_bytes is below the library's largest static LDS use, or arch is not a
 * build arch. */
int32_t turbine_ctx_set_profile(turbine_ctx *ctx,
                                const turbine_card_profile *p);

/* ======== v2.5 (additive, optional): copy streams and asynchronous copies
 * ========
 * The rest of the Phase 4 memory functions, as an optional group on top of
 * v2.3 (the tiered KV cache moves blocks between device memory and pinned host
 * memory without stalling the compute stream). Resolved only when
 * turbine_abi_minor() >= 5, all five symbols exist and the v2.3 group is
 * resolved; a library without them runs with the KV cache on the device only.
 *
 * A copy stream belongs to the caller from turbine_copy_stream_create until
 * turbine_copy_stream_destroy, which waits for its copies and must come before
 * turbine_ctx_destroy. turbine_memcpy_async enqueues one copy of kind
 * TURBINE_COPY_* on stream s (NULL = the compute stream); the host side must be
 * pinned memory of the same context. It only enqueues: neither buffer may be
 * freed, and the pinned side may not be read or written by the host, until an
 * event recorded on s after the copy has completed. From v2.5
 * turbine_event_record accepts a copy stream as s. turbine_event_query returns
 * 1 when the recorded work has completed (or the event was never recorded), 0
 * while it is pending, < 0 on error, and never blocks.
 * turbine_stream_wait_event makes work enqueued on s after the call wait until
 * e's recorded work has completed, without blocking the host. */
#define TURBINE_COPY_H2D 0
#define TURBINE_COPY_D2H 1
#define TURBINE_COPY_D2D 2
int32_t turbine_copy_stream_create(turbine_ctx *ctx, turbine_stream **out);
int32_t turbine_copy_stream_destroy(turbine_ctx *ctx, turbine_stream *s);
int32_t turbine_memcpy_async(turbine_ctx *ctx, turbine_stream *s, void *dst,
                             const void *src, size_t bytes, int32_t kind);
int32_t turbine_event_query(turbine_ctx *ctx, turbine_event *e);
int32_t turbine_stream_wait_event(turbine_ctx *ctx, turbine_stream *s,
                                  turbine_event *e);

#ifdef __cplusplus
}
#endif
#endif /* TURBINE_KERNELS_H */
