/* Turbine vendor-neutral kernel C ABI, version 1 (contract section 9).
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
 * turbine_stream_sync is the only blocking call. Host buffers passed to
 * turbine_memcpy_h2d / turbine_memcpy_d2h must stay valid until the next
 * turbine_stream_sync.
 *
 * Layout. Row-major everywhere; strides are in elements; "leading dimension"
 * is the row stride in elements.
 *
 * Symbols. Only turbine_* symbols are exported.
 *
 * Ownership. Every device pointer is allocated by the library (turbine_malloc)
 * and freed exactly once by its owner (turbine_free). The library never retains
 * a caller pointer beyond the call. The context owns its streams, library
 * handles and workspace and releases them in turbine_ctx_destroy.
 *
 * Errors. turbine_last_error(ctx, buf, len) copies the NUL-terminated message
 * of the most recent failure on ctx (with ctx == NULL: of the most recent
 * failed turbine_ctx_create on the calling thread) into the host buffer buf and
 * returns the full message length excluding the NUL. Device runtime messages
 * start with the runtime's own error name. */
#ifndef TURBINE_KERNELS_H
#define TURBINE_KERNELS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TURBINE_ABI_VERSION 1u

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

#ifdef __cplusplus
}
#endif
#endif /* TURBINE_KERNELS_H */
