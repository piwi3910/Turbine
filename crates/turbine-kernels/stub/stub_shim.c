/* Test-only stand-in for a kernel shim library (turbine-kernels shim tests).
 *
 * Implements every ABI v1 symbol of kernels/include/turbine_kernels.h without
 * a GPU: "device" memory is host heap memory, copies are memcpy, the stream is
 * always idle, and every op reports TURBINE_E_UNSUPPORTED. build.rs compiles
 * it with the host C compiler once per variant:
 *   -DSTUB_ABI=<n>u  -DSTUB_BACKEND="<name>"  -DSTUB_ARCHS="<a,b>"
 *
 * stub_live_contexts() is a test hook (not part of the ABI): the number of
 * contexts created and not yet destroyed, so tests can prove the Rust side
 * destroys each context exactly once. */
#include "turbine_kernels.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define STUB_TOTAL_BYTES ((size_t)1 << 30)
#define STUB_ERROR_LEN 256

struct turbine_ctx {
  int32_t ordinal;
  char last_error[STUB_ERROR_LEN];
};

static atomic_int live_contexts;
/* Message of the most recent failed turbine_ctx_create on this thread. */
static _Thread_local char create_error[STUB_ERROR_LEN];

static void set_error(char *dst, const char *msg) {
  strncpy(dst, msg, STUB_ERROR_LEN - 1);
  dst[STUB_ERROR_LEN - 1] = '\0';
}

int32_t stub_live_contexts(void) { return atomic_load(&live_contexts); }

/* Test hook: sizeof each op descriptor, indexed in header order (gemm,
 * attention, rmsnorm, rope, silu_mul, embedding, add); 0 past the end. */
size_t stub_desc_size(int32_t which) {
  switch (which) {
  case 0:
    return sizeof(turbine_gemm_desc);
  case 1:
    return sizeof(turbine_attention_desc);
  case 2:
    return sizeof(turbine_rmsnorm_desc);
  case 3:
    return sizeof(turbine_rope_desc);
  case 4:
    return sizeof(turbine_silu_mul_desc);
  case 5:
    return sizeof(turbine_embedding_desc);
  case 6:
    return sizeof(turbine_add_desc);
  default:
    return 0;
  }
}

uint32_t turbine_abi_version(void) { return STUB_ABI; }
const char *turbine_backend_name(void) { return STUB_BACKEND; }
const char *turbine_build_archs(void) { return STUB_ARCHS; }

int32_t turbine_ctx_create(int32_t device_ordinal, turbine_ctx **out) {
  if (out == NULL || device_ordinal < 0) {
    set_error(create_error, "stub: invalid device ordinal");
    return TURBINE_E_ARGUMENT;
  }
  turbine_ctx *ctx = calloc(1, sizeof *ctx);
  if (ctx == NULL) {
    set_error(create_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  ctx->ordinal = device_ordinal;
  atomic_fetch_add(&live_contexts, 1);
  *out = ctx;
  return TURBINE_OK;
}

void turbine_ctx_destroy(turbine_ctx *ctx) {
  if (ctx != NULL) {
    atomic_fetch_sub(&live_contexts, 1);
    free(ctx);
  }
}

int32_t turbine_malloc(turbine_ctx *ctx, size_t bytes, void **out) {
  if (bytes > STUB_TOTAL_BYTES) {
    set_error(ctx->last_error, "stub: out of device memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  *out = malloc(bytes == 0 ? 1 : bytes);
  if (*out == NULL) {
    set_error(ctx->last_error, "stub: out of device memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  return TURBINE_OK;
}

int32_t turbine_free(turbine_ctx *ctx, void *ptr) {
  (void)ctx;
  free(ptr);
  return TURBINE_OK;
}

int32_t turbine_memcpy_h2d(turbine_ctx *ctx, void *dst_device,
                           const void *src_host, size_t bytes) {
  (void)ctx;
  memcpy(dst_device, src_host, bytes);
  return TURBINE_OK;
}

int32_t turbine_memcpy_d2h(turbine_ctx *ctx, void *dst_host,
                           const void *src_device, size_t bytes) {
  (void)ctx;
  memcpy(dst_host, src_device, bytes);
  return TURBINE_OK;
}

int32_t turbine_stream_sync(turbine_ctx *ctx) {
  (void)ctx;
  return TURBINE_OK;
}

int32_t turbine_mem_info(turbine_ctx *ctx, size_t *free_bytes,
                         size_t *total_bytes) {
  (void)ctx;
  *free_bytes = STUB_TOTAL_BYTES;
  *total_bytes = STUB_TOTAL_BYTES;
  return TURBINE_OK;
}

size_t turbine_last_error(turbine_ctx *ctx, char *buf, size_t len) {
  const char *msg = ctx != NULL ? ctx->last_error : create_error;
  size_t n = strlen(msg);
  if (len > 0) {
    size_t copy = n < len - 1 ? n : len - 1;
    memcpy(buf, msg, copy);
    buf[copy] = '\0';
  }
  return n;
}

#define STUB_OP(op, desc)                                                      \
  int32_t turbine_##op(turbine_ctx *ctx, const desc *d) {                      \
    (void)d;                                                                   \
    set_error(ctx->last_error, "stub: " #op " is not implemented");            \
    return TURBINE_E_UNSUPPORTED;                                              \
  }                                                                            \
  int32_t turbine_##op##_supported(const desc *d) {                            \
    (void)d;                                                                   \
    return 0;                                                                  \
  }                                                                            \
  const char *turbine_##op##_impl(const desc *d) {                             \
    (void)d;                                                                   \
    return "stub_" #op;                                                        \
  }

STUB_OP(gemm, turbine_gemm_desc)
STUB_OP(attention_prefill, turbine_attention_prefill_desc)
STUB_OP(attention_decode, turbine_attention_decode_desc)
STUB_OP(rmsnorm, turbine_rmsnorm_desc)
STUB_OP(rope, turbine_rope_desc)
STUB_OP(silu_mul, turbine_silu_mul_desc)
STUB_OP(embedding, turbine_embedding_desc)
STUB_OP(add, turbine_add_desc)
