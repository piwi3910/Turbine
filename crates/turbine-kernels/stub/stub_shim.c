/* Test-only stand-in for a kernel shim library (turbine-kernels shim tests).
 *
 * Implements every ABI v2 symbol of kernels/include/turbine_kernels.h without
 * a GPU: "device" memory is host heap memory, copies are memcpy, the stream is
 * always idle, and every op reports TURBINE_E_UNSUPPORTED. build.rs compiles
 * it with the host C compiler once per variant:
 *   -DSTUB_ABI=<n>u  -DSTUB_BACKEND="<name>"  -DSTUB_ARCHS="<a,b>"
 *   [-DTURBINE_STUB_V21] [-DTURBINE_STUB_V24] [-DTURBINE_STUB_V25]
 *   [-DTURBINE_STUB_V26] [-DTURBINE_STUB_V27]
 * With TURBINE_STUB_V21 it also exports the optional ABI v2.1 and v2.3
 * symbols: turbine_abi_minor (3, a v2.3 library), context options
 * (TURBINE_OPTION_GEMM_AUTOTUNE kept per context,
 * TURBINE_OPTION_GEMM_TUNED_SHAPES always 0), the add_rmsnorm and
 * logits_reduce trios (unsupported like every op), graphs that record
 * nothing (stub_live_graphs() counts graphs not yet destroyed), and host
 * staging memory and events (see the v2.3 section below). With
 * TURBINE_STUB_V24 as well it reports minor 4 and exports the v2.4
 * implementation group (see the v2.4 section); with TURBINE_STUB_V25 as well
 * it reports minor 5 and exports the v2.5 copy streams (see the v2.5
 * section); with TURBINE_STUB_V26 as well it reports minor 6 and exports the
 * v2.6 group (see the v2.6 section); with TURBINE_STUB_V27 as well it reports
 * minor TURBINE_ABI_MINOR (7) and exports the v2.7 host-mapped group, whose
 * collective runs the protocol on the calling thread (see the v2.7 section at
 * the end).
 *
 * stub_live_contexts() is a test hook (not part of the ABI): the number of
 * contexts created and not yet destroyed, so tests can prove the Rust side
 * destroys each context exactly once. */
/* clock_gettime and nanosleep for the v2.7 section. */
#define _POSIX_C_SOURCE 200809L
#include "turbine_kernels.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define STUB_TOTAL_BYTES ((size_t)1 << 30)
#define STUB_ERROR_LEN 256

struct turbine_ctx {
  int32_t ordinal;
  char last_error[STUB_ERROR_LEN];
  /* v2.1 */
  int64_t gemm_autotune;
  int32_t capturing;
  /* v2.4: the moe_small_max_rows of the last turbine_ctx_set_profile, -1
   * before one */
  int64_t profile_small_rows;
  /* v2.6: its address is the compute stream's native handle */
  char compute_stream_tag;
};

static atomic_int live_contexts;
/* Message of the most recent failed turbine_ctx_create on this thread. */
static _Thread_local char create_error[STUB_ERROR_LEN];

static void set_error(char *dst, const char *msg) {
  strncpy(dst, msg, STUB_ERROR_LEN - 1);
  dst[STUB_ERROR_LEN - 1] = '\0';
}

int32_t stub_live_contexts(void) { return atomic_load(&live_contexts); }

/* Test hook: sizeof each descriptor, indexed in header order (gemm,
 * attention, rmsnorm, rope, silu_mul, embedding, add, then v2: ctx_info,
 * attention_paged, copy_blocks, moe_route, moe_experts, then v2.1:
 * add_rmsnorm, logits_reduce, then v2.6: row_sumsq, rmsnorm_sharded, then
 * v2.7: mapped_collective); 0 past the end. */
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
  case 7:
    return sizeof(turbine_ctx_info);
  case 8:
    return sizeof(turbine_attention_paged_desc);
  case 9:
    return sizeof(turbine_copy_blocks_desc);
  case 10:
    return sizeof(turbine_moe_route_desc);
  case 11:
    return sizeof(turbine_moe_experts_desc);
  case 12:
    return sizeof(turbine_add_rmsnorm_desc);
  case 13:
    return sizeof(turbine_logits_reduce_desc);
  case 14:
    return sizeof(turbine_row_sumsq_desc);
  case 15:
    return sizeof(turbine_rmsnorm_sharded_desc);
  case 16:
    return sizeof(turbine_mapped_collective_desc);
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
  ctx->gemm_autotune = 1;
  ctx->profile_small_rows = -1;
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

/* Workspace 1 MiB, no compute capability, device arch = first build arch. */
int32_t turbine_ctx_get_info(turbine_ctx *ctx, turbine_ctx_info *out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null info pointer");
    return TURBINE_E_ARGUMENT;
  }
  memset(out, 0, sizeof *out);
  out->workspace_bytes = (uint64_t)1 << 20;
  out->compute_major = -1;
  out->compute_minor = -1;
  const char *archs = STUB_ARCHS;
  size_t n = strcspn(archs, ",");
  if (n > sizeof out->device_arch - 1) {
    n = sizeof out->device_arch - 1;
  }
  memcpy(out->device_arch, archs, n);
  return TURBINE_OK;
}

size_t turbine_last_error(turbine_ctx *ctx, char *buf, size_t len) {
  const char *msg = ctx != NULL ? ctx->last_error : create_error;
  size_t n = strlen(msg);
  if (buf != NULL && len > 0) {
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
STUB_OP(attention_prefill_paged, turbine_attention_prefill_paged_desc)
STUB_OP(attention_decode_paged, turbine_attention_decode_paged_desc)
STUB_OP(copy_blocks, turbine_copy_blocks_desc)
STUB_OP(moe_route, turbine_moe_route_desc)
STUB_OP(moe_experts, turbine_moe_experts_desc)

#ifdef TURBINE_STUB_V21
STUB_OP(add_rmsnorm, turbine_add_rmsnorm_desc)
STUB_OP(logits_reduce, turbine_logits_reduce_desc)

#if defined(TURBINE_STUB_V27)
uint32_t turbine_abi_minor(void) { return TURBINE_ABI_MINOR; }
#elif defined(TURBINE_STUB_V26)
uint32_t turbine_abi_minor(void) { return 6u; }
#elif defined(TURBINE_STUB_V25)
uint32_t turbine_abi_minor(void) { return 5u; }
#elif defined(TURBINE_STUB_V24)
uint32_t turbine_abi_minor(void) { return 4u; }
#else
uint32_t turbine_abi_minor(void) { return 3u; }
#endif

int32_t turbine_ctx_set_option(turbine_ctx *ctx, int32_t option,
                               int64_t value) {
  switch (option) {
  case TURBINE_OPTION_GEMM_AUTOTUNE:
    if (value != 0 && value != 1) {
      set_error(ctx->last_error, "stub: gemm autotune takes 0 or 1");
      return TURBINE_E_ARGUMENT;
    }
    ctx->gemm_autotune = value;
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_TUNED_SHAPES:
    set_error(ctx->last_error, "stub: tuned shapes is read only");
    return TURBINE_E_ARGUMENT;
  default:
    set_error(ctx->last_error, "stub: unknown option");
    return TURBINE_E_UNSUPPORTED;
  }
}

int32_t turbine_ctx_get_option(turbine_ctx *ctx, int32_t option, int64_t *out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null option pointer");
    return TURBINE_E_ARGUMENT;
  }
  switch (option) {
  case TURBINE_OPTION_GEMM_AUTOTUNE:
    *out = ctx->gemm_autotune;
    return TURBINE_OK;
  case TURBINE_OPTION_GEMM_TUNED_SHAPES:
    *out = 0;
    return TURBINE_OK;
  default:
    set_error(ctx->last_error, "stub: unknown option");
    return TURBINE_E_UNSUPPORTED;
  }
}

struct turbine_graph {
  turbine_ctx *ctx;
};

static atomic_int live_graphs;

int32_t stub_live_graphs(void) { return atomic_load(&live_graphs); }

int32_t turbine_graph_begin(turbine_ctx *ctx) {
  if (ctx->capturing) {
    set_error(ctx->last_error, "stub: already capturing");
    return TURBINE_E_ARGUMENT;
  }
  ctx->capturing = 1;
  return TURBINE_OK;
}

int32_t turbine_graph_end(turbine_ctx *ctx, turbine_graph **out) {
  if (!ctx->capturing || out == NULL) {
    set_error(ctx->last_error, "stub: graph_end without graph_begin");
    return TURBINE_E_ARGUMENT;
  }
  ctx->capturing = 0;
  turbine_graph *g = calloc(1, sizeof *g);
  if (g == NULL) {
    set_error(ctx->last_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  g->ctx = ctx;
  atomic_fetch_add(&live_graphs, 1);
  *out = g;
  return TURBINE_OK;
}

int32_t turbine_graph_launch(turbine_ctx *ctx, turbine_graph *g) {
  if (g == NULL || g->ctx != ctx) {
    set_error(ctx->last_error, "stub: graph of another context");
    return TURBINE_E_ARGUMENT;
  }
  return TURBINE_OK;
}

int32_t turbine_graph_destroy(turbine_ctx *ctx, turbine_graph *g) {
  if (g == NULL || g->ctx != ctx) {
    set_error(ctx->last_error, "stub: graph of another context");
    return TURBINE_E_ARGUMENT;
  }
  atomic_fetch_sub(&live_graphs, 1);
  free(g);
  return TURBINE_OK;
}

/* v2.3: host staging memory is heap memory and events record nothing (the
 * stream is always idle). stub_live_host_buffers() and stub_live_events()
 * count what is not yet freed or destroyed; stub_event_syncs() counts
 * turbine_event_synchronize calls. */
struct turbine_event {
  turbine_ctx *ctx;
  /* v2.5: recorded at least once */
  int32_t recorded;
};

static atomic_int live_host_buffers;
static atomic_int live_events;
static atomic_int event_syncs;

int32_t stub_live_host_buffers(void) { return atomic_load(&live_host_buffers); }
int32_t stub_live_events(void) { return atomic_load(&live_events); }
int32_t stub_event_syncs(void) { return atomic_load(&event_syncs); }

int32_t turbine_host_alloc_pinned(turbine_ctx *ctx, size_t bytes, void **out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null host pointer");
    return TURBINE_E_ARGUMENT;
  }
  /* Page-locked memory is limited like the stub's device (`ulimit -l`). */
  *out = bytes > STUB_TOTAL_BYTES ? NULL : malloc(bytes == 0 ? 1 : bytes);
  if (*out == NULL) {
    set_error(ctx->last_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  atomic_fetch_add(&live_host_buffers, 1);
  return TURBINE_OK;
}

int32_t turbine_host_free_pinned(turbine_ctx *ctx, void *ptr) {
  (void)ctx;
  if (ptr != NULL) {
    atomic_fetch_sub(&live_host_buffers, 1);
    free(ptr);
  }
  return TURBINE_OK;
}

int32_t turbine_event_create(turbine_ctx *ctx, turbine_event **out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null event pointer");
    return TURBINE_E_ARGUMENT;
  }
  turbine_event *e = calloc(1, sizeof *e);
  if (e == NULL) {
    set_error(ctx->last_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  e->ctx = ctx;
  atomic_fetch_add(&live_events, 1);
  *out = e;
  return TURBINE_OK;
}

int32_t turbine_event_record(turbine_ctx *ctx, turbine_event *e,
                             turbine_stream *s) {
#ifdef TURBINE_STUB_V25
  const int32_t stream_ok = 1; /* v2.5: copy streams may be named */
#else
  const int32_t stream_ok = s == NULL;
#endif
  if (e == NULL || e->ctx != ctx || !stream_ok) {
    set_error(ctx->last_error, "stub: event of another context");
    return TURBINE_E_ARGUMENT;
  }
  e->recorded = 1;
  return TURBINE_OK;
}

int32_t turbine_event_synchronize(turbine_ctx *ctx, turbine_event *e) {
  if (e == NULL || e->ctx != ctx) {
    set_error(ctx->last_error, "stub: event of another context");
    return TURBINE_E_ARGUMENT;
  }
  atomic_fetch_add(&event_syncs, 1);
  return TURBINE_OK;
}

int32_t turbine_event_destroy(turbine_ctx *ctx, turbine_event *e) {
  if (e == NULL || e->ctx != ctx) {
    set_error(ctx->last_error, "stub: event of another context");
    return TURBINE_E_ARGUMENT;
  }
  atomic_fetch_sub(&live_events, 1);
  free(e);
  return TURBINE_OK;
}
#endif /* TURBINE_STUB_V21 */

#ifdef TURBINE_STUB_V24
/* v2.4: rmsnorm has two implementations, "stub_a" (provider "stub",
 * supports every descriptor) and "stub_b" (provider "stub_alt", refuses dim
 * 4096); every other op (the v2.6 ops only with TURBINE_STUB_V26) has one
 * implementation "stub_<op>" (provider "stub") that supports nothing, like
 * its turbine_<op>_supported. turbine_impl_run
 * records op * 100 + index for stub_last_impl_run() and does nothing else.
 * turbine_ctx_set_profile accepts a profile of a build arch and keeps its
 * moe_small_max_rows for stub_profile_small_rows(). */
static const char *const stub_op_names[] = {
    "stub_gemm",
    "stub_attention_prefill",
    "stub_attention_decode",
    "stub_rmsnorm",
    "stub_rope",
    "stub_silu_mul",
    "stub_embedding",
    "stub_add",
    "stub_attention_prefill_paged",
    "stub_attention_decode_paged",
    "stub_copy_blocks",
    "stub_moe_route",
    "stub_moe_experts",
    "stub_add_rmsnorm",
    "stub_logits_reduce",
#ifdef TURBINE_STUB_V26
    "stub_row_sumsq",
    "stub_rmsnorm_sharded",
#endif
};
#define STUB_OPS ((int32_t)(sizeof stub_op_names / sizeof stub_op_names[0]))

static atomic_int last_impl_run = -1;

int32_t stub_last_impl_run(void) { return atomic_load(&last_impl_run); }

int64_t stub_profile_small_rows(const turbine_ctx *ctx) {
  return ctx->profile_small_rows;
}

int32_t turbine_impl_count(int32_t op) {
  if (op < 0 || op >= STUB_OPS) {
    return TURBINE_E_ARGUMENT;
  }
  return op == TURBINE_OP_RMSNORM ? 2 : 1;
}

int32_t turbine_impl_info(int32_t op, int32_t index, turbine_impl_entry *out) {
  if (out == NULL || index < 0 || index >= turbine_impl_count(op)) {
    return TURBINE_E_ARGUMENT;
  }
  if (op == TURBINE_OP_RMSNORM) {
    out->name = index == 0 ? "stub_a" : "stub_b";
    out->provider = index == 0 ? "stub" : "stub_alt";
  } else {
    out->name = stub_op_names[op];
    out->provider = "stub";
  }
  out->flags = 0;
  return TURBINE_OK;
}

int32_t turbine_impl_supports(int32_t op, int32_t index, const void *desc) {
  if (desc == NULL || index < 0 || index >= turbine_impl_count(op)) {
    return TURBINE_E_ARGUMENT;
  }
  if (op != TURBINE_OP_RMSNORM) {
    return 0;
  }
  const turbine_rmsnorm_desc *d = desc;
  return index == 0 || d->dim != 4096 ? 1 : 0;
}

int32_t turbine_impl_run(turbine_ctx *ctx, int32_t op, int32_t index,
                         const void *desc) {
  int32_t supported = turbine_impl_supports(op, index, desc);
  if (supported < 0) {
    set_error(ctx->last_error, "stub: unknown implementation");
    return supported;
  }
  if (supported == 0) {
    set_error(ctx->last_error, "stub: implementation does not support desc");
    return TURBINE_E_UNSUPPORTED;
  }
  atomic_store(&last_impl_run, op * 100 + index);
  return TURBINE_OK;
}

int32_t turbine_ctx_set_profile(turbine_ctx *ctx,
                                const turbine_card_profile *p) {
  if (p == NULL || p->struct_bytes < sizeof *p || p->arch == NULL) {
    set_error(ctx->last_error, "stub: invalid card profile");
    return TURBINE_E_ARGUMENT;
  }
  /* The arch must be one of the comma-separated build archs. */
  const char *archs = STUB_ARCHS;
  size_t len = strlen(p->arch);
  for (const char *a = archs; *a != '\0';) {
    size_t n = strcspn(a, ",");
    if (n == len && strncmp(a, p->arch, n) == 0) {
      ctx->profile_small_rows = p->moe_small_max_rows;
      return TURBINE_OK;
    }
    a += n;
    if (*a == ',') {
      a++;
    }
  }
  set_error(ctx->last_error, "stub: profile arch is not a build arch");
  return TURBINE_E_UNSUPPORTED;
}
#endif /* TURBINE_STUB_V24 */

#ifdef TURBINE_STUB_V25
/* v2.5: copy streams are bookkeeping only and asynchronous copies are memcpy
 * (the stub has no device). stub_live_streams() counts streams not yet
 * destroyed; stub_hold_events(1) makes turbine_event_query report pending
 * until stub_hold_events(0); stub_stream_waits() counts
 * turbine_stream_wait_event calls. */
struct turbine_stream {
  turbine_ctx *ctx;
};

static atomic_int live_streams;
static atomic_int hold_events;
static atomic_int stream_waits;

int32_t stub_live_streams(void) { return atomic_load(&live_streams); }
void stub_hold_events(int32_t hold) { atomic_store(&hold_events, hold); }
int32_t stub_stream_waits(void) { return atomic_load(&stream_waits); }

int32_t turbine_copy_stream_create(turbine_ctx *ctx, turbine_stream **out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null stream pointer");
    return TURBINE_E_ARGUMENT;
  }
  turbine_stream *st = calloc(1, sizeof *st);
  if (st == NULL) {
    set_error(ctx->last_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  st->ctx = ctx;
  atomic_fetch_add(&live_streams, 1);
  *out = st;
  return TURBINE_OK;
}

int32_t turbine_copy_stream_destroy(turbine_ctx *ctx, turbine_stream *st) {
  if (st == NULL)
    return TURBINE_OK;
  if (st->ctx != ctx) {
    set_error(ctx->last_error, "stub: stream of another context");
    return TURBINE_E_ARGUMENT;
  }
  atomic_fetch_sub(&live_streams, 1);
  free(st);
  return TURBINE_OK;
}

int32_t turbine_memcpy_async(turbine_ctx *ctx, turbine_stream *st, void *dst,
                             const void *src, size_t bytes, int32_t kind) {
  if (st != NULL && st->ctx != ctx) {
    set_error(ctx->last_error, "stub: stream of another context");
    return TURBINE_E_ARGUMENT;
  }
  if (kind < TURBINE_COPY_H2D || kind > TURBINE_COPY_D2D) {
    set_error(ctx->last_error, "stub: unknown copy kind");
    return TURBINE_E_ARGUMENT;
  }
  if (bytes == 0)
    return TURBINE_OK;
  if (dst == NULL || src == NULL) {
    set_error(ctx->last_error, "stub: null copy pointer");
    return TURBINE_E_ARGUMENT;
  }
  memcpy(dst, src, bytes);
  return TURBINE_OK;
}

int32_t turbine_event_query(turbine_ctx *ctx, turbine_event *e) {
  if (e == NULL || e->ctx != ctx) {
    set_error(ctx->last_error, "stub: event of another context");
    return TURBINE_E_ARGUMENT;
  }
  if (!e->recorded)
    return 1;
  return atomic_load(&hold_events) ? 0 : 1;
}

int32_t turbine_stream_wait_event(turbine_ctx *ctx, turbine_stream *st,
                                  turbine_event *e) {
  if (e == NULL || e->ctx != ctx || (st != NULL && st->ctx != ctx)) {
    set_error(ctx->last_error, "stub: object of another context");
    return TURBINE_E_ARGUMENT;
  }
  atomic_fetch_add(&stream_waits, 1);
  return TURBINE_OK;
}
#endif /* TURBINE_STUB_V25 */

#ifdef TURBINE_STUB_V26
/* v2.6: the row_sumsq and rmsnorm_sharded trios (unsupported like every op)
 * and native stream handles: the compute stream's handle is the address of
 * the context's compute_stream_tag (stub_compute_stream(ctx) returns it), a
 * copy stream's handle is the turbine_stream pointer itself. */
STUB_OP(row_sumsq, turbine_row_sumsq_desc)
STUB_OP(rmsnorm_sharded, turbine_rmsnorm_sharded_desc)

void *stub_compute_stream(turbine_ctx *ctx) { return &ctx->compute_stream_tag; }

int32_t turbine_stream_native_handle(turbine_ctx *ctx, turbine_stream *s,
                                     void **out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null handle pointer");
    return TURBINE_E_ARGUMENT;
  }
  if (s != NULL && s->ctx != ctx) {
    set_error(ctx->last_error, "stub: stream of another context");
    return TURBINE_E_ARGUMENT;
  }
  *out = s != NULL ? (void *)s : stub_compute_stream(ctx);
  return TURBINE_OK;
}
#endif /* TURBINE_STUB_V26 */

#ifdef TURBINE_STUB_V27
/* v2.7: host-mapped memory is heap memory (the stub's "device" memory is host
 * memory too, so the device address is the host address), and
 * turbine_mapped_collective runs the one-shot protocol of the header on the
 * calling thread as one block: it writes this rank's contribution into its
 * slot, publishes flag word rank * max_blocks with a release store, waits
 * (sleeping 20 us between polls) for every peer's flag word with acquire
 * loads, then combines in rank order. A wait past timeout_ns stores the
 * timeout reason into abort_word and returns TURBINE_OK, like the kernel.
 * stub_live_mapped() counts allocations not yet freed. */
#include <time.h>

static atomic_int live_mapped;

int32_t stub_live_mapped(void) { return atomic_load(&live_mapped); }

int32_t turbine_host_alloc_mapped(turbine_ctx *ctx, size_t bytes, void **out) {
  if (out == NULL) {
    set_error(ctx->last_error, "stub: null out pointer");
    return TURBINE_E_ARGUMENT;
  }
  *out = calloc(1, bytes == 0 ? 1 : bytes);
  if (*out == NULL) {
    set_error(ctx->last_error, "stub: out of host memory");
    return TURBINE_E_OUT_OF_MEMORY;
  }
  atomic_fetch_add(&live_mapped, 1);
  return TURBINE_OK;
}

int32_t turbine_host_mapped_device_ptr(turbine_ctx *ctx, void *host,
                                       void **out) {
  if (host == NULL || out == NULL) {
    set_error(ctx->last_error, "stub: null pointer");
    return TURBINE_E_ARGUMENT;
  }
  *out = host;
  return TURBINE_OK;
}

int32_t turbine_host_free_mapped(turbine_ctx *ctx, void *host) {
  (void)ctx;
  if (host != NULL) {
    free(host);
    atomic_fetch_sub(&live_mapped, 1);
  }
  return TURBINE_OK;
}

static int64_t stub_round16(int64_t v) { return (v + 15) / 16 * 16; }

static int stub_reduction(int32_t kind) {
  return kind == TURBINE_MAPPED_ALL_REDUCE ||
         kind == TURBINE_MAPPED_REDUCE_SCATTER;
}

static int stub_mapped_ok(const turbine_mapped_collective_desc *d) {
  if (d == NULL || d->kind < 0 || d->kind > TURBINE_MAPPED_BROADCAST ||
      d->world < 1 || d->world > TURBINE_MAPPED_MAX_WORLD || d->rank < 0 ||
      d->rank >= d->world || d->max_blocks < 1 ||
      d->max_blocks > TURBINE_MAPPED_MAX_BLOCKS || d->timeout_ns <= 0 ||
      d->seq < 1 || d->bytes < 0 || d->slot_bytes % 16 != 0) {
    return 0;
  }
  if (d->kind == TURBINE_MAPPED_BROADCAST &&
      (d->root < 0 || d->root >= d->world)) {
    return 0;
  }
  int64_t e = 1;
  if (stub_reduction(d->kind)) {
    if (d->dtype != TURBINE_DTYPE_BF16 && d->dtype != TURBINE_DTYPE_F32) {
      return 0;
    }
    if (d->reduce_op != TURBINE_REDUCE_SUM &&
        d->reduce_op != TURBINE_REDUCE_MAX) {
      return 0;
    }
    e = d->dtype == TURBINE_DTYPE_BF16 ? 2 : 4;
  }
  if (d->bytes % e != 0) {
    return 0;
  }
  const int64_t parts = d->kind == TURBINE_MAPPED_REDUCE_SCATTER ? d->world : 1;
  return parts * stub_round16(d->bytes) <= d->slot_bytes;
}

int32_t
turbine_mapped_collective_supported(const turbine_mapped_collective_desc *d) {
  return stub_mapped_ok(d);
}

const char *
turbine_mapped_collective_impl(const turbine_mapped_collective_desc *d) {
  (void)d;
  return "stub_mapped_collective";
}

static uint64_t stub_now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static float stub_load(const uint8_t *p, int32_t dtype) {
  if (dtype == TURBINE_DTYPE_BF16) {
    uint16_t b;
    memcpy(&b, p, 2);
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
  }
  float f;
  memcpy(&f, p, 4);
  return f;
}

static void stub_store(uint8_t *p, int32_t dtype, float v) {
  if (dtype == TURBINE_DTYPE_BF16) {
    uint32_t u;
    memcpy(&u, &v, 4);
    uint16_t b;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
      b = (uint16_t)((u >> 16) | 0x0040u);
    } else {
      u += 0x7fffu + ((u >> 16) & 1u);
      b = (uint16_t)(u >> 16);
    }
    memcpy(p, &b, 2);
    return;
  }
  memcpy(p, &v, 4);
}

int32_t turbine_mapped_collective(turbine_ctx *ctx,
                                  const turbine_mapped_collective_desc *d) {
  if (!stub_mapped_ok(d)) {
    set_error(ctx->last_error, "stub: mapped_collective descriptor refused");
    return TURBINE_E_UNSUPPORTED;
  }
  if (d->bytes == 0) {
    return TURBINE_OK;
  }
  _Atomic uint32_t *abort_word = (_Atomic uint32_t *)d->abort_word;
  if (atomic_load(abort_word) != 0) {
    return TURBINE_OK;
  }
  const int64_t part = stub_round16(d->bytes);
  uint8_t *base = (uint8_t *)d->slots;
  const int64_t parity = (int64_t)(d->seq & 1u);
#define STUB_SLOT(r) (base + (parity * d->world + (r)) * d->slot_bytes)
  const uint8_t *send = (const uint8_t *)d->send;
  uint8_t *recv = (uint8_t *)d->recv;
  uint8_t *mine = STUB_SLOT(d->rank);
  switch (d->kind) {
  case TURBINE_MAPPED_REDUCE_SCATTER:
    for (int32_t q = 0; q < d->world; ++q) {
      if (q != d->rank) {
        memcpy(mine + q * part, send + q * d->send_stride, (size_t)d->bytes);
      }
    }
    break;
  case TURBINE_MAPPED_BROADCAST:
    if (d->rank == d->root) {
      memcpy(mine, send, (size_t)d->bytes);
    }
    break;
  default:
    memcpy(mine, send, (size_t)d->bytes);
    break;
  }
  const uint64_t tag = (d->seq << 24) | 1u;
  atomic_store_explicit(
      (_Atomic uint64_t *)&d->flags[(int64_t)d->rank * d->max_blocks], tag,
      memory_order_release);
  const uint64_t deadline = stub_now_ns() + (uint64_t)d->timeout_ns;
  for (int32_t q = 0; q < d->world; ++q) {
    if (q == d->rank) {
      continue;
    }
    _Atomic uint64_t *f =
        (_Atomic uint64_t *)&d->flags[(int64_t)q * d->max_blocks];
    while (atomic_load_explicit(f, memory_order_acquire) < tag) {
      if (atomic_load(abort_word) != 0) {
        return TURBINE_OK;
      }
      if (stub_now_ns() > deadline) {
        uint32_t expected = 0;
        atomic_compare_exchange_strong(abort_word, &expected,
                                       (TURBINE_MAPPED_ABORT_TIMEOUT << 24) |
                                           ((uint32_t)d->kind << 16) |
                                           (uint32_t)d->rank);
        return TURBINE_OK;
      }
      struct timespec nap = {0, 20000};
      nanosleep(&nap, NULL);
    }
  }
  switch (d->kind) {
  case TURBINE_MAPPED_ALL_GATHER:
    for (int32_t q = 0; q < d->world; ++q) {
      memcpy(recv + q * d->recv_stride, q == d->rank ? send : STUB_SLOT(q),
             (size_t)d->bytes);
    }
    break;
  case TURBINE_MAPPED_BROADCAST:
    if (d->rank != d->root) {
      memcpy(recv, STUB_SLOT(d->root), (size_t)d->bytes);
    } else if (recv != send) {
      memcpy(recv, send, (size_t)d->bytes);
    }
    break;
  default: {
    const int scatter = d->kind == TURBINE_MAPPED_REDUCE_SCATTER;
    const int64_t e = d->dtype == TURBINE_DTYPE_BF16 ? 2 : 4;
    for (int64_t i = 0; i < d->bytes; i += e) {
      float acc = 0.0f;
      for (int32_t q = 0; q < d->world; ++q) {
        const uint8_t *src =
            q == d->rank ? send + (scatter ? d->rank * d->send_stride : 0)
                         : STUB_SLOT(q) + (scatter ? d->rank * part : 0);
        const float x = stub_load(src + i, d->dtype);
        if (q == 0) {
          acc = x;
        } else if (d->reduce_op == TURBINE_REDUCE_SUM) {
          acc = acc + x;
        } else {
          acc = x > acc ? x : acc;
        }
      }
      stub_store(recv + i, d->dtype, acc);
    }
    break;
  }
  }
#undef STUB_SLOT
  return TURBINE_OK;
}
#endif /* TURBINE_STUB_V27 */
