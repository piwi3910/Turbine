// ABI v2.5 (Phase 4): copy streams, asynchronous copies, event queries and
// stream waits, on top of the v2.3 pinned host memory and events (memory.cpp);
// and ABI v2.6 (Phase 5): the native handle of a stream, for a collective
// library that enqueues on it.
//
// Ownership: a copy stream belongs to the caller until
// turbine_copy_stream_destroy, which drains it first and must come before
// turbine_ctx_destroy. A turbine_stream is a hipStream_t and a turbine_event a
// hipEvent_t, cast to the ABI's opaque types (as in memory.cpp); a NULL stream
// argument means the context's compute stream. turbine_memcpy_async only
// enqueues: the caller keeps both buffers alive, and the pinned side untouched,
// until an event recorded on the same stream after the copy has completed.
// While the compute stream is captured into a graph (graph.cpp) these
// functions return TURBINE_E_ARGUMENT, like the v2.3 ones.
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;
using turbine_hip::refuse_while_capturing;

namespace {

hipStream_t stream_of(turbine_ctx *ctx, turbine_stream *s) {
  return s != nullptr ? reinterpret_cast<hipStream_t>(s) : ctx->stream;
}

} // namespace

extern "C" {

int32_t turbine_copy_stream_create(turbine_ctx *ctx, turbine_stream **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_copy_stream_create");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_copy_stream_create: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  hipStream_t stream = nullptr;
  if (int32_t rc = check_hip(
          ctx, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags");
      rc != TURBINE_OK)
    return rc;
  *out = reinterpret_cast<turbine_stream *>(stream);
  return TURBINE_OK;
}

int32_t turbine_copy_stream_destroy(turbine_ctx *ctx, turbine_stream *s) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (s == nullptr)
    return TURBINE_OK;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  hipStream_t stream = reinterpret_cast<hipStream_t>(s);
  // Drain first: copies still in flight read or write caller buffers.
  const int32_t drained =
      check_hip(ctx, hipStreamSynchronize(stream), "hipStreamSynchronize");
  const int32_t destroyed =
      check_hip(ctx, hipStreamDestroy(stream), "hipStreamDestroy");
  return drained != TURBINE_OK ? drained : destroyed;
}

int32_t turbine_memcpy_async(turbine_ctx *ctx, turbine_stream *s, void *dst,
                             const void *src, size_t bytes, int32_t kind) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  // A device-to-device copy on the compute stream is captured as a graph node
  // (tensor-parallel decode graphs reorder the gathered logits with them).
  if (ctx->capturing && (s != nullptr || kind != TURBINE_COPY_D2D))
    return refuse_while_capturing(ctx, "turbine_memcpy_async");
  if (bytes == 0)
    return TURBINE_OK;
  if (dst == nullptr || src == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_memcpy_async: NULL pointer");
  }
  hipMemcpyKind hip_kind;
  switch (kind) {
  case TURBINE_COPY_H2D:
    hip_kind = hipMemcpyHostToDevice;
    break;
  case TURBINE_COPY_D2H:
    hip_kind = hipMemcpyDeviceToHost;
    break;
  case TURBINE_COPY_D2D:
    hip_kind = hipMemcpyDeviceToDevice;
    break;
  default:
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_memcpy_async: unknown kind " + std::to_string(kind));
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx,
                   hipMemcpyAsync(dst, src, bytes, hip_kind, stream_of(ctx, s)),
                   "hipMemcpyAsync");
}

int32_t turbine_event_query(turbine_ctx *ctx, turbine_event *e) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (e == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_event_query: NULL event");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  const hipError_t err = hipEventQuery(reinterpret_cast<hipEvent_t>(e));
  if (err == hipErrorNotReady)
    return 0;
  const int32_t rc = check_hip(ctx, err, "hipEventQuery");
  return rc == TURBINE_OK ? 1 : rc;
}

int32_t turbine_stream_wait_event(turbine_ctx *ctx, turbine_stream *s,
                                  turbine_event *e) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_stream_wait_event");
  if (e == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_stream_wait_event: NULL event");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(
      ctx,
      hipStreamWaitEvent(stream_of(ctx, s), reinterpret_cast<hipEvent_t>(e), 0),
      "hipStreamWaitEvent");
}

// ABI v2.6: the hipStream_t itself. Allowed while capturing (it only reads
// the handle); the stream stays owned by the context or the copy stream's
// owner.
int32_t turbine_stream_native_handle(turbine_ctx *ctx, turbine_stream *s,
                                     void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_stream_native_handle: out is NULL");
  }
  *out = reinterpret_cast<void *>(stream_of(ctx, s));
  return TURBINE_OK;
}

} // extern "C"
