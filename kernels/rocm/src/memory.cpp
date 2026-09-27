// Device allocation, host<->device copies, stream synchronisation and memory
// info, and (ABI v2.3) pinned host memory and stream events (v2.5 copy streams:
// copy_stream.cpp). Copies are
// enqueued on the context's compute stream; host buffers must stay valid until
// the next turbine_stream_sync, or for pinned memory until an event recorded
// after the copy has completed (ABI contract). While the stream is captured
// into a graph (graph.cpp) allocations, copies, syncs and the v2.3 functions
// return TURBINE_E_ARGUMENT: only op calls may be captured.
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;
using turbine_hip::refuse_while_capturing;

extern "C" {

int32_t turbine_malloc(turbine_ctx *ctx, size_t bytes, void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_malloc");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_malloc: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  const std::string what = "hipMalloc " + std::to_string(bytes) + " bytes";
  return check_hip(ctx, hipMalloc(out, bytes), what.c_str());
}

int32_t turbine_free(turbine_ctx *ctx, void *ptr) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_free");
  if (ptr == nullptr)
    return TURBINE_OK;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipFree(ptr), "hipFree");
}

int32_t turbine_memcpy_h2d(turbine_ctx *ctx, void *dst_device,
                           const void *src_host, size_t bytes) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_memcpy_h2d");
  if (bytes == 0)
    return TURBINE_OK;
  if (dst_device == nullptr || src_host == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_memcpy_h2d: NULL pointer");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx,
                   hipMemcpyAsync(dst_device, src_host, bytes,
                                  hipMemcpyHostToDevice, ctx->stream),
                   "hipMemcpyAsync host to device");
}

int32_t turbine_memcpy_d2h(turbine_ctx *ctx, void *dst_host,
                           const void *src_device, size_t bytes) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_memcpy_d2h");
  if (bytes == 0)
    return TURBINE_OK;
  if (dst_host == nullptr || src_device == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_memcpy_d2h: NULL pointer");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx,
                   hipMemcpyAsync(dst_host, src_device, bytes,
                                  hipMemcpyDeviceToHost, ctx->stream),
                   "hipMemcpyAsync device to host");
}

int32_t turbine_stream_sync(turbine_ctx *ctx) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_stream_sync");
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipStreamSynchronize(ctx->stream),
                   "hipStreamSynchronize");
}

int32_t turbine_mem_info(turbine_ctx *ctx, size_t *free_bytes,
                         size_t *total_bytes) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (free_bytes == nullptr || total_bytes == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_mem_info: NULL output");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipMemGetInfo(free_bytes, total_bytes),
                   "hipMemGetInfo");
}

// ABI v2.3: page-locked host staging memory and stream events. A turbine_event
// is a hipEvent_t (a pointer to the runtime's event object) cast to the ABI's
// opaque type.

int32_t turbine_host_alloc_pinned(turbine_ctx *ctx, size_t bytes, void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_host_alloc_pinned");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_host_alloc_pinned: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  const std::string what = "hipHostMalloc " + std::to_string(bytes) + " bytes";
  return check_hip(
      ctx, hipHostMalloc(out, bytes == 0 ? 1 : bytes, hipHostMallocDefault),
      what.c_str());
}

int32_t turbine_host_free_pinned(turbine_ctx *ctx, void *ptr) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_host_free_pinned");
  if (ptr == nullptr)
    return TURBINE_OK;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipHostFree(ptr), "hipHostFree");
}

int32_t turbine_event_create(turbine_ctx *ctx, turbine_event **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_event_create");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_event_create: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  hipEvent_t event = nullptr;
  if (int32_t rc =
          check_hip(ctx, hipEventCreateWithFlags(&event, hipEventDisableTiming),
                    "hipEventCreateWithFlags");
      rc != TURBINE_OK)
    return rc;
  *out = reinterpret_cast<turbine_event *>(event);
  return TURBINE_OK;
}

int32_t turbine_event_record(turbine_ctx *ctx, turbine_event *e,
                             turbine_stream *s) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_event_record");
  if (e == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_event_record: NULL event");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  // s NULL = the compute stream; from v2.5 a copy stream (copy_stream.cpp).
  hipStream_t stream =
      s != nullptr ? reinterpret_cast<hipStream_t>(s) : ctx->stream;
  return check_hip(ctx, hipEventRecord(reinterpret_cast<hipEvent_t>(e), stream),
                   "hipEventRecord");
}

int32_t turbine_event_synchronize(turbine_ctx *ctx, turbine_event *e) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_event_synchronize");
  if (e == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_event_synchronize: NULL event");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipEventSynchronize(reinterpret_cast<hipEvent_t>(e)),
                   "hipEventSynchronize");
}

int32_t turbine_event_destroy(turbine_ctx *ctx, turbine_event *e) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_event_destroy");
  if (e == nullptr)
    return TURBINE_OK;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipEventDestroy(reinterpret_cast<hipEvent_t>(e)),
                   "hipEventDestroy");
}

} // extern "C"
