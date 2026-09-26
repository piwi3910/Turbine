// Graphs (ABI v2.1): capture of the context's compute stream into a HIP graph,
// instantiated once and replayed with hipGraphLaunch.
//
// turbine_graph_begin starts a thread-local capture of ctx->stream and sets
// ctx->capturing; while it is set, turbine_malloc, turbine_free,
// turbine_memcpy_h2d, turbine_memcpy_d2h, turbine_stream_sync and a growing
// MoE scratch return TURBINE_E_ARGUMENT (memory.cpp, moe.cpp) instead of
// breaking the capture. turbine_graph_end always ends the capture, so a failed
// capture (an op error, or a runtime call the capture rejected) leaves the
// context usable: the caller runs the work eagerly instead.
//
// Ownership: a turbine_graph owns its instantiated hipGraphExec_t and remembers
// the context it was captured on; it records the device pointers of the
// captured ops, which the caller keeps alive until turbine_graph_destroy.
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

struct turbine_graph {
  turbine_ctx *ctx = nullptr;
  hipGraphExec_t exec = nullptr;
};

namespace turbine_hip {

int32_t refuse_while_capturing(turbine_ctx *ctx, const char *what) {
  return fail(ctx, TURBINE_E_ARGUMENT,
              std::string(what) +
                  ": not allowed while the compute stream is captured into a "
                  "graph (only op calls are)");
}

void abandon_capture(turbine_ctx *ctx) {
  if (!ctx->capturing)
    return;
  ctx->capturing = false;
  hipGraph_t graph = nullptr;
  if (hipStreamEndCapture(ctx->stream, &graph) == hipSuccess &&
      graph != nullptr) {
    (void)hipGraphDestroy(graph);
  }
  (void)hipGetLastError();
}

} // namespace turbine_hip

extern "C" {

int32_t turbine_graph_begin(turbine_ctx *ctx) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_graph_begin: a capture is already in progress");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = check_hip(
          ctx,
          hipStreamBeginCapture(ctx->stream, hipStreamCaptureModeThreadLocal),
          "hipStreamBeginCapture");
      rc != TURBINE_OK) {
    return rc;
  }
  ctx->capturing = true;
  return TURBINE_OK;
}

int32_t turbine_graph_end(turbine_ctx *ctx, turbine_graph **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (out != nullptr)
    *out = nullptr;
  if (!ctx->capturing) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_graph_end: no capture in progress");
  }
  if (out == nullptr) {
    turbine_hip::abandon_capture(ctx);
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_graph_end: out is NULL");
  }
  (void)hipSetDevice(ctx->device);
  ctx->capturing = false;
  hipGraph_t graph = nullptr;
  if (int32_t rc = check_hip(ctx, hipStreamEndCapture(ctx->stream, &graph),
                             "hipStreamEndCapture");
      rc != TURBINE_OK) {
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
    return rc;
  }
  if (graph == nullptr) {
    return fail(ctx, TURBINE_E_DEVICE, "hipStreamEndCapture returned no graph");
  }
  hipGraphExec_t exec = nullptr;
  const int32_t rc =
      check_hip(ctx, hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                "hipGraphInstantiate");
  (void)hipGraphDestroy(graph);
  if (rc != TURBINE_OK)
    return rc;
  auto *g = new turbine_graph();
  g->ctx = ctx;
  g->exec = exec;
  *out = g;
  return TURBINE_OK;
}

int32_t turbine_graph_launch(turbine_ctx *ctx, turbine_graph *g) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (g == nullptr || g->ctx != ctx) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_graph_launch: graph is NULL or of another context");
  }
  if (ctx->capturing)
    return turbine_hip::refuse_while_capturing(ctx, "turbine_graph_launch");
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipGraphLaunch(g->exec, ctx->stream), "hipGraphLaunch");
}

int32_t turbine_graph_destroy(turbine_ctx *ctx, turbine_graph *g) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (g == nullptr)
    return TURBINE_OK;
  if (g->ctx != ctx) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_graph_destroy: graph of another context");
  }
  (void)hipSetDevice(ctx->device);
  int32_t rc = TURBINE_OK;
  // A launched replay may still run: drain the stream first (not possible, nor
  // needed, while capturing: nothing launched in the capture ran yet).
  if (!ctx->capturing) {
    rc = check_hip(ctx, hipStreamSynchronize(ctx->stream),
                   "hipStreamSynchronize (graph destroy)");
  }
  const int32_t destroyed =
      check_hip(ctx, hipGraphExecDestroy(g->exec), "hipGraphExecDestroy");
  delete g;
  return rc != TURBINE_OK ? rc : destroyed;
}

} // extern "C"
