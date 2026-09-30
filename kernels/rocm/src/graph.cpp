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
// Every capture holds an entry in the process-wide capture gate from
// turbine_graph_begin until the capture ends (turbine_graph_end or
// abandon_capture); turbine_ctx_create waits until the gate is empty and keeps
// new captures from beginning meanwhile (turbine_hip.hpp, CreationGuard): HIP
// breaks every capture of the process on the synchronous hipMemset inside
// hipblasLtCreate.
//
// Ownership: a turbine_graph owns its instantiated hipGraphExec_t and remembers
// the context it was captured on; it records the device pointers of the
// captured ops, which the caller keeps alive until turbine_graph_destroy.
#include <algorithm>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

struct turbine_graph {
  turbine_ctx *ctx = nullptr;
  hipGraphExec_t exec = nullptr;
};

namespace {

// The capture gate: one entry (the beginning thread) per open capture, and the
// number of context creations in progress. Allocated once and never freed, so
// a thread still ending a capture while the process exits never touches a
// destroyed mutex.
struct CaptureGate {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::thread::id> capture_threads;
  int creating = 0;
};

CaptureGate &gate() {
  static CaptureGate *g = new CaptureGate();
  return *g;
}

// Registers a capture about to begin on the calling thread; waits while a
// context is being created.
void gate_enter_capture() {
  CaptureGate &g = gate();
  std::unique_lock<std::mutex> lock(g.mutex);
  g.changed.wait(lock, [&g] { return g.creating == 0; });
  g.capture_threads.push_back(std::this_thread::get_id());
}

// Removes the entry of a capture begun on thread `id` (it ended, or failed to
// begin).
void gate_leave_capture(std::thread::id id) {
  CaptureGate &g = gate();
  {
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = std::find(g.capture_threads.begin(), g.capture_threads.end(), id);
    if (it != g.capture_threads.end())
      g.capture_threads.erase(it);
  }
  g.changed.notify_all();
}

// Marks ctx's capture ended (whatever became of it) and leaves the gate.
void capture_ended(turbine_ctx *ctx) {
  ctx->capturing = false;
  gate_leave_capture(ctx->capture_thread);
}

} // namespace

namespace turbine_hip {

CreationGuard::CreationGuard() {
  CaptureGate &g = gate();
  std::unique_lock<std::mutex> lock(g.mutex);
  if (std::find(g.capture_threads.begin(), g.capture_threads.end(),
                std::this_thread::get_id()) != g.capture_threads.end()) {
    return;
  }
  ++g.creating;
  g.changed.wait(lock, [&g] { return g.capture_threads.empty(); });
  ok_ = true;
}

CreationGuard::~CreationGuard() {
  if (!ok_)
    return;
  CaptureGate &g = gate();
  {
    std::lock_guard<std::mutex> lock(g.mutex);
    --g.creating;
  }
  g.changed.notify_all();
}

int32_t refuse_while_capturing(turbine_ctx *ctx, const char *what) {
  return fail(ctx, TURBINE_E_ARGUMENT,
              std::string(what) +
                  ": not allowed while the compute stream is captured into a "
                  "graph (only op calls are)");
}

void abandon_capture(turbine_ctx *ctx) {
  if (!ctx->capturing)
    return;
  hipGraph_t graph = nullptr;
  const hipError_t ended = hipStreamEndCapture(ctx->stream, &graph);
  // Leave the gate only once the stream no longer captures.
  capture_ended(ctx);
  if (ended == hipSuccess && graph != nullptr) {
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
  gate_enter_capture();
  if (int32_t rc = check_hip(
          ctx,
          hipStreamBeginCapture(ctx->stream, hipStreamCaptureModeThreadLocal),
          "hipStreamBeginCapture");
      rc != TURBINE_OK) {
    gate_leave_capture(std::this_thread::get_id());
    return rc;
  }
  ctx->capture_thread = std::this_thread::get_id();
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
  hipGraph_t graph = nullptr;
  const hipError_t ended = hipStreamEndCapture(ctx->stream, &graph);
  capture_ended(ctx);
  if (int32_t rc = check_hip(ctx, ended, "hipStreamEndCapture");
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
