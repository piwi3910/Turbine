# Filed: A synchronous hipMemset on one thread invalidates a thread-local capture on a non-blocking stream of another thread

Target: ROCm/rocm-systems (clr / hipamd). Status: filed 2026-10-02 as https://github.com/ROCm/rocm-systems/issues/12677 (user said file).

## Environment

- ROCm 7.14.1 (TheRock build), HIP runtime 7.14.60850, AMD clang 23.0.0git
- GPU: AMD Radeon AI PRO R9700 (gfx1201), one device visible (`ROCR_VISIBLE_DEVICES=0`)
- OS: Debian GNU/Linux 13 (trixie), Linux 7.1.8, in-kernel amdgpu

## Summary

Thread A captures its own non-blocking stream with `hipStreamCaptureModeThreadLocal`. Thread B, which captures
nothing, calls the synchronous `hipMemset` on an unrelated buffer. B's call fails with
`hipErrorStreamCaptureImplicit` (906, "operation would make the legacy stream depend on a capturing blocking
stream"), and A's capture is invalidated: A's next launch and `hipStreamEndCapture` return
`hipErrorStreamCaptureInvalidated` and no graph is produced.

## Expected (CUDA semantics)

- Thread-local mode: by the CUDA documentation for `cudaStreamBeginCapture` / `cudaThreadExchangeStreamCaptureMode`,
  only calls made by the thread that began the capture are restricted; other threads are not affected.
- Legacy stream: a synchronous memset on the legacy stream synchronizes implicitly only with blocking streams. A
  stream created with the non-blocking flag never takes part in that, so capturing it cannot make "the legacy stream
  depend on a capturing blocking stream".

So both calls should succeed: B's memset completes and A gets a valid graph. We have not run the CUDA equivalent;
this expectation comes from the CUDA documentation.

## Actual (ROCm 7.14.1)

Output of the repro below, built and run on the environment above:

```
hipStreamCreateWithFlags(&s, hipStreamNonBlocking)         -> hipSuccess (0)
hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal)  -> hipSuccess (0)
hipGetLastError()                                          -> hipSuccess (0)
hipMemset(b, 0, 64 * sizeof(int))                          -> hipErrorStreamCaptureImplicit (906)
hipGetLastError()                                          -> hipErrorStreamCaptureInvalidated (901)
hipStreamEndCapture(s, &g)                                 -> hipErrorStreamCaptureInvalidated (901)
capture produced a graph: no
a[0] after one replay: 0 (expected 2)
```

(The last line shows `0` because no graph was ever made, so nothing was replayed.)

## Source involved (rocm-systems, tag therock-7.14.1)

- `projects/clr/hipamd/src/hip_internal.hpp`, `INVALIDATE_ALL_CAPTURING_AND_RETURN` and `CHECK_STREAM_CAPTURING`
  (around line 274): "Sync APIs (hipMemset, hipMemcpy, etc.) cannot be called when stream capture is active for any
  capture mode (Global, ThreadLocal, or Relaxed)". If `g_allCapturingStreams` is non-empty, the macro marks _every_
  capturing stream of the process invalidated and returns `hipErrorStreamCaptureImplicit`. It does not check the
  calling thread, the capture mode or the stream's non-blocking flag.
- `projects/clr/hipamd/src/hip_memory.cpp`: `hipMemset_common` (line 3388) and `hipMemcpy_common` (line 868) start
  with `CHECK_STREAM_CAPTURING()`.
- For comparison, `Device::StreamCaptureBlocking()` (`hip_device.cpp`) does skip non-blocking streams; that is the
  check the error text describes, but the sync-API path does not use it.

## Minimal repro

`capture_cross_thread.hip` (next to this file):

```cpp
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <thread>

__global__ void inc(int *p) { p[threadIdx.x] += 1; }

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    std::printf("%-58s -> %s (%d)\n", #x, hipGetErrorName(e_), (int)e_);     \
  } while (0)

int main() {
  int *a = nullptr, *b = nullptr;
  CHECK(hipMalloc(&a, 64 * sizeof(int)));
  CHECK(hipMalloc(&b, 64 * sizeof(int)));
  CHECK(hipMemset(a, 0, 64 * sizeof(int)));
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  std::atomic<int> phase{0};
  std::thread capturer([&] {
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    inc<<<1, 64, 0, s>>>(a);
    CHECK(hipGetLastError());
    phase = 1;                       // let thread B run its hipMemset
    while (phase.load() != 2) std::this_thread::yield();
    inc<<<1, 64, 0, s>>>(a);
    CHECK(hipGetLastError());
    hipGraph_t g = nullptr;
    CHECK(hipStreamEndCapture(s, &g));
    std::printf("capture produced a graph: %s\n", g ? "yes" : "no");
    if (g) {
      hipGraphExec_t ge = nullptr;
      CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      if (ge) { CHECK(hipGraphLaunch(ge, s)); CHECK(hipStreamSynchronize(s)); }
    }
  });
  std::thread other([&] {
    while (phase.load() != 1) std::this_thread::yield();
    // Unrelated work on another thread: a synchronous memset of another buffer.
    CHECK(hipMemset(b, 0, 64 * sizeof(int)));
    phase = 2;
  });
  capturer.join();
  other.join();
  int host[64] = {};
  CHECK(hipMemcpy(host, a, sizeof host, hipMemcpyDeviceToHost));
  std::printf("a[0] after one replay: %d (expected 2)\n", host[0]);
  return 0;
}
```

Build and run:

```
hipcc -std=c++17 -O2 --offload-arch=gfx1201 capture_cross_thread.hip -o capture_cross_thread -lpthread
./capture_cross_thread
```

## Impact

Any multi-threaded process that captures graphs on one thread breaks whenever another thread makes a synchronous
memset or memcpy. The other thread may be a library initializing itself (hipBLASLt's `hipblasLtCreate` does this,
see the companion report) or simply other work on another device. Thread-local capture mode, which exists to isolate
captures per thread, gives no protection. Our inference engine hit this as an intermittent process exit in a test
binary that runs GPU tests on several threads.

## Our workaround

A process-wide gate in our HIP layer: creating a context (which calls `hipblasLtCreate`) waits until no capture in
the process is open, and a new capture waits while a context is being created. This only covers the synchronous calls
we know about. A third-party library making one on another thread during a capture would still break it.

## Suggested fix

Apply `CHECK_STREAM_CAPTURING` only where CUDA does. In thread-local mode, restrict only the capturing thread. Only
count capturing streams that are blocking (as `StreamCaptureBlocking()` does) when deciding whether a legacy-stream
operation conflicts. Do not invalidate captures owned by other threads in thread-local mode.
