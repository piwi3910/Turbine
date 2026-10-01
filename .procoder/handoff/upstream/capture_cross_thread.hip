// A synchronous hipMemset on thread B breaks a thread-local capture on a
// non-blocking stream owned by thread A.
// Build: hipcc -std=c++17 -O2 capture_cross_thread.hip -o capture_cross_thread -lpthread
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
