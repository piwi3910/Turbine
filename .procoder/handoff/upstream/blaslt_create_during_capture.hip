// hipblasLtCreate on thread B while thread A captures (thread-local mode, non-blocking
// stream): hipblasLtCreate's synchronous hipMemset fails and hipBLASLt exit(1)s.
// Build: hipcc -std=c++17 -O2 blaslt_create_during_capture.hip -o blaslt_create_during_capture -lhipblaslt -lpthread
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

static void report_exit() { std::fprintf(stderr, "process is exiting (atexit handler ran)\n"); }

int main() {
  std::atexit(report_exit);
  hipStream_t s;
  (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  // Control: creating a handle with no capture open works.
  hipblasLtHandle_t h0 = nullptr;
  std::printf("hipblasLtCreate, no capture open: status %d\n", (int)hipblasLtCreate(&h0));
  (void)hipblasLtDestroy(h0);

  std::atomic<int> phase{0};
  std::thread capturer([&] {
    std::printf("hipStreamBeginCapture(ThreadLocal): %s\n",
                hipGetErrorName(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal)));
    phase = 1;
    while (phase.load() != 2) std::this_thread::yield();
    hipGraph_t g = nullptr;
    std::printf("hipStreamEndCapture: %s\n", hipGetErrorName(hipStreamEndCapture(s, &g)));
  });
  std::thread creator([&] {
    while (phase.load() != 1) std::this_thread::yield();
    hipblasLtHandle_t h = nullptr;
    hipblasStatus_t st = hipblasLtCreate(&h);  // never returns on ROCm 7.14.1
    std::printf("hipblasLtCreate during another thread's capture: status %d\n", (int)st);
    if (st == HIPBLAS_STATUS_SUCCESS) (void)hipblasLtDestroy(h);
    phase = 2;
  });
  creator.join();
  capturer.join();
  std::printf("done, exit 0\n");
  return 0;
}
