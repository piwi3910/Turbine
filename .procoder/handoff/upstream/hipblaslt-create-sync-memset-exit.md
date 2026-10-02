# Filed: hipblasLtCreate calls the process-wide exit(1) when its synchronous hipMemset fails (for example, during another thread's graph capture)

Target: ROCm/rocm-libraries (projects/hipblaslt). Status: filed 2026-10-02 as https://github.com/ROCm/rocm-libraries/issues/12895 (user said file). Fix PR: https://github.com/ROCm/rocm-libraries/pull/12900 (2026-10-02).

## Environment

- hipBLASLt 1.4.1 from ROCm 7.14.1 (TheRock build), HIP runtime 7.14.60850, AMD clang 23.0.0git
- GPU: AMD Radeon AI PRO R9700 (gfx1201), one device visible (`ROCR_VISIBLE_DEVICES=0`)
- OS: Debian GNU/Linux 13 (trixie), Linux 7.1.8, in-kernel amdgpu

## Summary

`hipblasLtCreate` allocates its synchronizer with `hipMalloc` and clears it with the synchronous, legacy-stream
`hipMemset`, both wrapped in a local `CHECK_HIP_ERROR` macro that prints and calls `exit(EXIT_FAILURE)` on any error.
On ROCm 7.14.1, that `hipMemset` fails whenever any stream of the process is capturing a graph, even another
thread's thread-local capture on a non-blocking stream (see the companion report on clr). So one thread creating a
hipBLASLt handle while another captures terminates the whole process. The caller never gets a status back.

## Expected

`hipblasLtCreate` returns an error status (e.g. `HIPBLAS_STATUS_INTERNAL_ERROR` or `HIPBLAS_STATUS_ALLOC_FAILED`)
instead of exiting. Ideally it does not use a synchronous legacy-stream operation at all, so it cannot conflict with
captures elsewhere in the process. The cuBLASLt equivalent, `cublasLtCreate`, returns a status and does not
terminate the process.

## Actual

Output of the repro below, built and run on the environment above (stdout and stderr interleaved; stdout was still
buffered when `exit` ran):

```
Hip error: 'operation would make the legacy stream depend on a capturing blocking stream'(906) at /__w/rockrel/rockrel/rocm-libraries/projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp:165
process is exiting (atexit handler ran)
hipblasLtCreate, no capture open: status 0
hipStreamBeginCapture(ThreadLocal): hipSuccess
exit status 1
```

The line after `hipblasLtCreate` in the creating thread is never reached. In a larger process, the other threads
still running GPU work while `exit` runs the static destructors then crash with SIGSEGV.

## Source involved (rocm-libraries, tag therock-7.14.1)

`projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp`:

- lines 132-143: `CHECK_HIP_ERROR` prints `Hip error: ...` and calls `exit(EXIT_FAILURE)`
- line 164: `CHECK_HIP_ERROR(hipMalloc(&d_Synchronizer, 16 * 409600 * sizeof(int)));`
- line 165: `CHECK_HIP_ERROR(hipMemset(d_Synchronizer, 0, sizeof(int) * 16 * 409600));`
- line 187 (`hipblasLtDestroy`): `CHECK_HIP_ERROR(hipFree(...))`, with the same exit-on-error behavior

## Minimal repro

`blaslt_create_during_capture.hip` (next to this file):

```cpp
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
```

Build and run:

```
hipcc -std=c++17 -O2 --offload-arch=gfx1201 blaslt_create_during_capture.hip -o blaslt_create_during_capture -lhipblaslt -lpthread
./blaslt_create_during_capture; echo "exit status $?"
```

## Impact

A library function terminates the host process. Any application that creates hipBLASLt handles lazily, or on several
threads (for example one handle per device or per worker), while other threads capture graphs dies without an error
it can handle. In our inference engine this appeared as an intermittent test-binary exit with status 1 and then
SIGSEGV, and it took a dedicated investigation to trace back to this line.

## Our workaround

Our HIP layer serializes handle creation against graph capture process-wide: `hipblasLtCreate` runs only while no
capture is open, and new captures wait until it returns.

## Suggested fix

1. Replace `exit(EXIT_FAILURE)` in `CHECK_HIP_ERROR` with returning a `hipblasStatus_t` (and free what was already
   allocated).
2. Clear the synchronizer with `hipMemsetAsync` on a stream the handle owns (or a non-blocking internal stream)
   followed by a stream synchronize, rather than the synchronous legacy-stream `hipMemset`.
