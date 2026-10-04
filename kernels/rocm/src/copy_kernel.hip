// ABI v2.11 (Phase 6b, decision "6b: KV promotions slow decode — which fix",
// A): turbine_memcpy_h2d_kernel, a host-to-device copy kernel for KV
// promotions out of pinned host memory.
//
// Why a kernel: on the R9700 (ROCm 7.14) an SDMA copy of pinned host memory
// stalls the compute queue for about the copy's own length (rocprofv3: rmsnorm
// 6 -> 80 us while a 1 MiB promotion copy runs; decode loses ~0.05-0.065 ms per
// MiB promoted). The stall grows with the host reads in flight, not with the
// engine: a copy kernel with 16 or more workgroups stalls as much, while one
// with 2 workgroups moves ~22 GB/s (SDMA: ~12.4) and costs decode ~0.003 ms
// per MiB (.procoder/perf-log.md, 6b "Promotion copy kernel"). The runtime's
// own blit kernels (HSA_ENABLE_SDMA=0) cannot be limited this way.
//
// Reuse first: SGLang's sgl-kernel kvcacheio (transfer_kv_*) and LMCache's
// multi_layer_kv_transfer are CUDA kernels of the same shape (a few blocks
// reading pinned host memory through its device address); neither ships for
// ROCm, and the copy itself is a vector load/store loop, so it is ours.
//
// Ownership: the segments are read on the host during the call and passed to
// the kernels by value (kernel arguments); the caller keeps both buffers alive
// and the host side untouched until an event recorded on s after the call has
// completed (the turbine_memcpy_async rules).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;
using turbine_hip::refuse_while_capturing;

namespace {

constexpr int kThreads = 256;
// Workgroups when the caller passes 0: the measured knee (2: 22 GB/s and no
// visible compute stall; 4: no faster, 4x the stall; 16: SDMA's stall).
constexpr int kDefaultWorkgroups = 2;
constexpr int kMaxWorkgroups = 64;
constexpr int kSegsPerLaunch = 32;

struct Seg {
  uint8_t *dst;
  const uint8_t *src;
  int64_t bytes;
};

struct Segs {
  Seg seg[kSegsPerLaunch];
  int32_t count;
};

__device__ inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

__global__ __launch_bounds__(kThreads) void h2d_copy_kernel(Segs segs) {
  const int64_t tid = blockIdx.x * int64_t{kThreads} + threadIdx.x;
  const int64_t stride = int64_t{gridDim.x} * kThreads;
  for (int k = 0; k < segs.count; ++k) {
    const Seg s = segs.seg[k];
    int64_t done = 0;
    if (aligned16(s.dst) && aligned16(s.src)) {
      // 16-byte vectors, four in flight per thread.
      const uint4 *src = reinterpret_cast<const uint4 *>(s.src);
      uint4 *dst = reinterpret_cast<uint4 *>(s.dst);
      const int64_t n = s.bytes / 16;
      int64_t i = tid;
      for (; i + 3 * stride < n; i += 4 * stride) {
        const uint4 a = src[i];
        const uint4 b = src[i + stride];
        const uint4 c = src[i + 2 * stride];
        const uint4 d = src[i + 3 * stride];
        dst[i] = a;
        dst[i + stride] = b;
        dst[i + 2 * stride] = c;
        dst[i + 3 * stride] = d;
      }
      for (; i < n; i += stride)
        dst[i] = src[i];
      done = n * 16;
    }
    // The tail, or a misaligned segment: bytes.
    for (int64_t i = done + tid; i < s.bytes; i += stride)
      s.dst[i] = s.src[i];
  }
}

} // namespace

extern "C" {

int32_t turbine_memcpy_h2d_kernel(turbine_ctx *ctx, turbine_stream *s,
                                  const turbine_copy_seg *segs, int32_t count,
                                  int32_t workgroups) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_memcpy_h2d_kernel");
  if (count < 0 || (count > 0 && segs == nullptr)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_memcpy_h2d_kernel: no segments (count " +
                    std::to_string(count) + ")");
  }
  if (workgroups < 0 || workgroups > kMaxWorkgroups) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_memcpy_h2d_kernel: workgroups " +
                    std::to_string(workgroups) + " outside 0.." +
                    std::to_string(kMaxWorkgroups));
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  // Resolve every segment before launching anything, so a bad one refuses the
  // call with nothing enqueued.
  std::vector<Seg> resolved;
  resolved.reserve(static_cast<size_t>(count));
  for (int32_t i = 0; i < count; ++i) {
    const turbine_copy_seg &g = segs[i];
    if (g.bytes == 0)
      continue;
    if (g.dst == nullptr || g.src == nullptr) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_memcpy_h2d_kernel: NULL pointer in segment " +
                      std::to_string(i));
    }
    void *dev_src = nullptr;
    const hipError_t err =
        hipHostGetDevicePointer(&dev_src, const_cast<void *>(g.src), 0);
    if (err != hipSuccess || dev_src == nullptr) {
      (void)hipGetLastError();
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_memcpy_h2d_kernel: segment " + std::to_string(i) +
                      " source is not pinned host memory of this context (" +
                      hipGetErrorName(err) + ")");
    }
    resolved.push_back({static_cast<uint8_t *>(g.dst),
                        static_cast<const uint8_t *>(dev_src),
                        static_cast<int64_t>(g.bytes)});
  }
  hipStream_t stream =
      s != nullptr ? reinterpret_cast<hipStream_t>(s) : ctx->stream;
  const int grid = workgroups == 0 ? kDefaultWorkgroups : workgroups;
  for (size_t at = 0; at < resolved.size(); at += kSegsPerLaunch) {
    Segs launch{};
    for (size_t k = at; k < resolved.size() && k < at + kSegsPerLaunch; ++k)
      launch.seg[launch.count++] = resolved[k];
    h2d_copy_kernel<<<grid, kThreads, 0, stream>>>(launch);
    if (int32_t rc = check_hip(ctx, hipGetLastError(),
                               "turbine_memcpy_h2d_kernel launch");
        rc != TURBINE_OK)
      return rc;
  }
  return TURBINE_OK;
}

} // extern "C"
