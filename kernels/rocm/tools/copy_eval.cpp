// turbine_copy_eval: a decode-like compute loop on one stream while H2D KV
// "promotions" run on a second non-blocking stream; prints the compute
// slowdown per MiB copied (slope), the copy rate with and without compute and
// the step time (perf log 6b "Promotion copy kernel"). A lab tool, not loaded
// by the server. Environment: CB_NOUP (no per-step upload), CB_ONLY
// (small|stream: one compute kernel kind), CB_SRC=dev (copy from device
// memory: no host link), CB_DIR=d2h (device -> pinned host copies, a KV
// demotion: SDMA, or the copy kernel writing the mapped host buffer), CB_ENV
// (a label echoed in the output).
//
//   turbine_copy_eval <mode> <seg_kib> <grid> <total_mib> [steps] [pairs]
//     mode: none | sdma | kernel
//     seg_kib: copy segment size (one hipMemcpyAsync / one kernel op)
//     grid: workgroups of the copy kernel (kernel mode)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t err_ = (x);                                                     \
    if (err_ != hipSuccess) {                                                  \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                \
              hipGetErrorString(err_));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Small latency-bound kernel (rmsnorm-like, one row per block).
__global__ void small_k(const float *x, float *y, int n) {
  __shared__ float s[256];
  float a = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    a += x[blockIdx.x * n + i] * x[blockIdx.x * n + i];
  s[threadIdx.x] = a;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o)
      s[threadIdx.x] += s[threadIdx.x + o];
    __syncthreads();
  }
  float r = rsqrtf(s[0] / n + 1e-6f);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    y[blockIdx.x * n + i] = x[blockIdx.x * n + i] * r;
}

// Bandwidth-bound kernel (GEMV-like weight stream).
__global__ void stream_k(const uint4 *w, uint4 *out, int64_t n) {
  uint4 acc = {0, 0, 0, 0};
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    uint4 v = w[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x & 0xffff) == 0x1234567)
    out[0] = acc;
}

struct Op {
  uint4 *dst;
  const uint4 *src;
  int64_t n16;
};
constexpr int kMaxOps = 32;
struct Ops {
  Op op[kMaxOps];
  int count;
};

// Copy kernel: each op split across all blocks; uint4 loads from host memory.
__global__ void __launch_bounds__(256) copy_k(Ops ops) {
  for (int k = 0; k < ops.count; ++k) {
    const Op o = ops.op[k];
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    for (; i + 3 * stride < o.n16; i += 4 * stride) {
      uint4 a = *(o.src + i);
      uint4 b = *(o.src + i + stride);
      uint4 c = *(o.src + i + 2 * stride);
      uint4 d = *(o.src + i + 3 * stride);
      o.dst[i] = a;
      o.dst[i + stride] = b;
      o.dst[i + 2 * stride] = c;
      o.dst[i + 3 * stride] = d;
    }
    for (; i < o.n16; i += stride)
      o.dst[i] = *(o.src + i);
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: turbine_copy_eval none|sdma|kernel seg_kib grid total_mib "
            "[steps] [pairs]\n");
    return 2;
  }
  std::string mode = argv[1];
  const int64_t seg = (int64_t)atoi(argv[2]) * 1024;
  const int grid = atoi(argv[3]);
  const int64_t total = (int64_t)atoi(argv[4]) << 20;
  const int steps = argc > 5 ? atoi(argv[5]) : 40;
  const int pairs = argc > 6 ? atoi(argv[6]) : 300;
  const int ops_per_launch = 16; // one block's layers per launch (kernel mode)

  CK(hipSetDevice(0));
  hipStream_t cs, xs;
  CK(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking));
  CK(hipStreamCreateWithFlags(&xs, hipStreamNonBlocking));

  // Compute buffers: 8 MiB weights per stream kernel (L2-busting ring of 64).
  const int64_t wbytes = 8 << 20, wring = 64;
  uint4 *w, *wout;
  CK(hipMalloc(&w, wbytes * wring));
  CK(hipMemset(w, 1, wbytes * wring));
  CK(hipMalloc(&wout, 64));
  float *nx, *ny;
  const int rows = 16, cols = 2048;
  CK(hipMalloc(&nx, rows * cols * 4));
  CK(hipMalloc(&ny, rows * cols * 4));
  CK(hipMemset(nx, 0, rows * cols * 4));
  // Per-step small H2D upload on the compute stream (decode's batch upload).
  void *up_h, *up_d;
  CK(hipHostMalloc(&up_h, 16384, hipHostMallocDefault));
  CK(hipMalloc(&up_d, 16384));

  // Promotion buffers: pinned host (like L1: hipHostMallocDefault) -> device.
  const int64_t ring = std::max<int64_t>(total, 64 << 20);
  uint8_t *hsrc, *ddst;
  CK(hipHostMalloc(&hsrc, ring, hipHostMallocDefault));
  for (int64_t i = 0; i < ring; i += 8)
    *reinterpret_cast<uint64_t *>(hsrc + i) =
        (uint64_t)i * 0x9E3779B97F4A7C15ull;
  void *hsrc_dev = nullptr;
  CK(hipHostGetDevicePointer(&hsrc_dev, hsrc, 0));
  CK(hipMalloc(&ddst, ring));

  const bool noup = getenv("CB_NOUP") != nullptr;
  const char *only = getenv("CB_ONLY"); // small | stream
  if (getenv("CB_SRC") && std::string(getenv("CB_SRC")) == "dev") {
    // Copy source in device memory instead (no PCIe): isolates the link.
    uint8_t *dsrc;
    CK(hipMalloc(&dsrc, ring));
    CK(hipMemcpy(dsrc, hsrc, ring, hipMemcpyHostToDevice));
    hsrc_dev = dsrc;
  }
  const hipMemcpyKind skind = hsrc_dev != (void *)hsrc && getenv("CB_SRC")
                                  ? hipMemcpyDeviceToDevice
                                  : hipMemcpyHostToDevice;
  const uint8_t *ssrc =
      getenv("CB_SRC") ? static_cast<uint8_t *>(hsrc_dev) : hsrc;
  // CB_DIR=d2h: demotions. The device buffer holds the pattern and is copied
  // to a second pinned buffer (SDMA, or the kernel storing to its mapping).
  const bool d2h = getenv("CB_DIR") && std::string(getenv("CB_DIR")) == "d2h";
  uint8_t *hdst = nullptr;
  void *hdst_dev = nullptr;
  if (d2h) {
    CK(hipMemcpy(ddst, hsrc, ring, hipMemcpyHostToDevice));
    CK(hipHostMalloc(&hdst, ring, hipHostMallocDefault));
    memset(hdst, 0, ring);
    CK(hipHostGetDevicePointer(&hdst_dev, hdst, 0));
  }
  // One copy batch of `total` bytes on stream xs (the promotion, or with
  // CB_DIR=d2h the demotion).
  auto issue_copy = [&]() {
    if (mode == "sdma") {
      for (int64_t off = 0; off < total; off += seg) {
        const int64_t b = std::min(seg, total - off);
        if (d2h)
          CK(hipMemcpyAsync(hdst + off % ring, ddst + off % ring, b,
                            hipMemcpyDeviceToHost, xs));
        else
          CK(hipMemcpyAsync(ddst + off % ring, ssrc + off % ring, b, skind,
                            xs));
      }
    } else if (mode == "kernel") {
      Ops ops{};
      for (int64_t off = 0; off < total; off += seg) {
        const int64_t b = std::min(seg, total - off);
        if (d2h)
          ops.op[ops.count++] = {
              reinterpret_cast<uint4 *>(static_cast<uint8_t *>(hdst_dev) +
                                        off % ring),
              reinterpret_cast<const uint4 *>(ddst + off % ring), b / 16};
        else
          ops.op[ops.count++] = {
              reinterpret_cast<uint4 *>(ddst + off % ring),
              reinterpret_cast<const uint4 *>(static_cast<uint8_t *>(hsrc_dev) +
                                              off % ring),
              b / 16};
        if (ops.count == ops_per_launch || off + seg >= total) {
          copy_k<<<grid, 256, 0, xs>>>(ops);
          ops.count = 0;
        }
      }
    }
  };
  auto step = [&](int s) {
    if (!noup)
      CK(hipMemcpyAsync(up_d, up_h, 16384, hipMemcpyHostToDevice, cs));
    for (int p = 0; p < pairs; ++p) {
      if (!only || std::string(only) == "small")
        small_k<<<rows, 256, 0, cs>>>(nx, ny, cols);
      if (!only || std::string(only) == "stream")
        stream_k<<<256, 256, 0, cs>>>(
            reinterpret_cast<uint4 *>(reinterpret_cast<uint8_t *>(w) +
                                      ((s * pairs + p) % wring) * wbytes),
            wout, wbytes / 16);
    }
  };

  std::vector<hipEvent_t> ev(steps + 1);
  for (auto &e : ev)
    CK(hipEventCreate(&e));
  hipEvent_t c0, c1, gate;
  CK(hipEventCreate(&c0));
  CK(hipEventCreate(&c1));
  CK(hipEventCreate(&gate));

  auto run_steps = [&](bool copy, std::vector<float> &st, float &copy_ms) {
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(ev[0], cs));
    for (int s = 0; s < steps; ++s) {
      step(s);
      CK(hipEventRecord(ev[s + 1], cs));
      if (copy && s == 1) {
        // Copies start once step 2 begins.
        CK(hipStreamWaitEvent(xs, ev[2], 0));
        CK(hipEventRecord(c0, xs));
        issue_copy();
        CK(hipEventRecord(c1, xs));
      }
    }
    CK(hipDeviceSynchronize());
    st.resize(steps);
    for (int s = 0; s < steps; ++s)
      CK(hipEventElapsedTime(&st[s], ev[s], ev[s + 1]));
    copy_ms = 0;
    if (copy)
      CK(hipEventElapsedTime(&copy_ms, c0, c1));
  };

  std::vector<float> base, with;
  float cm = 0;
  run_steps(false, base, cm); // warm-up
  if (mode != "none")
    run_steps(true, with, cm); // warm the copy path
  run_steps(false, base, cm);
  std::vector<float> sb(base.begin() + 2, base.end());
  std::sort(sb.begin(), sb.end());
  const float bmed = sb[sb.size() / 2];

  float copy_ms = 0;
  double extra = 0, worst = 0;
  int overlapped = 0;
  if (mode != "none") {
    run_steps(true, with, copy_ms);
    // Steps from index 2 on; count extra over the baseline median.
    for (int s = 2; s < steps; ++s) {
      double x = with[s] - bmed;
      extra += x;
      worst = std::max(worst, (double)with[s]);
      if (x > 0.05 * bmed)
        ++overlapped;
    }
    // Verify the bytes.
    std::vector<uint8_t> back(std::min<int64_t>(total, ring));
    if (d2h)
      memcpy(back.data(), hdst, back.size());
    else
      CK(hipMemcpy(back.data(), ddst, back.size(), hipMemcpyDeviceToHost));
    if (memcmp(back.data(), hsrc, back.size()) != 0) {
      fprintf(stderr, "MISMATCH\n");
      return 1;
    }
  }
  // Copy alone (no compute).
  float alone_ms = 0;
  if (mode != "none") {
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(c0, xs));
    issue_copy();
    CK(hipEventRecord(c1, xs));
    CK(hipDeviceSynchronize());
    CK(hipEventElapsedTime(&alone_ms, c0, c1));
  }
  const double mib = (double)total / (1 << 20);
  printf("mode=%s seg_kib=%lld grid=%d mib=%.0f base_step_ms=%.3f "
         "copy_ms=%.2f gbps=%.2f alone_gbps=%.2f extra_ms=%.2f "
         "slope_ms_per_mib=%.4f worst_step_ms=%.3f slowed_steps=%d dir=%s "
         "env=%s\n",
         mode.c_str(), (long long)(seg / 1024), grid, mib, bmed, copy_ms,
         copy_ms > 0 ? total / copy_ms / 1e6 : 0.0,
         alone_ms > 0 ? total / alone_ms / 1e6 : 0.0, extra,
         mib > 0 ? extra / mib : 0.0, worst, overlapped, d2h ? "d2h" : "h2d",
         getenv("CB_ENV") ? getenv("CB_ENV") : "-");
  return 0;
}
