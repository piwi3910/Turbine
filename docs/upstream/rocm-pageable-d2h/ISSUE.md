# Concurrent device-to-host copies into pageable memory corrupt a host page shared by two destinations (ROCm 7.14.1, gfx1201)

## Summary

Two host threads each copy 8 MiB from their own GPU into **pageable** host memory. When the two
destinations share one host page (one ends and the other begins inside the same 4 KiB page, as two
threads' heap allocations can), the bytes of that shared page are sometimes wrong in one of the
destinations after the copy has completed (`hipStreamSynchronize` after `hipMemcpyAsync`, or
`hipMemcpy`). The device data is correct: the same buffer read into pinned memory, or into
pageable memory that shares no page with another in-flight copy, is always right.

## Environment

- ROCm 7.14.1 (`hipRuntimeGetVersion` / `hipDriverGetVersion`: 71460850), AMD SMI 26.5.0
- 2 × AMD Radeon AI PRO R9700 (Navi 48, gfx1201, 32 GB), no peer-to-peer link between them
- Linux 7.1.8 (Debian 13), in-kernel amdgpu driver; firmware CP_MEC1 3430, RLC 12484000,
  SDMA0/1 7966358, PSP_SOSDRV 00.3A.12.14
- One process, one host thread per GPU

## Expected and actual behaviour

Expected: after `hipMemcpyAsync(dst, dev, n, hipMemcpyDeviceToHost, stream)` and
`hipStreamSynchronize(stream)` (or after `hipMemcpy(dst, dev, n, hipMemcpyDeviceToHost)`), `dst`
holds the device bytes, whatever other threads copy concurrently into other host memory.

Actual: when another thread's concurrent device-to-host copy targets memory in the same host
page, the bytes of the shared page in one destination are occasionally wrong: garbage that
matches neither the source, nor the previous contents, nor the other copy's data. The wrong
range is always within the shared page (up to the whole 4,080-byte part of it that belongs to
the destination), and it covers exactly the destination's part of that page.

## Minimal repro

`repro.hip` below (plain HIP, no other library):

```
hipcc -O2 --offload-arch=gfx1201 -o repro repro.hip -lpthread
./repro async-shared 2 busy 2000
./repro sync-shared  2 busy 2000
./repro async-shared 2 idle 2000
./repro pinned       2 busy 2000
./repro async-fresh  2 busy 2000
./repro async-shared 1 busy 2000
```

Each thread fills an 8 MiB device buffer with a known pattern by a kernel, then every round
writes 4 MiB of junk to a second device buffer (pageable host-to-device) and reads the 8 MiB
back into its destination, then compares. `-shared`: one fresh `calloc`'d region per round holds
both threads' destinations back to back at byte `16 + t × 8 MiB`, so the first destination's
last (partial) page is the second destination's first page. `-fresh`: separate `calloc`'d
destinations (no shared page). `busy`: kernels on a second stream of each GPU stream through
mapped coherent host memory during the loop (host-memory traffic from both GPUs, as a
collective library exchanging data through host memory produces).

## Results (2,000 rounds per thread and variant)

| variant                                               | destination          | device 0 bad rounds | device 1 bad rounds         |
| ----------------------------------------------------- | -------------------- | ------------------- | --------------------------- |
| `hipMemcpyAsync`, 2 threads, busy                     | shared page          | 0                   | 1 (68 words, first page)    |
| `hipMemcpy`, 2 threads, busy                          | shared page          | 0                   | 2 (1,016 words, first page) |
| `hipMemcpyAsync`, 2 threads, idle                     | shared page          | 0                   | 1 (660 words, first page)   |
| `hipMemcpyAsync`, pinned destination, 2 threads, busy | pinned               | 0                   | 0                           |
| `hipMemcpyAsync`, 2 threads, busy                     | separate allocations | 0                   | 0                           |
| `hipMemcpyAsync`, 1 thread (device 0), busy           | shared page          | 0                   | —                           |

A second run gave 1/2,000 and 4/2,000 in the first two rows. The wrong rounds are among the first
rounds of a run. In our application (8 MiB reads into freshly allocated heap buffers, two
threads, with two GPUs exchanging data through host memory between the reads) the rate was much
higher: 20 of 240 reads, always the destination's last partial page, exactly `(end address mod
4096) / 4` words, i.e. the bytes of the page shared with the next heap allocation.

## Workaround

Copy through pinned (page-locked, `hipHostMalloc`) staging buffers owned by each thread and
memcpy on the host; no wrong read was seen through pinned memory in any variant.
In our two-GPU application runs, copying through pinned staging gave no wrong reads: a stress
test read back and compared 221,000 collective outputs, and tensor-parallel serving cross-checked
about 657,000 collective results between the two GPUs with no mismatch detected.

## `repro.hip`

```cpp
// Two threads, two GPUs: concurrent device-to-host copies into PAGEABLE host memory whose
// destinations share one host page return wrong bytes in that shared page.
//
// Build: hipcc -O2 --offload-arch=gfx1201 -o repro repro.hip -lpthread
// Run:   ./repro <mode> <threads> <busy|idle> [rounds]
//   mode    async   hipMemcpyAsync (device to host) on a non-blocking stream, then
//                   hipStreamSynchronize
//           sync    hipMemcpy (device to host)
//           pinned  hipMemcpyAsync into hipHostMalloc memory, then hipStreamSynchronize
//           with a suffix choosing the pageable destination:
//             -shared  one fresh calloc'd region per round holds every thread's 8 MiB
//                      destination back to back at byte 16 + t * 8 MiB, so thread t's last
//                      (partial) page is thread t + 1's first
//             -fresh   a fresh calloc'd 8 MiB per thread per round (no page shared)
//             (none)   one 8 MiB buffer per thread, allocated once and memset each round
//   threads 2 (devices 0 and 1 concurrently) or 1 (device 0 alone, the control)
//   busy    each device also runs, on a second stream for the whole loop, kernels streaming
//           read-modify-writes through 16 MiB of mapped coherent host memory
//   rounds  default 200
// Each thread fills an 8 MiB device buffer with a known pattern by a kernel (no host-to-device
// copy of it), then every round writes 4 MiB of junk to a second device buffer (pageable
// host-to-device copy), reads the 8 MiB back into its destination and compares it with the
// pattern. Prints per thread the rounds with wrong bytes and the first one's wrong range.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kWords = 2u << 20;     // 8 MiB of uint32 words
constexpr size_t kJunkWords = 1u << 20; // 4 MiB
int g_rounds = 200;
int g_threads = 2;

std::atomic<int> g_arrived{0};
std::atomic<int> g_generation{0};
std::atomic<uint8_t *> g_region{nullptr};

void barrier() {
  const int gen = g_generation.load();
  if (g_arrived.fetch_add(1) + 1 == g_threads) {
    g_arrived.store(0);
    g_generation.fetch_add(1);
  } else {
    while (g_generation.load() == gen) {
    }
  }
}

__host__ __device__ inline uint32_t pattern(uint64_t i, uint32_t seed) {
  uint64_t z = i * 0x9E3779B97F4A7C15ull + seed;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return static_cast<uint32_t>(z ^ (z >> 31));
}

__global__ void fill(uint32_t *p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    p[i] = pattern(i, seed);
}

__global__ void churn(uint32_t *host, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    host[i] = host[i] * 3u + 1u;
}

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_));      \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

bool ends_with(const std::string &s, const std::string &suffix) {
  return s.size() > suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void run(int device, std::string mode, bool busy, std::string *report) {
  const bool shared = ends_with(mode, "-shared");
  const bool fresh = ends_with(mode, "-fresh");
  const std::string base =
      shared ? mode.substr(0, mode.size() - 7)
             : fresh ? mode.substr(0, mode.size() - 6) : mode;
  CHECK(hipSetDevice(device));
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  std::atomic<bool> stop{false};
  std::thread churner;
  uint32_t *mapped = nullptr;
  constexpr size_t kMapped = 4u << 20;
  if (busy) {
    CHECK(hipHostMalloc(reinterpret_cast<void **>(&mapped), kMapped * 4,
                        hipHostMallocMapped | hipHostMallocCoherent |
                            hipHostMallocPortable));
    churner = std::thread([device, mapped, &stop] {
      CHECK(hipSetDevice(device));
      hipStream_t b;
      CHECK(hipStreamCreateWithFlags(&b, hipStreamNonBlocking));
      uint32_t *dev = nullptr;
      CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&dev), mapped, 0));
      while (!stop.load()) {
        hipLaunchKernelGGL(churn, dim3(64), dim3(256), 0, b, dev, kMapped);
        CHECK(hipStreamSynchronize(b));
      }
      CHECK(hipStreamDestroy(b));
    });
  }
  uint32_t *buf = nullptr, *junk_dev = nullptr;
  CHECK(hipMalloc(&buf, kWords * 4));
  CHECK(hipMalloc(&junk_dev, kJunkWords * 4));
  const uint32_t seed = 1234 + device;
  hipLaunchKernelGGL(fill, dim3(256), dim3(256), 0, s, buf, kWords, seed);
  CHECK(hipStreamSynchronize(s));
  std::vector<uint32_t> want(kWords);
  for (size_t i = 0; i < kWords; ++i)
    want[i] = pattern(i, seed);
  std::vector<uint32_t> junk(kJunkWords, 0xdeadbeefu + device);
  std::vector<uint32_t> reused(kWords);
  uint32_t *pinned = nullptr;
  if (base == "pinned")
    CHECK(hipHostMalloc(reinterpret_cast<void **>(&pinned), kWords * 4,
                        hipHostMallocDefault));
  int bad = 0;
  std::string first;
  for (int round = 0; round < g_rounds; ++round) {
    if (base == "sync") {
      CHECK(hipMemcpy(junk_dev, junk.data(), kJunkWords * 4,
                      hipMemcpyHostToDevice));
    } else {
      CHECK(hipMemcpyAsync(junk_dev, junk.data(), kJunkWords * 4,
                           hipMemcpyHostToDevice, s));
      CHECK(hipStreamSynchronize(s));
    }
    uint32_t *dst = reused.data();
    void *owned = nullptr;
    if (shared) {
      if (device == 0)
        g_region.store(
            static_cast<uint8_t *>(std::calloc(1, 2 * kWords * 4 + 64)));
      barrier();
      dst = reinterpret_cast<uint32_t *>(g_region.load() + 16 +
                                         device * kWords * 4);
    } else if (fresh) {
      owned = std::calloc(kWords, 4);
      dst = static_cast<uint32_t *>(owned);
    } else {
      std::memset(dst, 0, kWords * 4);
    }
    if (base == "sync") {
      CHECK(hipMemcpy(dst, buf, kWords * 4, hipMemcpyDeviceToHost));
    } else if (base == "pinned") {
      CHECK(hipMemcpyAsync(pinned, buf, kWords * 4, hipMemcpyDeviceToHost, s));
      CHECK(hipStreamSynchronize(s));
      dst = pinned;
    } else {
      CHECK(hipMemcpyAsync(dst, buf, kWords * 4, hipMemcpyDeviceToHost, s));
      CHECK(hipStreamSynchronize(s));
    }
    size_t lo = kWords, hi = 0, n = 0;
    for (size_t i = 0; i < kWords; ++i) {
      if (dst[i] != want[i]) {
        lo = i < lo ? i : lo;
        hi = i;
        ++n;
      }
    }
    if (n > 0) {
      if (bad == 0)
        first = "round " + std::to_string(round) + ": " + std::to_string(n) +
                " words wrong in [" + std::to_string(lo) + ", " +
                std::to_string(hi) + "]";
      ++bad;
    }
    std::free(owned);
    if (shared) {
      barrier();
      if (device == 0)
        std::free(g_region.exchange(nullptr));
      barrier();
    }
  }
  if (busy) {
    stop.store(true);
    churner.join();
    CHECK(hipHostFree(mapped));
  }
  *report = "device " + std::to_string(device) + " mode " + mode +
            (busy ? " busy" : "") + ": " + std::to_string(bad) + "/" +
            std::to_string(g_rounds) + " rounds wrong" +
            (bad ? "; first " + first : "");
  if (pinned)
    CHECK(hipHostFree(pinned));
  CHECK(hipFree(buf));
  CHECK(hipFree(junk_dev));
  CHECK(hipStreamDestroy(s));
}

} // namespace

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "async-shared";
  g_threads = argc > 2 ? std::atoi(argv[2]) : 2;
  const bool busy = argc > 3 && std::string(argv[3]) == "busy";
  if (argc > 4)
    g_rounds = std::atoi(argv[4]);
  int runtime = 0, driver = 0;
  CHECK(hipRuntimeGetVersion(&runtime));
  CHECK(hipDriverGetVersion(&driver));
  std::vector<std::string> reports(g_threads);
  std::vector<std::thread> pool;
  for (int t = 0; t < g_threads; ++t)
    pool.emplace_back(run, t, mode, busy, &reports[t]);
  for (auto &t : pool)
    t.join();
  for (const auto &r : reports)
    std::printf("repro hip runtime %d driver %d threads %d %s\n", runtime,
                driver, g_threads, r.c_str());
  return 0;
}
```
