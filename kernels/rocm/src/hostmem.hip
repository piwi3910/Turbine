// ABI v2.7 (Phase 5, decision "P5: small-message all-reduce latency on
// novanas"): host-mapped memory and the one-shot collectives over it, for
// devices of one process without a peer-to-peer path (the two R9700s of
// novanas: no GPU<->GPU link, so RCCL stages through host memory with its own
// proxy thread and ~21 us per small all-reduce).
//
// Reuse first: no existing kernel serves this. vLLM's custom all-reduce and
// quick reduce and mscclpp exchange through peer (IPC) mappings, which need
// peer-to-peer access; RCCL's no-p2p path is its SHM transport (measured in
// the decision above).
//
// Protocol (one kernel per step and rank; see the header for the contract):
// block b of every rank handles the same byte range of the part, split into
// kStepBytes pipeline steps. For step k the block writes its contribution to
// its rank's slot, publishes tag(seq, k) into its flag word with a
// system-scope release (every thread fences, a barrier, then thread 0 stores),
// writes step k + 1 before it waits, then thread 0 polls every peer's flag
// word of block b with system-scope loads until it reaches the tag, the block
// acquires and reads the peers' contributions. A reduction combines the
// ranks' values in rank order 0 .. world - 1 in F32 (one rounding for BF16):
// every rank computes the same bits, and they equal the host reference
// backend's (crates/turbine-distributed collective/host.rs).
//
// Slot reuse: rank r writes slot (seq & 1) at step seq only after its step
// seq - 1 completed, and that step waited for a flag every peer published
// during its own step seq - 1, which runs after the peer's step seq - 2 (the
// last reader of that slot) has completed on its stream.
//
// Device-sequenced steps (v2.8, turbine_mapped_collective_dseq): instead of
// the seq baked into the launch, thread 0 of every block reads the channel's
// device counter (seq = counter + 1) and counts its block in the completion
// word; the last block to arrive (acq_rel chain on that word) resets it and
// stores seq into the counter. Every block's read happens before that store,
// and the next step on the stream starts after this one completes, so a
// captured graph advances the sequence on every replay.
//
// Every wait is bounded: thread 0 checks abort_word and the device wall clock
// every 16 polls; past the deadline it stores the timeout reason into
// abort_word (a plain system-scope store: no PCIe atomics are needed) and the
// block ends, and every other waiting block of every rank then sees the
// non-zero word and ends as well.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "bf16.hpp"
#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;
using turbine_hip::refuse_while_capturing;

namespace {

constexpr const char *kImpl = "turbine_hip_mapped_oneshot";
constexpr int kThreads = 256;
// 16-byte vectors each thread keeps in flight.
constexpr int kUnroll = 4;
// Bytes of one pipeline step of a block.
constexpr int64_t kStepBytes = 16384;
// A step spreads over as many blocks as have at least this many bytes each
// (one 16-byte vector per thread), up to max_blocks.
constexpr int64_t kBlockMinBytes = 4096;
// tag(seq, k) = seq << kSeqShift | (k + 1).
constexpr int kSeqShift = 24;
constexpr int64_t kMaxSteps = (int64_t{1} << kSeqShift) - 2;
constexpr uint64_t kMaxSeq = (uint64_t{1} << (64 - kSeqShift)) - 1;

struct Args {
  const uint8_t *send;
  uint8_t *recv;
  int64_t bytes, send_stride, recv_stride, part_stride, slot_bytes, per_block;
  uint8_t *slots;
  uint64_t *flags;
  uint32_t *abort_word;
  uint64_t seq;
  // v2.8: device counter (uint64) followed by the completion count (uint32);
  // null for a host-sequenced step.
  uint64_t *seq_counter;
  uint64_t timeout_ticks;
  int32_t kind, rank, world, root, max_blocks;
};

__device__ inline uint64_t load_flag(uint64_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ inline uint32_t load_abort(uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ inline int64_t lmin(int64_t a, int64_t b) { return a < b ? a : b; }

__device__ inline uint64_t tag(uint64_t seq, int64_t k) {
  return (seq << kSeqShift) | static_cast<uint64_t>(k + 1);
}

// ---- element access: F32 values of 16-byte vectors and single elements ----

template <typename T> struct Elem;
template <> struct Elem<float> {
  static constexpr int kPerVec = 4;
  __device__ static void unpack(const uint4 &v, float *o) {
    o[0] = __uint_as_float(v.x);
    o[1] = __uint_as_float(v.y);
    o[2] = __uint_as_float(v.z);
    o[3] = __uint_as_float(v.w);
  }
  __device__ static uint4 pack(const float *a) {
    return uint4{__float_as_uint(a[0]), __float_as_uint(a[1]),
                 __float_as_uint(a[2]), __float_as_uint(a[3])};
  }
  __device__ static float load(const uint8_t *p) {
    return *reinterpret_cast<const float *>(p);
  }
  __device__ static void store(uint8_t *p, float v) {
    *reinterpret_cast<float *>(p) = v;
  }
};
template <> struct Elem<turbine_hip::Bf16> {
  static constexpr int kPerVec = 8;
  __device__ static float half_word(uint32_t w, int hi) {
    return turbine_hip::bf16_to_f32(
        turbine_hip::Bf16{static_cast<uint16_t>(hi ? w >> 16 : w & 0xffffu)});
  }
  __device__ static void unpack(const uint4 &v, float *o) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    for (int i = 0; i < 4; ++i) {
      o[2 * i] = half_word(w[i], 0);
      o[2 * i + 1] = half_word(w[i], 1);
    }
  }
  __device__ static uint32_t word(float lo, float hi) {
    return static_cast<uint32_t>(turbine_hip::f32_to_bf16(lo).bits) |
           (static_cast<uint32_t>(turbine_hip::f32_to_bf16(hi).bits) << 16);
  }
  __device__ static uint4 pack(const float *a) {
    return uint4{word(a[0], a[1]), word(a[2], a[3]), word(a[4], a[5]),
                 word(a[6], a[7])};
  }
  __device__ static float load(const uint8_t *p) {
    return turbine_hip::bf16_to_f32(
        turbine_hip::Bf16{*reinterpret_cast<const uint16_t *>(p)});
  }
  __device__ static void store(uint8_t *p, float v) {
    *reinterpret_cast<uint16_t *>(p) = turbine_hip::f32_to_bf16(v).bits;
  }
};

// The host reference's combine: a + x, or x when x > a (so NaN handling and
// ties match it bit for bit).
template <int OP> __device__ inline float combine(float a, float x) {
  if constexpr (OP == TURBINE_REDUCE_SUM) {
    return a + x;
  } else {
    return x > a ? x : a;
  }
}

// dst[0, n) = src[0, n); VEC: both 16-byte aligned.
template <bool VEC>
__device__ void copy_bytes(uint8_t *dst, const uint8_t *src, int64_t n) {
  int64_t done = 0;
  if constexpr (VEC) {
    const int64_t nv = n / 16;
    const uint4 *s4 = reinterpret_cast<const uint4 *>(src);
    uint4 *d4 = reinterpret_cast<uint4 *>(dst);
    int64_t i = threadIdx.x;
    // kUnroll independent loads in flight per thread before their stores.
    for (; i + (kUnroll - 1) * kThreads < nv; i += kUnroll * kThreads) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        v[u] = s4[i + u * kThreads];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        d4[i + u * kThreads] = v[u];
      }
    }
    for (; i < nv; i += kThreads) {
      d4[i] = s4[i];
    }
    done = nv * 16;
  }
  for (int64_t i = done + threadIdx.x; i < n; i += kThreads) {
    dst[i] = src[i];
  }
}

// out[0, n) = combination of srcs[0 .. world)[0, n) in rank order; n is a
// multiple of sizeof(T); VEC: every pointer 16-byte aligned.
template <typename T, int OP, bool VEC>
__device__ void reduce_bytes(uint8_t *out, const uint8_t *const *srcs,
                             int world, int64_t n) {
  int64_t done = 0;
  if constexpr (VEC) {
    constexpr int E = Elem<T>::kPerVec;
    const int64_t nv = n / 16;
    int64_t i = threadIdx.x;
    // kUnroll independent vectors per thread: each source's loads are in flight
    // together.
    for (; i + (kUnroll - 1) * kThreads < nv; i += kUnroll * kThreads) {
      float acc[kUnroll][E];
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        v[u] = reinterpret_cast<const uint4 *>(srcs[0])[i + u * kThreads];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        Elem<T>::unpack(v[u], acc[u]);
      }
      for (int q = 1; q < world; ++q) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          v[u] = reinterpret_cast<const uint4 *>(srcs[q])[i + u * kThreads];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          float x[E];
          Elem<T>::unpack(v[u], x);
#pragma unroll
          for (int e = 0; e < E; ++e) {
            acc[u][e] = combine<OP>(acc[u][e], x[e]);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        reinterpret_cast<uint4 *>(out)[i + u * kThreads] =
            Elem<T>::pack(acc[u]);
      }
    }
    for (; i < nv; i += kThreads) {
      float acc[E];
      float x[E];
      Elem<T>::unpack(reinterpret_cast<const uint4 *>(srcs[0])[i], acc);
      for (int q = 1; q < world; ++q) {
        Elem<T>::unpack(reinterpret_cast<const uint4 *>(srcs[q])[i], x);
#pragma unroll
        for (int e = 0; e < E; ++e) {
          acc[e] = combine<OP>(acc[e], x[e]);
        }
      }
      reinterpret_cast<uint4 *>(out)[i] = Elem<T>::pack(acc);
    }
    done = nv * 16;
  }
  constexpr int64_t S = sizeof(T);
  for (int64_t i = done / S + threadIdx.x; i < n / S; i += kThreads) {
    float acc = Elem<T>::load(srcs[0] + i * S);
    for (int q = 1; q < world; ++q) {
      acc = combine<OP>(acc, Elem<T>::load(srcs[q] + i * S));
    }
    Elem<T>::store(out + i * S, acc);
  }
}

__device__ inline uint8_t *slot_of(const Args &a, int r) {
  return a.slots +
         (static_cast<int64_t>(a.seq & 1) * a.world + r) * a.slot_bytes;
}

// Rank a.rank's contribution to bytes [r0, r0 + n) of the part into its slot.
template <bool VEC>
__device__ void write_step(const Args &a, int64_t r0, int64_t n) {
  uint8_t *mine = slot_of(a, a.rank);
  switch (a.kind) {
  case TURBINE_MAPPED_REDUCE_SCATTER:
    for (int d = 0; d < a.world; ++d) {
      if (d != a.rank) {
        copy_bytes<VEC>(mine + d * a.part_stride + r0,
                        a.send + d * a.send_stride + r0, n);
      }
    }
    break;
  case TURBINE_MAPPED_BROADCAST:
    if (a.rank == a.root) {
      copy_bytes<VEC>(mine + r0, a.send + r0, n);
    }
    break;
  default:
    copy_bytes<VEC>(mine + r0, a.send + r0, n);
    break;
  }
}

// Every thread's slot writes are visible system-wide before thread 0
// publishes tag(seq, k).
__device__ void publish(const Args &a, int64_t k) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_store(a.flags + a.rank * a.max_blocks + blockIdx.x,
                       tag(a.seq, k), __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Thread 0: waits until every peer's flag of this block reaches want. False
// when the group is aborted or the deadline passes (then the timeout is
// recorded in abort_word).
__device__ bool wait_peers(const Args &a, uint64_t want, uint64_t deadline) {
  for (int q = 0; q < a.world; ++q) {
    if (q == a.rank) {
      continue;
    }
    uint64_t *f = a.flags + q * a.max_blocks + blockIdx.x;
    uint32_t polls = 0;
    while (load_flag(f) < want) {
      if ((++polls & 15u) == 0) {
        if (load_abort(a.abort_word) != 0) {
          return false;
        }
        if (static_cast<uint64_t>(wall_clock64()) > deadline) {
          if (load_abort(a.abort_word) == 0) {
            __hip_atomic_store(a.abort_word,
                               (TURBINE_MAPPED_ABORT_TIMEOUT << 24) |
                                   (static_cast<uint32_t>(a.kind) << 16) |
                                   static_cast<uint32_t>(a.rank),
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          return false;
        }
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
  return true;
}

// Bytes [r0, r0 + n) of the result from every rank's contribution.
template <typename T, int OP, bool VEC>
__device__ void consume_step(const Args &a, int64_t r0, int64_t n) {
  const uint8_t *srcs[TURBINE_MAPPED_MAX_WORLD];
  switch (a.kind) {
  case TURBINE_MAPPED_ALL_GATHER:
    for (int q = 0; q < a.world; ++q) {
      const uint8_t *src = q == a.rank ? a.send : slot_of(a, q);
      copy_bytes<VEC>(a.recv + q * a.recv_stride + r0, src + r0, n);
    }
    break;
  case TURBINE_MAPPED_BROADCAST:
    if (a.rank != a.root) {
      copy_bytes<VEC>(a.recv + r0, slot_of(a, a.root) + r0, n);
    } else if (a.recv != a.send) {
      copy_bytes<VEC>(a.recv + r0, a.send + r0, n);
    }
    break;
  default:
    if constexpr (!std::is_same_v<T, uint8_t>) {
      const bool scatter = a.kind == TURBINE_MAPPED_REDUCE_SCATTER;
      for (int q = 0; q < a.world; ++q) {
        srcs[q] =
            q == a.rank
                ? a.send + (scatter ? a.rank * a.send_stride : 0) + r0
                : slot_of(a, q) + (scatter ? a.rank * a.part_stride : 0) + r0;
      }
      reduce_bytes<T, OP, VEC>(a.recv + r0, srcs, a.world, n);
    }
    break;
  }
}

// v2.8: the step's seq from the device counter (see the header comment);
// every block of the step must call this first, before any early return.
__device__ uint64_t device_seq(const Args &a) {
  __shared__ uint64_t s_seq;
  if (threadIdx.x == 0) {
    uint32_t *done = reinterpret_cast<uint32_t *>(a.seq_counter + 1);
    const uint64_t seq = __hip_atomic_load(a.seq_counter, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT) +
                         1;
    s_seq = seq;
    const uint32_t before = __hip_atomic_fetch_add(
        done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (before + 1 == gridDim.x) {
      __hip_atomic_store(done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(a.seq_counter, seq, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  __syncthreads();
  return s_seq;
}

template <typename T, int OP, bool VEC>
__global__ __launch_bounds__(kThreads) void mapped_collective_kernel(Args a0) {
  __shared__ int s_stop;
  Args a = a0;
  if (a.seq_counter != nullptr) {
    a.seq = device_seq(a);
  }
  const int64_t start = static_cast<int64_t>(blockIdx.x) * a.per_block;
  const int64_t end = lmin(start + a.per_block, a.bytes);
  if (start >= end) {
    return;
  }
  const int64_t steps = (end - start + kStepBytes - 1) / kStepBytes;
  uint64_t deadline = 0;
  if (threadIdx.x == 0) {
    s_stop = load_abort(a.abort_word) != 0;
    deadline = static_cast<uint64_t>(wall_clock64()) + a.timeout_ticks;
  }
  __syncthreads();
  if (s_stop) {
    return;
  }
  auto range = [&](int64_t k, int64_t *n) {
    const int64_t r0 = start + k * kStepBytes;
    *n = lmin(r0 + kStepBytes, end) - r0;
    return r0;
  };
  int64_t n = 0;
  int64_t r0 = range(0, &n);
  write_step<VEC>(a, r0, n);
  publish(a, 0);
  for (int64_t k = 0; k < steps; ++k) {
    if (k + 1 < steps) {
      int64_t n1 = 0;
      const int64_t r1 = range(k + 1, &n1);
      write_step<VEC>(a, r1, n1);
      publish(a, k + 1);
    }
    // Every thread has read the previous s_stop before thread 0 rewrites it.
    __syncthreads();
    if (threadIdx.x == 0) {
      s_stop = !wait_peers(a, tag(a.seq, k), deadline);
    }
    __syncthreads();
    if (s_stop) {
      return;
    }
    // Invalidate this thread's caches before it reads the peers' writes.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    r0 = range(k, &n);
    consume_step<T, OP, VEC>(a, r0, n);
  }
}

template <typename T, int OP>
void launch_typed(hipStream_t s, const Args &a, bool vec, unsigned blocks) {
  if (vec) {
    hipLaunchKernelGGL((mapped_collective_kernel<T, OP, true>), dim3(blocks),
                       dim3(kThreads), 0, s, a);
  } else {
    hipLaunchKernelGGL((mapped_collective_kernel<T, OP, false>), dim3(blocks),
                       dim3(kThreads), 0, s, a);
  }
}

bool is_reduction(int32_t kind) {
  return kind == TURBINE_MAPPED_ALL_REDUCE ||
         kind == TURBINE_MAPPED_REDUCE_SCATTER;
}

int64_t elem_bytes(const turbine_mapped_collective_desc *d) {
  if (!is_reduction(d->kind)) {
    return 1;
  }
  return d->dtype == TURBINE_DTYPE_BF16 ? 2 : 4;
}

int64_t round16(int64_t v) { return (v + 15) / 16 * 16; }

int64_t lmin_host(int64_t a, int64_t b) { return a < b ? a : b; }

// Blocks and bytes per block of a step: equal on every rank for equal bytes
// and max_blocks.
void geometry(const turbine_mapped_collective_desc *d, unsigned *blocks,
              int64_t *per_block) {
  const int64_t want = (d->bytes + kBlockMinBytes - 1) / kBlockMinBytes;
  const int64_t n =
      want < 1 ? 1 : (want > d->max_blocks ? d->max_blocks : want);
  *blocks = static_cast<unsigned>(n);
  *per_block = round16((d->bytes + n - 1) / n);
}

// Why d is not supported, or empty.
std::string refusal(const turbine_mapped_collective_desc *d) {
  if (d == nullptr) {
    return "descriptor is NULL";
  }
  if (d->kind < TURBINE_MAPPED_ALL_REDUCE ||
      d->kind > TURBINE_MAPPED_BROADCAST) {
    return "unknown kind " + std::to_string(d->kind);
  }
  if (d->world < 1 || d->world > TURBINE_MAPPED_MAX_WORLD || d->rank < 0 ||
      d->rank >= d->world) {
    return "rank " + std::to_string(d->rank) + " of world " +
           std::to_string(d->world);
  }
  if (d->kind == TURBINE_MAPPED_BROADCAST &&
      (d->root < 0 || d->root >= d->world)) {
    return "root " + std::to_string(d->root);
  }
  if (d->max_blocks < 1 || d->max_blocks > TURBINE_MAPPED_MAX_BLOCKS) {
    return "max_blocks " + std::to_string(d->max_blocks);
  }
  if (d->timeout_ns <= 0 || d->seq < 1 || d->seq > kMaxSeq) {
    return "timeout_ns " + std::to_string(d->timeout_ns) + " seq " +
           std::to_string(d->seq);
  }
  if (is_reduction(d->kind)) {
    if (d->dtype != TURBINE_DTYPE_BF16 && d->dtype != TURBINE_DTYPE_F32) {
      return "dtype " + std::to_string(d->dtype);
    }
    if (d->reduce_op != TURBINE_REDUCE_SUM &&
        d->reduce_op != TURBINE_REDUCE_MAX) {
      return "reduce_op " + std::to_string(d->reduce_op);
    }
  }
  const int64_t e = elem_bytes(d);
  if (d->bytes < 0 || d->bytes % e != 0) {
    return "bytes " + std::to_string(d->bytes) + " not whole elements";
  }
  if (d->kind == TURBINE_MAPPED_REDUCE_SCATTER &&
      (d->send_stride < d->bytes || d->send_stride % e != 0)) {
    return "send_stride " + std::to_string(d->send_stride);
  }
  if (d->kind == TURBINE_MAPPED_ALL_GATHER && d->recv_stride < d->bytes) {
    return "recv_stride " + std::to_string(d->recv_stride);
  }
  const int64_t parts = d->kind == TURBINE_MAPPED_REDUCE_SCATTER ? d->world : 1;
  if (d->slot_bytes % 16 != 0 || parts * round16(d->bytes) > d->slot_bytes) {
    return "slot_bytes " + std::to_string(d->slot_bytes) + " below " +
           std::to_string(parts) + " x " + std::to_string(round16(d->bytes));
  }
  unsigned blocks = 0;
  int64_t per_block = 0;
  geometry(d, &blocks, &per_block);
  if ((per_block + kStepBytes - 1) / kStepBytes > kMaxSteps) {
    return "too many steps per block";
  }
  return {};
}

bool aligned(const void *p, int64_t a) {
  return reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(a) == 0;
}

// ---- v2.8 copy-engine all-reduce (turbine_mapped_all_reduce_dma) ----
//
// For messages where the one-shot kernels are bound by their ~6.3 GB/s reads
// of mapped host memory, the copy engines move the bytes instead (~12.5-13
// GB/s per direction on novanas). Per chunk k of the message (seq from the
// device counter, as the dseq steps):
//   copy stream:    D2H send[k] -> own slot[k]; publish tag(seq, k) in flag
//                   word rank * max_blocks + k (a one-thread kernel, after
//                   the copy completed);
//   compute stream: wait until every rank's (own included) flag word k
//                   reaches tag(seq, k); H2D each peer's slot[k] -> scratch;
//                   reduce recv[k] = combination of every rank's chunk in
//                   rank order 0 .. world - 1 (F32, one rounding for BF16:
//                   the bits of the one-shot kernel and of the host
//                   reference); after the last chunk, counter += 1.
// The copy stream first waits for the compute stream's work before the call
// (an event), so send is final. Waiting for the own flag orders the reduce
// after the own chunk left the device (recv may equal send). The chunks of
// the D2H and the H2D directions overlap. The slot parity is the caller's
// DMA call count (d->seq & 1), equal on every rank, since the device counter
// cannot address a host-side copy; see the header for the reuse argument.

__global__ void dma_publish_kernel(const uint64_t *counter, uint64_t *flag,
                                   int64_t k) {
  if (threadIdx.x == 0) {
    const uint64_t seq =
        __hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) +
        1;
    __threadfence_system();
    __hip_atomic_store(flag, tag(seq, k), __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

struct DmaWait {
  const uint64_t *counter;
  uint64_t *flags;
  uint32_t *abort_word;
  uint64_t timeout_ticks;
  int32_t world, rank, max_blocks;
};

// Thread 0 waits until every rank's flag word k reaches tag(seq, k); bounded
// like wait_peers (a timeout stores its reason into abort_word).
__global__ void dma_wait_kernel(DmaWait w, int64_t k) {
  if (threadIdx.x != 0) {
    return;
  }
  const uint64_t want =
      tag(__hip_atomic_load(w.counter, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) +
              1,
          k);
  if (load_abort(w.abort_word) != 0) {
    return;
  }
  const uint64_t deadline =
      static_cast<uint64_t>(wall_clock64()) + w.timeout_ticks;
  for (int q = 0; q < w.world; ++q) {
    uint64_t *f = w.flags + q * w.max_blocks + k;
    uint32_t polls = 0;
    while (load_flag(f) < want) {
      if ((++polls & 15u) == 0) {
        if (load_abort(w.abort_word) != 0) {
          return;
        }
        if (static_cast<uint64_t>(wall_clock64()) > deadline) {
          if (load_abort(w.abort_word) == 0) {
            __hip_atomic_store(
                w.abort_word,
                (TURBINE_MAPPED_ABORT_TIMEOUT << 24) |
                    (static_cast<uint32_t>(TURBINE_MAPPED_ALL_REDUCE) << 16) |
                    static_cast<uint32_t>(w.rank),
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
          return;
        }
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
  // Later readers of the peers' chunks are copies ordered after this kernel.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

struct DmaReduce {
  uint8_t *out;
  const uint8_t *own;
  const uint8_t *scratch;
  int64_t scratch_stride, n, per_block;
  uint32_t *abort_word;
  int32_t world, rank;
  // Peer q's part is scratch part q (its slot) instead of q < rank ? q : q - 1.
  bool direct;
};

// out[0, n) = combination of the ranks' chunks in rank order (own: `own`,
// peer q: scratch part q < rank ? q : q - 1); blocks split n.
template <typename T, int OP, bool VEC>
__global__ __launch_bounds__(kThreads) void dma_reduce_kernel(DmaReduce r) {
  if (load_abort(r.abort_word) != 0) {
    return;
  }
  const int64_t start = static_cast<int64_t>(blockIdx.x) * r.per_block;
  const int64_t end = lmin(start + r.per_block, r.n);
  if (start >= end) {
    return;
  }
  const uint8_t *srcs[TURBINE_MAPPED_MAX_WORLD];
  for (int q = 0; q < r.world; ++q) {
    const int part = r.direct ? q : (q < r.rank ? q : q - 1);
    srcs[q] =
        (q == r.rank ? r.own : r.scratch + part * r.scratch_stride) + start;
  }
  reduce_bytes<T, OP, VEC>(r.out + start, srcs, r.world, end - start);
}

__global__ void dma_advance_kernel(uint64_t *counter) {
  if (threadIdx.x == 0) {
    *counter += 1;
  }
}

template <typename T, int OP>
void launch_dma_reduce(hipStream_t s, const DmaReduce &r, bool vec,
                       unsigned blocks) {
  if (vec) {
    hipLaunchKernelGGL((dma_reduce_kernel<T, OP, true>), dim3(blocks),
                       dim3(kThreads), 0, s, r);
  } else {
    hipLaunchKernelGGL((dma_reduce_kernel<T, OP, false>), dim3(blocks),
                       dim3(kThreads), 0, s, r);
  }
}

} // namespace

extern "C" {

int32_t turbine_host_alloc_mapped(turbine_ctx *ctx, size_t bytes, void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_host_alloc_mapped");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_host_alloc_mapped: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  const size_t n = bytes == 0 ? 1 : bytes;
  const std::string what =
      "hipHostMalloc(portable|mapped|coherent) " + std::to_string(n) + " bytes";
  void *p = nullptr;
  if (int32_t rc =
          check_hip(ctx,
                    hipHostMalloc(&p, n,
                                  hipHostMallocPortable | hipHostMallocMapped |
                                      hipHostMallocCoherent),
                    what.c_str());
      rc != TURBINE_OK)
    return rc;
  std::memset(p, 0, n);
  *out = p;
  return TURBINE_OK;
}

int32_t turbine_host_alloc_dma(turbine_ctx *ctx, size_t bytes, void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_host_alloc_dma");
  if (out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_host_alloc_dma: out is NULL");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  const size_t n = bytes == 0 ? 1 : bytes;
  const std::string what =
      "hipHostMalloc(portable) " + std::to_string(n) + " bytes";
  void *p = nullptr;
  if (int32_t rc = check_hip(ctx, hipHostMalloc(&p, n, hipHostMallocPortable),
                             what.c_str());
      rc != TURBINE_OK)
    return rc;
  *out = p;
  return TURBINE_OK;
}

int32_t turbine_host_mapped_device_ptr(turbine_ctx *ctx, void *host,
                                       void **out) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (host == nullptr || out == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_host_mapped_device_ptr: NULL pointer");
  }
  *out = nullptr;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipHostGetDevicePointer(out, host, 0),
                   "hipHostGetDevicePointer");
}

int32_t turbine_host_free_mapped(turbine_ctx *ctx, void *host) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_host_free_mapped");
  if (host == nullptr)
    return TURBINE_OK;
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  return check_hip(ctx, hipHostFree(host), "hipHostFree");
}

int32_t
turbine_mapped_collective_supported(const turbine_mapped_collective_desc *d) {
  return refusal(d).empty() ? 1 : 0;
}

const char *
turbine_mapped_collective_impl(const turbine_mapped_collective_desc *d) {
  (void)d;
  return kImpl;
}

} // extern "C"

namespace {

// The device's wall clock rate (kHz), read once per context.
int32_t ensure_wall_clock(turbine_ctx *ctx) {
  if (ctx->wall_clock_khz <= 0) {
    int khz = 0;
    if (int32_t rc =
            check_hip(ctx,
                      hipDeviceGetAttribute(
                          &khz, hipDeviceAttributeWallClockRate, ctx->device),
                      "hipDeviceGetAttribute(hipDeviceAttributeWallClockRate)");
        rc != TURBINE_OK)
      return rc;
    if (khz <= 0) {
      return fail(ctx, TURBINE_E_DEVICE,
                  "turbine_mapped_collective: the device reports no wall "
                  "clock rate");
    }
    ctx->wall_clock_khz = khz;
  }
  return TURBINE_OK;
}

// Ticks of the device wall clock in timeout_ns (rounded up), saturating.
uint64_t timeout_ticks(const turbine_ctx *ctx, int64_t timeout_ns) {
  // ns * kHz / 1e6, without overflow for any timeout below ~290 years.
  const double ticks = static_cast<double>(timeout_ns) *
                       static_cast<double>(ctx->wall_clock_khz) / 1e6;
  return ticks >= 1.8e19 ? UINT64_MAX : static_cast<uint64_t>(ticks) + 1;
}

// Both entry points: validates d and enqueues its step; seq_counter null =
// host-sequenced (d->seq).
int32_t enqueue_step(turbine_ctx *ctx, const turbine_mapped_collective_desc *d,
                     uint64_t *seq_counter) {
  if (const std::string why = refusal(d); !why.empty()) {
    return fail(ctx, d == nullptr ? TURBINE_E_ARGUMENT : TURBINE_E_UNSUPPORTED,
                "turbine_mapped_collective: " + why);
  }
  if (d->bytes == 0) {
    return TURBINE_OK;
  }
  const int64_t e = elem_bytes(d);
  if (d->send == nullptr || d->recv == nullptr || d->slots == nullptr ||
      d->flags == nullptr || d->abort_word == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_mapped_collective: NULL operand");
  }
  if (!aligned(d->send, e) || !aligned(d->recv, e) || !aligned(d->slots, 16) ||
      !aligned(d->flags, 8) || !aligned(d->abort_word, 4)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_mapped_collective: misaligned operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = ensure_wall_clock(ctx); rc != TURBINE_OK)
    return rc;
  Args a{};
  a.send = static_cast<const uint8_t *>(d->send);
  a.recv = static_cast<uint8_t *>(d->recv);
  a.bytes = d->bytes;
  a.send_stride = d->send_stride;
  a.recv_stride = d->recv_stride;
  a.part_stride = round16(d->bytes);
  a.slot_bytes = d->slot_bytes;
  a.slots = static_cast<uint8_t *>(d->slots);
  a.flags = d->flags;
  a.abort_word = d->abort_word;
  a.seq = d->seq;
  a.seq_counter = seq_counter;
  a.timeout_ticks = timeout_ticks(ctx, d->timeout_ns);
  a.kind = d->kind;
  a.rank = d->rank;
  a.world = d->world;
  a.root = d->root;
  a.max_blocks = d->max_blocks;
  unsigned blocks = 0;
  geometry(d, &blocks, &a.per_block);
  const bool vec = aligned(d->send, 16) && aligned(d->recv, 16) &&
                   d->send_stride % 16 == 0 && d->recv_stride % 16 == 0;
  hipStream_t s = ctx->stream;
  if (!is_reduction(d->kind)) {
    launch_typed<uint8_t, TURBINE_REDUCE_SUM>(s, a, vec, blocks);
  } else if (d->dtype == TURBINE_DTYPE_BF16) {
    if (d->reduce_op == TURBINE_REDUCE_SUM)
      launch_typed<turbine_hip::Bf16, TURBINE_REDUCE_SUM>(s, a, vec, blocks);
    else
      launch_typed<turbine_hip::Bf16, TURBINE_REDUCE_MAX>(s, a, vec, blocks);
  } else {
    if (d->reduce_op == TURBINE_REDUCE_SUM)
      launch_typed<float, TURBINE_REDUCE_SUM>(s, a, vec, blocks);
    else
      launch_typed<float, TURBINE_REDUCE_MAX>(s, a, vec, blocks);
  }
  return check_hip(ctx, hipGetLastError(), "mapped_collective kernel launch");
}


// v2.8: the copy-engine all-reduce (header; the kernels' comment above).
int32_t enqueue_dma(turbine_ctx *ctx, const turbine_mapped_collective_desc *d,
                    const turbine_mapped_dma_desc *m) {
  const char *what = "turbine_mapped_all_reduce_dma";
  if (const std::string why = refusal(d); !why.empty()) {
    return fail(ctx, d == nullptr ? TURBINE_E_ARGUMENT : TURBINE_E_UNSUPPORTED,
                std::string(what) + ": " + why);
  }
  if (d->kind != TURBINE_MAPPED_ALL_REDUCE) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + ": only ALL_REDUCE");
  }
  if (m == nullptr || m->copy == nullptr || m->event == nullptr ||
      m->seq_counter == nullptr || !aligned(m->seq_counter, 8) ||
      m->chunk_bytes <= 0 || m->chunk_bytes % 16 != 0 ||
      (m->flags & ~TURBINE_MAPPED_DMA_PEER_READ) != 0 ||
      (d->world > 1 && m->scratch == nullptr &&
       (m->flags & TURBINE_MAPPED_DMA_PEER_READ) == 0)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": invalid DMA descriptor");
  }
  if (d->bytes == 0) {
    return TURBINE_OK;
  }
  const int64_t chunks = (d->bytes + m->chunk_bytes - 1) / m->chunk_bytes;
  if (chunks > d->max_blocks) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string(what) + ": " + std::to_string(chunks) +
                    " chunks exceed max_blocks " +
                    std::to_string(d->max_blocks));
  }
  const int64_t e = elem_bytes(d);
  if (d->send == nullptr || d->recv == nullptr || d->slots == nullptr ||
      d->flags == nullptr || d->abort_word == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, std::string(what) + ": NULL operand");
  }
  if (!aligned(d->send, e) || !aligned(d->recv, e) || !aligned(d->slots, 16) ||
      !aligned(d->flags, 8) || !aligned(d->abort_word, 4) ||
      (m->scratch != nullptr && !aligned(m->scratch, 16))) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string(what) + ": misaligned operand");
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  if (int32_t rc = ensure_wall_clock(ctx); rc != TURBINE_OK)
    return rc;
  hipStream_t cs = ctx->stream;
  hipStream_t xs = reinterpret_cast<hipStream_t>(m->copy);
  hipEvent_t ev = reinterpret_cast<hipEvent_t>(m->event);
  if (int32_t rc = check_hip(ctx, hipEventRecord(ev, cs), "hipEventRecord");
      rc != TURBINE_OK)
    return rc;
  if (int32_t rc =
          check_hip(ctx, hipStreamWaitEvent(xs, ev, 0), "hipStreamWaitEvent");
      rc != TURBINE_OK)
    return rc;
  const auto *send = static_cast<const uint8_t *>(d->send);
  auto *recv = static_cast<uint8_t *>(d->recv);
  auto *slots = static_cast<uint8_t *>(d->slots);
  auto *scratch = static_cast<uint8_t *>(m->scratch);
  const int64_t parity = static_cast<int64_t>(d->seq & 1);
  auto slot = [&](int q) {
    return slots + (parity * d->world + q) * d->slot_bytes;
  };
  auto chunk_len = [&](int64_t k) {
    return lmin_host(m->chunk_bytes, d->bytes - k * m->chunk_bytes);
  };
  // Copy engine: every chunk out, each followed by its flag.
  for (int64_t k = 0; k < chunks; ++k) {
    const int64_t off = k * m->chunk_bytes;
    if (int32_t rc = check_hip(ctx,
                               hipMemcpyAsync(slot(d->rank) + off, send + off,
                                              static_cast<size_t>(chunk_len(k)),
                                              hipMemcpyDefault, xs),
                               "hipMemcpyAsync (chunk out)");
        rc != TURBINE_OK)
      return rc;
    hipLaunchKernelGGL(dma_publish_kernel, dim3(1), dim3(64), 0, xs,
                       m->seq_counter, d->flags + d->rank * d->max_blocks + k,
                       k);
  }
  DmaWait w{};
  w.counter = m->seq_counter;
  w.flags = d->flags;
  w.abort_word = d->abort_word;
  w.timeout_ticks = timeout_ticks(ctx, d->timeout_ns);
  w.world = d->world;
  w.rank = d->rank;
  w.max_blocks = d->max_blocks;
  const bool direct = (m->flags & TURBINE_MAPPED_DMA_PEER_READ) != 0;
  for (int64_t k = 0; k < chunks; ++k) {
    const int64_t off = k * m->chunk_bytes;
    const int64_t n = chunk_len(k);
    hipLaunchKernelGGL(dma_wait_kernel, dim3(1), dim3(64), 0, cs, w, k);
    for (int q = 0; q < d->world; ++q) {
      if (q == d->rank || direct)
        continue;
      uint8_t *dst = scratch + (q < d->rank ? q : q - 1) * m->chunk_bytes;
      if (int32_t rc =
              check_hip(ctx,
                        hipMemcpyAsync(dst, slot(q) + off,
                                       static_cast<size_t>(n),
                                       hipMemcpyDefault, cs),
                        "hipMemcpyAsync (chunk in)");
          rc != TURBINE_OK)
        return rc;
    }
    DmaReduce r{};
    r.out = recv + off;
    r.own = send + off;
    r.scratch = direct ? slot(0) + off : scratch;
    r.scratch_stride = direct ? d->slot_bytes : m->chunk_bytes;
    r.direct = direct;
    r.n = n;
    r.abort_word = d->abort_word;
    r.world = d->world;
    r.rank = d->rank;
    const int64_t want = (n + kBlockMinBytes - 1) / kBlockMinBytes;
    const int64_t nb = want < 1 ? 1 : (want > 256 ? 256 : want);
    r.per_block = round16((n + nb - 1) / nb);
    const unsigned blocks = static_cast<unsigned>(nb);
    const bool vec = aligned(r.out, 16) && aligned(r.own, 16);
    if (d->dtype == TURBINE_DTYPE_BF16) {
      if (d->reduce_op == TURBINE_REDUCE_SUM)
        launch_dma_reduce<turbine_hip::Bf16, TURBINE_REDUCE_SUM>(cs, r, vec,
                                                                 blocks);
      else
        launch_dma_reduce<turbine_hip::Bf16, TURBINE_REDUCE_MAX>(cs, r, vec,
                                                                 blocks);
    } else {
      if (d->reduce_op == TURBINE_REDUCE_SUM)
        launch_dma_reduce<float, TURBINE_REDUCE_SUM>(cs, r, vec, blocks);
      else
        launch_dma_reduce<float, TURBINE_REDUCE_MAX>(cs, r, vec, blocks);
    }
  }
  hipLaunchKernelGGL(dma_advance_kernel, dim3(1), dim3(1), 0, cs,
                     m->seq_counter);
  return check_hip(ctx, hipGetLastError(), "mapped DMA all-reduce launch");
}

} // namespace

extern "C" {

int32_t turbine_mapped_collective(turbine_ctx *ctx,
                                  const turbine_mapped_collective_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  // A captured step would replay its baked seq: v2.8 steps only (header).
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_mapped_collective");
  return enqueue_step(ctx, d, nullptr);
}

int32_t turbine_mapped_collective_dseq(turbine_ctx *ctx,
                                       const turbine_mapped_collective_desc *d,
                                       uint64_t *seq_counter) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (seq_counter == nullptr || !aligned(seq_counter, 8)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_mapped_collective_dseq: seq_counter NULL or "
                "misaligned");
  }
  return enqueue_step(ctx, d, seq_counter);
}

int32_t turbine_mapped_all_reduce_dma(turbine_ctx *ctx,
                                      const turbine_mapped_collective_desc *d,
                                      const turbine_mapped_dma_desc *m) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_mapped_all_reduce_dma");
  return enqueue_dma(ctx, d, m);
}

} // extern "C"
