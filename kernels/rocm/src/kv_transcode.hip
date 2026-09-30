// KV transcode (ABI v2.11 turbine_kv_transcode) on ROCm: encode a batch of KV
// blocks from their L0 pages into a lower tier's codec, in a device buffer of
// coded slots, and decode them back (Phase 6b S-1). The coded bytes then cross
// the host link through the existing pinned copies; this file never touches
// host memory but the page table of the descriptor.
//
// `turbine_hip_fp8`: the FP8 e4m3 codec of crates/turbine-kv (fp8_e4m3.rs),
// BF16 pages only. An own kernel per the decision "P6b: KV transcode —
// provider evaluation (kernel reuse rule)": CK's elementwise and
// batched_transpose building blocks take one strided tensor per call, have no
// per-(layer, K or V) absmax reduction feeding a scale, gather no scattered
// page addresses and convert through the hardware e4m3 instruction, whose
// rounding and NaN code the codec's contract does not pin down.
//
// Bit-exact with the CPU codec. A slot is, per layer, K and V scale F32
// little-endian (header: layers x [k, v]), then one e4m3 byte per page element
// in page order (layer, K then V, token, head, dim). One workgroup handles one
// (block, layer, K or V): it takes the absmax of its page half, derives
// max(amax / 448, 1 / (448 * 512)) (qgemm_quantize.hpp dynamic_fp8_scale, the
// same rule), divides each element by it in F32 (HIP's correctly rounded
// division) and rounds to e4m3 with fp8_e4m3_round, which transcribes the
// reference; decode multiplies the e4m3 value by the scale in F32 and rounds to
// BF16 with ties to even (bf16.hpp). No hardware conversion is used.
//
// Slots for TurboQuant (Task 8) go in kv_transcode_tq.hip behind their own
// implementation entries; the descriptor and the page table upload below are
// shared with them (kv_transcode.hpp).
#include "kv_transcode.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "bf16.hpp"
#include "qgemm_quantize.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace turbine_hip {

// The per-context page-table upload (turbine_ctx::kv_transcode): a ring of
// kSlots tables, each a pinned host copy of the descriptor's pointer array, the
// device copy the kernels read, and an event that says when the pinned copy has
// been consumed. A call takes the next slot and waits only for the call kSlots
// calls earlier, so the engine thread does not wait for the previous batch.
// Grown outside graph capture only; freed with the context.
struct KvTranscodeScratch {
  static constexpr int kSlots = 4;
  struct Table {
    void **host = nullptr;
    void **device = nullptr;
    size_t entries = 0;
    hipEvent_t consumed = nullptr;
  };
  Table tables[kSlots];
  int next = 0;

  KvTranscodeScratch() = default;
  KvTranscodeScratch(const KvTranscodeScratch &) = delete;
  KvTranscodeScratch &operator=(const KvTranscodeScratch &) = delete;
  ~KvTranscodeScratch() {
    for (Table &t : tables) {
      if (t.consumed != nullptr)
        (void)hipEventDestroy(t.consumed);
      if (t.host != nullptr)
        (void)hipHostFree(t.host);
      if (t.device != nullptr)
        (void)hipFree(t.device);
    }
  }
};

} // namespace turbine_hip

namespace {

constexpr const char *kFp8Impl = "turbine_hip_fp8";
constexpr int kThreads = turbine_hip::kQuantThreads;

// Elements of one layer's K (or V) of a block.
int64_t half_elems(const turbine_kv_transcode_desc *d) {
  return static_cast<int64_t>(d->block_tokens) * d->num_kv_heads * d->head_dim;
}

// Bytes of an FP8 slot: the scale header plus one byte per page element.
int64_t fp8_slot_bytes(const turbine_kv_transcode_desc *d) {
  return static_cast<int64_t>(d->layers) * (8 + 2 * half_elems(d));
}

// Shape and dtype checks shared by the descriptor's probe (no pointers, no
// slot size) and its run.
bool fp8_supported(const turbine_kv_transcode_desc *d) {
  if (d == nullptr)
    return false;
  if (d->format != TURBINE_KVFMT_FP8_E4M3)
    return false;
  if (d->direction != TURBINE_KV_ENCODE && d->direction != TURBINE_KV_DECODE)
    return false;
  // FP8 pages are the codec's copy path (a slot is the page), not a transcode.
  if (d->page_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->num_blocks < 0 || d->layers <= 0 || d->block_tokens <= 0 ||
      d->num_kv_heads <= 0 || d->head_dim <= 0) {
    return false;
  }
  // The scale and element offsets below are 64-bit; bound the page so the
  // slot size cannot overflow.
  return half_elems(d) <= (int64_t{1} << 31) && d->layers <= (1 << 16);
}

std::string describe(const turbine_kv_transcode_desc *d) {
  return "format=" + std::to_string(d->format) +
         " direction=" + std::to_string(d->direction) +
         " page_dtype=" + std::to_string(d->page_dtype) +
         " blocks=" + std::to_string(d->num_blocks) +
         " layers=" + std::to_string(d->layers) +
         " block_tokens=" + std::to_string(d->block_tokens) +
         " kv_heads=" + std::to_string(d->num_kv_heads) +
         " head_dim=" + std::to_string(d->head_dim);
}

// Grid: x = 2 * layers ((layer, K or V) pairs), y = blocks. `pages` is the
// device copy of the descriptor's page table: pages[b * layers + l].
__global__ void __launch_bounds__(kThreads)
    fp8_encode_kernel(const uint16_t *const *pages, uint8_t *coded,
                      int64_t slot_bytes, int32_t layers, int64_t half) {
  __shared__ float red[kThreads];
  const int t = threadIdx.x;
  const int32_t layer = blockIdx.x >> 1;
  const int32_t kind = blockIdx.x & 1;
  const int64_t b = blockIdx.y;
  const uint16_t *src = pages[b * layers + layer] + kind * half;
  uint8_t *slot = coded + b * slot_bytes;

  float amax = 0.0f;
  for (int64_t i = t; i < half; i += kThreads) {
    amax = fmaxf(amax, fabsf(turbine_hip::bf16_to_f32(
                           turbine_hip::Bf16{src[i]})));
  }
  const float scale =
      turbine_hip::dynamic_fp8_scale(turbine_hip::block_max(amax, red));
  const int64_t pair = static_cast<int64_t>(layer) * 2 + kind;
  if (t == 0) {
    // Bytewise: a slot is not guaranteed 4-byte aligned.
    uint32_t bits;
    __builtin_memcpy(&bits, &scale, sizeof(bits));
    for (int k = 0; k < 4; ++k)
      slot[pair * 4 + k] = static_cast<uint8_t>(bits >> (8 * k));
  }
  uint8_t *body = slot + static_cast<int64_t>(layers) * 8 + pair * half;
  for (int64_t i = t; i < half; i += kThreads) {
    const float x =
        turbine_hip::bf16_to_f32(turbine_hip::Bf16{src[i]});
    body[i] = turbine_hip::fp8_e4m3_round(x / scale);
  }
}

__global__ void __launch_bounds__(kThreads)
    fp8_decode_kernel(uint16_t *const *pages, const uint8_t *coded,
                      int64_t slot_bytes, int32_t layers, int64_t half) {
  const int t = threadIdx.x;
  const int32_t layer = blockIdx.x >> 1;
  const int32_t kind = blockIdx.x & 1;
  const int64_t b = blockIdx.y;
  uint16_t *dst = pages[b * layers + layer] + kind * half;
  const uint8_t *slot = coded + b * slot_bytes;
  const int64_t pair = static_cast<int64_t>(layer) * 2 + kind;
  uint32_t bits = 0;
  for (int k = 0; k < 4; ++k)
    bits |= static_cast<uint32_t>(slot[pair * 4 + k]) << (8 * k);
  float scale;
  __builtin_memcpy(&scale, &bits, sizeof(scale));
  const uint8_t *body = slot + static_cast<int64_t>(layers) * 8 + pair * half;
  for (int64_t i = t; i < half; i += kThreads) {
    dst[i] = turbine_hip::f32_to_bf16(turbine_hip::fp8_e4m3_value(body[i]) *
                                      scale)
                 .bits;
  }
}

// Makes a device table hold the descriptor's pointers (num_blocks * layers of
// them) and returns it in `device`; the next kSlots - 1 calls use other tables.
int32_t upload_pages(turbine_ctx *ctx, const turbine_kv_transcode_desc *d,
                     void ***device) {
  const size_t entries =
      static_cast<size_t>(d->num_blocks) * static_cast<size_t>(d->layers);
  if (!ctx->kv_transcode)
    ctx->kv_transcode = std::make_shared<turbine_hip::KvTranscodeScratch>();
  turbine_hip::KvTranscodeScratch &scratch = *ctx->kv_transcode;
  turbine_hip::KvTranscodeScratch::Table &t = scratch.tables[scratch.next];
  if (t.consumed == nullptr) {
    const int32_t rc = check_hip(
        ctx, hipEventCreateWithFlags(&t.consumed, hipEventDisableTiming),
        "hipEventCreate (kv_transcode)");
    if (rc != TURBINE_OK)
      return rc;
  } else {
    // The pinned copy this slot held kSlots calls ago is overwritten below.
    const int32_t rc = check_hip(ctx, hipEventSynchronize(t.consumed),
                                 "hipEventSynchronize (kv_transcode)");
    if (rc != TURBINE_OK)
      return rc;
  }
  if (t.entries < entries) {
    if (ctx->capturing) {
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_kv_transcode: page table of " +
                      std::to_string(t.entries) + " entries cannot grow to " +
                      std::to_string(entries) + " while a graph is captured");
    }
    void **host = nullptr;
    void **dev = nullptr;
    int32_t rc = check_hip(
        ctx,
        hipHostMalloc(reinterpret_cast<void **>(&host),
                      entries * sizeof(void *), hipHostMallocDefault),
        "hipHostMalloc (kv_transcode page table)");
    if (rc != TURBINE_OK)
      return rc;
    rc = check_hip(ctx,
                   hipMalloc(reinterpret_cast<void **>(&dev),
                             entries * sizeof(void *)),
                   "hipMalloc (kv_transcode page table)");
    if (rc != TURBINE_OK) {
      (void)hipHostFree(host);
      return rc;
    }
    if (t.device != nullptr) {
      // Queued work may still read the old table.
      (void)hipStreamSynchronize(ctx->stream);
      (void)hipFree(t.device);
    }
    if (t.host != nullptr)
      (void)hipHostFree(t.host);
    t.host = host;
    t.device = dev;
    t.entries = entries;
  }
  std::memcpy(t.host, d->pages, entries * sizeof(void *));
  int32_t rc = check_hip(ctx,
                         hipMemcpyAsync(t.device, t.host,
                                        entries * sizeof(void *),
                                        hipMemcpyHostToDevice, ctx->stream),
                         "hipMemcpyAsync (kv_transcode page table)");
  if (rc != TURBINE_OK)
    return rc;
  rc = check_hip(ctx, hipEventRecord(t.consumed, ctx->stream),
                 "hipEventRecord (kv_transcode)");
  if (rc != TURBINE_OK)
    return rc;
  *device = t.device;
  scratch.next = (scratch.next + 1) % turbine_hip::KvTranscodeScratch::kSlots;
  return TURBINE_OK;
}

} // namespace

extern "C" {

int32_t turbine_kv_transcode_supported(const turbine_kv_transcode_desc *d) {
  return turbine_hip::default_entry(turbine_hip::kDefaultProfile,
                                    TURBINE_OP_KV_TRANSCODE, d) != nullptr
             ? 1
             : 0;
}

const char *turbine_kv_transcode_impl(const turbine_kv_transcode_desc *d) {
  return turbine_hip::default_name(TURBINE_OP_KV_TRANSCODE, d);
}

int32_t turbine_kv_transcode(turbine_ctx *ctx,
                             const turbine_kv_transcode_desc *d) {
  return turbine_hip::run_default(ctx, TURBINE_OP_KV_TRANSCODE, d);
}

} // extern "C"

namespace turbine_hip {

bool kv_transcode_fp8_supports(const turbine_kv_transcode_desc *d) {
  return fp8_supported(d);
}

int32_t kv_transcode_fp8_run(turbine_ctx *ctx,
                             const turbine_kv_transcode_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: descriptor is NULL");
  }
  if (!fp8_supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                std::string("turbine_kv_transcode (") + kFp8Impl +
                    "): unsupported configuration " + describe(d));
  }
  if (d->num_blocks == 0)
    return TURBINE_OK;
  if (d->pages == nullptr || d->coded == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_kv_transcode: NULL page table or coded buffer");
  }
  if (d->coded_block_bytes != fp8_slot_bytes(d)) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                std::string("turbine_kv_transcode (") + kFp8Impl +
                    "): coded_block_bytes " +
                    std::to_string(d->coded_block_bytes) + " is not the " +
                    std::to_string(fp8_slot_bytes(d)) + " bytes of a slot (" +
                    describe(d) + ")");
  }
  for (int64_t i = 0; i < static_cast<int64_t>(d->num_blocks) * d->layers;
       ++i) {
    if (d->pages[i] == nullptr)
      return fail(ctx, TURBINE_E_ARGUMENT,
                  "turbine_kv_transcode: NULL page " + std::to_string(i));
  }
  // grid.y is at most 65535 blocks.
  if (d->num_blocks > 65535) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_kv_transcode: more than 65535 blocks in one call");
  }
  if (ctx->capturing)
    return refuse_while_capturing(ctx, "turbine_kv_transcode");
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;
  void **table = nullptr;
  if (int32_t rc = upload_pages(ctx, d, &table); rc != TURBINE_OK)
    return rc;
  const dim3 grid(static_cast<uint32_t>(2 * d->layers),
                  static_cast<uint32_t>(d->num_blocks));
  auto *coded = static_cast<uint8_t *>(d->coded);
  if (d->direction == TURBINE_KV_ENCODE) {
    hipLaunchKernelGGL(fp8_encode_kernel, grid, dim3(kThreads), 0, ctx->stream,
                       reinterpret_cast<const uint16_t *const *>(table), coded,
                       d->coded_block_bytes, d->layers, half_elems(d));
  } else {
    hipLaunchKernelGGL(fp8_decode_kernel, grid, dim3(kThreads), 0, ctx->stream,
                       reinterpret_cast<uint16_t *const *>(table), coded,
                       d->coded_block_bytes, d->layers, half_elems(d));
  }
  return check_hip(ctx, hipGetLastError(), "turbine_kv_transcode launch");
}

} // namespace turbine_hip
