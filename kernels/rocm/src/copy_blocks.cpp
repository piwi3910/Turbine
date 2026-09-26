// copy_blocks: forks KV blocks across every layer of the pool.
//
// Layer l's block b is the block_bytes bytes at
// pool + l * layer_stride_bytes + b * block_bytes. For each layer (outer) and
// each pair i (inner, in order), block src_blocks[i] is copied to
// dst_blocks[i] with one device-to-device hipMemcpyAsync on the compute stream,
// so the copies run in the same order as the cpu-reference loop. The host
// arrays are read during the call only.
#include <string>

#include "turbine_hip.hpp"

using turbine_hip::check_hip;
using turbine_hip::enter;
using turbine_hip::fail;

namespace {

constexpr const char *kImpl = "hip_memcpy_d2d";

bool supported(const turbine_copy_blocks_desc *d) {
  return d != nullptr && d->block_bytes > 0 &&
         d->layer_stride_bytes >= d->block_bytes && d->num_layers >= 0 &&
         d->count >= 0;
}

std::string describe(const turbine_copy_blocks_desc *d) {
  return "layers=" + std::to_string(d->num_layers) +
         " layer_stride_bytes=" + std::to_string(d->layer_stride_bytes) +
         " block_bytes=" + std::to_string(d->block_bytes) +
         " count=" + std::to_string(d->count);
}

} // namespace

extern "C" {

int32_t turbine_copy_blocks_supported(const turbine_copy_blocks_desc *d) {
  return supported(d) ? 1 : 0;
}

const char *turbine_copy_blocks_impl(const turbine_copy_blocks_desc *d) {
  (void)d;
  return kImpl;
}

int32_t turbine_copy_blocks(turbine_ctx *ctx,
                            const turbine_copy_blocks_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (d == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT,
                "turbine_copy_blocks: descriptor is NULL");
  }
  if (!supported(d)) {
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "turbine_copy_blocks: unsupported configuration " +
                    describe(d));
  }
  if (d->count == 0 || d->num_layers == 0)
    return TURBINE_OK;
  if (d->pool == nullptr || d->src_blocks == nullptr ||
      d->dst_blocks == nullptr) {
    return fail(ctx, TURBINE_E_ARGUMENT, "turbine_copy_blocks: NULL operand");
  }
  // Every block must lie inside its layer.
  const int64_t blocks_per_layer = d->layer_stride_bytes / d->block_bytes;
  for (int32_t i = 0; i < d->count; ++i) {
    for (const int32_t b : {d->src_blocks[i], d->dst_blocks[i]}) {
      if (b < 0 || b >= blocks_per_layer) {
        return fail(ctx, TURBINE_E_ARGUMENT,
                    "turbine_copy_blocks: block id " + std::to_string(b) +
                        " is outside the " + std::to_string(blocks_per_layer) +
                        " blocks of a layer");
      }
    }
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;

  auto *pool = static_cast<char *>(d->pool);
  const auto bytes = static_cast<size_t>(d->block_bytes);
  for (int32_t layer = 0; layer < d->num_layers; ++layer) {
    char *base = pool + static_cast<int64_t>(layer) * d->layer_stride_bytes;
    for (int32_t i = 0; i < d->count; ++i) {
      const char *src =
          base + static_cast<int64_t>(d->src_blocks[i]) * d->block_bytes;
      char *dst =
          base + static_cast<int64_t>(d->dst_blocks[i]) * d->block_bytes;
      if (src == dst)
        continue;
      const int32_t rc = check_hip(
          ctx,
          hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, ctx->stream),
          "hipMemcpyAsync device to device (copy_blocks)");
      if (rc != TURBINE_OK)
        return rc;
    }
  }
  return TURBINE_OK;
}

} // extern "C"
