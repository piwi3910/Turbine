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

// Per-class page addressing (ABI v2.11, P6b S-5 / S-7): the byte offset of
// block b's page within one layer's region, by the pair's TURBINE_KVFMT_*
// code, or -1 when the id or the format is not one of the pool's. Flat pools
// (page_classes NULL) keep `b * block_bytes`.
int64_t page_offset(const turbine_copy_blocks_desc *d, int32_t b, int fmt) {
  const int64_t base_page = d->block_bytes;
  int64_t per = base_page;
  if (d->page_classes != nullptr) {
    per = 0;
    if (b >= d->base_blocks) {
      for (int32_t i = 0; i < d->num_page_classes; ++i) {
        if (d->page_classes[i].fmt == fmt) {
          per = d->page_classes[i].per_layer_bytes;
          break;
        }
      }
    }
    if (per <= 0 || d->slab_stride <= 0 || d->slab_base_blocks <= 0)
      return -1;
  }
  int64_t off;
  if (d->page_classes == nullptr || b < d->base_blocks) {
    if (b < 0)
      return -1;
    off = static_cast<int64_t>(b) * base_page;
  } else {
    const int64_t rel = b - d->base_blocks;
    off = rel / d->slab_stride * static_cast<int64_t>(d->slab_base_blocks) *
              base_page +
          rel % d->slab_stride * per;
  }
  // The page must sit inside one layer's region.
  if (off < 0 || off + per > d->layer_stride_bytes)
    return -1;
  return off;
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
  // Every block must lie inside its layer (flat pools); a classed pool
  // validates the ids against its classes below.
  const int64_t blocks_per_layer = d->layer_stride_bytes / d->block_bytes;
  const bool classed = d->page_classes != nullptr;
  for (int32_t i = 0; i < d->count; ++i) {
    if (classed) {
      if (d->pair_formats == nullptr) {
        return fail(ctx, TURBINE_E_ARGUMENT,
                    "turbine_copy_blocks: page classes without pair_formats");
      }
      const int fmt = d->pair_formats[i];
      for (const int32_t b : {d->src_blocks[i], d->dst_blocks[i]}) {
        if (page_offset(d, b, fmt) < 0) {
          return fail(ctx, TURBINE_E_ARGUMENT,
                      "turbine_copy_blocks: block id " + std::to_string(b) +
                          " is outside the pool's id space");
        }
      }
    } else {
      for (const int32_t b : {d->src_blocks[i], d->dst_blocks[i]}) {
        if (b < 0 || b >= blocks_per_layer) {
          return fail(ctx, TURBINE_E_ARGUMENT,
                      "turbine_copy_blocks: block id " + std::to_string(b) +
                          " is outside the " +
                          std::to_string(blocks_per_layer) +
                          " blocks of a layer");
        }
      }
    }
  }
  if (int32_t rc = enter(ctx); rc != TURBINE_OK)
    return rc;

  auto *pool = static_cast<char *>(d->pool);
  for (int32_t layer = 0; layer < d->num_layers; ++layer) {
    char *base = pool + static_cast<int64_t>(layer) * d->layer_stride_bytes;
    for (int32_t i = 0; i < d->count; ++i) {
      const int fmt = d->pair_formats != nullptr ? d->pair_formats[i] : 0;
      const int64_t off = page_offset(d, d->src_blocks[i], fmt);
      const int64_t dst_off = page_offset(d, d->dst_blocks[i], fmt);
      if (off < 0 || dst_off < 0)
        return fail(ctx, TURBINE_E_ARGUMENT,
                    "turbine_copy_blocks: block id outside the pool");
      // The class's page bytes (the base page on a flat pool).
      int64_t bytes = d->block_bytes;
      if (d->page_classes != nullptr && d->src_blocks[i] >= d->base_blocks) {
        for (int32_t c = 0; c < d->num_page_classes; ++c) {
          if (d->page_classes[c].fmt == fmt)
            bytes = d->page_classes[c].per_layer_bytes;
        }
      }
      const char *src = base + off;
      char *dst = base + dst_off;
      const auto move_bytes = static_cast<size_t>(bytes);
      if (src == dst)
        continue;
      const int32_t rc =
          check_hip(ctx,
                    hipMemcpyAsync(dst, src, move_bytes,
                                   hipMemcpyDeviceToDevice, ctx->stream),
                    "hipMemcpyAsync device to device (copy_blocks)");
      if (rc != TURBINE_OK)
        return rc;
    }
  }
  return TURBINE_OK;
}

} // extern "C"
