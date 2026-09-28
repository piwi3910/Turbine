// INT4 group-quantized GEMM (ABI v2.9 turbine_qgemm, schemes INT4_GROUP_ZP and
// INT4_GROUP_SYM, BF16 activations, act_quant NONE) on RDNA4 (gfx12 WMMA):
// qgemm_int4.hip. Weight layout as the header specifies: b = n x k/2 bytes
// (low nibble = even column), b_scales = F32 [n x k/group], b_zeros = bytes
// [n x k/group] (ZP only; SYM uses 8); value (q - z) * s.
//
// Implementations (decision "P6: INT4 group GEMM — provider evaluation (kernel
// reuse rule)"):
//   turbine_hip_int4_wmma     the fused kernel: codes streamed from memory
//                             into WMMA registers, any m (16-row tiles).
//   turbine_hip_int4_dequant  dequantize column chunks to BF16 then the
//                             hipBLASLt BF16 GEMM (large m).
#pragma once

#include "turbine_hip.hpp"

namespace turbine_hip {

enum class Int4Path { Wmma, Dequant };

// The implementation name of path (impl_table.cpp's).
const char *qgemm_int4_name(Int4Path path);
bool qgemm_int4_supports(const turbine_qgemm_desc *d, Int4Path path);
int32_t qgemm_int4_run(turbine_ctx *ctx, const turbine_qgemm_desc *d,
                       Int4Path path);

// ---- launch level (the implementations above and tools/qgemm_int4_eval) ----

// One INT4 GEMM on raw device pointers: c[m, n] = alpha * a[m, k] .
// dequant(b)[n, k]^T. zeros NULL means symmetric (8). c_f32 selects an F32 c
// (else BF16).
struct Int4Gemm {
  const void *a;
  const uint8_t *b;
  const float *scales;
  const uint8_t *zeros;
  void *c;
  int64_t m, n, k, lda, ldc;
  int32_t group;
  bool c_f32;
  float alpha;
};

// The shape rules of the fused kernel: group 64, 128 or 256, k a multiple of
// group, lda a multiple of 8.
bool int4_wmma_shape(const Int4Gemm &g);
// The shape rules of the dequant path: group a multiple of 32, k a multiple of
// group.
bool int4_dequant_shape(const Int4Gemm &g);
// The fused kernel on stream. Operands 16-byte aligned.
hipError_t launch_int4_wmma(const Int4Gemm &g, hipStream_t stream);
// Dequantizes weight rows [row0, row0 + rows) to BF16 out ([rows, k], dense):
// out = bf16((q - z) * s), rounded to nearest even.
hipError_t launch_int4_dequant(const Int4Gemm &g, int64_t row0, int64_t rows,
                               void *out, hipStream_t stream);

} // namespace turbine_hip
