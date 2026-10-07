// Single-row BF16 GEMV on RDNA (gfx11 / gfx12, wave32): vllm_wvsplitk, the
// m = 1 implementation of turbine_gemm (c[1, n] = a[1, k] . b[n, k]^T, BF16
// a and b, BF16 or F32 c), on vLLM's wvSplitK kernel.
//
// Third-party code: the device code between the "vLLM" markers below is
// vLLM's csrc/rocm/skinny_gemms.cu at commit
// b267fe488102eb26555d553a005127b4a523ff4b (github.com/vllm-project/vllm),
// SPDX-License-Identifier: Apache-2.0, Copyright contributors to the vLLM
// project; the license text is kernels/rocm/third_party/LICENSES/vllm-LICENSE.
// Taken: the architecture macros, UNREACHABLE_CODE, loadnt, DOT2C, min__, the
// wvSplitK_hf_sml_ kernel (the activations staged once in LDS, one persistent
// block per CU looping over weight rows) and the host helper mindiv. Changed
// (Turbine, 2026-10-06): a trailing template parameter out_t types the output,
// and the two output stores call turbine_store<out_t> (round to nearest even
// for BF16, unrounded for F32: the LM head's logits); a trailing template
// parameter GLU and argument UpOff make the LDS staging loop store
// turbine_silu_mul_chunk(A[k..], A[k + UpOff..]) instead of A[k..] (the
// SwiGLU input of kernel ABI v2.14, 5p GEMV stage B); nothing else. The
// launcher below is Turbine's, written after vLLM's wvSplitK() host code for
// N = 1 on gfx1x (not gfx1151): YTILE 1 / UNRL 4 when the rows fit one round
// of 4 per CU, else YTILE 2 / UNRL 2; 16 waves of 32 lanes per block, one
// block per CU.
//
// Why vendored (decision "5p fused batch-1 GEMV: provider evaluation and
// proposal", user decision B, staged, stage A): the fastest correct m = 1 GEMV
// of the evaluated providers on every Llama-3.2-3B decode shape on the R9700
// (qkv 51.8 us vs hipBLASLt's served 58.9, o_proj 32.2 vs 36.5, gate_up 160.0
// vs 159.8, down 83.7 vs 85.8, LM head 1242.5 vs 1244.6), plain HIP, no torch,
// and the persistent structure the later fused stages build on.
//
// Which calls run the kernel: m = 1 calls of a decode-only step
// (TURBINE_OPTION_GEMM_PREFILL off), alpha 1, beta 0, 16-byte aligned a and b,
// on a wave32 device. Every other single-row call (prefill steps, whose rows
// must equal the invariant prefill solution's bit for bit; a scaled or
// accumulating call) runs hipblaslt (gemm.cpp) unchanged, so the entry serves
// every call its supports() accepts: m = 1 and the kernel's shapes. The card
// profile puts it first in the one-row GEMM tier, for the configs of dense
// BF16 decoders on one device (GemmConfig::gemv on the Rust side).
//
// Numerics: products of two BF16 values are exact in F32; each lane sums its
// k slice in F32 in k order, the 32 lanes are combined by a DPP / shuffle
// tree, and the sum is rounded once to c's dtype. The result differs from
// hipBLASLt's and the CPU reference's by the F32 summation order only, and a
// row depends only on its weight row and a (no split across blocks), so it is
// deterministic run to run.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>

#include "bf16.hpp"
#include "turbine_hip.hpp"

// Turbine: the SwiGLU input of one 8-element chunk, element by element exactly
// as elementwise.hip's silu_mul kernels compute it (SiLU in F32, rounded to
// BF16, times up in F32, rounded to BF16), so the staged operand equals the
// silu_mul op's output bit for bit.
template <typename chunk_t>
__device__ __forceinline__ chunk_t turbine_silu_mul_chunk(const chunk_t &g,
                                                          const chunk_t &u) {
  using turbine_hip::Bf16;
  chunk_t o;
  constexpr int kElems = sizeof(o.h) / sizeof(o.h[0]);
#pragma unroll
  for (int i = 0; i < kElems; ++i) {
    const float gf =
        turbine_hip::bf16_to_f32(__builtin_bit_cast(Bf16, g.h[i]));
    const float uf =
        turbine_hip::bf16_to_f32(__builtin_bit_cast(Bf16, u.h[i]));
    const float silu = gf / (1.0f + expf(-gf));
    o.h[i] = __builtin_bit_cast(
        __hip_bfloat16,
        turbine_hip::f32_to_bf16(turbine_hip::round_bf16(silu) * uf));
  }
  return o;
}

// Turbine: the output conversion of the vendored kernel's stores.
template <typename out_t>
__device__ __forceinline__ out_t turbine_store(float v);
template <>
__device__ __forceinline__ __hip_bfloat16
turbine_store<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ float turbine_store<float>(float v) {
  return v;
}

// ---- vLLM (csrc/rocm/skinny_gemms.cu @ b267fe4, Apache-2.0) ----
// clang-format off

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

// Combined RDNA macro (gfx11 + gfx12) - both use 32-wide wavefronts
#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__MI3XX__
#endif

#if defined(__gfx950__)
  #define LDS_SIZE 160 * 1024
#else
  #define LDS_SIZE 64 * 1024
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

#if defined(__HIP__GFX9__) && !defined(__HIP__GFX1X__)
  #define DOT2C(V0, V2, V3)                                          \
    if constexpr (std::is_same_v<scalar_t, half>) {                  \
      asm("v_dot2c_f32_f16 %0, %2, %3"                               \
          : "=v"(V0)                                                 \
          : "0"(V0), "v"(V2), "v"(V3));                              \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) { \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *  \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));   \
      V0 += (s.x + s.y);                                             \
    }
#elif defined(__HIP__GFX1X__)
  // gfx1x: v_dot2_f32_f16 (VOP3-P, dot10-insts, available on gfx11+gfx12)
  #define DOT2C(V0, V2, V3)                                               \
    if constexpr (std::is_same_v<scalar_t, half>) {                       \
      asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(V0) : "v"(V2), "v"(V3)); \
    } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {      \
      float2 s = __bfloat1622float2(*((__hip_bfloat162*)(&(V2)))) *       \
                 __bfloat1622float2(*((__hip_bfloat162*)(&(V3))));        \
      V0 += (s.x + s.y);                                                  \
    }
#endif

// To avoid LLVM silently upcasting to double
__device__ inline unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

#if defined(__HIP__GFX9__) || defined(__HIP__GFX1X__)
// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, typename out_t, bool GLU>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap, const int M,
                     const int Bx, const int By, const scalar_t* B,
                     const scalar_t* __restrict__ A,
                     const scalar_t* __restrict__ BIAS, out_t* C,
                     const int _WvPrGrp, const int CuCount, const int UpOff) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  #if defined(__HIP__MI3XX__)
  constexpr bool use_mfma = (std::is_same_v<scalar_t, __hip_bfloat16>);
  #else
  constexpr bool use_mfma = false;
  #endif
  using scalar8 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
  using half4 =
      __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(__bf16)))) __bf16;
  union bigType {
    scalar_t h[A_CHUNK];
    float f[A_CHUNK / 2];
    float2 f2[A_CHUNK / 4];
    double d[A_CHUNK / 4];
    half4 h4[A_CHUNK / 4];
    scalar8 h8;
  };

  //----------------------------------------------------
  // Reserving 64/160 KB of LDS to have 1 WG / CU
  // Goal is to bring the activation matrix A to the LDS
  // and use it across the lifetime of the work group
  // TODO: When activation matrix is larger than 64 KB
  //	     then this is not going to work!
  //----------------------------------------------------
  __shared__ scalar_t s[max_lds_len];

  //----------------------------------------------------
  // Fetch the activation matrix to LDS
  // Loop iteration:
  // - Each thread (lane) is fetching 8 elements (A_Chunk)
  // - Each wave will fetch 64*8=> 512 elements
  // - Each WG will fetch 512 * 16 => 8K elements
  // - Then the WG will move to another 8 K elements
  // TODO: Logic below will only work when K is multiple of 8
  //----------------------------------------------------
  for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
       k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
  #if defined(__gfx950__)
    __builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0);
  #else
    if constexpr (GLU)  // Turbine (stage B): the SwiGLU input
      *((bigType*)(&s[k])) = turbine_silu_mul_chunk(
          *((const bigType*)(&A[k])), *((const bigType*)(&A[k + UpOff])));
    else
      *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
  #endif
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  //----------------------------------------------------
  // Each wave works on a single column of weight matrix.
  // There are 16 waves per WG, and hence, each WG is
  // working on 16 columns of weight matrix. Moreover,
  // we tile in column direction by YTILE, so when YTILE=1
  // the above math is right, however, when YTILE=2 then
  // each wave  will be working on 2 columns and WG will
  // be working on 32 columns.
  //
  // Top level loop that makes WGs persistent!
  // - WGs iterates across columns of weight matrix
  // - Each wave within WG works on a given column(s)
  // - After completing first set of columns, WGs start
  //   working on the next set of available columns
  //----------------------------------------------------
  while (m < M) {
    //----------------------------------------------------
    // 'sum' accumulates the matrix A x B computation
    // split across 64 lanes.
    //
    // YTILE represents how many column of weight matrix
    // are being worked on by each wave.
    //----------------------------------------------------
    float sum[N][YTILE] = {};
    scalar8 sum4[N][YTILE] = {};

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
      bigType bigA[N][UNRL] = {};
      bigType bigB[YTILE][UNRL];
      // Fetch the weight matrix from memory!
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        const scalar_t* B_ = &B[min__(k_, K - A_CHUNK)];
        for (int y = 0; y < YTILE; y++)
          bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[min__(y + m, M - 1) * Kbp])));
      }

      // Fetch activation matrix from either just LDS or from both LDS / memory
  #pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= K) break;
        for (int n = 0; n < N; n++) {
          bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
        }
      }

      // Do the matrix multiplication in interleaved manner
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        for (uint32_t n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (!use_mfma)
              for (uint32_t b = 0; b < A_CHUNK / 2; b++) {
                DOT2C(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b])
              }
            else
              for (uint32_t b = 0; b < A_CHUNK / 4; b++)
                sum4[n][y] = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(
                    bigA[n][k2].h4[b], bigB[y][k2].h4[b], sum4[n][y], 0, 0, 0);
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    //----------------------------------------------------
    // Final reduction step using shuffle
    //----------------------------------------------------
    if constexpr (!use_mfma) {
      for (int n = 0; n < N; n++) {
        for (int y = 0; y < YTILE; y++) {
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf,
                                                1);  // row_shr8
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf,
                                                1);  // row_shr4
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf,
                                                1);  // row_shr2
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf,
                                                1);  // row_shr1
  #if defined(__HIP__GFX9__)
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf,
                                                1);  // ROW_BCAST15
          sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf,
                                                1);  // ROW_BCAST31
  #else
          sum[n][y] += __shfl_xor(sum[n][y], 16);
  #endif
        }
      }

      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            if constexpr (std::is_same_v<scalar_t, half>) {
              sum[n][y] += __half2float(biases[n][y]);
            } else if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
              sum[n][y] += __bfloat162float(biases[n][y]);
            }
            C[m + y + n * M] = turbine_store<out_t>(sum[n][y]);
          }
        }
      }
    } else {
  #ifdef __HIP__GFX9__
    #pragma unroll
      for (int n = 0; n < N; n++) {
    #pragma unroll
        for (int y = 0; y < YTILE; y++) {
          /*float accm1 = 0;
           for (int i=0; i<64; i++)
              accm1 += __shfl(sum4[n][y][i%4], i);
          sum4[n][y][0] = accm1;*/
          float accm = sum4[n][y][0];
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][1], 0x101, 0xf, 0xf,
                                           1);  // row_shl1
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][2], 0x102, 0xf, 0xf,
                                           1);  // row_shl2
          accm += __builtin_amdgcn_mov_dpp(sum4[n][y][3], 0x103, 0xf, 0xf,
                                           1);  // row_shl3
          accm += __builtin_amdgcn_mov_dpp(accm, 0x104, 0xf, 0xf,
                                           1);  // row_shl4
          accm += __builtin_amdgcn_mov_dpp(accm, 0x108, 0xf, 0xf,
                                           1);  // row_shl8
          accm = __builtin_amdgcn_mov_dpp(accm, 0x11f, 0xf, 0xf,
                                          1);  // row_shr15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x142, 0xf, 0xf,
                                           1);  // ROW_BCAST15
          accm += __builtin_amdgcn_mov_dpp(accm, 0x143, 0xf, 0xf,
                                           1);  // ROW_BCAST31

          sum4[n][y][0] = accm;
        }
      }
      if (threadIdx.x == (THRDS - 1)) {
        scalar_t biases[N][YTILE] = {};
        if (BIAS)
          for (int n = 0; n < N; n++) {
            for (int y = 0; y < YTILE; y++) {
              biases[n][y] = BIAS[(m + y) % Bx + (n % By) * Bx];
            }
          }
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < YTILE; y++) {
            sum4[n][y][0] += __bfloat162float(biases[n][y]);
            C[m + y + n * M] = turbine_store<out_t>(sum4[n][y][0]);
          }
        }
      }
  #endif  // __HIP__GFX9__ (MFMA path)
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, typename out_t, bool GLU>
__global__ void wvSplitK_hf_sml_(const int K, const int Kbp, const int Kap,
                                 const int M, const int Bx, const int By,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ BIAS, out_t* C,
                                 const int _WvPrGrp, const int CuCount,
                                 const int UpOff) {
  UNREACHABLE_CODE
}
#endif

int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int rnds[13];
  for (int i = 0; i < 13; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = 12; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}
// clang-format on
// ---- end vLLM ----

namespace turbine_hip {
namespace {

// The launch shape vLLM's wvSplitK() picks on gfx1x: wave32 lanes, 16 waves
// per block, 8 BF16 elements per lane load.
constexpr int kThreads = 32;
constexpr int kWaves = 16;
constexpr int kChunk = 8;
// Activation elements the block stages in LDS (the kernel's max_lds_len).
constexpr int64_t kLdsElems = LDS_SIZE / 2;

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// The shapes the kernel computes: the whole activation row in LDS, k in whole
// 8-element loads, weight rows of whole 16-byte loads, an even row count (the
// two-row tile), and 32-bit element offsets.
bool kernel_shape(const turbine_gemm_desc *d) {
  if (d->a_dtype != TURBINE_DTYPE_BF16 || d->b_dtype != TURBINE_DTYPE_BF16)
    return false;
  if (d->c_dtype != TURBINE_DTYPE_BF16 && d->c_dtype != TURBINE_DTYPE_F32)
    return false;
  if (d->trans_b != 1 || d->k <= 0 || d->k % kChunk != 0 || d->k > kLdsElems)
    return false;
  if (d->n <= 0 || d->n % 2 != 0 || d->ldb % kChunk != 0)
    return false;
  return d->n * d->ldb < (int64_t{1} << 32);
}

template <typename out_t, bool GLU, int YTILE, int UNRL>
void launch(turbine_ctx *ctx, const turbine_gemm_desc *d, int cu) {
  const int n = static_cast<int>(d->n);
  const int k = static_cast<int>(d->k);
  const int wv = mindiv(n, cu * YTILE, kWaves);
  // The kernel's arguments: K, the weight row stride, the activation row
  // stride (one row: k), M = weight rows, the bias dims, weights,
  // activations, bias, out, waves per block in use, CUs, the up columns'
  // offset from the gate columns (GLU).
  wvSplitK_hf_sml_<__hip_bfloat16, kThreads, YTILE, kWaves, kChunk, UNRL, 1,
                   out_t, GLU>
      <<<dim3(cu), dim3(kThreads, kWaves), 0, ctx->stream>>>(
          k, static_cast<int>(d->ldb), k, n, 1, 1,
          static_cast<const __hip_bfloat16 *>(d->b),
          static_cast<const __hip_bfloat16 *>(d->a), nullptr,
          static_cast<out_t *>(d->c), wv, cu,
          GLU ? static_cast<int>(d->a_up_offset) : 0);
}

// vLLM's gfx1x tile choice for one activation row.
template <typename out_t, bool GLU>
void launch_tiled(turbine_ctx *ctx, const turbine_gemm_desc *d, int cu) {
  const int64_t round_rows = int64_t{cu} * 4;
  if ((d->n + round_rows - 1) / round_rows <= 1)
    launch<out_t, GLU, 1, 4>(ctx, d, cu);
  else
    launch<out_t, GLU, 2, 2>(ctx, d, cu);
}

} // namespace

bool gemm_wvsplitk_supports(const turbine_gemm_desc *d) {
  if (d == nullptr || d->a_silu_mul == 0)
    return turbine_gemm_supported(d) == 1 && d->m == 1 && kernel_shape(d);
  // The SwiGLU input (ABI v2.14): the plain call's shapes, alpha 1 and beta 0
  // (the forms the kernel computes; hipblaslt serves its other fallbacks after
  // an in-place silu_mul), and up columns whole 16-byte loads away from the
  // gate columns, not overlapping them, within 32-bit offsets.
  turbine_gemm_desc plain = *d;
  plain.a_silu_mul = 0;
  plain.a_up_offset = 0;
  return gemm_wvsplitk_supports(&plain) && d->alpha == 1.0f &&
         d->beta == 0.0f && d->a_up_offset % kChunk == 0 &&
         (d->a_up_offset >= d->k || d->a_up_offset <= -d->k) &&
         d->a_up_offset < (int64_t{1} << 31) - d->k &&
         d->a_up_offset > -(int64_t{1} << 31);
}

namespace {

// The SwiGLU input's fallback (a prefill step, an unaligned operand or a
// non-wave32 device): silu_mul(gate, up) written over the gate columns of a,
// then hipblaslt on them, bit for bit silu_mul into a buffer then
// turbine_gemm.
int32_t silu_mul_in_place_then_gemm(turbine_ctx *ctx,
                                    const turbine_gemm_desc *d) {
  turbine_silu_mul_desc s{};
  s.gate = d->a;
  s.up = static_cast<const __hip_bfloat16 *>(d->a) + d->a_up_offset;
  s.out = const_cast<void *>(d->a);
  s.rows = d->m;
  s.cols = d->k;
  s.gate_stride_row = d->lda;
  s.up_stride_row = d->lda;
  s.out_stride_row = d->lda;
  s.dtype = d->a_dtype;
  const int32_t rc = turbine_silu_mul(ctx, &s);
  if (rc != TURBINE_OK)
    return rc;
  turbine_gemm_desc plain = *d;
  plain.a_silu_mul = 0;
  plain.a_up_offset = 0;
  return turbine_gemm(ctx, &plain);
}

} // namespace

int32_t gemm_wvsplitk_run(turbine_ctx *ctx, const turbine_gemm_desc *d) {
  if (ctx == nullptr)
    return TURBINE_E_ARGUMENT;
  if (!gemm_wvsplitk_supports(d))
    return fail(ctx, TURBINE_E_UNSUPPORTED,
                "vllm_wvsplitk: unsupported gemm configuration");
  const bool glu = d->a_silu_mul != 0;
  const bool up_aligned =
      !glu || aligned16(static_cast<const __hip_bfloat16 *>(d->a) +
                        d->a_up_offset);
  if (ctx->gemm_prefill || d->alpha != 1.0f || d->beta != 0.0f ||
      !aligned16(d->a) || !up_aligned || !aligned16(d->b) ||
      ctx->wave_size != kThreads)
    return glu ? silu_mul_in_place_then_gemm(ctx, d) : turbine_gemm(ctx, d);
  if (ctx->cu_count == 0) {
    int cu = 0;
    const int32_t rc = check_hip(
        ctx,
        hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount,
                              ctx->device),
        "hipDeviceGetAttribute(multiprocessor count)");
    if (rc != TURBINE_OK)
      return rc;
    ctx->cu_count = cu;
  }
  const bool f32 = d->c_dtype == TURBINE_DTYPE_F32;
  if (f32 && glu)
    launch_tiled<float, true>(ctx, d, ctx->cu_count);
  else if (f32)
    launch_tiled<float, false>(ctx, d, ctx->cu_count);
  else if (glu)
    launch_tiled<__hip_bfloat16, true>(ctx, d, ctx->cu_count);
  else
    launch_tiled<__hip_bfloat16, false>(ctx, d, ctx->cu_count);
  return check_hip(ctx, hipGetLastError(), "vllm_wvsplitk launch");
}

} // namespace turbine_hip
