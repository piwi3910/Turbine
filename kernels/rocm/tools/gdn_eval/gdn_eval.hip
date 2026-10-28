// turbine gdn_eval: Gated DeltaNet provider evaluation (Phase 7 Task 29).
//
// Timing inputs are seeded random data, never zeros (all-zero inputs run
// >20 % faster on AMD and would skew the timings; issue #10).
//
// Qwen3.5-9B / Qwen3.6-35B GDN shapes: 16 key heads x 128, 32 value heads x 128,
// conv k=4 over q|k|v channels (2048 + 2048 + 4096 = 8192), F32 recurrent state.
//
// Candidates:
//   llama.cpp  gated_delta_net_cuda (ggml-cuda/gated_delta_net.cu, compiled for HIP
//              through ggml_shim.h) -- the token-sequential recurrence, F32 in/out.
//              ssm_conv_f32 / ssm_conv_long_token_f32 (ggml-cuda/ssm-conv.cu), F32.
//   aiter      chunk_gated_delta_rule_fwd_h_hip (csrc/kernels, gfx12 wave64 WMMA, BF16
//              k/w/u, F32 state) for the chunk-state stage of the FLA chunked prefill;
//              the other FLA stages (cumsum, kkt, solve_tril, w/u, chunk_o) are Triton
//              in AITER, so they are Turbine prototype HIP kernels here (own_prep,
//              own_o: scalar FP32 in LDS, unoptimized).
//              causal_conv1d_update (decode) and causal_conv1d_fwd_split_qkv (prefill), BF16.
//
// Reference: the transformers torch_recurrent_gated_delta_rule math in F64 on the
// host (q, k pre-L2-normalized, q scaled by 1/sqrt(128)). Inputs are generated
// BF16-representable so both F32 and BF16 candidates see identical values.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "ggml_shim.h"
#include "llama_kernels.inc"

void aiter_eval_chunk_h(int bv, const void* k, const void* w, const void* u, const float* g, const void* h0,
                        void* h, void* v_new, void* ht, const int32_t* dummy, int T, int H, int Hg, hipStream_t s);
namespace aiter {
void causal_conv1d_update(aiter_tensor_t& x, aiter_tensor_t& conv_state, aiter_tensor_t& weight,
                          aiter_tensor_t& bias, aiter_tensor_t& out, bool use_silu,
                          aiter_tensor_t& cache_seqlens, aiter_tensor_t& conv_state_indices, int pad_slot_id);
void causal_conv1d_fwd_split_qkv_hip(aiter_tensor_t x, aiter_tensor_t weight, aiter_tensor_t bias,
                                     aiter_tensor_t conv_states, aiter_tensor_t cache_indices,
                                     aiter_tensor_t has_initial_state, aiter_tensor_t query_start_loc,
                                     aiter_tensor_t batch_ptr, aiter_tensor_t token_chunk_offset_ptr,
                                     aiter_tensor_t q, aiter_tensor_t k, aiter_tensor_t v, int64_t k_dim,
                                     int64_t v_dim, int64_t n_programs, int64_t block_m, bool has_bias, bool silu,
                                     int64_t pad_slot_id);
}  // namespace aiter

#define CK(x) HIP_CALL(x)

static constexpr int HK = 16, HV = 32, DK = 128, DV = 128, BT = 64;
static constexpr int CH = 2 * HK * DK + HV * DV;  // 8192 conv channels
static constexpr int KW = 4;

// ---------------------------------------------------------------- helpers
static float bf16r(float x) {  // round to nearest even BF16
  uint32_t u;
  memcpy(&u, &x, 4);
  u += 0x7fff + ((u >> 16) & 1);
  u &= 0xffff0000u;
  float r;
  memcpy(&r, &u, 4);
  return r;
}
static uint16_t bf16bits(float x) {
  uint32_t u;
  float r = bf16r(x);
  memcpy(&u, &r, 4);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float bits2f_dev(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bits_dev(float x) {
  uint32_t u = __float_as_uint(x);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bits2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float r;
  memcpy(&r, &u, 4);
  return r;
}
template <class T> static T* dalloc(size_t n) {
  T* p;
  CK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
template <class T> static T* dup(const std::vector<T>& h) {
  T* p = dalloc<T>(h.size());
  CK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
static std::vector<uint16_t> tobf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); i++) o[i] = bf16bits(v[i]);
  return o;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}
static aiter_tensor_t tens(void* p, AiterDtype dt, std::vector<int64_t> shape, std::vector<int64_t> strides = {}) {
  aiter_tensor_t t{};
  t.ptr = p;
  t.dtype_ = dt;
  t.device_id = 0;
  t.ndim = (int)shape.size();
  size_t n = 1;
  for (int i = 0; i < t.ndim; i++) {
    t.shape[i] = shape[i];
    n *= shape[i];
  }
  t.numel_ = n;
  if (strides.empty()) {
    int64_t s = 1;
    for (int i = t.ndim - 1; i >= 0; i--) {
      t.strides[i] = s;
      s *= shape[i];
    }
  } else {
    for (int i = 0; i < t.ndim; i++) t.strides[i] = strides[i];
  }
  return t;
}
static aiter_tensor_t empty_t(AiterDtype dt) { return tens(nullptr, dt, {0}); }

struct Err {
  double max_abs = 0, max_ref = 0;
  void add(double ref, double got) {
    max_abs = std::max(max_abs, std::fabs(ref - got));
    max_ref = std::max(max_ref, std::fabs(ref));
  }
  double rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
};

static hipStream_t g_stream;
template <class F> static double time_us(F f, int warm = 5, int iters = 50) {
  for (int i = 0; i < warm; i++) f();
  CK(hipStreamSynchronize(g_stream));
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  std::vector<double> t;
  for (int i = 0; i < iters; i++) {
    CK(hipEventRecord(a, g_stream));
    f();
    CK(hipEventRecord(b, g_stream));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    t.push_back(ms * 1000.0);
  }
  std::sort(t.begin(), t.end());
  CK(hipEventDestroy(a));
  CK(hipEventDestroy(b));
  return t[t.size() / 2];
}

// ---------------------------------------------------------------- problem data
// q, k: [S][T][HK][DK] (L2-normalized per head, BF16-representable)
// v:    [S][T][HV][DV] (BF16-representable); g, beta: [S][T][HV]
// s0:   [S][HV][DV][DK] (state, "VK" layout: s0[v][k] = S[k][v])
struct Problem {
  int S, T;
  std::vector<float> q, k, v, g, beta, s0;
};
static Problem make_problem(int S, int T, uint64_t seed) {
  Problem p{S, T};
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto norm_heads = [&](std::vector<float>& x, int heads, int d) {
    x.resize((size_t)S * T * heads * d);
    for (size_t r = 0; r < (size_t)S * T * heads; r++) {
      double ss = 0;
      for (int i = 0; i < d; i++) {
        float z = n01(rng);
        x[r * d + i] = z;
        ss += (double)z * z;
      }
      float inv = (float)(1.0 / std::sqrt(ss + 1e-6));
      for (int i = 0; i < d; i++) x[r * d + i] = bf16r(x[r * d + i] * inv);
    }
  };
  norm_heads(p.q, HK, DK);
  norm_heads(p.k, HK, DK);
  p.v.resize((size_t)S * T * HV * DV);
  for (auto& x : p.v) x = bf16r(n01(rng));
  p.g.resize((size_t)S * T * HV);
  p.beta.resize(p.g.size());
  for (size_t i = 0; i < p.g.size(); i++) {
    float a = n01(rng);
    p.g[i] = -0.5f * std::log1p(std::exp(a));  // -0.5 softplus: (-inf, 0)
    p.beta[i] = 1.f / (1.f + std::exp(-n01(rng)));
  }
  p.s0.resize((size_t)S * HV * DV * DK);
  for (auto& x : p.s0) x = 0.1f * n01(rng);
  return p;
}

// mapping: 0 = repeat_interleave (h / (HV/HK), transformers + AITER), 1 = tiled (h % HK, llama.cpp)
static int khead(int h, int mapping) { return mapping == 0 ? h / (HV / HK) : h % HK; }

// out: [S][T][HV][DV], s_out: [S][HV][DV][DK]
static void ref_gdn(const Problem& p, int mapping, std::vector<double>& out, std::vector<double>& s_out) {
  out.assign((size_t)p.S * p.T * HV * DV, 0);
  s_out.assign((size_t)p.S * HV * DV * DK, 0);
  const double scale = 1.0 / std::sqrt((double)DK);
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int s = 0; s < p.S; s++)
    for (int h = 0; h < HV; h++) {
      int hk = khead(h, mapping);
      std::vector<double> st(DV * DK);  // st[v][k]
      for (int i = 0; i < DV * DK; i++) st[i] = p.s0[((size_t)s * HV + h) * DV * DK + i];
      for (int t = 0; t < p.T; t++) {
        const float* q = &p.q[(((size_t)s * p.T + t) * HK + hk) * DK];
        const float* k = &p.k[(((size_t)s * p.T + t) * HK + hk) * DK];
        const float* v = &p.v[(((size_t)s * p.T + t) * HV + h) * DV];
        size_t gi = ((size_t)s * p.T + t) * HV + h;
        double dec = std::exp((double)p.g[gi]), b = p.beta[gi];
        for (int c = 0; c < DV; c++) {
          double* row = &st[c * DK];
          double kv = 0;
          for (int i = 0; i < DK; i++) {
            row[i] *= dec;
            kv += row[i] * k[i];
          }
          double delta = (v[c] - kv) * b, o = 0;
          for (int i = 0; i < DK; i++) {
            row[i] += k[i] * delta;
            o += row[i] * q[i];
          }
          out[(((size_t)s * p.T + t) * HV + h) * DV + c] = o * scale;
        }
      }
      for (int i = 0; i < DV * DK; i++) s_out[((size_t)s * HV + h) * DV * DK + i] = st[i];
    }
}

// ---------------------------------------------------------------- llama.cpp GDN
struct LlamaGdn {
  float *q, *k, *v, *g, *b, *s0, *out, *s1;
  int S, T;
  explicit LlamaGdn(const Problem& p) : S(p.S), T(p.T) {
    q = dup(p.q), k = dup(p.k), v = dup(p.v), g = dup(p.g), b = dup(p.beta), s0 = dup(p.s0);
    out = dalloc<float>((size_t)S * T * HV * DV);
    s1 = dalloc<float>((size_t)S * HV * DV * DK);
  }
  void run() {
    // ggml layout: q/k ne = [DK, HK, T, S], v ne = [DV, HV, T, S], g/beta ne = [1, HV, T, S]
    launch_gated_delta_net<false, false>(q, k, v, g, b, s0, out, s1, DV, HV, T, S,
                                         /*sq1*/ DK, /*sq2*/ (int64_t)HK * DK, /*sq3*/ (int64_t)T * HK * DK,
                                         /*sv1*/ DV, /*sv2*/ (int64_t)HV * DV, /*sv3*/ (int64_t)T * HV * DV,
                                         /*sb1*/ 1, /*sb2*/ HV, /*sb3*/ (int64_t)T * HV,
                                         /*neqk1*/ HK, /*rq3*/ 1, 1.0f / std::sqrt((float)DV),
                                         (int64_t)DV * DV * HV * S, 1, g_stream);
  }
  ~LlamaGdn() {
    for (void* x : {(void*)q, (void*)k, (void*)v, (void*)g, (void*)b, (void*)s0, (void*)out, (void*)s1})
      CK(hipFree(x));
  }
};

// ---------------------------------------------------------------- own FLA glue stages
// own_prep: one workgroup per (chunk, value head). Chunk-local cumsum of g, then
// A[i][j] = beta_i (k_i.k_j) exp(gc_i - gc_j) (j < i), T = (I + A)^-1 by forward
// substitution, w = T (beta exp(gc) k), u = T (beta v). Writes gc [T][HV] F32,
// w [HV][T][DK] BF16, u [HV][T][DV] BF16.
__global__ __launch_bounds__(256) void own_prep(const uint16_t* __restrict__ kx, const uint16_t* __restrict__ vx,
                                                const float* __restrict__ g, const float* __restrict__ beta,
                                                float* __restrict__ gc_out, uint16_t* __restrict__ w,
                                                uint16_t* __restrict__ u, int T) {
  const int c = blockIdx.x, h = blockIdx.y, hk = h / (HV / HK), tid = threadIdx.x;
  const int t0 = c * BT, L = min(BT, T - t0);
  __shared__ float sk[BT][DK + 1];  // k (then v) rows, F32
  __shared__ float sA[BT][BT + 1];
  __shared__ float sg[BT], sb[BT];
  if (tid < BT) {
    sb[tid] = tid < L ? beta[(size_t)(t0 + tid) * HV + h] : 0.f;
    sg[tid] = tid < L ? g[(size_t)(t0 + tid) * HV + h] : 0.f;
  }
  for (int e = tid; e < BT * DK; e += 256) {
    int r = e / DK, d = e % DK;
    sk[r][d] = r < L ? bits2f_dev(kx[((size_t)(t0 + r) * HK + hk) * DK + d]) : 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0;
    for (int r = 0; r < BT; r++) {
      acc += sg[r];
      sg[r] = acc;
    }
  }
  __syncthreads();
  if (tid < L) gc_out[(size_t)(t0 + tid) * HV + h] = sg[tid];
  for (int e = tid; e < BT * BT; e += 256) {
    int i = e / BT, j = e % BT;
    float a = 0;
    if (j < i) {
      float d = 0;
      for (int x = 0; x < DK; x++) d += sk[i][x] * sk[j][x];
      a = sb[i] * d * __expf(sg[i] - sg[j]);
    }
    sA[i][j] = a;
  }
  __syncthreads();
  // T = (I + A)^-1 in place of A, row by row: T[i][:] = e_i - sum_{j<i} A[i][j] T[j][:]
  for (int i = 0; i < BT; i++) {
    float acc = 0.f;
    if (tid < BT) {
      int col = tid;
      acc = (col == i) ? 1.f : 0.f;
      for (int j = col; j < i; j++) acc -= sA[i][j] * sA[j][col];
      if (col > i) acc = 0.f;
    }
    __syncthreads();
    if (tid < BT) sA[i][tid] = acc;
    __syncthreads();
  }
#define sT sA
  // w = T @ (beta exp(gc) k)
  for (int e = tid; e < BT * DK; e += 256) {
    int i = e / DK, d = e % DK;
    float acc = 0;
    for (int j = 0; j <= i; j++) acc += sT[i][j] * sb[j] * __expf(sg[j]) * sk[j][d];
    if (i < L) w[((size_t)h * T + t0 + i) * DK + d] = f2bits_dev(acc);
  }
  __syncthreads();
  for (int e = tid; e < BT * DV; e += 256) {
    int r = e / DV, d = e % DV;
    sk[r][d] = r < L ? bits2f_dev(vx[((size_t)(t0 + r) * HV + h) * DV + d]) : 0.f;
  }
  __syncthreads();
  for (int e = tid; e < BT * DV; e += 256) {
    int i = e / DV, d = e % DV;
    float acc = 0;
    for (int j = 0; j <= i; j++) acc += sT[i][j] * sb[j] * sk[j][d];
    if (i < L) u[((size_t)h * T + t0 + i) * DV + d] = f2bits_dev(acc);
  }
#undef sT
}

// own_o: one workgroup per (chunk, value head):
// o_i = scale (exp(gc_i) q_i . h_c[v][:] + sum_{j<=i} (q_i.k_j) exp(gc_i - gc_j) v_new_j[v])
// h: [NT][HV][DV][DK] BF16 (AITER snapshot = state at chunk start), v_new [HV][T][DV] BF16.
__global__ __launch_bounds__(256) void own_o(const uint16_t* __restrict__ qx, const uint16_t* __restrict__ kx,
                                             const float* __restrict__ gc, const uint16_t* __restrict__ hs,
                                             const uint16_t* __restrict__ vnew, float* __restrict__ out, int T,
                                             float scale) {
  const int c = blockIdx.x, h = blockIdx.y, hk = h / (HV / HK), tid = threadIdx.x;
  const int t0 = c * BT, L = min(BT, T - t0);
  __shared__ float sq[BT][DK + 1];
  __shared__ float sP[BT][BT + 1];
  __shared__ float sg[BT];
  if (tid < BT) sg[tid] = tid < L ? gc[(size_t)(t0 + tid) * HV + h] : 0.f;
  for (int e = tid; e < BT * DK; e += 256) {
    int r = e / DK, d = e % DK;
    sq[r][d] = r < L ? bits2f_dev(qx[((size_t)(t0 + r) * HK + hk) * DK + d]) : 0.f;
  }
  __syncthreads();
  for (int e = tid; e < BT * BT; e += 256) {
    int i = e / BT, j = e % BT;
    float a = 0;
    if (j <= i && j < L) {
      const uint16_t* kr = &kx[((size_t)(t0 + j) * HK + hk) * DK];
      for (int x = 0; x < DK; x++) a += sq[i][x] * bits2f_dev(kr[x]);
      a *= __expf(sg[i] - sg[j]);
    }
    sP[i][j] = a;
  }
  __syncthreads();
  const int vcol = tid % DV, r0 = tid / DV;  // rows r0, r0+2, ...
  float acc[BT / 2];
#pragma unroll
  for (int r = 0; r < BT / 2; r++) acc[r] = 0;
  const uint16_t* hrow = &hs[(((size_t)c * HV + h) * DV + vcol) * DK];
  for (int d = 0; d < DK; d++) {
    float hv = bits2f_dev(hrow[d]);
#pragma unroll
    for (int r = 0; r < BT / 2; r++) acc[r] += sq[r0 + 2 * r][d] * hv;
  }
#pragma unroll
  for (int r = 0; r < BT / 2; r++) acc[r] *= __expf(sg[r0 + 2 * r]);
  for (int j = 0; j < L; j++) {
    float vn = bits2f_dev(vnew[((size_t)h * T + t0 + j) * DV + vcol]);
#pragma unroll
    for (int r = 0; r < BT / 2; r++) acc[r] += sP[r0 + 2 * r][j] * vn;
  }
#pragma unroll
  for (int r = 0; r < BT / 2; r++) {
    int i = r0 + 2 * r;
    if (i < L) out[((size_t)(t0 + i) * HV + h) * DV + vcol] = acc[r] * scale;
  }
}

struct AiterChunk {  // one sequence, T tokens
  int T, NT;
  uint16_t *q, *k, *v, *w, *u, *hs, *vnew;
  float *g, *b, *gc, *s0, *s1, *out;
  int32_t* dummy;
  int bv;
  explicit AiterChunk(const Problem& p, int bv_) : T(p.T), NT((p.T + BT - 1) / BT), bv(bv_) {
    q = dup(tobf(p.q)), k = dup(tobf(p.k)), v = dup(tobf(p.v));
    g = dup(p.g), b = dup(p.beta), s0 = dup(p.s0);
    w = dalloc<uint16_t>((size_t)HV * T * DK), u = dalloc<uint16_t>((size_t)HV * T * DV);
    vnew = dalloc<uint16_t>((size_t)HV * T * DV);
    hs = dalloc<uint16_t>((size_t)NT * HV * DV * DK);
    gc = dalloc<float>((size_t)T * HV);
    s1 = dalloc<float>((size_t)HV * DV * DK);
    out = dalloc<float>((size_t)T * HV * DV);
    dummy = dalloc<int32_t>(2);
  }
  void prep() { hipLaunchKernelGGL(own_prep, dim3(NT, HV), dim3(256), 0, g_stream, k, v, g, b, gc, w, u, T); }
  void chunk_h() {
    aiter_eval_chunk_h(bv, k, w, u, gc, s0, hs, vnew, s1, dummy, T, HV, HK, g_stream);
  }
  void o() {
    hipLaunchKernelGGL(own_o, dim3(NT, HV), dim3(256), 0, g_stream, q, k, gc, hs, vnew, out, T,
                       1.0f / std::sqrt((float)DK));
  }
  void run() {
    prep();
    chunk_h();
    o();
  }
  ~AiterChunk() {
    for (void* x : {(void*)q, (void*)k, (void*)v, (void*)w, (void*)u, (void*)hs, (void*)vnew, (void*)g, (void*)b,
                    (void*)gc, (void*)s0, (void*)s1, (void*)out, (void*)dummy})
      CK(hipFree(x));
  }
};

static void compare(const char* tag, const std::vector<double>& ro, const std::vector<double>& rs,
                    const std::vector<float>& go, const std::vector<float>& gs, Err& eo, Err& es) {
  for (size_t i = 0; i < ro.size(); i++) eo.add(ro[i], go[i]);
  for (size_t i = 0; i < rs.size(); i++) es.add(rs[i], gs[i]);
  (void)tag;
}

static int aiter_default_bv(int total_chunks, int max_seq_chunks) {  // aiter/ops _compute_bv
  hipDeviceProp_t pr;
  CK(hipGetDeviceProperties(&pr, 0));
  int lds = (int)pr.maxSharedMemoryPerMultiProcessor;
  if (lds <= 0) lds = (int)pr.sharedMemPerBlock;
  for (int bv : {64, 32, 16}) {
    int per = 32 * 1024 + 512 * bv;
    int resident = std::min(std::max(1, lds / per), 2);
    long total = (long)(128 / bv) * HV * total_chunks;
    long thr = (long)std::max(1, (pr.multiProcessorCount * resident) / 2) * max_seq_chunks;
    if (total >= thr) return bv;
  }
  return 16;
}

// ---------------------------------------------------------------- GDN runs
static void gdn_decode(int B) {
  Problem p = make_problem(B, 1, 1000 + B);
  std::vector<double> ro, rs;
  ref_gdn(p, 1, ro, rs);
  LlamaGdn L(p);
  L.run();
  CK(hipStreamSynchronize(g_stream));
  Err eo, es;
  compare("", ro, rs, down(L.out, ro.size()), down(L.s1, rs.size()), eo, es);
  double us = time_us([&] { L.run(); });
  double bytes = 2.0 * B * HV * DV * DK * 4;  // state read + write
  printf("GDN decode B=%-3d llama.cpp        %9.1f us  state %6.1f GB/s  out max|d| %.2e (rel %.2e) state max|d| %.2e (rel %.2e)\n",
         B, us, bytes / us / 1e3, eo.max_abs, eo.rel(), es.max_abs, es.rel());
}

static void gdn_prefill(int T, bool timing) {
  Problem p = make_problem(1, T, 2000 + T);
  std::vector<double> ro_t, rs_t, ro_i, rs_i;
  ref_gdn(p, 1, ro_t, rs_t);  // llama.cpp head mapping
  ref_gdn(p, 0, ro_i, rs_i);  // AITER / transformers head mapping
  {
    LlamaGdn L(p);
    L.run();
    CK(hipStreamSynchronize(g_stream));
    Err eo, es;
    compare("", ro_t, rs_t, down(L.out, ro_t.size()), down(L.s1, rs_t.size()), eo, es);
    double us = timing ? time_us([&] { L.run(); }, 2, T >= 16384 ? 10 : 50) : 0;
    printf("GDN prefill T=%-5d llama.cpp       %9.1f us                    out max|d| %.2e (rel %.2e) state max|d| %.2e (rel %.2e)\n",
           T, us, eo.max_abs, eo.rel(), es.max_abs, es.rel());
  }
  const int NT = (T + BT - 1) / BT;
  int def_bv = aiter_default_bv(NT, NT);
  for (int bv : {16, 32, 64}) {
    AiterChunk A(p, bv);
    A.run();
    CK(hipStreamSynchronize(g_stream));
    Err eo, es;
    compare("", ro_i, rs_i, down(A.out, ro_i.size()), down(A.s1, rs_i.size()), eo, es);
    double tp = 0, th = 0, to = 0, tt = 0;
    if (timing) {
      int it = T >= 16384 ? 20 : 50;
      tp = time_us([&] { A.prep(); }, 2, it);
      th = time_us([&] { A.chunk_h(); }, 2, it);
      to = time_us([&] { A.o(); }, 2, it);
      tt = time_us([&] { A.run(); }, 2, it);
    }
    printf("GDN prefill T=%-5d aiter bv=%-2d%s  %9.1f us  (own_prep %.1f + aiter chunk_h %.1f + own_o %.1f)  out max|d| %.2e (rel %.2e) state max|d| %.2e (rel %.2e)\n",
           T, bv, bv == def_bv ? "*" : " ", tt, tp, th, to, eo.max_abs, eo.rel(), es.max_abs, es.rel());
  }
}

// ---------------------------------------------------------------- conv
// x: [S][T][CH] token-major (the in_proj output), w: [CH][KW], bias [CH], state [S][CH][KW-1]
static void ref_conv(int S, int T, const std::vector<float>& x, const std::vector<float>& w,
                     const std::vector<float>& bias, const std::vector<float>& st, std::vector<double>& y,
                     std::vector<double>& st_out) {
  y.assign((size_t)S * T * CH, 0);
  st_out.assign((size_t)S * CH * (KW - 1), 0);
  for (int s = 0; s < S; s++)
    for (int c = 0; c < CH; c++) {
      std::vector<double> xf(KW - 1 + T);
      for (int j = 0; j < KW - 1; j++) xf[j] = st[((size_t)s * CH + c) * (KW - 1) + j];
      for (int t = 0; t < T; t++) xf[KW - 1 + t] = x[((size_t)s * T + t) * CH + c];
      for (int t = 0; t < T; t++) {
        double a = bias[c];
        for (int j = 0; j < KW; j++) a += w[c * KW + j] * xf[t + j];
        y[((size_t)s * T + t) * CH + c] = a / (1 + std::exp(-a));
      }
      for (int j = 0; j < KW - 1; j++) st_out[((size_t)s * CH + c) * (KW - 1) + j] = xf[T + j];
    }
}

static void conv_case(int S, int T) {
  std::mt19937_64 rng(3000 + S * 7 + T);
  std::normal_distribution<float> n01(0.f, 1.f);
  std::vector<float> x((size_t)S * T * CH), w(CH * KW), bias(CH), st((size_t)S * CH * (KW - 1));
  for (auto& e : x) e = bf16r(n01(rng));
  for (auto& e : w) e = bf16r(0.5f * n01(rng));
  for (auto& e : bias) e = bf16r(0.1f * n01(rng));
  for (auto& e : st) e = bf16r(n01(rng));
  std::vector<double> ry, rst;
  ref_conv(S, T, x, w, bias, st, ry, rst);
  const char* stage = T == 1 ? "decode " : "prefill";
  // llama.cpp: conv_x [S][CH][KW-1+T] channel-major (state ++ transposed x), F32
  {
    std::vector<float> cx((size_t)S * CH * (KW - 1 + T));
    for (int s = 0; s < S; s++)
      for (int c = 0; c < CH; c++) {
        float* r = &cx[((size_t)s * CH + c) * (KW - 1 + T)];
        for (int j = 0; j < KW - 1; j++) r[j] = st[((size_t)s * CH + c) * (KW - 1) + j];
        for (int t = 0; t < T; t++) r[KW - 1 + t] = x[((size_t)s * T + t) * CH + c];
      }
    float *dcx = dup(cx), *dw = dup(w), *db = dup(bias), *dy = dalloc<float>((size_t)S * T * CH);
    auto run = [&] {
      ssm_conv_f32_cuda<true>(dcx, dw, db, 4, (KW - 1 + T) * 4, CH * (KW - 1 + T) * 4, KW * 4, dy, 4, CH * 4,
                              T * CH * 4, KW, CH, T, S, g_stream);
    };
    run();
    CK(hipStreamSynchronize(g_stream));
    auto gy = down(dy, ry.size());
    Err e;
    for (size_t i = 0; i < ry.size(); i++) e.add(ry[i], gy[i]);
    double us = time_us(run);
    printf("conv %s S=%-3d T=%-5d llama.cpp ssm_conv   %8.1f us  y max|d| %.2e (rel %.2e)  [state concat/transposed input built outside]\n",
           stage, S, T, us, e.max_abs, e.rel());
    for (void* p : {(void*)dcx, (void*)dw, (void*)db, (void*)dy}) CK(hipFree(p));
  }
  uint16_t *dx = dup(tobf(x)), *dw = dup(tobf(w)), *db = dup(tobf(bias));
  std::vector<uint16_t> stb = tobf(st);
  if (T == 1) {  // AITER causal_conv1d_update: x [S][CH][1]
    uint16_t *dst = dup(stb), *dy = dalloc<uint16_t>((size_t)S * CH);
    aiter_tensor_t tx = tens(dx, AITER_DTYPE_bf16, {S, CH, 1}), ts = tens(dst, AITER_DTYPE_bf16, {S, CH, KW - 1}),
                   tw = tens(dw, AITER_DTYPE_bf16, {CH, KW}), tb = tens(db, AITER_DTYPE_bf16, {CH}),
                   ty = tens(dy, AITER_DTYPE_bf16, {S, CH, 1}), e1 = empty_t(AITER_DTYPE_i32),
                   e2 = empty_t(AITER_DTYPE_i32);
    aiter::causal_conv1d_update(tx, ts, tw, tb, ty, true, e1, e2, -1);
    CK(hipStreamSynchronize(g_stream));
    auto gy = down(dy, ry.size());
    auto gs = down(dst, rst.size());
    Err e, es;
    for (size_t i = 0; i < ry.size(); i++) e.add(ry[i], bits2f(gy[i]));
    for (size_t i = 0; i < rst.size(); i++) es.add(rst[i], bits2f(gs[i]));
    double us = time_us([&] { aiter::causal_conv1d_update(tx, ts, tw, tb, ty, true, e1, e2, -1); });
    printf("conv %s S=%-3d T=%-5d aiter conv1d_update  %8.1f us  y max|d| %.2e (rel %.2e) state max|d| %.2e\n", stage, S,
           T, us, e.max_abs, e.rel(), es.max_abs);
    CK(hipFree(dst));
    CK(hipFree(dy));
  } else {  // AITER causal_conv1d_fwd_split_qkv: x channel-last (shape [CH, S*T], strides [1, CH])
    int bm = 64;
    for (auto tm : {std::pair<int, int>{12, 8}, {24, 16}, {48, 32}})
      if ((double)T <= tm.first) {
        bm = tm.second;
        break;
      }
    std::vector<int32_t> qsl(S + 1), bptr, tco;
    for (int s = 0; s <= S; s++) qsl[s] = s * T;
    for (int s = 0; s < S; s++)
      for (int c = 0; c < (T + bm - 1) / bm; c++) bptr.push_back(s), tco.push_back(c);
    std::vector<int32_t> cidx(S);
    for (int s = 0; s < S; s++) cidx[s] = s;
    std::vector<uint8_t> hinit(S, 1);
    int32_t *dq = dup(qsl), *dbp = dup(bptr), *dtc = dup(tco), *dci = dup(cidx);
    uint8_t* dhi = dup(hinit);
    uint16_t *dst = dup(stb), *oq = dalloc<uint16_t>((size_t)S * T * HK * DK),
             *ok = dalloc<uint16_t>((size_t)S * T * HK * DK), *ov = dalloc<uint16_t>((size_t)S * T * HV * DV);
    int ST = S * T, kd = HK * DK, vd = HV * DV;
    auto run = [&](uint16_t* states) {
      aiter::causal_conv1d_fwd_split_qkv_hip(
          tens(dx, AITER_DTYPE_bf16, {CH, ST}, {1, CH}), tens(dw, AITER_DTYPE_bf16, {CH, KW}),
          tens(db, AITER_DTYPE_bf16, {CH}), tens(states, AITER_DTYPE_bf16, {S, CH, KW - 1}),
          tens(dci, AITER_DTYPE_i32, {S}), tens(dhi, AITER_DTYPE_u8, {S}), tens(dq, AITER_DTYPE_i32, {S + 1}),
          tens(dbp, AITER_DTYPE_i32, {(int64_t)bptr.size()}), tens(dtc, AITER_DTYPE_i32, {(int64_t)tco.size()}),
          tens(oq, AITER_DTYPE_bf16, {ST, kd}), tens(ok, AITER_DTYPE_bf16, {ST, kd}),
          tens(ov, AITER_DTYPE_bf16, {ST, vd}), kd, vd, (int64_t)bptr.size(), bm, true, true, -1);
    };
    run(dst);
    CK(hipStreamSynchronize(g_stream));
    auto gq = down(oq, (size_t)ST * kd), gk = down(ok, (size_t)ST * kd), gv = down(ov, (size_t)ST * vd);
    auto gs = down(dst, rst.size());
    Err e, es;
    for (int r = 0; r < ST; r++)
      for (int c = 0; c < CH; c++) {
        float got = c < kd ? bits2f(gq[(size_t)r * kd + c])
                           : (c < 2 * kd ? bits2f(gk[(size_t)r * kd + c - kd]) : bits2f(gv[(size_t)r * vd + c - 2 * kd]));
        e.add(ry[(size_t)r * CH + c], got);
      }
    for (size_t i = 0; i < rst.size(); i++) es.add(rst[i], bits2f(gs[i]));
    uint16_t* dst2 = dup(stb);
    double us = time_us([&] { run(dst2); });
    printf("conv %s S=%-3d T=%-5d aiter split_qkv bm=%-2d %7.1f us  y max|d| %.2e (rel %.2e) state max|d| %.2e\n", stage,
           S, T, bm, us, e.max_abs, e.rel(), es.max_abs);
    for (void* p : {(void*)dq, (void*)dbp, (void*)dtc, (void*)dci, (void*)dhi, (void*)dst, (void*)dst2, (void*)oq,
                    (void*)ok, (void*)ov})
      CK(hipFree(p));
  }
  for (void* p : {(void*)dx, (void*)dw, (void*)db}) CK(hipFree(p));
}

int main(int argc, char** argv) {
  std::string only = argc > 1 ? argv[1] : "all";
  CK(hipSetDevice(0));
  CK(hipStreamCreate(&g_stream));
  aiter::eval_stream() = g_stream;
  hipDeviceProp_t pr;
  CK(hipGetDeviceProperties(&pr, 0));
  printf("device %s (%s), %d CUs, LDS/CU %zu\n", pr.name, pr.gcnArchName, pr.multiProcessorCount,
         (size_t)pr.maxSharedMemoryPerMultiProcessor);
  if (only == "all" || only == "decode")
    for (int B : {1, 16, 64}) gdn_decode(B);
  if (only == "all" || only == "prefill") {
    gdn_prefill(100, false);  // correctness only: a tail chunk (100 = 64 + 36)
    for (int T : {128, 2048, 16384}) gdn_prefill(T, true);
  }
  if (only == "all" || only == "conv") {
    for (int B : {1, 16, 64}) conv_case(B, 1);
    for (int T : {128, 2048, 16384}) conv_case(1, T);
  }
  return 0;
}
