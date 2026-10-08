// Turbine gdn_eval shim: the subset of AITER's aiter_hip_common.h the three
// evaluated kernels use (checks, dtype enum, device guard). Not AITER code.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>
enum AiterDtype {
  AITER_DTYPE_fp32,
  AITER_DTYPE_bf16,
  AITER_DTYPE_fp16,
  AITER_DTYPE_i32,
  AITER_DTYPE_u8
};
inline const char *AiterDtype_to_str(AiterDtype d) {
  switch (d) {
  case AITER_DTYPE_fp32:
    return "fp32";
  case AITER_DTYPE_bf16:
    return "bf16";
  case AITER_DTYPE_fp16:
    return "fp16";
  case AITER_DTYPE_i32:
    return "i32";
  default:
    return "u8";
  }
}
inline size_t AiterDtype_element_size(AiterDtype d) {
  return (d == AITER_DTYPE_fp32 || d == AITER_DTYPE_i32)
             ? 4
             : (d == AITER_DTYPE_u8 ? 1 : 2);
}
template <class... A> std::string aiter_msg(A &&...a) {
  std::ostringstream o;
  (o << ... << a);
  return o.str();
}
#define AITER_CHECK(cond, ...)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "AITER_CHECK %s: %s\n", #cond,                           \
              aiter_msg(__VA_ARGS__).c_str());                                 \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define HIP_CALL(x)                                                            \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__,    \
              __LINE__);                                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define HIP_CALL_LAUNCH(x) HIP_CALL(x)
struct HipDeviceGuard {
  explicit HipDeviceGuard(int) {}
};
