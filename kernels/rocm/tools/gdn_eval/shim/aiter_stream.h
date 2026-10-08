#pragma once
#include <hip/hip_runtime.h>
namespace aiter {
inline hipStream_t &eval_stream() {
  static hipStream_t s = nullptr;
  return s;
}
inline hipStream_t getCurrentHIPStream() { return eval_stream(); }
} // namespace aiter
