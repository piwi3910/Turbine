#pragma once
#include "aiter_hip_common.h"
struct aiter_tensor_t {
  void *ptr;
  size_t numel_;
  int ndim;
  int64_t shape[8];
  int64_t strides[8];
  AiterDtype dtype_;
  int device_id;
  int64_t size(int i) const { return (i < 0) ? shape[ndim + i] : shape[i]; }
  int64_t stride(int i) const {
    return (i < 0) ? strides[ndim + i] : strides[i];
  }
  void *data_ptr() const { return ptr; }
  size_t numel() const { return numel_; }
  int dim() const { return ndim; }
  AiterDtype dtype() const { return dtype_; }
  size_t element_size() const { return AiterDtype_element_size(dtype_); }
  bool is_gpu() const { return device_id >= 0; }
  bool is_cpu() const { return device_id == -1; }
  bool is_contiguous() const {
    if (numel_ == 0)
      return true;
    int64_t e = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      if (shape[d] != 1 && strides[d] != e)
        return false;
      e *= shape[d];
    }
    return true;
  }
};
