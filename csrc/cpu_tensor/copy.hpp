#pragma once

#include "dtype.hpp"
#include <vector>

namespace copy {

template <typename T>
void copy_ker(const std::vector<size_t> &shape, const T *in, T *out,
              const std::vector<size_t> &in_strides,
              const std::vector<size_t> &out_strides) {

  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
#pragma omp parallel for
  for (int index = 0; index < total_elements; index++) {
    size_t offset_in = 0;
    size_t offset_out = 0;
    size_t index_copy = index;
    for (int dim_idx = shape.size() - 1; dim_idx >= 0; dim_idx--) {
      offset_in +=
          (index_copy % shape[dim_idx]) * in_strides[dim_idx] / sizeof(T);
      offset_out +=
          (index_copy % shape[dim_idx]) * out_strides[dim_idx] / sizeof(T);

      index_copy /= shape[dim_idx];
    }
    out[offset_out] = in[offset_in];
  }
}

static inline void dispatch_copy(const shape_t &shape,
                                 const shape_t &in_strides,
                                 const shape_t &out_strides, const void *in,
                                 void *out, DType dtype) {
  switch (dtype) {
  case DType::Float32:
    copy_ker<float>(shape, static_cast<const float *>(in),
                    static_cast<float *>(out), in_strides, out_strides);
    break;
  case DType::Float64:
    copy_ker<double>(shape, static_cast<const double *>(in),
                     static_cast<double *>(out), in_strides, out_strides);
    break;
  case DType::Int32:
    copy_ker<int32_t>(shape, static_cast<const int32_t *>(in),
                      static_cast<int32_t *>(out), in_strides, out_strides);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}
} // namespace copy
  // namespace cont