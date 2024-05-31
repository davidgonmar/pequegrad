#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cpu {
template <typename T>
void copy_ker(const shape_t &shape, const T *in, T *out,
              const strides_t &in_strides, const strides_t &out_strides) {

  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
#pragma omp parallel for collapse(1)
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

// Casting
template <typename T, typename U>
void cast_ker(const shape_t &shape, const T *in, U *out,
              const strides_t &in_strides, const strides_t &out_strides) {
  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
#pragma omp parallel for collapse(1)
  for (int index = 0; index < total_elements; index++) {
    size_t offset_in = 0;
    size_t offset_out = 0;
    size_t index_copy = index;
    for (int dim_idx = shape.size() - 1; dim_idx >= 0; dim_idx--) {
      offset_in +=
          (index_copy % shape[dim_idx]) * in_strides[dim_idx] / sizeof(T);
      offset_out +=
          (index_copy % shape[dim_idx]) * out_strides[dim_idx] / sizeof(U);

      index_copy /= shape[dim_idx];
    }
    out[offset_out] = static_cast<U>(in[offset_in]);
  }
}

namespace view {
View as_contiguous(const View &view, bool force = false);
View astype(const View &view, DType dtype);
void copy_data(const View &src, View &dst);
} // namespace view
} // namespace cpu
} // namespace pg