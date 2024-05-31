#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include <functional>
#include <vector>
namespace pg {
namespace cpu {
template <typename T> using TerOp = std::function<T(T, T, T)>;

template <typename T>
void ternar_op_ker(const T *a, const T *b, const T *c, T *result,
                   const strides_t &a_strides, const strides_t &b_strides,
                   const strides_t &c_strides,

                   const strides_t &result_strides, const shape_t &shape,
                   TerOp<T> op) {

  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
#pragma omp parallel for
  for (int index = 0; index < total_elements; index++) {
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t offset_c = 0;
    size_t offset_result = 0;
    size_t index_copy = index;
    for (int dim_idx = shape.size() - 1; dim_idx >= 0; dim_idx--) {
      offset_a +=
          (index_copy % shape[dim_idx]) * a_strides[dim_idx] / sizeof(T);
      offset_b +=
          (index_copy % shape[dim_idx]) * b_strides[dim_idx] / sizeof(T);
      offset_c +=
          (index_copy % shape[dim_idx]) * c_strides[dim_idx] / sizeof(T);
      offset_result +=
          (index_copy % shape[dim_idx]) * result_strides[dim_idx] / sizeof(T);
      index_copy /= shape[dim_idx];
    }
    result[offset_result] = op(a[offset_a], b[offset_b], c[offset_c]);
  }
}
} // namespace cpu
} // namespace pg
