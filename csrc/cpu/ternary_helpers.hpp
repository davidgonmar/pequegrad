#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include <functional>
#include <vector>
namespace pg {
namespace cpu {
enum class TernaryOpType {
  Where,
};

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

template <typename T>
static inline void _dispatch_teranry_op_helper(
    const shape_t &shape, const strides_t &a_strides,
    const strides_t &b_strides, const strides_t &c_strides,
    const strides_t &result_strides, const void *a, const void *b,
    const void *c, void *result, TernaryOpType op) {
  switch (op) {
  case TernaryOpType::Where:
    ternar_op_ker<T>(static_cast<const T *>(a), static_cast<const T *>(b),
                     static_cast<const T *>(c), static_cast<T *>(result),
                     a_strides, b_strides, c_strides, result_strides, shape,
                     [](T a, T b, T c) { return a ? b : c; });
    break;
  default:
    throw std::runtime_error("Unsupported operation");
  }
}

static inline void
dispatch_ternary_op(const shape_t &shape, const strides_t &a_strides,
                    const strides_t &b_strides, const strides_t &c_strides,
                    const strides_t &result_strides, const void *a,
                    const void *b, const void *c, void *result, DType dtype,
                    TernaryOpType op) {
  switch (dtype) {
  case DType::Float32:
    _dispatch_teranry_op_helper<float>(shape, a_strides, b_strides, c_strides,
                                       result_strides, a, b, c, result, op);
    break;
  case DType::Float64:
    _dispatch_teranry_op_helper<double>(shape, a_strides, b_strides, c_strides,
                                        result_strides, a, b, c, result, op);
    break;
  case DType::Int32:
    _dispatch_teranry_op_helper<int32_t>(shape, a_strides, b_strides, c_strides,
                                         result_strides, a, b, c, result, op);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}
} // namespace cpu
} // namespace pg
