#include "dtype.hpp"
#include <vector>

namespace bh {
using shape_t = std::vector<size_t>;

enum class BinaryOpType {
  Add,
  Sub,
  Mul,
  Div,
  Gt,
  Lt,
  Eq,
  Neq,
  Ge,
  Le,
  Pow,
  Max
};

template <typename T> using Op = std::function<T(T, T)>;

#include <vector>

template <typename T>
void binary_op_ker(const T *lhs, const T *rhs, T *result,
                   const std::vector<size_t> &lhs_strides,
                   const std::vector<size_t> &rhs_strides,
                   const std::vector<size_t> &result_strides,
                   const std::vector<size_t> &shape, Op<T> op) {

  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
#pragma omp parallel for
  for (int index = 0; index < total_elements; index++) {
    size_t offset_lhs = 0;
    size_t offset_rhs = 0;
    size_t offset_result = 0;
    size_t index_copy = index;
    for (int dim_idx = shape.size() - 1; dim_idx >= 0; dim_idx--) {
      offset_lhs +=
          (index_copy % shape[dim_idx]) * lhs_strides[dim_idx] / sizeof(T);
      offset_rhs +=
          (index_copy % shape[dim_idx]) * rhs_strides[dim_idx] / sizeof(T);
      offset_result +=
          (index_copy % shape[dim_idx]) * result_strides[dim_idx] / sizeof(T);
      index_copy /= shape[dim_idx];
    }
    result[offset_result] = op(lhs[offset_lhs], rhs[offset_rhs]);
  }
}

#define BINOP_HELPER_CASE(BINOP_TYPE, LAMBDA)                                  \
  case BinaryOpType::BINOP_TYPE:                                               \
    binary_op_ker<T>(static_cast<const T *>(lhs), static_cast<const T *>(rhs), \
                     static_cast<T *>(result), lhs_strides, rhs_strides,       \
                     result_strides, shape, LAMBDA);                           \
    break;

template <typename T>
static inline void
_dispatch_binary_op_helper(const shape_t &shape, const shape_t &lhs_strides,
                           const shape_t &rhs_strides,
                           const shape_t &result_strides, const void *lhs,
                           const void *rhs, void *result, BinaryOpType op) {
  switch (op) {
    BINOP_HELPER_CASE(Add, [](T a, T b) { return a + b; })
    BINOP_HELPER_CASE(Sub, [](T a, T b) { return a - b; })
    BINOP_HELPER_CASE(Mul, [](T a, T b) { return a * b; })
    BINOP_HELPER_CASE(Div, [](T a, T b) { return a / b; })
    BINOP_HELPER_CASE(Gt, [](T a, T b) { return a > b; })
    BINOP_HELPER_CASE(Lt, [](T a, T b) { return a < b; })
    BINOP_HELPER_CASE(Eq, [](T a, T b) { return a == b; })
    BINOP_HELPER_CASE(Neq, [](T a, T b) { return a != b; })
    BINOP_HELPER_CASE(Ge, [](T a, T b) { return a >= b; })
    BINOP_HELPER_CASE(Le, [](T a, T b) { return a <= b; })
    BINOP_HELPER_CASE(Pow, [](T a, T b) { return std::pow(a, b); })
    BINOP_HELPER_CASE(Max, [](T a, T b) { return std::max(a, b); })
  }
}

static inline void
dispatch_binary_op(const shape_t &shape, const shape_t &lhs_strides,
                   const shape_t &rhs_strides, const shape_t &result_strides,
                   const void *lhs, const void *rhs, void *result, DType dtype,
                   BinaryOpType op) {
  switch (dtype) {
  case DType::Float32:
    _dispatch_binary_op_helper<float>(shape, lhs_strides, rhs_strides,
                                      result_strides, lhs, rhs, result, op);
    break;
  case DType::Int32:
    _dispatch_binary_op_helper<int>(shape, lhs_strides, rhs_strides,
                                    result_strides, lhs, rhs, result, op);
    break;
  case DType::Float64:
    _dispatch_binary_op_helper<double>(shape, lhs_strides, rhs_strides,
                                       result_strides, lhs, rhs, result, op);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

} // namespace bh