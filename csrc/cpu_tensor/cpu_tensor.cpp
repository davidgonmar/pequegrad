#include "cpu_tensor.hpp"
#include "binary_helpers.hpp"
#include "immintrin.h"
#include "shape.hpp"
#include "unary_vectorized.hpp"
#include "utils.hpp"
#include <cblas.h>

bool CpuTensor::is_contiguous() const {
  // if it is not dense, also not contiguous (todo -- handle this case)
  if (!is_dense()) {
    return false;
  }
  if (offset != 0) {
    return false;
  }
  if (strides.size() != shape.size()) {
    return false;
  }
  if (strides.size() == 0) { // scalar
    return true;
  }
  shape_t expected_strides(shape.size());
  expected_strides[shape.size() - 1] = dtype_to_size(dtype);
  for (int i = shape.size() - 2; i >= 0; --i) {
    expected_strides[i] = expected_strides[i + 1] * shape[i + 1];
  }
  if (expected_strides != strides) {
    return false;
  }
  return true;
}

bool CpuTensor::is_dense() const {
  // dense means that it might not be contiguous, but
  // there are no holes in the array
  // that is, the total number of elements is equal to
  // the size of the underlying storage
  size_t total_in_storage = nbytes;
  size_t total_size_in_bytes = size() * dtype_to_size(dtype);
  return total_in_storage == total_size_in_bytes;
}

size_t CpuTensor::compute_nbytes(const shape_t &shape, DType dtype) const {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size * dtype_to_size(dtype);
}
// Returns a view of the tensor with the given shape
CpuTensor CpuTensor::broadcast_to(const shape_t &new_shape) const {
  if (shape == new_shape) {
    return *this;
  }
  shape_t new_strides =
      get_strides_for_broadcasting(this->shape, this->strides, new_shape);

  return CpuTensor(nbytes, new_shape, new_strides, ptr, dtype);
}

CpuTensor binary_op(const CpuTensor &lhs, const CpuTensor &rhs,
                    bh::BinaryOpType op) {
  // we need to broadcast
  if (lhs.shape != rhs.shape) {
    // broadcast rhs -> lhs
    if (lhs.shape.size() > rhs.shape.size()) {
      return binary_op(lhs, rhs.broadcast_to(lhs.shape), op);
    } else if (lhs.shape.size() < rhs.shape.size()) {
      return binary_op(lhs.broadcast_to(rhs.shape), rhs, op);
    } else {
      // we need to try both, improve this logic
      try {
        return binary_op(lhs, rhs.broadcast_to(lhs.shape), op);
      } catch (...) {
        return binary_op(lhs.broadcast_to(rhs.shape), rhs, op);
      }
    }
  }
  if (lhs.dtype != rhs.dtype) { // Only float32 supported
    throw std::runtime_error("Data types do not match");
  }

  void *result =
      (void *)new char[std::accumulate(lhs.shape.begin(), lhs.shape.end(), 1,
                                       std::multiplies<size_t>()) *
                       dtype_to_size(lhs.dtype)];
  auto res_tens = CpuTensor(
      std::accumulate(lhs.shape.begin(), lhs.shape.end(), 1,
                      std::multiplies<size_t>()) *
          dtype_to_size(lhs.dtype),
      lhs.shape, compute_natural_strides(lhs.shape, lhs.dtype),
      std::shared_ptr<void>(result, [](void *p) { delete[] p; }), lhs.dtype);
  bh::dispatch_binary_op(lhs.shape, lhs.strides, rhs.strides, res_tens.strides,
                         lhs.ptr.get(), rhs.ptr.get(), res_tens.ptr.get(),
                         lhs.dtype, op);

  return res_tens;
}

CpuTensor CpuTensor::add(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Add);
}

CpuTensor CpuTensor::sub(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Sub);
}

CpuTensor CpuTensor::mul(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Mul);
}

CpuTensor CpuTensor::div(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Div);
}

CpuTensor CpuTensor::gt(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Gt);
}

CpuTensor CpuTensor::lt(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Lt);
}

CpuTensor CpuTensor::eq(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Eq);
}

CpuTensor CpuTensor::ne(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Neq);
}

CpuTensor CpuTensor::ge(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Ge);
}

CpuTensor CpuTensor::le(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Le);
}

CpuTensor CpuTensor::pow(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Pow);
}

CpuTensor CpuTensor::el_wise_max(const CpuTensor &other) const {
  return binary_op(*this, other, bh::BinaryOpType::Max);
}

CpuTensor prepare_for_unary_op(const CpuTensor &a) {
  void *ptr1 = a.ptr.get();
  void *ptr2 = malloc(a.nbytes);

  return CpuTensor(a.nbytes, a.shape, a.strides,
                   std::shared_ptr<void>(ptr2, [](void *p) { free(p); }),
                   a.dtype);
}
CpuTensor CpuTensor::exp() const {
  CpuTensor result = prepare_for_unary_op(*this);
  dispatch_unary_op(dtype, UnaryOp::Exp, ptr.get(), result.ptr.get(),
                    nbytes / dtype_to_size(dtype));
  return result;
}

CpuTensor CpuTensor::log() const {
  CpuTensor result = prepare_for_unary_op(*this);
  dispatch_unary_op(dtype, UnaryOp::Log, ptr.get(), result.ptr.get(),
                    nbytes / dtype_to_size(dtype));
  return result;
}
