#include "cpu_tensor.hpp"
#include "binary_helpers.hpp"
#include "immintrin.h"
#include "unary_vectorized.hpp"
#include "utils.hpp"
#include <cblas.h>

size_t CpuTensor::compute_nbytes(const shape_t &shape, DType dtype) const {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size * dtype_to_size(dtype);
}

CpuTensor binary_op(const CpuTensor &lhs, const CpuTensor &rhs,
                    bh::BinaryOpType op) {
  if (lhs.shape != rhs.shape) {
    throw std::runtime_error("Shapes do not match");
  }
  if (lhs.dtype != rhs.dtype) { // Only float32 supported
    throw std::runtime_error("Data types do not match");
  }

  void *result = (void *)new char[lhs.nbytes];
  auto res_tens = CpuTensor(
      lhs.shape, rhs.strides,
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

  return CpuTensor(a.shape, a.strides,
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
