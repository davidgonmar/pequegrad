#include "cpu_tensor.hpp"
#include "./copy.hpp"
#include "./matmul.hpp"
#include "binary_helpers.hpp"
#include "immintrin.h"
#include "shape.hpp"
#include "unary_vectorized.hpp"
#include "utils.hpp"
#include <cblas.h>
#include "./reducers.hpp"

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

CpuTensor CpuTensor::as_contiguous() const {
  if (is_contiguous()) {
    return *this;
  } else {
    CpuTensor out(shape, dtype);
    copy::dispatch_copy(shape, strides, out.strides, ptr.get(), out.ptr.get(),
                        dtype);
    return out;
  }
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

CpuTensor CpuTensor::matmul(const CpuTensor &other) const {

  if (dtype != other.dtype) {
    throw std::runtime_error("Data types do not match");
  }
  bool added_a_dim = false;
  bool added_b_dim = false;
  // if a is a vector and the other is a matrix, add a dimension to a
  CpuTensor a = (this->ndim() == 1 && other.ndim() != 1)
                    ? this->unsqueeze(0)
                    : this->as_contiguous();
  if (this->ndim() == 1 && other.ndim() != 1) {
    added_a_dim = true;
  }
  CpuTensor b = (other.ndim() == 1 && this->ndim() != 1)
                    ? other.unsqueeze(1)
                    : other.as_contiguous();
  if (other.ndim() == 1 && this->ndim() != 1) {
    added_b_dim = true;
  }
  shape_t new_shape;
  size_t size1, midsize, size2;
  if (a.ndim() == 1 && b.ndim() == 1) {
    CpuTensor out({1}, dtype);
    dispatch_contiguous_dot_ker(a.ptr.get(), b.ptr.get(), out.ptr.get(),
                                a.shape[0], dtype);
    return out.squeeze();
  } else {
    int a_prod = std::accumulate(a.shape.begin(), a.shape.end() - 2, 1,
                                 std::multiplies<int>());
    int b_prod = std::accumulate(b.shape.begin(), b.shape.end() - 2, 1,
                                 std::multiplies<int>());
    if (a.ndim() > b.ndim()) { // we will try to broadcast, but keep
                               // last to dims
      shape_t b_new = shape_t(a.shape);
      b_new[b_new.size() - 1] = b.shape[b.shape.size() - 1];
      b_new[b_new.size() - 2] = b.shape[b.shape.size() - 2];
      b = b.broadcast_to(b_new).as_contiguous();
    } else if (a.ndim() < b.ndim()) {
      shape_t a_new = shape_t(b.shape);
      a_new[a_new.size() - 1] = a.shape[a.shape.size() - 1];
      a_new[a_new.size() - 2] = a.shape[a.shape.size() - 2];
      a = a.broadcast_to(a_new).as_contiguous();
      // if ndim are equal, we will broadcast the one with the smallest product
    } else if (a_prod >= b_prod) {
      shape_t b_new = shape_t(a.shape);
      b_new[b_new.size() - 1] = b.shape[b.shape.size() - 1];
      b_new[b_new.size() - 2] = b.shape[b.shape.size() - 2];
      b = b.broadcast_to(b_new).as_contiguous();
    } else if (a_prod < b_prod) {
      shape_t a_new = shape_t(b.shape);
      a_new[a_new.size() - 1] = a.shape[a.shape.size() - 1];
      a_new[a_new.size() - 2] = a.shape[a.shape.size() - 2];
      a = a.broadcast_to(a_new).as_contiguous();
    }
    size1 = a.shape.at(a.ndim() - 2);
    midsize = a.shape.at(a.ndim() - 1);
    size2 = b.shape.at(b.ndim() - 1);
    new_shape = a.shape;
    new_shape[new_shape.size() - 1] = size2;
    if (added_a_dim) {
      new_shape.erase(new_shape.begin());
    }
    if (added_b_dim) {
      new_shape.erase(new_shape.end() - 1);
    }
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>());

    CpuTensor out(new_shape, dtype);
    size_t M = size1;
    size_t N = size2;
    size_t K = midsize;
    size_t B = new_size / (M * N); // batch size
    dispatch_contiguous_matmul_ker(a.ptr.get(), b.ptr.get(), out.ptr.get(), M,
                                   N, K, B, dtype);
    return out;
  }
}


CpuTensor CpuTensor::reduce(ReduceOp ker, axis_t axis,
                              bool keepdims) const {
  if (!is_contiguous()) {
    return as_contiguous().reduce(ker, axis, keepdims);
  }
  // if axis is negative, we need to convert it to a positive axis
  if (axis < 0) {
    axis = shape.size() + axis;
  }
  PG_CHECK_ARG(axis < shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  shape_t new_shape = shape;
  new_shape[axis] = 1;
  size_t new_size = size() / shape[axis];
  size_t n_dims = shape.size();
  CpuTensor out(new_shape, dtype);
  dispatch_reduce(ptr.get(), out.ptr.get(), strides, shape, axis, dtype, ker);
  if (keepdims) {
    return out;
  }
  return out.squeeze(axis);
}

CpuTensor CpuTensor::reduce(ReduceOp ker, axes_t axes,
                              bool keepdims) const {
  CpuTensor out = *this;
  for (size_t axis : axes) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze(axes);
}

CpuTensor CpuTensor::reduce(ReduceOp ker, bool keepdims) const {
  CpuTensor out = *this;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze();
}

CpuTensor CpuTensor::sum(axis_t axis, bool keepdims) const {
  return reduce(ReduceOp::Sum, axis, keepdims);
}
CpuTensor CpuTensor::sum(axes_t axes, bool keepdims) const {
  return reduce(ReduceOp::Sum, axes, keepdims);
}
CpuTensor CpuTensor::sum(bool keepdims) const {
  return reduce(ReduceOp::Sum, keepdims);
}
CpuTensor CpuTensor::max(axis_t axis, bool keepdims) const {
  return reduce(ReduceOp::Max, axis, keepdims);
}
CpuTensor CpuTensor::max(axes_t axes, bool keepdims) const {
  return reduce(ReduceOp::Max, axes, keepdims);
}
CpuTensor CpuTensor::max(bool keepdims) const {
  return reduce(ReduceOp::Max, keepdims);
}

CpuTensor CpuTensor::mean(axis_t axis, bool keepdims) const {
  return reduce(ReduceOp::Mean, axis, keepdims);
}
CpuTensor CpuTensor::mean(axes_t axes, bool keepdims) const {
  return reduce(ReduceOp::Mean, axes, keepdims);
}
CpuTensor CpuTensor::mean(bool keepdims) const {
  return reduce(ReduceOp::Mean, keepdims);
}