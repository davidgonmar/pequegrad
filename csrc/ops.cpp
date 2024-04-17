#pragma once

#include "ad_primitives.hpp"
#include "init_primitives.hpp"
#include "tensor.hpp"

namespace pg {
Tensor add(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Add>(), {a, b});
}
Tensor mul(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Mul>(), {a, b});
}
Tensor sub(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Sub>(), {a, b});
}
Tensor div(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Div>(), {a, b});
}

Tensor gt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Gt>(), {a, b});
}

Tensor lt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Lt>(), {a, b});
}

Tensor eq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Eq>(), {a, b});
}

Tensor neq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Neq>(), {a, b});
}

Tensor pow(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Pow>(), {a, b});
}

Tensor log(const Tensor &a) {
  return Tensor::from_primitive(std::make_shared<Log>(), {a});
}

Tensor fill(const shape_t &shape, DType dtype, double value,
            device::DeviceKind device) {
  Tensor t = Tensor(shape, dtype, device);
  if (device == device::CPU) {
    cpu::fill(t, value, shape);
    return t;
  } else {
    cuda::fill(t, value, shape);
  }
  return t;
}

Tensor neg(const Tensor &a) {
  Tensor minus_one = fill(a.shape(), a.dtype(), -1.0, a.device());
  return mul(a, minus_one);
}

#define DEFINE_REDUCE_OP(op_name, functor)                                     \
  Tensor op_name(const Tensor &a, const axes_t &axes, bool keepdims) {         \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims),   \
                                  {a});                                        \
  }                                                                            \
  Tensor op_name(const Tensor &a, bool keepdims) {                             \
    axes_t axes(a.shape().size());                                             \
    std::iota(axes.begin(), axes.end(), 0);                                    \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims),   \
                                  {a});                                        \
  }                                                                            \
  Tensor op_name(const Tensor &a, axis_t axis, bool keepdims) {                \
    axes_t axes = {axis};                                                      \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims),   \
                                  {a});                                        \
  }

DEFINE_REDUCE_OP(sum, Sum)
DEFINE_REDUCE_OP(max_reduce, MaxReduce)
DEFINE_REDUCE_OP(mean, Mean)

Tensor broadcast_to(const Tensor &a, const shape_t &shape) {
  return Tensor::from_primitive(std::make_shared<BroadcastTo>(shape), {a});
}

Tensor broadcast_as(const Tensor &a, const Tensor &b) {
  return broadcast_to(a, b.shape());
}

Tensor squeeze(const Tensor &a, const axes_t &axes) {
  return Tensor::from_primitive(std::make_shared<Squeeze>(axes), {a});
}

Tensor squeeze(const Tensor &a, axis_t axis) {
  return squeeze(a, axes_t{axis});
}

Tensor squeeze(const Tensor &a) {
  axes_t axes;
  for (size_t i = 0; i < a.shape().size(); i++) {
    if (a.shape()[i] == 1) {
      axes.push_back(i);
    }
  }
  return squeeze(a, axes);
}

Tensor unsqueeze(const Tensor &a, const axes_t &axes) {
  return Tensor::from_primitive(std::make_shared<Unsqueeze>(axes), {a});
}

Tensor unsqueeze(const Tensor &a, axis_t axis) {
  return Tensor::from_primitive(std::make_shared<Unsqueeze>(axes_t{axis}), {a});
}

Tensor expand_dims(const Tensor &a, axis_t axis) { return unsqueeze(a, axis); }

Tensor expand_dims(const Tensor &a, const axes_t &axes) {
  return unsqueeze(a, axes);
}

Tensor permute(const Tensor &a, const axes_t &axes) {
  return Tensor::from_primitive(std::make_shared<Permute>(axes), {a});
}

Tensor t(const Tensor &a) {
  if (a.shape().size() < 2) {
    return a;
  }
  axes_t axes(a.shape().size());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
  return permute(a, axes);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<MatMul>(), {a, b});
}

} // namespace pg