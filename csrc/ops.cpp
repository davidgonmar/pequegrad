#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"
#include "init_primitives.hpp"

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

Tensor fill(const shape_t &shape, DType dtype, double value) {
  Tensor t = Tensor(shape, dtype);
  cpu::fill(t, value, shape);
  return t;
}

Tensor neg(const Tensor &a) {
  Tensor minus_one = fill(a.shape(), a.dtype(), -1.0);
  return mul(a, minus_one);
}


#define DEFINE_REDUCE_OP(op_name, functor) \
  Tensor op_name(const Tensor &a, const axes_t &axes, bool keepdims) { \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims), {a}); \
  } \
  Tensor op_name(const Tensor &a, bool keepdims) { \
    axes_t axes(a.shape().size()); \
    std::iota(axes.begin(), axes.end(), 0); \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims), {a}); \
  } \
  Tensor op_name(const Tensor &a, axis_t axis, bool keepdims) { \
    axes_t axes = {axis}; \
    return Tensor::from_primitive(std::make_shared<functor>(axes, keepdims), {a}); \
  }


DEFINE_REDUCE_OP(sum, Sum)
DEFINE_REDUCE_OP(max_reduce, MaxReduce)
DEFINE_REDUCE_OP(mean, Mean)

} // namespace pg