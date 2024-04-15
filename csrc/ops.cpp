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

Tensor fill(const shape_t &shape, DType dtype, double value) {
  Tensor t = Tensor(shape, dtype);
  cpu::fill(t, value, shape);
  return t;
}

Tensor neg(const Tensor &a) {
  Tensor minus_one = fill(a.shape(), a.dtype(), -1.0);
  return mul(a, minus_one);
}

} // namespace pg