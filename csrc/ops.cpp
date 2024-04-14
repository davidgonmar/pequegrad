#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"

namespace pg {
Tensor add(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Add>(), {a, b});
}
Tensor mul(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Mul>(), {a, b});
}
} // namespace pg