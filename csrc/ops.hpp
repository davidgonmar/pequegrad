#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"

namespace pg {
Tensor add(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor pow(const Tensor &a, const Tensor &b);
Tensor gt(const Tensor &a, const Tensor &b);
Tensor lt(const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor neq(const Tensor &a, const Tensor &b);
Tensor log(const Tensor &a);
Tensor neg(const Tensor &a);
Tensor fill(const shape_t &shape, DType dtype, double value);
} // namespace pg