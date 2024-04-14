#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"

namespace pg {
Tensor add(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
} // namespace pg