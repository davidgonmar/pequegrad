#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"


#define DEFINE_REDUCE_OP(name) \
  Tensor name(const Tensor &a, const axes_t &axes, bool keepdims); \
  Tensor name(const Tensor &a, bool keepdims); \
  Tensor name(const Tensor &a, axis_t axis, bool keepdims);


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



DEFINE_REDUCE_OP(sum)
DEFINE_REDUCE_OP(max_reduce)
DEFINE_REDUCE_OP(mean)

Tensor fill(const shape_t &shape, DType dtype, double value);

Tensor broadcast_to(const Tensor &a, const shape_t &shape);
Tensor broadcast_as(const Tensor &a, const Tensor &b);

Tensor squeeze(const Tensor &a, const axes_t &axes);
Tensor squeeze(const Tensor &a, axis_t axis);
Tensor squeeze(const Tensor &a);

Tensor expand_dims(const Tensor &a, axis_t axis);
Tensor expand_dims(const Tensor &a, const axes_t &axes);

Tensor unsqueeze(const Tensor &a, axis_t axis);
Tensor unsqueeze(const Tensor &a, const axes_t &axes);

Tensor permute(const Tensor &a, const axes_t &axes);
} // namespace pg