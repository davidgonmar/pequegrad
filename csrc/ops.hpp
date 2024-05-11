#pragma once

#include "ad_primitives.hpp"
#include "tensor.hpp"

#define DEFINE_REDUCE_OP(name)                                                 \
  Tensor name(const Tensor &a, const axes_t &axes, bool keepdims);             \
  Tensor name(const Tensor &a, bool keepdims);                                 \
  Tensor name(const Tensor &a, axis_t axis, bool keepdims);

namespace pg {
using hl_select_t =
    std::variant<SelectKeepDim, SelectWithSlice, SelectWithSingleIdx, Tensor>;
Tensor select(const Tensor &a, const std::vector<hl_select_t> &_items);
Tensor add(const Tensor &a, const Tensor &b);
Tensor add_inplace(Tensor &a, const Tensor &b);
Tensor add(const Tensor &a, double b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, double b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, double b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, double b);
Tensor pow(const Tensor &a, const Tensor &b);
Tensor pow(const Tensor &a, double b);

Tensor gt(const Tensor &a, const Tensor &b);
Tensor lt(const Tensor &a, const Tensor &b);
Tensor eq(const Tensor &a, const Tensor &b);
Tensor neq(const Tensor &a, const Tensor &b);
Tensor log(const Tensor &a);
Tensor neg(const Tensor &a);

DEFINE_REDUCE_OP(sum)
DEFINE_REDUCE_OP(max_reduce)
DEFINE_REDUCE_OP(mean)

Tensor fill(const shape_t &shape, DType dtype, double value,
            device::DeviceKind device);

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

Tensor t(const Tensor &a);
Tensor matmul(const Tensor &a, const Tensor &b);

Tensor where(const Tensor &condition, const Tensor &a, const Tensor &b);
Tensor max(const Tensor &a, const Tensor &b);

Tensor exp(const Tensor &a);

Tensor im2col(const Tensor &a, const shape_t &kernel_shape,
              const shape_t &stride, const shape_t &padding,
              const shape_t &dilation);
Tensor col2im(const Tensor &a, const shape_t &output_shape,
              const shape_t &kernel_shape, const shape_t &stride,
              const shape_t &padding, const shape_t &dilation);
Tensor reshape(const Tensor &a, const axes_t &shape);
Tensor reshape(const Tensor &a, const shape_t &shape);

Tensor as_contiguous(const Tensor &a);

std::vector<hl_select_t>
convert_from_select_t_to_hl_select_t(const select_t &items,
                                     const std::vector<Tensor> &t_indices);

Tensor assign_at(const Tensor &dst, const Tensor &src,
                 const std::vector<hl_select_t> &items);

Tensor astype(const Tensor &a, DType dtype);
} // namespace pg