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

#define DECLARE_BINARY_OP(name)                                                \
  Tensor name(const Tensor &a, const Tensor &b);                               \
  Tensor name(const Tensor &a, double b);                                      \
  Tensor name(double a, const Tensor &b);

Tensor select(const Tensor &a, const std::vector<hl_select_t> &_items);

Tensor add_inplace(Tensor &a, const Tensor &b);
DECLARE_BINARY_OP(add)
DECLARE_BINARY_OP(sub)
DECLARE_BINARY_OP(mul)
DECLARE_BINARY_OP(div)
DECLARE_BINARY_OP(pow)
DECLARE_BINARY_OP(gt)
DECLARE_BINARY_OP(lt)
DECLARE_BINARY_OP(eq)
DECLARE_BINARY_OP(neq)

Tensor log(const Tensor &a);
Tensor neg(const Tensor &a);

DEFINE_REDUCE_OP(sum)
DEFINE_REDUCE_OP(max_reduce)
DEFINE_REDUCE_OP(mean)

Tensor fill(const shape_t &shape, DType dtype, double value,
            std::shared_ptr<device::Device> device);
Tensor bilinear_resize(const Tensor &a, const shape_t &new_shape);
Tensor one_hot(const Tensor &a, int num_classes);
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
DECLARE_BINARY_OP(max)
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

Tensor binomial(const double p, const shape_t &shape, const DType dtype,
                std::shared_ptr<device::Device> device);
} // namespace pg