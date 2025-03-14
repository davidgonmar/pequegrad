#pragma once

#include "ops.hpp"
#include "ad_primitives.hpp"
#include "tensor.hpp"

namespace pg {

Tensor fill(const shape_t &shape, DType dtype, double value,
            std::shared_ptr<device::Device> device) {
  return Tensor::from_primitive_one(
      std::make_shared<Fill>(value, dtype, shape, device), {}, device);
}

Tensor broadcast_to(const Tensor &a, const shape_t &shape) {
  return Tensor::from_primitive_one(std::make_shared<BroadcastTo>(shape), {a});
}

Tensor _cudnn_sdpa(const Tensor &q, const Tensor &k, const Tensor &v) {
  return Tensor::from_primitive_one(std::make_shared<CudnnSdpa>(), {q, k, v});
}

static shape_t get_broadcasted_shapes(const shape_t &_a, const shape_t &_b) {
  auto a = shape_t(_a);
  auto b = shape_t(_b);
  std::reverse(a.begin(), a.end());
  std::reverse(b.begin(), b.end());
  size_t max_dim = std::max(a.size(), b.size());
  size_t min_dim = std::min(a.size(), b.size());
  shape_t new_shape(max_dim);
  for (size_t i = 0; i < max_dim; i++) {
    size_t a_dim = i < a.size() ? a[i] : 1;
    size_t b_dim = i < b.size() ? b[i] : 1;
    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::runtime_error("Shapes are not broadcastable: " +
                               vec_to_string(_a) + " and " + vec_to_string(_b));
    }
    new_shape[i] = std::max(a_dim, b_dim);
  }
  std::reverse(new_shape.begin(), new_shape.end());
  std::reverse(a.begin(), a.end());
  std::reverse(b.begin(), b.end());
  return new_shape;
}

static shape_t get_broadcasted_shapes(const shape_t &a, const shape_t &b,
                                      const shape_t &c) {
  size_t max_dim = std::max(a.size(), std::max(b.size(), c.size()));
  shape_t new_shape(max_dim);
  for (size_t i = 0; i < max_dim; i++) {
    size_t a_dim = i < a.size() ? a[i] : 1;
    size_t b_dim = i < b.size() ? b[i] : 1;
    size_t c_dim = i < c.size() ? c[i] : 1;
    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::runtime_error(
          "Shapes are not broadcastable: " + vec_to_string(a) + " and " +
          vec_to_string(b) + " and " + vec_to_string(c));
    }
    if (a_dim != c_dim && a_dim != 1 && c_dim != 1) {
      throw std::runtime_error(
          "Shapes are not broadcastable: " + vec_to_string(a) + " and " +
          vec_to_string(b) + " and " + vec_to_string(c));
    }
    if (b_dim != c_dim && b_dim != 1 && c_dim != 1) {
      throw std::runtime_error(
          "Shapes are not broadcastable: " + vec_to_string(b) + " and " +
          vec_to_string(b) + " and " + vec_to_string(c));
    }
    new_shape[i] = std::max(a_dim, std::max(b_dim, c_dim));
  }
  return new_shape;
}

static std::vector<Tensor> broadcast_tensors(const Tensor &a, const Tensor &b) {
  shape_t new_shape = get_broadcasted_shapes(a.shape(), b.shape());
  Tensor a_broadcasted = broadcast_to(a, new_shape);
  Tensor b_broadcasted = broadcast_to(b, new_shape);
  return {a_broadcasted, b_broadcasted};
}

#define DEFINE_BINOP_SCALAR_OVERLOAD(name)                                     \
  Tensor name(const Tensor &a, double b) {                                     \
    Tensor t = fill({}, a.dtype(), b, a.device());                             \
    return name(a, t);                                                         \
  }                                                                            \
  Tensor name(double a, const Tensor &b) {                                     \
    Tensor t = fill({}, b.dtype(), a, b.device());                             \
    return name(t, b);                                                         \
  }

Tensor add(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Add>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(add)

Tensor mul(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Mul>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(mul)

Tensor sub(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Sub>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(sub)

Tensor div(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Div>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(div)

Tensor gt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Gt>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(gt)

Tensor lt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Lt>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(lt)

Tensor eq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Eq>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(eq)

Tensor neq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Neq>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(neq)

Tensor pow(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Pow>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(pow)

Tensor max(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Max>(),
                                    broadcast_tensors(a, b));
}
DEFINE_BINOP_SCALAR_OVERLOAD(max)

Tensor log(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<Log>(), {a});
}

Tensor neg(const Tensor &a) {
  Tensor minus_one = fill({}, a.dtype(), -1.0, a.device());
  return mul(a, minus_one);
}

#define DEFINE_REDUCE_OP(op_name, functor)                                     \
  Tensor op_name(const Tensor &a, const axes_t &axes, bool keepdims) {         \
    return Tensor::from_primitive_one(                                         \
        std::make_shared<functor>(axes, keepdims), {a});                       \
  }                                                                            \
  Tensor op_name(const Tensor &a, bool keepdims) {                             \
    axes_t axes(a.shape().size());                                             \
    std::iota(axes.begin(), axes.end(), 0);                                    \
    return Tensor::from_primitive_one(                                         \
        std::make_shared<functor>(axes, keepdims), {a});                       \
  }                                                                            \
  Tensor op_name(const Tensor &a, axis_t axis, bool keepdims) {                \
    axes_t axes = {axis};                                                      \
    return Tensor::from_primitive_one(                                         \
        std::make_shared<functor>(axes, keepdims), {a});                       \
  }

DEFINE_REDUCE_OP(sum, Sum)
DEFINE_REDUCE_OP(max_reduce, MaxReduce)
DEFINE_REDUCE_OP(mean, Mean)

Tensor broadcast_as(const Tensor &a, const Tensor &b) {
  return broadcast_to(a, b.shape());
}

Tensor squeeze(const Tensor &a, const axes_t &axes) {
  return Tensor::from_primitive_one(std::make_shared<Squeeze>(axes), {a});
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
  return Tensor::from_primitive_one(std::make_shared<Unsqueeze>(axes), {a});
}

Tensor unsqueeze(const Tensor &a, axis_t axis) {
  return Tensor::from_primitive_one(std::make_shared<Unsqueeze>(axes_t{axis}),
                                    {a});
}

Tensor expand_dims(const Tensor &a, axis_t axis) { return unsqueeze(a, axis); }

Tensor expand_dims(const Tensor &a, const axes_t &axes) {
  return unsqueeze(a, axes);
}

Tensor permute(const Tensor &a, const axes_t &axes) {
  return Tensor::from_primitive_one(std::make_shared<Permute>(axes), {a});
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
static std::vector<Tensor> prepare_shapes_for_matmul(const Tensor &_a,
                                                     const Tensor &_b) {

  shape_t shape_a = _a.shape();
  shape_t shape_b = _b.shape();
  shape_t new_shape_a;
  shape_t new_shape_b;
  // CASE 1: vec x vec
  if (shape_a.size() == 1 && shape_b.size() == 1) {
    // [1, a] x [b, 1] -> [1, 1]. We will handle squeezing later so [1, 1] -> []
    Tensor a = unsqueeze(_a, 0); // [a] -> [1, a]
    Tensor b = unsqueeze(_b, 1); // [b] -> [b, 1]
    return {a, b};
  }
  // CASE 2: vec x mat
  if (shape_a.size() == 1 && shape_b.size() >= 2) {
    // vec x mat can be seen as [1, a] x [d1, d2, ..., a, b] where d's are batch
    // sizes (optional)
    new_shape_a = {1, shape_a[0]};
    // now, we need to try to broadcast a's shape to match the batch size of b
    for (size_t i = 0; i < shape_b.size() - 2; i++) {
      new_shape_a.insert(new_shape_a.begin(), shape_b[i]);
    }
    // this makes the op as [d1, d2, ..., 1, a] x [d1, d2, ..., a, b] -> [d1,
    // d2, ..., 1, b]
    Tensor a = broadcast_to(_a, new_shape_a);
    return {a, _b};
  }
  // CASE 3: mat x vec
  if (shape_a.size() >= 2 && shape_b.size() == 1) {
    // mat x vec can be seen as [d1, d2, ..., a, b] x [b, 1] where d's are batch
    // sizes (optional)
    new_shape_b = {shape_b[0], 1};
    // now, we need to try to broadcast b's shape to match the batch size of a
    for (size_t i = 0; i < shape_a.size() - 2; i++) {
      new_shape_b.insert(new_shape_b.begin(), shape_a[i]);
    }
    // this makes the op as [d1, d2, ..., a, b] x [d1, d2, ..., b, 1] -> [d1,
    // d2, ..., a, 1]
    Tensor b = broadcast_to(_b, new_shape_b);
    return {_a, b};
  }
  // CASE 4: mat x mat
  if (shape_a.size() >= 2 && shape_b.size() >= 2) {
    // mat x mat can be seen as [d1, d2, ..., a, b] x [d1, d2, ..., b, c] where
    // d's are batch sizes (optional) we need to keep the last 2 dims of the
    // shapes equal, and try to broadcast the rest
    shape_t batch_shape_a = shape_a;
    shape_t batch_shape_b = shape_b;
    // remove last two dims from the shapes
    batch_shape_a.pop_back();
    batch_shape_a.pop_back();
    batch_shape_b.pop_back();
    batch_shape_b.pop_back();
    // now, we need to try to broadcast the batch shapes
    shape_t new_batch_shape =
        get_broadcasted_shapes(batch_shape_a, batch_shape_b);
    // now, we need to append the last two dims
    new_shape_a = new_batch_shape;
    new_shape_a.push_back(shape_a[shape_a.size() - 2]);
    new_shape_a.push_back(shape_a[shape_a.size() - 1]);
    new_shape_b = new_batch_shape;
    new_shape_b.push_back(shape_b[shape_b.size() - 2]);
    new_shape_b.push_back(shape_b[shape_b.size() - 1]);
    Tensor a = broadcast_to(_a, new_shape_a);
    Tensor b = broadcast_to(_b, new_shape_b);
    return {a, b};
  }
  throw std::runtime_error(
      "Invalid shapes for matmul: " + vec_to_string(shape_a) + " and " +
      vec_to_string(shape_b));
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  std::vector<Tensor> tensors = prepare_shapes_for_matmul(a, b);
  Tensor res = Tensor::from_primitive_one(std::make_shared<MatMul>(), tensors);
  // Now, we need to squeeze the result if needed
  if (a.shape().size() == 1 && b.shape().size() == 1) {
    return squeeze(res);
  }
  // On matrix-vector multiplication, we need to squeeze the result on the last
  // dim
  if (a.shape().size() == 2 && b.shape().size() == 1) {
    return squeeze(res, -1);
  }
  // On vector-matrix multiplication, we need to squeeze the result on the first
  // dim
  if (a.shape().size() == 1 && b.shape().size() == 2) {
    return squeeze(res, 0);
  }
  return res;
}

Tensor where(const Tensor &condition, const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive_one(std::make_shared<Where>(),
                                    {condition, a, b});
}

Tensor exp(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<Exp>(), {a});
}

Tensor im2col(const Tensor &a, const shape_t &kernel_shape,
              const shape_t &stride, const shape_t &padding,
              const shape_t &dilation) {
  return Tensor::from_primitive_one(
      std::make_shared<Im2Col>(kernel_shape, stride, padding, dilation), {a});
}

Tensor col2im(const Tensor &a, const shape_t &output_shape,
              const shape_t &kernel_shape, const shape_t &stride,
              const shape_t &padding, const shape_t &dilation) {
  return Tensor::from_primitive_one(
      std::make_shared<Col2Im>(output_shape, kernel_shape, stride, padding,
                               dilation),
      {a});
}

// We can allow negative shapes like (-1, 2) to mean "2" and "whatever is left"
Tensor reshape(const Tensor &a, const axes_t &shape) {
  return Tensor::from_primitive_one(std::make_shared<Reshape>(shape), {a});
}
Tensor reshape(const Tensor &a, const shape_t &shape) {
  return reshape(a, axes_t(shape.begin(), shape.end()));
}

std::vector<hl_select_t>
convert_from_select_t_to_hl_select_t(const select_t &items,
                                     const std::vector<Tensor> &t_indices) {
  std::vector<hl_select_t> _items;
  int curr_tensor_idx = 0;
  for (auto &item : items) {
    if (std::holds_alternative<SelectKeepDim>(item)) {
      _items.push_back(SelectKeepDim());
    } else if (std::holds_alternative<SelectWithSlice>(item)) {
      auto _item = std::get<SelectWithSlice>(item);
      _items.push_back(_item);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      _items.push_back(std::get<SelectWithSingleIdx>(item));
    } else if (std::holds_alternative<SelectWithTensor>(item)) {
      _items.push_back(t_indices[curr_tensor_idx]);
      curr_tensor_idx++;
    } else {
      throw std::runtime_error("[select] Invalid select item");
    }
  }
  return _items;
}

std::pair<select_t, std::vector<Tensor>>
convert_from_hl_select_t_to_select_t(const std::vector<hl_select_t> &_items) {
  select_t items;
  std::vector<Tensor> t_indices;
  for (auto &item : _items) {
    if (std::holds_alternative<SelectKeepDim>(item)) {
      items.push_back(SelectKeepDim());
    } else if (std::holds_alternative<SelectWithSlice>(item)) {
      auto _item = std::get<SelectWithSlice>(item);
      items.push_back(_item);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      items.push_back(std::get<SelectWithSingleIdx>(item));
    } else if (std::holds_alternative<Tensor>(item)) {
      t_indices.push_back(std::get<Tensor>(item));
      items.push_back(SelectWithTensor());
    } else {
      throw std::runtime_error("[select] Invalid select item");
    }
  }
  return {items, t_indices};
}

Tensor select(const Tensor &a, const std::vector<hl_select_t> &_items) {
  select_t items;
  std::vector<Tensor> t_indices;
  for (auto &item : _items) {
    if (std::holds_alternative<SelectKeepDim>(item)) {
      items.push_back(std::get<SelectKeepDim>(item));
    } else if (std::holds_alternative<SelectWithSlice>(item)) {
      auto _item = std::get<SelectWithSlice>(item);
      items.push_back(_item);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      items.push_back(std::get<SelectWithSingleIdx>(item));
    } else if (std::holds_alternative<Tensor>(item)) {
      t_indices.push_back(std::get<Tensor>(item));
      items.push_back(SelectWithTensor());
    } else {
      throw std::runtime_error("[select] Invalid select item");
    }
  }
  // now pad With SelectKeepDim until ndim == _items.size()
  while (items.size() < a.ndim()) {
    items.push_back(SelectKeepDim());
  }
  std::vector<Tensor> inputs = {a};
  inputs.insert(inputs.end(), t_indices.begin(), t_indices.end());
  return Tensor::from_primitive_one(std::make_shared<Select>(items), inputs);
}

Tensor as_contiguous(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<AsContiguous>(), {a});
}

Tensor assign_at(const Tensor &a, const Tensor &b,
                 const std::vector<hl_select_t> &indices) {
  auto [items, t_indices] = convert_from_hl_select_t_to_select_t(indices);
  while (items.size() < a.ndim()) {
    items.push_back(SelectKeepDim());
  }
  std::vector<Tensor> inputs = {a, b};
  inputs.insert(inputs.end(), t_indices.begin(), t_indices.end());
  return Tensor::from_primitive_one(std::make_shared<AssignAt>(items), inputs);
}

Tensor astype(const Tensor &a, DType dtype) {
  return Tensor::from_primitive_one(std::make_shared<AsType>(dtype), {a});
}

Tensor add_inplace(Tensor &dst, const Tensor &other) {
  Tensor added = add(dst, other);
  dst.inplace_update(added);
  return dst;
}

Tensor binomial(const double p, const shape_t &shape, const DType dtype,
                std::shared_ptr<device::Device> device) {
  return Tensor::from_primitive_one(
      std::make_shared<Binomial>(p, shape, dtype, device), {}, device);
}

Tensor bilinear_resize(const Tensor &a, const shape_t &new_shape) {
  return Tensor::from_primitive_one(std::make_shared<BilinearResize>(new_shape),
                                    {a});
}
Tensor one_hot(const Tensor &a, int num_classes) {
  return Tensor::from_primitive_one(std::make_shared<OneHotVector>(num_classes),
                                    {a});
}

Tensor to_device(const Tensor &a, std::shared_ptr<device::Device> device_to) {
  return Tensor::from_primitive_one(std::make_shared<ToDevice>(device_to), {a});
}

Tensor sin(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<Sin>(), {a});
}

Tensor cos(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<Cos>(), {a});
}

Tensor copy(const Tensor &a) {
  return Tensor::from_primitive_one(std::make_shared<Copy>(), {a});
}

} // namespace pg