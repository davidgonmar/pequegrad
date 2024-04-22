#pragma once

#include "ad_primitives.hpp"
#include "init_primitives.hpp"
#include "tensor.hpp"

namespace pg {

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

Tensor broadcast_to(const Tensor &a, const shape_t &shape) {
  return Tensor::from_primitive(std::make_shared<BroadcastTo>(shape), {a});
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
                               vec_to_string(a) + " and " + vec_to_string(b));
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

Tensor add(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Add>(),
                                broadcast_tensors(a, b));
}

Tensor add(const Tensor &a, double b) {
  Tensor t = fill(a.shape(), a.dtype(), b, a.device());
  return add(a, t);
}

Tensor mul(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Mul>(),
                                broadcast_tensors(a, b));
}

Tensor mul(const Tensor &a, double b) {
  Tensor t = fill(a.shape(), a.dtype(), b, a.device());
  return mul(a, t);
}
Tensor sub(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Sub>(),
                                broadcast_tensors(a, b));
}

Tensor sub(const Tensor &a, double b) {
  Tensor t = fill(a.shape(), a.dtype(), b, a.device());
  return sub(a, t);
}

Tensor div(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Div>(),
                                broadcast_tensors(a, b));
}

Tensor div(const Tensor &a, double b) {
  Tensor t = fill(a.shape(), a.dtype(), b, a.device());
  return div(a, t);
}

Tensor gt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Gt>(),
                                broadcast_tensors(a, b));
}

Tensor lt(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Lt>(),
                                broadcast_tensors(a, b));
}

Tensor eq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Eq>(),
                                broadcast_tensors(a, b));
}

Tensor neq(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Neq>(),
                                broadcast_tensors(a, b));
}

Tensor pow(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Pow>(),
                                broadcast_tensors(a, b));
}

Tensor pow(const Tensor &a, double b) {
  Tensor t = fill(a.shape(), a.dtype(), b, a.device());
  return pow(a, t);
}

Tensor max(const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Max>(),
                                broadcast_tensors(a, b));
}

Tensor log(const Tensor &a) {
  return Tensor::from_primitive(std::make_shared<Log>(), {a});
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
    Tensor a = broadcast_to(a, new_shape_a);
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
    Tensor b = broadcast_to(b, new_shape_b);
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
  Tensor res = Tensor::from_primitive(std::make_shared<MatMul>(), tensors);
  // Now, we need to squeeze the result if needed
  if (a.shape().size() == 1 && b.shape().size() == 1) {
    return squeeze(res);
  }
  return res;
}

Tensor where(const Tensor &condition, const Tensor &a, const Tensor &b) {
  return Tensor::from_primitive(std::make_shared<Where>(), {condition, a, b});
}

Tensor exp(const Tensor &a) {
  return Tensor::from_primitive(std::make_shared<Exp>(), {a});
}

} // namespace pg