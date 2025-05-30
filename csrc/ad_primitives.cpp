#include "ad_primitives.hpp"
#include "common/view_helpers.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <vector>

namespace pg {
std::vector<Tensor> ADPrimitive::backward(const std::vector<Tensor> &primals,
                                          const std::vector<Tensor> &tangents,
                                          const std::vector<Tensor> &outputs) {
  throw std::runtime_error("backward not implemented for " + str());
}

void ADPrimitive::dispatch_cpu(const std::vector<Tensor> &inputs,
                               std::vector<Tensor> &outputs) {
  throw std::runtime_error("dispatch_cpu not implemented for " + str());
}

void ADPrimitive::dispatch_cuda(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  throw std::runtime_error("dispatch_cuda not implemented for " + str());
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {tangents[0], tangents[0]};
}

std::vector<View> FromNumpy::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions()
              .dtype(_dtype)
              .shape(_shape)
              .offset(0)
              .strides(_strides)
              .device(_device)
              .build()};
}

void FromNumpy::_dispatch_general(std::vector<Tensor> &outputs,
                                  std::shared_ptr<device::Device> _device) {
  auto device = _device->kind();
  auto _ptr = device::allocate(_buffer_size * dtype_to_size(_dtype), device,
                               /*pinned=*/true);
  if (device == device::DeviceKind::CUDA) {
    copy_from_cpu_to_cuda(_data_ptr, _ptr,
                          _buffer_size * dtype_to_size(_dtype));

  } else {
    std::memcpy(_ptr.get(), _data_ptr, _buffer_size * dtype_to_size(_dtype));
  }
  outputs[0].view_ptr()->set_ptr(_ptr, _buffer_size * dtype_to_size(_dtype));
}

void FromNumpy::dispatch_cpu(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  _dispatch_general(outputs, device::from_str("cpu"));
}

void FromNumpy::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  _dispatch_general(outputs, device::from_str("cuda"));
}

std::vector<View> Add::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Mul::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Sub::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Div::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Pow::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Max::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Gt::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Lt::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Eq::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Neq::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Ge::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Le::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::tuple<shape_t, int, int, shape_t>
reduce_shape(const shape_t &shape, const axes_t &axes, bool keepdims) {
  shape_t new_shape;
  shape_t new_shape_assuming_keepdims;
  axes_t sorted_axes(axes);
  int total_reduced_per_out_elem = 1;
  int total_out_elems = 1;
  // substitute negative axes
  for (auto &axis : sorted_axes) {
    if (axis < 0) {
      axis += shape.size();
    }
  }
  std::sort(sorted_axes.begin(), sorted_axes.end());

  for (size_t i = 0; i < shape.size(); i++) {
    if (std::find(sorted_axes.begin(), sorted_axes.end(), i) ==
        sorted_axes.end()) {
      new_shape.push_back(shape[i]);
      new_shape_assuming_keepdims.push_back(shape[i]);
      total_out_elems *= shape[i];
    } else if (keepdims) {
      new_shape.push_back(1);
      total_reduced_per_out_elem *= shape[i];
      new_shape_assuming_keepdims.push_back(1);
    } else {
      total_reduced_per_out_elem *= shape[i];
      new_shape_assuming_keepdims.push_back(1);
    }
  }
  return {new_shape, total_out_elems, total_reduced_per_out_elem,
          new_shape_assuming_keepdims};
}

std::vector<View> Sum::precompute(const std::vector<Tensor> &inputs) {
  auto [new_shape, total_out_elems, total_reduced_per_out_elem,
        shape_assuming_keepdims] =
      reduce_shape(inputs[0].shape(), _axes, _keepdims);
  this->_total_out_numel = total_out_elems;
  this->_total_reduce_numel = total_reduced_per_out_elem;
  this->reduced_shape_assuming_keepdims = shape_assuming_keepdims;
  return {ViewOptions()
              .device(inputs[0].device())
              .dtype(inputs[0].dtype())
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> MaxReduce::precompute(const std::vector<Tensor> &inputs) {
  auto [new_shape, total_out_elems, total_reduced_per_out_elem,
        shape_assuming_keepdims] =
      reduce_shape(inputs[0].shape(), _axes, _keepdims);
  this->_total_out_numel = total_out_elems;
  this->_total_reduce_numel = total_reduced_per_out_elem;
  this->reduced_shape_assuming_keepdims = shape_assuming_keepdims;
  return {ViewOptions()
              .device(inputs[0].device())
              .dtype(inputs[0].dtype())
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> Mean::precompute(const std::vector<Tensor> &inputs) {
  auto [new_shape, total_out_elems, total_reduced_per_out_elem,
        shape_assuming_keepdims] =
      reduce_shape(inputs[0].shape(), _axes, _keepdims);
  this->_total_out_numel = total_out_elems;
  this->_total_reduce_numel = total_reduced_per_out_elem;
  this->reduced_shape_assuming_keepdims = shape_assuming_keepdims;
  return {ViewOptions()
              .device(inputs[0].device())
              .dtype(inputs[0].dtype())
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> Log::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Exp::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Sin::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<View> Cos::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<Tensor> Exp::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {mul(tangents[0], exp(primals[0]))};
}

std::vector<Tensor> Sin::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {mul(tangents[0], cos(primals[0]))};
}

std::vector<Tensor> Cos::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {mul(tangents[0], neg(sin(primals[0])))};
}

std::vector<View> BroadcastTo::precompute(const std::vector<Tensor> &inputs) {
  auto [view, broadcasted_axis, created_axes] =
      view::broadcasted_to(inputs[0].view(), _shape_to);
  this->_broadcasted_axes = broadcasted_axis;
  this->_created_axes = created_axes;
  return {ViewOptions().like(view).build()};
}

std::vector<View> Squeeze::precompute(const std::vector<Tensor> &inputs) {
  View view = view::squeeze(inputs[0].view(), _axes);
  return {ViewOptions().like(view).build()};
}

std::vector<View> Unsqueeze::precompute(const std::vector<Tensor> &inputs) {
  View view = view::unsqueeze(inputs[0].view(), _axes);
  return {ViewOptions().like(view).build()};
}

std::vector<View> Permute::precompute(const std::vector<Tensor> &inputs) {
  View view = view::permute(inputs[0].view(), _axes);
  return {ViewOptions().like(view).build()};
}

std::vector<Tensor> Mul::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {mul(tangents[0], primals[1]), mul(tangents[0], primals[0])};
}

std::vector<Tensor> Sub::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {tangents[0], neg(tangents[0])};
}

std::vector<Tensor> Div::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  Tensor tangent = tangents[0];
  Tensor x = primals[0];
  Tensor y = primals[1];
  return {div(tangent, y), div(mul(x, neg(tangent)), mul(y, y))};
}

std::vector<Tensor> Pow::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  Tensor tangent = tangents[0];
  Tensor x = primals[0];
  Tensor y = primals[1];
  return {
      mul(mul(y, pow(x, sub(y, fill({}, y.dtype(), 1, x.device())))), tangent),
      mul(log(x), mul(pow(x, y), tangent))};
}
static Tensor zeros_like(const Tensor &t) {
  return fill(t.shape(), t.dtype(), 0, t.device());
}

std::vector<Tensor> Max::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  Tensor a = primals[0];
  Tensor b = primals[1];
  Tensor mask = gt(a, b);
  return {where(mask, tangents[0], zeros_like(a)),
          where(mask, zeros_like(b), tangents[0])};
}

std::vector<Tensor> Sum::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  // if we did not keep dims, and our output is not a scalar, we need to
  // unsqueeze first
  if (!_keepdims && _axes.size() != primals[0].shape().size()) {
    return {broadcast_to(unsqueeze(tangents[0], _axes), primals[0].shape())};
  }
  return {broadcast_to(tangents[0], primals[0].shape())};
}

std::vector<Tensor> MaxReduce::backward(const std::vector<Tensor> &primals,
                                        const std::vector<Tensor> &tangents,
                                        const std::vector<Tensor> &outputs) {
  bool cond = !_keepdims && _axes.size() != primals[0].shape().size();
  // now, instead of a sum, it is max reduce
  Tensor g =
      cond ? broadcast_to(unsqueeze(tangents[0], _axes), primals[0].shape())
           : broadcast_to(tangents[0], primals[0].shape());
  Tensor a = primals[0];
  Tensor b = outputs[0];
  Tensor mask = eq(a, _keepdims ? b : unsqueeze(b, _axes));
  Tensor r = where(mask, g, zeros_like(a));
  return {r};
}

std::vector<Tensor> Mean::backward(const std::vector<Tensor> &primals,
                                   const std::vector<Tensor> &tangents,
                                   const std::vector<Tensor> &outputs) {
  long total_els_reduced = 1;
  for (auto &axis : _axes) {
    total_els_reduced *= primals[0].shape()[axis];
  }
  if (!_keepdims && _axes.size() != primals[0].shape().size()) {
    Tensor g = broadcast_to(unsqueeze(tangents[0], _axes), primals[0].shape());
    return {
        broadcast_to(div(g, fill({}, g.dtype(), total_els_reduced, g.device())),
                     primals[0].shape())};
  }
  Tensor g = broadcast_to(tangents[0], primals[0].shape());
  return {
      broadcast_to(div(g, fill({}, g.dtype(), total_els_reduced, g.device())),
                   primals[0].shape())};
}

std::vector<Tensor> Log::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {div(tangents[0], primals[0])};
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor> &primals,
                                          const std::vector<Tensor> &tangents,
                                          const std::vector<Tensor> &outputs) {
  if (_broadcasted_axes.empty() && _created_axes.empty()) { // no broadcast
    return {tangents[0]};
  }
  Tensor t = tangents[0];
  Tensor s = _broadcasted_axes.empty()
                 ? t
                 : sum(t, _broadcasted_axes,
                       true); // sum along broadcasted axes, keeping the dims
  return {_created_axes.empty()
              ? s
              : sum(s, _created_axes, false)}; // sum along created axes,
                                               // removing the dims
}

std::vector<Tensor> Permute::backward(const std::vector<Tensor> &primals,
                                      const std::vector<Tensor> &tangents,
                                      const std::vector<Tensor> &outputs) {

  // we need to compute the 'inverse' permutation
  auto argsort = [](const axes_t &v) {
    axes_t idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
  };
  axes_t inv_permutation = argsort(_axes);

  return {permute(tangents[0], inv_permutation)};
}

static bool is_vec(const Tensor &t) { return t.shape().size() == 1; }

static bool is_mat_at_least(const Tensor &t) { return t.shape().size() >= 2; }

static bool is_vec_mat(const std::vector<Tensor> &t) {
  return is_vec(t[0]) && is_mat_at_least(t[1]);
}

static bool is_mat_vec(const std::vector<Tensor> &t) {
  return is_mat_at_least(t[0]) && is_vec(t[1]);
}

static bool is_mat_mat(const std::vector<Tensor> &t) {
  return is_mat_at_least(t[0]) && is_mat_at_least(t[1]);
}

static bool is_vec_vec(const std::vector<Tensor> &t) {
  return is_vec(t[0]) && is_vec(t[1]);
}

static shape_t get_broadcasted_shapes(const shape_t &_a, const shape_t &_b,
                                      std::string caller_info = "") {
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
      throw std::runtime_error(caller_info +
                               " -> shapes are not broadcastable: " +
                               vec_to_string(a) + " and " + vec_to_string(b));
    }
    new_shape[i] = std::max(a_dim, b_dim);
  }
  std::reverse(new_shape.begin(), new_shape.end());
  std::reverse(a.begin(), a.end());
  std::reverse(b.begin(), b.end());
  return new_shape;
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
    shape_t new_batch_shape = get_broadcasted_shapes(
        batch_shape_a, batch_shape_b, "MatMul<mat, mat>");
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

std::vector<View> MatMul::precompute(const std::vector<Tensor> &inputs) {
  std::vector<Tensor> prepared =
      prepare_shapes_for_matmul(inputs[0], inputs[1]);
  Tensor a = prepared[0];
  Tensor b = prepared[1];
  shape_t shape_a = a.shape();
  shape_t shape_b = b.shape();
  shape_t new_shape;
  // we need to do 2 checks:
  // Given two inputs [D1, D2, .., A, B1] and [D1, D2, .., B2, C], we need to
  // make sure the batch dimensions are equal (not broadcastable, that is
  // handled externally, here they should be equal) and make sure B1 == B2
  PG_CHECK_ARG(
      shape_a.size() == shape_b.size(),
      "MatMul expects inputs to have the same number of dimensions, got ",
      shape_a.size(), " and ", shape_b.size());
  for (size_t i = 0; i < shape_a.size() - 2; i++) {
    PG_CHECK_ARG(shape_a[i] == shape_b[i],
                 "MatMul expects inputs to have the same shape in the batch "
                 "dimensions, got ",
                 vec_to_string(shape_a), " and ", vec_to_string(shape_b));
    new_shape.push_back(shape_a[i]);
  }
  int M = shape_a[shape_a.size() - 2];
  int N = shape_b[shape_b.size() - 1];
  int K = shape_a[shape_a.size() - 1];
  PG_CHECK_ARG(K == shape_b[shape_b.size() - 2],
               "MatMul expects inputs to have the same shape in the inner "
               "dimensions, got ",
               vec_to_string(shape_a), " and ", vec_to_string(shape_b));
  new_shape.push_back(M);
  new_shape.push_back(N);

  return {ViewOptions()
              .dtype(inputs[0].dtype())
              .device(inputs[0].device())
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  return {matmul(tangents[0], primals[1].T()),
          matmul(primals[0].T(), tangents[0])};
}

std::vector<Tensor> Squeeze::backward(const std::vector<Tensor> &primals,
                                      const std::vector<Tensor> &tangents,
                                      const std::vector<Tensor> &outputs) {
  return {unsqueeze(tangents[0], _axes)};
}

std::vector<Tensor> Unsqueeze::backward(const std::vector<Tensor> &primals,
                                        const std::vector<Tensor> &tangents,
                                        const std::vector<Tensor> &outputs) {
  return {squeeze(tangents[0], _axes)};
}

std::vector<Tensor> Im2Col::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  // For output shape, we need primals[0].shape()[-2:]
  shape_t out_shape = {primals[0].shape()[2], primals[0].shape()[3]};

  return {col2im(tangents[0], out_shape, _kernel_shape, _strides, _padding,
                 _dilation)};
}

std::vector<View> Im2Col::precompute(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 1, "Im2Col expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(_kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               _kernel_shape.size());
  const Tensor &a = inputs[0];
  size_t k_h = _kernel_shape[0];
  size_t k_w = _kernel_shape[1];
  size_t stride_y = _strides[0];
  size_t stride_x = _strides[1];
  size_t dilation_y = _dilation[0];
  size_t dilation_x = _dilation[1];
  size_t batch_size = a.shape()[0];
  size_t in_channels = a.shape()[1];
  size_t in_h = a.shape()[2];
  size_t in_w = a.shape()[3];

  size_t out_h = (in_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (in_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, in_channels * k_h * k_w, out_h * out_w};
  return {ViewOptions()
              .device(a.device())
              .dtype(a.dtype())
              .shape(out_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> Col2Im::precompute(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 1, "Col2Im expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(_kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               _kernel_shape.size());
  const Tensor &a = inputs[0];
  size_t k_h = _kernel_shape[0];
  size_t k_w = _kernel_shape[1];
  size_t stride_y = _strides[0];
  size_t stride_x = _strides[1];
  size_t dilation_y = _dilation[0];
  size_t dilation_x = _dilation[1];
  size_t batch_size = a.shape()[0];
  size_t in_h = a.shape()[1];
  size_t in_w = a.shape()[2];

  size_t out_h = _output_shape[0];
  size_t out_w = _output_shape[1];
  size_t out_channels = a.shape()[1] / (k_h * k_w);

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, out_channels, out_h, out_w};
  return {ViewOptions()
              .device(a.device())
              .dtype(a.dtype())
              .shape(out_shape)
              .with_natural_strides()
              .build()};
}

std::vector<Tensor> Col2Im::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  return {im2col(tangents[0], _kernel_shape, _strides, _padding, _dilation)};
}

std::vector<Tensor> Reshape::backward(const std::vector<Tensor> &primals,
                                      const std::vector<Tensor> &tangents,
                                      const std::vector<Tensor> &outputs) {
  return {reshape(tangents[0], primals[0].shape())};
}

std::vector<View> Reshape::precompute(const std::vector<Tensor> &inputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  shape_t shape = inputs[0].shape();
  Tensor a = inputs[0];
  axes_t _new_shape = _shape_to;
  shape_t new_shape(_new_shape.size());
  size_t total_new = 1;

  int neg_pos = -1;
  for (size_t i = 0; i < _new_shape.size(); i++) {
    if (_new_shape[i] < 0) {
      PG_CHECK_ARG(
          neg_pos == -1,
          "Can only specify one unknown dimension (-1) for reshape, got ",
          neg_pos, " and ", i, " for shape ", vec_to_string(_new_shape));
      neg_pos = i;
    }
    new_shape[i] = _new_shape[i];
    total_new *= new_shape[i] == -1 ? 1 : new_shape[i];
  }

  size_t total_old =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (neg_pos != -1) {
    new_shape[neg_pos] = total_old / total_new;
    PG_CHECK_ARG(
        total_old % total_new == 0,
        "New shape is not compatible with old shape: ", vec_to_string(shape),
        " not compatible with ", vec_to_string(_new_shape));
  }
  total_new = total_old;

  if (a.is_contiguous()) {
    return {ViewOptions()
                .like(view::nocopy_reshape_nocheck(a.view(), new_shape))
                .build()};
  } else {
    return {ViewOptions()
                .device(a.device())
                .dtype(a.dtype())
                .shape(new_shape)
                .with_natural_strides()
                .build()};
  }
}

std::vector<View> Select::precompute(const std::vector<Tensor> &inputs) {
  shape_t new_shape;
  strides_t new_strides;
  int _offset = 0;
  bool slice_with_array = false;
  int visited_tensors = 0;
  Tensor inp = inputs[0];
  PG_CHECK_ARG(inp.ndim() == _items.size(),
               "Number of slices must match number of dimensions");
  for (int i = 0; i < _items.size(); i++) {
    select_item_t item = _items[i];
    if (std::holds_alternative<SelectWithSlice>(item)) {
      // at the moment, only positive slices are supported
      auto _item = std::get<SelectWithSlice>(item);
      int start = _item.start;
      int stop = _item.stop;
      int step = _item.step;
      PG_CHECK_ARG(start < inp.shape()[i] && stop <= inp.shape()[i],
                   "Slice out of bounds, start: " + std::to_string(start) +
                       ", end: " + std::to_string(stop) +
                       ", shape: " + std::to_string(inp.shape()[i]));
      _offset += start * inp.strides()[i];
      new_shape.push_back((stop - start + step - 1) / step);
      new_strides.push_back(inp.strides()[i] * step);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      int _item = std::get<SelectWithSingleIdx>(item).index;
      PG_CHECK_ARG(_item >= 0, "Only positive slices are supported, got: " +
                                   std::to_string(_item));
      PG_CHECK_ARG(_item < inp.shape()[i], "Slice out of bounds, index: ",
                   std::to_string(_item) +
                       ", shape: " + std::to_string(inp.shape()[i]));
      _offset += _item * inp.strides()[i];
      // but here, since we are doing something like [:, 1], we dont add
      // anything to the shape we also dont add anything to the strides
    } else if (std::holds_alternative<SelectWithTensor>(item)) {
      // this is something like [:, [1, 2, 3]], where we are indexing over the i
      // dimension with an array we cant work with memory views here, so we just
      // run through a kernel to copy the values into a new array
      slice_with_array = true;
      new_shape.push_back(inputs[visited_tensors + 1].numel());
      visited_tensors++;
    } else if (std::holds_alternative<SelectKeepDim>(item)) {
      new_shape.push_back(inp.shape()[i]);
      new_strides.push_back(inp.strides()[i]);
    }
  }
  if (slice_with_array) {
    return {ViewOptions()
                .device(inp.device())
                .dtype(inp.dtype())
                .shape(new_shape)
                .with_natural_strides()
                .build()};
  }

  return {ViewOptions()
              .dtype(inp.dtype())
              .shape(new_shape)
              .strides(new_strides)
              .offset(_offset)
              .device(inp.device())
              .build()};
}

std::vector<View> AsContiguous::precompute(const std::vector<Tensor> &inputs) {
  auto i = inputs[0];
  return {ViewOptions()
              .device(i.device())
              .dtype(i.dtype())
              .shape(i.shape())
              .with_natural_strides()
              .build()};
}

std::vector<Tensor> AsContiguous::backward(const std::vector<Tensor> &primals,
                                           const std::vector<Tensor> &tangents,
                                           const std::vector<Tensor> &outputs) {
  return {tangents[0]};
}

std::vector<Tensor> Select::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  std::vector<hl_select_t> st = convert_from_select_t_to_hl_select_t(
      _items, std::vector<Tensor>{primals.begin() + 1, primals.end()});
  Tensor g =
      fill(primals[0].shape(), primals[0].dtype(), 0, primals[0].device());
  std::vector<Tensor> gs = {assign_at(g, tangents[0], st)};
  // Now fill with 0s the rest of the tensors (indices are not differentiable)
  for (size_t i = 1; i < primals.size(); i++) {
    gs.push_back(
        fill(primals[i].shape(), primals[i].dtype(), 0, primals[i].device()));
  }
  return gs;
}

std::vector<View> AssignAt::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like_natural(inputs[0]).build()};
}

std::vector<Tensor> AssignAt::backward(const std::vector<Tensor> &primals,
                                       const std::vector<Tensor> &tangents,
                                       const std::vector<Tensor> &outputs) {
  // assign is of the form out = assign_at(dst, src, indices). dst = primals[0],
  // src = primals[1] so the gradient wrt
  std::vector<hl_select_t> st = convert_from_select_t_to_hl_select_t(
      _items, std::vector<Tensor>{primals.begin() + 2, primals.end()});
  Tensor selected_tan_src = select(tangents[0], st);
  Tensor zeros_like_src =
      fill(primals[1].shape(), primals[1].dtype(), 0, primals[1].device());
  return {assign_at(tangents[0], zeros_like_src, st), selected_tan_src};
}

std::vector<Tensor> AsType::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  return {astype(tangents[0], primals[0].dtype())};
}

std::vector<View> AsType::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions()
              .like_natural(inputs[0])
              .dtype(_dtype_to)
              .with_natural_strides()
              .build()};
}

std::vector<View> Where::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions()
              .device(inputs[1].device())
              .dtype(inputs[1].dtype())
              .shape(inputs[1].shape())
              .with_natural_strides()
              .build()};
}

std::vector<Tensor> Fill::backward(const std::vector<Tensor> &primals,
                                   const std::vector<Tensor> &tangents,
                                   const std::vector<Tensor> &outputs) {
  return {};
}

std::vector<View> Fill::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions()
              .device(_device)
              .dtype(_dtype)
              .shape(_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> Binomial::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions()
              .device(_device)
              .dtype(_dtype)
              .shape(_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View>
BilinearResize::precompute(const std::vector<Tensor> &inputs) {
  size_t new_height = _output_shape[0];
  size_t new_width = _output_shape[1];
  shape_t new_shape = {inputs[0].shape()[0], inputs[0].shape()[1], new_height,
                       new_width};
  return {ViewOptions()
              .device(inputs[0].device())
              .dtype(inputs[0].dtype())
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<View> OneHotVector::precompute(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 1, "OneHotVector expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(inputs[0].ndim() == 1, "OneHotVector expects 1D input, got ",
               inputs[0].ndim());
  shape_t new_shape = {inputs[0].shape()[0], size_t(num_classes)};
  return {ViewOptions()
              .device(inputs[0].device())
              .dtype(DType::Float32)
              .shape(new_shape)
              .with_natural_strides()
              .build()};
}

std::vector<Tensor> OneHotVector::backward(const std::vector<Tensor> &primals,
                                           const std::vector<Tensor> &tangents,
                                           const std::vector<Tensor> &outputs) {
  return {pg::sum(tangents[0], {1}, /*keepdims*/ false)};
}

/*
class Copy : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Copy)
  DEFINE_PRECOMPUTE
};*/

std::vector<View> Copy::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like(inputs[0]).build()};
}

std::vector<Tensor> Copy::backward(const std::vector<Tensor> &primals,
                                   const std::vector<Tensor> &tangents,
                                   const std::vector<Tensor> &outputs) {
  return {tangents[0]};
}

// implementations
void Copy::dispatch_cpu(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->allocate();
  std::memcpy(outputs[0].view_ptr()->get_base_ptr(),
              inputs[0].view_ptr()->get_base_ptr(), inputs[0].nbytes());
}

std::vector<View> ToDevice::precompute(const std::vector<Tensor> &inputs) {
  return {ViewOptions().like(inputs[0]).device(_device_to).build()};
}

std::vector<Tensor> ToDevice::backward(const std::vector<Tensor> &primals,
                                       const std::vector<Tensor> &tangents,
                                       const std::vector<Tensor> &outputs) {
  return {to_device(tangents[0], primals[0].device())};
}
} // namespace pg
