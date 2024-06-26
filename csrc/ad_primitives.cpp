#include "ad_primitives.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <vector>

// for backward and output_shapes, default is throwing an exception

namespace pg {
std::vector<shape_t>
ADPrimitive::infer_output_shapes(const std::vector<Tensor> &inputs) {
  throw std::runtime_error("output_shapes not implemented for " + str());
}
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
  throw std::runtime_error("dispatch_cuda not implemented" + str());
}

std::vector<Tensor> Add::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {tangents[0], tangents[0]};
}

std::vector<DType>
FromNumpy::infer_output_dtypes(const std::vector<Tensor> &inputs) {
  return {_dtype};
}

std::vector<shape_t>
FromNumpy::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {_shape};
}

void FromNumpy::_dispatch_general(std::vector<Tensor> &outputs) {
  auto _ptr = device::allocate(_buffer_size * dtype_to_size(_dtype),
                               device::DeviceKind::CPU);
  std::memcpy(_ptr.get(), _data_ptr, _buffer_size * dtype_to_size(_dtype));
  Tensor arr(_buffer_size * dtype_to_size(_dtype), _shape, _strides, _ptr,
             _dtype, device::DeviceKind::CPU);
  outputs[0].init_view(std::make_shared<View>(arr.view()));
}

void FromNumpy::dispatch_cpu(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  _dispatch_general(outputs);
}

void FromNumpy::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  _dispatch_general(outputs);
  outputs[0].to_(device::DeviceKind::CUDA);
}

std::vector<shape_t>
Add::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Mul::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Sub::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Div::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Pow::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Max::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Gt::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Lt::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Eq::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Neq::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Ge::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Le::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

shape_t reduce_shape(const shape_t &shape, const axes_t &axes, bool keepdims) {
  shape_t new_shape;
  axes_t sorted_axes(axes);
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
    } else if (keepdims) {
      new_shape.push_back(1);
    }
  }
  return new_shape;
}

std::vector<shape_t>
Sum::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t>
MaxReduce::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t>
Mean::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t>
Log::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t>
Exp::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<Tensor> Exp::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  return {mul(tangents[0], exp(primals[0]))};
}

std::vector<shape_t>
BroadcastTo::infer_output_shapes(const std::vector<Tensor> &inputs) {
  // We need here to populate the _created_axis and _broadcasted_axis
  auto [broadcasted_axes, created_axes] =
      view::get_broadcasting_info(inputs[0].shape(), _shape_to);
  this->_broadcasted_axes = broadcasted_axes;
  this->_created_axes = created_axes;
  return {_shape_to};
}

std::vector<shape_t>
Where::infer_output_shapes(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 3, "Where expects 3 inputs, got ",
               inputs.size());
  return {inputs[0].shape()};
}

std::vector<DType>
Where::infer_output_dtypes(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 3, "Where expects 3 inputs, got ",
               inputs.size());
  PG_CHECK_ARG(inputs[1].dtype() == inputs[2].dtype() &&
                   inputs[0].dtype() == inputs[1].dtype(),
               "Where expects inputs to have the same dtype, got ",
               dtype_to_string(inputs[0].dtype()), ", ",
               dtype_to_string(inputs[1].dtype()), " and ",
               dtype_to_string(inputs[2].dtype()));
  return {inputs[1].dtype()};
}

std::vector<shape_t>
Squeeze::infer_output_shapes(const std::vector<Tensor> &inputs) {
  shape_t new_shape = inputs[0].shape();
  for (auto &axis : std::vector<axis_t>(_axes.rbegin(), _axes.rend())) {
    if (axis < 0) {
      axis += new_shape.size();
    }
    PG_CHECK_ARG(new_shape[axis] == 1, "cannot squeeze axis ", axis,
                 " as it is not 1, got ", new_shape[axis]);
    new_shape.erase(new_shape.begin() + axis);
  }
  return {new_shape};
}

std::vector<shape_t>
Unsqueeze::infer_output_shapes(const std::vector<Tensor> &inputs) {
  shape_t new_shape = inputs[0].shape();
  for (auto &axis : axes_t(_axes)) {
    if (axis < 0) {
      axis += new_shape.size() + 1;
    }
    new_shape.insert(new_shape.begin() + axis, 1);
  }

  return {new_shape};
}

std::vector<shape_t>
Permute::infer_output_shapes(const std::vector<Tensor> &inputs) {
  shape_t new_shape = inputs[0].shape();
  // Permute basically reorders the axes
  // First normalize them (if negative)
  shape_t new_shape_permuted(new_shape.size());
  for (size_t i = 0; i < _axes.size(); i++) {
    if (_axes[i] < 0) {
      _axes[i] += new_shape.size();
    }
    PG_CHECK_ARG(_axes[i] < new_shape.size(),
                 "Permute axis out of bounds: got axes ", vec_to_string(_axes),
                 " and shape ", vec_to_string(new_shape));
    new_shape_permuted[i] = new_shape[_axes[i]];
  }
  return {new_shape_permuted};
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
  return {mul(mul(y, pow(x, sub(y, fill(y.shape(), y.dtype(), 1, x.device())))),
              tangent),
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
    return {broadcast_to(
        div(g, fill(g.shape(), g.dtype(), total_els_reduced, g.device())),
        primals[0].shape())};
  }
  Tensor g = broadcast_to(tangents[0], primals[0].shape());
  return {broadcast_to(
      div(g, fill(g.shape(), g.dtype(), total_els_reduced, g.device())),
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
std::vector<shape_t>
MatMul::infer_output_shapes(const std::vector<Tensor> &inputs) {
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
  return {new_shape};
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

std::vector<shape_t>
Im2Col::infer_output_shapes(const std::vector<Tensor> &inputs) {
  PG_CHECK_ARG(inputs.size() == 1, "Im2Col expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(_kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               _kernel_shape.size());
  const Tensor &a = inputs[0];
  PG_CHECK_ARG(a.ndim() == 4, "Im2Col expects input to have 4 dimensions, got ",
               a.ndim());

  size_t k_h = _kernel_shape[0];
  size_t k_w = _kernel_shape[1];
  size_t stride_y = _strides[0];
  size_t stride_x = _strides[1];
  size_t dilation_y = _dilation[0];
  size_t dilation_x = _dilation[1];

  size_t batch_size = a.shape()[0];
  size_t in_channels = a.shape()[1];

  size_t x_h = a.shape()[2];
  size_t x_w = a.shape()[3];

  size_t out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, in_channels * k_h * k_w, out_h * out_w};
  return {out_shape};
}

std::vector<shape_t>
Col2Im::infer_output_shapes(const std::vector<Tensor> &inputs) {
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
  return {out_shape};
}

std::vector<Tensor> Im2Col::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  // For output shape, we need primals[0].shape()[-2:]
  shape_t out_shape = {primals[0].shape()[2], primals[0].shape()[3]};

  return {col2im(tangents[0], out_shape, _kernel_shape, _strides, _padding,
                 _dilation)};
}

std::vector<Tensor> Col2Im::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  return {im2col(tangents[0], _kernel_shape, _strides, _padding, _dilation)};
}

std::vector<shape_t>
Reshape::infer_output_shapes(const std::vector<Tensor> &inputs) {
  // we need to get -1 elements to 'infer' the shape
  shape_t new_shape = shape_t(_shape_to.size());
  size_t neg_idx = 0;
  size_t total = 1;
  for (size_t i = 0; i < _shape_to.size(); i++) {
    if (_shape_to[i] == -1) {
      neg_idx = i;
    } else {
      total *= _shape_to[i];
    }
  }
  PG_CHECK_ARG(total > 0, "Reshape: total elements must be > 0");
  size_t neg_val = inputs[0].numel() / total;
  new_shape[neg_idx] = neg_val;
  for (size_t i = 0; i < _shape_to.size(); i++) {
    if (_shape_to[i] == -1) {
      continue;
    }
    new_shape[i] = _shape_to[i];
  }
  return {new_shape};
}

std::vector<Tensor> Reshape::backward(const std::vector<Tensor> &primals,
                                      const std::vector<Tensor> &tangents,
                                      const std::vector<Tensor> &outputs) {
  return {reshape(tangents[0], primals[0].shape())};
}

std::vector<shape_t>
Select::infer_output_shapes(const std::vector<Tensor> &inputs) {
  // 1st input -> src tensor
  // 2nd..nth input -> indices
  select_t items = _items;
  shape_t orig_shape = inputs[0].shape();
  shape_t new_shape;

  size_t n_tensor_indices = 0;
  for (size_t i = 0; i < items.size(); i++) {
    if (std::holds_alternative<SelectWithSlice>(items[i])) {
      SelectWithSlice sss = std::get<SelectWithSlice>(items[i]);
      int start = sss.start;
      int stop = sss.stop;
      int step = sss.step;
      if (start < 0) {
        start += orig_shape[i];
      }
      if (stop < 0) {
        stop += orig_shape[i];
      }
      if (step < 0) {
        step += orig_shape[i];
      }
      new_shape.push_back((stop - start + step - 1) / step);
    } else if (std::holds_alternative<SelectWithTensor>(items[i])) {
      SelectWithTensor swt = std::get<SelectWithTensor>(items[i]);
      Tensor t = inputs[n_tensor_indices + 1];
      PG_CHECK_ARG(t.ndim() == 1,
                   "Select: tensor indices must be 1D, got shape ",
                   vec_to_string(t.shape()));
      new_shape.push_back(t.numel());
      n_tensor_indices++;
    } else if (std::holds_alternative<SelectWithSingleIdx>(items[i])) {
      // Deletes the dimension
    } else if (std::holds_alternative<SelectKeepDim>(items[i])) {
      new_shape.push_back(orig_shape[i]);
    }
  }

  return {new_shape};
}

std::vector<shape_t>
AsContiguous::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
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

std::vector<shape_t>
AssignAt::infer_output_shapes(const std::vector<Tensor> &inputs) {
  // Just return Dst shape
  return {inputs[0].shape()};
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

std::vector<shape_t>
AsType::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<Tensor> AsType::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {
  return {astype(tangents[0], primals[0].dtype())};
}

std::vector<DType>
AsType::infer_output_dtypes(const std::vector<Tensor> &inputs) {
  return {_dtype_to};
}

std::vector<shape_t>
Fill::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {_shape};
}

std::vector<DType>
Fill::infer_output_dtypes(const std::vector<Tensor> &inputs) {
  return {_dtype};
}

std::vector<Tensor> Fill::backward(const std::vector<Tensor> &primals,
                                   const std::vector<Tensor> &tangents,
                                   const std::vector<Tensor> &outputs) {
  return {};
}

} // namespace pg
