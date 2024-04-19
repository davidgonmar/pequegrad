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

std::vector<shape_t> Add::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Mul::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Sub::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Div::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Pow::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Max::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Gt::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Lt::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Eq::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Neq::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Ge::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> Le::infer_output_shapes(const std::vector<Tensor> &inputs) {
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
    if (std::find(sorted_axes.begin(), sorted_axes.end(), i) == sorted_axes.end()) {
      new_shape.push_back(shape[i]);
    } else if (keepdims) {
      new_shape.push_back(1);
    }
  }
  return new_shape;
}

std::vector<shape_t> Sum::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t> MaxReduce::infer_output_shapes(
    const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t> Mean::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {reduce_shape(inputs[0].shape(), _axes, _keepdims)};
}

std::vector<shape_t> Log::infer_output_shapes(const std::vector<Tensor> &inputs) {
  return {inputs[0].shape()};
}

std::vector<shape_t> BroadcastTo::infer_output_shapes(
    const std::vector<Tensor> &inputs) {
  return {_shape_to};
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
  Tensor max_val = outputs[0];
  Tensor argmax = primals[1];
  Tensor tangent = tangents[0];
  Tensor mask = eq(argmax, max_val);
  return {mul(mask, tangent)};
}

std::vector<Tensor> Mean::backward(const std::vector<Tensor> &primals,
                                   const std::vector<Tensor> &tangents,
                                   const std::vector<Tensor> &outputs) {
  long total_els_reduced = 1;
  for (auto &axis : _axes) {
    total_els_reduced *= primals[0].shape()[axis];
  }
  return {broadcast_to(
      div(tangents[0], fill(primals[0].shape(), primals[0].dtype(),
                            total_els_reduced, primals[0].device())),
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
  if (_axes_to_reduce_in_bw.size() == 0) { // means we did not broadcast
    return {tangents[0]};
  }
  return {broadcast_to(sum(tangents[0], _axes_to_reduce_in_bw, false),
                       primals[0].shape())};
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

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &primals,
                                     const std::vector<Tensor> &tangents,
                                     const std::vector<Tensor> &outputs) {

  Tensor a = primals[0];
  Tensor b = primals[1];

  if (is_mat_mat(primals)) {
    return {matmul(tangents[0], b.T()), matmul(a.T(), tangents[0])};
  } else if (is_vec_vec(primals)) {
    PG_CHECK_RUNTIME(tangents[0].ndim() == 0,
                     "[MatMul::backward] expected scalar tangent for "
                     "vector-vector matmul, got ",
                     vec_to_string(tangents[0].shape()));
    return {mul(broadcast_to(tangents[0], b.shape()), b),
            mul(broadcast_to(tangents[0], a.shape()), a)};
  }
  throw std::runtime_error(
      "MatMul::backward not implemented for the given shapes");
}
} // namespace pg
