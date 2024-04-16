#include "ad_primitives.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <vector>

// for backward and output_shapes, default is throwing an exception

namespace pg {
std::vector<shape_t>
ADPrimitive::output_shapes(const std::vector<Tensor> &inputs) {
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
  return {mul(mul(y, pow(x, sub(y, fill(y.shape(), y.dtype(), 1)))), tangent),
          mul(log(x), mul(pow(x, y), tangent))};
}

std::vector<Tensor> Sum::backward(const std::vector<Tensor> &primals,
                                  const std::vector<Tensor> &tangents,
                                  const std::vector<Tensor> &outputs) {
  // if we did not keep dims, and our output is not a scalar, we need to unsqueeze first
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
  return {broadcast_to(div(tangents[0], fill(primals[0].shape(), primals[0].dtype(), total_els_reduced)), primals[0].shape())};
}

std::vector<Tensor> Log::backward(const std::vector<Tensor>& primals,
    const std::vector<Tensor>& tangents,
    const std::vector<Tensor>& outputs) {
    return { div(tangents[0], primals[0]) };
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor>& primals,
    const std::vector<Tensor>& tangents,
    const std::vector<Tensor>& outputs) {
    if (_axes_to_reduce_in_bw.size() == 0) { // means we did not broadcast
        return { tangents[0] };
    }
    return { broadcast_to(sum(tangents[0], _axes_to_reduce_in_bw, false), primals[0].shape()) };
}
} // namespace pg
