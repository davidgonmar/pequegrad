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
  
  throw std::runtime_error("Sum::backward not implemented");
}

std::vector<Tensor> Log::backward(const std::vector<Tensor>& primals,
    const std::vector<Tensor>& tangents,
    const std::vector<Tensor>& outputs) {
    return { div(tangents[0], primals[0]) };
}

std::vector<Tensor> BroadcastTo::backward(const std::vector<Tensor>& primals,
    const std::vector<Tensor>& tangents,
    const std::vector<Tensor>& outputs) {
    return { sum(tangents[0], _axes_to_reduce_in_bw, false) };
}
} // namespace pg
