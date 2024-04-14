#include "ad_primitives.hpp"
#include "./binary_helpers.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace pg {
void Add::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];
  CHECK_SAME_SHAPE(a, b);
  // We need to create a view for the output tensor
  outputs[0].init_view(std::make_shared<View>(a.shape(), a.dtype()));
  cpu::dispatch_binary_op(a.shape(), a.strides(), b.strides(),
                          outputs[0].strides(), a.get_base_ptr(),
                          b.get_base_ptr(), outputs[0].get_base_ptr(),
                          a.dtype(), cpu::BinaryOpType::Add);
}

void Mul::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];
  CHECK_SAME_SHAPE(a, b);
  // We need to create a view for the output tensor
  outputs[0].init_view(std::make_shared<View>(a.shape(), a.dtype()));
  cpu::dispatch_binary_op(a.shape(), a.strides(), b.strides(),
                          outputs[0].strides(), a.get_base_ptr(),
                          b.get_base_ptr(), outputs[0].get_base_ptr(),
                          a.dtype(), cpu::BinaryOpType::Mul);
}
} // namespace pg