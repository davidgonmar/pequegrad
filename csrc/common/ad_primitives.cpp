#include "ad_primitives.hpp"

namespace pg {
void BroadcastTo::dispatch_cpu(const std::vector<Tensor> &inputs,
                               std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void BroadcastTo::dispatch_cuda(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Squeeze::dispatch_cpu(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Squeeze::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Unsqueeze::dispatch_cpu(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Unsqueeze::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Permute::dispatch_cpu(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

void Permute::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->set_ptr(inputs[0].view().shared_ptr(),
                                 inputs[0].nbytes());
}

} // namespace pg