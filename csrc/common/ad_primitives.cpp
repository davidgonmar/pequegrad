#include "ad_primitives.hpp"
#include "cpu/mem.hpp"
#include "cuda/mem.hpp"

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

void to_device_dispatch_general(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  auto inp = inputs[0];
  auto out = outputs[0];
  out.view_ptr()->allocate();
  auto in_ptr = inp.view().get_base_ptr();
  auto out_ptr = out.view().get_base_ptr();
  // TO(Device) out has the same layout as input
  if (out.device() == inp.device() &&
      out.device()->kind() == device::DeviceKind::CPU) {
    out.view_ptr()->set_ptr(inp.view().shared_ptr(), inp.nbytes());
  } else if (out.device() == inp.device() &&
             out.device()->kind() == device::DeviceKind::CUDA) {
    out.view_ptr()->set_ptr(inp.view().shared_ptr(), inp.nbytes());
  } else {
    // cpu -> cuda
    if (device::is_cuda(out.device())) {
      copy_from_cpu_to_cuda(inp.view().shared_ptr(), out.view().shared_ptr(),
                            inp.nbytes());
    } else { // cuda -> cpu
      copy_from_cuda_to_cpu(inp.view().shared_ptr(), out.view().shared_ptr(),
                            inp.nbytes());
    }
  }
}
void ToDevice::dispatch_cpu(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  to_device_dispatch_general(inputs, outputs);
}

void ToDevice::dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  to_device_dispatch_general(inputs, outputs);
}

} // namespace pg