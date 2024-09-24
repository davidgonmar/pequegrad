#include "unary.hpp"
#include "ad_primitives.hpp"
#include "dispatch.hpp"

namespace pg {

void Log::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::from_str("cpu")));
  PG_DISPATCH_FLOATING_TYPES(a.dtype(), "dispatch_log_kernel", [&] {
    cpu::vec_log(a.get_casted_base_ptr<scalar_t>(),
                 outputs[0].get_casted_base_ptr<scalar_t>(),
                 a.nbytes() / dtype_to_size(a.dtype()));
  });
}

void Exp::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::from_str("cpu")));
  PG_DISPATCH_FLOATING_TYPES(a.dtype(), "dispatch_exp_kernel", [&] {
    cpu::vec_exp(a.get_casted_base_ptr<scalar_t>(),
                 outputs[0].get_casted_base_ptr<scalar_t>(),
                 a.nbytes() / dtype_to_size(a.dtype()));
  });
}
} // namespace pg