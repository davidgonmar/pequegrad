#include "ternary.hpp"
#include "ad_primitives.hpp"
#include "dispatch.hpp"

namespace pg {
void Where::dispatch_cpu(const std::vector<Tensor> &inputs,
                         std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 3);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &condition = inputs[0];
  const Tensor &a = inputs[1];
  const Tensor &b = inputs[2];
  CHECK_SAME_SHAPE(a, b);
  CHECK_SAME_SHAPE(condition, a);
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::from_str("cpu")));
  PG_DISPATCH_ALL_TYPES(a.dtype(), "where_cpu", [&] {
    cpu::ternar_op_ker<scalar_t>(
        condition.get_casted_base_ptr<scalar_t>(),
        a.get_casted_base_ptr<scalar_t>(), b.get_casted_base_ptr<scalar_t>(),
        outputs[0].get_casted_base_ptr<scalar_t>(), condition.strides(),
        a.strides(), b.strides(), outputs[0].strides(), a.shape(),
        [](scalar_t a, scalar_t b, scalar_t c) { return a ? b : c; });
  });
}
} // namespace pg