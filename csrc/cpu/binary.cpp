#include "binary.hpp"
#include "ad_primitives.hpp"
#include "dispatch.hpp"

#define DEF_BINARY_OP(NAME, OP)                                                \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    PG_CHECK_ARG(                                                              \
        inputs[0].dtype() == inputs[1].dtype(),                                \
        "Binary operation expects inputs to have the same dtype, got ",        \
        dtype_to_string(inputs[0].dtype()), " and ",                           \
        dtype_to_string(inputs[1].dtype()), " for ", #NAME);                   \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].view_ptr()->allocate();                                         \
    PG_DISPATCH_ALL_TYPES(a.dtype(), "dispatch_binary_op", [&] {               \
      cpu::_dispatch_binary_op_helper<scalar_t>(                               \
          a.shape(), a.strides(), b.strides(), outputs[0].strides(),           \
          a.get_casted_base_ptr<scalar_t>(),                                   \
          b.get_casted_base_ptr<scalar_t>(),                                   \
          outputs[0].get_casted_base_ptr<scalar_t>(), OP);                     \
    });                                                                        \
  }

namespace pg {
DEF_BINARY_OP(Add, cpu::BinaryOpType::Add)
DEF_BINARY_OP(Mul, cpu::BinaryOpType::Mul)
DEF_BINARY_OP(Sub, cpu::BinaryOpType::Sub)
DEF_BINARY_OP(Div, cpu::BinaryOpType::Div)
DEF_BINARY_OP(Gt, cpu::BinaryOpType::Gt)
DEF_BINARY_OP(Lt, cpu::BinaryOpType::Lt)
DEF_BINARY_OP(Eq, cpu::BinaryOpType::Eq)
DEF_BINARY_OP(Neq, cpu::BinaryOpType::Neq)
DEF_BINARY_OP(Ge, cpu::BinaryOpType::Ge)
DEF_BINARY_OP(Le, cpu::BinaryOpType::Le)
DEF_BINARY_OP(Pow, cpu::BinaryOpType::Pow)
DEF_BINARY_OP(Max, cpu::BinaryOpType::Max)

} // namespace pg