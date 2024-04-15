#include "ad_primitives.hpp"
#include "./binary_helpers.hpp"
#include "tensor.hpp"
#include "utils.hpp"


#define DECL_BINARY_OP(NAME, OP)                                               \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                  \
                          std::vector<Tensor> &outputs) {                       \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                           \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].init_view(std::make_shared<View>(a.shape(), a.dtype()));        \
    cpu::dispatch_binary_op(a.shape(), a.strides(), b.strides(),               \
                            outputs[0].strides(), a.get_base_ptr(),            \
                            b.get_base_ptr(), outputs[0].get_base_ptr(),       \
                            a.dtype(), cpu::BinaryOpType::OP);                 \
  }
namespace pg {
DECL_BINARY_OP(Add, Add)
DECL_BINARY_OP(Mul, Mul)
DECL_BINARY_OP(Sub, Sub)
DECL_BINARY_OP(Div, Div)
DECL_BINARY_OP(Gt, Gt)
DECL_BINARY_OP(Lt, Lt)
DECL_BINARY_OP(Eq, Eq)
DECL_BINARY_OP(Neq, Neq)
DECL_BINARY_OP(Ge, Ge)
DECL_BINARY_OP(Le, Le)
DECL_BINARY_OP(Pow, Pow)
DECL_BINARY_OP(Max, Max)
} // namespace pg
