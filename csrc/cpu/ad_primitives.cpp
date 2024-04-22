#include "ad_primitives.hpp"
#include "./binary_helpers.hpp"
#include "./matmul_helpers.hpp"
#include "./reduce_helpers.hpp"
#include "./ternary_helpers.hpp"
#include "./unary_helpers.hpp"
#include "common/view_helpers.hpp"
#include "cpu/view_helpers.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#define DECL_BINARY_OP(NAME, OP)                                               \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    PG_CHECK_ARG(                                                              \
        inputs[0].dtype() == inputs[1].dtype(),                                \
        "Binary operation expects inputs to have the same dtype, got ",        \
        dtype_to_string(inputs[0].dtype()), " and ",                           \
        dtype_to_string(inputs[1].dtype()));                                   \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].init_view(                                                      \
        std::make_shared<View>(a.shape(), a.dtype(), device::CPU));            \
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

void Log::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::CPU));
  cpu::dispatch_unary_op(
      a.dtype(), cpu::UnaryOpType::Log, a.get_base_ptr(),
      outputs[0].get_base_ptr(),
      a.nbytes() / dtype_to_size(a.dtype())); // todo -- this assumes contiguity
}

void Exp::dispatch_cpu(const std::vector<Tensor> &inputs,
                       std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::CPU));
  cpu::dispatch_unary_op(
      a.dtype(), cpu::UnaryOpType::Exp, a.get_base_ptr(),
      outputs[0].get_base_ptr(),
      a.nbytes() / dtype_to_size(a.dtype())); // todo -- this assumes contiguity
}

#define DECL_REDUCE_OP(NAME, OP)                                               \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 1);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const axes_t &axes = _axes;                                                \
    if (axes.size() == 0) {                                                    \
      throw std::runtime_error("Reduce expects at least one axis");            \
    }                                                                          \
    const bool keepdims = _keepdims;                                           \
    const bool is_sum = OP == cpu::ReduceOp::Sum;                              \
    View old_view = inputs[0].view();                                          \
    View new_view;                                                             \
    for (int i = 0; i < axes.size(); i++) {                                    \
      const shape_t new_shape =                                                \
          _reduce_single_shape_assuming_keepdims(old_view, axes[i]);           \
      new_view = View(new_shape, a.dtype(), device::CPU);                      \
      axis_t axis = axes[i];                                                   \
      cpu::dispatch_reduce(old_view.get_base_ptr(), new_view.get_base_ptr(),   \
                           old_view.strides(), old_view.shape(), axis,         \
                           old_view.dtype(), OP);                              \
      old_view = new_view;                                                     \
    }                                                                          \
    if (!keepdims) { /* squeeze the axes if keepdims is false*/                \
      new_view = view::squeeze(old_view, axes);                                \
    }                                                                          \
    outputs[0].init_view(std::make_shared<View>(new_view));                    \
  }

DECL_REDUCE_OP(Sum, cpu::ReduceOp::Sum)
DECL_REDUCE_OP(MaxReduce, cpu::ReduceOp::Max)
DECL_REDUCE_OP(Mean, cpu::ReduceOp::Mean)

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
      std::make_shared<View>(a.shape(), a.dtype(), device::CPU));
  cpu::dispatch_ternary_op(condition.shape(), condition.strides(), a.strides(),
                           b.strides(), outputs[0].strides(),
                           condition.get_base_ptr(), a.get_base_ptr(),
                           b.get_base_ptr(), outputs[0].get_base_ptr(),
                           a.dtype(), cpu::TernaryOpType::Where);
}

void MatMul::dispatch_cpu(const std::vector<Tensor> &inputs,
                          std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  View a = pg::cpu::view::as_contiguous(inputs[0].view());
  View b = pg::cpu::view::as_contiguous(inputs[1].view());
  PG_CHECK_ARG(a.dtype() == b.dtype(),
               "MatMul expects inputs to have the same dtype, got ",
               dtype_to_string(a.dtype()), " and ", dtype_to_string(b.dtype()));
  // We need to do 2 checks:
  // Given two inputs [D1, D2, .., A, B1] and [D1, D2, .., B2, C], we need to
  // make sure the batch dimensions are equal (not broadcastable, that is
  // handled externally, here they should be equal) and make sure B1 == B2
  PG_CHECK_ARG(
      a.ndim() == b.ndim(),
      "MatMul expects inputs to have the same number of dimensions, got ",
      a.ndim(), " and ", b.ndim());

  shape_t new_shape;
  int B = 1;
  for (size_t i = 0; i < a.ndim() - 2; i++) {
    PG_CHECK_ARG(a.shape()[i] == b.shape()[i],
                 "MatMul expects inputs to have the same shape in the batch "
                 "dimensions, got ",
                 vec_to_string(a.shape()), " and ", vec_to_string(b.shape()));
    new_shape.push_back(a.shape()[i]);
    B *= a.shape()[i];
  }
  int M = a.shape()[a.ndim() - 2];
  int N = b.shape()[b.ndim() - 1];
  int K = a.shape()[a.ndim() - 1];
  PG_CHECK_ARG(K == b.shape()[b.ndim() - 2],
               "MatMul expects inputs to have the same shape in the inner "
               "dimensions, got ",
               vec_to_string(a.shape()), " and ", vec_to_string(b.shape()));
  new_shape.push_back(M);
  new_shape.push_back(N);
  View out_view(new_shape, a.dtype(), device::CPU);
  dispatch_contiguous_matmul_ker(a.get_base_ptr(), b.get_base_ptr(),
                                 out_view.get_base_ptr(), M, N, K, B,
                                 a.dtype());
  outputs[0].init_view(std::make_shared<View>(out_view));
}
} // namespace pg
