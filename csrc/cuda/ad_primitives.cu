#include "./binary_helpers.cuh"
#include "ad_primitives.hpp"
#include "cuda_tensor/cuda_utils.cuh"
#include "tensor.hpp"

#define DECL_BINARY_OP(NAME, OP)                                               \
  void NAME::dispatch_cuda(const std::vector<Tensor> &inputs,                  \
                           std::vector<Tensor> &outputs) {                     \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].init_view(                                                      \
        std::make_shared<View>(a.shape(), a.dtype(), device::CUDA));           \
    size_t numels = a.numel();                                                 \
    auto d_strides_a =                                                         \
        cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());     \
    auto d_strides_b =                                                         \
        cuda_unique_ptr_from_host(b.ndim(), b.strides().data());               \
    auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());      \
    cuda::dispatch_binary_op(cuda::BinaryKernelType::OP, a.dtype(), numels,    \
                             d_strides_a.get(), d_strides_b.get(),             \
                             d_shape.get(), a.ndim(), a.get_base_ptr(),        \
                             b.get_base_ptr(), outputs[0].get_base_ptr());     \
    PG_CUDA_KERNEL_END;                                                        \
  }
namespace pg {
void Add::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];
  CHECK_SAME_SHAPE(a, b);
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::CUDA));
  size_t numels = a.numel();
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_strides_b = cuda_unique_ptr_from_host(b.ndim(), b.strides().data());
  auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  cuda::dispatch_binary_op(cuda::BinaryKernelType::ADD, a.dtype(), numels,
                           d_strides_a.get(), d_strides_b.get(), d_shape.get(),
                           a.ndim(), a.get_base_ptr(), b.get_base_ptr(),
                           outputs[0].get_base_ptr());
  PG_CUDA_KERNEL_END;
}

DECL_BINARY_OP(Sub, SUB)
DECL_BINARY_OP(Mul, MULT)
DECL_BINARY_OP(Div, DIV)
DECL_BINARY_OP(Gt, GREATER)
DECL_BINARY_OP(Lt, LESS)
DECL_BINARY_OP(Eq, EQUAL)
DECL_BINARY_OP(Neq, NOT_EQUAL)
DECL_BINARY_OP(Ge, GREATER_EQUAL)
DECL_BINARY_OP(Le, LESS_EQUAL)

} // namespace pg