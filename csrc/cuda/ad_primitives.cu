#include "./binary_helpers.cuh"
#include "./reduce_helpers.cuh"
#include "./ternary_helpers.cuh"
#include "./unary_helpers.cuh"
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
DECL_BINARY_OP(Add, ADD)
DECL_BINARY_OP(Pow, POW)
DECL_BINARY_OP(Sub, SUB)
DECL_BINARY_OP(Max, ELEMENT_WISE_MAX)
DECL_BINARY_OP(Mul, MULT)
DECL_BINARY_OP(Div, DIV)
DECL_BINARY_OP(Gt, GREATER)
DECL_BINARY_OP(Lt, LESS)
DECL_BINARY_OP(Eq, EQUAL)
DECL_BINARY_OP(Neq, NOT_EQUAL)
DECL_BINARY_OP(Ge, GREATER_EQUAL)
DECL_BINARY_OP(Le, LESS_EQUAL)

void Log::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::CUDA));
  dim3 blocksize(DEFAULT_BLOCK_SIZE);
  dim3 gridsize((a.numel() + blocksize.x - 1) / blocksize.x);
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  cuda::dispatch_unary_kernel(cuda::UnaryKernelType::LOG, a.dtype(), gridsize,
                              blocksize, d_strides_a.get(), d_shape.get(),
                              a.ndim(), a.get_base_ptr(),
                              outputs[0].get_base_ptr());
  PG_CUDA_KERNEL_END;
}

void Sum::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  if (axes.size() == 0) {
    throw std::runtime_error("Reduce expects at least one axis");
  }
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    cuda::launch_reduce_kernel(cuda::ReduceKernelType::SUM, a.dtype(), gridsize,
                               blocksize, old_view.get_base_ptr(),
                               new_view.get_base_ptr(), d_strides.get(),
                               d_shape.get(), old_view.ndim(), axis);
    PG_CUDA_KERNEL_END;
    old_view = new_view;
  }
  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(old_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}

void Mean::dispatch_cuda(const std::vector<Tensor> &inputs,
                         std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  if (axes.size() == 0) {
    throw std::runtime_error("Reduce expects at least one axis");
  }
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    cuda::launch_reduce_kernel(cuda::ReduceKernelType::MEAN, a.dtype(),
                               gridsize, blocksize, old_view.get_base_ptr(),
                               new_view.get_base_ptr(), d_strides.get(),
                               d_shape.get(), old_view.ndim(), axis);
    PG_CUDA_KERNEL_END;
    old_view = new_view;
  }
  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(old_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}

void MaxReduce::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  if (axes.size() == 0) {
    throw std::runtime_error("Reduce expects at least one axis");
  }
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    cuda::launch_reduce_kernel(cuda::ReduceKernelType::MAX, a.dtype(), gridsize,
                               blocksize, old_view.get_base_ptr(),
                               new_view.get_base_ptr(), d_strides.get(),
                               d_shape.get(), old_view.ndim(), axis);
    PG_CUDA_KERNEL_END;
    old_view = new_view;
  }
  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(old_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}

void Where::dispatch_cuda(const std::vector<Tensor> &inputs,
                          std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 3);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &condition = inputs[0];
  const Tensor &a = inputs[1];
  const Tensor &b = inputs[2];
  CHECK_SAME_SHAPE(a, b);
  CHECK_SAME_SHAPE(condition, a);
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), device::CUDA));
  size_t numels = a.numel();
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_strides_b = cuda_unique_ptr_from_host(b.ndim(), b.strides().data());
  auto d_strides_condition = cuda_unique_ptr_from_host<stride_t>(
      condition.ndim(), condition.strides().data());
  auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  dim3 blocksize(DEFAULT_BLOCK_SIZE);
  dim3 gridsize((numels + blocksize.x - 1) / blocksize.x);
  cuda::launch_ternary_kernel(
      cuda::TernaryKernelType::WHERE, a.dtype(), gridsize, blocksize,
      d_strides_condition.get(), d_strides_a.get(), d_strides_b.get(),
      d_shape.get(), a.ndim(), condition.get_base_ptr(), a.get_base_ptr(),
      b.get_base_ptr(), outputs[0].get_base_ptr());
  PG_CUDA_KERNEL_END;
}
} // namespace pg