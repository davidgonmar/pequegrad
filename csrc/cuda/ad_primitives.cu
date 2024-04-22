#include "./binary_helpers.cuh"
#include "./reduce_helpers.cuh"
#include "./ternary_helpers.cuh"
#include "./unary_helpers.cuh"
#include "ad_primitives.hpp"
#include "cuda_tensor/cuda_utils.cuh"
#include "matmul_helpers.cuh"
#include "tensor.hpp"
#include "view_helpers.cuh"

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

void Exp::dispatch_cuda(const std::vector<Tensor> &inputs,
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
  cuda::dispatch_unary_kernel(cuda::UnaryKernelType::EXP, a.dtype(), gridsize,
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

void MatMul::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  View a = pg::cuda::view::as_contiguous(inputs[0].view());
  View b = pg::cuda::view::as_contiguous(inputs[1].view());
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
  View out_view(new_shape, a.dtype(), device::CUDA);
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_strides_b = cuda_unique_ptr_from_host(b.ndim(), b.strides().data());
  auto d_strides_out =
      cuda_unique_ptr_from_host(out_view.ndim(), out_view.strides().data());
  auto d_shape_a = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  auto d_shape_b = cuda_unique_ptr_from_host(b.ndim(), b.shape().data());
  dim3 blocksize(DEFAULT_BLOCK_SIZE);
  dim3 gridsize((out_view.numel() + blocksize.x - 1) / blocksize.x);
  cuda::launch_batched_matmul_kernel(
      gridsize, blocksize, a.dtype(), a.get_base_ptr(), b.get_base_ptr(),
      out_view.get_base_ptr(), d_shape_a.get(), d_shape_b.get(), a.ndim());
  PG_CUDA_KERNEL_END;
  outputs[0].init_view(std::make_shared<View>(out_view));
}
} // namespace pg