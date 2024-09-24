#include "./unary.cuh"
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "utils.hpp"
#include <cmath>

namespace pg {

void Log::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  if (!a.is_dense()) {
    outputs[0].init_view(
        std::make_shared<View>(a.shape(), a.dtype(), a.device()));
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((a.numel() + blocksize.x - 1) / blocksize.x);
    auto d_strides_a =
        cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
    auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
    size_t smem_size = sizeof(stride_t) * a.ndim() + sizeof(size_t) * a.ndim();
    PG_DISPATCH_FLOATING_TYPES(a.dtype(), "log_cuda", [&]() {
      cuda::log_kernel<<<gridsize, blocksize, smem_size>>>(
          d_strides_a.get(), d_shape.get(), a.ndim(),
          a.get_casted_base_ptr<scalar_t>(),
          outputs[0].get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
  } else {
    outputs[0].init_view(
        std::make_shared<View>(a.shape(), a.strides(), a.dtype(), a.device()));
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((a.numel() + blocksize.x - 1) / blocksize.x);
    PG_DISPATCH_FLOATING_TYPES(a.dtype(), "log_cuda", [&]() {
      cuda::log_kernel_dense<<<gridsize, blocksize>>>(
          a.numel(), a.get_casted_base_ptr<scalar_t>(),
          outputs[0].get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
  }
}

void Exp::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  if (!a.is_dense()) {
    outputs[0].init_view(
        std::make_shared<View>(a.shape(), a.dtype(), device::from_str("cuda")));
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((a.numel() + blocksize.x - 1) / blocksize.x);
    auto d_strides_a =
        cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
    auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
    size_t smem_size = sizeof(stride_t) * a.ndim() + sizeof(size_t) * a.ndim();
    PG_DISPATCH_FLOATING_TYPES(a.dtype(), "exp_cuda", [&]() {
      cuda::exp_kernel<<<gridsize, blocksize, smem_size>>>(
          d_strides_a.get(), d_shape.get(), a.ndim(),
          a.get_casted_base_ptr<scalar_t>(),
          outputs[0].get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
  } else {
    outputs[0].init_view(std::make_shared<View>(
        a.shape(), a.strides(), a.dtype(), device::from_str("cuda")));
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((a.numel() + blocksize.x - 1) / blocksize.x);
    PG_DISPATCH_FLOATING_TYPES(a.dtype(), "exp_cuda", [&]() {
      cuda::exp_kernel_dense<<<gridsize, blocksize>>>(
          a.numel(), a.get_casted_base_ptr<scalar_t>(),
          outputs[0].get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
  }
}

namespace cuda {
DEF_UNARY_OP_KERNEL(copy_kernel, x, float)
DEF_UNARY_OP_KERNEL(copy_kernel, x, double)
DEF_UNARY_OP_KERNEL(copy_kernel, x, int)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x), float)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((double)x), double)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x), int)
DEF_UNARY_OP_KERNEL(log_kernel, log((float)x), float)
DEF_UNARY_OP_KERNEL(log_kernel, log((double)x), double)
DEF_UNARY_OP_KERNEL(log_kernel, log((float)x), int)

DEF_UNARY_OP_KERNEL_DENSE(exp_kernel_dense, exp((float)x), float)
DEF_UNARY_OP_KERNEL_DENSE(exp_kernel_dense, exp((double)x), double)
DEF_UNARY_OP_KERNEL_DENSE(exp_kernel_dense, exp((float)x), int)
DEF_UNARY_OP_KERNEL_DENSE(log_kernel_dense, log((float)x), float)
DEF_UNARY_OP_KERNEL_DENSE(log_kernel_dense, log((double)x), double)
DEF_UNARY_OP_KERNEL_DENSE(log_kernel_dense, log((float)x), int)
} // namespace cuda
} // namespace pg