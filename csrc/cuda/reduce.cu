#include "ad_primitives.hpp"
#include "common/view_helpers.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "reduce.cuh"
#include "shape.hpp"
#include "utils.hpp"

namespace pg {
void Sum::dispatch_cuda(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  const shape_t new_shape = this->reduced_shape_assuming_keepdims;
  View new_view = View(new_shape, a.dtype(), device::CUDA);
  PG_CHECK_RUNTIME(new_shape.size() == old_view.ndim());
  // normalize axes so they are positive
  axes_t axes = std::vector<axis_t>(_axes);
  for (int i = 0; i < axes.size(); i++) {
    axes[i] = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
  }
  auto d_strides = cuda_unique_ptr_from_host<stride_t>(
      old_view.ndim(), old_view.strides().data());
  auto d_shape =
      cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
  auto d_axes = cuda_unique_ptr_from_host(axes.size(), axes.data());
  dim3 blocksize(REDUCE_N_WARPS * REDUCE_WARP_SIZE);
  dim3 gridsize(new_view.numel());
  size_t n_dims = old_view.ndim();
  size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
  PG_DISPATCH_ALL_TYPES(a.dtype(), "sum_cuda", [&]() {
    cuda::sum_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
        old_view.get_casted_base_ptr<scalar_t>(),
        new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
        d_shape.get(), old_view.ndim(), d_axes.get(), axes.size(),
        this->_total_out_numel, this->_total_reduce_numel);
  });
  PG_CUDA_KERNEL_END;
  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(new_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}

void Mean::dispatch_cuda(const std::vector<Tensor> &inputs,
                         std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  const shape_t new_shape = this->reduced_shape_assuming_keepdims;
  PG_CHECK_RUNTIME(new_shape.size() == old_view.ndim());
  View new_view = View(new_shape, a.dtype(), device::CUDA);
  // normalize axes so they are positive
  axes_t axes = std::vector<axis_t>(_axes);
  for (int i = 0; i < axes.size(); i++) {
    axes[i] = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
  }
  auto d_strides = cuda_unique_ptr_from_host<stride_t>(
      old_view.ndim(), old_view.strides().data());
  auto d_shape =
      cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
  auto d_axes = cuda_unique_ptr_from_host(axes.size(), axes.data());
  dim3 blocksize(REDUCE_N_WARPS * REDUCE_WARP_SIZE);
  dim3 gridsize(new_view.numel());
  size_t n_dims = old_view.ndim();
  size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
  PG_DISPATCH_ALL_TYPES(a.dtype(), "mean_cuda", [&]() {
    cuda::mean_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
        old_view.get_casted_base_ptr<scalar_t>(),
        new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
        d_shape.get(), old_view.ndim(), d_axes.get(), axes.size(),
        this->_total_out_numel, this->_total_reduce_numel);
  });
  PG_CUDA_KERNEL_END;

  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(new_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}

void MaxReduce::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  const shape_t new_shape = this->reduced_shape_assuming_keepdims;
  View new_view = View(new_shape, a.dtype(), device::CUDA);

  // normalize axes so they are positive
  axes_t axes = std::vector<axis_t>(_axes);
  for (int i = 0; i < axes.size(); i++) {
    axes[i] = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
  }
  auto d_strides = cuda_unique_ptr_from_host<stride_t>(
      old_view.ndim(), old_view.strides().data());
  auto d_shape =
      cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
  auto d_axes = cuda_unique_ptr_from_host(axes.size(), axes.data());
  dim3 blocksize(REDUCE_N_WARPS * REDUCE_WARP_SIZE);
  dim3 gridsize(new_view.numel());
  size_t n_dims = old_view.ndim();
  size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
  PG_DISPATCH_ALL_TYPES(a.dtype(), "max_cuda", [&]() {
    cuda::max_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
        old_view.get_casted_base_ptr<scalar_t>(),
        new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
        d_shape.get(), old_view.ndim(), d_axes.get(), axes.size(),
        this->_total_out_numel, this->_total_reduce_numel);
  });
  PG_CUDA_KERNEL_END;

  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(new_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}
} // namespace pg