#include "ad_primitives.hpp"
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
  const axes_t &axes = _axes;
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    size_t n_dims = old_view.ndim();
    size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
    PG_DISPATCH_ALL_TYPES(a.dtype(), "sum_cuda", [&]() {
      cuda::sum_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
          old_view.get_casted_base_ptr<scalar_t>(),
          new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
          d_shape.get(), old_view.ndim(), axis);
    });
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
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    size_t n_dims = old_view.ndim();
    size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
    PG_DISPATCH_FLOATING_TYPES(a.dtype(), "mean_cuda", [&]() {
      cuda::mean_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
          old_view.get_casted_base_ptr<scalar_t>(),
          new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
          d_shape.get(), old_view.ndim(), axis);
    });
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
  const bool keepdims = _keepdims;
  View old_view = inputs[0].view();
  View new_view;
  for (int i = 0; i < axes.size(); i++) {
    const shape_t new_shape =
        _reduce_single_shape_assuming_keepdims(old_view, axes[i]);
    new_view = View(new_shape, a.dtype(), device::CUDA);
    axis_t axis = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];
    auto d_strides = cuda_unique_ptr_from_host<stride_t>(
        old_view.ndim(), old_view.strides().data());
    auto d_shape =
        cuda_unique_ptr_from_host(old_view.ndim(), old_view.shape().data());
    dim3 blocksize(DEFAULT_BLOCK_SIZE);
    dim3 gridsize((old_view.numel() + blocksize.x - 1) / blocksize.x);
    size_t n_dims = old_view.ndim();
    size_t smem = sizeof(size_t) * n_dims + sizeof(stride_t) * n_dims;
    PG_DISPATCH_ALL_TYPES(a.dtype(), "max_cuda", [&]() {
      cuda::max_kernel<scalar_t><<<gridsize, blocksize, smem>>>(
          old_view.get_casted_base_ptr<scalar_t>(),
          new_view.get_casted_base_ptr<scalar_t>(), d_strides.get(),
          d_shape.get(), old_view.ndim(), axis);
    });
    PG_CUDA_KERNEL_END;
    old_view = new_view;
  }
  if (!keepdims) { /* squeeze the axes if keepdims is false*/
    new_view = view::squeeze(old_view, axes);
  }
  outputs[0].init_view(std::make_shared<View>(new_view));
}
} // namespace pg