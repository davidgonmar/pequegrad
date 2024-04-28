#include "cuda_tensor/cuda_utils.cuh"
#include "unary_helpers.cuh"
#include "view_helpers.cuh"
namespace pg {
namespace cuda {
namespace view {
View as_contiguous(const View &view) {
  if (view.is_contiguous()) {
    return view;
  }
  View contiguous_view = View(view.shape(), view.dtype(), view.device());
  auto d_shape =
      cuda_unique_ptr_from_host(view.shape().size(), view.shape().data());
  auto d_strides =
      cuda_unique_ptr_from_host(view.strides().size(), view.strides().data());

  dispatch_unary_kernel(
      UnaryKernelType::COPY, view.dtype(),
      dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
      dim3(DEFAULT_BLOCK_SIZE), d_strides.get(), d_shape.get(),
      view.shape().size(), view.get_base_ptr(), contiguous_view.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return contiguous_view;
}
} // namespace view
} // namespace cuda
} // namespace pg