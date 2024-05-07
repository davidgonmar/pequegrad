#include "cuda_utils.cuh"
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
void copy(const View &src, const View &dst) {
  auto d_src_shape =
      cuda_unique_ptr_from_host(src.shape().size(), src.shape().data());
  auto d_src_strides =
      cuda_unique_ptr_from_host(src.strides().size(), src.strides().data());
  auto d_dst_shape =
      cuda_unique_ptr_from_host(dst.shape().size(), dst.shape().data());
  auto d_dst_strides =
      cuda_unique_ptr_from_host(dst.strides().size(), dst.strides().data());

  launch_copy_with_out_strides_kernel(
      src.dtype(),
      dim3((src.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
      dim3(DEFAULT_BLOCK_SIZE), d_src_strides.get(), d_src_shape.get(),
      d_dst_strides.get(), d_dst_shape.get(), src.shape().size(),
      dst.shape().size(), src.get_base_ptr(), dst.get_base_ptr());

  PG_CUDA_KERNEL_END;
}

View astype(const View &view, const DType &dtype) {
  if (view.dtype() == dtype) {
    return view;
  }
  View new_view = View(view.shape(), dtype, view.device());
  auto d_shape =
      cuda_unique_ptr_from_host(view.shape().size(), view.shape().data());
  auto d_strides =
      cuda_unique_ptr_from_host(view.strides().size(), view.strides().data());

  launch_astype_kernel(
      view.dtype(), dtype,
      dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
      dim3(DEFAULT_BLOCK_SIZE), d_strides.get(), d_shape.get(),
      view.shape().size(), view.get_base_ptr(), new_view.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return new_view;
}
} // namespace view
} // namespace cuda
} // namespace pg