#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "unary.cuh"
#include "utils.hpp"
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

  PG_DISPATCH_ALL_TYPES(view.dtype(), "as_contiguous", [&]() {
    size_t smem = sizeof(size_t) * view.ndim() + sizeof(stride_t) * view.ndim();
    cuda::copy_kernel<<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) /
                             DEFAULT_BLOCK_SIZE),
                        dim3(DEFAULT_BLOCK_SIZE), smem>>>(
        d_strides.get(), d_shape.get(), view.shape().size(),
        view.get_casted_base_ptr<scalar_t>(),
        contiguous_view.get_casted_base_ptr<scalar_t>());
  });
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

  PG_DISPATCH_ALL_TYPES(src.dtype(), "copy_with_out_strides_kernel", [&]() {
    cuda::copy_with_out_strides_kernel<scalar_t>
        <<<dim3((src.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
           dim3(DEFAULT_BLOCK_SIZE)>>>(d_src_strides.get(), d_src_shape.get(),
                                       d_dst_strides.get(), d_dst_shape.get(),
                                       src.shape().size(), dst.shape().size(),
                                       src.get_casted_base_ptr<scalar_t>(),
                                       dst.get_casted_base_ptr<scalar_t>());
  });
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
  PG_DISPATCH_ALL_TYPES_TWO_TYPES(view.dtype(), dtype, "astype", [&]() {
    cuda::astype_kernel<scalar_t1, scalar_t2>
        <<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
           dim3(DEFAULT_BLOCK_SIZE)>>>(
            d_strides.get(), d_shape.get(), view.shape().size(),
            view.get_casted_base_ptr<scalar_t1>(),
            new_view.get_casted_base_ptr<scalar_t2>());
  });
  PG_CUDA_KERNEL_END;
  return new_view;
}
} // namespace view
} // namespace cuda
} // namespace pg