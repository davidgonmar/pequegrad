#include "cuda_array.cuh"


CudaArray CudaArray::ternaryop(const CudaArray &second, const CudaArray &third,
                               TernaryKernelType ker) const {
  if (second.shape != third.shape || shape != second.shape ||
      shape != third.shape) {
    size_t biggest_size =
        std::max({shape.size(), second.shape.size(), third.shape.size()});
    shape_t target_shape(biggest_size, 1);
    for (size_t i = 0; i < biggest_size; i++) {
      if (i < shape.size()) {
        target_shape[i] = shape[i];
      }
      if (i < second.shape.size()) {
        target_shape[i] = std::max(target_shape[i], second.shape[i]);
      }
      if (i < third.shape.size()) {
        target_shape[i] = std::max(target_shape[i], third.shape[i]);
      }
    }

    return broadcast_to(target_shape)
        .ternaryop(second.broadcast_to(target_shape),
                   third.broadcast_to(target_shape), ker);
  }
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));

  // Default stride calculation
  CudaArray out(size, shape, dtype);
  size_t n_dims = shape.size();
  cuda_unique_ptr<size_t> d_first_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_second_strides =
      cuda_unique_ptr_from_host(n_dims, second.strides.data());
  cuda_unique_ptr<size_t> d_third_strides =
      cuda_unique_ptr_from_host(n_dims, third.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  launch_ternary_kernel(ker, dtype, grid_size, block_size,
                        d_first_strides.get(), d_second_strides.get(),
                        d_third_strides.get(), d_shape.get(), n_dims, get_base_ptr(),
                        second.get_base_ptr(), third.get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return out;
}