#include "cuda_tensor.cuh"

CudaTensor CudaTensor::elwiseop(UnaryKernelType ker) const {
  if (!is_dense() || ker == UnaryKernelType::COPY) {
    dim3 block_size(DEFAULT_BLOCK_SIZE);
    dim3 grid_size(ceil(size() / (float)DEFAULT_BLOCK_SIZE));
    size_t n_dims = shape.size();
    cuda_unique_ptr<size_t> d_strides =
        cuda_unique_ptr_from_host(n_dims, strides.data());
    cuda_unique_ptr<size_t> d_shape =
        cuda_unique_ptr_from_host(n_dims, shape.data());
    CudaTensor out(shape, dtype);

    launch_unary_kernel(ker, dtype, grid_size, block_size, d_strides.get(),
                        d_shape.get(), n_dims, get_base_ptr(),
                        out.get_base_ptr());
    PG_CUDA_KERNEL_END;
    return out;
  } else {
    dim3 block_size(DEFAULT_BLOCK_SIZE);
    dim3 grid_size(ceil(size() / (float)DEFAULT_BLOCK_SIZE));
    CudaTensor out(shape, strides, dtype);
    launch_unary_kernel_dense(ker, dtype, grid_size, block_size, size(),
                              get_base_ptr(), out.get_base_ptr());
    PG_CUDA_KERNEL_END;
    return out;
  }
}
