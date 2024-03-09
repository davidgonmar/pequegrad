#include "cuda_array.cuh"


CudaArray CudaArray::binop_same_dtype(const CudaArray &other,
                                      BinaryKernelType kt) const {
  if (shape != other.shape) {
    // try to broadcast, from smaller to larger
    if (shape.size() < other.shape.size()) {
      return broadcast_to(other.shape).binop_same_dtype(other, kt);
    } else if (shape.size() > other.shape.size()) {
      return binop_same_dtype(other.broadcast_to(shape), kt);
    } else {
      // we need to check the one with less product of shape, and try to
      // broadcast
      int64_t prod_shape = 1;
      int64_t prod_other_shape = 1;
      for (int i = 0; i < shape.size(); i++) {
        prod_shape *= shape[i];
        prod_other_shape *= other.shape[i];
      }
      if (prod_shape < prod_other_shape) {
        return broadcast_to(other.shape).binop_same_dtype(other, kt);
      } else {
        return binop_same_dtype(other.broadcast_to(shape), kt);
      }
    }
  }
  assert(shape == other.shape);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  // Default stride calculation
  CudaArray out(size, shape, dtype);
  size_t n_dims = shape.size();

  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_other_strides =
      cuda_unique_ptr_from_host(n_dims, other.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());

  launch_binary_kernel(kt, dtype, grid_size, block_size, d_strides.get(),
                       d_other_strides.get(), d_shape.get(), n_dims, get_base_ptr(),
                       other.get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return out;
}

CudaArray CudaArray::binop(const CudaArray &other, BinaryKernelType kt) const {
  // we need to decide on the casting, always biggest type
  return binop_same_dtype(other.astype(this->dtype), kt);
}
