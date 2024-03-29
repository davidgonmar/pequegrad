#pragma once

#include "cuda_tensor.cuh"
#include "cuda_tensor/cuda_utils.cuh"
#include "dtype.hpp"
#include "kernels/all.cuh"
#include <cmath>
#include <iostream>
#include <string>

bool CudaTensor::is_contiguous() const {
  // if it is not dense, also not contiguous (todo -- handle this case)
  if (!is_dense()) {
    return false;
  }
  if (offset != 0) {
    return false;
  }
  if (strides.size() != shape.size()) {
    return false;
  }
  if (strides.size() == 0) { // scalar
    return true;
  }
  shape_t expected_strides(shape.size());
  expected_strides[shape.size() - 1] = dtype_to_size(dtype);
  for (int i = shape.size() - 2; i >= 0; --i) {
    expected_strides[i] = expected_strides[i + 1] * shape[i + 1];
  }
  if (expected_strides != strides) {
    return false;
  }
  return true;
}

bool CudaTensor::is_dense() const {
  // dense means that it might not be contiguous, but
  // there are no holes in the array
  // that is, the total number of elements is equal to
  // the size of the underlying storage
  size_t total_in_storage = nbytes;
  size_t total_size_in_bytes = size() * dtype_to_size(dtype);
  return total_in_storage == total_size_in_bytes;
}

CudaTensor CudaTensor::astype(DType new_type) const {
  if (dtype == new_type) {
    return *this;
  }
  CudaTensor out(shape, new_type);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size() / (float)DEFAULT_BLOCK_SIZE));
  auto &in_strides = cuda_unique_ptr_from_host(shape.size(), strides.data());
  auto &in_shape = cuda_unique_ptr_from_host(shape.size(), this->shape.data());
  launch_astype_kernel(dtype, new_type, grid_size, block_size, in_strides.get(),
                       in_shape.get(), ndim(), get_base_ptr(),
                       out.get_base_ptr());
  PG_CUDA_KERNEL_END;

  return out;
}

int CudaTensor::ndim() const { return shape.size(); }

CudaTensor CudaTensor::clone() const {
  CudaTensor out(shape, strides, dtype);
  CHECK_CUDA(cudaMemcpy(out.get_base_ptr(), get_base_ptr(),
                        size() * dtype_to_size(dtype),
                        cudaMemcpyDeviceToDevice));
  return out;
}

CudaTensor CudaTensor::as_contiguous() const {
  return is_contiguous() ? *this : elwiseop(UnaryKernelType::COPY);
}
