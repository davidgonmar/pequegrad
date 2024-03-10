#pragma once

#include "cuda_tensor.cuh"
#include "dtype.hpp"
#include "kernels/all.cuh"
#include "utils.cuh"
#include <cmath>
#include <iostream>
#include <string>


bool CudaTensor::is_contiguous() const {
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

CudaTensor CudaTensor::astype(DType new_type) const {
  if (dtype == new_type) {
    return *this;
  }
  CudaTensor out(size, shape, new_type);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  auto &in_strides = cuda_unique_ptr_from_host(shape.size(), strides.data());
  auto &in_shape = cuda_unique_ptr_from_host(shape.size(), this->shape.data());
  launch_astype_kernel(dtype, new_type, grid_size, block_size, in_strides.get(),
                       in_shape.get(), ndim(), get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;

  return out;
}

int CudaTensor::ndim() const { return shape.size(); }


std::string CudaTensor::to_string() const {
  /*void *host = malloc(size * dtype_to_size(dtype));
  CHECK_CUDA(
      cudaMemcpy(host, get_base_ptr(), size * sizeof(T), cudaMemcpyDeviceToHost));
  */
  std::stringstream ss;
  ss << "CudaTensor<" << dtype_to_string(dtype) << ">(" << size
     << ") with shape " << vec_to_string(shape) << " and strides "
     << vec_to_string(strides);
  return ss.str();
}

CudaTensor CudaTensor::clone() const {
  CudaTensor out(size, shape, strides, dtype);
  CHECK_CUDA(cudaMemcpy(out.get_base_ptr(), get_base_ptr(), size * dtype_to_size(dtype),
                        cudaMemcpyDeviceToDevice));
  return out;
}


CudaTensor CudaTensor::as_contiguous() const {
  return is_contiguous() ? *this : elwiseop(UnaryKernelType::COPY);
}

