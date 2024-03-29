#include "cuda_tensor.cuh"

shape_t compute_natural_strides(const shape_t &shape, const DType dtype) {
  if (shape.size() == 0) {
    return shape_t(); // if scalar, return empty strides
  }
  shape_t strides(shape.size());
  size_t dtype_size = dtype_to_size(dtype);
  strides[shape.size() - 1] = dtype_size;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

CudaTensor::CudaTensor(size_t nbytes, const shape_t &shape,
                       const shape_t &strides, const std::shared_ptr<void> &ptr,
                       DType dtype)
    : nbytes(nbytes), shape(shape), strides(strides), ptr(ptr), dtype(dtype),
      offset(0) {}

CudaTensor::CudaTensor(size_t nbytes, const shape_t &shape,
                       const shape_t &strides, const std::shared_ptr<void> &ptr,
                       DType dtype, int offset)
    : nbytes(nbytes), shape(shape), strides(strides), ptr(ptr), dtype(dtype),
      offset(offset) {}

CudaTensor::CudaTensor(shape_t shape, shape_t strides, DType dtype)
    : shape(shape), strides(strides), dtype(dtype), offset(0) {
  PG_CHECK_ARG(shape.size() == strides.size(),
               "shape and strides must have the same size");
  nbytes = size() * dtype_to_size(dtype);
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, nbytes));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

CudaTensor::CudaTensor(shape_t shape, DType dtype)
    : shape(shape), dtype(dtype), offset(0) {
  strides = compute_natural_strides(shape, dtype);
  nbytes = size() * dtype_to_size(dtype);
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, nbytes));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

CudaTensor::~CudaTensor() {}

CudaTensor::CudaTensor(const CudaTensor &other)
    : nbytes(other.nbytes), shape(other.shape), strides(other.strides),
      ptr(other.ptr), dtype(other.dtype), offset(other.offset) {}

CudaTensor &CudaTensor::operator=(const CudaTensor &other) {
  if (this != &other) {
    nbytes = other.nbytes;
    shape = other.shape;
    strides = other.strides;
    ptr = other.ptr;
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}

CudaTensor::CudaTensor(CudaTensor &&other)
    : nbytes(other.nbytes), shape(std::move(other.shape)),
      strides(std::move(other.strides)), ptr(std::move(other.ptr)),
      dtype(other.dtype), offset(other.offset) {}

CudaTensor &CudaTensor::operator=(CudaTensor &&other) {
  if (this != &other) {
    nbytes = other.nbytes;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    ptr = std::move(other.ptr);
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}