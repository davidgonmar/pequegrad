#include "cuda_array.cuh"


CudaArray::CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
                     const std::shared_ptr<void> &ptr, DType dtype)
    : size(size), shape(shape), strides(strides), ptr(ptr), dtype(dtype), offset(0) {}

CudaArray::CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
                     const std::shared_ptr<void> &ptr, DType dtype, int offset)
    : size(size), shape(shape), strides(strides), ptr(ptr), dtype(dtype), offset(offset) {}
    
CudaArray::CudaArray(size_t size, shape_t shape, shape_t strides, DType dtype)
    : size(size), shape(shape), strides(strides), dtype(dtype), offset(0) {
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * dtype_to_size(dtype)));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}


CudaArray::CudaArray(size_t size, shape_t shape, DType dtype)
    : size(size), shape(shape), dtype(dtype), offset(0) {
  strides.resize(shape.size());
  // Only calculate strides if we don't have a scalar
  if (shape.size() > 0) {
    strides[shape.size() - 1] = dtype_to_size(dtype);
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * dtype_to_size(dtype)));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

CudaArray::~CudaArray() {}

CudaArray::CudaArray(const CudaArray &other)
    : size(other.size), shape(other.shape), strides(other.strides),
      ptr(other.ptr), dtype(other.dtype), offset(other.offset) {}

CudaArray &CudaArray::operator=(const CudaArray &other) {
  if (this != &other) {
    size = other.size;
    shape = other.shape;
    strides = other.strides;
    ptr = other.ptr;
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}

CudaArray::CudaArray(CudaArray &&other)
    : size(other.size), shape(std::move(other.shape)),
      strides(std::move(other.strides)), ptr(std::move(other.ptr)),
      dtype(other.dtype), offset(other.offset) {}

CudaArray &CudaArray::operator=(CudaArray &&other) {
  if (this != &other) {
    size = other.size;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    ptr = std::move(other.ptr);
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}