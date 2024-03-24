#include "cpu_tensor.hpp"
#include "immintrin.h"
#include "unary_vectorized.hpp"
#include <cblas.h>

size_t CpuTensor::compute_nbytes(const shape_t &shape, DType dtype) const {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size * dtype_to_size(dtype);
}

void add_vectorized(float *ptr1, float *ptr2, float *result, int strx, int stry,
                    int size) {
  cblas_scopy(size, ptr1, strx, result, stry);
  cblas_saxpy(size, 1.0, ptr2, strx, result, stry);
}

CpuTensor CpuTensor::add(const CpuTensor &other) const {
  if (shape != other.shape) {
    throw std::runtime_error("Shapes do not match");
  }
  if (dtype != other.dtype) {
    throw std::runtime_error("Data types do not match");
  }
  float *ptr1 = static_cast<float *>(ptr.get());
  float *ptr2 = static_cast<float *>(other.ptr.get());

  // The strategy is: an n-dim tensor is an (n-1)-dim tensor of vectors
  // So, for each vector, we can use a vectorized operation
  float *result = new float[nbytes / sizeof(float)];

  // Vector + Vector case
  if (shape.size() == 1) {
    add_vectorized(ptr1, ptr2, result, strides[0] / sizeof(float),
                   other.strides[0] / sizeof(float), shape[0]);
  } else {
    // We need to iterate over the vectors (last dimension)
    int vecsize = shape[shape.size() - 1];
    int total_vectors = nbytes / (vecsize * sizeof(float));

    for (int i = 0; i < total_vectors; i++) {
      add_vectorized(ptr1 + i * vecsize, ptr2 + i * vecsize,
                     result + i * vecsize,
                     strides[shape.size() - 1] / sizeof(float),
                     other.strides[shape.size() - 1] / sizeof(float), vecsize);
    }
  }

  return CpuTensor(shape, strides,
                   std::shared_ptr<void>(result, [](float *p) { delete[] p; }),
                   dtype);
}

CpuTensor prepare_for_unary_op(const CpuTensor &a) {
  void *ptr1 = a.ptr.get();
  void *ptr2 = malloc(a.nbytes);

  return CpuTensor(a.shape, a.strides,
                   std::shared_ptr<void>(ptr2, [](void *p) { free(p); }),
                   a.dtype);
}
CpuTensor CpuTensor::exp() const {
  CpuTensor result = prepare_for_unary_op(*this);
  dispatch_unary_op(dtype, UnaryOp::Exp, ptr.get(), result.ptr.get(),
                    nbytes / dtype_to_size(dtype));
  return result;
}

CpuTensor CpuTensor::log() const {
  CpuTensor result = prepare_for_unary_op(*this);
  dispatch_unary_op(dtype, UnaryOp::Log, ptr.get(), result.ptr.get(),
                    nbytes / dtype_to_size(dtype));
  return result;
}
