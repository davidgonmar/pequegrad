#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t _err = x;                                                      \
    if (_err != cudaSuccess) {                                                 \
      std::ostringstream oss;                                                  \
      oss << "CUDA error " << _err << " on file/line " << __FILE__ << ":"      \
          << __LINE__ << " " << cudaGetErrorString(_err);                      \
    }                                                                          \
  } while (0)

#define ELEM_SIZE sizeof(float)
#define DEFAULT_BLOCK_SIZE 256

__device__ int get_idx_from_strides(const size_t *shape, const size_t *strides,
                                    const size_t num_dims, const int abs_idx);

__device__ int get_max_idx(const size_t *shape, const size_t num_dims);

template <typename T> using cuda_unique_ptr = std::unique_ptr<T, void (*)(T *)>;

template <typename T>
cuda_unique_ptr<T> cuda_unique_ptr_from_host(const size_t size,
                                             const T *host_ptr) {
  T *device_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&device_ptr, size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(device_ptr, host_ptr, size * sizeof(T),
                        cudaMemcpyHostToDevice));
  return cuda_unique_ptr<T>(device_ptr,
                            [](T *ptr) { CHECK_CUDA(cudaFree(ptr)); });
}

template <typename T, typename... Args>
void PG_CHECK_ARG(T cond, Args... args) {
  if (!cond) {
    std::ostringstream stream;
    (stream << ... << args);
    throw std::invalid_argument(stream.str());
  }
}

template <typename T, typename... Args>
void PG_CHECK_RUNTIME(T cond, Args... args) {
  if (!cond) {
    std::ostringstream stream;
    (stream << ... << args);
    throw std::runtime_error(stream.str());
  }
}