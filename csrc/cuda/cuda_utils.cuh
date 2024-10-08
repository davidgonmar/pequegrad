#pragma once

#include "shape.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t _err = x;                                                      \
    if (_err != cudaSuccess) {                                                 \
      std::ostringstream oss;                                                  \
      oss << "CUDA error " << _err << " on file/line " << __FILE__ << ":"      \
          << __LINE__ << " " << cudaGetErrorString(_err);                      \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define DEFAULT_BLOCK_SIZE 256

template <typename T>
__device__ int get_idx_from_strides(const size_t *shape, const size_t *strides,
                                    const size_t num_dims, const int abs_idx) {
  int tmp_idx = abs_idx;
  int idx = 0;
  for (int d = num_dims - 1; d >= 0; d--) {
    int curr_dim = tmp_idx % shape[d]; // 'how much of dimension d'
    idx += strides[d] * curr_dim;
    tmp_idx /= shape[d];
  }
  return idx / sizeof(T); // strides are in bytes
}

template <typename T>
__device__ int get_idx_from_strides(const size_t *shape,
                                    const stride_t *strides,
                                    const size_t num_dims, const int abs_idx) {
  int tmp_idx = abs_idx;
  int idx = 0;
  for (int d = num_dims - 1; d >= 0; d--) {
    int curr_dim = tmp_idx % shape[d]; // 'how much of dimension d'
    idx += strides[d] * curr_dim;
    tmp_idx /= shape[d];
  }
  return idx / sizeof(T); // strides are in bytes
}

static inline __device__ int get_max_idx(const size_t *shape,
                                         const size_t num_dims) {
  int accum = 1;
  for (int d = 0; d < num_dims; d++) {
    accum *= shape[d];
  }
  return accum;
}

template <typename T> using cuda_unique_ptr = std::unique_ptr<T, void (*)(T *)>;

template <typename T>
cuda_unique_ptr<T> cuda_unique_ptr_from_host(const size_t size,
                                             const T *host_ptr) {
  T *device_ptr = nullptr;
  CHECK_CUDA(cudaMallocAsync(&device_ptr, size * sizeof(T), 0));
  CHECK_CUDA(cudaMemcpyAsync(device_ptr, host_ptr, size * sizeof(T),
                             cudaMemcpyHostToDevice));
  return cuda_unique_ptr<T>(device_ptr,
                            [](T *ptr) { CHECK_CUDA(cudaFreeAsync(ptr, 0)); });
}

#define SHOULD_SYNC 0
#define PG_CUDA_KERNEL_END                                                     \
  do {                                                                         \
    if (SHOULD_SYNC) {                                                         \
      CHECK_CUDA(cudaDeviceSynchronize());                                     \
    }                                                                          \
    CHECK_CUDA(cudaGetLastError());                                            \
                                                                               \
  } while (0)