#pragma once

#include "cuda/mem.hpp"
#include "shape.hpp"
#include "state.hpp"
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

template <typename T> using cuda_unique_ptr = std::shared_ptr<T>;

template <typename T>
cuda_unique_ptr<T> cuda_unique_ptr_from_host(const size_t size,
                                             const T *host_ptr) {
  auto sptr = allocate_cuda(size * sizeof(T));
  CHECK_CUDA(cudaMemcpyAsync(
      sptr.get(), host_ptr, size * sizeof(T), cudaMemcpyHostToDevice,
      GlobalState::getInstance()->get_cuda_stream()->get()));
  return std::static_pointer_cast<T>(sptr);
}

#define SHOULD_SYNC 0
#define PG_CUDA_KERNEL_END                                                     \
  do {                                                                         \
    if (SHOULD_SYNC == 1) {                                                    \
      CHECK_CUDA(cudaDeviceSynchronize());                                     \
    } else if (SHOULD_SYNC == 2) {                                             \
      CHECK_CUDA(cudaStreamSynchronize(                                        \
          GlobalState::getInstance()->get_cuda_stream()->get()));              \
    }                                                                          \
    CHECK_CUDA(cudaGetLastError());                                            \
                                                                               \
  } while (0)