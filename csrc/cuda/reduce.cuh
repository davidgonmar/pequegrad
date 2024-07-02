#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include <cub/block/block_reduce.cuh>

#define REDUCE_WARP_SIZE 32
#define REDUCE_N_WARPS 2

namespace pg {

template <typename T> __device__ bool isin(T el, const T *arr, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (arr[i] == el) {
      return true;
    }
  }
  return false;
}

namespace cuda {
// operation and initial accumulator value
template <typename Op, typename T, int WARP_SIZE, int N_WARPS = 1>
__device__ void
reduce_base_fn(const T *in, T *out, const stride_t *_in_strides,
               const size_t *_in_shape, const size_t n_dims,
               const stride_t *red_axes, const size_t n_red_axes,
               const int total_out_elements, const int red_elements) {
  extern __shared__ int8_t smem[];

  int idx = blockIdx.x;

  // Allocate shared memory for input shapes and strides
  size_t *in_shape = (size_t *)smem;
  stride_t *in_strides = (stride_t *)(smem + n_dims * sizeof(size_t));

  if (threadIdx.x < n_dims) {
    in_shape[threadIdx.x] = _in_shape[threadIdx.x];
    in_strides[threadIdx.x] = _in_strides[threadIdx.x];
  }
  __syncthreads();

  Op op;

  if (idx >= total_out_elements) {
    return;
  }

  T accum = op.initial_value();

  for (int i = threadIdx.x; i < red_elements; i += WARP_SIZE * N_WARPS) {
    int reduced_idx = idx;
    int in_idx = 0;
    int remaining_i = i;

    for (int j = n_dims - 1; j >= 0; j--) {
      size_t s = in_shape[j];
      if (isin((stride_t)j, red_axes, n_red_axes)) {
        int current_dim_idx = remaining_i % s;
        in_idx += current_dim_idx * (in_strides[j] / sizeof(T));
        remaining_i /= s;
      } else {
        int current_dim_idx = reduced_idx % s;
        in_idx += current_dim_idx * (in_strides[j] / sizeof(T));
        reduced_idx /= s;
      }
    }

    T el = in[in_idx];
    accum = op.apply(accum, el);
  }

  // reduce within warp
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    accum = op.apply(accum, __shfl_down_sync(0xffffffff, accum, offset));
  }

  // share result of each warp with other warps
  __shared__ T warp_results[N_WARPS];
  if (threadIdx.x % WARP_SIZE == 0) {
    warp_results[threadIdx.x / WARP_SIZE] = accum;
  }

  // reduce again from shared memory
  __syncthreads();

  if (threadIdx.x < N_WARPS) {
    accum = warp_results[threadIdx.x];
    for (int offset = N_WARPS / 2; offset > 0; offset /= 2) {
      accum = op.apply(accum, __shfl_down_sync(0xffffffff, accum, offset));
    }
  }

  // write result to output
  if (threadIdx.x == 0) {
    out[idx] = op.post_reduce(accum, red_elements);
  }
}

template <typename T> struct SumOp {
  __device__ T operator()(T a, T b) { return a + b; }
  __device__ T apply(T a, T b) { return a + b; }
  __device__ T initial_value() { return (T)0; }
  __device__ T post_reduce(T a, size_t n) { return a; }
};

template <typename T> struct MaxOp {
  __device__ T operator()(T a, T b) { return max(a, b); }
  __device__ T apply(T a, T b) { return max(a, b); }
  // depending on the type, we might want to use the smallest possible value
  __device__ T initial_value() {
    if (std::is_same<T, float>::value) {
      return -INFINITY;
    } else if (std::is_same<T, double>::value) {
      return -INFINITY;
    } else if (std::is_same<T, int>::value) {
      return INT_MIN;
    } else {
      return 0;
    }
  }
  __device__ T post_reduce(T a, size_t n) { return a; }
};

template <typename T> struct MeanOp {
  __device__ T operator()(T a, T b) { return a + b; }
  __device__ T apply(T a, T b) { return a + b; }
  __device__ T initial_value() { return (T)0; }
  __device__ T post_reduce(T a, size_t n) { return a / n; }
};

template <typename T>
__global__ void sum_kernel(const T *in, T *out, const stride_t *in_strides,
                           const size_t *in_shape, const size_t n_dims,
                           const stride_t *red_axes, const size_t n_red_axes,
                           const int total_out_elements,
                           const int red_elements) {
  reduce_base_fn<SumOp<T>, T, REDUCE_WARP_SIZE, REDUCE_N_WARPS>(
      in, out, in_strides, in_shape, n_dims, red_axes, n_red_axes,
      total_out_elements, red_elements);
}

template <typename T>
__global__ void max_kernel(const T *in, T *out, const stride_t *in_strides,
                           const size_t *in_shape, const size_t n_dims,
                           const stride_t *red_axes, const size_t n_red_axes,
                           const int total_out_elements,
                           const int red_elements) {
  reduce_base_fn<MaxOp<T>, T, REDUCE_WARP_SIZE, REDUCE_N_WARPS>(
      in, out, in_strides, in_shape, n_dims, red_axes, n_red_axes,
      total_out_elements, red_elements);
}

template <typename T>
__global__ void mean_kernel(const T *in, T *out, const stride_t *in_strides,
                            const size_t *in_shape, const size_t n_dims,
                            const stride_t *red_axes, const size_t n_red_axes,
                            const int total_out_elements,
                            const int red_elements) {
  reduce_base_fn<MeanOp<T>, T, REDUCE_WARP_SIZE, REDUCE_N_WARPS>(
      in, out, in_strides, in_shape, n_dims, red_axes, n_red_axes,
      total_out_elements, red_elements);
}

enum class ReduceKernelType { SUM, MAX, MEAN };

} // namespace cuda
} // namespace pg