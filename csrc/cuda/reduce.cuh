#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include <cub/block/block_reduce.cuh>

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
template <typename Op, typename T, int THREADS_PER_BLOCK>
__device__ void reduce_base_fn(const T *in, T *out, const stride_t *_in_strides,
                               const size_t *_in_shape, const size_t n_dims,
                               const stride_t *red_axes,
                               const size_t n_red_axes) {
  // the general explanation is:
  // each idx represents one output value, so each thread will reduce to said
  // output value therefore, we'll loop accross the reduced dimension, and
  // accumulate those values in order to calculate where we take the input from,
  // we can just do 2 things:
  // 1. calculate actual index as normally (impl in cuda_utils.cuh)
  // 2. if the dim we are iterating over is the one we are reducing over,
  // we must use the 'i' value (current iteration over the dimension we are
  // reducing)

  // example: in_shape = (4, 3, 2), contiguous (strides = (6, 2, 1)), wanna
  // reduce axis=1, so output shape will be (4, 1, 2), with strides not being
  // relevant

  // therefore, idx will be valid if it is < 4 * 1 * 2 (remember, idx represents
  // 1 output value) so, we will calculate the actual index to take the input
  // from based on those strides, but since we are 'moving' accross the
  // shape[dim=1] in the loop, we will not use the idx accross that dimension,
  // but the iterator value (i).
  extern __shared__ int8_t smem[];
  // cub
  typedef cub::BlockReduce<T, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int idx = blockIdx.x; // one element of the output array

  size_t *in_shape = (size_t *)smem;
  stride_t *in_strides = (stride_t *)(smem + n_dims * sizeof(size_t));

  if (threadIdx.x < n_dims) {
    in_shape[threadIdx.x] = _in_shape[threadIdx.x];
    in_strides[threadIdx.x] = _in_strides[threadIdx.x];
  }
  __syncthreads();
  Op op;

  int total_out_elements = 1;
  int red_elements = 1;
  for (int i = 0; i < n_dims; i++) {
    if (!isin((stride_t)i, red_axes, n_red_axes)) {
      total_out_elements *= in_shape[i];
    } else {
      red_elements *= in_shape[i];
    }
  }

  if (idx >= total_out_elements) {
    return;
  }

  T accum = op.initial_value();

  for (int i = threadIdx.x; i < red_elements; i += THREADS_PER_BLOCK) {
    int reduced_idx = idx; // idx -> idxs in output
    int in_idx = 0;
    int remaining_i = i; // i -> idxs in reduced dimension
    for (int j = n_dims - 1; j >= 0; j--) {
      if (isin((stride_t)j, red_axes, n_red_axes)) { // if we are reducing
        int current_dim_idx = remaining_i % in_shape[j];
        in_idx += current_dim_idx * in_strides[j] / sizeof(T);
        remaining_i /= in_shape[j];
      } else { // do the general algorithm to go from idx -> actual displacement
        int current_dim_idx = reduced_idx % in_shape[j];
        in_idx += current_dim_idx * in_strides[j] / sizeof(T);
        reduced_idx /= in_shape[j];
      }
    }
    T el = in[in_idx];
    accum = op.apply(accum, el);
  }
  // now, we have accumulated value in 'accum' per thread
  // block reduce it
  accum = BlockReduce(temp_storage).Reduce(accum, op);
  __syncthreads();
  if (threadIdx.x == 0) {
    accum = op.post_reduce(accum, red_elements);
    out[idx] = accum;
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
                           const stride_t *red_axes, const size_t n_red_axes) {
  reduce_base_fn<SumOp<T>, T, DEFAULT_BLOCK_SIZE>(in, out, in_strides, in_shape,
                                                  n_dims, red_axes, n_red_axes);
}

template <typename T>
__global__ void max_kernel(const T *in, T *out, const stride_t *in_strides,
                           const size_t *in_shape, const size_t n_dims,
                           const stride_t *red_axes, const size_t n_red_axes) {
  reduce_base_fn<MaxOp<T>, T, DEFAULT_BLOCK_SIZE>(in, out, in_strides, in_shape,
                                                  n_dims, red_axes, n_red_axes);
}

template <typename T>
__global__ void mean_kernel(const T *in, T *out, const stride_t *in_strides,
                            const size_t *in_shape, const size_t n_dims,
                            const stride_t *red_axes, const size_t n_red_axes) {
  reduce_base_fn<MeanOp<T>, T, DEFAULT_BLOCK_SIZE>(
      in, out, in_strides, in_shape, n_dims, red_axes, n_red_axes);
}

enum class ReduceKernelType { SUM, MAX, MEAN };

} // namespace cuda
} // namespace pg