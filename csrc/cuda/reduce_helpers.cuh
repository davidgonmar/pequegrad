#pragma once

#include "dtype.hpp"
namespace pg {
namespace cuda {
// operation and initial accumulator value
template <typename Op, typename T>
__device__ void reduce_base_fn(const T *in, T *out, const size_t *in_strides,
                               const size_t *in_shape, const size_t n_dims,
                               const size_t red_axis) {
  // the general explanation is:
  // each idx represents one output value, so each thread will reduce to said
  // output value therefore, we'll loop accross the reduced dimension, and
  // accumulate those values in order to calculate where we take the input from,
  // we can just do 2 things:
  // 1. calculate actual index as normally (impl in cuda_tensor/cuda_utils.cuh)
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
  int idx =
      blockDim.x * blockIdx.x + threadIdx.x; // one element of the output array

  Op op;

  int total_out_elements = 1;
  for (int i = 0; i < n_dims; i++) {
    total_out_elements *= in_shape[i];
  }
  total_out_elements /= in_shape[red_axis];

  if (idx >= total_out_elements) {
    return;
  }

  int red_elements = in_shape[red_axis];

  T accum = op.initial_value();

  for (int i = 0; i < red_elements; i++) {
    int reduced_idx = idx;
    int in_idx = 0;
    for (int j = n_dims - 1; j >= 0; j--) {
      if (j == red_axis) {
        in_idx +=
            i * in_strides[j] / sizeof(T); // simply advance by 'i * stride'
      } else { // do the general algorithm to go from idx -> actual displacement
        int current_dim_idx = reduced_idx % in_shape[j];
        in_idx += current_dim_idx * in_strides[j] / sizeof(T);
        reduced_idx /= in_shape[j];
      }
    }
    T el = in[in_idx];
    accum = op.apply(accum, el);
  }
  accum = op.post_reduce(accum, red_elements);
  out[idx] = accum;
}

template <typename T> struct SumOp {
  __device__ T apply(T a, T b) { return a + b; }
  __device__ T initial_value() { return (T)0; }
  __device__ T post_reduce(T a, size_t n) { return a; }
};

template <typename T> struct MaxOp {
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
  __device__ T apply(T a, T b) { return a + b; }
  __device__ T initial_value() { return (T)0; }
  __device__ T post_reduce(T a, size_t n) { return a / n; }
};

template <typename T>
__global__ void sum_kernel(const T *in, T *out, const size_t *in_strides,
                           const size_t *in_shape, const size_t n_dims,
                           const size_t red_axis) {
  reduce_base_fn<SumOp<T>, T>(in, out, in_strides, in_shape, n_dims, red_axis);
}

template <typename T>
__global__ void max_kernel(const T *in, T *out, const size_t *in_strides,
                           const size_t *in_shape, const size_t n_dims,
                           const size_t red_axis) {
  reduce_base_fn<MaxOp<T>, T>(in, out, in_strides, in_shape, n_dims, red_axis);
}

template <typename T>
__global__ void mean_kernel(const T *in, T *out, const size_t *in_strides,
                            const size_t *in_shape, const size_t n_dims,
                            const size_t red_axis) {
  reduce_base_fn<MeanOp<T>, T>(in, out, in_strides, in_shape, n_dims, red_axis);
}

enum class ReduceKernelType { SUM, MAX, MEAN };

template <typename T>
void launch_reduce_kernel_helper(ReduceKernelType type, dim3 blocks,
                                 dim3 threads, const T *in, T *out,
                                 const size_t *in_strides,
                                 const size_t *in_shape, const size_t n_dims,
                                 const size_t red_axis) {
  if (type == ReduceKernelType::SUM) {
    sum_kernel<T>
        <<<blocks, threads>>>(in, out, in_strides, in_shape, n_dims, red_axis);
  } else if (type == ReduceKernelType::MAX) {
    max_kernel<T>
        <<<blocks, threads>>>(in, out, in_strides, in_shape, n_dims, red_axis);
  } else if (type == ReduceKernelType::MEAN) {
    mean_kernel<T>
        <<<blocks, threads>>>(in, out, in_strides, in_shape, n_dims, red_axis);
  }
}

void launch_reduce_kernel(ReduceKernelType type, DType dtype, dim3 blocks,
                          dim3 threads, const void *in, void *out,
                          const size_t *in_strides, const size_t *in_shape,
                          const size_t n_dims, const size_t red_axis);
} // namespace cuda
} // namespace pg