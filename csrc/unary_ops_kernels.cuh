#pragma once

#include "utils.cuh"

#define KERNEL_PARAMS_UN(T)                                                    \
  const size_t *in_strides, const size_t *shape, const size_t num_dims,        \
      const T *in, T *out

__global__ void copy_kernel(KERNEL_PARAMS_UN(float));
__global__ void copy_kernel(KERNEL_PARAMS_UN(double));
__global__ void copy_kernel(KERNEL_PARAMS_UN(int));
__global__ void exp_kernel(KERNEL_PARAMS_UN(float));
__global__ void exp_kernel(KERNEL_PARAMS_UN(double));
__global__ void exp_kernel(KERNEL_PARAMS_UN(int));
__global__ void log_kernel(KERNEL_PARAMS_UN(float));
__global__ void log_kernel(KERNEL_PARAMS_UN(double));
__global__ void log_kernel(KERNEL_PARAMS_UN(int));

template <typename T>
__global__ void
copy_with_out_strides_kernel(const size_t *in_strides, const size_t *in_shape,
                             const size_t *out_strides, const size_t *out_shape,
                             const size_t in_num_dims,
                             const size_t out_num_dims, const T *in, T *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, in_num_dims) < idx ||
      get_max_idx(out_shape, out_num_dims) < idx)
    return;
  int in_idx = get_idx_from_strides<T>(in_shape, in_strides, in_num_dims, idx);
  int out_idx =
      get_idx_from_strides<T>(out_shape, out_strides, out_num_dims, idx);
  out[out_idx] = in[in_idx];
}

template <typename InT, typename OutT> // both have same strides and everything
__global__ void astype_kernel(const size_t *in_strides, const size_t *in_shape,
                              const size_t num_dims, const InT *in, OutT *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, num_dims) < idx)
    return;
  int in_idx = get_idx_from_strides<InT>(in_shape, in_strides, num_dims, idx);

  out[idx] = static_cast<OutT>(in[in_idx]);
}