#pragma once

#define KERNEL_PARAMS_UN                                                       \
  const size_t *in_strides, const size_t *shape, const size_t num_dims,        \
      const float *in, float *out

__global__ void copy_kernel(KERNEL_PARAMS_UN);
__global__ void exp_kernel(KERNEL_PARAMS_UN);
__global__ void log_kernel(KERNEL_PARAMS_UN);

__global__ void copy_with_out_strides_kernel(
    const size_t *in_strides, const size_t *in_shape, const size_t *out_strides,
    const size_t *out_shape, const size_t in_num_dims,
    const size_t out_num_dims, const float *in, float *out);