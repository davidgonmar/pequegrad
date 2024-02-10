#pragma once

#define KERNEL_PARAMS_UN                                                   \
  const int *in_strides, const int *shape, const int num_dims,                 \
      const float *in, float *out

__global__ void copy_kernel(KERNEL_PARAMS_UN);
__global__ void exp_kernel(KERNEL_PARAMS_UN);
__global__ void log_kernel(KERNEL_PARAMS_UN);