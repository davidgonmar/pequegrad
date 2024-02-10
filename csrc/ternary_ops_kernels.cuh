#pragma once

#define KERNEL_PARAMS_TER                                                  \
  const int *first_strides, const int *second_strides,                         \
      const int *third_strides, const int *shape, const int num_dims,          \
      const float *first, const float *second, const float *third, float *out

__global__ void where_kernel(KERNEL_PARAMS_TER);