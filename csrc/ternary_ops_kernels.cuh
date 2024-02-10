#pragma once

#define KERNEL_PARAMS_TER                                                      \
  const size_t *first_strides, const size_t *second_strides,                   \
      const size_t *third_strides, const size_t *shape, const size_t num_dims, \
      const float *first, const float *second, const float *third, float *out

__global__ void where_kernel(KERNEL_PARAMS_TER);