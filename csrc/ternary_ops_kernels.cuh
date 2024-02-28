#pragma once

#define KERNEL_PARAMS_TER(T)                                                   \
  const size_t *first_strides, const size_t *second_strides,                   \
      const size_t *third_strides, const size_t *shape, const size_t num_dims, \
      const T *first, const T *second, const T *third, T *out

__global__ void where_kernel(KERNEL_PARAMS_TER(float));
__global__ void where_kernel(KERNEL_PARAMS_TER(double));
__global__ void where_kernel(KERNEL_PARAMS_TER(int));