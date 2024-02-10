#pragma once

#define KERNEL_PARAMS_BIN                                                  \
  const int *lhs_strides, const int *rhs_strides, const int *shape,            \
      int num_dims, const float *lhs, const float *rhs, float *out


__global__ void add_kernel(KERNEL_PARAMS_BIN);
__global__ void sub_kernel(KERNEL_PARAMS_BIN);
__global__ void mult_kernel(KERNEL_PARAMS_BIN);
__global__ void div_kernel(KERNEL_PARAMS_BIN);
__global__ void greater_kernel(KERNEL_PARAMS_BIN);
__global__ void less_kernel(KERNEL_PARAMS_BIN);
__global__ void equal_kernel(KERNEL_PARAMS_BIN);
__global__ void not_equal_kernel(KERNEL_PARAMS_BIN);
__global__ void greater_equal_kernel(KERNEL_PARAMS_BIN);
__global__ void less_equal_kernel(KERNEL_PARAMS_BIN);
__global__ void element_wise_max_kernel(KERNEL_PARAMS_BIN);
__global__ void pow_kernel(KERNEL_PARAMS_BIN);
