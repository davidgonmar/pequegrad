#pragma once

#define ADD_KERNEL_PARAMS_BIN                                                  \
  const int *lhs_strides, const int *rhs_strides, const int *shape,            \
      int num_dims, const float *lhs, const float *rhs, float *out

__global__ void AddKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void SubKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void MultKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void DivKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void GreaterKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void LessKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void EqualKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void NotEqualKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void GreaterEqualKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void LessEqualKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void ElementWiseMaxKernel(ADD_KERNEL_PARAMS_BIN);
__global__ void PowKernel(ADD_KERNEL_PARAMS_BIN);
