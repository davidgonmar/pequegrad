#pragma once

#define ADD_KERNEL_PARAMS_UN                                                   \
  const int *in_strides, const int *shape, const int num_dims,                 \
      const float *in, float *out

__global__ void CopyKernel(ADD_KERNEL_PARAMS_UN);
__global__ void ExpKernel(ADD_KERNEL_PARAMS_UN);
__global__ void LogKernel(ADD_KERNEL_PARAMS_UN);