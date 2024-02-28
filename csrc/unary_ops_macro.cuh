#pragma once

#include "utils.cuh"

#define DEF_UNARY_OP_KERNEL(KERNEL_NAME, FN, T)                                \
  __global__ void KERNEL_NAME(const size_t *in_strides, const size_t *shape,   \
                              const size_t num_dims, const T *in, T *out) {    \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    int in_idx = get_idx_from_strides<T>(shape, in_strides, num_dims, idx);    \
    T x = in[in_idx];                                                          \
    out[idx] = FN;                                                             \
  }