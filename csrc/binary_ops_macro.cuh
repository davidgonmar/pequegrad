#pragma once

#include "utils.cuh"

#define DEF_BIN_OP_KERNEL(KERNEL_NAME, FN)                                     \
  __global__ void KERNEL_NAME(                                                 \
      const size_t *lhs_strides, /* in bytes */                                \
      const size_t *rhs_strides, /* in bytes */                                \
      const size_t *shape,   /* both lhs and rhs should have equal shape, we   \
                             dont   handle broadcasting here */                \
      const size_t num_dims, /* equals len of strides and shape */             \
      const float *lhs, const float *rhs, float *out) {                        \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    /* calculate correct index based on strides */                             \
    int idx_lhs = get_idx_from_strides(shape, lhs_strides, num_dims, idx);     \
    int idx_rhs = get_idx_from_strides(shape, rhs_strides, num_dims, idx);     \
    float x = lhs[idx_lhs];                                                    \
    float y = rhs[idx_rhs];                                                    \
    out[idx] = FN;                                                             \
  }