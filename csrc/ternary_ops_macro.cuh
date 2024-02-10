#pragma once

#include "utils.cuh"
#define DEF_TERNARY_OP_KERNEL(KERNEL_NAME, FN)                                 \
  __global__ void KERNEL_NAME(                                                 \
      const size_t *first_strides,  /* in bytes */                             \
      const size_t *second_strides, /* in bytes */                             \
      const size_t *third_strides,  /* in bytes */                             \
      const size_t *shape,   /* both lhs and rhs should have equal shape, we   \
                             dont   handle broadcasting here */                \
      const size_t num_dims, /* equals len of strides and shape */             \
      const float *first, const float *second, const float *third,             \
      float *out) {                                                            \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                           \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    /* calculate correct index based on strides */                             \
    int idx_f = get_idx_from_strides(shape, first_strides, num_dims, idx);     \
    int idx_s = get_idx_from_strides(shape, second_strides, num_dims, idx);    \
    int idx_t = get_idx_from_strides(shape, third_strides, num_dims, idx);     \
    float x = first[idx_f];                                                    \
    float y = second[idx_s];                                                   \
    float z = third[idx_t];                                                    \
    out[idx] = FN;                                                             \
  }
