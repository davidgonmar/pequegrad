#pragma once

#include "utils.cuh"
#define DEF_TERNARY_OP_KERNEL(KERNEL_NAME, FN, T)                              \
  __global__ void KERNEL_NAME(                                                 \
      const size_t *first_strides,  /* in bytes */                             \
      const size_t *second_strides, /* in bytes */                             \
      const size_t *third_strides,  /* in bytes */                             \
      const size_t *shape,   /* both lhs and rhs should have equal shape, we   \
                             dont   handle broadcasting here */                \
      const size_t num_dims, /* equals len of strides and shape */             \
      const T *first, const T *second, const T *third, T *out) {               \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                           \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    /* calculate correct index based on strides */                             \
    int idx_f = get_idx_from_strides<T>(shape, first_strides, num_dims, idx);  \
    int idx_s = get_idx_from_strides<T>(shape, second_strides, num_dims, idx); \
    int idx_t = get_idx_from_strides<T>(shape, third_strides, num_dims, idx);  \
    T x = first[idx_f];                                                        \
    T y = second[idx_s];                                                       \
    T z = third[idx_t];                                                        \
    out[idx] = FN;                                                             \
  }
