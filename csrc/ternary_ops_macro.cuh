#pragma once

#define DEF_TERNARY_OP_KERNEL(KERNEL_NAME, FN)                                 \
  __global__ void KERNEL_NAME(                                                 \
      const int *first_strides,  /* in bytes */                                \
      const int *second_strides, /* in bytes */                                \
      const int *third_strides,  /* in bytes */                                \
      const int *shape,   /* both lhs and rhs should have equal shape, we dont \
                             handle broadcasting here */                       \
      const int num_dims, /* equals len of strides and shape */                \
      const float *first, const float *second, const float *third,             \
      float *out) {                                                            \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    int num_elements = 1;                                                      \
    for (int i = 0; i < num_dims; i++) {                                       \
      num_elements *= shape[i];                                                \
    }                                                                          \
    if (idx >= num_elements) {                                                 \
      return;                                                                  \
    }                                                                          \
    /* calculate correct index based on strides */                             \
    int idx_f = 0;                                                             \
    int idx_s = 0;                                                             \
    int idx_t = 0;                                                             \
    int tmp_i = idx;                                                           \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmp_i % shape[d];                                              \
      idx_f += (dim *                                                          \
               first_strides[d]); /* Convert byte stride to element stride */  \
      idx_s += (dim *                                                          \
               second_strides[d]); /* Convert byte stride to element stride */ \
      idx_t += (dim *                                                          \
               third_strides[d]); /* Convert byte stride to element stride */  \
      tmp_i /= shape[d];                                                       \
    }                                                                          \
    idx_f /= sizeof(float); /* Convert byte index to element index */          \
    idx_s /= sizeof(float); /* Convert byte index to element index */          \
    idx_t /= sizeof(float); /* Convert byte index to element index */          \
    float x = first[idx_f];                                                    \
    float y = second[idx_s];                                                   \
    float z = third[idx_t];                                                    \
    out[idx] = FN;                                                             \
  }
