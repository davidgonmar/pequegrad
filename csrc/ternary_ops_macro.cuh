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
    int idxF = 0;                                                              \
    int idxS = 0;                                                              \
    int idxT = 0;                                                              \
    int tmpI = idx;                                                            \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmpI % shape[d];                                               \
      idxF += (dim *                                                           \
               first_strides[d]); /* Convert byte stride to element stride */  \
      idxS += (dim *                                                           \
               second_strides[d]); /* Convert byte stride to element stride */ \
      idxT += (dim *                                                           \
               third_strides[d]); /* Convert byte stride to element stride */  \
      tmpI /= shape[d];                                                        \
    }                                                                          \
    idxF /= sizeof(float); /* Convert byte index to element index */           \
    idxS /= sizeof(float); /* Convert byte index to element index */           \
    idxT /= sizeof(float); /* Convert byte index to element index */           \
    float x = first[idxF];                                                     \
    float y = second[idxS];                                                    \
    float z = second[idxT];                                                    \
    out[idx] = FN;                                                             \
  }