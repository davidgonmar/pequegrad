#pragma once

#define DEF_BIN_OP_KERNEL(KERNEL_NAME, FN)                                     \
  __global__ void KERNEL_NAME(                                                 \
      const int *lhs_strides, /* in bytes */                                   \
      const int *rhs_strides, /* in bytes */                                   \
      const int *shape,   /* both lhs and rhs should have equal shape, we dont \
                             handle broadcasting here */                       \
      const int num_dims, /* equals len of strides and shape */                \
      const float *lhs, const float *rhs, float *out) {                        \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    int num_elements = 1;                                                      \
    for (int i = 0; i < num_dims; i++) {                                       \
      num_elements *= shape[i];                                                \
    }                                                                          \
    if (idx >= num_elements) {                                                 \
      return;                                                                  \
    }                                                                          \
    /* calculate correct index based on strides */                             \
    int idxLhs = 0;                                                            \
    int idxRhs = 0;                                                            \
    int tmpI = idx;                                                            \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmpI % shape[d];                                               \
      idxLhs +=                                                                \
          (dim * lhs_strides[d]); /* Convert byte stride to element stride */  \
      idxRhs +=                                                                \
          (dim * rhs_strides[d]); /* Convert byte stride to element stride */  \
      tmpI /= shape[d];                                                        \
    }                                                                          \
    idxLhs /= sizeof(float); /* Convert byte index to element index */         \
    idxRhs /= sizeof(float); /* Convert byte index to element index */         \
    float x = lhs[idxLhs];                                                     \
    float y = rhs[idxRhs];                                                     \
    out[idx] = FN;                                                             \
  }