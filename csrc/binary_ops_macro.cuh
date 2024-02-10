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
    int idx_lhs = 0;                                                            \
    int idx_rhs = 0;                                                            \
    int tmp_idx = idx;                                                            \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmp_idx % shape[d];                                               \
      idx_lhs +=                                                                \
          (dim * lhs_strides[d]); /* Convert byte stride to element stride */  \
      idx_rhs +=                                                                \
          (dim * rhs_strides[d]); /* Convert byte stride to element stride */  \
      tmp_idx /= shape[d];                                                        \
    }                                                                          \
    idx_lhs /= sizeof(float); /* Convert byte index to element index */         \
    idx_rhs /= sizeof(float); /* Convert byte index to element index */         \
    float x = lhs[idx_lhs];                                                     \
    float y = rhs[idx_rhs];                                                     \
    out[idx] = FN;                                                             \
  }