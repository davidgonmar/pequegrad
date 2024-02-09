#pragma once

#define DEF_UNARY_OP_KERNEL(KERNEL_NAME, FN)                                   \
  __global__ void KERNEL_NAME(const int *in_strides, const int *shape,         \
                              const int num_dims, const float *in,             \
                              float *out) {                                    \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    int totalEls = 1;                                                          \
    for (int i = 0; i < num_dims; i++) {                                       \
      totalEls *= shape[i];                                                    \
    }                                                                          \
    if (idx >= totalEls) {                                                     \
      return;                                                                  \
    }                                                                          \
                                                                               \
    int tmpIdx = idx;                                                          \
    int inIdx = 0;                                                             \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmpIdx % shape[d];                                             \
      inIdx += dim * in_strides[d];                                            \
      tmpIdx /= shape[d];                                                      \
    }                                                                          \
    inIdx /= sizeof(float);                                                    \
    float x = in[inIdx];                                                       \
    out[idx] = FN;                                                             \
  }