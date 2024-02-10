#pragma once

#define DEF_UNARY_OP_KERNEL(KERNEL_NAME, FN)                                   \
  __global__ void KERNEL_NAME(const size_t *in_strides, const size_t *shape,   \
                              const size_t num_dims, const float *in,          \
                              float *out) {                                    \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    int total_els = 1;                                                         \
    for (int i = 0; i < num_dims; i++) {                                       \
      total_els *= shape[i];                                                   \
    }                                                                          \
    if (idx >= total_els) {                                                    \
      return;                                                                  \
    }                                                                          \
                                                                               \
    int tmp_idx = idx;                                                         \
    int in_idx = 0;                                                            \
    for (int d = num_dims - 1; d >= 0; d--) {                                  \
      int dim = tmp_idx % shape[d];                                            \
      in_idx += dim * in_strides[d];                                           \
      tmp_idx /= shape[d];                                                     \
    }                                                                          \
    in_idx /= sizeof(float);                                                   \
    float x = in[in_idx];                                                      \
    out[idx] = FN;                                                             \
  }