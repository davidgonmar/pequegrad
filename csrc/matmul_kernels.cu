#include "matmul_kernels.cuh"
#include <stdio.h>

__global__ void matmul_kernel(const float *a, const float *b,
                              float *out, // assume contiguous memory
                              const size_t *a_shape, const size_t *b_shape,
                              const size_t a_ndim, const size_t b_ndim) {
  // each idx corresponds to a single element in the output matrix
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int size1 =
      a_ndim == 1 ? 1 : a_shape[0]; // handle vector * vector, vector * matrix,
                                    // matrix * vector, matrix * matrix cases
  const int sizemid = a_ndim == 1 ? a_shape[0] : a_shape[1];
  const int size2 = b_ndim == 1 ? 1 : b_shape[1];

  if (idx >= size1 * size2) {
    return;
  }

  const int row = idx / size2;
  const int col = idx % size2;

  float accum = 0;
  for (int i = 0; i < sizemid; i++) {
    accum += a[row * sizemid + i] *
             b[col + size2 * i]; // just standard matmul kernel!
  }
  out[row * size2 + col] = accum;
}