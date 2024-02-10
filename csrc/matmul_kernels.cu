#include "matmul_kernels.cuh"

__global__ void matmul_kernel(const float *a, const float *b, float *out,
                              const int size1, const int sizemid,
                              const int size2) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = idx / size2;
  const int col = idx % size2;

  if (row >= size1 || col >= size2)
    return;
  float accum = 0;
  for (int i = 0; i < sizemid; i++) {
    accum += a[row * sizemid + i] * b[i * size2 + col];
  }
  out[row * size2 + col] = accum;
}