#include "matmul_kernels.cuh"
#include <stdio.h>
// All kernels here assume contiguous (natural) memory

// Similar to
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. If
// size exceeds block size, then we need to do a reduction across blocks, kernel
// caller does that. Kernel computes accumulates vector_a * vector_b. Max
// accumulation range depends on the maximum block size of the GPU.
__global__ void vector_dot_product_accum(const float *a, const float *b,
                                         float *out, size_t size) {
  extern __shared__ float shared[]; // declared in kernel call
  const int idx = threadIdx.x;

  // only compute product if we are within size
  // else just set to 0, since uninitialized memory is undefined
  if (blockIdx.x * blockDim.x + idx < size) {
    shared[idx] =
        a[blockIdx.x * blockDim.x + idx] * b[blockIdx.x * blockDim.x + idx];
  } else {
    shared[idx] = 0;
  }

  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (idx % (stride * 2) == 0) {
      shared[idx] += shared[idx + stride];
    }
    __syncthreads();
  }
  if (idx == 0)
    out[blockIdx.x] = shared[0];
}

__global__ void matmul_kernel(const float *a, const float *b,
                              float *out, // assume contiguous memory
                              const size_t *a_shape, const size_t *b_shape,
                              const size_t a_ndim, const size_t b_ndim) {
  // each idx corresponds to a single element in the output matrix
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int size1 =
      a_ndim == 1
          ? 1
          : a_shape[a_ndim - 2]; // handle vector * vector, vector * matrix,
                                 // matrix * vector, matrix * matrix cases
  const int sizemid = a_ndim == 1 ? a_shape[0] : a_shape[a_ndim - 1];
  const int size2 = b_ndim == 1 ? 1 : b_shape[b_ndim - 1];
  const int batch_size =
      a_ndim == 3 ? a_shape[0]
                  : 1; // 1 means not batched (or just batchsize of 1 which is
                       // really the same in terms of memory)

  if (idx >= size1 * size2 * batch_size) {
    return;
  }

  const int row = (idx / size2) % size1;
  const int col = idx % size2;
  const int batch = idx / size2 / size1;

  // batch is both for a and b
  float accum = 0;
  for (int i = 0; i < sizemid; i++) {
    accum += a[size1 * batch * sizemid + row * sizemid + i] *
             b[sizemid * size2 * batch + col +
               size2 * i]; // just standard matmul kernel!
  }
  out[size1 * size2 * batch + row * size2 + col] = accum;
}

__global__ void vector_outer_product_kernel(float *a, float *b, float *out,
                                            size_t m, size_t n) {
  // vector a -> size m
  // vector b -> size n
  // matrix out -> m x n
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  const int curr_m = idx / n;
  const int curr_n = idx % n;

  if (curr_m >= m || curr_n >= n)
    return;
  out[curr_m * n + curr_n] = a[curr_m] * b[curr_n];
}