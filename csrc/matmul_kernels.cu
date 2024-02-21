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

__global__ void batched_matmul_kernel(
    const float *a, const float *b, float *out, const size_t *a_shape,
    const size_t *b_shape, // assume both have same shape length (ndim) and are
                           // contiguous on memory
    const size_t n_dims) {
  // each idx corresponds to a single element in the output matrix. So, if there
  // is no extra dimensions, each idx will correspond to one row/col
  // combination. in the example of a batched matmul, each idx would correspond
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  const int size1 = a_shape[n_dims - 2];
  const int sizemid = a_shape[n_dims - 1];
  const int size2 = b_shape[n_dims - 1];

  const int row = (idx / size2) % size1;
  const int col = idx % size2;

  int batch_idx_a = 0;
  int total_size_a = size1 * sizemid;
  int batch_idx_b = 0;
  int total_size_b = size2 * sizemid;
  int batch_idx_out = 0;
  int total_size_out = size2 * size1;

  // already 2 dims taken
  int rest = (idx / size2 / size1);

  // - 3 because 2 dims taken
  for (int i = n_dims - 2; i >= 0; i--) {
    int current_idx = rest % a_shape[i];
    batch_idx_a += total_size_a * current_idx;
    batch_idx_b += total_size_b * current_idx;
    batch_idx_out += total_size_out * current_idx;
    total_size_a = total_size_a * a_shape[i];
    total_size_b = total_size_b * a_shape[i];
    total_size_out = total_size_out * a_shape[i];
    rest = rest / a_shape[i];
  }

  int max_idx = size1 * size2;
  for (int i = 0; i < n_dims - 2; i++) {
    max_idx *= a_shape[i];
  }
  if (idx >= max_idx) {
    return;
  }

  float accum = 0;
  for (int i = 0; i < sizemid; i++) {
    accum += a[batch_idx_a + row * sizemid + i] *
             b[batch_idx_b + col +
               size2 * i]; // just standard matmul kernel taking into account
                           // d1, d2, d3... batched matmul
  }
  out[batch_idx_out + row * size2 + col] = accum;
}