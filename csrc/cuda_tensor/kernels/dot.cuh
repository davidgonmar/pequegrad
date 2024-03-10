#pragma once
#include "dtype.hpp"
// All kernels here assume contiguous (natural) memory

// Similar to
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. If
// size exceeds block size, then we need to do a reduction across blocks, kernel
// caller does that. Kernel computes accumulates vector_a * vector_b. Max
// accumulation range depends on the maximum block size of the GPU.
template <typename T>
__global__ void vector_dot_product_accum_kernel(const T *a, const T *b, T *out,
                                                size_t size) {
  // https://stackoverflow.com/a/27570775
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T *shared = reinterpret_cast<T *>(my_smem);
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

template <typename T>
__global__ void vector_outer_product_kernel(T *lhs, T *rhs, T *out, size_t lhs_size,
                                            size_t rhs_size) {
  // vector 'lhs' -> size 'lhs_size'
  // vector 'rhs' -> size 'rhs_size'
  // matrix 'out' -> shape (lhs_size, rhs_size)
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int curr_row = idx / rhs_size;
  const int curr_col = idx % rhs_size;

  if (curr_row >= lhs_size || curr_col >= rhs_size) {
    return;
  }

  out[curr_row * rhs_size + curr_col] = lhs[curr_row] * rhs[curr_col];
}

template <typename T>
__global__ void batched_matmul_kernel(
    const T *a, const T *b, T *out, const size_t *a_shape,
    const size_t *b_shape,
    const size_t n_dims) {
  // this kernel assumes correctness of the inputs + contiguous memory
  // a and b are, at minimum, 2D matrices. Both have the same number of
  // dimensions. Extra dimensions are treated as 'batch' dimensions. The kernel
  // computes the batched matmul of a and b, and stores the result in out. The

  // Each idx corresponds to a single element in the output matrix. So, if there
  // is no extra dimensions, each idx will correspond to one row, column pair.

  // In case there were 2 batch dimensions, for example: batch_1, batch_2,
  // the matrix a would be of shape (batch_1, batch_2, d1, d2), and b would be
  // of shape (batch_1, batch_2, d2, d3). The output matrix would be of shape
  // (batch_1, batch_2, d1, d3). The 'idx' variable would correspond to a
  // single element in the output matrix.
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  const int size1 = a_shape[n_dims - 2];
  const int sizemid = a_shape[n_dims - 1];
  const int size2 = b_shape[n_dims - 1];


  // First, we compute current the row and column of the output matrix.
  const int row = (idx / size2) % size1;
  const int col = idx % size2;


  // Then, we compute a 'batch offset' for each input matrix and the output
  // matrix
  int batch_offset_a = 0;
  int total_size_a = size1 * sizemid;
  int batch_offset_b = 0;
  int total_size_b = size2 * sizemid;
  int batch_offset_out = 0;
  int total_size_out = size2 * size1;

  // Since we took into consideration the last 2 dimensions,
  // we start from n_dims - 2
  int rest = (idx / size2 / size1);

  for (int i = n_dims - 2; i >= 0; i--) {
    int current_idx = rest % a_shape[i];
    batch_offset_a += total_size_a * current_idx;
    batch_offset_b += total_size_b * current_idx;
    batch_offset_out += total_size_out * current_idx;
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

  T accum = 0;
  for (int i = 0; i < sizemid; i++) {
    accum += a[batch_offset_a + row * sizemid + i] *
             b[batch_offset_b + col +
               size2 * i];
  }
  out[batch_offset_out + row * size2 + col] = accum;
}

void launch_batched_matmul_kernel(dim3 grid_size, dim3 block_size, DType dtype,
                                  const void *a, const void *b, void *out,
                                  const size_t *a_shape, const size_t *b_shape,
                                  const size_t n_dims);


void launch_vector_dot_product_accum_kernel(dim3 grid_size, dim3 block_size,
                                            size_t smem_size, DType dtype,
                                            const void *a, const void *b,
                                            void *out, size_t size);


void launch_vector_outer_product_kernel(dim3 grid_size, dim3 block_size,
                                        DType dtype, void *a, void *b,
                                        void *out, size_t m, size_t n);