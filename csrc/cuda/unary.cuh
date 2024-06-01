#pragma once

#include "common.hpp"
#include "cuda_utils.cuh"
#include "dtype.hpp"
namespace pg {
namespace cuda {

#define KERNEL_PARAMS_UNARY(T)                                                 \
  const stride_t *_in_strides, const size_t *_shape, const size_t num_dims,    \
      const T *in, T *out

__global__ void copy_kernel(KERNEL_PARAMS_UNARY(float));
__global__ void copy_kernel(KERNEL_PARAMS_UNARY(double));
__global__ void copy_kernel(KERNEL_PARAMS_UNARY(int));
__global__ void exp_kernel(KERNEL_PARAMS_UNARY(float));
__global__ void exp_kernel(KERNEL_PARAMS_UNARY(double));
__global__ void exp_kernel(KERNEL_PARAMS_UNARY(int));
__global__ void log_kernel(KERNEL_PARAMS_UNARY(float));
__global__ void log_kernel(KERNEL_PARAMS_UNARY(double));
__global__ void log_kernel(KERNEL_PARAMS_UNARY(int));

#define DEF_UNARY_OP_KERNEL(KERNEL_NAME, FN, T)                                \
  __global__ void KERNEL_NAME(KERNEL_PARAMS_UNARY(T)) {                        \
    extern __shared__ int8_t smem[];                                           \
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;                \
    size_t *shape = (size_t *)smem;                                            \
    stride_t *in_strides = (stride_t *)(smem + num_dims * sizeof(size_t));     \
    if (threadIdx.x < num_dims) {                                              \
      in_strides[threadIdx.x] = _in_strides[threadIdx.x];                      \
      shape[threadIdx.x] = _shape[threadIdx.x];                                \
    }                                                                          \
    __syncthreads();                                                           \
    for (int i = 0; i < 4; ++i) {                                              \
      int idx = base_idx + i * blockDim.x * gridDim.x;                         \
      if (get_max_idx(shape, num_dims) <= idx)                                 \
        return;                                                                \
      int in_idx = get_idx_from_strides<T>(shape, in_strides, num_dims, idx);  \
      T x = in[in_idx];                                                        \
      out[idx] = FN;                                                           \
    }                                                                          \
  }

#define KERNEL_PARAMS_UNARY_DENSE(T)                                           \
  const size_t total_size, const T *in, T *out

// In this case, we are guaranteed that both 'in' and 'out' are dense tensors,
// that is, they might not be contiguous but all memory from the start of *in
// and *out to *in + total_size and *out + total_size belongs to in and out
// respectively. They must also have the same memory layout (strides and shape)
// In this case, we can use a simpler kernel that doesn't need to compute the
// index from the strides
#define DEF_UNARY_OP_KERNEL_DENSE(KERNEL_NAME, FN, T)                          \
  __global__ void KERNEL_NAME(KERNEL_PARAMS_UNARY_DENSE(T)) {                  \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    if (total_size <= idx)                                                     \
      return;                                                                  \
    T x = in[idx];                                                             \
    out[idx] = FN;                                                             \
  }

__global__ void exp_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(float));
__global__ void exp_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(double));
__global__ void exp_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(int));
__global__ void log_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(float));
__global__ void log_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(double));
__global__ void log_kernel_dense(KERNEL_PARAMS_UNARY_DENSE(int));

enum class UnaryKernelType {
  COPY,
  EXP,
  LOG,
};

template <typename T>
__global__ void
copy_with_out_strides_kernel(const stride_t *in_strides, const size_t *in_shape,
                             const stride_t *out_strides,
                             const size_t *out_shape, const size_t in_num_dims,
                             const size_t out_num_dims, const T *in, T *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, in_num_dims) < idx ||
      get_max_idx(out_shape, out_num_dims) < idx)
    return;
  int in_idx = get_idx_from_strides<T>(in_shape, in_strides, in_num_dims, idx);
  int out_idx =
      get_idx_from_strides<T>(out_shape, out_strides, out_num_dims, idx);
  out[out_idx] = in[in_idx];
}

template <typename InT, typename OutT>
__global__ void astype_kernel(const stride_t *in_strides,
                              const size_t *in_shape, const size_t num_dims,
                              const InT *in, OutT *out) {

  // 'out' is assumed to be contiguous in memory, and have the same shape as
  // 'in'
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, num_dims) <= idx)
    return;
  int in_idx = get_idx_from_strides<InT>(in_shape, in_strides, num_dims, idx);

  out[idx] = static_cast<OutT>(in[in_idx]);
}

} // namespace cuda
} // namespace pg