#pragma once

#include "cuda_tensor/cuda_utils.cuh"
#include "dtype.hpp"

#define KERNEL_PARAMS_UNARY(T)                                                 \
  const size_t *in_strides, const size_t *shape, const size_t num_dims,        \
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
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    int in_idx = get_idx_from_strides<T>(shape, in_strides, num_dims, idx);    \
    T x = in[in_idx];                                                          \
    out[idx] = FN;                                                             \
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
copy_with_out_strides_kernel(const size_t *in_strides, const size_t *in_shape,
                             const size_t *out_strides, const size_t *out_shape,
                             const size_t in_num_dims,
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
__global__ void astype_kernel(const size_t *in_strides, const size_t *in_shape,
                              const size_t num_dims, const InT *in, OutT *out) {

  // 'out' is assumed to be contiguous in memory, and have the same shape as
  // 'in'
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, num_dims) <= idx)
    return;
  int in_idx = get_idx_from_strides<InT>(in_shape, in_strides, num_dims, idx);

  out[idx] = static_cast<OutT>(in[in_idx]);
}

template <typename T>
void launch_unary_kernel_helper(UnaryKernelType type, dim3 blocks, dim3 threads,
                                const size_t *in_strides, const size_t *shape,
                                const size_t num_dims, const T *in, T *out) {
  switch (type) {
  case UnaryKernelType::COPY:
    copy_kernel<<<blocks, threads>>>(in_strides, shape, num_dims, in, out);
    break;
  case UnaryKernelType::EXP:
    exp_kernel<<<blocks, threads>>>(in_strides, shape, num_dims, in, out);
    break;
  case UnaryKernelType::LOG:
    log_kernel<<<blocks, threads>>>(in_strides, shape, num_dims, in, out);
    break;
  default:
    throw std::runtime_error("Invalid UnaryKernelType");
  }
}

template <typename T>
void launch_unary_kernel_dense_helper(UnaryKernelType type, dim3 blocks,
                                      dim3 threads, const size_t total_size,
                                      const T *in, T *out) {
  switch (type) {
  case UnaryKernelType::EXP:
    exp_kernel_dense<<<blocks, threads>>>(total_size, in, out);
    break;
  case UnaryKernelType::LOG:
    log_kernel_dense<<<blocks, threads>>>(total_size, in, out);
    break;
  default:
    throw std::runtime_error("Invalid UnaryKernelType");
  }
}

void launch_unary_kernel_dense(UnaryKernelType type, DType dtype, dim3 blocks,
                               dim3 threads, const size_t total_size,
                               const void *in, void *out);

template <typename T>
void launch_copy_with_out_strides_kernel_helper(
    dim3 blocks, dim3 threads, const size_t *in_strides, const size_t *in_shape,
    const size_t *out_strides, const size_t *out_shape,
    const size_t in_num_dims, const size_t out_num_dims, const T *in, T *out) {
  copy_with_out_strides_kernel<T>
      <<<blocks, threads>>>(in_strides, in_shape, out_strides, out_shape,
                            in_num_dims, out_num_dims, in, out);
}

template <typename InT, typename OutT>
void launch_astype_kernel_helper(dim3 blocks, dim3 threads,
                                 const size_t *in_strides,
                                 const size_t *in_shape, const size_t num_dims,
                                 const InT *in, OutT *out) {
  astype_kernel<InT, OutT>
      <<<blocks, threads>>>(in_strides, in_shape, num_dims, in, out);
}

void launch_copy_with_out_strides_kernel(
    DType dtype, dim3 blocks, dim3 threads, const size_t *in_strides,
    const size_t *in_shape, const size_t *out_strides, const size_t *out_shape,
    const size_t in_num_dims, const size_t out_num_dims, const void *in,
    void *out);

void launch_astype_kernel(DType in_dtype, DType out_dtype, dim3 blocks,
                          dim3 threads, const size_t *in_strides,
                          const size_t *in_shape, const size_t num_dims,
                          const void *in, void *out);

void launch_unary_kernel(UnaryKernelType type, DType dtype, dim3 blocks,
                         dim3 threads, const size_t *in_strides,
                         const size_t *shape, const size_t num_dims,
                         const void *in, void *out);