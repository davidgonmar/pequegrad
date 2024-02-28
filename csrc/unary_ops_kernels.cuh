#pragma once

#include "dtype.cuh"
#include "utils.cuh"

#define KERNEL_PARAMS_UN(T)                                                    \
  const size_t *in_strides, const size_t *shape, const size_t num_dims,        \
      const T *in, T *out

__global__ void copy_kernel(KERNEL_PARAMS_UN(float));
__global__ void copy_kernel(KERNEL_PARAMS_UN(double));
__global__ void copy_kernel(KERNEL_PARAMS_UN(int));
__global__ void exp_kernel(KERNEL_PARAMS_UN(float));
__global__ void exp_kernel(KERNEL_PARAMS_UN(double));
__global__ void exp_kernel(KERNEL_PARAMS_UN(int));
__global__ void log_kernel(KERNEL_PARAMS_UN(float));
__global__ void log_kernel(KERNEL_PARAMS_UN(double));
__global__ void log_kernel(KERNEL_PARAMS_UN(int));

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

template <typename InT, typename OutT> // both have same strides and everything
__global__ void astype_kernel(const size_t *in_strides, const size_t *in_shape,
                              const size_t num_dims, const InT *in, OutT *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, num_dims) < idx)
    return;
  int in_idx = get_idx_from_strides<InT>(in_shape, in_strides, num_dims, idx);

  out[idx] = static_cast<OutT>(in[in_idx]);
}

enum class UnaryKernelType {
  COPY,
  EXP,
  LOG,
  ASTYPE,
};

template <typename T>
void __launch_unary_kernel(UnaryKernelType type, dim3 blocks, dim3 threads,
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

void launch_unary_kernel(UnaryKernelType type, DType dtype, dim3 blocks,
                         dim3 threads, const size_t *in_strides,
                         const size_t *shape, const size_t num_dims,
                         const void *_in, void *_out);

template <typename T>
void __launch_copy_with_out_strides_kernel(
    dim3 blocks, dim3 threads, const size_t *in_strides, const size_t *in_shape,
    const size_t *out_strides, const size_t *out_shape,
    const size_t in_num_dims, const size_t out_num_dims, const T *in, T *out) {
  copy_with_out_strides_kernel<T>
      <<<blocks, threads>>>(in_strides, in_shape, out_strides, out_shape,
                            in_num_dims, out_num_dims, in, out);
}

void launch_copy_with_out_strides_kernel(
    DType dtype, dim3 blocks, dim3 threads, const size_t *in_strides,
    const size_t *in_shape, const size_t *out_strides, const size_t *out_shape,
    const size_t in_num_dims, const size_t out_num_dims, const void *in,
    void *out);

template <typename InT, typename OutT>
void __launch_astype_kernel(dim3 blocks, dim3 threads, const size_t *in_strides,
                            const size_t *in_shape, const size_t num_dims,
                            const InT *in, OutT *out) {
  astype_kernel<InT, OutT>
      <<<blocks, threads>>>(in_strides, in_shape, num_dims, in, out);
}

void launch_astype_kernel(DType in_dtype, DType out_dtype, dim3 blocks,
                          dim3 threads, const size_t *in_strides,
                          const size_t *in_shape, const size_t num_dims,
                          const void *in, void *out);