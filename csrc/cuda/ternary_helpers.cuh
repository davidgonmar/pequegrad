#pragma once
#include "cuda_tensor/cuda_utils.cuh"
#include "dtype.hpp"
#include "shape.hpp"

namespace pg {
namespace cuda {
#define KERNEL_PARAMS_TER(T)                                                   \
  const stride_t *first_strides, const stride_t *second_strides,               \
      const stride_t *third_strides, const size_t *shape,                      \
      const size_t num_dims, const T *first, const T *second, const T *third,  \
      T *out

__global__ void where_kernel(KERNEL_PARAMS_TER(float));
__global__ void where_kernel(KERNEL_PARAMS_TER(double));
__global__ void where_kernel(KERNEL_PARAMS_TER(int));

#define DEF_TERNARY_OP_KERNEL(KERNEL_NAME, FN, T)                              \
  __global__ void KERNEL_NAME(                                                 \
      const stride_t *first_strides,  /* in bytes */                           \
      const stride_t *second_strides, /* in bytes */                           \
      const stride_t *third_strides,  /* in bytes */                           \
      const size_t *shape,   /* all inputs should have equal shape, we         \
                             don't handle broadcasting here */                 \
      const size_t num_dims, /* equals len of strides and len of shape */      \
      const T *first, const T *second, const T *third, T *out) {               \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;                           \
    if (get_max_idx(shape, num_dims) <= idx)                                   \
      return;                                                                  \
    /* calculate correct index based on strides */                             \
    int idx_f = get_idx_from_strides<T>(shape, first_strides, num_dims, idx);  \
    int idx_s = get_idx_from_strides<T>(shape, second_strides, num_dims, idx); \
    int idx_t = get_idx_from_strides<T>(shape, third_strides, num_dims, idx);  \
    T x = first[idx_f];                                                        \
    T y = second[idx_s];                                                       \
    T z = third[idx_t];                                                        \
    out[idx] = FN;                                                             \
  }

enum class TernaryKernelType { WHERE };

template <typename T>
void launch_ternary_kernel_helper(
    TernaryKernelType kernel_type, dim3 grid_size, dim3 block_size,
    const stride_t *first_strides, const stride_t *second_strides,
    const stride_t *third_strides, const size_t *shape, const size_t num_dims,
    const T *first, const T *second, const T *third, T *out) {
  switch (kernel_type) {
  case TernaryKernelType::WHERE:
    where_kernel<<<grid_size, block_size>>>(first_strides, second_strides,
                                            third_strides, shape, num_dims,
                                            first, second, third, out);
    break;
  }
}

void launch_ternary_kernel(TernaryKernelType kt, DType dtype, dim3 grid_size,
                           dim3 block_size, const stride_t *first_strides,
                           const stride_t *second_strides,
                           const stride_t *third_strides, const size_t *shape,
                           const size_t num_dims, const void *first,
                           const void *second, const void *third, void *out);

} // namespace cuda
} // namespace pg