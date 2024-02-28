#pragma once
#include "dtype.cuh"

#define KERNEL_PARAMS_TER(T)                                                   \
  const size_t *first_strides, const size_t *second_strides,                   \
      const size_t *third_strides, const size_t *shape, const size_t num_dims, \
      const T *first, const T *second, const T *third, T *out

__global__ void where_kernel(KERNEL_PARAMS_TER(float));
__global__ void where_kernel(KERNEL_PARAMS_TER(double));
__global__ void where_kernel(KERNEL_PARAMS_TER(int));

enum class TernaryKernelType { WHERE };

template <typename T>
void __launch_ternary_kernel(TernaryKernelType kernel_type, dim3 grid_size,
                             dim3 block_size, const size_t *first_strides,
                             const size_t *second_strides,
                             const size_t *third_strides, const size_t *shape,
                             const size_t num_dims, const T *first,
                             const T *second, const T *third, T *out) {
  switch (kernel_type) {
  case TernaryKernelType::WHERE:
    where_kernel<<<grid_size, block_size>>>(first_strides, second_strides,
                                            third_strides, shape, num_dims,
                                            first, second, third, out);
    break;
  }
}

void launch_ternary_kernel(TernaryKernelType kt, DType dtype, dim3 grid_size,
                           dim3 block_size, const size_t *first_strides,
                           const size_t *second_strides,
                           const size_t *third_strides, const size_t *shape,
                           const size_t num_dims, const void *first,
                           const void *second, const void *third, void *out);