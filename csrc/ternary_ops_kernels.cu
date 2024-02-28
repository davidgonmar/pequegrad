#include "ternary_ops_kernels.cuh"
#include "ternary_ops_macro.cuh"
#include <cmath>

DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, float)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, double)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, int)

void launch_ternary_kernel(TernaryKernelType kt, DType dtype, dim3 grid_size,
                           dim3 block_size, const size_t *first_strides,
                           const size_t *second_strides,
                           const size_t *third_strides, const size_t *shape,
                           const size_t num_dims, const void *first,
                           const void *second, const void *third, void *out) {
  switch (dtype) {
  case DType::Float32:
    __launch_ternary_kernel<float>(
        kt, grid_size, block_size, first_strides, second_strides, third_strides,
        shape, num_dims, static_cast<const float *>(first),
        static_cast<const float *>(second), static_cast<const float *>(third),
        static_cast<float *>(out));
    break;
  case DType::Float64:
    __launch_ternary_kernel<double>(
        kt, grid_size, block_size, first_strides, second_strides, third_strides,
        shape, num_dims, static_cast<const double *>(first),
        static_cast<const double *>(second), static_cast<const double *>(third),
        static_cast<double *>(out));
    break;
  case DType::Int32:
    __launch_ternary_kernel<int>(
        kt, grid_size, block_size, first_strides, second_strides, third_strides,
        shape, num_dims, static_cast<const int *>(first),
        static_cast<const int *>(second), static_cast<const int *>(third),
        static_cast<int *>(out));
    break;
  }
}