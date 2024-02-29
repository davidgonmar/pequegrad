#include "binary.cuh"
#include "dtype.hpp"
#include "utils.cuh"
#include <cmath>

DEF_BIN_OP_KERNEL(add_kernel, x + y, float)
DEF_BIN_OP_KERNEL(add_kernel, x + y, double)
DEF_BIN_OP_KERNEL(add_kernel, x + y, int)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, float)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, double)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, int)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, float)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, double)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, int)
DEF_BIN_OP_KERNEL(div_kernel, x / y, float)
DEF_BIN_OP_KERNEL(div_kernel, x / y, double)
DEF_BIN_OP_KERNEL(div_kernel, x / y, int)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, float)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, double)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, int)
DEF_BIN_OP_KERNEL(less_kernel, x < y, float)
DEF_BIN_OP_KERNEL(less_kernel, x < y, double)
DEF_BIN_OP_KERNEL(less_kernel, x < y, int)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, float)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, double)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, int)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, float)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, double)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, int)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, float)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, double)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, int)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, float)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, double)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, int)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, float)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, double)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, int)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), float)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), double)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), int)

void launch_binary_kernel(BinaryKernelType kernel_type, DType dtype,
                          dim3 grid_size, dim3 block_size,
                          const size_t *lhs_strides, const size_t *rhs_strides,
                          const size_t *shape, const size_t num_dims,
                          const void *lhs, const void *rhs, void *out) {
  switch (dtype) {
  case DType::Float32:
    launch_binary_kernel_casted_helper(
        kernel_type, grid_size, block_size, lhs_strides, rhs_strides, shape,
        num_dims, static_cast<const float *>(lhs), static_cast<const float *>(rhs), static_cast<float *>(out));
    break;
  case DType::Float64:
    launch_binary_kernel_casted_helper(
        kernel_type, grid_size, block_size, lhs_strides, rhs_strides, shape,
        num_dims, static_cast<const double *>(lhs), static_cast<const double *>(rhs), static_cast<double *>(out));
    break;
  case DType::Int32:
    launch_binary_kernel_casted_helper(kernel_type, grid_size, block_size, lhs_strides,
                                rhs_strides, shape, num_dims, static_cast<const int *>(lhs),
                                static_cast<const int *>(rhs), static_cast<int *>(out));
    break;
  }
}