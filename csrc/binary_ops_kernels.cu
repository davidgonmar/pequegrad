#include "binary_ops_kernels.cuh"
#include "binary_ops_macro.cuh"
#include <cmath>

DEF_BIN_OP_KERNEL(add_kernel, x + y)
DEF_BIN_OP_KERNEL(sub_kernel, x - y)
DEF_BIN_OP_KERNEL(mult_kernel, x * y)
DEF_BIN_OP_KERNEL(div_kernel, x / y)
DEF_BIN_OP_KERNEL(greater_kernel, x > y)
DEF_BIN_OP_KERNEL(less_kernel, x < y)
DEF_BIN_OP_KERNEL(equal_kernel, x == y)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y)
DEF_BIN_OP_KERNEL(pow_kernel, powf(x, y))
