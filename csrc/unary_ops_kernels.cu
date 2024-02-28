#include "unary_ops_kernels.cuh"
#include "unary_ops_macro.cuh"
#include <cmath>

DEF_UNARY_OP_KERNEL(copy_kernel, x, float)
DEF_UNARY_OP_KERNEL(copy_kernel, x, double)
DEF_UNARY_OP_KERNEL(copy_kernel, x, int)

DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x), float)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((double)x), double)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x), int)

DEF_UNARY_OP_KERNEL(log_kernel, log((float)x), float)
DEF_UNARY_OP_KERNEL(log_kernel, log((double)x), double)
DEF_UNARY_OP_KERNEL(log_kernel, log((float)x), int)
