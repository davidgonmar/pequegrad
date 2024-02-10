#include "unary_ops_kernels.cuh"
#include "unary_ops_macro.cuh"
#include <cmath>

DEF_UNARY_OP_KERNEL(copy_kernel, x)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x))
DEF_UNARY_OP_KERNEL(log_kernel, log((float)x))