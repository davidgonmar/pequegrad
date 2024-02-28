#include "ternary_ops_kernels.cuh"
#include "ternary_ops_macro.cuh"
#include <cmath>

DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, float)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, double)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, int)