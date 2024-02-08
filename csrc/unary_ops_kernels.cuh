#include "unary_ops_macro.cuh"
#include <cmath>

DEF_UNARY_OP_KERNEL(CopyKernel, x)
DEF_UNARY_OP_KERNEL(ExpKernel, exp((float)x))
DEF_UNARY_OP_KERNEL(LogKernel, log((float)x))