#include "ternary_ops_macro.cuh"
#include <cmath>

DEF_TERNARY_OP_KERNEL(WhereKernel, x ? y : z)