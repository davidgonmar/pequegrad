#include "binary_ops_kernels.cuh"
#include "binary_ops_macro.cuh"
#include <cmath>

DEF_BIN_OP_KERNEL(AddKernel, x + y)
DEF_BIN_OP_KERNEL(SubKernel, x - y)
DEF_BIN_OP_KERNEL(MultKernel, x *y)
DEF_BIN_OP_KERNEL(DivKernel, x / y)
DEF_BIN_OP_KERNEL(GreaterKernel, x > y)
DEF_BIN_OP_KERNEL(LessKernel, x < y)
DEF_BIN_OP_KERNEL(EqualKernel, x == y)
DEF_BIN_OP_KERNEL(NotEqualKernel, x != y)
DEF_BIN_OP_KERNEL(GreaterEqualKernel, x >= y)
DEF_BIN_OP_KERNEL(LessEqualKernel, x <= y)
DEF_BIN_OP_KERNEL(ElementWiseMaxKernel, x > y ? x : y)
DEF_BIN_OP_KERNEL(PowKernel, powf(x, y))
