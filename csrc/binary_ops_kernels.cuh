#include <binary_ops_macro.cuh>

DEF_BIN_OP_KERNEL(AddKernel, x + y);
DEF_BIN_OP_KERNEL(SubKernel, x - y);
DEF_BIN_OP_KERNEL(MultKernel, x * y);
DEF_BIN_OP_KERNEL(DivKernel, x / y);