// TODO -- this assumes both lhs and rhs are contiguous
#define DEF_BIN_OP_KERNEL(KERNEL_NAME, FN)\
__global__ void KERNEL_NAME(\
    const size_t numel, \
    const float *lhs, \
    const float *rhs, \
    float *out\
) {\
    const int i = blockDim.x * blockIdx.x + threadIdx.x;\
    if (i >= numel) return;\
    float x = lhs[i];\
    float y = rhs[i];\
    out[i] = FN;\
}\


// to be used like DEF_BIN_OP(AddKernel, x + y)