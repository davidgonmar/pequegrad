#define DEF_BIN_OP_KERNEL(KERNEL_NAME, FN)\
__global__ void KERNEL_NAME(\
    const int *lhs_strides, \
    const int *rhs_strides,\
    const int *shape, /* both lhs and rhs should have equal shape, we dont handle broadcasting here */\
    const int num_dims, /* equals len of strides and shape */\
    const float *lhs,\
    const float *rhs,\
    float *out\
) {\
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\
    /* calculate correct index based on strides */\
    int idxLhs = 0;\
    int idxRhs = 0;\
    int tmpI = idx;\
    for (int d = num_dims - 1; d >= 0; d--){\
        int dim = tmpI % shape[d];\
        idxLhs += dim * lhs_strides[d] / sizeof(float);\
        idxRhs += dim * rhs_strides[d] / sizeof(float);\
        tmpI /= shape[d];\
    }\
    float x = lhs[idxLhs];\
    float y = rhs[idxRhs];\
    out[idx] = FN; /* assume we have contiguous output */\
}