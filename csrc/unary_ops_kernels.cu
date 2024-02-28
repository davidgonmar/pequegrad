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


void launch_unary_kernel(UnaryKernelType type, DType dtype, dim3 blocks, dim3 threads,
              const size_t *in_strides, const size_t *shape, const size_t num_dims,
              const void *_in, void *_out) {
  switch (dtype) {
  case DType::Float32:
    __launch_unary_kernel<float>(type, blocks, threads, in_strides, shape, num_dims, (const float *)_in, (float *)_out);
    break;
  case DType::Float64:
    __launch_unary_kernel<double>(type, blocks, threads, in_strides, shape, num_dims, (const double *)_in, (double *)_out);
    break;
  case DType::Int32:
    __launch_unary_kernel<int>(type, blocks, threads, in_strides, shape, num_dims, (const int *)_in, (int *)_out);
    break;
  }
}

void launch_copy_with_out_strides_kernel(DType dtype, dim3 blocks, dim3 threads,
                                         const size_t *in_strides,
                                         const size_t *in_shape,
                                         const size_t *out_strides,
                                         const size_t *out_shape,
                                         const size_t in_num_dims,
                                         const size_t out_num_dims, const void *in,
                                         void *out) {
    switch (dtype) {
    case DType::Float32:
      __launch_copy_with_out_strides_kernel<float>(blocks, threads, in_strides, in_shape, out_strides, out_shape, in_num_dims, out_num_dims, (const float *)in, (float *)out);
      break;
    case DType::Float64:
        __launch_copy_with_out_strides_kernel<double>(blocks, threads, in_strides, in_shape, out_strides, out_shape, in_num_dims, out_num_dims, (const double *)in, (double *)out);
        break;
    case DType::Int32:
        __launch_copy_with_out_strides_kernel<int>(blocks, threads, in_strides, in_shape, out_strides, out_shape, in_num_dims, out_num_dims, (const int *)in, (int *)out);
        break;
    }
}

void launch_astype_kernel(DType in_dtype, DType out_dtype, dim3 blocks, dim3 threads,
                          const size_t *in_strides, const size_t *in_shape,
                          const size_t num_dims, const void *in, void *out) {
    switch (in_dtype) {
    case DType::Float32:
        switch (out_dtype) {
        case DType::Float32:
            __launch_astype_kernel<float, float>(blocks, threads, in_strides, in_shape, num_dims, (const float *)in, (float *)out);
            break;
        case DType::Float64:
            __launch_astype_kernel<float, double>(blocks, threads, in_strides, in_shape, num_dims, (const float *)in, (double *)out);
            break;
        case DType::Int32:
            __launch_astype_kernel<float, int>(blocks, threads, in_strides, in_shape, num_dims, (const float *)in, (int *)out);
            break;
    
        }
        break;
    case DType::Float64:
        switch (out_dtype) {
        case DType::Float32:
            __launch_astype_kernel<double, float>(blocks, threads, in_strides, in_shape, num_dims, (const double *)in, (float *)out);
            break;
        case DType::Float64:
            __launch_astype_kernel<double, double>(blocks, threads, in_strides, in_shape, num_dims, (const double *)in, (double *)out);
            break;
        case DType::Int32:
            __launch_astype_kernel<double, int>(blocks, threads, in_strides, in_shape, num_dims, (const double *)in, (int *)out);
            break;
        }
        break;
    case DType::Int32:
        switch (out_dtype) {
        case DType::Float32:
            __launch_astype_kernel<int, float>(blocks, threads, in_strides, in_shape, num_dims, (const int *)in, (float *)out);
            break;
        case DType::Float64:
            __launch_astype_kernel<int, double>(blocks, threads, in_strides, in_shape, num_dims, (const int *)in, (double *)out);
            break;
        case DType::Int32:
            __launch_astype_kernel<int, int>(blocks, threads, in_strides, in_shape, num_dims, (const int *)in, (int *)out);
            break;
        }
        break;
    }
}