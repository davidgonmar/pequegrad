#include "unary.cuh"
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

void launch_unary_kernel(UnaryKernelType type, DType dtype, dim3 blocks,
                         dim3 threads, const size_t *in_strides,
                         const size_t *shape, const size_t num_dims,
                         const void *_in, void *_out) {
  switch (dtype) {
  case DType::Float32:
    launch_unary_kernel_helper<float>(type, blocks, threads, in_strides, shape,
                                 num_dims, static_cast<const float *>(_in), static_cast<float *>(_out));
    break;
  case DType::Float64:
    launch_unary_kernel_helper<double>(type, blocks, threads, in_strides, shape,
                                  num_dims, static_cast<const double *>(_in),
                                  static_cast<double *>(_out));
    break;
  case DType::Int32:
    launch_unary_kernel_helper<int>(type, blocks, threads, in_strides, shape,
                               num_dims, static_cast<const int *>(_in), static_cast<int *>(_out));
    break;
  }
}

void launch_copy_with_out_strides_kernel(
    DType dtype, dim3 blocks, dim3 threads, const size_t *in_strides,
    const size_t *in_shape, const size_t *out_strides, const size_t *out_shape,
    const size_t in_num_dims, const size_t out_num_dims, const void *in,
    void *out) {
  switch (dtype) {
  case DType::Float32:
    launch_copy_with_out_strides_kernel_helper<float>(
        blocks, threads, in_strides, in_shape, out_strides, out_shape,
        in_num_dims, out_num_dims, static_cast<const float *>(in), static_cast<float *>(out));
    break;
  case DType::Float64:
    launch_copy_with_out_strides_kernel_helper<double>(
        blocks, threads, in_strides, in_shape, out_strides, out_shape,
        in_num_dims, out_num_dims, static_cast<const double *>(in), static_cast<double *>(out));
    break;
  case DType::Int32:
    launch_copy_with_out_strides_kernel_helper<int>(
        blocks, threads, in_strides, in_shape, out_strides, out_shape,
        in_num_dims, out_num_dims, static_cast<const int *>(in), static_cast<int *>(out));
    break;
  }
}

void launch_astype_kernel(DType in_dtype, DType out_dtype, dim3 blocks,
                          dim3 threads, const size_t *in_strides,
                          const size_t *in_shape, const size_t num_dims,
                          const void *in, void *out) {
  switch (in_dtype) {
  case DType::Float32:
    switch (out_dtype) {
    case DType::Float32:
      launch_astype_kernel_helper<float, float>(blocks, threads, in_strides,
                                           in_shape, num_dims,
                                           static_cast<const float *>(in), static_cast<float *>(out));
      break;
    case DType::Float64:
      launch_astype_kernel_helper<float, double>(blocks, threads, in_strides,
                                            in_shape, num_dims,
                                            static_cast<const float *>(in), static_cast<double *>(out));
      break;
    case DType::Int32:
      launch_astype_kernel_helper<float, int>(blocks, threads, in_strides, in_shape,
                                         num_dims, static_cast<const float *>(in),
                                         static_cast<int *>(out));
      break;
    }
    break;
  case DType::Float64:
    switch (out_dtype) {
    case DType::Float32:
      launch_astype_kernel_helper<double, float>(blocks, threads, in_strides,
                                            in_shape, num_dims,
                                            static_cast<const double *>(in), static_cast<float *>(out));
      break;
    case DType::Float64:
      launch_astype_kernel_helper<double, double>(blocks, threads, in_strides,
                                             in_shape, num_dims,
                                             static_cast<const double *>(in), static_cast<double *>(out));
      break;
    case DType::Int32:
      launch_astype_kernel_helper<double, int>(blocks, threads, in_strides, in_shape,
                                          num_dims, static_cast<const double *>(in),
                                          static_cast<int *>(out));
      break;
    }
    break;
  case DType::Int32:
    switch (out_dtype) {
    case DType::Float32:
      launch_astype_kernel_helper<int, float>(blocks, threads, in_strides, in_shape,
                                         num_dims, static_cast<const int *>(in),
                                         static_cast<float *>(out));
      break;
    case DType::Float64:
      launch_astype_kernel_helper<int, double>(blocks, threads, in_strides, in_shape,
                                          num_dims, static_cast<const int *>(in),
                                          static_cast<double *>(out));
      break;
    case DType::Int32:
      launch_astype_kernel_helper<int, int>(blocks, threads, in_strides, in_shape,
                                       num_dims, static_cast<const int *>(in), static_cast<int *>(out));
      break;
    }
    break;
  }
}