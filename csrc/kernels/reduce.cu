#include "reduce.cuh"
#include "dtype.hpp"

void launch_reduce_kernel(ReduceKernelType type, DType dtype, dim3 blocks,
                          dim3 threads, const void *in, void *out,
                          const size_t *in_strides, const size_t *in_shape,
                          const size_t n_dims, const size_t red_axis) {
  if (dtype == DType::Float32) {
    launch_reduce_kernel_helper<float>(type, blocks, threads, static_cast<const float *>(in),
                                  static_cast<float *>(out), in_strides, in_shape, n_dims,
                                  red_axis);
  } else if (dtype == DType::Int32) {
    launch_reduce_kernel_helper<int>(type, blocks, threads, static_cast<const int *>(in),
                                static_cast<int *>(out), in_strides, in_shape, n_dims,
                                red_axis);
  } else if (dtype == DType::Float64) {
    launch_reduce_kernel_helper<double>(type, blocks, threads, static_cast<const double *>(in),
                                   static_cast<double *>(out), in_strides, in_shape, n_dims,
                                   red_axis);
  }
}