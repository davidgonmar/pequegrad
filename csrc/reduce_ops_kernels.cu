#include "cuda_array.cuh"
#include "reduce_ops_kernels.cuh"
void launch_reduce_kernel(ReduceKernelType type, DType dtype, dim3 blocks,
                          dim3 threads, const void *in, void *out,
                          const size_t *in_strides, const size_t *in_shape,
                          const size_t n_dims, const size_t red_axis) {
  if (dtype == DType::Float32) {
    __launch_reduce_kernel<float>(type, blocks, threads, (const float *)in,
                                  (float *)out, in_strides, in_shape, n_dims,
                                  red_axis);
  } else if (dtype == DType::Int32) {
    __launch_reduce_kernel<int>(type, blocks, threads, (const int *)in,
                                (int *)out, in_strides, in_shape, n_dims,
                                red_axis);
  } else if (dtype == DType::Float64) {
    __launch_reduce_kernel<double>(type, blocks, threads, (const double *)in,
                                   (double *)out, in_strides, in_shape, n_dims,
                                   red_axis);
  }
}