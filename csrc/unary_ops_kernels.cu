#include "unary_ops_kernels.cuh"
#include "unary_ops_macro.cuh"
#include <cmath>

DEF_UNARY_OP_KERNEL(copy_kernel, x)
DEF_UNARY_OP_KERNEL(exp_kernel, exp((float)x))
DEF_UNARY_OP_KERNEL(log_kernel, log((float)x))

__global__ void copy_with_out_strides_kernel(
    const size_t *in_strides, const size_t *in_shape, const size_t *out_strides,
    const size_t *out_shape, const size_t in_num_dims,
    const size_t out_num_dims, const float *in, float *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (get_max_idx(in_shape, in_num_dims) < idx ||
      get_max_idx(out_shape, out_num_dims) < idx)
    return;
  int in_idx = get_idx_from_strides(in_shape, in_strides, in_num_dims, idx);
  int out_idx = get_idx_from_strides(out_shape, out_strides, out_num_dims, idx);
  out[out_idx] = in[in_idx];
}