#include "reduce_ops_kernels.cuh"

struct SumOp {
  __device__ float apply(float a, float b) { return a + b; }
  __device__ float initial_value() { return 0.0f; }
};

struct MaxOp {
  __device__ float apply(float a, float b) { return max(a, b); }
  __device__ float initial_value() { return -INFINITY; }
};

__global__ void sum_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t red_axis) {
  reduce_base_fn<SumOp>(in, out, in_strides, in_shape, n_dims, red_axis);
}

__global__ void max_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t red_axis) {
  reduce_base_fn<MaxOp>(in, out, in_strides, in_shape, n_dims, red_axis);
}
