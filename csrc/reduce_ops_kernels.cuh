__global__ void sum_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t reduce_axis);

__global__ void max_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t reduce_axis);