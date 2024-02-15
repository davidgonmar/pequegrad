__global__ void im2col_kernel(float *in, float *out, size_t k_h, size_t k_w,
                              size_t x_h, size_t x_w, size_t stride,
                              size_t batch_size, size_t in_channels);

__global__ void col2im_kernel(float *in, float *out, size_t in_channels,
                              size_t k_h, size_t k_w, size_t in_h, size_t in_w,
                              size_t batch_size, size_t out_h, size_t out_w,
                              size_t stride);