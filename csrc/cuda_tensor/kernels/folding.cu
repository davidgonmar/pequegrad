#include "folding.cuh"

void launch_im2col_kernel(DType dtype, dim3 blocks, dim3 threads, void *in,
                          void *out, size_t k_h, size_t k_w, size_t x_h,
                          size_t x_w, size_t stride_x, size_t stride_y,
                          size_t batch_size, size_t in_channels,
                          size_t dilation_x, size_t dilation_y) {
  switch (dtype) {
  case DType::Float32:
    im2col_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(in), static_cast<float *>(out), k_h, k_w, x_h, x_w,
        stride_x, stride_y, batch_size, in_channels, dilation_x, dilation_y);
    break;
  case DType::Float64:
    im2col_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(in), static_cast<double *>(out), k_h, k_w, x_h,
        x_w, stride_x, stride_y, batch_size, in_channels, dilation_x,
        dilation_y);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

void launch_col2im_kernel(DType dtype, dim3 blocks, dim3 threads, void *in,
                          void *out, size_t out_channels, size_t k_h,
                          size_t k_w, size_t in_h, size_t in_w,
                          size_t batch_size, size_t out_h, size_t out_w,
                          size_t stride_x, size_t stride_y, size_t dilation_x,
                          size_t dilation_y) {
  switch (dtype) {
  case DType::Float32:
    col2im_kernel<float><<<blocks, threads>>>(
        static_cast<float *>(in), static_cast<float *>(out), out_channels, k_h,
        k_w, in_h, in_w, batch_size, out_h, out_w, stride_x, stride_y,
        dilation_x, dilation_y);
    break;
  case DType::Float64:
    col2im_kernel<double><<<blocks, threads>>>(
        static_cast<double *>(in), static_cast<double *>(out), out_channels,
        k_h, k_w, in_h, in_w, batch_size, out_h, out_w, stride_x, stride_y,
        dilation_x, dilation_y);
    break;
    throw std::runtime_error("Unsupported data type");
  default:
    throw std::runtime_error("Unsupported data type");
  }
}