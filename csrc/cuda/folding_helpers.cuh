#pragma once
#include "dtype.hpp"
namespace pg {

namespace cuda {

template <typename T>
__global__ void im2col_kernel(T *in, T *out, size_t k_h, size_t k_w, size_t x_h,
                              size_t x_w, size_t stride_x, size_t stride_y,
                              size_t batch_size, size_t in_channels,
                              size_t dilation_x, size_t dilation_y) {
  int out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  int out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // out shape (batch_size, in_channels * k_h * k_w, out_h * out_w)
  // in shape (batch_size, in_channels, x_h, x_w)
  // each col in out means a 'sliding window' in the in
  // number of total cols = out_h * out_w
  int col = idx % (out_h * out_w);
  int channel = idx / (out_h * out_w) % in_channels;
  int h = idx / (out_h * out_w) / in_channels % k_h;
  int w = idx / (out_h * out_w) / in_channels / k_h % k_w;
  int batch = idx / (out_h * out_w) / in_channels / k_h / k_w;

  if (col >= out_h * out_w || batch >= batch_size || channel >= in_channels ||
      h >= k_h || w >= k_w) {
    return;
  }

  int in_x_offset = col % out_w * stride_x;
  int in_y_offset = col / out_w * stride_y;

  // for each output in the current col, of size in_channels * k_h * k_w
  int row = (channel * k_h * k_w) + h * k_w + w;
  out[batch * in_channels * k_h * k_w * out_h * out_w + out_w * out_h * row +
      col] =
      in[batch * in_channels * x_h * x_w + channel * x_h * x_w +
         (h * dilation_y + in_y_offset) * x_w + (w * dilation_x + in_x_offset)];
}

void launch_im2col_kernel(DType dtype, dim3 blocks, dim3 threads, void *in,
                          void *out, size_t k_h, size_t k_w, size_t x_h,
                          size_t x_w, size_t stride_x, size_t stride_y,
                          size_t batch_size, size_t in_channels,
                          size_t dilation_x, size_t dilation_y);

template <typename T>
__global__ void col2im_kernel(T *in, T *out, size_t out_channels, size_t k_h,
                              size_t k_w, size_t in_h, size_t in_w,
                              size_t batch_size, size_t out_h, size_t out_w,
                              size_t stride_x, size_t stride_y,
                              size_t dilation_x, size_t dilation_y) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int col = idx % in_w;
  int ky = idx / in_w % k_h;
  int kx = idx / in_w / k_h % k_w;
  int channel = idx / in_w / k_h / k_w % out_channels;
  int batch = idx / in_w / k_h / k_w / out_channels;

  if (batch >= batch_size || channel >= out_channels || ky >= k_h ||
      kx >= k_w || col >= in_w) {
    return;
  }

  int n_horizontal_slides = (out_w - (k_w - 1) * dilation_x - 1) / stride_x + 1;

  int out_x_offset = col % n_horizontal_slides * stride_x;
  int out_y_offset = col / n_horizontal_slides * stride_y;
  int in_row = ky * k_w + kx + channel * k_w * k_h;

  atomicAdd(&out[batch * out_channels * out_h * out_w +
                 channel * out_h * out_w + out_y_offset * out_w +
                 ky * out_w * dilation_y + out_x_offset + kx * dilation_x],
            in[batch * in_w * in_h + in_row * in_w + col]);
}

void launch_col2im_kernel(DType dtype, dim3 blocks, dim3 threads, void *in,
                          void *out, size_t out_channels, size_t k_h,
                          size_t k_w, size_t in_h, size_t in_w,
                          size_t batch_size, size_t out_h, size_t out_w,
                          size_t stride_x, size_t stride_y, size_t dilation_x,
                          size_t dilation_y);
} // namespace cuda
} // namespace pg