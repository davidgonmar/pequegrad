#include <stdio.h>

__global__ void im2col_kernel(float *in, float *out, size_t k_h, size_t k_w,
                              size_t x_h, size_t x_w, size_t stride,
                              size_t batch_size, size_t in_channels) {
  int out_h = (x_h - k_h) / stride + 1;
  int out_w = (x_w - k_w) / stride + 1;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // out shape (batch_size, in_channels * k_h * k_w, out_h * out_w)
  // in shape (batch_size, in_channels, x_h, x_w)
  // each col in out means a 'sliding window' in the in
  // number of total cols = out_h * out_w
  int col = idx % (out_h * out_w);
  int batch = idx / (out_h * out_w);

  if (col >= out_h * out_w || batch >= batch_size) {
    return;
  }

  int in_x_offset = col % out_w * stride;
  int in_y_offset = col / out_w * stride;

  // for each output in the current col, of size in_channels * k_h * k_w
  for (int channel = 0; channel < in_channels; channel++) {
    for (int h = 0; h < k_h; h++) {
      for (int w = 0; w < k_w; w++) {
        int row = (channel * k_h * k_w) + h * k_w + w;
        out[batch * in_channels * k_h * k_w * out_h * out_w +
            out_w * out_h * row + col] =
            in[batch * in_channels * x_h * x_w + channel * x_h * x_w +
               (h + in_y_offset) * x_w + (w + in_x_offset)];
      }
    }
  }
}

__global__ void col2im_kernel(float *in, float *out, size_t out_channels,
                              size_t k_h, size_t k_w, size_t in_h, size_t in_w,
                              size_t batch_size, size_t out_h, size_t out_w,
                              size_t stride) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int batch = idx;
  if (batch >= batch_size) {
    return;
  }
  int n_horizontal_slides = (out_w - k_w) / stride + 1;
  for (int chann = 0; chann < out_channels; chann++) {
    for (int col = 0; col < in_w; col++) {
      int out_x_offset = col % n_horizontal_slides * stride;
      int out_y_offset = col / n_horizontal_slides * stride;
      for (int ky = 0; ky < k_h; ky++) {
        for (int kx = 0; kx < k_w; kx++) {
          int in_row = ky * k_w + kx + chann * k_w * k_h;
          out[batch * out_channels * out_h * out_w + chann * out_h * out_w +
              out_y_offset * out_w + ky * out_w + out_x_offset + kx] +=
              in[batch * in_w * in_h + in_row * in_w + col];
        }
      }
    }
  }
}