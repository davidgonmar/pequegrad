#include <dtype.hpp>
#include <vector>

namespace pg {
namespace cpu {
template <typename T>
void im2col_cpu(const T *in, T *out, size_t k_h, size_t k_w, size_t x_h,
                size_t x_w, size_t stride_x, size_t stride_y, size_t batch_size,
                size_t in_channels, size_t dilation_x, size_t dilation_y) {
  size_t out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;
#pragma omp parallel for collapse(5)
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < in_channels; ++channel) {
      for (int h = 0; h < k_h; ++h) {
        for (int w = 0; w < k_w; ++w) {
          for (int out_y = 0; out_y < out_h; ++out_y) {
            for (int out_x = 0; out_x < out_w; ++out_x) {
              size_t in_y = h * dilation_y + out_y * stride_y;
              size_t in_x = w * dilation_x + out_x * stride_x;
              out[batch * (in_channels * k_h * k_w * out_h * out_w) +
                  channel * (k_h * k_w * out_h * out_w) +
                  h * (k_w * out_h * out_w) + w * (out_h * out_w) +
                  out_y * out_w + out_x] =
                  in[batch * (in_channels * x_h * x_w) + channel * (x_h * x_w) +
                     in_y * x_w + in_x];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void col2im_cpu(const T *in, T *out, size_t out_channels, size_t k_h,
                size_t k_w, size_t in_h, size_t in_w, size_t batch_size,
                size_t out_h, size_t out_w, size_t stride_x, size_t stride_y,
                size_t dilation_x, size_t dilation_y) {
#pragma omp parallel for collapse(5)
  for (int col = 0; col < in_w; ++col) {
    for (int channel = 0; channel < out_channels; ++channel) {
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int ky = 0; ky < k_h; ++ky) {
          for (int kx = 0; kx < k_w; ++kx) {
            int n_horizontal_slides =
                (out_w - (k_w - 1) * dilation_x - 1) / stride_x + 1;
            int out_x_offset = col % n_horizontal_slides * stride_x;
            int out_y_offset = col / n_horizontal_slides * stride_y;
            int in_row = ky * k_w + kx + channel * k_w * k_h;
#pragma omp atomic
            out[batch * out_channels * out_h * out_w + channel * out_h * out_w +
                out_y_offset * out_w + ky * out_w * dilation_y + out_x_offset +
                kx * dilation_x] +=
                in[batch * in_w * in_h + in_row * in_w + col];
          }
        }
      }
    }
  }
}

void dispatch_im2col_kernel(DType dtype, void *in, void *out, size_t k_h,
                            size_t k_w, size_t x_h, size_t x_w, size_t stride_x,
                            size_t stride_y, size_t batch_size,
                            size_t in_channels, size_t dilation_x,
                            size_t dilation_y);

void dispatch_col2im_kernel(DType dtype, void *in, void *out,
                            size_t out_channels, size_t k_h, size_t k_w,
                            size_t in_h, size_t in_w, size_t batch_size,
                            size_t out_h, size_t out_w, size_t stride_x,
                            size_t stride_y, size_t dilation_x,
                            size_t dilation_y);
} // namespace cpu
} // namespace pg