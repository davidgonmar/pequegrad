#include "cuda_tensor.cuh"

CudaTensor CudaTensor::im2col(shape_t kernel_shape, int stride_y, int stride_x,
                              int dilation_y, int dilation_x) const {
  if (!is_contiguous()) {
    return as_contiguous().im2col(kernel_shape, stride_y, stride_x, dilation_y,
                                  dilation_x);
  }
  PG_CHECK_ARG(ndim() == 4, "ndim has to be 4 in im2col, got shape ",
               vec_to_string(shape));
  PG_CHECK_ARG(kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               vec_to_string(kernel_shape));
  size_t k_h = kernel_shape[0];
  size_t k_w = kernel_shape[1];

  size_t batch_size = shape[0];
  size_t in_channels = shape[1];
  size_t x_h = shape[2];
  size_t x_w = shape[3];

  size_t out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, in_channels * k_h * k_w, out_h * out_w};
  size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                    std::multiplies<size_t>());

  CudaTensor out(out_shape, dtype);

  int total_iters = batch_size * out_h * out_w * in_channels * k_h *
                    k_w; // check kernel code for more details
  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_iters / (float)block_size);

  launch_im2col_kernel(dtype, grid_size, block_size, get_base_ptr(),
                       out.get_base_ptr(), k_h, k_w, x_h, x_w, stride_x,
                       stride_y, batch_size, in_channels, dilation_x,
                       dilation_y);
  PG_CUDA_KERNEL_END;
  return out;
}

CudaTensor CudaTensor::col2im(shape_t kernel_shape, shape_t out_shape,
                              int stride_y, int stride_x, int dilation_y,
                              int dilation_x) const {
  if (!is_contiguous()) {
    return as_contiguous().col2im(kernel_shape, out_shape, stride_y, stride_x,
                                  dilation_y, dilation_x);
  }

  PG_CHECK_ARG(ndim() == 3, "ndim has to be 3 in col2im, got shape ",
               vec_to_string(shape));
  PG_CHECK_ARG(kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               vec_to_string(kernel_shape));
  PG_CHECK_ARG(out_shape.size() == 2, "out shape size must be 2, got ",
               vec_to_string(out_shape));

  size_t k_h = kernel_shape[0];
  size_t k_w = kernel_shape[1];
  size_t out_h = out_shape[0];
  size_t out_w = out_shape[1];
  size_t in_h = shape[1];
  size_t in_w = shape[2];

  // out_shape is just (out_h, out_w)
  size_t out_channels = shape[1] / (k_h * k_w);

  size_t out_batch_size = shape[0];
  shape_t _out_shape = {out_batch_size, out_channels, out_h, out_w};
  size_t out_size = std::accumulate(_out_shape.begin(), _out_shape.end(), 1,
                                    std::multiplies<size_t>());
  CudaTensor out(_out_shape, dtype);
  CHECK_CUDA(
      cudaMemset(out.get_base_ptr(), 0, out.nbytes)); // set output to zero

  dim3 block_size(DEFAULT_BLOCK_SIZE);
  /*int col = idx % in_w;
  int channel = (idx / in_w) % out_channels;
  int batch = idx / in_w / out_channels % batch_size;
  int ky = (idx / in_w / out_channels / batch_size) % k_h;
  int kx = (idx / in_w / out_channels / batch_size / k_h) % k_w;
*/
  dim3 grid_size(
      ceil(10000000 / (float)block_size.x)); // 1000000 is a random number
  launch_col2im_kernel(dtype, grid_size, block_size, get_base_ptr(),
                       out.get_base_ptr(), out_channels, k_h, k_w, in_h, in_w,
                       out_batch_size, out_h, out_w, stride_x, stride_y,
                       dilation_x, dilation_y);
  PG_CUDA_KERNEL_END;
  return out;
}