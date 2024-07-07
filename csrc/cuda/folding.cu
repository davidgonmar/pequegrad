#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "folding.cuh"
#include "view_helpers.cuh"

namespace pg {

void Im2Col::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 1, "Im2Col expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(outputs.size() == 1, "Im2Col expects 1 output, got ",
               outputs.size());
  PG_CHECK_ARG(_kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               _kernel_shape.size());
  const Tensor &at = inputs[0];
  PG_CHECK_ARG(at.ndim() == 4,
               "Im2Col expects input to have 4 dimensions, got ", at.ndim());

  View a = pg::cuda::view::as_contiguous(at.view());
  size_t k_h = _kernel_shape[0];
  size_t k_w = _kernel_shape[1];
  size_t stride_y = _strides[0];
  size_t stride_x = _strides[1];
  size_t dilation_y = _dilation[0];
  size_t dilation_x = _dilation[1];

  size_t batch_size = a.shape()[0];
  size_t in_channels = a.shape()[1];

  size_t x_h = a.shape()[2];
  size_t x_w = a.shape()[3];

  size_t out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  outputs[0].view_ptr()->allocate();

  int total_iters = batch_size * out_h * out_w * in_channels * k_h *
                    k_w; // check kernel code for more details

  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_iters / (float)block_size);

  PG_DISPATCH_FLOATING_TYPES(a.dtype(), "im2col", [&]() {
    cuda::im2col_kernel<scalar_t><<<grid_size, block_size>>>(
        a.get_casted_base_ptr<scalar_t>(),
        outputs[0].get_casted_base_ptr<scalar_t>(), k_h, k_w, x_h, x_w,
        stride_x, stride_y, batch_size, in_channels, dilation_x, dilation_y);
  });

  PG_CUDA_KERNEL_END;
}

void Col2Im::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 1, "Col2Im expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(outputs.size() == 1, "Col2Im expects 1 output, got ",
               outputs.size());
  PG_CHECK_ARG(_kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               _kernel_shape.size());
  const Tensor &at = inputs[0];
  PG_CHECK_ARG(at.ndim() == 3,
               "Col2Im expects input to have 3 dimensions, got ", at.ndim());

  View a = pg::cuda::view::as_contiguous(at.view());
  size_t k_h = _kernel_shape[0];
  size_t k_w = _kernel_shape[1];
  size_t stride_y = _strides[0];
  size_t stride_x = _strides[1];
  size_t dilation_y = _dilation[0];
  size_t dilation_x = _dilation[1];
  size_t batch_size = a.shape()[0];
  size_t in_h = a.shape()[1];
  size_t in_w = a.shape()[2];

  size_t out_h = _output_shape[0];
  size_t out_w = _output_shape[1];
  size_t out_channels = a.shape()[1] / (k_h * k_w);

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  outputs[0].view_ptr()->allocate();
  cudaMemset(outputs[0].get_base_ptr(), 0,
             outputs[0].numel() *
                 dtype_to_size(a.dtype())); // since we'll accumulate
  int total_iters = batch_size * out_h * out_w * out_channels * k_h *
                    k_w; // check kernel code for more details

  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_iters / (float)block_size);

  PG_DISPATCH_FLOATING_TYPES(a.dtype(), "col2im", [&]() {
    cuda::col2im_kernel<scalar_t><<<grid_size, block_size>>>(
        a.get_casted_base_ptr<scalar_t>(),
        outputs[0].get_casted_base_ptr<scalar_t>(), out_channels, k_h, k_w,
        in_h, in_w, batch_size, out_h, out_w, stride_x, stride_y, dilation_x,
        dilation_y);
  });
  PG_CUDA_KERNEL_END;
}
} // namespace pg