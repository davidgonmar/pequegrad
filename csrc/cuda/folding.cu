#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "folding.cuh"
#include "view_helpers.cuh"

// cudnn
#include <cudnn.h>
#include <cudnn_cnn.h>

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

#define PG_CHECK_CUDNN(expression)                                             \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    std::string errstring = cudnnGetErrorString(status);                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      PG_CHECK_RUNTIME(false, "CUDNN Error at line ", __LINE__, ": ",          \
                       errstring);                                             \
    }                                                                          \
  }

void CudnnConv2D::dispatch_cuda(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 2, "CudnnConv2d expects 2 inputs, got ",
               inputs.size());
  auto &input = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &weight = pg::cuda::view::as_contiguous(inputs[1].view());
  auto &output = outputs[0].view_ptr();
  output->allocate();
  PG_CHECK_ARG(input.ndim() == 4, "Input tensor must have 4 dimensions, got ",
               input.ndim());
  PG_CHECK_ARG(weight.ndim() == 4, "Weight tensor must have 4 dimensions, got ",
               weight.ndim());
  PG_CHECK_ARG(output->ndim() == 4,
               "Output tensor must have 4 dimensions, got ", output->ndim());
  PG_CHECK_RUNTIME(input.dtype() == weight.dtype() &&
                       input.dtype() == DType::Float32,
                   "Input and weight tensors must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  PG_CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  PG_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  int batch_size = input.shape()[0];
  int in_channels = input.shape()[1];
  int in_h = input.shape()[2];
  int in_w = input.shape()[3];
  int out_channels = weight.shape()[0];
  int k_h = weight.shape()[2];
  int k_w = weight.shape()[3];
  int out_h = output->shape()[2];
  int out_w = output->shape()[3];

  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, out_channels,
                                            in_channels, k_h, k_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            out_channels, out_h, out_w));
  PG_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT)); // usually what is called 'convolution' in dl
                          // frameworks is actually cross-correlation

  // Select an algorithm for convolution
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults;
  PG_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
      handle, input_desc, filter_desc, conv_desc, output_desc, 1,
      &returned_algo_count, &perfResults));

  size_t workspace_size;
  void *workspace;
  // Determine workspace size
  PG_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      handle, input_desc, filter_desc, conv_desc, output_desc, perfResults.algo,
      &workspace_size));

  // Allocate workspace
  cudaMalloc(&workspace, workspace_size);

  float alpha = 1.0f, beta = 0.0f;
  PG_CHECK_CUDNN(cudnnConvolutionForward(
      handle, &alpha, input_desc, input.get_base_ptr(), filter_desc,
      weight.get_base_ptr(), conv_desc, perfResults.algo, workspace,
      workspace_size, &beta, output_desc, output->get_base_ptr()));

  cudaDeviceSynchronize();
  // Cleanup
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  PG_CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
  PG_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  PG_CHECK_CUDNN(cudnnDestroy(handle));
  cudaFree(workspace);

  // Sync
  cudaDeviceSynchronize();
}

} // namespace pg