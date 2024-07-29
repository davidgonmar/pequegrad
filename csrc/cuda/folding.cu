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

  // set stream to 0
  PG_CHECK_CUDNN(cudnnSetStream(handle, 0));
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

  shape_t dilation = this->dilation;
  shape_t strides = this->strides;
  shape_t padding = this->padding;
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
      conv_desc, padding[0], padding[1], strides[0], strides[1], dilation[0],
      dilation[1], CUDNN_CROSS_CORRELATION,
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
  cudaMallocAsync(&workspace, workspace_size, 0);
  float alpha = 1.0f, beta = 0.0f;
  PG_CHECK_CUDNN(cudnnConvolutionForward(
      handle, &alpha, input_desc, input.get_base_ptr(), filter_desc,
      weight.get_base_ptr(), conv_desc, perfResults.algo, workspace,
      workspace_size, &beta, output_desc, output->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  PG_CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
  PG_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  PG_CHECK_CUDNN(cudnnDestroy(handle));
  cudaFreeAsync(workspace, 0);
}

// backward pass FOR THE WEIGHTS
void CudnnConv2dVjpWeight::dispatch_cuda(const std::vector<Tensor> &inputs,
                                         std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 2,
               "CudnnConv2dVjpWeight expects 2 inputs, got ", inputs.size());
  auto &input = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &output_grad = pg::cuda::view::as_contiguous(inputs[1].view());
  auto &weight_grad = outputs[0].view_ptr();
  weight_grad->allocate();
  PG_CHECK_ARG(input.ndim() == 4, "Input tensor must have 4 dimensions, got ",
               input.ndim());
  PG_CHECK_ARG(output_grad.ndim() == 4,
               "Output tensor must have 4 dimensions, got ",
               output_grad.ndim());
  PG_CHECK_ARG(weight_grad->ndim() == 4,
               "Weight_grad tensor must have 4 dimensions, got ",
               weight_grad->ndim());
  PG_CHECK_RUNTIME(input.dtype() == output_grad.dtype() &&
                       input.dtype() == DType::Float32,
                   "Input and output_grad tensors must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc, output_grad_desc;
  cudnnFilterDescriptor_t weight_grad_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  cudnnSetStream(handle, 0);
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  int batch_size = input.shape()[0];
  int in_channels = input.shape()[1];
  int in_h = input.shape()[2];
  int in_w = input.shape()[3];
  int out_channels = output_grad.shape()[1];
  int k_h = weight_grad->shape()[2];
  int k_w = weight_grad->shape()[3];
  int out_h = output_grad.shape()[2];
  int out_w = output_grad.shape()[3];

  shape_t dilation = this->dilation;
  shape_t strides = this->strides;
  shape_t padding = this->padding;
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_grad_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, out_channels,
                                            in_channels, k_h, k_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            out_channels, out_h, out_w));
  PG_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, padding[0], padding[1], strides[0], strides[1], dilation[0],
      dilation[1], CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT)); // usually what is called 'convolution' in dl
                          // frameworks is actually cross-correlation
  // Select an algorithm for convolution
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perfResults;

  PG_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      handle, input_desc, output_grad_desc, conv_desc, weight_grad_desc, 1,
      &returned_algo_count, &perfResults));

  size_t workspace_size;
  void *workspace;
  // Determine workspace size
  PG_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc, output_grad_desc, conv_desc, weight_grad_desc,
      perfResults.algo, &workspace_size));

  // Allocate workspace
  cudaMallocAsync(&workspace, workspace_size, 0);
  float alpha = 1.0f, beta = 0.0f;
  PG_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      handle, &alpha, input_desc, input.get_base_ptr(), output_grad_desc,
      output_grad.get_base_ptr(), conv_desc, perfResults.algo, workspace,
      workspace_size, &beta, weight_grad_desc, weight_grad->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  PG_CHECK_CUDNN(cudnnDestroy(handle));
  cudaFreeAsync(workspace, 0);
}

void CudnnConv2dVjpInput::dispatch_cuda(const std::vector<Tensor> &inputs,
                                        std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 2, "CudnnConv2dVjpInput expects 2 inputs, got ",
               inputs.size());
  auto &weight = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &output_grad = pg::cuda::view::as_contiguous(inputs[1].view());
  auto &input_grad = outputs[0].view_ptr();
  input_grad->allocate();
  PG_CHECK_ARG(weight.ndim() == 4, "Weight tensor must have 4 dimensions, got ",
               inputs[0].ndim());
  PG_CHECK_ARG(output_grad.ndim() == 4,
               "Output tensor must have 4 dimensions, got ",
               output_grad.ndim());
  PG_CHECK_ARG(input_grad->ndim() == 4,
               "Input_grad tensor must have 4 dimensions, got ",
               input_grad->ndim());
  PG_CHECK_RUNTIME(weight.dtype() == output_grad.dtype() &&
                       weight.dtype() == DType::Float32,
                   "Weight and output_grad tensors must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_grad_desc, output_grad_desc;
  cudnnFilterDescriptor_t weight_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  cudnnSetStream(handle, 0);
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
  PG_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  int batch_size = output_grad.shape()[0];
  int in_channels = input_grad->shape()[1];
  int in_h = input_grad->shape()[2];
  int in_w = input_grad->shape()[3];
  int out_channels = output_grad.shape()[1];
  int k_h = weight.shape()[2];
  int k_w = weight.shape()[3];
  int out_h = output_grad.shape()[2];
  int out_w = output_grad.shape()[3];

  shape_t dilation = this->dilation;
  shape_t strides = this->strides;
  shape_t padding = this->padding;
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, out_channels,
                                            in_channels, k_h, k_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            out_channels, out_h, out_w));
  PG_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, padding[0], padding[1], strides[0], strides[1], dilation[0],
      dilation[1], CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT)); // usually what is called 'convolution' in dl
                          // frameworks is actually cross-correlation
  // Select an algorithm for convolution
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perfResults;

  PG_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      handle, weight_desc, output_grad_desc, conv_desc, input_grad_desc, 1,
      &returned_algo_count, &perfResults));

  size_t workspace_size;
  void *workspace;
  // Determine workspace size
  PG_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, weight_desc, output_grad_desc, conv_desc, input_grad_desc,
      perfResults.algo, &workspace_size));

  // Allocate workspace
  cudaMallocAsync(&workspace, workspace_size, 0);
  float alpha = 1.0f, beta = 0.0f;
  PG_CHECK_CUDNN(cudnnConvolutionBackwardData(
      handle, &alpha, weight_desc, weight.get_base_ptr(), output_grad_desc,
      output_grad.get_base_ptr(), conv_desc, perfResults.algo, workspace,
      workspace_size, &beta, input_grad_desc, input_grad->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  // Cleanup
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_desc));
  PG_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  PG_CHECK_CUDNN(cudnnDestroy(handle));
  cudaFreeAsync(workspace, 0);
}

void CudnnPooling2D::dispatch_cuda(const std::vector<Tensor> &inputs,
                                   std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 1, "CudnnPooling2d expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(outputs.size() == 1, "CudnnPooling2d expects 1 output, got ",
               outputs.size());
  auto &input = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &output = outputs[0].view_ptr();
  output->allocate();
  PG_CHECK_ARG(input.ndim() == 4, "Input tensor must have 4 dimensions, got ",
               input.ndim());
  PG_CHECK_ARG(output->ndim() == 4,
               "Output tensor must have 4 dimensions, got ", output->ndim());
  PG_CHECK_RUNTIME(input.dtype() == DType::Float32,
                   "Input tensor must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnPoolingDescriptor_t pooling_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  PG_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));

  int batch_size = input.shape()[0];
  int in_channels = input.shape()[1];
  int in_h = input.shape()[2];
  int in_w = input.shape()[3];
  int out_h = output->shape()[2];
  int out_w = output->shape()[3];

  // pads are always 0
  shape_t strides = this->strides;
  shape_t kernel_shape = this->kernel_shape;
  shape_t padding = {0, 0};

  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, out_h, out_w));
  PG_CHECK_CUDNN(cudnnSetPooling2dDescriptor(
      pooling_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, kernel_shape[0],
      kernel_shape[1], padding[0], padding[1], strides[0], strides[1]));

  float alpha = 1.0f, beta = 0.0f;
  PG_CHECK_CUDNN(cudnnPoolingForward(handle, pooling_desc, &alpha, input_desc,
                                     input.get_base_ptr(), &beta, output_desc,
                                     output->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  // Cleanup
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  PG_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
  PG_CHECK_CUDNN(cudnnDestroy(handle));
}

void CudnnLRN::dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 1, "CudnnLRN expects 1 input, got ",
               inputs.size());
  PG_CHECK_ARG(outputs.size() == 1, "CudnnLRN expects 1 output, got ",
               outputs.size());
  auto &input = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &output = outputs[0].view_ptr();
  output->allocate();
  PG_CHECK_ARG(input.ndim() == 4, "Input tensor must have 4 dimensions, got ",
               input.ndim());
  PG_CHECK_ARG(output->ndim() == 4,
               "Output tensor must have 4 dimensions, got ", output->ndim());
  PG_CHECK_RUNTIME(input.dtype() == DType::Float32,
                   "Input tensor must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnLRNDescriptor_t lrn_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  cudnnSetStream(handle, 0);
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  PG_CHECK_CUDNN(cudnnCreateLRNDescriptor(&lrn_desc));

  int batch_size = input.shape()[0];
  int in_channels = input.shape()[1];
  int in_h = input.shape()[2];
  int in_w = input.shape()[3];

  // pads are always 0
  int size = this->size;
  float alpha = this->alpha;
  float beta = this->beta;
  float k = this->k;

  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetLRNDescriptor(lrn_desc, size, alpha, beta, k));

  float alpha_ = 1.0f, beta_ = 0.0f;
  PG_CHECK_CUDNN(cudnnLRNCrossChannelForward(
      handle, lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha_, input_desc,
      input.get_base_ptr(), &beta_, output_desc, output->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  // Cleanup
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  PG_CHECK_CUDNN(cudnnDestroyLRNDescriptor(lrn_desc));

  PG_CHECK_CUDNN(cudnnDestroy(handle));
}

void CudnnLRNVjpInput::dispatch_cuda(const std::vector<Tensor> &inputs,
                                     std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(inputs.size() == 3, "CudnnLRNVjpInput expects 2 inputs, got ",
               inputs.size());
  auto &forward_output = pg::cuda::view::as_contiguous(inputs[0].view());
  auto &output_grad = pg::cuda::view::as_contiguous(inputs[1].view());
  auto &input = pg::cuda::view::as_contiguous(inputs[2].view());
  auto &input_grad = outputs[0].view_ptr();

  input_grad->allocate();

  PG_CHECK_ARG(forward_output.ndim() == 4,
               "Forward output tensor must have 4 dimensions, got ",
               forward_output.ndim());
  PG_CHECK_ARG(output_grad.ndim() == 4,
               "Output tensor must have 4 dimensions, got ",
               output_grad.ndim());
  PG_CHECK_ARG(input.ndim() == 4, "Input tensor must have 4 dimensions, got ",
               input.ndim());
  PG_CHECK_ARG(input_grad->ndim() == 4,
               "Input_grad tensor must have 4 dimensions, got ",
               input_grad->ndim());
  PG_CHECK_RUNTIME(
      forward_output.dtype() == output_grad.dtype() &&
          forward_output.dtype() == DType::Float32,
      "Forward output and output_grad tensors must have dtype float32");

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t forward_output_desc, output_grad_desc, input_desc,
      input_grad_desc;
  cudnnLRNDescriptor_t lrn_desc;

  PG_CHECK_CUDNN(cudnnCreate(&handle));
  cudnnSetStream(handle, 0);
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&forward_output_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  PG_CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_grad_desc));
  PG_CHECK_CUDNN(cudnnCreateLRNDescriptor(&lrn_desc));

  int batch_size = forward_output.shape()[0];
  int in_channels = forward_output.shape()[1];
  int in_h = forward_output.shape()[2];
  int in_w = forward_output.shape()[3];

  // pads are always 0
  int size = this->size;
  float alpha = this->alpha;
  float beta = this->beta;
  float k = this->k;

  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      forward_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size,
      in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, batch_size,
                                            in_channels, in_h, in_w));
  PG_CHECK_CUDNN(cudnnSetLRNDescriptor(lrn_desc, size, alpha, beta, k));

  float alpha_ = 1.0f, beta_ = 0.0f;
  PG_CHECK_CUDNN(cudnnLRNCrossChannelBackward(
      handle, lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha_,
      forward_output_desc, forward_output.get_base_ptr(), forward_output_desc,
      output_grad.get_base_ptr(), input_desc, input.get_base_ptr(), &beta_,
      input_grad_desc, input_grad->get_base_ptr()));
  PG_CUDA_KERNEL_END;
  // Cleanup
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(forward_output_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  PG_CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_grad_desc));
  PG_CHECK_CUDNN(cudnnDestroyLRNDescriptor(lrn_desc));

  PG_CHECK_CUDNN(cudnnDestroy(handle));
}

} // namespace pg