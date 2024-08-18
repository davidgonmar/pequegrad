#include "./binary.cuh"
#include "./unary.cuh"
#include "ad_primitives.hpp"
#include "common/view_helpers.hpp"
#include "cuda_utils.cuh"
#include "random.cuh"
#include "tensor.hpp"
#include "view_helpers.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

// cutlass for fused linear bias relu
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/gemm.h>
namespace pg {
void FusedLinearBiasReLU::dispatch_cuda(const std::vector<Tensor> &inputs,
                                        std::vector<Tensor> &outputs) {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm75;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

  constexpr int NumStages = 2;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  auto &input = cuda::view::as_contiguous(inputs[0].view());
  auto &weight = cuda::view::as_contiguous(inputs[1].view());
  auto &bias = cuda::view::as_contiguous(inputs[2].view());
  Tensor &output = outputs[0];

  output.view_ptr()->allocate();

  int m = input.shape()[0];
  int k = input.shape()[1];
  int n = weight.shape()[1];

  PG_CHECK_RUNTIME(input.shape().size() == 2);
  PG_CHECK_RUNTIME(weight.shape().size() == 2);
  PG_CHECK_RUNTIME(bias.shape().size() == 1);
  PG_CHECK_RUNTIME(output.shape().size() == 2);

  ElementInputA *d_input = input.get_casted_base_ptr<ElementInputA>();
  ElementInputB *d_weight = weight.get_casted_base_ptr<ElementInputB>();
  ElementOutput *d_output = output.get_casted_base_ptr<ElementOutput>();
  ElementOutput *d_bias = bias.get_casted_base_ptr<ElementOutput>();

  if (input.shape()[1] != weight.shape()[0]) {
    throw std::runtime_error("Input and weight dimensions are incompatible for "
                             "matrix multiplication.");
  }

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cudaMemsetAsync(d_output, 0, m * n * sizeof(ElementOutput));
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  int split_k_slices = 1;
  typename Gemm::Arguments arguments{
      problem_size,  {d_input, k}, {d_weight, n}, {d_bias, 0},
      {d_output, n}, {alpha},      split_k_slices};

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        "Operation cannot be implemented by CUTLASS. Reason: " +
        std::string(cutlass::cutlassGetStatusString(status)));
  }
  status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("GEMM operation failed!");
  }

  PG_CUDA_KERNEL_END;
}

void Reshape::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  shape_t shape = inputs[0].shape();
  Tensor a = inputs[0];
  axes_t _new_shape = _shape_to;
  shape_t new_shape(_new_shape.size());
  size_t total_new = 1;

  int neg_pos = -1;
  for (size_t i = 0; i < _new_shape.size(); i++) {
    if (_new_shape[i] < 0) {
      PG_CHECK_ARG(
          neg_pos == -1,
          "Can only specify one unknown dimension (-1) for reshape, got ",
          neg_pos, " and ", i, " for shape ", vec_to_string(_new_shape));
      neg_pos = i;
    }
    new_shape[i] = _new_shape[i];
    total_new *= new_shape[i] == -1 ? 1 : new_shape[i];
  }

  size_t total_old =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (neg_pos != -1) {
    new_shape[neg_pos] = total_old / total_new;
    PG_CHECK_ARG(
        total_old % total_new == 0,
        "New shape is not compatible with old shape: ", vec_to_string(shape),
        " not compatible with ", vec_to_string(_new_shape));
  }
  total_new = total_old;

  if (a.is_contiguous()) {
    outputs[0].init_view(std::make_shared<View>(
        view::nocopy_reshape_nocheck(a.view(), new_shape)));
    return;
  } else {
    View cont_view = cuda::view::as_contiguous(a.view());
    outputs[0].init_view(std::make_shared<View>(
        view::nocopy_reshape_nocheck(cont_view, new_shape)));
    return;
  }
}

void AsContiguous::dispatch_cuda(const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  outputs[0].init_view(
      std::make_shared<View>(cuda::view::as_contiguous(inputs[0].view())));
}

void AsType::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(cuda::view::astype(a.view(), _dtype_to)));
}

__global__ void bilinear_resize_kernel(const float *__restrict__ input,
                                       float *__restrict__ output,
                                       int batch_size, int input_height,
                                       int input_width, int output_height,
                                       int output_width, int channels,
                                       float height_scale, float width_scale) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total_output_size = batch_size * output_height * output_width * channels;

  if (index >= total_output_size)
    return;

  int n = index / (channels * output_height * output_width);
  int c = (index / (output_height * output_width)) % channels;
  int y = (index / output_width) % output_height;
  int x = index % output_width;

  float in_x = (x + 0.5f) * width_scale - 0.5f;
  float in_y = (y + 0.5f) * height_scale - 0.5f;

  int x0 = floor(in_x);
  int x1 = min(x0 + 1, input_width - 1);
  int y0 = floor(in_y);
  int y1 = min(y0 + 1, input_height - 1);

  float x_weight = in_x - x0;
  float y_weight = in_y - y0;

  const float *input_ptr = input + (n * channels * input_height * input_width);
  float *output_ptr = output + (n * channels * output_height * output_width);

  float v0 =
      input_ptr[(c * input_height + y0) * input_width + x0] * (1 - x_weight) +
      input_ptr[(c * input_height + y0) * input_width + x1] * x_weight;
  float v1 =
      input_ptr[(c * input_height + y1) * input_width + x0] * (1 - x_weight) +
      input_ptr[(c * input_height + y1) * input_width + x1] * x_weight;
  output_ptr[(c * output_height + y) * output_width + x] =
      v0 * (1 - y_weight) + v1 * y_weight;
}

void BilinearResize::dispatch_cuda(const std::vector<Tensor> &inputs,
                                   std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &input = inputs[0];
  Tensor &output = outputs[0];
  output.view_ptr()->allocate();
  PG_CHECK_RUNTIME(input.dtype() == DType::Float32);
  PG_CHECK_RUNTIME(output.dtype() == DType::Float32);
  PG_CHECK_RUNTIME(input.shape().size() == 4);
  PG_CHECK_RUNTIME(output.shape().size() == 4);

  int batch_size = input.shape()[0];
  int input_height = input.shape()[2];
  int input_width = input.shape()[3];
  int output_height = output.shape()[2];
  int output_width = output.shape()[3];
  int channels = input.shape()[1];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;

  int total_output_size = batch_size * output_height * output_width * channels;
  int block_size = 256;
  int grid_size = (total_output_size + block_size - 1) / block_size;

  bilinear_resize_kernel<<<grid_size, block_size>>>(
      input.get_casted_base_ptr<float>(), output.get_casted_base_ptr<float>(),
      batch_size, input_height, input_width, output_height, output_width,
      channels, height_scale, width_scale);

  PG_CUDA_KERNEL_END;
}

__global__ void one_hot_kernel(const int *__restrict__ input,
                               float *__restrict__ output, int num_elements,
                               int num_classes) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_elements) {
    int category = input[index];
    if (category >= 0 && category < num_classes) {
      output[index * num_classes + category] = 1.0f;
    }
  }
}

void OneHotVector::dispatch_cuda(const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &input = inputs[0];
  Tensor &output = outputs[0];
  output.view_ptr()->allocate();
  int num_elements = input.shape()[0];
  cudaMemsetAsync(output.get_casted_base_ptr<float>(), 0,
                  num_elements * num_classes * sizeof(float));
  PG_CHECK_RUNTIME(input.dtype() == DType::Int32,
                   "Input must be of type Int32");
  PG_CHECK_RUNTIME(output.dtype() == DType::Float32,
                   "Output must be of type Float32");
  PG_CHECK_RUNTIME(input.shape().size() == 1, "Input must be 1D");
  PG_CHECK_RUNTIME(output.shape().size() == 2, "Output must be 2D");
  PG_CHECK_RUNTIME(num_elements == output.shape()[0]);
  int num_classes = output.shape()[1];

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  one_hot_kernel<<<grid_size, block_size>>>(input.get_casted_base_ptr<int>(),
                                            output.get_casted_base_ptr<float>(),
                                            num_elements, num_classes);

  PG_CUDA_KERNEL_END;
}

} // namespace pg