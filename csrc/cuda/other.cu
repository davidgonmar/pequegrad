#include "./binary.cuh"
#include "./unary.cuh"
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "tensor.hpp"
#include "view_helpers.cuh"
#include <cuda.h>

namespace pg {

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

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

void CompiledPrimitive::dispatch_cuda(const std::vector<Tensor> &inputs,
                                      std::vector<Tensor> &outputs) {
  PG_CHECK_ARG(_handle != nullptr, "Function pointer is null");

  // Prepare grid and block dimensions
  dim3 threads_per_block(128, 1, 1);
  size_t num_elements = inputs[0].numel();
  dim3 blocks_per_grid(
      (num_elements + threads_per_block.x - 1) / threads_per_block.x, 1, 1);

  outputs[0].init_view(std::make_shared<View>(
      outputs[0].shape(), outputs[0].dtype(), device::CUDA));

  // Prepare kernel arguments
  std::vector<void *> kernel_args;
  for (const auto &input : inputs) {
    float *in_data = static_cast<float *>(input.get_base_ptr());
    kernel_args.push_back(&in_data);
  }
  float *out_data = static_cast<float *>(outputs[0].get_base_ptr());
  kernel_args.push_back(&out_data);

  // Convert to array of pointers
  std::vector<void *> kernel_args_ptrs;
  for (auto &arg : kernel_args) {
    kernel_args_ptrs.push_back(arg);
  }

  // Launch the kernel
  CUresult launch_result = cuLaunchKernel(
      (CUfunction)_handle, blocks_per_grid.x, blocks_per_grid.y,
      blocks_per_grid.z, threads_per_block.x, threads_per_block.y,
      threads_per_block.z, 0, NULL, kernel_args_ptrs.data(), NULL);

  if (launch_result != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(launch_result, &error_string);
    PG_CHECK_RUNTIME(false,
                     "Error launching kernel: " + std::string(error_string));
  }

  // Synchronize to ensure kernel execution is complete
  PG_CUDA_KERNEL_END;
}

} // namespace pg