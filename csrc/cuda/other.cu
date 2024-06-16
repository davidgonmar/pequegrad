#include "./binary.cuh"
#include "./unary.cuh"
#include "ad_primitives.hpp"
#include "compiler/expr.hpp"
#include "cuda_utils.cuh"
#include "tensor.hpp"
#include "view_helpers.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

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

std::string dtype_to_cpp_string(DType dtype) {
  return dtype == DType::Float32   ? "float"
         : dtype == DType::Float64 ? "double"
                                   : "int";
}

void CompiledPrimitive::dispatch_cuda(const std::vector<Tensor> &inputs,
                                      std::vector<Tensor> &outputs) {
  if (this->fn_ptr == nullptr) {
    // first get the inputs of ast
    std::vector<std::shared_ptr<AstLoadExpr>> inputs_ast = get_leafs(ast);
    // for each input, set its strides to the strides of the corresponding
    // tensor
    PG_CHECK_RUNTIME(
        inputs.size() == inputs_ast.size(),
        "Number of inputs does not match number of AST inputs, got ",
        inputs.size(), " and ", inputs_ast.size());
    for (size_t i = 0; i < inputs_ast.size(); i++) {
      inputs_ast[i]->strides = inputs[i].strides();
    }

    // assert that ast is a store expr and set its strides and shape
    outputs[0].init_view(std::make_shared<View>(
        outputs[0].shape(), outputs[0].dtype(), device::CUDA));

    // check we can cast to store
    PG_CHECK_RUNTIME(std::dynamic_pointer_cast<AstStoreExpr>(ast) != nullptr,
                     "AST is not a store expression");
    auto store = std::dynamic_pointer_cast<AstStoreExpr>(ast);
    store->shape = outputs[0].shape();
    store->strides = outputs[0].strides();
    store->propagate_movement_ops();
    std::string x = store->render_idxs() + store->render();
    std::string ker_inner =
        "size_t idx = blockDim.x * blockIdx.x + threadIdx.x;\n";
    ker_inner +=
        "if (idx >= " + std::to_string(outputs[0].numel()) + ") return;\n";
    ker_inner += x;

    // now render the kernel
    std::string ker =
        "__global__ void kernel_" + std::to_string(outputs[0].id) + "(";
    std::string kernel_name = "kernel_" + std::to_string(outputs[0].id);
    // for each input, render dtype and name
    for (size_t i = 0; i < inputs.size(); i++) {
      ker += dtype_to_cpp_string(inputs[i].dtype()) + " *" +
             inputs_ast[i]->name + ",\n";
    }

    ker += dtype_to_cpp_string(outputs[0].dtype()) + " *out) {\n" + ker_inner +
           "\n}";

    nvrtcProgram prog;
    // apend extern C
    std::string file = "extern \"C\" {\n" + ker + "\n}";
    nvrtcCreateProgram(&prog, file.c_str(), nullptr, 0, nullptr, nullptr);

    if (std::getenv("PG_KERNEL_DB") != nullptr) {
      std::cout << "file: " << file << std::endl;
    }
    const char *opts[] = {"--use_fast_math"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

    // Check for compilation errors
    if (compileResult != NVRTC_SUCCESS) {
      size_t logSize;
      nvrtcGetProgramLogSize(prog, &logSize);
      char *log = new char[logSize];
      nvrtcGetProgramLog(prog, log);
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error("NVRTC compilation failed: " + std::string(log));
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    CUmodule cuModule;
    CUfunction cuFunction;
    CUcontext cuContext;
    CUresult R1 = cuModuleLoadData(&cuModule, ptx);
    PG_CHECK_RUNTIME(R1 == CUDA_SUCCESS,
                     "Failed to load data: got " + std::to_string(R1));
    CUresult R =
        cuModuleGetFunction(&cuFunction, cuModule, kernel_name.c_str());
    PG_CHECK_RUNTIME(R == CUDA_SUCCESS, "Failed to get function: got " +
                                            std::to_string(R) + " for kernel " +
                                            kernel_name);

    PG_CHECK_RUNTIME(cuFunction != nullptr, "Failed to get function");
    // Store the function pointer in a void*
    void *function_ptr = reinterpret_cast<void *>(cuFunction);
    PG_CHECK_RUNTIME(function_ptr != nullptr, "Failed to get function pointer");
    // Clean up
    nvrtcDestroyProgram(&prog);
    delete[] ptx;
    this->fn_ptr = reinterpret_cast<void *>(cuFunction);
    this->_cuda_code = ker;
    this->_name = kernel_name;
  }
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
    void *in_data = input.get_base_ptr();
    kernel_args.push_back(in_data);
  }
  void *out_data = outputs[0].get_base_ptr();
  kernel_args.push_back(out_data);

  // Convert to array of pointers
  std::vector<void *> kernel_args_ptrs;
  for (auto &arg : kernel_args) {
    kernel_args_ptrs.push_back(&arg);
  }

  // Launch the kernel
  // create stream to launch kernel
  // first check if function is valid

  CUresult launch_result = cuLaunchKernel(
      (CUfunction)this->fn_ptr, blocks_per_grid.x, blocks_per_grid.y,
      blocks_per_grid.z, threads_per_block.x, threads_per_block.y,
      threads_per_block.z, 0, NULL, kernel_args_ptrs.data(), NULL);

  if (launch_result != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(launch_result, &error_string);
    PG_CHECK_RUNTIME(
        false, "Error launching kernel: " + std::string(error_string) +
                   " "
                   "for kernel " +
                   std::to_string(outputs[0].id) + " with code \n" +
                   this->_cuda_code + "\n Launched with: \n" +
                   vec_to_string(kernel_args_ptrs) +
                   "\n and fn_ptr: " + std::to_string((size_t)this->fn_ptr));
  }

  // Synchronize to ensure kernel execution is complete
  PG_CUDA_KERNEL_END;

  // cache
}

} // namespace pg