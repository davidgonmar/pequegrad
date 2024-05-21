#pragma once
#include "ad_primitives.hpp"
#include "common.hpp"
#include <cuda.h>
#include <nvrtc.h>

namespace pg {
bool is_unary(ADPrimitive &primitive) {
  return dynamic_cast<UnaryPrimitive *>(&primitive) != nullptr;
}

bool is_unary(Tensor &tensor) {
  return is_unary(*tensor.ad_node().primitive().get());
}

template <typename Other> bool is(ADPrimitive &primitive) {
  return typeid(primitive) == typeid(Other);
}

bool is(DType dt1, DType dt2) { return dt1 == dt2; }

std::string render_unary(ADPrimitive &p, DType dt, std::string varname) {
  PG_CHECK_RUNTIME(is_unary(p), "Primitive is not unary: " + p.str());
  if (is<Log>(p)) {
    if (is(DType::Float32, dt)) {
      return "logf(" + varname + ")";
    } else if (is(DType::Float64, dt)) {
      return "log(" + varname + ")";
    }
    PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
  } else if (is<Exp>(p)) {
    if (is(DType::Float32, dt)) {
      return "expf(" + varname + ")";
    } else if (is(DType::Float64, dt)) {
      return "exp(" + varname + ")";
    }
    PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
  }
  PG_CHECK_RUNTIME(false, "Unsupported unary primitive: " + p.str());
}

std::string render_dtype(DType dt) {
  if (is(DType::Float32, dt)) {
    return "float";
  } else if (is(DType::Float64, dt)) {
    return "double";
  }
  PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
}

void fuse_unary(Tensor &out) {
  ADPrimitive &primitive = *out.ad_node().primitive().get();
  bool out_is_unary = is_unary(out);
  if (!out_is_unary) {
    return;
  }

  std::string varname = "in[idx_x]";
  std::string inputs_str = "const " + render_dtype(out.dtype()) + " *in, " +
                           render_dtype(out.dtype()) + " *out";
  std::string expr = render_unary(primitive, out.dtype(), varname);
  // if parent is also unary, render it also
  Tensor &child = out.ad_node().children()[0];
  while (is_unary(child)) {
    std::cout << "Child is unary: " << child.str() << "\n";
    ADPrimitive &child_primitive = *child.ad_node().primitive().get();
    expr = render_unary(child_primitive, out.dtype(), expr);
    child = child.ad_node().children()[0];
  }
  std::string kernel_guard = render_guard("idx_x", std::to_string(out.numel()));
  std::string kernel_body = kernel_guard + "out[idx_x] = " + expr + ";\n";
  std::string kernel_name = "unary_" + primitive.str();
  std::string kernel_code = get_x_gid() + kernel_body;
  std::string file = render_kernel_file(kernel_name, kernel_code, inputs_str);
  // std::cout << " Kernel file: " << file << "\n";
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, file.c_str(), nullptr, 0, nullptr, nullptr);

  nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, nullptr);

  // Check for compilation errors
  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);

    std::cerr << "NVRTC compilation failed:\n" << log << std::endl;
    delete[] log;

    nvrtcDestroyProgram(&prog);
    return;
  }

  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char *ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);

  CUmodule cuModule;
  CUfunction cuFunction;
  cuInit(0);
  CUcontext cuContext;
  CUresult R0 = cuCtxCreate(&cuContext, 0, 0);
  PG_CHECK_RUNTIME(R0 == CUDA_SUCCESS,
                   "Failed to create context: got " + std::to_string(R0));
  CUresult R1 = cuModuleLoadData(&cuModule, ptx);
  PG_CHECK_RUNTIME(R1 == CUDA_SUCCESS,
                   "Failed to load data: got " + std::to_string(R1));
  CUresult R = cuModuleGetFunction(&cuFunction, cuModule, kernel_name.c_str());
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

  CompiledPrimitive compiled(kernel_name, function_ptr, file);

  out.ad_node().set_primitive(std::make_shared<CompiledPrimitive>(compiled));
  out.ad_node().set_children({child});
}

} // namespace pg