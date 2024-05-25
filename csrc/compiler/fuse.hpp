#pragma once
#include "ad_primitives.hpp"
#include "common.hpp"
#include <cuda.h>
#include <nvrtc.h>

namespace pg {

bool is_unary(ADPrimitive &primitive) {
  return dynamic_cast<UnaryPrimitive *>(&primitive) != nullptr;
}
bool is_binary(ADPrimitive &primitive) {
  return dynamic_cast<BinaryPrimitive *>(&primitive) != nullptr;
}

bool is_unary(Tensor &tensor) {
  return is_unary(*tensor.ad_node().primitive().get());
}
bool is_binary(Tensor &tensor) {
  return is_binary(*tensor.ad_node().primitive().get());
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

std::string render_binary(ADPrimitive &p, DType dt, std::string varname1,
                          std::string varname2) {
  PG_CHECK_RUNTIME(!is_unary(p), "Primitive is not binary: " + p.str());
  if (is<Add>(p)) {
    if (is(DType::Float32, dt)) {
      return varname1 + " + " + varname2;
    } else if (is(DType::Float64, dt)) {
      return varname1 + " + " + varname2;
    }
    PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
  } else if (is<Mul>(p)) {
    if (is(DType::Float32, dt)) {
      return varname1 + " * " + varname2;
    } else if (is(DType::Float64, dt)) {
      return varname1 + " * " + varname2;
    }
    PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
  }
  PG_CHECK_RUNTIME(false, "Unsupported binary primitive: " + p.str());
}

std::string render_dtype(DType dt) {
  if (is(DType::Float32, dt)) {
    return "float";
  } else if (is(DType::Float64, dt)) {
    return "double";
  }
  PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dt));
}

struct Expr {
  std::vector<Tensor> &inputs;
  Tensor &output;
  std::string rendered;

  Expr(std::vector<Tensor> &inputs, Tensor &output, std::string rendered)
      : inputs(inputs), output(output), rendered(rendered) {}
};

void apply_expr(Expr expr) {
  Tensor &out = expr.output;
  std::string kernel_guard = render_guard("idx_x", std::to_string(out.numel()));
  std::string kernel_body =
      kernel_guard + "out[idx_x] = " + expr.rendered + ";";
  std::string inputs_str = "";
  for (size_t i = 0; i < expr.inputs.size(); i++) {
    Tensor &t = expr.inputs[i];
    inputs_str +=
        "const " + render_dtype(t.dtype()) + " *in" + std::to_string(i);
    inputs_str += ", ";
  }
  inputs_str += render_dtype(out.dtype()) + " *out";
  std::string kernel_name = "kernel_" + std::to_string(rand());
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
  out.ad_node().set_children(expr.inputs);
}

void fuse(Tensor &out) {
  ADPrimitive &primitive = *out.ad_node().primitive().get();
  std::vector<std::string> inputs;
  std::vector<std::string> outputs = {"out"};
  if (is_unary(out)) {
    inputs.push_back("in0");
    std::string varname = "in0[idx_x]";
    std::vector<Tensor *> chain;
    Tensor *current = &out;
    while (is_unary(*current)) {
      chain.push_back(current);
      current = &current->ad_node().children()[0];
      PG_CHECK_RUNTIME(current->shape() == out.shape(),
                       "[fuse_unary_chain] Shape mismatch: " + current->str() +
                           " vs " + out.str());
      PG_CHECK_RUNTIME(current->dtype() == out.dtype(),
                       "[fuse_unary_chain] DType mismatch: " + current->str() +
                           " vs " + out.str());
    }
    // Step 2: Reverse the chain (for example, if the chain is a.log().exp(),
    // out.primitive = Exp, and current chain is [Exp, Log]) So we want to
    // reverse it to [Log, Exp]
    std::reverse(chain.begin(), chain.end());
    // Step 3: Render the chain
    DType dt = out.dtype();
    for (Tensor *t : chain) {
      ADPrimitive &p = *t->ad_node().primitive().get();
      if (is_unary(p)) {
        std::string expr = render_unary(p, dt, varname);
        varname = expr;
      } else {
        PG_CHECK_RUNTIME(false, "[fuse_unary_chain] Primitive is not unary: " +
                                    p.str());
      }
    }
    std::vector<Tensor> inputs = {*current};
    Expr e(inputs, out, varname);
    apply_expr(e);
  } else if (is_binary(out)) {
    std::string varname1 = "in0[idx_x]";
    std::string varname2 = "in1[idx_x]";
    inputs.push_back("in0");
    inputs.push_back("in1");

    std::vector<Tensor> inputs = {out.ad_node().children()[0],
                                  out.ad_node().children()[1]};

    int i = 0;
    for (Tensor &t : inputs) {
      if (is_unary(t)) {
        std::vector<Tensor *> chain;
        Tensor *current = &t;
        while (is_unary(*current)) {
          chain.push_back(current);
          current = &current->ad_node().children()[0];
          PG_CHECK_RUNTIME(current->shape() == out.shape(),
                           "[fuse_unary_chain] Shape mismatch: " +
                               current->str() + " vs " + out.str());
          PG_CHECK_RUNTIME(current->dtype() == out.dtype(),
                           "[fuse_unary_chain] DType mismatch: " +
                               current->str() + " vs " + out.str());
        }
        // Step 2: Reverse the chain (for example, if the chain is
        // a.log().exp(), out.primitive = Exp, and current chain is [Exp, Log])
        // So we want to reverse it to [Log, Exp]
        std::reverse(chain.begin(), chain.end());
        // Step 3: Render the chain
        DType dt = out.dtype();
        for (Tensor *t : chain) {
          ADPrimitive &p = *t->ad_node().primitive().get();
          if (is_unary(p)) {
            std::string expr =
                render_unary(p, dt, i == 0 ? varname1 : varname2);
            if (i == 0) {
              varname1 = expr;
            } else {
              varname2 = expr;
            }
          } else {
            PG_CHECK_RUNTIME(
                false, "[fuse_unary_chain] Primitive is not unary: " + p.str());
          }
        }
        inputs[i] = *current;
      }
      i++;
    }

    std::string expr =
        render_binary(primitive, out.dtype(), varname1, varname2);
    Expr e(inputs, out, expr);
    apply_expr(e);
  } else {
    // check recursively until we find a unary or binary primitive
    for (Tensor &t : out.ad_node().children()) {
      fuse(t);
    }
  }
}

} // namespace pg