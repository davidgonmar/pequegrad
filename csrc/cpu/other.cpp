#include "ad_primitives.hpp"
#include "common/view_helpers.hpp"
#include "compiler/expr.hpp"
#include "cpu/view_helpers.hpp"
#include "dtype.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <cstdlib>
#ifdef __linux__
#include <dlfcn.h> // for dynamic loading on Unix
#endif
#ifdef _WIN32
#include <Windows.h>      // for dynamic loading on Windows
#include <libloaderapi.h> // for LoadLibrary and GetProcAddress on Windows
#endif
#include <fstream>

namespace pg {
void Reshape::dispatch_cpu(const std::vector<Tensor> &inputs,
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
    View cont_view = cpu::view::as_contiguous(a.view());
    outputs[0].init_view(std::make_shared<View>(
        view::nocopy_reshape_nocheck(cont_view, new_shape)));
    return;
  }
}

void AsContiguous::dispatch_cpu(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  outputs[0].init_view(
      std::make_shared<View>(cpu::view::as_contiguous(inputs[0].view(), true)));
}

void AsType::dispatch_cpu(const std::vector<Tensor> &inputs,
                          std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  outputs[0].init_view(
      std::make_shared<View>(cpu::view::astype(inputs[0].view(), _dtype_to)));
}

static std::string dtype_to_cpp_string(DType dtype) {
  return dtype == DType::Float32   ? "float"
         : dtype == DType::Float64 ? "double"
                                   : "int";
}

void CompiledPrimitive::dispatch_cpu(const std::vector<Tensor> &inputs,
                                     std::vector<Tensor> &outputs) {
  outputs[0].init_view(std::make_shared<View>(outputs[0].shape(),
                                              outputs[0].dtype(), device::CPU));
  if (this->fn_ptr == nullptr) {
    std::vector<std::shared_ptr<AstLoadExpr>> inputs_ast = get_leafs(ast);
    PG_CHECK_RUNTIME(
        inputs.size() == inputs_ast.size(),
        "Number of inputs does not match number of AST inputs, got ",
        inputs.size(), " and ", inputs_ast.size());
    for (size_t i = 0; i < inputs_ast.size(); i++) {
      inputs_ast[i]->strides = inputs[i].strides();
    }

    // Check we can cast to store
    PG_CHECK_RUNTIME(std::dynamic_pointer_cast<AstStoreExpr>(ast) != nullptr,
                     "AST is not a store expression");
    auto store = std::dynamic_pointer_cast<AstStoreExpr>(ast);
    store->shape = outputs[0].shape();
    store->strides = outputs[0].strides();
    store->propagate_movement_ops();
    std::string x = store->render_idxs() + store->render();
    std::string loop_inner = "size_t idx = i;\n" + x;
    std::string kernel_name =
        "kernel_" + std::to_string(reinterpret_cast<size_t>(this));

    // now render the CPU function
    std::string func = "#include <math.h>\n"
                       "extern \"C\" {\n"
                       "using size_t = unsigned long long;\n"
                       "void __declspec(dllexport)" +
                       kernel_name + "(";
    // for each input, render dtype and name
    for (size_t i = 0; i < inputs.size(); i++) {
      func += dtype_to_cpp_string(inputs[i].dtype()) + " *" +
              inputs_ast[i]->name + ",\n";
    }

    func += dtype_to_cpp_string(outputs[0].dtype()) + " *out) {\n";
    func += "#pragma omp parallel for\n";
    func += "for (size_t i = 0; i < " + std::to_string(outputs[0].numel()) +
            "; ++i) {\n";
    func += loop_inner + "\n";
    func += "}\n";
    func += "}\n";
    func += "}\n";
    // if PG_KERNEL_DB is set, print code
    if (getenv("PG_KERNEL_DB") != nullptr) {
      std::cout << "Generated CPU code for kernel " << kernel_name << ":\n";
      std::cout << func << std::endl;
    }

    // compile and load the function
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);
    std::string file_name = std::string(tempPath) + kernel_name + ".cpp";
    std::ofstream out(file_name, std::ios::out | std::ios::trunc);
    if (out.is_open()) {
      out << func;
      out.flush();
      out.close();
    } else {
      PG_CHECK_RUNTIME(false, "Failed to open file for writing: " + file_name);
    }
    std::string compile_command =
        "clang++ -O3 -march=native -m64 -shared -o " + std::string(tempPath) +
        kernel_name + ".dll " + file_name +
        " -fopenmp > nul"; // Compile in the Windows temp folder
    int compile_result = std::system(compile_command.c_str());
    PG_CHECK_RUNTIME(compile_result == 0,
                     "Compilation failed for kernel: " + kernel_name);
    HMODULE handle =
        LoadLibraryA((std::string(tempPath) + kernel_name + ".dll").c_str());
    PG_CHECK_RUNTIME(handle != nullptr,
                     "Failed to load DLL for kernel: " + kernel_name +
                         ", errors: " + std::to_string(GetLastError()));
    void *func_ptr = GetProcAddress(handle, kernel_name.c_str());
    PG_CHECK_RUNTIME(func_ptr != nullptr,
                     "Failed to get function pointer for kernel: " +
                         kernel_name);

    this->fn_ptr = func_ptr;
    this->_cuda_code = func;
    this->_name = kernel_name;
  }

  // prepare args
  std::vector<void *> kernel_args;
  for (const auto &input : inputs) {
    void *in_data = input.get_base_ptr();
    kernel_args.push_back(in_data);
  }
  void *out_data = outputs[0].get_base_ptr();
  kernel_args.push_back(out_data);

  // cast fn back
  using KernelFunc = void (*)(...);
  KernelFunc kernel_func = reinterpret_cast<KernelFunc>(this->fn_ptr);

  switch (kernel_args.size()) {
  case 1:
    kernel_func(kernel_args[0]);
    break;
  case 2:
    kernel_func(kernel_args[0], kernel_args[1]);
    break;
  case 3:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2]);
    break;
  case 4:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3]);
    break;
  case 5:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4]);
    break;
  case 6:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4], kernel_args[5]);
  case 7:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4], kernel_args[5], kernel_args[6]);
    break;
  case 8:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4], kernel_args[5], kernel_args[6], kernel_args[7]);
    break;
  case 9:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4], kernel_args[5], kernel_args[6], kernel_args[7],
                kernel_args[8]);
    break;
  case 10:
    kernel_func(kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3],
                kernel_args[4], kernel_args[5], kernel_args[6], kernel_args[7],
                kernel_args[8], kernel_args[9]);
    break;
  default:
    PG_CHECK_RUNTIME(false, "Too many arguments for kernel " + this->_name);
  }
}

} // namespace pg
