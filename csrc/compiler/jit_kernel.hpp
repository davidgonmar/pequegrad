#include "utils.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdexcept>
#include <string>
#include <vector>

// We keep a global kernel database of all the kernels that have been compiled
// Avoids recompiling the same kernel multiple times
class KernelDatabase {
private:
  std::unordered_map<std::string, void *> _kernels;
  KernelDatabase() {} // private constructor to prevent instantiation

public:
  // Delete copy constructor and assignment operator to prevent copying
  KernelDatabase(const KernelDatabase &) = delete;
  KernelDatabase &operator=(const KernelDatabase &) = delete;

  static KernelDatabase &get_instance() {
    static KernelDatabase instance;
    return instance;
  }

  void add_kernel(const std::string &src, void *kernel) {
    _kernels[src] = kernel;
  }

  void *get_kernel(const std::string &src) {
    if (_kernels.find(src) == _kernels.end()) {
      return nullptr;
    }
    return _kernels[src];
  }
};

class AbstractKernel {
public:
  virtual void launch(std::vector<void *> &args) = 0;
  virtual void compile() = 0;
};

class CudaKernel : public AbstractKernel {
private:
  CUfunction _func;
  dim3 _blocks_per_grid;
  dim3 _threads_per_block;
  std::string _name;
  std::string _src;

  void compile() {
    // put cuda_fp16 into a string
    nvrtcProgram prog;
    // extern C to avoid name mangling
    std::string file = "#include <cuda_fp16.h>\n";
    // add fp16 max overloads
    file += "__device__ half max(half a, half b) { return a > b ? a : b; }\n";
    file += "\nextern \"C\" {\n" + _src + "\n};";
    nvrtcCreateProgram(&prog, file.c_str(), nullptr, 0, nullptr, nullptr);
    // fast math

    const char *opts[] = {"--use_fast_math", "-arch=sm_70",
                          "--include-path=C:\\Program Files\\NVIDIA GPU "
                          "Computing Toolkit\\CUDA\\v12.1\\include"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 3, opts);
    // Check for compilation errors
    if (compileResult != NVRTC_SUCCESS) {
      size_t logSize;
      nvrtcGetProgramLogSize(prog, &logSize);
      char *log = new char[logSize];
      nvrtcGetProgramLog(prog, log);
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error("NVRTC compilation failed: " + std::string(log) +
                               " kernel source: " + _src);
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
    CUresult R = cuModuleGetFunction(&cuFunction, cuModule, _name.c_str());
    PG_CHECK_RUNTIME(R == CUDA_SUCCESS, "Failed to get function: got " +
                                            std::to_string(R) + " for kernel " +
                                            _name);

    PG_CHECK_RUNTIME(cuFunction != nullptr, "Failed to get function");
    // Store the function pointer in a void*
    void *function_ptr = reinterpret_cast<void *>(cuFunction);
    PG_CHECK_RUNTIME(function_ptr != nullptr, "Failed to get function pointer");
    // Clean up
    nvrtcDestroyProgram(&prog);
    delete[] ptx;
    this->_func = cuFunction;
  }

  void launch(std::vector<void *> &args) {
    if (_func == nullptr) {
      throw std::runtime_error("Kernel not compiled");
    }
    CUresult R = cuLaunchKernel(_func, _blocks_per_grid.x, _blocks_per_grid.y,
                                _blocks_per_grid.z, _threads_per_block.x,
                                _threads_per_block.y, _threads_per_block.z, 0,
                                0, args.data(), nullptr);
    PG_CHECK_RUNTIME(R == CUDA_SUCCESS, "Failed to launch kernel: got " +
                                            std::to_string(R) + " for kernel " +
                                            _name);
  }

public:
  CudaKernel(const std::string &name, const std::string &src)
      : _name(name), _src(src) {
    // if the kernel has already been compiled, get the function pointer
    void *kernel = KernelDatabase::get_instance().get_kernel(src);
    if (kernel != nullptr) {
      _func = reinterpret_cast<CUfunction>(kernel);
    } else {
      this->compile();
      KernelDatabase::get_instance().add_kernel(src, _func);
    }
  }

  dim3 blocks_per_grid() const { return _blocks_per_grid; }
  dim3 threads_per_block() const { return _threads_per_block; }
  void set_blocks_per_grid(dim3 blocks_per_grid) {
    _blocks_per_grid = blocks_per_grid;
  }
  void set_threads_per_block(dim3 threads_per_block) {
    _threads_per_block = threads_per_block;
  }
  void set_blocks_per_grid(int x) { _blocks_per_grid.x = x; }
  void set_blocks_per_grid(int x, int y) {
    _blocks_per_grid.x = x;
    _blocks_per_grid.y = y;
  }
  void set_blocks_per_grid(int x, int y, int z) {
    _blocks_per_grid.x = x;
    _blocks_per_grid.y = y;
    _blocks_per_grid.z = z;
  }

  void set_threads_per_block(int x) { _threads_per_block.x = x; }
  void set_threads_per_block(int x, int y) {
    _threads_per_block.x = x;
    _threads_per_block.y = y;
  }
  void set_threads_per_block(int x, int y, int z) {
    _threads_per_block.x = x;
    _threads_per_block.y = y;
    _threads_per_block.z = z;
  }
};