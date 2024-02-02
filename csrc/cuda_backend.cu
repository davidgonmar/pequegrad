#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#define ELEM_SIZE sizeof(float)
#define DEFAULT_BLOCK_SIZE 256

namespace pequegrad{
namespace cuda {

class CudaArray {
 public:
  float* ptr;
  size_t size;

  CudaArray(size_t size) : size(size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }

  ~CudaArray() { cudaFree(ptr); }

  int64_t ptr_as_int() const { return (int64_t)ptr; }
};

__global__ void AddKernel(const float* a, const float* b, float* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}


void Add(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  dim3 block(DEFAULT_BLOCK_SIZE);
  dim3 grid(ceil(out->size / (float)DEFAULT_BLOCK_SIZE));

  AddKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


}  // namespace cuda
}  // namespace pequegrad

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;
  using namespace pequegrad;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);


  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<float> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("add", &Add);
}