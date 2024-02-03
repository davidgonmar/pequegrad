#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <binary_ops_kernels.cuh>

#include <iostream>
#include <sstream>

#define ELEM_SIZE sizeof(float)
#define DEFAULT_BLOCK_SIZE 256


namespace pequegrad{
namespace cuda {
namespace py = pybind11;

typedef void(*BinaryOpKernel)(size_t, const float*, const float*, float*);

class CudaArray {
public:
    float* ptr;
    size_t size;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
  
    CudaArray(size_t size, std::vector<py::ssize_t> shape, std::vector<py::ssize_t> strides) :size(size), shape(shape), strides(strides) {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        //printf("CudaArray(%d) constructor\n", size);
        //printf("ptr: %d\n", ptr);
        //printf("ptr_as_int: %d\n", ptr_as_int());
    }
    CudaArray binop(const CudaArray& other, BinaryOpKernel Ker) const {
        if (size != other.size) {
            throw std::runtime_error("Size mismatch");
        }
        CudaArray out(size, shape, strides);
        dim3 block(DEFAULT_BLOCK_SIZE);
        dim3 grid((out.size + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE);
        Ker<<<grid, block>>>(out.size, this->ptr, other.ptr, out.ptr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        return out;
    }

    float getitem(std::vector<py::ssize_t> index) const {
        if (index.size() != shape.size()) {
            throw std::runtime_error("Index dimension mismatch");
        }

        // Calculate the offset for the multi-dimensional index
        size_t offset = 0;
        for (size_t i = 0; i < index.size(); i++) {
          if (index[i] < 0 || index[i] >= shape[i]) {
                throw std::runtime_error("Index out of bounds");
          }
          offset += index[i] * strides[i] / ELEM_SIZE;
        }
        // Copy the requested element from device to host
        float value;
        cudaError_t err = cudaMemcpy(&value, ptr + offset, ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return value;
    }

    static CudaArray fromNumpy(py::array_t<float> np_array) {
        py::buffer_info buffer_info = np_array.request();
        std::vector<py::ssize_t> py_strides = buffer_info.strides;
        std::vector<size_t> strides(py_strides.begin(), py_strides.end());
        auto size = buffer_info.size;
        auto* ptr = static_cast<float*>(buffer_info.ptr);
        std::vector<py::ssize_t> py_shape = buffer_info.shape;
        CudaArray arr(size, py_shape, py_strides);
        auto err = cudaMemcpy(arr.ptr, ptr, size * ELEM_SIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        return arr;
    }

    py::array_t<float> toNumpy() const {
        py::array_t<float> result(shape, strides);
        auto err = cudaMemcpy(result.mutable_data(), ptr, size * ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        return result;
    }
   
    std::string toString() const {
        std::stringstream ss;
        ss << "CudaArray(" << size << ") [";
        float* host = (float*)malloc(size * ELEM_SIZE);
        if (host == nullptr) {
            throw std::runtime_error("failed to allocate host memory");
        }
        // Ensure all operations on the device are complete before copying memory
        cudaDeviceSynchronize();
        auto err = cudaMemcpy(host, ptr, size * ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            free(host); // Ensure we free the allocated memory in case of an error
            throw std::runtime_error("got cuda error: " + std::string(cudaGetErrorString(err)));
        }
        for (size_t i = 0; i < size; i++) {
            ss << host[i] << " ";
        }
        free(host);
        ss << "]";
        return ss.str();
    }

    ~CudaArray() {
        //printf("CudaArray destructor\n");
        //printf("ptr: %d\n", ptr);
        cudaFree(ptr);
    }

    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;

    CudaArray(CudaArray&& other) {
        //printf("CudaArray move constructor\n");
        //printf("other.ptr: %d\n", other.ptr);
        ptr = other.ptr;
        size = other.size;
        shape = other.shape;
        strides = other.strides;
        other.ptr = nullptr;
    }

    CudaArray& operator=(CudaArray&& other) {
        //printf("CudaArray move assignment\n");
        //printf("other.ptr: %d\n", other.ptr);
        if (this != &other) {
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            shape = other.shape;
            strides = other.strides;
        }
        return *this;
    }

    CudaArray clone() const {
        CudaArray out(size, shape, strides);
        cudaError_t err = cudaMemcpy(out.ptr, ptr, size * ELEM_SIZE, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        return out;
    }


    int64_t ptr_as_int() const { return (int64_t)ptr; }

};






}  // namespace cuda
}  // namespace pequegrad

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;
  using namespace pequegrad;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  py::class_<CudaArray>(m, "Array")
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int)
      .def_readonly("shape", &CudaArray::shape)
      .def_readonly("strides", &CudaArray::strides)
      .def("clone", &CudaArray::clone)
      .def("to_numpy", &CudaArray::toNumpy)
      .def("from_numpy", [](py::array_t<float> np_array) {
        return CudaArray::fromNumpy(np_array);
      }).def("__repr__", [](const CudaArray& arr) {
        return arr.toString();
      }).def("add", [](const CudaArray& arr, const CudaArray& other) {
        return arr.binop(other, AddKernel);
      })
      .def("sub", [](const CudaArray& arr, const CudaArray& other) {
        return arr.binop(other, SubKernel);
      }).
      def("mul", [](const CudaArray& arr, const CudaArray& other) {
        return arr.binop(other, MultKernel);
      }).
      def("div", [](const CudaArray& arr, const CudaArray& other) {
        return arr.binop(other, DivKernel);
      }).
      def("__getitem__", [](const CudaArray& arr, std::vector<py::ssize_t> index) {
        return arr.getitem(index);
    });


  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<float> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

}