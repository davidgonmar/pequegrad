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
    CudaArray broadcastTo(const std::vector<py::ssize_t> shape) const {
        const std::vector<py::ssize_t> shape_from = this->shape;
        const std::vector<py::ssize_t> shape_to = shape;
        // determine if we can broadcast
        const int from_ndim = shape_from.size();
        const int to_ndim = shape_to.size();
        // cannot broadcast if the number of dimensions of the from array is greater than the number of dimensions of the to array
        if (from_ndim > to_ndim) {
            throw std::runtime_error("got incompatible shapes, to_ndim < from_ndim: " + std::to_string(to_ndim) + " < " + std::to_string(from_ndim));
        }
        
        int new_size = 1;
        std::vector<py::ssize_t> new_strides(to_ndim, 0);
        // reverse test if the dim is 1 or they are equal
        for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
            py::ssize_t dim_to = shape_to[i];
            py::ssize_t dim_from = (j >= 0) ? shape_from[j] : 1; // assume non existing shape is 1
            if (dim_to != dim_from && dim_from != 1) {
                throw std::runtime_error("got incompatible shapes");
            }
            new_size *= dim_to;
            new_strides[i] = (dim_from == 1) ? 0 : strides[j];
        }
        CudaArray out(new_size, shape_to, new_strides);
        // copy the data to the new array
        
        cudaError_t err = cudaMemcpy(out.ptr, ptr, size * ELEM_SIZE, cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) throw std::runtime_error("cuda error: " + std::string(cudaGetErrorString(err)));

        return out;
        
    }
    CudaArray binop(const CudaArray& other, BinaryOpKernel Ker) const {
        if (size != other.size) {
            throw std::runtime_error("got incompatible shapes");
        }
        dim3 block_size(DEFAULT_BLOCK_SIZE);
        dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
        CudaArray out(size, shape, strides);
        Ker<<<grid_size, block_size>>>(size, ptr, other.ptr, out.ptr);
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
          offset += index[i] * strides[i] / ELEM_SIZE; // since strides are in bytes,
          //we need to divide by ELEM_SIZE to get the correct offset
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

    CudaArray(const CudaArray& other) {
    size = other.size;
    shape = other.shape;
    strides = other.strides;
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMemcpy(ptr, other.ptr, size * ELEM_SIZE, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(ptr);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

CudaArray& operator=(const CudaArray& other) {
    if (this != &other) {
        cudaFree(ptr); // Free existing device memory
        size = other.size;
        shape = other.shape;
        strides = other.strides;
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        err = cudaMemcpy(ptr, other.ptr, size * ELEM_SIZE, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFree(ptr);
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    return *this;
}

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
      .def("broadcast_to", &CudaArray::broadcastTo)
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


}