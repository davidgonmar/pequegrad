#include "binary_ops_kernels.cuh"
#include "unary_ops_kernels.cuh"
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#define ELEM_SIZE sizeof(float)
#define DEFAULT_BLOCK_SIZE 256

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t _err = x;                                                      \
    if (_err != cudaSuccess) {                                                 \
      std::ostringstream oss;                                                  \
      oss << "CUDA error " << _err << " on file/line " << __FILE__ << ":"      \
          << __LINE__ << " " << cudaGetErrorString(_err);                      \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

namespace pequegrad {
namespace cuda {
namespace py = pybind11;

typedef void (*BinaryOpKernel)(const int *strides, const int *ostrides,
                               const int *shape, const int ndim, const float *a,
                               const float *b, float *out);

typedef void (*ElementWiseOpKernel)(
  const int *in_strides,const int *shape, const int num_dims,const float *in, float *out
);

class CudaArray {
public:
  std::shared_ptr<float> ptr;
  size_t size;
  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;

  bool isContiguous() const {
    if (strides.size() != shape.size()) {
      return false;
    }
    std::vector<py::ssize_t> expected_strides(shape.size());
    expected_strides[shape.size() - 1] = ELEM_SIZE;
    for (int i = shape.size() - 2; i >= 0; --i) {
      expected_strides[i] = expected_strides[i + 1] * shape[i + 1];
    }
    if (expected_strides != strides) {
      return false;
    }
    return true;
  }
  CudaArray(size_t size, const std::vector<py::ssize_t> &shape,
            const std::vector<py::ssize_t> &strides,
            const std::shared_ptr<float> &sharedPtr)
      : size(size), shape(shape), strides(strides), ptr(sharedPtr) {}
  CudaArray permute(std::vector<py::ssize_t> axes) const {
    // TODO - check that axes is from 0 to shape.size - 1, in any order
    if (axes.size() != shape.size()) {
      throw std::runtime_error("axes must have same size as shape");
    }
    std::vector<py::ssize_t> newShape(shape.size());
    std::vector<py::ssize_t> newStrides(strides.size());

    for (size_t i = 0; i < axes.size(); ++i) {
      newShape[i] = shape[axes[i]];
      newStrides[i] = strides[axes[i]];
    }

    CudaArray out(size, newShape, newStrides, ptr);
    return out;
  }

  CudaArray(size_t size, std::vector<py::ssize_t> shape,
            std::vector<py::ssize_t> strides)
      : size(size), shape(shape), strides(strides) {
    float *raw_ptr;
    CHECK_CUDA(cudaMalloc(&raw_ptr, size * ELEM_SIZE));
    ptr = std::shared_ptr<float>(raw_ptr, [](float *p) { cudaFree(p); });
  }

  CudaArray(size_t size, std::vector<py::ssize_t> shape)
      : size(size), shape(shape) {
    strides.resize(shape.size());
    strides[shape.size() - 1] = ELEM_SIZE;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    float *raw_ptr;
    CHECK_CUDA(cudaMalloc(&raw_ptr, size * ELEM_SIZE));
    ptr = std::shared_ptr<float>(raw_ptr, [](float *p) { cudaFree(p); });
  }

  CudaArray broadcastTo(const std::vector<py::ssize_t> _shape) const {
    const std::vector<py::ssize_t> shape_from = this->shape;
    const std::vector<py::ssize_t> shape_to = _shape;
    // determine if we can broadcast
    const int from_ndim = (const int)shape_from.size();
    const int to_ndim = (const int)shape_to.size();
    // cannot broadcast if the number of dimensions of the from array is greater
    // than the number of dimensions of the to array
    if (from_ndim > to_ndim) {
      throw std::runtime_error(
          "got incompatible shapes, to_ndim < from_ndim: " +
          std::to_string(to_ndim) + " < " + std::to_string(from_ndim));
    }

    int new_size = 1;
    std::vector<py::ssize_t> new_strides(to_ndim, 0);
    // reverse test if the dim is 1 or they are equal
    for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
      py::ssize_t dim_to = shape_to[i];
      py::ssize_t dim_from =
          (j >= 0) ? shape_from[j]
                   : -1; // -1 means we 'ran' out of dimensions for j
      if (dim_to != dim_from && dim_from != 1 && dim_from != -1) {
        // we can only 'broadcast' a dimension if dim_from == 1 or we ran out of
        // dimensions.
        throw std::runtime_error(
            "got incompatible shapes, dim_to != dim_from: " +
            std::to_string(dim_to) + " != " + std::to_string(dim_from));
      }
      if (dim_from != 1 && dim_from != -1) {
        new_strides[i] = strides[j];
      }
      new_size *= dim_to;
    }
    CudaArray out(new_size, shape_to, new_strides);
    CHECK_CUDA(cudaMemcpy(out.ptr.get(), ptr.get(), size * ELEM_SIZE,
                          cudaMemcpyDeviceToDevice));
    return out;
  }

  CudaArray binop(const CudaArray &other, BinaryOpKernel Ker) const {
    if (shape != other.shape) {
      // try to broadcast, from smaller to larger
      if (shape.size() < other.shape.size()) {
        return broadcastTo(other.shape).binop(other, Ker);
      } else if (shape.size() > other.shape.size()) {
        return binop(other.broadcastTo(shape), Ker);
      } else {
        // we need to check the one with less product of shape, and try to
        // broadcast
        int64_t prod_shape = 1;
        int64_t prod_other_shape = 1;
        for (int i = 0; i < shape.size(); i++) {
          prod_shape *= shape[i];
          prod_other_shape *= other.shape[i];
        }
        if (prod_shape < prod_other_shape) {
          return broadcastTo(other.shape).binop(other, Ker);
        } else {
          return binop(other.broadcastTo(shape), Ker);
        }
      }
    }
    assert(shape == other.shape);
    dim3 block_size(DEFAULT_BLOCK_SIZE);
    dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
    // Default stride calculation
    CudaArray out(size, shape);
    int n_dims = shape.size();
    int *d_strides, *d_other_strides, *d_shape;
    CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_other_strides, n_dims * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(int)));

    int *host_strides = (int *)malloc(n_dims * sizeof(int));
    int *host_other_strides = (int *)malloc(n_dims * sizeof(int));
    int *host_shape = (int *)malloc(n_dims * sizeof(int));

    for (int i = 0; i < n_dims; i++) {
      host_strides[i] = strides[i];
      host_other_strides[i] = other.strides[i];
      host_shape[i] = shape[i];
    }

    CHECK_CUDA(cudaMemcpy(d_strides, host_strides, n_dims * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_other_strides, host_other_strides,
                          n_dims * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_shape, host_shape, n_dims * sizeof(int),
                          cudaMemcpyHostToDevice));
    Ker<<<grid_size, block_size>>>(d_strides, d_other_strides, d_shape, n_dims,
                                   ptr.get(), other.ptr.get(), out.ptr.get());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
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
      offset +=
          index[i] * strides[i] / ELEM_SIZE; // since strides are in bytes,
      // we need to divide by ELEM_SIZE to get the correct offset
    }
    // Copy the requested element from device to host
    float value;
    CHECK_CUDA(cudaMemcpy(&value, ptr.get() + offset, ELEM_SIZE,
                          cudaMemcpyDeviceToHost));
    return value;
  }

  static CudaArray fromNumpy(py::array_t<float> np_array) {
    py::buffer_info buffer_info = np_array.request();
    std::vector<py::ssize_t> py_strides = buffer_info.strides;
    std::vector<size_t> strides(py_strides.begin(), py_strides.end());
    auto size = buffer_info.size;
    auto *ptr = static_cast<float *>(buffer_info.ptr);
    std::vector<py::ssize_t> py_shape = buffer_info.shape;
    CudaArray arr(size, py_shape, py_strides);
    CHECK_CUDA(cudaMemcpy(arr.ptr.get(), ptr, size * ELEM_SIZE,
                          cudaMemcpyHostToDevice));
    return arr;
  }

  py::array_t<float> toNumpy() const {
    py::array_t<float> result(shape, strides);
    CHECK_CUDA(cudaMemcpy(result.mutable_data(), ptr.get(), size * ELEM_SIZE,
                          cudaMemcpyDeviceToHost));
    float *host = (float *)malloc(size * ELEM_SIZE);
    if (host == nullptr) {
      throw std::runtime_error("failed to allocate host memory");
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(
        cudaMemcpy(host, ptr.get(), size * ELEM_SIZE, cudaMemcpyDeviceToHost));
    return result;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "CudaArray(" << size << ") [";
    float *host = (float *)malloc(size * ELEM_SIZE);
    if (host == nullptr) {
      throw std::runtime_error("failed to allocate host memory");
    }
    CHECK_CUDA(
        cudaMemcpy(host, ptr.get(), size * ELEM_SIZE, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      ss << host[i] << " ";
    }
    free(host);
    ss << "]";
    return ss.str();
  }

  ~CudaArray() {}

  CudaArray(const CudaArray &other)
      : size(other.size), shape(other.shape), strides(other.strides),
        ptr(other.ptr) {}

  CudaArray &operator=(const CudaArray &other) {
    if (this != &other) {
      size = other.size;
      shape = other.shape;
      strides = other.strides;
      ptr = other.ptr;
    }
    return *this;
  }

  CudaArray(CudaArray &&other) noexcept
      : size(other.size), shape(std::move(other.shape)),
        strides(std::move(other.strides)), ptr(std::move(other.ptr)) {}

  CudaArray &operator=(CudaArray &&other) noexcept {
    if (this != &other) {
      size = other.size;
      shape = std::move(other.shape);
      strides = std::move(other.strides);
      ptr = std::move(other.ptr);
    }
    return *this;
  }

  CudaArray clone() const {
    CudaArray out(size, shape, strides);
    CHECK_CUDA(cudaMemcpy(out.ptr.get(), ptr.get(), size * ELEM_SIZE,
                          cudaMemcpyDeviceToDevice));
    return out;
  }

  CudaArray elwiseop(ElementWiseOpKernel Ker) const {
    dim3 block_size(DEFAULT_BLOCK_SIZE);
    dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
    int n_dims = shape.size();
    int *d_strides, *d_shape;

    CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(int)));

    int *host_strides = (int *)malloc(n_dims * sizeof(int));
    int *host_shape = (int *)malloc(n_dims * sizeof(int));

    for (int i = 0; i < n_dims; i++) {
      host_strides[i] = strides[i];
      host_shape[i] = shape[i];
    }

    CHECK_CUDA(cudaMemcpy(d_strides, host_strides, n_dims * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_shape, host_shape, n_dims * sizeof(int),
                          cudaMemcpyHostToDevice));

    CudaArray out(size, shape);
    Ker<<<grid_size, block_size>>>(d_strides, d_shape, n_dims,
                                          this->ptr.get(), out.ptr.get());

    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    return out;
  }

  CudaArray asContiguous() const {
    return elwiseop(CopyKernel);
  }

  int64_t ptr_as_int() const { return (int64_t)ptr.get(); }
};

} // namespace cuda
} // namespace pequegrad

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
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return CudaArray::fromNumpy(np_array);
           })
      .def("__repr__", [](const CudaArray &arr) { return arr.toString(); })
      .def("add",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, AddKernel);
           })
      .def("sub",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, SubKernel);
           })
      .def("mul",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, MultKernel);
           })
      .def("div",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, DivKernel);
           })
      .def("eq",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, EqualKernel);
           })
      .def("ne",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, NotEqualKernel);
           })
      .def("lt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, LessKernel);
           })
      .def("le",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, LessEqualKernel);
           })
      .def("gt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, GreaterKernel);
           })
      .def("ge",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, GreaterEqualKernel);
           })
      .def("contiguous",
           [](const CudaArray &arr) { return arr.asContiguous();})
      .def("exp",
      [](const CudaArray &arr) {
          return arr.elwiseop(ExpKernel);
      }).
      def("log",
      [](const CudaArray &arr) {
          return arr.elwiseop(LogKernel);
      })
      .def("permute",
           [](const CudaArray &arr, std::vector<py::ssize_t> axes) {
             return arr.permute(axes);
           })
      .def("is_contiguous", &CudaArray::isContiguous)
      .def("__getitem__",
           [](const CudaArray &arr, std::vector<py::ssize_t> index) {
             return arr.getitem(index);
           });
}