#include "dtype.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

class CpuTensor {

private:
  size_t compute_nbytes(const shape_t &shape, DType dtype) const;

public:
  bool is_contiguous() const;
  std::shared_ptr<void> ptr;
  size_t nbytes;
  shape_t shape;
  shape_t strides;
  size_t offset;
  DType dtype;

  CpuTensor(const shape_t &shape, const shape_t &strides,
            const std::shared_ptr<void> &ptr, DType dtype)
      : nbytes(compute_nbytes(shape, dtype)), shape(shape), strides(strides),
        ptr(ptr), offset(0), dtype(dtype) {}

  template <typename T> static CpuTensor from_numpy(py::array_t<T> np_array) {
    py::buffer_info buffer_info = np_array.request();
    auto size = buffer_info.size;
    shape_t shape;
    shape_t strides;

    if (buffer_info.ndim == 0) { // Handle scalar as a special case
      shape = {};                // Empty shape for scalar
      strides = {};              // Empty strides for scalar
    } else {
      std::vector<py::ssize_t> py_strides = buffer_info.strides;
      strides.assign(py_strides.begin(), py_strides.end());
      std::vector<py::ssize_t> py_shape = buffer_info.shape;
      shape.assign(py_shape.begin(), py_shape.end());
    }

    auto _ptr = std::shared_ptr<T>(new T[size], [](T *p) { delete[] p; });
    std::memcpy(_ptr.get(), buffer_info.ptr, size * sizeof(T));
    CpuTensor arr(shape, strides, _ptr, dtype_from_pytype<T>());
    return arr;
  }

  template <typename T> py::array_t<T> to_numpy() const {
    // TODO -- Convert to contiguous array
    py::array_t<T> np_array(shape, strides);
    std::memcpy(np_array.mutable_data(), ptr.get(), nbytes);
    return np_array;
  }

  CpuTensor broadcast_to(const shape_t &new_shape) const;

  CpuTensor add(const CpuTensor &other) const;
  CpuTensor sub(const CpuTensor &other) const;
  CpuTensor mul(const CpuTensor &other) const;
  CpuTensor div(const CpuTensor &other) const;

  CpuTensor gt(const CpuTensor &other) const;
  CpuTensor lt(const CpuTensor &other) const;
  CpuTensor ge(const CpuTensor &other) const;
  CpuTensor le(const CpuTensor &other) const;
  CpuTensor eq(const CpuTensor &other) const;
  CpuTensor ne(const CpuTensor &other) const;
  CpuTensor pow(const CpuTensor &other) const;
  CpuTensor el_wise_max(const CpuTensor &other) const;

  CpuTensor exp() const;
  CpuTensor log() const;
};