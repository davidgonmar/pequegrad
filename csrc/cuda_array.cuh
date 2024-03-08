#pragma once

#include <iostream>
#include <vector>

#include "cuda_array.cuh"
#include "kernels/all.cuh"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using shape_t = std::vector<size_t>;

// for example, we might want to pass reduction indices around, and they can be
// neagitve like array.sum(-1)
using axis_t = int;
using axes_t = std::vector<axis_t>;

class CudaArray {
public:
  bool is_contiguous() const;
  std::shared_ptr<void> ptr;
  size_t size;
  shape_t shape;
  shape_t strides;
  size_t offset;
  DType dtype;

  // returns the base pointer of the array, offsetted by the offset
  void *get_base_ptr() const {
    return static_cast<char *>(ptr.get()) + offset;
  }
  CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
            const std::shared_ptr<void> &shared_ptr, DType dtype);
  CudaArray(size_t size, shape_t shape, shape_t strides, DType dtype);
  CudaArray(size_t size, shape_t shape, DType dtype);

  ~CudaArray();
  CudaArray(const CudaArray &other);
  CudaArray &operator=(const CudaArray &other);
  CudaArray(CudaArray &&other);
  CudaArray &operator=(CudaArray &&other);
  CudaArray clone() const;

  CudaArray broadcast_to(const shape_t _shape) const;

  CudaArray binop(const CudaArray &other, BinaryKernelType kt) const;

  template <typename T>
  CudaArray binop(const py::array_t<T> &other, BinaryKernelType kt) const;

  template <typename T> // scalars
  CudaArray binop(const T other, BinaryKernelType kt) const;

  CudaArray ternaryop(const CudaArray &second, const CudaArray &third,
                      TernaryKernelType ker) const;

  template <typename T>
  CudaArray ternaryop(const py::array_t<T> &second, const py::array_t<T> &third,
                      TernaryKernelType ker) const;
  template <typename T>
  CudaArray ternaryop(const CudaArray &second, const py::array_t<T> &third,
                      TernaryKernelType ker) const;

  template <typename T>
  CudaArray ternaryop(const py::array_t<T> &second, const CudaArray &third,
                      TernaryKernelType ker) const;

  CudaArray elwiseop(UnaryKernelType kt) const;

  template <typename T> T getitem(shape_t index) const;
  int ndim() const;
  CudaArray mat_mul(const CudaArray &other) const;
  CudaArray outer_product(const CudaArray &other) const;
  CudaArray permute(shape_t axes) const;
  CudaArray as_contiguous() const;
  CudaArray sum(axis_t axis, bool keepdims) const;
  CudaArray sum(bool keepdims) const;
  CudaArray sum(axes_t axes, bool keepdims) const;
  CudaArray max(axis_t axis, bool keepdims) const;
  CudaArray max(bool keepdims) const;
  CudaArray max(axes_t axis, bool keepdims) const;

  CudaArray squeeze(axis_t axis) const;
  CudaArray squeeze() const;
  CudaArray squeeze(axes_t axes) const;
  CudaArray unsqueeze(axis_t axis) const;
  CudaArray unsqueeze(axes_t axes) const;

  CudaArray reshape(std::vector<int> &new_shape) const;

  CudaArray im2col(shape_t kernel_shape, int stride_y, int stride_x,
                   int dilation_y, int dilation_x) const;
  CudaArray col2im(shape_t kernel_shape, shape_t out_shape, int stride_y,
                   int stride_x, int dilation_y, int dilation_x) const;

  template <typename T> static CudaArray from_numpy(py::array_t<T> np_array);

  template <typename T> py::array_t<T> to_numpy() const;

  std::string to_string() const;

  template <typename T> static CudaArray fill(shape_t shape, T value);

  CudaArray astype(DType new_dtype) const;

private:
  CudaArray reduce(ReduceKernelType ker, axes_t axes, bool keepdims) const;
  CudaArray reduce(ReduceKernelType ker, axis_t axis, bool keepdims) const;
  CudaArray reduce(ReduceKernelType ker, bool keepdims) const;

  CudaArray binop_same_dtype(const CudaArray &other, BinaryKernelType kt) const;
};

// TEMPLATE IMPLEMENTATIONS
template <typename T>
CudaArray CudaArray::binop(const py::array_t<T> &np_array,
                           BinaryKernelType kt) const {
  CudaArray other = CudaArray::from_numpy(np_array);
  return binop(other, kt);
}

template <typename T>
CudaArray CudaArray::binop(const T scalar, BinaryKernelType kt) const {
  CudaArray other = CudaArray::fill({}, scalar);
  return binop(other, kt);
}

template <typename T>
CudaArray CudaArray::ternaryop(const py::array_t<T> &second,
                               const py::array_t<T> &third,
                               TernaryKernelType ker) const {
  CudaArray second_arr = CudaArray::from_numpy(second);
  CudaArray third_arr = CudaArray::from_numpy(third);
  return ternaryop(second_arr, third_arr, ker);
}

template <typename T>
CudaArray CudaArray::ternaryop(const CudaArray &second,
                               const py::array_t<T> &third,
                               TernaryKernelType ker) const {
  CudaArray third_arr = CudaArray::from_numpy(third);
  return ternaryop(second, third_arr, ker);
}

template <typename T>
CudaArray CudaArray::ternaryop(const py::array_t<T> &second,
                               const CudaArray &third,
                               TernaryKernelType ker) const {
  CudaArray second_arr = CudaArray::from_numpy(second);
  return ternaryop(second_arr, third, ker);
}

template <typename T> T CudaArray::getitem(shape_t index) const {
  PG_CHECK_ARG(index.size() == shape.size(),
               "index size must be equal to shape size, got ", index.size(),
               " and ", shape.size());
  // Calculate the offset for the multi-dimensional index
  size_t elemsize = dtype_to_size(dtype);
  size_t offset = 0;
  for (size_t i = 0; i < index.size(); i++) {
    PG_CHECK_ARG(index[i] < shape[i] && index[i] >= 0,
                 "index out of bounds, got ", index[i], " for shape ",
                 vec_to_string(shape));
    offset += index[i] * strides[i];
  }
  offset /= elemsize; // since strides are in bytes, divide by element size to
                      // get the correct offset

  T value;
  CHECK_CUDA(cudaMemcpy(&value, static_cast<char *>(this->get_base_ptr()) + offset,
                        elemsize, cudaMemcpyDeviceToHost));
  return value;
}

template <typename T> CudaArray CudaArray::from_numpy(py::array_t<T> np_array) {
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

  auto *ptr = static_cast<T *>(buffer_info.ptr);
  CudaArray arr((size_t)size, shape, strides, dtype_from_pytype<T>());
  CHECK_CUDA(cudaMemcpy(arr.get_base_ptr(), ptr, (size_t)size * sizeof(T),
                        cudaMemcpyHostToDevice));
  return arr;
}

template <typename T> py::array_t<T> CudaArray::to_numpy() const {
  // assert that the array has compatible type
  PG_CHECK_ARG(dtype == dtype_from_pytype<T>(),
               "cannot convert to numpy array, expected type ",
               dtype_to_string(dtype), " but got ",
               dtype_to_string(dtype_from_pytype<T>()));
  py::array_t<T> result(shape, strides);
  CHECK_CUDA(cudaMemcpy(result.mutable_data(), this->get_base_ptr(), size * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return result;
}

template <typename T> CudaArray CudaArray::fill(shape_t shape, T value) {
  // calculate correct dtype
  DType _dtype = dtype_from_cpptype<T>();
  CudaArray out(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()),
      shape, _dtype);
  fill_kernel<<<ceil(out.size / (float)DEFAULT_BLOCK_SIZE),
                DEFAULT_BLOCK_SIZE>>>((T *)out.get_base_ptr(), out.size, value);
  PG_CUDA_KERNEL_END;
  return out;
};