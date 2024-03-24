#pragma once

#include "cuda_tensor/cuda_utils.cuh"
#include "kernels/all.cuh"
#include "shape.hpp"
#include "utils.hpp"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#define MAX_THREADS_PER_BLOCK 512

namespace py = pybind11;

struct SliceFromSSS {
  int start;
  int stop;
  int step;
  SliceFromSSS(int start, int stop, int step)
      : start(start), stop(stop), step(step) {}
};

struct SliceFromIdxArray {
  std::vector<int> indices;
  SliceFromIdxArray(const std::vector<int> &indices) : indices(indices) {}
};

struct SliceFromSingleIdx {
  int index;
  SliceFromSingleIdx(int index) : index(index) {}
};

struct SliceKeepDim {
  SliceKeepDim() {}
};

enum class Device_SliceType {
  SliceFromSSS,
  SliceFromIdxArray,
  SliceFromSingleIdx,
  SliceKeepDim
};
struct Device_Slice {
  Device_SliceType type;
  int start;
  int stop;
  int step;
  // this is a device pointer
  int *indices;
  int indexSize;
  int index;
};

// single item, start:stop, or [idx1, idx2, idx3, ...]
using slice_item_t = std::variant<SliceFromSSS, SliceFromIdxArray,
                                  SliceFromSingleIdx, SliceKeepDim>;
using slice_t = std::vector<slice_item_t>;

class CudaTensor {
public:
  bool is_contiguous() const;
  std::shared_ptr<void> ptr;
  size_t size;
  shape_t shape;
  shape_t strides;
  size_t offset;
  DType dtype;

  // returns the base pointer of the array, offsetted by the offset
  void *get_base_ptr() const { return static_cast<char *>(ptr.get()) + offset; }
  CudaTensor(size_t size, const shape_t &shape, const shape_t &strides,
             const std::shared_ptr<void> &shared_ptr, DType dtype);
  CudaTensor(size_t size, const shape_t &shape, const shape_t &strides,
             const std::shared_ptr<void> &shared_ptr, DType dtype, int offset);
  CudaTensor(size_t size, shape_t shape, shape_t strides, DType dtype);
  CudaTensor(size_t size, shape_t shape, DType dtype);

  ~CudaTensor();
  CudaTensor(const CudaTensor &other);
  CudaTensor &operator=(const CudaTensor &other);
  CudaTensor(CudaTensor &&other);
  CudaTensor &operator=(CudaTensor &&other);
  CudaTensor clone() const;

  CudaTensor broadcast_to(const shape_t _shape) const;

  CudaTensor binop(const CudaTensor &other, BinaryKernelType kt) const;

  template <typename T>
  CudaTensor binop(const py::array_t<T> &other, BinaryKernelType kt) const;

  template <typename T> // scalars
  CudaTensor binop(const T other, BinaryKernelType kt) const;

  CudaTensor ternaryop(const CudaTensor &second, const CudaTensor &third,
                       TernaryKernelType ker) const;

  template <typename T>
  CudaTensor ternaryop(const py::array_t<T> &second,
                       const py::array_t<T> &third,
                       TernaryKernelType ker) const;
  template <typename T>
  CudaTensor ternaryop(const CudaTensor &second, const py::array_t<T> &third,
                       TernaryKernelType ker) const;

  template <typename T>
  CudaTensor ternaryop(const py::array_t<T> &second, const CudaTensor &third,
                       TernaryKernelType ker) const;

  template <typename T>
  CudaTensor ternaryop(const T second, const T third,
                       TernaryKernelType ker) const;

  template <typename T>
  CudaTensor ternaryop(const CudaTensor &second, const T third,
                       TernaryKernelType ker) const;

  template <typename T>
  CudaTensor ternaryop(const T second, const CudaTensor &third,
                       TernaryKernelType ker) const;

  CudaTensor elwiseop(UnaryKernelType kt) const;

  template <typename T> T getitem(shape_t index) const;
  int ndim() const;
  CudaTensor mat_mul(const CudaTensor &other) const;
  CudaTensor outer_product(const CudaTensor &other) const;
  CudaTensor permute(shape_t axes) const;
  CudaTensor as_contiguous() const;
  CudaTensor sum(axis_t axis, bool keepdims) const;
  CudaTensor sum(bool keepdims) const;
  CudaTensor sum(axes_t axes, bool keepdims) const;
  CudaTensor max(axis_t axis, bool keepdims) const;
  CudaTensor max(bool keepdims) const;
  CudaTensor max(axes_t axis, bool keepdims) const;
  CudaTensor mean(axis_t axis, bool keepdims) const;
  CudaTensor mean(bool keepdims) const;
  CudaTensor mean(axes_t axis, bool keepdims) const;

  CudaTensor squeeze(axis_t axis) const;
  CudaTensor squeeze() const;
  CudaTensor squeeze(axes_t axes) const;
  CudaTensor unsqueeze(axis_t axis) const;
  CudaTensor unsqueeze(axes_t axes) const;

  CudaTensor reshape(const std::vector<int> &new_shape) const;

  CudaTensor im2col(shape_t kernel_shape, int stride_y, int stride_x,
                    int dilation_y, int dilation_x) const;
  CudaTensor col2im(shape_t kernel_shape, shape_t out_shape, int stride_y,
                    int stride_x, int dilation_y, int dilation_x) const;

  template <typename T> static CudaTensor from_numpy(py::array_t<T> np_array);

  template <typename T> py::array_t<T> to_numpy() const;

  std::string to_string() const;

  template <typename T> static CudaTensor fill(shape_t shape, T value);

  CudaTensor astype(DType new_dtype) const;

  CudaTensor slice(const slice_t &slices) const;
  CudaTensor assign(const slice_t &slices, const CudaTensor &vals);

private:
  CudaTensor reduce(ReduceKernelType ker, axes_t axes, bool keepdims) const;
  CudaTensor reduce(ReduceKernelType ker, axis_t axis, bool keepdims) const;
  CudaTensor reduce(ReduceKernelType ker, bool keepdims) const;

  CudaTensor binop_same_dtype(const CudaTensor &other,
                              BinaryKernelType kt) const;
};

// TEMPLATE IMPLEMENTATIONS
template <typename T>
CudaTensor CudaTensor::binop(const py::array_t<T> &np_array,
                             BinaryKernelType kt) const {
  CudaTensor other = CudaTensor::from_numpy(np_array);
  return binop(other, kt);
}

template <typename T>
CudaTensor CudaTensor::binop(const T scalar, BinaryKernelType kt) const {
  CudaTensor other = CudaTensor::fill({}, scalar);
  return binop(other, kt);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const py::array_t<T> &second,
                                 const py::array_t<T> &third,
                                 TernaryKernelType ker) const {
  CudaTensor second_arr = CudaTensor::from_numpy(second);
  CudaTensor third_arr = CudaTensor::from_numpy(third);
  return ternaryop(second_arr, third_arr, ker);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const CudaTensor &second,
                                 const py::array_t<T> &third,
                                 TernaryKernelType ker) const {
  CudaTensor third_arr = CudaTensor::from_numpy(third);
  return ternaryop(second, third_arr, ker);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const py::array_t<T> &second,
                                 const CudaTensor &third,
                                 TernaryKernelType ker) const {
  CudaTensor second_arr = CudaTensor::from_numpy(second);
  return ternaryop(second_arr, third, ker);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const T second, const T third,
                                 TernaryKernelType ker) const {
  CudaTensor second_arr = CudaTensor::fill({}, second);
  CudaTensor third_arr = CudaTensor::fill({}, third);
  return ternaryop(second_arr, third_arr, ker);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const CudaTensor &second, const T third,
                                 TernaryKernelType ker) const {
  CudaTensor third_arr = CudaTensor::fill({}, third);
  return ternaryop(second, third_arr, ker);
}

template <typename T>
CudaTensor CudaTensor::ternaryop(const T second, const CudaTensor &third,
                                 TernaryKernelType ker) const {
  CudaTensor second_arr = CudaTensor::fill({}, second);
  return ternaryop(second_arr, third, ker);
}

template <typename T> T CudaTensor::getitem(shape_t index) const {
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
  CHECK_CUDA(cudaMemcpy(&value,
                        static_cast<char *>(this->get_base_ptr()) + offset,
                        elemsize, cudaMemcpyDeviceToHost));
  return value;
}

template <typename T>
CudaTensor CudaTensor::from_numpy(py::array_t<T> np_array) {
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

  auto *_ptr = static_cast<T *>(buffer_info.ptr);
  CudaTensor arr((size_t)size, shape, strides, dtype_from_pytype<T>());
  CHECK_CUDA(cudaMemcpy(arr.get_base_ptr(), _ptr, (size_t)size * sizeof(T),
                        cudaMemcpyHostToDevice));
  return arr;
}

template <typename T> py::array_t<T> CudaTensor::to_numpy() const {
  if (!is_contiguous()) {
    return as_contiguous().to_numpy<T>();
  }
  // assert that the array has compatible type
  PG_CHECK_ARG(dtype == dtype_from_pytype<T>(),
               "cannot convert to numpy array, expected type ",
               dtype_to_string(dtype), " but got ",
               dtype_to_string(dtype_from_pytype<T>()));
  py::array_t<T> result(shape, strides);
  CHECK_CUDA(cudaMemcpy(result.mutable_data(), this->get_base_ptr(),
                        size * dtype_to_size(dtype), cudaMemcpyDeviceToHost));
  return result;
}

template <typename T> CudaTensor CudaTensor::fill(shape_t shape, T value) {
  // calculate correct dtype
  DType _dtype = dtype_from_cpptype<T>();
  CudaTensor out(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()),
      shape, _dtype);
  fill_kernel<<<ceil(out.size / (float)DEFAULT_BLOCK_SIZE),
                DEFAULT_BLOCK_SIZE>>>((T *)out.get_base_ptr(), out.size, value);
  PG_CUDA_KERNEL_END;
  return out;
};