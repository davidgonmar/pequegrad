#pragma once

#include <iostream>
#include <vector>

#include "binary_ops_kernels.cuh"
#include "dtype.cuh"
#include "reduce_ops_kernels.cuh"
#include "ternary_ops_kernels.cuh"
#include "unary_ops_kernels.cuh"
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
  DType dtype;

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

  CudaArray im2col(shape_t kernel_shape, int stride) const;
  CudaArray col2im(shape_t kernel_shape, shape_t out_shape, int stride) const;

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
