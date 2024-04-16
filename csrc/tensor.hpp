
#pragma once
#include "dtype.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

static inline strides_t _compute_natural_strides(const shape_t &shape,
                                                 const DType dtype) {
  if (shape.size() == 0) {
    return strides_t(); // if scalar, return empty strides
  }
  strides_t strides(shape.size());
  size_t dtype_size = dtype_to_size(dtype);
  strides[shape.size() - 1] = dtype_size;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

static inline size_t compute_nbytes(const shape_t &shape, const DType dtype) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<size_t>()) *
         dtype_to_size(dtype);
}

#define _throw_if_not_initialized(msg)                                         \
  if (!is_initialized()) {                                                       \
    std::cout << "Exception on line " << __LINE__ << " in file " << __FILE__    \
              << ": " << msg << std::endl;                                       \
    throw std::runtime_error(msg);                                              \
  }

namespace pg {

class ADPrimitive; // forward declaration
class Tensor;

namespace py = pybind11;
class View {

public:
  void *get_base_ptr() const;

  std::shared_ptr<void> shared_ptr() const;
  shape_t shape() const;

  strides_t strides() const;

  size_t offset() const;

  size_t nbytes() const;

  DType dtype() const;

  View(const std::shared_ptr<void> &ptr, const size_t nbytes,
       const shape_t &shape, const strides_t &strides, const size_t offset,
       DType dtype);

  View(const shape_t shape, DType dtype) {
    _shape = shape;
    size_t nbytes = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<size_t>()) *
                    dtype_to_size(dtype);
    _nbytes = nbytes;
    _strides = _compute_natural_strides(shape, dtype);
    _ptr = std::shared_ptr<void>(new char[nbytes], [](void *p) { delete[] p; });
    _offset = 0;
    _initialized = true;
    _dtype = dtype;
  }

  void init_view(const std::shared_ptr<View> &view) {
    if (is_initialized()) {
      throw std::runtime_error(
          "View already initialized, but tried to call init_view.");
    }
    _ptr = view->_ptr;
    _nbytes = view->_nbytes;
    _shape = view->_shape;
    _strides = view->_strides;
    _offset = view->_offset;
    _dtype = view->_dtype;
    _initialized = true;
  }

  View() = default;

  bool is_initialized() const { return _initialized; }

  // Copy and move constructors
  View(const View &other) {
    _ptr = other._ptr;
    _nbytes = other._nbytes;
    _shape = other._shape;
    _strides = other._strides;
    _offset = other._offset;
    _dtype = other._dtype;
  }

  View(View &&other) {
    _ptr = std::move(other._ptr);
    _nbytes = other._nbytes;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _offset = other._offset;
    _dtype = other._dtype;
  }

  View &operator=(const View &other) {
    _ptr = other._ptr;
    _nbytes = other._nbytes;
    _shape = other._shape;
    _strides = other._strides;
    _offset = other._offset;
    _dtype = other._dtype;
    return *this;
  }

  View &operator=(View &&other) {
    _ptr = std::move(other._ptr);
    _nbytes = other._nbytes;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _offset = other._offset;
    _dtype = other._dtype;
    return *this;
  }

private:
  std::shared_ptr<void> _ptr;
  size_t _nbytes; // number of bytes of the pointer. Does not necessarily match
                  // shape product * dtype size
  shape_t _shape;
  strides_t _strides;
  size_t _offset; // offset in bytes
  DType _dtype;
  bool _initialized = false;
};

class ADNode {
public:
  explicit ADNode(std::shared_ptr<ADPrimitive> primitive,
                  std::vector<Tensor> children);
  explicit ADNode() = default;
  static ADNode create_leaf();
  bool is_leaf() { return _primitive == nullptr || _children.empty(); };
  std::shared_ptr<ADPrimitive> primitive() const;
  std::vector<Tensor> children() const;
  const std::shared_ptr<Tensor> grad() const;
  void accum_grad(Tensor &grad);

private:
  std::shared_ptr<ADPrimitive> _primitive = nullptr;
  std::vector<Tensor> _children;
  std::shared_ptr<Tensor> _grad = nullptr;
};

class Tensor {

public:
  const Tensor &grad() const { return *(_ad_node->grad().get()); }

  bool Tensor::is_contiguous() const {
    if (offset() != 0) {
      return false;
    }
    if (strides().size() == 0) { // scalar
      return true;
    }
    strides_t expected_strides(shape().size());
    expected_strides[shape().size() - 1] = dtype_to_size(dtype());
    for (int i = shape().size() - 2; i >= 0; --i) {
      expected_strides[i] = expected_strides[i + 1] * shape()[i + 1];
    }
    if (expected_strides != strides()) {
      return false;
    }
    return true;
  }

  // Copy and move constructors
  Tensor(const Tensor &other) {
    _view = other._view;
    _ad_node = other._ad_node;
  }

  Tensor(Tensor &&other) {
    _view = std::move(other._view);
    _ad_node = std::move(other._ad_node);
  }

  View view() const {
    _throw_if_not_initialized("view() called on uninitialized tensor.");
    return *_view;
  }

  void init_view(std::shared_ptr<View> view) {
    if (is_initialized()) {
      throw std::runtime_error(
          "Tensor already initialized, but tried to call init_view.");
    }
    _view->init_view(view);
  }

  shape_t shape() const {
    _throw_if_not_initialized("shape() called on uninitialized tensor.");
    return view().shape(); }

  strides_t strides() const {
    _throw_if_not_initialized("strides() called on uninitialized tensor.");
    return view().strides(); }

  size_t offset() const {
    _throw_if_not_initialized("offset() called on uninitialized tensor.");
    return view().offset(); }

  size_t nbytes() const {
    _throw_if_not_initialized("nbytes() called on uninitialized tensor.");
    return view().nbytes(); }

  DType dtype() const {
    _throw_if_not_initialized("dtype() called on uninitialized tensor.");
    return view().dtype(); }

  void *get_base_ptr() const {
    _throw_if_not_initialized("get_base_ptr() called on uninitialized tensor.");
    return view().get_base_ptr(); }

  template <typename T> T* get_casted_base_ptr() {
    _throw_if_not_initialized("get_casted_base_ptr() called on uninitialized tensor.");
    if (dtype_from_cpptype<T>() != this->dtype()) {
      throw std::runtime_error("Cannot cast pointer to different dtype.");
    }
    return static_cast<T *>(view().get_base_ptr());
  }
  template <typename T> static Tensor from_numpy(py::array_t<T> np_array) {
    py::buffer_info buffer_info = np_array.request();
    auto size = buffer_info.size;
    shape_t shape;
    strides_t strides;

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
    Tensor arr(buffer_info.size * dtype_to_size(dtype_from_pytype<T>()), shape,
               strides, _ptr, dtype_from_pytype<T>());
    return arr;
  }
  template <typename T> py::array_t<T> to_numpy() {
    if (!is_evaled()) {
      eval();
    }
    py::array_t<T> np_array(shape(), strides());
    std::memcpy(np_array.mutable_data(), get_base_ptr(), nbytes());
    return np_array;
  }

  static Tensor from_primitive(const std::shared_ptr<ADPrimitive> &primitive,
                               std::vector<Tensor> inputs) {
    return Tensor(primitive, inputs);
  }

  void eval() const;

  void backward(Tensor &tangent);

  bool is_evaled() const { return is_initialized(); }

  bool is_initialized() const { return _view->is_initialized(); }

  Tensor(const shape_t &shape, const DType dtype) {
    _view = std::make_shared<View>(shape, dtype);
  }

private:
  std::shared_ptr<View> _view = std::make_shared<View>();

  

  std::shared_ptr<ADNode> _ad_node =
      std::make_shared<ADNode>(); // creates a leaf node by default

  Tensor(const size_t nbytes, const shape_t &shape, const strides_t &strides,
         const std::shared_ptr<void> &ptr, DType dtype)
      : _view(std::make_shared<View>(ptr, nbytes, shape, strides, 0, dtype)) {}
  
  

  Tensor(const std::shared_ptr<ADPrimitive> &primitive,
         std::vector<Tensor> inputs) {
    _ad_node = std::make_shared<ADNode>(primitive, inputs);
  }
};

} // namespace pg
