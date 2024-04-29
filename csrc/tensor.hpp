
#pragma once
#include "cpu/mem.hpp"
#include "cuda/mem.hpp"
#include "device.hpp"
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
  if (!is_initialized()) {                                                     \
    std::cout << "Exception on line " << __LINE__ << " in file " << __FILE__   \
              << ": " << msg << std::endl;                                     \
    throw std::runtime_error("not initialized!!!!");                           \
  }

namespace pg {

class ADPrimitive; // forward declaration
class Tensor;

namespace py = pybind11;
class View {

public:
  bool is_contiguous() const {
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
  void *get_base_ptr() const;
  void set_ptr(const std::shared_ptr<void> &ptr) { _ptr = ptr; }
  std::shared_ptr<void> shared_ptr() const;
  shape_t shape() const;
  void set_dtype(DType dtype) { _dtype = dtype; }
  strides_t strides() const;

  size_t numel() {
    shape_t s = shape();
    return std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>());
  }
  size_t offset() const;

  size_t nbytes() const;

  DType dtype() const;

  size_t ndim() const { return _shape.size(); }
  size_t numel() const {
    return std::accumulate(_shape.begin(), _shape.end(), 1,
                           std::multiplies<size_t>());
  }
  void set_device(device::DeviceKind device) { _device = device; }
  void set_shape(const shape_t &shape) { _shape = shape; }

  device::DeviceKind device() const { return _device; }

  View(const std::shared_ptr<void> &ptr, const size_t nbytes,
       const shape_t &shape, const strides_t &strides, const size_t offset,
       DType dtype, device::DeviceKind device);

  View(const shape_t shape, DType dtype, device::DeviceKind device) {
    _shape = shape;
    size_t nbytes = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<size_t>()) *
                    dtype_to_size(dtype);
    _nbytes = nbytes;
    _strides = _compute_natural_strides(shape, dtype);
    _ptr = device::allocate(nbytes, device);
    _device = device;
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
    _device = view->_device;
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
    _device = other._device;
    _initialized = other._initialized;
  }

  View(View &&other) {
    _ptr = std::move(other._ptr);
    _nbytes = other._nbytes;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _offset = other._offset;
    _dtype = other._dtype;
    _device = other._device;
    _initialized = other._initialized;
  }

  View &operator=(const View &other) {
    _ptr = other._ptr;
    _nbytes = other._nbytes;
    _shape = other._shape;
    _strides = other._strides;
    _offset = other._offset;
    _dtype = other._dtype;
    _device = other._device;
    _initialized = other._initialized;
    return *this;
  }

  View &operator=(View &&other) {
    _ptr = std::move(other._ptr);
    _nbytes = other._nbytes;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _offset = other._offset;
    _dtype = other._dtype;
    _device = other._device;
    _initialized = other._initialized;
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
  device::DeviceKind _device;
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
  shape_t inferred_shape() const { return _inferred_shape; }

  // Copy and move constructors
  ADNode(const ADNode &other) {
    _primitive = other._primitive;
    _children = other._children;
    _grad = other._grad;
    _inferred_shape = other._inferred_shape;
  }

  ADNode(ADNode &&other) {
    _primitive = std::move(other._primitive);
    _children = std::move(other._children);
    _grad = std::move(other._grad);
    _inferred_shape = std::move(other._inferred_shape);
  }

private:
  std::shared_ptr<ADPrimitive> _primitive = nullptr;
  std::vector<Tensor> _children;
  std::weak_ptr<Tensor> _grad;
  shape_t _inferred_shape;
};

// forward declaration
Tensor t(const Tensor &t);
class Tensor {

public:
  std::vector<Tensor> children() const { return _ad_node->children(); }
  bool requires_grad() const { return _requires_grad; }
  ADNode &ad_node() const;
  void assign(const Tensor &other) {
    if (!other.is_evaled()) {
      other.eval();
    }
    PG_CHECK_RUNTIME(other.device() == device(),
                     "Cannot assign tensors on different devices.");
    PG_CHECK_RUNTIME(other.dtype() == dtype(),
                     "Cannot assign tensors of different dtypes.");
    PG_CHECK_RUNTIME(other.shape() == shape(),
                     "Cannot assign tensors of different shapes.");
    _view = other._view;
    _ad_node = std::make_shared<ADNode>(); // reset gradient information
  }
  Tensor detach() {
    if (!is_evaled()) {
      throw std::runtime_error("Cannot detach unevaluated tensor.");
    }
    Tensor detached = *this;
    detached._ad_node = std::make_shared<ADNode>(); // creates a leaf node
    return detached;
  }

  Tensor copy_but_lose_grad_info() {
    Tensor copy = *this;
    copy._ad_node = std::make_shared<ADNode>(copy._ad_node->primitive(),
                                             copy._ad_node->children());

    return copy;
  }
  Tensor detach_() {
    if (!is_evaled()) {
      throw std::runtime_error("Cannot detach unevaluated tensor.");
    }
    _ad_node = std::make_shared<ADNode>(); // creates a leaf node
    return *this;
  }
  Tensor T() const { return t(*this); }

  size_t numel() const {
    shape_t s = shape();
    return std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>());
  }

  size_t ndim() const { return shape().size(); }
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

  long long get_next_id() {
    static long long id = 0;
    return id++;
  }

  long long id = get_next_id();
  // Copy and move constructors
  Tensor(const Tensor &other) {
    _view = other._view;
    _ad_node = other._ad_node;
    id = other.id;
  }

  Tensor(Tensor &&other) {
    _view = std::move(other._view);
    _ad_node = std::move(other._ad_node);
    id = other.id;
  }

  Tensor &operator=(const Tensor &other) {
    _view = other._view;
    _ad_node = other._ad_node;
    id = other.id;
    return *this;
  }

  Tensor &operator=(Tensor &&other) {
    _view = std::move(other._view);
    _ad_node = std::move(other._ad_node);
    id = other.id;
    return *this;
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
    //_throw_if_not_initialized("shape() called on uninitialized tensor.");
    return _view->shape();
  }

  strides_t strides() const {
    _throw_if_not_initialized("strides() called on uninitialized tensor.");
    return view().strides();
  }

  size_t offset() const {
    _throw_if_not_initialized("offset() called on uninitialized tensor.");
    return view().offset();
  }

  size_t nbytes() const {
    _throw_if_not_initialized("nbytes() called on uninitialized tensor.");
    return view().nbytes();
  }

  DType dtype() const;

  void *get_base_ptr() const {
    _throw_if_not_initialized("get_base_ptr() called on uninitialized tensor.");
    return view().get_base_ptr();
  }

  device::DeviceKind device() const {
    // We need to know device before initialization
    return _view->device();
  }

  void set_requires_grad(bool requires_grad) {
    this->_requires_grad = requires_grad;
  }

  template <typename T> T *get_casted_base_ptr() const {
    _throw_if_not_initialized(
        "get_casted_base_ptr() called on uninitialized tensor.");
    if (dtype_from_cpptype<T>() != this->dtype()) {
      throw std::runtime_error("Cannot cast pointer to different dtype.");
    }
    return static_cast<T *>(view().get_base_ptr());
  }
  template <typename T>
  static Tensor
  from_numpy(py::array_t<T> np_array, bool requires_grad = false,
             device::DeviceKind device = device::DeviceKind::CPU) {
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

    auto _ptr = device::allocate(size * dtype_to_size(dtype_from_pytype<T>()),
                                 device::DeviceKind::CPU);
    std::memcpy(_ptr.get(), buffer_info.ptr, size * sizeof(T));
    Tensor arr(buffer_info.size * dtype_to_size(dtype_from_pytype<T>()), shape,
               strides, _ptr, dtype_from_pytype<T>(), device::DeviceKind::CPU);
    return arr.to(device);
  }
  template <typename T> py::array_t<T> to_numpy() {
    if (!is_evaled()) {
      eval();
    }
    // TODO -- maybe dont copy 2 times
    if (device() != device::DeviceKind::CPU) {
      return to_cpu().to_numpy<T>();
    }
    py::array_t<T> np_array(shape(), strides());
    std::memcpy(np_array.mutable_data(), get_base_ptr(), nbytes());
    return np_array;
  }

  Tensor to(device::DeviceKind device) {
    if (device == device::DeviceKind::CPU) {
      return to_cpu();
    } else if (device == device::DeviceKind::CUDA) {
      return to_cuda();
    }
    throw std::runtime_error("Unsupported device type.");
  }

  Tensor to_(device::DeviceKind _device) {
    // TODO -- Make this safer
    if (device() == _device) {
      return *this;
    }
    if (!is_initialized()) {
      throw std::runtime_error(
          "Cannot move uninitialized tensor. Eval it first.");
    }
    if (device() == device::DeviceKind::CUDA) {
      size_t nbytes = this->nbytes();
      auto new_ptr = device::allocate(nbytes, device::DeviceKind::CPU);
      copy_from_cuda_to_cpu(view().shared_ptr(), new_ptr, nbytes);
      this->_view->set_device(device::DeviceKind::CPU);
      this->_view->set_ptr(new_ptr);
    } else if (device() == device::DeviceKind::CPU) {
      size_t nbytes = this->nbytes();
      auto new_ptr = device::allocate(nbytes, device::DeviceKind::CUDA);
      copy_from_cpu_to_cuda(view().shared_ptr(), new_ptr, nbytes);
      this->_view->set_device(device::DeviceKind::CUDA);
      this->_view->set_ptr(new_ptr);
    }
    return *this;
  }

  std::string str() const;

  Tensor to_cpu() {
    if (device() == device::DeviceKind::CPU) {
      return *this;
    }
    if (!is_initialized()) {
      throw std::runtime_error(
          "Cannot move uninitialized tensor. Eval it first.");
    }
    size_t nbytes = this->nbytes();
    auto new_ptr = device::allocate(nbytes, device::DeviceKind::CPU);
    copy_from_cuda_to_cpu(view().shared_ptr(), new_ptr, nbytes);
    return Tensor(nbytes, shape(), strides(), new_ptr, dtype(),
                  device::DeviceKind::CPU);
  }

  Tensor to_cuda() {
    if (device() == device::DeviceKind::CUDA) {
      return *this;
    }
    if (!is_initialized()) {
      throw std::runtime_error(
          "Cannot move uninitialized tensor. Eval it first.");
    }
    size_t nbytes = this->nbytes();
    auto new_ptr = device::allocate(nbytes, device::DeviceKind::CUDA);
    copy_from_cpu_to_cuda(view().shared_ptr(), new_ptr, nbytes);
    return Tensor(nbytes, shape(), strides(), new_ptr, dtype(),
                  device::DeviceKind::CUDA);
  }

  static Tensor from_primitive(const std::shared_ptr<ADPrimitive> &primitive,
                               std::vector<Tensor> inputs) {

    return Tensor(primitive, inputs);
  }

  Tensor eval() const;

  Tensor() { throw std::runtime_error("Cannot create empty tensor."); }
  bool is_evaled() const { return is_initialized(); }

  bool is_initialized() const { return _view->is_initialized(); }

  Tensor(const shape_t &shape, const DType dtype, device::DeviceKind device) {
    _view = std::make_shared<View>(shape, dtype, device);
  }

private:
  bool _requires_grad = false;
  std::shared_ptr<View> _view = std::make_shared<View>();

  std::shared_ptr<ADNode> _ad_node =
      std::make_shared<ADNode>(); // creates a leaf node by default

  Tensor(const size_t nbytes, const shape_t &shape, const strides_t &strides,
         const std::shared_ptr<void> &ptr, DType dtype,
         device::DeviceKind device)
      : _view(std::make_shared<View>(ptr, nbytes, shape, strides, 0, dtype,
                                     device)) {}

  Tensor(const std::shared_ptr<ADPrimitive> &primitive,
         std::vector<Tensor> inputs);
};

std::vector<Tensor> grads(const std::vector<Tensor> &required_tensors,
                          const Tensor output,
                          const std::optional<Tensor> &tangent = std::nullopt);

} // namespace pg
