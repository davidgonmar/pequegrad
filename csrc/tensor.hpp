
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
void tensor_precompute_again(Tensor &t);
Tensor as_contiguous(const Tensor &t); // forward declaration
Tensor to_device(const Tensor &t, std::shared_ptr<device::Device> device);
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
  template <typename T> T *get_casted_base_ptr() const {
    return static_cast<T *>(get_base_ptr());
  }
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
  void set_device(std::shared_ptr<device::Device> device) { _device = device; }
  void set_shape(const shape_t &shape) { _shape = shape; }
  void set_strides(const strides_t &strides) { _strides = strides; }
  void set_ptr(const std::shared_ptr<void> &ptr, size_t nbytes) {
    _ptr = ptr;
    _nbytes = nbytes;
    _initialized = true;
  }
  std::shared_ptr<device::Device> device() const { return _device; }

  View(const std::shared_ptr<void> &ptr, const size_t nbytes,
       const shape_t &shape, const strides_t &strides, const size_t offset,
       DType dtype, std::shared_ptr<device::Device> device);

  bool is_initialized() const { return _initialized; }
  bool is_evaled() const { return is_initialized(); }
  bool is_dense() const {
    // dense means that it might not be contiguous, but
    // there are no holes in the array
    // that is, the total number of elements is equal to
    // the size of the underlying storage
    size_t total_in_storage = nbytes();
    size_t total_size_in_bytes = numel() * dtype_to_size(dtype());
    if (!is_evaled()) {
      throw std::runtime_error("Cannot check if unevaluated view is dense.");
    }
    return total_in_storage == total_size_in_bytes;
  }

  View(const shape_t shape, const strides_t strides, const size_t offset,
       const DType dtype, std::shared_ptr<device::Device> device)
      : _shape(shape), _strides(strides), _offset(offset), _dtype(dtype),
        _device(device) {
    size_t nbytes = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<size_t>()) *
                    dtype_to_size(dtype);
    _nbytes = nbytes;
    _ptr = device->allocate(nbytes);
    _initialized = true;

    PG_CHECK_RUNTIME(is_dense(), "Cannot create view with holes.");
  }

  View(const shape_t shape, const strides_t strides, const DType dtype,
       std::shared_ptr<device::Device> device)
      : _shape(shape), _strides(strides), _dtype(dtype), _device(device) {
    size_t nbytes = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<size_t>()) *
                    dtype_to_size(dtype);
    _nbytes = nbytes;
    _ptr = device->allocate(nbytes);
    _offset = 0;
    _initialized = true;

    PG_CHECK_RUNTIME(is_dense(), "Cannot create view with holes.");
  }

  View(const shape_t shape, DType dtype, std::shared_ptr<device::Device> device,
       bool init = true) {
    _shape = shape;

    if (init) {
      size_t nbytes = std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<size_t>()) *
                      dtype_to_size(dtype);
      _nbytes = nbytes;
      _strides = _compute_natural_strides(shape, dtype);
      _ptr = device->allocate(nbytes);
      _offset = 0;
    }
    _device = device;
    _initialized = init;
    _dtype = dtype;
  }

  void allocate() {
    size_t nbytes = compute_nbytes(_shape, _dtype);
    _ptr = _device->allocate(nbytes);
    _nbytes = nbytes;
    _initialized = true;
  }

  void deallocate() {
    if (_ptr) {
      _ptr = nullptr;
      _nbytes = 0;
      _initialized = false;
    }
  }

  void init_view(const std::shared_ptr<View> &view) {
    _ptr = view->_ptr;
    _nbytes = view->_nbytes;
    _shape = view->_shape;
    _strides = view->_strides;
    _offset = view->_offset;
    _dtype = view->_dtype;
    _initialized = true;
    _device = view->_device;
  }

  void copy_meta(const View &view) {
    _shape = view._shape;
    _strides = view._strides;
    _offset = view._offset;
    _dtype = view._dtype;
    _device = view._device;
  }

  View() = default;

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

  View(size_t nbytes, shape_t shape, strides_t strides, size_t offset,
       DType dtype, std::shared_ptr<device::Device> device)
      : _nbytes(nbytes), _shape(shape), _strides(strides), _offset(offset),
        _dtype(dtype), _device(device) {
    _initialized = false;
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
  std::shared_ptr<device::Device> _device;
};

// builder pattern for view options

class ViewOptions {
public:
  ViewOptions &dtype(DType dtype) {
    _dtype = dtype;
    return *this;
  }
  ViewOptions &device(std::shared_ptr<device::Device> device) {
    _device = device;
    return *this;
  }
  ViewOptions &shape(const shape_t &shape) {
    _shape = shape;
    return *this;
  }
  ViewOptions &strides(const strides_t &strides) {
    _strides = strides;
    _strides_set = true;
    return *this;
  }

  ViewOptions &with_natural_strides() {
    _strides = _compute_natural_strides(_shape, _dtype);
    _strides_set = true;
    return *this;
  }

  ViewOptions &offset(size_t offset) {
    _offset = offset;
    return *this;
  }
  ViewOptions &nbytes(size_t nbytes) {
    _nbytes = nbytes;
    return *this;
  }

  ViewOptions &like(const View &view) {
    _dtype = view.dtype();
    _device = view.device();
    _shape = view.shape();
    _strides = view.strides();
    _offset = view.offset();
    _nbytes = view.nbytes();
    _strides_set = true;
    return *this;
  }

  ViewOptions &like(const Tensor &t);
  ViewOptions &like_natural(const Tensor &t);

  ViewOptions &like_natural(const View &view) {
    _dtype = view.dtype();
    _device = view.device();
    _shape = view.shape();
    _strides = _compute_natural_strides(view.shape(), view.dtype());
    _offset = view.offset();
    _nbytes = compute_nbytes(view.shape(), view.dtype());
    _strides_set = true;
    return *this;
  }

  View build() {
    if (!_strides_set) {
      _strides = _compute_natural_strides(_shape, _dtype);
    }
    if (_device == nullptr) {
      PG_CHECK_RUNTIME(false, "Device not set for ViewOptions");
    }
    if (_nbytes == 0) {
      _nbytes = compute_nbytes(_shape, _dtype);
    }
    return View(_nbytes, _shape, _strides, _offset, _dtype, _device);
  }

  ViewOptions() = default;

private:
  DType _dtype = DType::Float32;
  std::shared_ptr<device::Device> _device = nullptr;
  shape_t _shape;
  strides_t _strides;
  size_t _offset = 0;
  size_t _nbytes = 0;
  bool _strides_set = false;
};
class ADNode {
public:
  explicit ADNode(std::shared_ptr<ADPrimitive> primitive,
                  std::vector<Tensor> children);
  explicit ADNode() = default;
  static ADNode create_leaf();
  bool is_leaf() { return _primitive == nullptr || _children.empty(); };
  std::shared_ptr<ADPrimitive> primitive() const;
  std::vector<Tensor> &children();
  shape_t inferred_shape() const { return _inferred_shape; }

  void detach_() {
    // remove children
    _children.clear();
    set_primitive(nullptr);
  }
  // Copy and move constructors
  ADNode(const ADNode &other) {
    set_primitive(other.primitive());
    _children = other._children;
    _siblings = other._siblings;
    _position = other._position;
    _inferred_shape = other._inferred_shape;
  }

  ADNode(ADNode &&other) {
    set_primitive(std::move(other.primitive()));
    _children = std::move(other._children);
    _inferred_shape = std::move(other._inferred_shape);
    _position = other._position;
    _siblings = std::move(other._siblings);
  }

  ADNode &operator=(const ADNode &other) {
    set_primitive(other.primitive());
    _children = other._children;
    _inferred_shape = other._inferred_shape;
    _position = other._position;
    _siblings = other._siblings;
    return *this;
  }

  ADNode &operator=(ADNode &&other) {
    set_primitive(std::move(other.primitive()));
    _children = std::move(other._children);
    _inferred_shape = std::move(other._inferred_shape);
    _position = other._position;
    _siblings = std::move(other._siblings);
    return *this;
  }

  void set_primitive(std::shared_ptr<ADPrimitive> &primitive);
  void set_primitive(const ADPrimitive &primitive);
  void set_primitive(std::shared_ptr<ADPrimitive> &&primitive);
  void set_children(const std::vector<Tensor> &children, bool prop = true);
  void replace_child(const Tensor &old_child, const Tensor &new_child);
  void set_position(int position) { _position = position; }
  void set_siblings(const std::vector<Tensor> &siblings);
  std::vector<Tensor> &siblings() { return _siblings; }

  int position() const { return _position; }

private:
  std::shared_ptr<ADPrimitive> _primitive = nullptr;
  std::vector<Tensor> _children = std::vector<Tensor>();
  std::vector<Tensor> _siblings = std::vector<Tensor>();
  shape_t _inferred_shape;
  int _position = 0;
};

// forward declaration
Tensor t(const Tensor &t);
Tensor astype(const Tensor &t, DType dtype);
class Tensor {

public:
  Tensor astype(DType dtype) const { return pg::astype(*this, dtype); }
  std::vector<Tensor> &children() const {
    PG_CHECK_RUNTIME(_ad_node != nullptr, "_ad_node == nullptr.");
    return _ad_node->children();
  }
  std::shared_ptr<ADNode> ad_node() const;
  void assign(Tensor other) {
    if (!other.is_evaled()) {
      other.eval();
    }
    PG_CHECK_RUNTIME(other.device() == device(),
                     "Cannot assign tensors on different devices, got " +
                         device_to_string(other.device()) + " and " +
                         device_to_string(device()));
    PG_CHECK_RUNTIME(other.dtype() == dtype(),
                     "Cannot assign tensors of different dtypes, got " +
                         dtype_to_string(other.dtype()) + " and " +
                         dtype_to_string(dtype()));
    PG_CHECK_RUNTIME(other.shape() == shape(),
                     "Cannot assign tensors of different shapes, got dst " +
                         this->str() + " and srd " + other.str());

    // we want to make our view shared ptr reference the view associated with
    // the other tensor
    _view->init_view(other._view);
  }

  void _inplace_as_copy(Tensor other);
  Tensor detach() {
    if (!is_evaled()) {
      throw std::runtime_error("Cannot detach unevaluated tensor.");
    }
    Tensor detached = *this;
    detached._ad_node = std::make_shared<ADNode>(); // creates a leaf node
    return detached;
  }

  Tensor inplace_update(const Tensor other) {
    /*
    EXPLANATION with ADD
    a = Tensor([1, 2, 3], id=0)
    b = Tensor([4, 5, 6], id=1)
    temp = a + b (Tensor([5, 7, 9], id=2))
    a.inplace_update(temp)
    then, a will be Tensor([5, 7, 9], id=2)
    and a will have as childs Tensor([1, 2, 3], id=0) and Tensor([4, 5, 6],
    id=1)
    */
    this->_ad_node = other._ad_node;
    this->_view = other._view;
    this->id = other.id;
    return *this;
  }
  Tensor copy_graph(
      std::vector<Tensor> &inputs,
      std::optional<std::shared_ptr<ADPrimitive>> primitive = std::nullopt);
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
    _ad_node->detach_();
    return *this;
  }
  // Assigns a new view with same dtype, device, and shape, but uninitialized
  // ptr, strides, etc
  void reset_view() {
    _view = std::make_shared<View>(shape(), dtype(), device(), false);
    PG_CHECK_RUNTIME(!is_initialized(), "View should be uninitialized.");
  }
  Tensor T() const { return t(*this); }

  size_t numel() const {
    shape_t s = shape();
    if (s.size() == 0)
      return 1;
    return std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>());
  }

  size_t ndim() const { return shape().size(); }
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

  Tensor(Tensor &&other);

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

  View view() const { return *_view; }
  void set_view(const View &view) { _view = std::make_shared<View>(view); }
  void copy_view_inplace(const View &view) {
    _view->set_shape(view.shape());
    _view->set_strides(view.strides());
    _view->set_dtype(view.dtype());
    _view->set_device(view.device());
  }
  void set_view(const std::shared_ptr<View> &view) { _view = view; }
  std::shared_ptr<View> view_ptr() const { return _view; }

  void init_view(std::shared_ptr<View> view) { _view->init_view(view); }

  shape_t shape() const {
    //_throw_if_not_initialized("shape() called on uninitialized tensor.");
    return _view->shape();
  }

  strides_t strides() const { return view().strides(); }

  size_t offset() const { return view().offset(); }

  size_t nbytes() const { return view().nbytes(); }

  DType dtype() const;

  void *get_base_ptr() const {
    _throw_if_not_initialized("get_base_ptr() called on uninitialized tensor.");
    return view().get_base_ptr();
  }

  std::shared_ptr<device::Device> device() const {
    // We need to know device before initialization
    return _view->device();
  }

  template <typename T> T *get_casted_base_ptr() const {
    _throw_if_not_initialized(
        "get_casted_base_ptr() called on uninitialized tensor.");

    if (dtype_from_cpptype<T>() != this->dtype()) {
      throw std::runtime_error("Cannot cast pointer to different dtype, got " +
                               dtype_to_string(this->dtype()) + " and " +
                               dtype_to_string(dtype_from_cpptype<T>()));
    }
    return static_cast<T *>(view().get_base_ptr());
  }

  static Tensor from_primitive_numpy(const shape_t &shape, DType dtype,
                                     const strides_t &strides,
                                     const py::buffer_info &buffer_info,
                                     const size_t size,
                                     std::shared_ptr<device::Device> device);

  template <typename T>
  static Tensor from_numpy(
      py::array_t<T> np_array,
      std::shared_ptr<device::Device> device = device::get_default_device()) {
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

    DType dtype = dtype_from_pytype<T>();
    return from_primitive_numpy(shape, dtype, strides, buffer_info, size,
                                device);
  }

  bool is_dense() const {
    // dense means that it might not be contiguous, but
    // there are no holes in the array
    // that is, the total number of elements is equal to
    // the size of the underlying storage
    size_t total_in_storage = nbytes();
    size_t total_size_in_bytes = numel() * dtype_to_size(dtype());
    if (!is_evaled()) {
      throw std::runtime_error("Cannot check if unevaluated tensor is dense.");
    }
    return total_in_storage == total_size_in_bytes;
  }

  Tensor contiguous() const { return as_contiguous(*this); }

  template <typename T> py::array_t<T> to_numpy() {
    if (!is_evaled()) {
      eval();
    }
    // TODO -- maybe dont copy 2 times
    if (!device::is_cpu(device())) {
      return to_cpu().to_numpy<T>();
    }
    // if is not dense, we need to copy to a dense tensor
    if (!is_contiguous()) {
      return contiguous().to_numpy<T>();
    }
    py::array_t<T> np_array(shape(), strides(), get_casted_base_ptr<T>());
    return np_array;
  }

  // Differentiable / graph-aware (jittable) operation
  // Differentiable / graph-aware (jittable) operation
  Tensor to(std::shared_ptr<device::Device> new_device) {
    return to_device(*this, new_device);
  }

  // This is inplace, not differentiable / graph-aware at the moment
  Tensor to_(std::shared_ptr<device::Device> _device) {
    // TODO -- Make this safer
    if (device() == _device) {
      return *this;
    }
    if (!is_initialized()) {
      this->eval();
    }
    if (device::is_cpu(_device)) {
      size_t nbytes = this->nbytes();
      auto new_ptr = _device->allocate(nbytes);
      copy_from_cuda_to_cpu(view().shared_ptr(), new_ptr, nbytes);
      this->_view->set_device(_device);
      this->_view->set_ptr(new_ptr);
    } else if (device::is_cuda(_device)) {
      size_t nbytes = this->nbytes();
      auto new_ptr = _device->allocate(nbytes);
      copy_from_cpu_to_cuda(view().shared_ptr(), new_ptr, nbytes);
      this->_view->set_device(_device);
      this->_view->set_ptr(new_ptr);
    }
    return *this;
  }

  std::string str() const;

  Tensor to_cpu(int idx = 0) {
    if (is_cpu(device())) {
      return *this;
    }
    if (!is_initialized()) {
      this->eval();
    }
    size_t nbytes = this->nbytes();
    auto str = "cpu:" + std::to_string(idx);
    auto new_ptr = device::from_str(str)->allocate(nbytes);
    copy_from_cuda_to_cpu(view().shared_ptr(), new_ptr, nbytes);
    return Tensor(nbytes, shape(), strides(), offset(), new_ptr, dtype(),
                  device::from_str(str));
  }

  Tensor to_cuda(int idx = 0) {
    if (is_cuda(device())) {
      return *this;
    }
    if (!is_initialized()) {
      this->eval();
    }
    size_t nbytes = this->nbytes();
    auto new_ptr =
        device::from_str("cuda:" + std::to_string(idx))->allocate(nbytes);
    copy_from_cpu_to_cuda(view().shared_ptr(), new_ptr, nbytes);
    return Tensor(nbytes, shape(), strides(), offset(), new_ptr, dtype(),
                  device::from_str("cuda:" + std::to_string(idx)));
  }

  static Tensor
  from_primitive_one(const std::shared_ptr<ADPrimitive> &primitive,
                     std::vector<Tensor> inputs,
                     std::shared_ptr<device::Device> device = nullptr);
  static std::vector<Tensor>
  from_primitive_multiple(const std::shared_ptr<ADPrimitive> &primitive,
                          std::vector<Tensor> inputs,
                          std::shared_ptr<device::Device> device = nullptr);
  Tensor eval(bool detach = true);

  Tensor() {}
  bool is_evaled() const { return is_initialized(); }

  bool is_initialized() const { return _view->is_initialized(); }

  Tensor(const shape_t &shape, const DType dtype,
         std::shared_ptr<device::Device> device)
      : _view(std::make_shared<View>(shape, dtype, device)) {}

  void set_ad_node(const ADNode &ad_node) {
    _ad_node = std::make_shared<ADNode>(ad_node);
  }
  void set_ad_node(std::shared_ptr<ADNode> ad_node) { _ad_node = ad_node; }

  Tensor(const size_t nbytes, const shape_t &shape, const strides_t &strides,
         const std::shared_ptr<void> &ptr, DType dtype,
         std::shared_ptr<device::Device> device)
      : _view(std::make_shared<View>(ptr, nbytes, shape, strides, 0, dtype,
                                     device)) {}
  Tensor(const size_t nbytes, const shape_t &shape, const strides_t &strides,
         size_t offset, const std::shared_ptr<void> &ptr, DType dtype,
         std::shared_ptr<device::Device> device)
      : _view(std::make_shared<View>(ptr, nbytes, shape, strides, offset, dtype,
                                     device)) {}

private:
  std::shared_ptr<View> _view = std::make_shared<View>();

  std::shared_ptr<ADNode> _ad_node =
      std::make_shared<ADNode>(); // creates a leaf node by default

  Tensor(const std::shared_ptr<ADPrimitive> &primitive,
         std::vector<Tensor> inputs, int position,
         std::shared_ptr<device::Device> device);
};

std::vector<Tensor> grads(const std::vector<Tensor> &required_tensors,
                          const Tensor output,
                          const std::optional<Tensor> &tangent = std::nullopt);

} // namespace pg
