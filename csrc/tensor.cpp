

#include "tensor.hpp"
#include "ad_primitives.hpp"
#include "cuda/cuda_utils.cuh"
#include "ops.hpp"
#include <algorithm>
#include <vector>

namespace pg {

class ADPrimitive; // forward declaration

void *View::get_base_ptr() const {
  return static_cast<char *>(_ptr.get()) + _offset;
}

std::shared_ptr<void> View::shared_ptr() const { return _ptr; }

shape_t View::shape() const { return _shape; }

strides_t View::strides() const { return _strides; }

size_t View::offset() const { return _offset; }

size_t View::nbytes() const { return _nbytes; }

DType View::dtype() const { return _dtype; }
View::View(const std::shared_ptr<void> &ptr, const size_t nbytes,
           const shape_t &shape, const strides_t &strides, const size_t offset,
           DType dtype, device::DeviceKind device)
    : _ptr(ptr), _nbytes(nbytes), _shape(shape), _strides(strides),
      _offset(offset), _dtype(dtype), _initialized(true), _device(device) {}

ADNode::ADNode(std::shared_ptr<ADPrimitive> primitive,
               std::vector<Tensor> children)
    : _primitive(std::move(primitive)), _children(std::move(children)) {}

ADNode ADNode::create_leaf() { return ADNode(); }

std::shared_ptr<ADPrimitive> ADNode::primitive() const {
  if (_primitive == nullptr) {
    return std::make_shared<ADPrimitive>();
  }
  return _primitive;
}

ViewOptions &ViewOptions::like(const Tensor &t) {
  _dtype = t.dtype();
  _device = t.device();
  _shape = t.shape();
  _strides = t.strides();
  _offset = t.offset();
  _nbytes = t.nbytes();
  _strides_set = true;
  return *this;
}

ViewOptions &ViewOptions::like_natural(const Tensor &t) {
  _dtype = t.dtype();
  _device = t.device();
  _shape = t.shape();
  _strides = _compute_natural_strides(t.shape(), t.dtype());
  _offset = 0;
  _nbytes = compute_nbytes(t.shape(), t.dtype());
  _strides_set = true;
  return *this;
}

std::vector<Tensor> &ADNode::children() { return _children; }
Tensor Tensor::from_primitive(const std::shared_ptr<ADPrimitive> &primitive,
                              std::vector<Tensor> inputs,
                              std::optional<device::DeviceKind> device) {
  if (inputs.size() == 0) {
    PG_CHECK_ARG(device.has_value(),
                 "Device must be specified for leaf nodes.");
  }

  Tensor t = Tensor(primitive, inputs, device);
  // check if primitive is marked as eager
  if (primitive->eager()) {
    t.eval(false);
  }
  return t;
}
Tensor Tensor::eval(bool detach) {
  if (is_evaled()) {
    if (detach) {
      this->detach_();
    }
    return *this;
  }
  ADPrimitive *primitive = (_ad_node->primitive().get());
  const device::DeviceKind this_device = this->device();
  std::vector<Tensor> &children = _ad_node->children();
  for (Tensor &child : children) {
    PG_CHECK_RUNTIME(child.device() == this_device,
                     "All children must be on the same device");
    child.eval(detach);
  }
  // outputs is just `this` tensor
  std::vector<Tensor> outputs = {*this};
  // assert all children are on the same device
  for (Tensor &child : children) {
    PG_CHECK_RUNTIME(child.device() == this_device,
                     "All children must be on the same device");
  }
  switch (this_device) {

  case device::DeviceKind::CPU:
    primitive->dispatch_cpu(children, outputs);
    break;
  case device::DeviceKind::CUDA:
    primitive->dispatch_cuda(children, outputs);
    break;
  default:
    throw std::runtime_error("Unsupported device");
  }
  if (detach) {
    this->detach_();
  }
  return *this;
}

std::vector<Tensor> grads(const std::vector<Tensor> &required_tensors,
                          const Tensor output,
                          const std::optional<Tensor> &_tangent) {
  Tensor tangent = _tangent.has_value() ? _tangent.value()
                                        : fill(output.shape(), output.dtype(),
                                               1, output.device());
  auto tensorComparator = [](const Tensor &lhs, const Tensor &rhs) {
    // ??? weird but works. todo figure this out
    return lhs.id < rhs.id;
  };

  if (tangent.shape() != output.shape()) {
    throw std::runtime_error("Tangent shape does not match tensor shape");
  }
  std::map<Tensor, Tensor, decltype(tensorComparator)> tangents_map(
      tensorComparator);
  auto accum_grad = [&](Tensor ten, Tensor tan) {
    bool has_grad = tangents_map.count(ten) == 1;
    PG_CHECK_RUNTIME(tan.shape() == ten.shape(),
                     "Tangent shape does not match tensor shape, got: ",
                     tan.str(), " and ", ten.str());
    if (!has_grad) {
      tangents_map.insert({ten, tan});
    } else {
      Tensor tmp = add(tangents_map[ten], tan);
      tangents_map.insert_or_assign(ten, tmp);
    }
  };
  accum_grad(output, tangent);
  auto flatten_tangents =
      [&](std::map<Tensor, Tensor, decltype(tensorComparator)> tangents) {
        std::vector<Tensor> flattened_tangents;
        for (auto tensor : required_tensors) {
          PG_CHECK_RUNTIME(
              tangents.count(tensor) == 1,
              "Tangent not found for required tensor: ", tensor.str());
          flattened_tangents.push_back(tangents.at(tensor));
        }
        return flattened_tangents;
      };
  if (output.ad_node().is_leaf()) {
    return flatten_tangents(tangents_map);
  }

  std::set<Tensor, decltype(tensorComparator)> visited(tensorComparator);
  std::vector<Tensor> nodes;

  std::function<void(Tensor)> toposort = [&](Tensor t) -> void {
    visited.insert(t);
    for (auto child : t.ad_node().children()) {
      if (!visited.count(child)) {
        // Calling the lambda function recursively
        toposort(child);
      }
    }
    nodes.push_back(t);
  };

  toposort(output);

  auto nodes_reversed = std::vector<Tensor>(nodes.rbegin(), nodes.rend());
  for (auto node : nodes_reversed) {
    if (node.ad_node().is_leaf()) {
      continue;
    }
    ADPrimitive *primitive = node.ad_node().primitive().get();
    std::vector<Tensor> children = node.ad_node().children();
    std::vector<Tensor> outputs = {node};
    PG_CHECK_RUNTIME(tangents_map.count(node) == 1,
                     "Tangent not found for node: ", node.str());
    std::vector<Tensor> tangents =
        primitive->backward(children, {tangents_map.at(node)}, outputs);
    PG_CHECK_RUNTIME(tangents.size() == children.size(),
                     "ADPrimitive::backward must return the same number of "
                     "tangents as inputs");
    for (size_t i = 0; i < children.size(); i++) {
      accum_grad(children[i], tangents[i]);
    }
  }

  return flatten_tangents(tangents_map);
}

std::string Tensor::str() const {
  std::stringstream ss;
  ss << "Tensor(shape=" << vec_to_string(shape())
     << ", strides=" << vec_to_string(strides())
     << ", dtype=" << dtype_to_string(dtype()) << ", device=" << device()
     << ", evaled=" << is_evaled()
     << ", primitive=" << _ad_node->primitive()->str() << ", id=" << this->id
     << ")";
  return ss.str();
}
DType Tensor::dtype() const {
  //_throw_if_not_initialized("dtype() called on uninitialized tensor, with
  // primitive: " + _ad_node->primitive()->str());
  return _view->dtype();
}
Tensor
Tensor::copy_graph(std::vector<Tensor> &inputs,
                   std::optional<std::shared_ptr<ADPrimitive>> primitive) {
  Tensor copy = Tensor(*this);
  // create new ad node
  copy._ad_node = std::make_shared<ADNode>(
      primitive.has_value() ? primitive.value() : this->_ad_node->primitive(),
      inputs);
  // reset view
  copy._view = std::make_shared<View>(*_view);
  copy._view->deallocate();
  copy.id = Tensor::get_next_id();
  return copy;
}

Tensor::Tensor(const std::shared_ptr<ADPrimitive> &primitive,
               std::vector<Tensor> inputs,
               std::optional<device::DeviceKind> _device) {
  _ad_node = std::make_shared<ADNode>(primitive, inputs);
  device::DeviceKind device =
      _device.has_value() ? _device.value() : inputs[0].device();
  for (const Tensor &input : inputs) {
    PG_CHECK_ARG(input.device() == device,
                 "All inputs to a primitive must be on the same device, got ",
                 device_to_string(input.device()), " and ",
                 device_to_string(device));
  }
  this->_view->set_device(device);
  ADPrimitive *primitive_ptr = primitive.get();
  View v = primitive_ptr->precompute(inputs)[0];
  this->_view = std::make_shared<View>(v);
  this->_view->set_device(device);
  // this->_view->set_shape(shape[0]);
  // this->_view->set_dtype(primitive_ptr->infer_output_dtypes(inputs)[0]);
}

ADNode &Tensor::ad_node() const { return *_ad_node; }
void ADNode::set_children(const std::vector<Tensor> &children) {
  _children = children;
}

void ADNode::replace_child(const Tensor &old_child, const Tensor &new_child) {
  // we need to specifically find the one with the same id
  for (size_t i = 0; i < _children.size(); i++) {
    if (_children[i].id == old_child.id) {
      _children[i] = new_child;
      return;
    }
  }

  PG_CHECK_RUNTIME(false, "Child not found in ADNode::replace_child");
}

void ADNode::set_primitive(const ADPrimitive &primitive) {
  _primitive = std::make_shared<ADPrimitive>(primitive);
}
void ADNode::set_primitive(std::shared_ptr<ADPrimitive> &primitive) {
  _primitive = primitive;
}
void ADNode::set_primitive(std::shared_ptr<ADPrimitive> &&primitive) {
  _primitive = std::move(primitive);
}
} // namespace pg
