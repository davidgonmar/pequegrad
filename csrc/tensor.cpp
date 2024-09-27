

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
           DType dtype, std::shared_ptr<device::Device> device)
    : _ptr(ptr), _nbytes(nbytes), _shape(shape), _strides(strides),
      _offset(offset), _dtype(dtype), _initialized(true), _device(device) {}

ADNode::ADNode(std::shared_ptr<ADPrimitive> primitive,
               std::vector<Tensor> children)
    : _primitive(primitive), _children(children) {}

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
typedef void (*FunPtr)();

std::vector<Tensor> &ADNode::children() { return _children; }
Tensor Tensor::from_primitive_one(const std::shared_ptr<ADPrimitive> &primitive,
                                  std::vector<Tensor> inputs,
                                  std::shared_ptr<device::Device> _device) {
  if (inputs.size() == 0) {
    PG_CHECK_ARG(_device != nullptr,
                 "Device must be specified for leaf nodes.");
  }

  auto device = _device != nullptr ? _device : inputs[0].device();
  Tensor t = Tensor(primitive, inputs, 0, device);

  // only check if it is not a ToDevice op. This shouldbe cleaner with primitve
  // traits
  if (true || primitive->str() != "ToDevice") {
    for (const Tensor &input : inputs) {
      PG_CHECK_ARG(input.device() == device,
                   "All inputs to a primitive must be on the same device, got ",
                   input.device()->str(), " and ", device->str(),
                   "for primitive", primitive->str());
    }
  }

  ADPrimitive *primitive_ptr = primitive.get();
  std::vector<View> vs = primitive_ptr->precompute(inputs);
  PG_CHECK_RUNTIME(vs.size() == 1, "precompute must return a single view");
  t.set_view(vs[0]);

  // maybe override device
  if (_device != nullptr) {
    t.view_ptr()->set_device(_device);
  }
  // check if primitive is marked as eager
  if (primitive->eager()) {
    t.eval(false);
  }
  return t;
}

Tensor Tensor::from_primitive_numpy(const shape_t &shape, DType dtype,
                                    const strides_t &strides,
                                    const py::buffer_info &buffer_info,
                                    const size_t size,
                                    std::shared_ptr<device::Device> device) {
  Tensor a = Tensor::from_primitive_one(
      std::make_shared<FromNumpy>(shape, dtype, strides, buffer_info.ptr, size,
                                  device),
      {}, device);
  return a;
}
std::vector<Tensor>
Tensor::from_primitive_multiple(const std::shared_ptr<ADPrimitive> &primitive,
                                std::vector<Tensor> inputs,
                                std::shared_ptr<device::Device> _device) {
  if (inputs.size() == 0) {
    PG_CHECK_ARG(_device != nullptr,
                 "Device must be specified for leaf nodes.");
  }

  std::shared_ptr<device::Device> device =
      _device != nullptr ? _device : inputs[0].device();
  std::vector<View> vs = primitive->precompute(inputs);
  int nouts = vs.size();
  std::vector<Tensor> tensors;
  for (int i = 0; i < nouts; i++) {
    Tensor t = Tensor(primitive, inputs, i, device);
    for (const Tensor &input : inputs) {
      PG_CHECK_ARG(input.device() == device,
                   "All inputs to a primitive must be on the same device, got ",
                   device_to_string(input.device()), " and ",
                   device_to_string(device));
    }
    t.set_view(vs[i]);
    t.view_ptr()->set_device(device);
    // check if primitive is marked as eager
    if (primitive->eager()) {
      t.eval(false);
    }
    tensors.push_back(t);
  }

  // for each tensor, set siblings to be the other tensors
  for (int i = 0; i < nouts; i++) {
    // copy the tensors list
    std::vector<Tensor> siblings = tensors;
    // remove the current tensor
    siblings.erase(siblings.begin() + i);
    tensors[i].ad_node()->set_siblings(siblings);
  }

  return tensors;
}

void Tensor::_inplace_as_copy(Tensor other) {
  auto newprim = std::make_shared<Copy>();
  this->ad_node()->set_primitive(newprim);
  this->ad_node()->set_children({other});
}

Tensor Tensor::eval(bool detach) {
  if (is_evaled()) {
    if (detach) {
      this->detach_();
    }
    return *this;
  }
  auto primitive = _ad_node->primitive();
  auto this_device = this->device();
  std::vector<Tensor> &children = _ad_node->children();
  for (Tensor &child : children) {
    child.eval(detach);
  }
  // outputs is just `this` tensor and the siblings (ordered by position)
  std::vector<Tensor> outputs = this->ad_node()->siblings();
  outputs.insert(outputs.begin(), *this);
  // sort by position
  std::sort(outputs.begin(), outputs.end(),
            [](const Tensor &a, const Tensor &b) {
              return a.ad_node()->position() < b.ad_node()->position();
            });
  switch (this_device->kind()) {
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
    throw std::runtime_error(
        "Tangent shape does not match tensor shape, got: " + tangent.str() +
        " and " + output.str());
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
          if (tangents.count(tensor) == 0) {
            // then gradient is zero
            flattened_tangents.push_back(
                fill(tensor.shape(), tensor.dtype(), 0, tensor.device()));
          } else {
            flattened_tangents.push_back(tangents.at(tensor));
          }
        }
        return flattened_tangents;
      };
  if (output.ad_node()->is_leaf()) {
    return flatten_tangents(tangents_map);
  }

  std::set<Tensor, decltype(tensorComparator)> visited(tensorComparator);
  std::vector<Tensor> nodes;

  std::function<void(Tensor)> toposort = [&](Tensor t) -> void {
    visited.insert(t);
    for (auto child : t.ad_node()->children()) {
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
    if (node.ad_node()->is_leaf()) {
      continue;
    }
    ADPrimitive *primitive = node.ad_node()->primitive().get();
    std::vector<Tensor> children = node.ad_node()->children();
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
     << ", position=" << _ad_node->position() << ")";
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

void ADNode::set_siblings(const std::vector<Tensor> &siblings) {
  // check that all siblings have same shape
  _siblings = siblings;
  std::sort(_siblings.begin(), _siblings.end(),
            [](const Tensor &a, const Tensor &b) {
              return a.ad_node()->position() < b.ad_node()->position();
            });
  // filter to make sure self is not marked as sibling
  _siblings.erase(std::remove_if(_siblings.begin(), _siblings.end(),
                                 [this](const Tensor &t) {
                                   return t.ad_node()->position() == _position;
                                 }),
                  _siblings.end());
}

Tensor::Tensor(Tensor &&other) {
  _view = std::move(other._view);
  _ad_node = std::move(other._ad_node);
  id = other.id;
}
Tensor::Tensor(const std::shared_ptr<ADPrimitive> &primitive,
               std::vector<Tensor> inputs, int position,
               std::shared_ptr<device::Device> device) {
  _ad_node = std::make_shared<ADNode>(primitive, inputs);
}

std::shared_ptr<ADNode> Tensor::ad_node() const {
  return _ad_node == nullptr ? std::make_shared<ADNode>() : _ad_node;
}
void ADNode::set_children(const std::vector<Tensor> &children, bool prop) {
  _children = children;
  // also set for each sibling
  if (!prop)
    return;
  for (Tensor &sibling : _siblings) {
    sibling.ad_node()->set_children(children, false);
  }
}

void ADNode::replace_child(const Tensor &old_child, const Tensor &new_child) {
  // we need to specifically find the one with the same id
  for (size_t i = 0; i < _children.size(); i++) {
    if (_children[i].id == old_child.id) {
      _children[i] = Tensor(new_child); // copy
      return;
    }
  }

  std::string childstr;
  for (const Tensor &child : _children) {
    childstr += child.str() + ", ";
  }
  PG_CHECK_RUNTIME(
      false,
      "Child not found in ADNode::replace_child. Old child: ", old_child.str(),
      " New child: ", new_child.str(), "List of children: ", childstr);
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
