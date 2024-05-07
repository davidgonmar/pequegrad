

#include "tensor.hpp"
#include "ad_primitives.hpp"
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

std::vector<Tensor> ADNode::children() const { return _children; }

Tensor Tensor::eval(bool force) const {
  if (is_evaled() && !force || ad_node().is_leaf()) {
    return *this;
  }
  ADPrimitive *primitive = (_ad_node->primitive().get());
  const device::DeviceKind this_device = this->device();
  std::vector<Tensor> children = _ad_node->children();
  for (const Tensor &child : children) {
    PG_CHECK_RUNTIME(child.device() == this_device,
                     "All children must be on the same device");
    child.eval(force);
  }
  // outputs is just `this` tensor
  std::vector<Tensor> outputs = {*this};
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
    // check if we already have computed the gradient for all required tensors
    // (even if they are not in the nodes list)
    bool all_grads = true;
    for (auto tensor : required_tensors) {
      if (tangents_map.count(tensor) == 0) {
        all_grads = false;
        break;
      }
    }
    if (all_grads) {
      break;
    }
  }

  return flatten_tangents(tangents_map);
}

std::string Tensor::str() const {
  std::stringstream ss;
  ss << "Tensor(shape=" << vec_to_string(shape())
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

Tensor::Tensor(const std::shared_ptr<ADPrimitive> &primitive,
               std::vector<Tensor> inputs) {
  _ad_node = std::make_shared<ADNode>(primitive, inputs);
  device::DeviceKind device = inputs[0].device();
  for (const Tensor &input : inputs) {
    PG_CHECK_ARG(input.device() == device,
                 "All inputs to a primitive must be on the same device, got ",
                 device_to_string(input.device()), " and ",
                 device_to_string(device));
  }
  this->_view->set_device(device);
  ADPrimitive *primitive_ptr = primitive.get();
  std::vector<shape_t> shape = primitive_ptr->infer_output_shapes(inputs);
  PG_CHECK_RUNTIME(shape.size() == 1, "Primitive must return a single shape");
  this->_view->set_shape(shape[0]);
  this->_view->set_dtype(primitive_ptr->infer_output_dtypes(inputs)[0]);
}

ADNode &Tensor::ad_node() const { return *_ad_node; }
} // namespace pg
