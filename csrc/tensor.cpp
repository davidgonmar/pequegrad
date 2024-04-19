

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

const std::shared_ptr<Tensor> ADNode::grad() const {
  if (_grad == nullptr) {
    throw std::runtime_error("Gradient not yet computed");
  }
  return _grad;
}

void ADNode::accum_grad(Tensor &grad) {
  if (this->_grad == nullptr) {
    this->_grad = std::make_shared<Tensor>(grad);
    return;
  }
  Tensor tmp = add(*(this->grad().get()), grad);
  this->_grad = std::make_shared<Tensor>(tmp);
}

ADNode ADNode::create_leaf() { return ADNode(); }

std::shared_ptr<ADPrimitive> ADNode::primitive() const { return _primitive; }

std::vector<Tensor> ADNode::children() const { return _children; }

void Tensor::eval() const {
  if (is_evaled()) {
    return;
  }
  ADPrimitive *primitive = (_ad_node->primitive().get());
  const device::DeviceKind this_device = this->device();
  std::vector<Tensor> children = _ad_node->children();
  for (const Tensor &child : children) {
    PG_CHECK_RUNTIME(child.device() == this_device,
                     "All children must be on the same device");
    child.eval();
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
}

void Tensor::backward(Tensor &tangent) {
  if (!is_evaled()) {
    throw std::runtime_error("Cannot call backward on unevaluated tensor");
  }
  if (tangent.shape() != shape()) {
    throw std::runtime_error("Tangent shape does not match tensor shape");
  }
  this->_ad_node->accum_grad(tangent);
  if (this->_ad_node->is_leaf()) {
    return;
  }
  auto tensorComparator = [](const Tensor &lhs, const Tensor &rhs) {
    // ??? weird but works. todo figure this out
    return lhs._ad_node < rhs._ad_node;
  };

  std::set<Tensor, decltype(tensorComparator)> visited(tensorComparator);
  std::vector<Tensor> nodes;

  std::function<void(Tensor)> toposort = [&](Tensor t) -> void {
    visited.insert(t);
    for (auto child : t._ad_node->children()) {
      if (!visited.count(child)) {
        // Calling the lambda function recursively
        toposort(child);
      }
    }
    nodes.push_back(t);
  };

  toposort(*this);

  auto nodes_reversed = std::vector<Tensor>(nodes.rbegin(), nodes.rend());
  for (auto node : nodes_reversed) {
    if (node._ad_node->is_leaf()) {
      continue;
    }
    ADPrimitive *primitive = node._ad_node->primitive().get();
    std::vector<Tensor> children = node._ad_node->children();
    std::vector<Tensor> outputs = {node};
    std::vector<Tensor> tangents =
        primitive->backward(children, {node.grad()}, outputs);
    PG_CHECK_RUNTIME(tangents.size() == children.size(),
                     "ADPrimitive::backward must return the same number of "
                     "tangents as inputs");
    for (size_t i = 0; i < children.size(); i++) {
      children[i]._ad_node->accum_grad(tangents[i]);
    }
  }
}

Tensor::Tensor(const std::shared_ptr<ADPrimitive>& primitive,
    std::vector<Tensor> inputs) {
    _ad_node = std::make_shared<ADNode>(primitive, inputs);
    device::DeviceKind device = inputs[0].device();
    for (const Tensor& input : inputs) {
        PG_CHECK_ARG(input.device() == device,
            "All inputs to a primitive must be on the same device, got ",
            device_to_string(input.device()), " and ",
            device_to_string(device));
    }
    this->_view->set_device(device);
    try {
        ADPrimitive* primitive_ptr = primitive.get();
        std::vector<shape_t> shape = primitive_ptr->infer_output_shapes(inputs);
        PG_CHECK_RUNTIME(shape.size() == 1,
            "Primitive must return a single shape");
        this->_view->set_shape(shape[0]);
    }
    catch (const std::exception& e) {
        // todo -- this should always throw
    }
}
} // namespace pg
