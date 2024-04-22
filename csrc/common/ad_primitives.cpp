#include "ad_primitives.hpp"

namespace pg {
void BroadcastTo::dispatch_cpu(const std::vector<Tensor> &inputs,
                               std::vector<Tensor> &outputs) {
  if (inputs[0].shape() == _shape_to) {
    outputs[0].init_view(std::make_shared<View>(inputs[0].view()));
    return;
  }

  auto [view, broadcasted_axis, created_axes] =
      view::broadcasted_to(inputs[0].view(), _shape_to);
  outputs[0].init_view(std::make_shared<View>(view));
  this->_broadcasted_axes = broadcasted_axis;
  this->_created_axes = created_axes;
}

void BroadcastTo::dispatch_cuda(const std::vector<Tensor> &inputs,
                                std::vector<Tensor> &outputs) {
  if (inputs[0].shape() == _shape_to) {
    outputs[0].init_view(std::make_shared<View>(inputs[0].view()));
    return;
  }
  auto [view, broadcasted_axis, created_axes] =
      view::broadcasted_to(inputs[0].view(), _shape_to);
  outputs[0].init_view(std::make_shared<View>(view));
  this->_broadcasted_axes = broadcasted_axis;
  this->_created_axes = created_axes;
}

void Squeeze::dispatch_cpu(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::squeeze(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

void Squeeze::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::squeeze(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

void Unsqueeze::dispatch_cpu(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::unsqueeze(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

void Unsqueeze::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::unsqueeze(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

void Permute::dispatch_cpu(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::permute(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

void Permute::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  const axes_t &axes = _axes;
  View view = view::permute(a.view(), axes);
  outputs[0].init_view(std::make_shared<View>(view));
}

} // namespace pg