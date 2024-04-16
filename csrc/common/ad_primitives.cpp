#include "ad_primitives.hpp"


namespace pg {
    void BroadcastTo::dispatch_cpu(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        auto [view, broadcasted_axis] = view::broadcasted_to(inputs[0].view(), _shape_to);
        outputs[0].init_view(std::make_shared<View>(view));
        this->_axes_to_reduce_in_bw = broadcasted_axis;
    }

    void Squeeze::dispatch_cpu(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        CHECK_INPUTS_LENGTH(inputs, 1);
        CHECK_OUTPUTS_LENGTH(outputs, 1);
        const Tensor &a = inputs[0];
        const axes_t &axes = _axes;
        View view = view::squeeze(a.view(), axes);
        outputs[0].init_view(std::make_shared<View>(view));
    }

    void Unsqueeze::dispatch_cpu(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        CHECK_INPUTS_LENGTH(inputs, 1);
        CHECK_OUTPUTS_LENGTH(outputs, 1);
        const Tensor &a = inputs[0];
        const axes_t &axes = _axes;
        View view = view::unsqueeze(a.view(), axes);
        outputs[0].init_view(std::make_shared<View>(view));
    }

    void Permute::dispatch_cpu(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        CHECK_INPUTS_LENGTH(inputs, 1);
        CHECK_OUTPUTS_LENGTH(outputs, 1);
        const Tensor &a = inputs[0];
        const axes_t &axes = _axes;
        View view = view::permute(a.view(), axes);
        outputs[0].init_view(std::make_shared<View>(view));
    }

}