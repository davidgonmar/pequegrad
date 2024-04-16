#include "ad_primitives.hpp"


namespace pg {
    void BroadcastTo::dispatch_cpu(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
        auto [view, broadcasted_axis] = view::broadcasted_to(inputs[0].view(), _shape_to);
        outputs[0].init_view(std::make_shared<View>(view));
        this->_axes_to_reduce_in_bw = broadcasted_axis;
    }
}