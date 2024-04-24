#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace view {
std::tuple<View, axes_t, axes_t> broadcasted_to(const View &view,
                                                const shape_t &shape_to);
View squeeze(const View &view, const axes_t &axes);
View squeeze(const View &view);
View squeeze(const View &view, const axis_t axis);
View unsqueeze(const View &view, const axes_t &axes);
View unsqueeze(const View &view, const axis_t axis);
View permute(const View &view, const axes_t &axes);
View transpose(
    const View &view); // transpose all dims (reverses the order of dims)
View nocopy_reshape_nocheck(const View &view, const shape_t &shape);
} // namespace view
} // namespace pg