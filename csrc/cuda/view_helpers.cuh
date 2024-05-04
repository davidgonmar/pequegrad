#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cuda {
namespace view {
View as_contiguous(const View &view);
void copy(const View &src, const View &dst);
} // namespace view
} // namespace cuda
} // namespace pg