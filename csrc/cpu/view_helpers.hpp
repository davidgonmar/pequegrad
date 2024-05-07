#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cpu {
namespace view {
View as_contiguous(const View &view, bool force = false);
View astype(const View &view, DType dtype);
} // namespace view
} // namespace cpu
} // namespace pg