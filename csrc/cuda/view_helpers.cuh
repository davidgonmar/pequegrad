#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cuda {
namespace view {
View as_contiguous(const View &view);
void copy(const View &src, const View &dst);
View astype(const View &view, const DType &dtype);
} // namespace view
} // namespace cuda
} // namespace pg