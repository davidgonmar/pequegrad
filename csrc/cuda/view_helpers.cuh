#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cuda {
namespace view {
View as_contiguous(const View &view);
}
} // namespace cuda
} // namespace pg