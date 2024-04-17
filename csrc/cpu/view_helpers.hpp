#pragma once
#include "shape.hpp"
#include "tensor.hpp"

namespace pg {
namespace cpu {
namespace view {
View as_contiguous(const View &view);
}
} // namespace cpu
} // namespace pg