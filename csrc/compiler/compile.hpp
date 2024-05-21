#pragma once
#include "fuse.hpp"

namespace pg {

void compile(Tensor &out) { fuse_unary(out); }
} // namespace pg