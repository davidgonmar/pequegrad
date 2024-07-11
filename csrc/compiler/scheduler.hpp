#pragma once;
#include "ir.hpp"

namespace pg {
void schedule(Tensor &out, Tensor &root,
              std::unordered_map<int, std::set<int>> &dependents);
} // namespace pg
