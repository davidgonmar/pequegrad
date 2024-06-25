#pragma once;
#include "compile.hpp"
#include "ir.hpp"

namespace pg {
void compilev2(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  remove_useless_broadcast(out);
}

void schedule(Tensor &out);
} // namespace pg
