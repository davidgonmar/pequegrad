#pragma once
#include "fuse.hpp"

namespace pg {

bool is_broadcast(ADPrimitive &primitive) {
  return dynamic_cast<BroadcastTo *>(&primitive) != nullptr;
}

bool is_broadcast(Tensor &tensor) {
  return is_broadcast(*tensor.ad_node().primitive().get());
}

BroadcastTo &get_broadcast(Tensor &tensor) {
  return dynamic_cast<BroadcastTo &>(*tensor.ad_node().primitive().get());
}

void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  for (Tensor &node : out.ad_node().children()) {
    if (is_broadcast(node)) {
      auto &broadcast = get_broadcast(node);
      if (broadcast.shape_to() == out.shape()) {
        Tensor &child = node.ad_node().children()[0];
        node.set_ad_node(child.ad_node());
      }
    }
  }
  fuse(out);
}
} // namespace pg