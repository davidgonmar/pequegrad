#pragma once
#include "expr.hpp"

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

void remove_useless_broadcast(Tensor &out) {
  for (Tensor &node : out.ad_node().children()) {
    if (is_broadcast(node)) {
      auto &broadcast = get_broadcast(node);
      // useless broadcast
      if (broadcast.shape_to() == node.children()[0].shape()) {
        Tensor &child = node.ad_node().children()[0];
        // connect out with the child
        out.ad_node().replace_child(node, child);
        remove_useless_broadcast(child);
        continue;
      }
    }
    remove_useless_broadcast(node);
  }
}
void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  remove_useless_broadcast(out);
  fuse(out);

  for (Tensor &node : out.ad_node().children()) {
    compile(node);
  }
}
} // namespace pg