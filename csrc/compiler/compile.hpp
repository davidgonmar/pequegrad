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
        continue;
      }
    }
  }

  // now, recursively call remove_useless_broadcast for each children
  for (Tensor &node : out.ad_node().children()) {
    remove_useless_broadcast(node);
  }
}
void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  remove_useless_broadcast(out);
  bool success = fuse(out);

  while (success) {
    success = fuse(out);
  }
LOOP:
  int n_children = out.ad_node().children().size();
  for (int i = 0; i < n_children; i++) {
    bool _success = fuse(out.ad_node().children()[i]);
    if (_success) {
      goto LOOP;
    }
  }
  // now, for each children, compile
  for (Tensor &node : out.ad_node().children()) {
    compile(node);
  }
}
} // namespace pg