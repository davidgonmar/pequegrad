#pragma once
#include "expr.hpp"
#include "scheduler.hpp"

namespace pg {
static bool is_broadcast(ADPrimitive &primitive) {
  return dynamic_cast<BroadcastTo *>(&primitive) != nullptr;
}

static bool is_broadcast(Tensor &tensor) {
  return is_broadcast(*tensor.ad_node().primitive().get());
}

static BroadcastTo &get_broadcast(Tensor &tensor) {
  return dynamic_cast<BroadcastTo &>(*tensor.ad_node().primitive().get());
}

static void remove_useless_broadcast(Tensor &out) {
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

static void rec_fuse(Tensor &out) {
  fuse(out);

  for (int i = 0; i < out.ad_node().children().size(); i++) {
    if (i < out.ad_node().children().size()) {
      rec_fuse(out.ad_node().children()[i]);
    }
  }
}
static void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  remove_useless_broadcast(out);
  // Second pass -> fuse
  // rec_fuse(out);
  schedule(out);
}
} // namespace pg