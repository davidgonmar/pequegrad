#pragma once
#include "scheduler.hpp"

namespace pg {
static bool is_broadcast(ADPrimitive &primitive) {
  return dynamic_cast<BroadcastTo *>(&primitive) != nullptr;
}

static bool is_broadcast(Tensor &tensor) {
  return is_broadcast(*tensor.ad_node()->primitive().get());
}

static BroadcastTo &get_broadcast(Tensor &tensor) {
  return dynamic_cast<BroadcastTo &>(*tensor.ad_node()->primitive().get());
}

static void remove_useless_broadcast(Tensor &out, std::set<int> &visited) {
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  for (Tensor &node : out.ad_node()->children()) {
    visited.insert(out.id);
    if (is_broadcast(node)) {
      auto &broadcast = get_broadcast(node);
      // useless broadcast
      if (broadcast.shape_to() == node.children()[0].shape()) {
        Tensor &child = node.ad_node()->children()[0];
        // connect out with the child
        out.ad_node()->replace_child(node, child);
        continue;
      }
    }
  }

  // now, recursively call remove_useless_broadcast for each children
  for (Tensor &node : out.ad_node()->children()) {
    remove_useless_broadcast(node, visited);
  }
}

static void rec_schedule(Tensor &root, Tensor &out, std::set<int> &visited) {
  // get a map of tensor -> tensors that have that tensor as a child (depend on
  // it)
  std::unordered_map<int, std::set<int>>
      dependents; // key: tensor id, value: vector of tensor ids

  std::set<int> _visited;
  using recurse_t = std::function<void(Tensor &)>;
  recurse_t recurse = [&](Tensor &node) {
    if (_visited.find(node.id) != _visited.end()) {
      return;
    }
    _visited.insert(node.id);
    for (Tensor child : node.ad_node()->children()) {
      dependents[child.id].insert(node.id);
      recurse(child);
    }
  };
  recurse(root);
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  visited.insert(out.id);

  schedule(out, root, dependents);
  for (int i = 0; i < out.ad_node()->children().size(); i++) {
    if (i < out.ad_node()->children().size()) {
      rec_schedule(root, out.ad_node()->children()[i], visited);
    }
  }
}

static void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  std::set<int> visited;
  remove_useless_broadcast(out, visited);
  // Second -> schedule (fuse)
  std::set<int> visited2;
  rec_schedule(out, out, visited2);
}
} // namespace pg