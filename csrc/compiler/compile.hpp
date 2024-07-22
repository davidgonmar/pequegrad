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

#define RETURN_FALSE_IF_PRIM_IS_NOT(tensor, prim)                              \
  if (!(tensor.ad_node()->primitive()->str() == prim)) {                       \
    return std::nullopt;                                                       \
  }

template <typename T> using Maybe = std::optional<T>;

struct Conv2dPatternMatcherResult {
  Tensor input;
  Tensor filter;
};

Maybe<Conv2dPatternMatcherResult> conv2d_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Col2Im");
  auto &foldchild = out.ad_node()->children()[0];
  // In conv, the folded output has 2 children: processed input and kernel
  RETURN_FALSE_IF_PRIM_IS_NOT(foldchild, "MatMul");
  auto &unfolded_input = foldchild.ad_node()->children()[1];
  auto &filter_permuted0 = foldchild.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted0,
                              "Broadcast"); // broadcasted filter

  auto &filter_permuted1 =
      filter_permuted0.ad_node()->children()[0]; // the actual permuted filter
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted1, "Permute");
  // now try to get filter
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted1.ad_node()->children()[0],
                              "Permute");
  auto &filter_permuted2 = filter_permuted1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted2.ad_node()->children()[0],
                              "Reshape");
  auto &filter =
      filter_permuted2.ad_node()->children()[0].ad_node()->children()[0];
  // now try to get input
  RETURN_FALSE_IF_PRIM_IS_NOT(unfolded_input, "Im2Col");
  // another requisite is that unfolded input stride, padding, dilation is one
  Im2Col &im2col =
      dynamic_cast<Im2Col &>(*unfolded_input.ad_node()->primitive().get());
  shape_t needed = {1, 1};
  if (im2col.strides() != needed || im2col.dilation() != needed) {
    return std::nullopt;
  }
  auto &input = unfolded_input.ad_node()->children()[0];

  return Conv2dPatternMatcherResult{input, filter};
}

bool try_convert_conv2d(Tensor &out) {
  auto maybe_result = conv2d_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  auto &input = result.input;
  auto &filter = result.filter;
  out.ad_node()->set_primitive(std::make_shared<CudnnConv2D>());
  out.ad_node()->set_children({input, filter});
  return true;
}

void recursive_conv2d(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_conv2d(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d(node);
  }
}

static void compile(Tensor &out) {
  // First pass -> remove unnecesary broadcast
  std::set<int> visited;
  remove_useless_broadcast(out, visited);
  // Second pass -> conv2d pattern matching
  recursive_conv2d(out);
  // Second -> schedule (fuse)
  std::set<int> visited2;
  rec_schedule(out, out, visited2);
}
} // namespace pg