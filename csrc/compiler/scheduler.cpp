#include "scheduler.hpp"

namespace pg {

using leaf_record_t = std::vector<Tensor>;

void schedule(Tensor &out) {
  leaf_record_t leafs;
  schedule_inner(out, leafs);

  // now, we have inputs and out
  Compiled compnode;

  out.ad_node().set_primitive(std::make_shared<CompiledPrimitive>(compnode));
  out.ad_node().set_children(leafs);
}

void schedule_inner(Tensor &node, leaf_record_t &leafs) {
  if (is<Log>(node.ad_node().primitive())) {
    auto &log = dynamic_cast<Log &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
  }
  if (is<Add>(node.ad_node().primitive())) {
    auto &add = dynamic_cast<Add &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
  }
  if (is<Mul>(node.ad_node().primitive())) {
    auto &mul = dynamic_cast<Mul &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
  }
  // ...
  // else, we have a leaf
  leafs.push_back(node);
}
} // namespace pg