#include "scheduler.hpp"
#include "ad_primitives.hpp"

namespace pg {

using namespace ir;

using leaf_record_t = std::vector<Tensor>;

static void schedule_inner(Tensor &node, leaf_record_t &leafs) {
  if (is<Log>(node.ad_node().primitive())) {
    auto &log = dynamic_cast<Log &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    return;
  }
  if (is<Add>(node.ad_node().primitive())) {
    auto &add = dynamic_cast<Add &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Mul>(node.ad_node().primitive())) {
    auto &mul = dynamic_cast<Mul &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  // ...
  // else, we have a leaf
  leafs.push_back(node);
}

void schedule(Tensor &out) {
  std::cout << "Scheduling " << out.id << std::endl;
  leaf_record_t leafs;
  schedule_inner(out, leafs);

  // now, we have inputs and out
  Compiled compnode;
  std::cout << "Compiling " << out.id << std::endl;

  auto [ir, ctx] = graph_to_ir(out, leafs);

  std::cout << "IR done" << std::endl;
  compnode.ir = ir;
  compnode.tensor_idx_to_strides = ctx.tensor_idx_to_strides;
  out.ad_node().set_primitive(std::make_shared<Compiled>(compnode));
  out.ad_node().set_children(leafs);
  std::cout << "Compiled " << out.id << std::endl;
}

} // namespace pg