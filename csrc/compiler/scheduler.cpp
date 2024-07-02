#include "scheduler.hpp"
#include "ad_primitives.hpp"

namespace pg {

using namespace ir;

using leaf_record_t = std::vector<Tensor>;

static void schedule_inner(Tensor &node, leaf_record_t &leafs,
                           bool is_root = false) {
  if (is<Log>(node.ad_node().primitive())) {
    auto &log = dynamic_cast<Log &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    return;
  }
  if (is<Exp>(node.ad_node().primitive())) {
    auto &exp = dynamic_cast<Exp &>(*node.ad_node().primitive());
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
  if (is<Sub>(node.ad_node().primitive())) {
    auto &sub = dynamic_cast<Sub &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Div>(node.ad_node().primitive())) {
    auto &div = dynamic_cast<Div &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Max>(node.ad_node().primitive())) {
    auto &max = dynamic_cast<Max &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Gt>(node.ad_node().primitive())) {
    auto &gt = dynamic_cast<Gt &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Lt>(node.ad_node().primitive())) {
    auto &lt = dynamic_cast<Lt &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Where>(node.ad_node().primitive())) {
    auto &where = dynamic_cast<Where &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    schedule_inner(node.ad_node().children()[2], leafs);
    return;
  }
  if (is<Eq>(node.ad_node().primitive())) {
    auto &eq = dynamic_cast<Eq &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  if (is<Pow>(node.ad_node().primitive())) {
    auto &pow = dynamic_cast<Pow &>(*node.ad_node().primitive());
    schedule_inner(node.ad_node().children()[0], leafs);
    schedule_inner(node.ad_node().children()[1], leafs);
    return;
  }
  auto prim = node.ad_node().primitive();
  if ((is<Sum>(prim) || is<MaxReduce>(prim) || is<Mean>(prim)) && is_root) {
    schedule_inner(node.ad_node().children()[0], leafs);
    return;
  }
  // ...
  // else, we have a leaf
  leafs.push_back(node);
}

void schedule(Tensor &out) {
  leaf_record_t leafs;
  schedule_inner(out, leafs, true);
  // if leafs is {out}, we failed to schedule and just return early
  if (leafs.size() == 1 && leafs[0].id == out.id) {
    return;
  }
  // now, we have inputs and out
  Compiled compnode;
  auto [ir, ctx, inputs_to_use] = graph_to_ir(out, leafs);
  compnode.ir = ir;
  out.ad_node().set_primitive(std::make_shared<Compiled>(compnode));
  std::vector<Tensor> tensors_to_use;
  for (int i = 0; i < inputs_to_use.size(); i++) {
    if (inputs_to_use[i]) {
      tensors_to_use.push_back(leafs[i]);
    }
  }
  out.ad_node().set_children(tensors_to_use);
}

} // namespace pg