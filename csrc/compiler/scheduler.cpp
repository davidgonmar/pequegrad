#include "scheduler.hpp"
#include "ad_primitives.hpp"

namespace pg {

using namespace ir;

using leaf_record_t = std::vector<Tensor>;

static void schedule_inner(Tensor &node, leaf_record_t &leafs,
                           std::vector<Tensor> &marked_as_out,
                           std::unordered_map<int, std::set<int>> &dependents,
                           bool is_root = false) {
  // if node is used by another node, mark it as output
  if (is<Log>(node.ad_node()->primitive())) {
    auto &log = dynamic_cast<Log &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Exp>(node.ad_node()->primitive())) {
    auto &exp = dynamic_cast<Exp &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Add>(node.ad_node()->primitive())) {
    auto &add = dynamic_cast<Add &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Mul>(node.ad_node()->primitive())) {
    auto &mul = dynamic_cast<Mul &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Sub>(node.ad_node()->primitive())) {
    auto &sub = dynamic_cast<Sub &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Div>(node.ad_node()->primitive())) {
    auto &div = dynamic_cast<Div &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Max>(node.ad_node()->primitive())) {
    auto &max = dynamic_cast<Max &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Gt>(node.ad_node()->primitive())) {
    auto &gt = dynamic_cast<Gt &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Lt>(node.ad_node()->primitive())) {
    auto &lt = dynamic_cast<Lt &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Where>(node.ad_node()->primitive())) {
    auto &where = dynamic_cast<Where &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[2], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Eq>(node.ad_node()->primitive())) {
    auto &eq = dynamic_cast<Eq &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  if (is<Pow>(node.ad_node()->primitive())) {
    auto &pow = dynamic_cast<Pow &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents);
    return;
  }
  auto prim = node.ad_node()->primitive();
  if ((is<Sum>(prim) || is<MaxReduce>(prim) || is<Mean>(prim)) && is_root) {
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents);
    return;
  }
  // else, we have a leaf
  leafs.push_back(node);
}

void schedule(
    Tensor &out, Tensor &root,
    std::unordered_map<int, std::set<int>>
        &dependents) { // dependents is a map {node_id: {dependent_node_id, ...}
                       // dependent means that the node is used by another node
  leaf_record_t leafs;
  std::vector<Tensor> marked_as_out;
  schedule_inner(out, leafs, marked_as_out, dependents, true);
  // if leafs is {out}, we failed to schedule and just return early
  if (leafs.size() == 0 | leafs.size() == 1 && leafs[0].id == out.id) {
    return;
  }
  PG_CHECK_RUNTIME(marked_as_out.size() == 0, "marked_as_out should be empty");
  // if out is reduce, marked_as_out is empty
  if (is<Sum>(out.ad_node()->primitive()) ||
      is<MaxReduce>(out.ad_node()->primitive()) ||
      is<Mean>(out.ad_node()->primitive())) {
    marked_as_out = {out};
    if (leafs.size() == 1 && leafs[0].id == out.id) {
      return;
    }
  } else {
    // if out is not in marked_as_out, add it
    bool found = false;
    for (Tensor &t : marked_as_out) {
      if (t.id == out.id) {
        found = true;
        break;
      }
    }
    if (!found) {
      marked_as_out.push_back(out);
    }
    // Now, given a subgraph SG (to be compiled) bounded by {inputs} and {out},
    // and the general graph GG started by {root}, we need to mark as out nodes
    // of SG that are used by nodes in GG - SG
    std::vector<Tensor> linearized_subgraph;
    using recurse_t = std::function<void(Tensor &)>;
    recurse_t recurse = [&](Tensor &node) {
      // check if we reached leaves (leafs)
      for (Tensor &leaf : leafs) {
        if (node.id == leaf.id) {
          return;
        }
      }
      // if node is used by another node, mark it as output
      for (Tensor &child : node.ad_node()->children()) {
        recurse(child);
      }
      linearized_subgraph.push_back(node);
    };

    recurse(out);

    // check that every tensor in the linearized_subgraph has same shape
    if (linearized_subgraph.size() > 0) {
      shape_t shape = linearized_subgraph[0].shape();
      for (Tensor &t : linearized_subgraph) {
        PG_CHECK_RUNTIME(
            t.shape() == shape,
            "All tensors in the subgraph should have the same shape, got " +
                t.str() + " with shape " + vec_to_string(t.shape()) +
                " and expected " + vec_to_string(shape));
      }
    }

    // linearized_subgraph = SG
    // root... = GG
    // so, for each node in SG, check if its dependants are not in SG (if they
    // are not, mark the node as out)
    for (Tensor &node : linearized_subgraph) {
      for (int dep : dependents[node.id]) {
        bool found = false;
        for (Tensor &t : linearized_subgraph) {
          if (t.id == dep) {
            found = true;
            break;
          }
        }
        if (!found) {
          marked_as_out.push_back(node);
          break;
        }
      }
    }

    // check that every node marked as out has the same shape
    if (marked_as_out.size() > 0) {
      shape_t shape = marked_as_out[0].shape();
      for (Tensor &t : marked_as_out) {
        PG_CHECK_RUNTIME(
            t.shape() == shape,
            "All tensors marked as out should have the same shape");
      }
    }

    // make them unique
    std::set<int> ids;
    std::vector<Tensor> marked_as_out_unique;
    for (Tensor &t : marked_as_out) {
      if (ids.find(t.id) == ids.end()) {
        marked_as_out_unique.push_back(t);
        ids.insert(t.id);
      }
    }
    marked_as_out = marked_as_out_unique;
    // now check for cycles
    // if there is a cycle, we cannot compile, we just return
    // being a children of a node means the parent depends on the child
    // a cycle means that, recursing from the parent, we reach a child (and
    // continuing recursing, we reach the parent)

    // so to check it, we check if we can reach, from each input, some output
    // if we can, we have a cycle
    std::set<int> visited;
    leaf_record_t leafs_copy = leafs;
    leaf_record_t leafs_copy_copy = leafs;
    using check_t = std::function<void(Tensor &, Tensor &)>;
    check_t check = [&](Tensor &node, Tensor &orig_node) {
      if (visited.find(node.id) != visited.end()) {
        return;
      }
      visited.insert(node.id);
      for (Tensor &child : node.ad_node()->children()) {
        check(child, orig_node);
      }

      for (Tensor &tout : marked_as_out) {
        if (tout.id == node.id) {
          // std::cout << "Cycle detected with node " << tout.str() <<
          // std::endl;
          // delete from marked_as_out
          for (int i = 0; i < marked_as_out.size(); i++) {
            if (marked_as_out[i].id == tout.id) {
              marked_as_out.erase(marked_as_out.begin() + i);
              break;
            }
          }
        }
      }
    };

    for (Tensor &t : leafs) {
      check(t, t);
    }
  }
  // if marked_as_out is empty, we failed to schedule and just return early
  if (marked_as_out.size() == 0) {
    return;
  }
  // now, we have inputs and out
  Compiled compnode;
  auto [ir, ctx, inputs_to_use] = graph_to_ir(out, marked_as_out, leafs);
  compnode.ir = ir;
  out.ad_node()->set_primitive(std::make_shared<Compiled>(compnode));
  std::vector<Tensor> tensors_to_use;
  for (int i = 0; i < inputs_to_use.size(); i++) {
    if (inputs_to_use[i]) {
      tensors_to_use.push_back(leafs[i]);
    }
  }
  // each tensor marked as out -> set as sibling to out
  std::vector<Tensor> sibs;
  for (Tensor &t : marked_as_out) {
    sibs.push_back(t);
    sibs.back().ad_node()->set_primitive(out.ad_node()->primitive());
  }
  // sibs.push_back(out);
  // now, for each tensor in sibs, set siblings to sib - {t}
  int pos = 0;
  for (Tensor &t : sibs) {
    t.ad_node()->set_position(pos++);
  }
  for (Tensor &t : sibs) {
    t.ad_node()->set_siblings(sibs);
  }
  for (Tensor &t : sibs) {
    t.ad_node()->set_children(tensors_to_use);
  }
}

} // namespace pg