#include "scheduler.hpp"
#include "ad_primitives.hpp"
#include "utils.hpp"

namespace pg {

using namespace ir;

using leaf_record_t = std::vector<Tensor>;

constexpr bool ALLOW_REDUCE_EPILOGUE = true;

static void schedule_inner(Tensor &node, leaf_record_t &leafs,
                           std::vector<Tensor> &marked_as_out,
                           std::unordered_map<int, std::set<int>> &dependents,
                           std::vector<Tensor> &allgraph, bool *allow_reduce) {
  auto prim = node.ad_node()->primitive();
  allgraph.push_back(node);
  // we will not continue recursing if there is some dependents of this node
  // that are not in the allgraph
  if (dependents.find(node.id) != dependents.end()) {
    for (int dep : dependents[node.id]) {
      bool found = false;
      for (Tensor &t : allgraph) {
        if (t.id == dep) {
          found = true;
          break;
        }
      }
      if (!found) {
        leafs.push_back(node);
        return;
      }
    }
  }
  if ((is<Sum>(prim) || is<MaxReduce>(prim) || is<Mean>(prim)) &&
      *allow_reduce) {
    *allow_reduce = false;
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents, allgraph,
                   allow_reduce); // dont allow more reduce if we
                                  // already have one
    return;
  }

  if (is<Log>(prim) || is<Exp>(prim)) {
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
    return;
  }
  // allow_reduce =false; // buggy if we allow reduce here

  if (is<Add>(prim) || is<Mul>(prim) || is<Sub>(prim) || is<Div>(prim) ||
      is<Max>(prim) || is<Gt>(prim) || is<Lt>(prim) || is<Eq>(prim) ||
      is<Pow>(prim)) {
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
    return;
  }
  if (is<Where>(node.ad_node()->primitive())) {
    auto &where = dynamic_cast<Where &>(*node.ad_node()->primitive());
    schedule_inner(node.ad_node()->children()[0], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
    schedule_inner(node.ad_node()->children()[1], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
    schedule_inner(node.ad_node()->children()[2], leafs, marked_as_out,
                   dependents, allgraph, allow_reduce);
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
  bool allow_reduce_ptr = true;
  std::vector<Tensor> allgraph;
  schedule_inner(out, leafs, marked_as_out, dependents, allgraph,
                 &allow_reduce_ptr);
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
    if (leafs.size() == 1 && leafs[0].id == out.children()[0].id) {
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