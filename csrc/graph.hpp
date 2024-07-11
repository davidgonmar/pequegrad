#include "ad_primitives.hpp"
#include "tensor.hpp"

namespace pg {

std::tuple<std::vector<Tensor>, std::vector<Tensor>>
clone_graph(std::vector<Tensor> &outputs, std::vector<Tensor> &inputs) {
  std::vector<Tensor> new_outputs;
  std::vector<Tensor> new_inputs;

  using copy_lambda = std::function<void(Tensor &)>;

  auto tensorComparator = [](const Tensor &lhs, const Tensor &rhs) {
    // ??? weird but works. todo figure this out
    return lhs.id < rhs.id;
  };

  std::map<Tensor, Tensor, decltype(tensorComparator)> old_to_new(
      tensorComparator);

  using notin_lambda_t =
      std::function<bool(const std::vector<Tensor> &, const Tensor &)>;
  notin_lambda_t notin = [&](const std::vector<Tensor> &vec, const Tensor &t) {
    for (const Tensor &v : vec) {
      if (v.id == t.id) {
        return false;
      }
    }
    return true;
  };

  std::set<int> visited;

  copy_lambda copy = [&](Tensor &t) {
    if ((old_to_new.find(t) != old_to_new.end()) ||
        (visited.find(t.id) != visited.end())) {
      return;
    }

    std::vector<Tensor> siblings = {t};
    for (Tensor &sib : t.ad_node()->siblings()) {
      siblings.push_back(sib);
    }

    // Mark tensors as visited
    for (Tensor &sib : siblings) {
      visited.insert(sib.id);
    }

    // input will not have siblings
    if (!notin(inputs, t)) {
      PG_CHECK_RUNTIME(t.ad_node()->siblings().size() == 0,
                       "Input should not have siblings");
      // if it's an input, we just copy it but set a 'breakpoint'
      std::vector<Tensor> empty = {};
      std::shared_ptr<JitBoundary> boundary = std::make_shared<JitBoundary>();
      Tensor new_t = t.copy_graph(empty, boundary);
      old_to_new[t] = new_t;
      return;
    }

    for (Tensor &child : t.children()) {
      copy(child);
    }

    // then get new children
    std::vector<Tensor> new_children;
    for (Tensor &child : t.children()) {
      // we did not find the child
      if (old_to_new.find(child) == old_to_new.end()) {
        // check which siblings (of the child) are found
        std::vector<Tensor> found_sibs;
        std::vector<Tensor> not_found_sibs = {child};
        for (Tensor &sib : child.ad_node()->siblings()) {
          if (old_to_new.find(sib) != old_to_new.end()) {
            found_sibs.push_back(sib);
          } else {
            not_found_sibs.push_back(sib);
          }
        }
        for (Tensor &sib : found_sibs) {
          std::cout << "Found sib: " << sib.str() << std::endl;
        }
        for (Tensor &sib : not_found_sibs) {
          std::cout << "Not found sib: " << sib.str() << std::endl;
        }
        // now print t and its siblings
        std::cout << "T: " << t.str() << std::endl;
        for (Tensor &sib : t.ad_node()->siblings()) {
          std::cout << "T Sibling: " << sib.str() << std::endl;
        }
        throw std::runtime_error("Child not found in old_to_new: " +
                                 child.str());
      }
      new_children.push_back(old_to_new.at(child));
    }

    std::vector<Tensor> newsibs = {};
    // if new_t has siblings, we need to copy them as well
    for (Tensor &sib : siblings) {
      Tensor new_sib = sib.copy_graph(new_children);
      new_sib.ad_node()->set_position(sib.ad_node()->position());
      old_to_new[sib] = new_sib;
      newsibs.push_back(new_sib);
    }

    // for each sibling, we need to set the new siblings
    for (Tensor &sib : newsibs) {
      sib.ad_node()->set_siblings(newsibs);
    }
  };

  for (Tensor &t : outputs) {
    copy(t);
    if (old_to_new.find(t) == old_to_new.end()) {
      throw std::runtime_error("Output not found in old_to_new: " + t.str());
    }
    new_outputs.push_back(old_to_new.at(t));
  }
  for (Tensor &t : inputs) {
    PG_CHECK_RUNTIME(t.ad_node()->siblings().size() == 0,
                     "Input should not have siblings");
    if (old_to_new.find(t) == old_to_new.end()) {
      // throw std::runtime_error("Input not found in old_to_new: " + t.str());
      // TODO -- maybe something cleaner
      std::vector<Tensor> empty = {};
      std::shared_ptr<JitBoundary> boundary = std::make_shared<JitBoundary>();
      new_inputs.push_back(
          t.copy_graph(empty, boundary)); // copy inputs as well
      continue; // means the output does not depend on this input
    }
    if (old_to_new.find(t) == old_to_new.end()) {
      throw std::runtime_error("Input not found in old_to_new: " + t.str());
    }
    new_inputs.push_back(old_to_new.at(t));
  }
  auto x = std::make_tuple(new_outputs, new_inputs);

  return x;
}

} // namespace pg
