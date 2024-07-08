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
  // First pass, create new  tensors
  copy_lambda copy = [&](Tensor &t) {
    // we need to find leafs that are not inputs (e.g. constants OR model
    // weights) those will be old_to_new[t] = t

    // check if tensor is already copied
    if (old_to_new.find(t) != old_to_new.end()) {
      return;
    }

    if (!notin(inputs, t)) {
      // if it's an input, we just copy it but set a 'breakpoint'
      Tensor &new_t =
          t.copy_graph(std::vector<Tensor>(), std::make_shared<JitBoundary>());
      old_to_new[t] = new_t;
      return;
    }

    // first children
    for (Tensor &child : t.children()) {
      if (old_to_new.find(child) == old_to_new.end()) {
        copy(child);
      }
    }
    // then get new children
    std::vector<Tensor> new_children;
    for (Tensor &child : t.children()) {
      new_children.push_back(old_to_new.at(child));
    }
    Tensor &new_t = t.copy_graph(new_children);
    old_to_new[t] = new_t;
  };

  for (Tensor &t : outputs) {
    copy(t);
    new_outputs.push_back(old_to_new.at(t));
  }
  for (Tensor &t : inputs) {
    if (old_to_new.find(t) == old_to_new.end()) {
      //throw std::runtime_error("Input not found in old_to_new: " + t.str());
      // TODO -- maybe something cleaner
      new_inputs.push_back(t.copy_graph(std::vector<Tensor>(), std::make_shared<JitBoundary>()));
      continue; // means the output does not depend on this input
    }
    new_inputs.push_back(old_to_new.at(t));
  }
  auto x = std::make_tuple(new_outputs, new_inputs);

  return x;
}

} // namespace pg