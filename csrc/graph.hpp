#include "tensor.hpp"

namespace pg {

std::pair<std::vector<Tensor>, std::vector<Tensor>>
clone_graph(std::vector<Tensor> &outputs, std::vector<Tensor> &inputs) {
  std::vector<Tensor> new_outputs;
  std::vector<Tensor> new_inputs;
  using copy_lambda = std::function<void(Tensor &)>;

  // copy inputs first
  std::map<Tensor *, Tensor *> old_to_new;

  std::cout << "here0" << std::endl;
  // First pass, create new  tensors
  copy_lambda copy = [&](Tensor &t) {
    // check if tensor is already copied
    if (old_to_new.find(&t) != old_to_new.end()) {
      return;
    }
    // first children
    for (Tensor &child : t.children()) {
      if (old_to_new.find(&child) == old_to_new.end()) {
        copy(child);
      }
    }

    std::cout << "here1" << std::endl;
    // then get new childrens
    std::vector<Tensor> new_children;
    for (Tensor &child : t.children()) {
      new_children.push_back(*old_to_new.at(&child));
    }
    Tensor new_t = t.copy_graph(new_children);
    old_to_new[&t] = &new_t;

    std::cout << "here2" << std::endl;
  };

  for (Tensor &t : outputs) {
    copy(t);
    new_outputs.push_back(*old_to_new.at(&t));
  }

  for (Tensor &t : inputs) {
    copy(t);
    new_inputs.push_back(*old_to_new.at(&t));
  }

  return std::make_pair(new_outputs, new_inputs);
}
class ComputeGraph {
public:
  ComputeGraph() = default;
  ComputeGraph(const ComputeGraph &) = default;
  ComputeGraph &operator=(const ComputeGraph &) = default;
  ComputeGraph(ComputeGraph &&) = default;
  ComputeGraph &operator=(ComputeGraph &&) = default;
  ~ComputeGraph() = default;

  void clone_from_outputs(std::vector<Tensor> &outputs,
                          std::vector<Tensor> &inputs) {
    this->outputs = outputs;
    this->inputs = inputs;
  }

  void clone_from_output(Tensor &output, std::vector<Tensor> &inputs) {
    this->outputs.push_back(output);
    this->inputs = inputs;
  }

  static ComputeGraph from_outputs(std::vector<Tensor> &outputs,
                                   std::vector<Tensor> &inputs) {
    ComputeGraph cg;
    cg.clone_from_outputs(outputs, inputs);
    return cg;
  }

  std::vector<Tensor> feed_data(const std::vector<Tensor> &inputs,
                                bool eval = true) {
    // Now, look for leafs and feed the data
    int curr = 0;
    for (Tensor &t : this->inputs) {
      t.assign(inputs[curr]);
      curr++;
    }
    // We first need to clear the data of the tensors
    using clear_lambda = std::function<void(Tensor &)>;
    clear_lambda clear_data = [&](Tensor &t) {
      if (t.ad_node().is_leaf()) {
        return;
      }
      t.reset_view();
      for (Tensor &child : t.children()) {
        clear_data(child);
      }
    };

    for (Tensor &t : this->outputs) {
      clear_data(t);
    }

    for (Tensor &t : this->outputs) {
      t.eval(); // force reevaluation
    }

    return this->outputs;
  }

  std::vector<Tensor> get_outputs() { return this->outputs; }

private:
  std::vector<Tensor> outputs;
  std::vector<Tensor> inputs;
  void clean_tensor_data(std::vector<Tensor> &tensors) {
    for (Tensor &t : tensors) {
      t.reset_view();
    }
  };
};

} // namespace pg