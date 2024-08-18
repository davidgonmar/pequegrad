#include "tensor.hpp"

namespace pg {
using consumers_map_t = std::unordered_map<Tensor, std::vector<Tensor>>;

void get_consumers_recursive(Tensor &out, consumers_map_t &consumers) {
  for (Tensor &node : out.ad_node()->children()) {
    consumers[node].push_back(out);
    get_consumers_recursive(node, consumers);
  }
}

consumers_map_t get_consumers(std::vector<Tensor> &outs) {
  consumers_map_t consumers;
  for (Tensor &out : outs) {
    get_consumers_recursive(out, consumers);
  }
  return consumers;
}
} // namespace pg