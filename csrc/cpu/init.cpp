
#include "ad_primitives.hpp"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "tensor.hpp"
#include <random>

namespace pg {
namespace cpu {
template <typename T> void fill(Tensor &t, T value, const shape_t &shape) {
  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
  T *data = t.get_casted_base_ptr<T>();
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    data[i] = value;
  }
}
} // namespace cpu
void Fill::dispatch_cpu(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->allocate();
  PG_DISPATCH_ALL_TYPES(_dtype, "dispatch_fill_kernel", [&] {
    cpu::fill(outputs[0], static_cast<scalar_t>(_value), _shape);
  });
}

void Binomial::dispatch_cpu(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  outputs[0].view_ptr()->allocate();
  PG_DISPATCH_ALL_TYPES(_dtype, "dispatch_binomial_kernel", [&] {
    std::default_random_engine generator;
    std::binomial_distribution<int> distribution(1, _p);
    scalar_t *data = outputs[0].get_casted_base_ptr<scalar_t>();
    for (long i = 0; i < outputs[0].numel(); i++) {
      data[i] = distribution(generator) == 1 ? 1 : 0;
    }
  });
}
} // namespace pg
