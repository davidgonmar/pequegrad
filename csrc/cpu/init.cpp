
#include "ad_primitives.hpp"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "tensor.hpp"
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
  outputs[0].init_view(
      std::make_shared<View>(_shape, _dtype, device::DeviceKind::CPU));
  PG_DISPATCH_ALL_TYPES(_dtype, "dispatch_fill_kernel", [&] {
    cpu::fill(outputs[0], static_cast<scalar_t>(_value), _shape);
  });
}
} // namespace pg