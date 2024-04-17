
#include "dtype.hpp"
#include "tensor.hpp"
namespace pg {
namespace cpu {
namespace helper {
template <typename T> void fill(Tensor &t, T value, const shape_t &shape) {
  size_t total_elements = 1;
  for (size_t dim : shape) {
    total_elements *= dim;
  }
  T *data = t.get_casted_base_ptr<T>();
  for (size_t i = 0; i < total_elements; i++) {
    data[i] = value;
  }
}
} // namespace helper

void fill(Tensor &t, double value, const shape_t &shape) {
  switch (t.dtype()) {
  case DType::Float32:
    helper::fill<float>(t, static_cast<float>(value), shape);
    break;
  case DType::Int32:
    helper::fill<int>(t, static_cast<int>(value), shape);
    break;
  case DType::Float64:
    helper::fill<double>(t, static_cast<double>(value), shape);
    break;
  default:
    throw std::runtime_error("Unsupported dtype: " +
                             dtype_to_string(t.dtype()));
  }
}
} // namespace cpu
} // namespace pg