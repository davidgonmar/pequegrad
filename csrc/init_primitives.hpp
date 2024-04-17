#include "dtype.hpp"
#include "tensor.hpp"
namespace pg {
namespace cpu {
namespace helper {
template <typename T> void fill(Tensor &t, T value, const shape_t &shape);
} // namespace helper
void fill(Tensor &t, double value, const shape_t &shape);
} // namespace cpu
} // namespace pg