
#include "cuda_tensor/cuda_utils.cuh"
#include "dtype.hpp"
#include "tensor.hpp"

namespace pg {
namespace cuda {
namespace helper {
template <typename T>
__global__ void fill(T *ptr, T value, const size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    ptr[idx] = value;
  }
}
} // namespace helper

void fill(Tensor &t, double value, const shape_t &_shape) {
  auto shape = cuda_unique_ptr_from_host(t.ndim(), _shape.data());
  size_t numels = std::accumulate(_shape.begin(), _shape.end(), 1,
                                  std::multiplies<size_t>());
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size((numels + block_size.x - 1) / block_size.x);
  switch (t.dtype()) {
  case DType::Float32:
    helper::fill<float><<<grid_size, block_size>>>(
        t.get_casted_base_ptr<float>(), static_cast<float>(value), numels);
    break;
  case DType::Int32:
    helper::fill<int><<<grid_size, block_size>>>(
        t.get_casted_base_ptr<int>(), static_cast<int>(value), numels);
    break;
  case DType::Float64:
    helper::fill<double><<<grid_size, block_size>>>(
        t.get_casted_base_ptr<double>(), value, numels);
    break;
  default:
    throw std::runtime_error("Unsupported dtype: " +
                             dtype_to_string(t.dtype()));
  }
}
} // namespace cuda
} // namespace pg