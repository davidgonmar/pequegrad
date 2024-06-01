
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "tensor.hpp"

namespace pg {
namespace cuda {
template <typename T>
__global__ void fill(T *ptr, T value, const size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    ptr[idx] = value;
  }
}
} // namespace cuda
void Fill::dispatch_cuda(const std::vector<Tensor> &inputs,
                         std::vector<Tensor> &outputs) {
  outputs[0].init_view(std::make_shared<View>(_shape, _dtype, device::CUDA));
  PG_DISPATCH_ALL_TYPES(_dtype, "fill_cuda", [&]() {
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t grid_size = ceil(outputs[0].numel() / (float)block_size);
    cuda::fill<scalar_t>
        <<<grid_size, block_size>>>(outputs[0].get_casted_base_ptr<scalar_t>(),
                                    (scalar_t)_value, outputs[0].numel());
  });
  PG_CUDA_KERNEL_END;
}
} // namespace pg