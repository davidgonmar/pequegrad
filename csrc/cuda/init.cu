
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "random.cuh"
#include "tensor.hpp"
#include <curand_kernel.h>

namespace pg {
namespace cuda {
template <typename T>
__global__ void fill(T *ptr, T value, const size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    ptr[idx] = value;
  }
}

template <typename T>
__global__ void binomial(T *ptr, const size_t numel, const double p,
                         const curandState *state) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    curandState localState = state[idx];
    ptr[idx] = curand_uniform(&localState) < p ? 1 : 0;
  }
}
} // namespace cuda

void Fill::dispatch_cuda(const std::vector<Tensor> &inputs,
                         std::vector<Tensor> &outputs) {
  outputs[0].init_view(
      std::make_shared<View>(_shape, _dtype, device::from_str("cuda")));
  PG_DISPATCH_ALL_TYPES(_dtype, "fill_cuda", [&]() {
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t grid_size = ceil(outputs[0].numel() / (float)block_size);
    cuda::fill<scalar_t>
        <<<grid_size, block_size>>>(outputs[0].get_casted_base_ptr<scalar_t>(),
                                    (scalar_t)_value, outputs[0].numel());
  });
  PG_CUDA_KERNEL_END;
}

void Binomial::dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  outputs[0].init_view(
      std::make_shared<View>(_shape, _dtype, device::from_str("cuda")));
  curandState *state =
      CudaRandomState::get_instance(outputs[0].numel()).get_states().get();
  PG_DISPATCH_ALL_TYPES(_dtype, "binomial_cuda", [&]() {
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t grid_size = ceil(outputs[0].numel() / (float)block_size);
    cuda::binomial<scalar_t>
        <<<grid_size, block_size>>>(outputs[0].get_casted_base_ptr<scalar_t>(),
                                    outputs[0].numel(), _p, state);
  });
  PG_CUDA_KERNEL_END;
}
} // namespace pg