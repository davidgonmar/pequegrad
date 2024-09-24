#include "./ternary.cuh"
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include <cmath>

namespace pg {
namespace cuda {
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, float)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, double)
DEF_TERNARY_OP_KERNEL(where_kernel, x ? y : z, int)

} // namespace cuda

void Where::dispatch_cuda(const std::vector<Tensor> &inputs,
                          std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 3);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &condition = inputs[0];
  const Tensor &a = inputs[1];
  const Tensor &b = inputs[2];

  CHECK_SAME_SHAPE(a, b);
  CHECK_SAME_SHAPE(condition, a);
  outputs[0].init_view(
      std::make_shared<View>(a.shape(), a.dtype(), a.device()));
  size_t numels = a.numel();
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_strides_b = cuda_unique_ptr_from_host(b.ndim(), b.strides().data());
  auto d_strides_condition = cuda_unique_ptr_from_host<stride_t>(
      condition.ndim(), condition.strides().data());
  auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  dim3 blocksize(DEFAULT_BLOCK_SIZE);
  dim3 gridsize((numels + blocksize.x - 1) / blocksize.x);
  PG_DISPATCH_ALL_TYPES(a.dtype(), "where_cuda", [&]() {
    cuda::where_kernel<<<gridsize, blocksize>>>(
        d_strides_condition.get(), d_strides_a.get(), d_strides_b.get(),
        d_shape.get(), a.ndim(), condition.get_casted_base_ptr<scalar_t>(),
        a.get_casted_base_ptr<scalar_t>(), b.get_casted_base_ptr<scalar_t>(),
        outputs[0].get_casted_base_ptr<scalar_t>());
  });
  PG_CUDA_KERNEL_END;
}

} // namespace pg