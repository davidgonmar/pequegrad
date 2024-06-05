#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "matmul.cuh"
#include "view_helpers.cuh"

namespace pg {
void MatMul::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  View a = pg::cuda::view::as_contiguous(inputs[0].view());
  View b = pg::cuda::view::as_contiguous(inputs[1].view());
  PG_CHECK_ARG(a.dtype() == b.dtype(),
               "MatMul expects inputs to have the same dtype, got ",
               dtype_to_string(a.dtype()), " and ", dtype_to_string(b.dtype()));
  // We need to do 2 checks:
  // Given two inputs [D1, D2, .., A, B1] and [D1, D2, .., B2, C], we need to
  // make sure the batch dimensions are equal (not broadcastable, that is
  // handled externally, here they should be equal) and make sure B1 == B2
  PG_CHECK_ARG(
      a.ndim() == b.ndim(),
      "MatMul expects inputs to have the same number of dimensions, got ",
      a.ndim(), " and ", b.ndim());

  shape_t new_shape;
  int B = 1;
  for (size_t i = 0; i < a.ndim() - 2; i++) {
    PG_CHECK_ARG(a.shape()[i] == b.shape()[i],
                 "MatMul expects inputs to have the same shape in the batch "
                 "dimensions, got ",
                 vec_to_string(a.shape()), " and ", vec_to_string(b.shape()));
    new_shape.push_back(a.shape()[i]);
    B *= a.shape()[i];
  }
  int M = a.shape()[a.ndim() - 2];
  int N = b.shape()[b.ndim() - 1];
  int K = a.shape()[a.ndim() - 1];
  PG_CHECK_ARG(K == b.shape()[b.ndim() - 2],
               "MatMul expects inputs to have the same shape in the inner "
               "dimensions, got ",
               vec_to_string(a.shape()), " and ", vec_to_string(b.shape()));
  new_shape.push_back(M);
  new_shape.push_back(N);
  View out_view(new_shape, a.dtype(), device::CUDA);
  auto d_strides_a =
      cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());
  auto d_strides_b = cuda_unique_ptr_from_host(b.ndim(), b.strides().data());
  auto d_strides_out =
      cuda_unique_ptr_from_host(out_view.ndim(), out_view.strides().data());
  auto d_shape_a = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());
  auto d_shape_b = cuda_unique_ptr_from_host(b.ndim(), b.shape().data());
  dim3 blocksize(DEFAULT_BLOCK_SIZE);
  dim3 gridsize((out_view.numel() + blocksize.x - 1) / blocksize.x);
  PG_DISPATCH_ALL_TYPES(a.dtype(), "matmul_cuda", [&]() {
    size_t smem_size = 2 * a.ndim() * sizeof(size_t);
    cuda::batched_matmul_kernel<scalar_t><<<gridsize, blocksize, smem_size>>>(
        a.get_casted_base_ptr<scalar_t>(), b.get_casted_base_ptr<scalar_t>(),
        out_view.get_casted_base_ptr<scalar_t>(), d_shape_a.get(),
        d_shape_b.get(), a.ndim());
  });
  PG_CUDA_KERNEL_END;
  outputs[0].init_view(std::make_shared<View>(out_view));
}
} // namespace pg