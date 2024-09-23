#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "matmul.cuh"
#include "view_helpers.cuh"
#include <cublas_v2.h>

#define PG_MATMUL_USE_CUBLAS 1
namespace pg {

#if !PG_MATMUL_USE_CUBLAS
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
#else
void MatMul::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 2);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  View a = pg::cuda::view::as_contiguous(inputs[0].view());
  View b = pg::cuda::view::as_contiguous(inputs[1].view());
  PG_CHECK_ARG(a.dtype() == b.dtype(),
               "MatMul expects inputs to have the same dtype, got ",
               dtype_to_string(a.dtype()), " and ", dtype_to_string(b.dtype()));
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  // set to stream 0
  cublasSetStream(cublas_handle, 0);
  int B = 1;
  for (size_t i = 0; i < a.ndim() - 2; i++) {
    PG_CHECK_ARG(a.shape()[i] == b.shape()[i],
                 "MatMul expects inputs to have the same shape in the batch "
                 "dimensions, got ",
                 vec_to_string(a.shape()), " and ", vec_to_string(b.shape()));
    B *= a.shape()[i];
  }
  int M = a.shape()[a.ndim() - 2];
  int N = b.shape()[b.ndim() - 1];
  int K = a.shape()[a.ndim() - 1];
  outputs[0].view_ptr()->allocate();

  // Call cuBLAS for matrix multiplication
  // TODO -- do checks
  if (a.dtype() == DType::Float32) {
    float alpha = 1.0f;
    float beta = 0.0f;
    float *a_ptr = a.get_casted_base_ptr<float>();
    float *b_ptr = b.get_casted_base_ptr<float>();
    float *out_ptr = outputs[0].get_casted_base_ptr<float>();

    // remember we use column major, so the order is reversed
    long long stride_out = M * N; // size of out
    long long stride_a = M * K;   // size of a
    long long stride_b = K * N;   // size of b
    auto res = cublasSgemmStridedBatched(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N,
        stride_b, a_ptr, K, stride_a, &beta, out_ptr, N, stride_out, B);

    PG_CUDA_KERNEL_END;
    // check res
    if (res != CUBLAS_STATUS_SUCCESS) {
      PG_CHECK_RUNTIME(false, "CUBLAS error: ", res);
    }
  } else if (a.dtype() == DType::Float64) {
    double alpha = 1.0;
    double beta = 0.0;
    double *a_ptr = a.get_casted_base_ptr<double>();
    double *b_ptr = b.get_casted_base_ptr<double>();
    double *out_ptr = outputs[0].get_casted_base_ptr<double>();

    // remember we use column major, so the order is reversed
    long long stride_out = M * N; // size of out
    long long stride_a = M * K;   // size of a
    long long stride_b = K * N;   // size of b
    auto res = cublasDgemmStridedBatched(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N,
        stride_b, a_ptr, K, stride_a, &beta, out_ptr, N, stride_out, B);
    PG_CUDA_KERNEL_END;
    // check res
    if (res != CUBLAS_STATUS_SUCCESS) {
      PG_CHECK_RUNTIME(false, "CUBLAS error: ", res);
    }
  } else if (a.dtype() == DType::Float16) {
    half alpha = 1.0f;
    half beta = 0.0f;
    half *a_ptr = a.get_casted_base_ptr<half>();
    half *b_ptr = b.get_casted_base_ptr<half>();
    half *out_ptr = outputs[0].get_casted_base_ptr<half>();

    // remember we use column major, so the order is reversed
    long long stride_out = M * N; // size of out
    long long stride_a = M * K;   // size of a
    long long stride_b = K * N;   // size of b
    auto res = cublasHgemmStridedBatched(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_ptr, N,
        stride_b, a_ptr, K, stride_a, &beta, out_ptr, N, stride_out, B);

    PG_CUDA_KERNEL_END;

  } else {
    PG_CHECK_RUNTIME(
        false, "Unsupported dtype for MatMul: ", dtype_to_string(a.dtype()));
  }

  cublasDestroy(cublas_handle);
}
#endif

} // namespace pg