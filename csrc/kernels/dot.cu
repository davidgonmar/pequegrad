#include "dot.cuh"

void launch_matmul_kernel(dim3 grid_size, dim3 block_size, DType dtype,
                          const void *a, const void *b, void *out,
                          const size_t *a_shape, const size_t *b_shape,
                          const size_t n_dims) {
  switch (dtype) {
  case DType::Float32:
    batched_matmul_kernel<float><<<grid_size, block_size, 0, 0>>>(
        static_cast<const float *>(a), static_cast<const float *>(b),
        static_cast<float *>(out), a_shape, b_shape, n_dims);
    break;
  case DType::Float64:
    batched_matmul_kernel<double><<<grid_size, block_size, 0, 0>>>(
        static_cast<const double *>(a), static_cast<const double *>(b),
        static_cast<double *>(out), a_shape, b_shape, n_dims);
    break;
  case DType::Int32:
    batched_matmul_kernel<int><<<grid_size, block_size, 0, 0>>>(
        static_cast<const int *>(a), static_cast<const int *>(b),
        static_cast<int *>(out), a_shape, b_shape, n_dims);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void launch_vector_outer_product_kernel(dim3 grid_size, dim3 block_size,
                                        DType dtype, void *a, void *b,
                                        void *out, size_t m, size_t n) {
  switch (dtype) {
  case DType::Float32:
    vector_outer_product_kernel<float><<<grid_size, block_size, 0, 0>>>(
        static_cast<float *>(a), static_cast<float *>(b),
        static_cast<float *>(out), m, n);
    break;
  case DType::Float64:

    vector_outer_product_kernel<double><<<grid_size, block_size, 0, 0>>>(
        static_cast<double *>(a), static_cast<double *>(b),
        static_cast<double *>(out), m, n);
    break;
  case DType::Int32:
    vector_outer_product_kernel<int><<<grid_size, block_size, 0, 0>>>(
        static_cast<int *>(a), static_cast<int *>(b), static_cast<int *>(out),
        m, n);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void launch_batched_matmul_kernel(dim3 grid_size, dim3 block_size, DType dtype,
                                  const void *a, const void *b, void *out,
                                  const size_t *a_shape, const size_t *b_shape,
                                  const size_t n_dims) {
  switch (dtype) {
  case DType::Float32:
    batched_matmul_kernel<float><<<grid_size, block_size, 0, 0>>>(
        static_cast<const float *>(a), static_cast<const float *>(b),
        static_cast<float *>(out), a_shape, b_shape, n_dims);
    break;
  case DType::Float64:
    batched_matmul_kernel<double><<<grid_size, block_size, 0, 0>>>(
        static_cast<const double *>(a), static_cast<const double *>(b),
        static_cast<double *>(out), a_shape, b_shape, n_dims);
    break;
  case DType::Int32:
    batched_matmul_kernel<int><<<grid_size, block_size, 0, 0>>>(
        static_cast<const int *>(a), static_cast<const int *>(b),
        static_cast<int *>(out), a_shape, b_shape, n_dims);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void launch_vector_dot_product_accum_kernel(dim3 grid_size, dim3 block_size,
                                            size_t smem_size, DType dtype,
                                            const void *a, const void *b,
                                            void *out, size_t size) {
  switch (dtype) {
  case DType::Float32:
    vector_dot_product_accum_kernel<float>
        <<<grid_size, block_size, smem_size, 0>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<float *>(out), size);
    break;
  case DType::Float64:

    vector_dot_product_accum_kernel<double>
        <<<grid_size, block_size, smem_size, 0>>>(
            static_cast<const double *>(a), static_cast<const double *>(b),
            static_cast<double *>(out), size);
    break;
  case DType::Int32:
    vector_dot_product_accum_kernel<int>
        <<<grid_size, block_size, smem_size, 0>>>(
            static_cast<const int *>(a), static_cast<const int *>(b),
            static_cast<int *>(out), size);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}