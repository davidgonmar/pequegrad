#pragma once

#include "cuda_utils.cuh"
#include "dtype.hpp"
#include "shape.hpp"

#define DEF_BIN_OP_KERNEL(NAME, FN, TYPE)                                      \
  __global__ void NAME(                                                        \
      const long *_lhs_strides, /* in bytes */                                 \
      const long *_rhs_strides, /* in bytes */                                 \
      const size_t *_shape,  /* both lhs and rhs should have equal shape, we   \
                                don't handle broadcasting here */              \
      const size_t num_dims, /* equals len of strides and shape */             \
      const TYPE *lhs, const TYPE *rhs, TYPE *out) {                           \
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;                     \
    extern __shared__ int8_t smem[];                                           \
    long *lhs_strides = (long *)smem;                                          \
    long *rhs_strides = (long *)(smem + num_dims * sizeof(long));              \
    size_t *shape = (size_t *)(smem + num_dims * sizeof(long) * 2);            \
                                                                               \
    if (threadIdx.x < num_dims) {                                              \
      lhs_strides[threadIdx.x] = _lhs_strides[threadIdx.x];                    \
      rhs_strides[threadIdx.x] = _rhs_strides[threadIdx.x];                    \
      shape[threadIdx.x] = _shape[threadIdx.x];                                \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    if (idx >= get_max_idx(shape, num_dims))                                   \
      return;                                                                  \
                                                                               \
    /* calculate correct index based on strides */                             \
    int idx_lhs =                                                              \
        get_idx_from_strides<TYPE>(shape, lhs_strides, num_dims, idx);         \
    int idx_rhs =                                                              \
        get_idx_from_strides<TYPE>(shape, rhs_strides, num_dims, idx);         \
                                                                               \
    TYPE x = lhs[idx_lhs];                                                     \
    TYPE y = rhs[idx_rhs];                                                     \
    out[idx] = FN;                                                             \
  }

#define KERNEL_PARAMS_BIN(TYPE)                                                \
  const long *lhs_strides, const long *rhs_strides, const size_t *shape,       \
      const size_t num_dims, const TYPE *lhs, const TYPE *rhs, TYPE *out

namespace pg {
namespace cuda {
__global__ void add_kernel(KERNEL_PARAMS_BIN(float));
__global__ void add_kernel(KERNEL_PARAMS_BIN(double));
__global__ void add_kernel(KERNEL_PARAMS_BIN(int));

__global__ void sub_kernel(KERNEL_PARAMS_BIN(float));
__global__ void sub_kernel(KERNEL_PARAMS_BIN(double));
__global__ void sub_kernel(KERNEL_PARAMS_BIN(int));

__global__ void mult_kernel(KERNEL_PARAMS_BIN(float));
__global__ void mult_kernel(KERNEL_PARAMS_BIN(double));
__global__ void mult_kernel(KERNEL_PARAMS_BIN(int));

__global__ void div_kernel(KERNEL_PARAMS_BIN(float));
__global__ void div_kernel(KERNEL_PARAMS_BIN(double));
__global__ void div_kernel(KERNEL_PARAMS_BIN(int));

__global__ void greater_kernel(KERNEL_PARAMS_BIN(float));
__global__ void greater_kernel(KERNEL_PARAMS_BIN(double));
__global__ void greater_kernel(KERNEL_PARAMS_BIN(int));

__global__ void less_kernel(KERNEL_PARAMS_BIN(float));
__global__ void less_kernel(KERNEL_PARAMS_BIN(double));
__global__ void less_kernel(KERNEL_PARAMS_BIN(int));

__global__ void equal_kernel(KERNEL_PARAMS_BIN(float));
__global__ void equal_kernel(KERNEL_PARAMS_BIN(double));
__global__ void equal_kernel(KERNEL_PARAMS_BIN(int));

__global__ void not_equal_kernel(KERNEL_PARAMS_BIN(float));
__global__ void not_equal_kernel(KERNEL_PARAMS_BIN(double));
__global__ void not_equal_kernel(KERNEL_PARAMS_BIN(int));

__global__ void greater_equal_kernel(KERNEL_PARAMS_BIN(float));
__global__ void greater_equal_kernel(KERNEL_PARAMS_BIN(double));
__global__ void greater_equal_kernel(KERNEL_PARAMS_BIN(int));

__global__ void less_equal_kernel(KERNEL_PARAMS_BIN(float));
__global__ void less_equal_kernel(KERNEL_PARAMS_BIN(double));
__global__ void less_equal_kernel(KERNEL_PARAMS_BIN(int));

__global__ void element_wise_max_kernel(KERNEL_PARAMS_BIN(float));
__global__ void element_wise_max_kernel(KERNEL_PARAMS_BIN(double));
__global__ void element_wise_max_kernel(KERNEL_PARAMS_BIN(int));

__global__ void pow_kernel(KERNEL_PARAMS_BIN(float));
__global__ void pow_kernel(KERNEL_PARAMS_BIN(double));
__global__ void pow_kernel(KERNEL_PARAMS_BIN(int));

enum class BinaryKernelType {
  ADD,
  SUB,
  MULT,
  DIV,
  GREATER,
  LESS,
  EQUAL,
  NOT_EQUAL,
  GREATER_EQUAL,
  LESS_EQUAL,
  ELEMENT_WISE_MAX,
  POW
};

template <typename T>
void launch_binary_kernel_casted_helper(BinaryKernelType kernel_type,
                                        dim3 grid_size, dim3 block_size,
                                        const stride_t *lhs_strides,
                                        const stride_t *rhs_strides,
                                        const size_t *shape,
                                        const size_t num_dims, const T *lhs,
                                        const T *rhs, T *out) {
  size_t total_dynamic_smem =
      sizeof(stride_t) * num_dims * 2 +
      sizeof(size_t) * num_dims; // two strides and shape
  switch (kernel_type) {
  case BinaryKernelType::ADD:
    add_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::SUB:
    sub_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::MULT:
    mult_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::DIV:
    div_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::GREATER:
    greater_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::LESS:
    less_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::EQUAL:
    equal_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::NOT_EQUAL:
    not_equal_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::GREATER_EQUAL:
    greater_equal_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::LESS_EQUAL:
    less_equal_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::ELEMENT_WISE_MAX:
    element_wise_max_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  case BinaryKernelType::POW:
    pow_kernel<<<grid_size, block_size, total_dynamic_smem>>>(
        lhs_strides, rhs_strides, shape, num_dims, lhs, rhs, out);
    break;
  default:
    throw std::runtime_error("Invalid kernel type");
  }
}

void dispatch_binary_op(BinaryKernelType kernel_type, DType dtype,
                        size_t numels, const stride_t *lhs_strides,
                        const stride_t *rhs_strides, const size_t *shape,
                        const size_t num_dims, const void *lhs, const void *rhs,
                        void *out);
} // namespace cuda
} // namespace pg