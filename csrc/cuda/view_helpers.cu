#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "unary.cuh"
#include "utils.hpp"
#include "view_helpers.cuh"

namespace pg {
namespace cuda {
template <typename T>
__global__ void copy_kernel_fast(const stride_t *in_strides,
                                 const size_t *in_shape, const size_t num_dims,
                                 const T *in, T *out) {
  extern __shared__ int8_t smem[];
  const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t *shape = (size_t *)smem;
  stride_t *in_strides_smem = (stride_t *)(smem + num_dims * sizeof(size_t));
  if (threadIdx.x < num_dims) {
    in_strides_smem[threadIdx.x] = in_strides[threadIdx.x];
    shape[threadIdx.x] = in_shape[threadIdx.x];
  }
  __syncthreads();
  for (int i = 0; i < 4; ++i) {
    int idx = base_idx + i * blockDim.x * gridDim.x;
    if (get_max_idx(shape, num_dims) <= idx)
      return;
    int in_idx = get_idx_from_strides<T>(shape, in_strides_smem, num_dims, idx);
    out[idx] = in[in_idx];
  }
}

template <typename T>
__global__ void copy_kernel_3d(const stride_t instr0, const stride_t instr1,
                               const stride_t instr2, const uint64_t inshp0,
                               const uint64_t inshp1, const uint64_t inshp2,
                               const uint64_t outshp0, const uint64_t outshp1,
                               const uint64_t outshp2, const T *in, T *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= inshp0 * inshp1 * inshp2)
    return;
  const int i = idx / (inshp1 * inshp2);
  const int j = (idx / inshp2) % inshp1;
  const int k = idx % inshp2;
  const int in_idx = i * instr0 + j * instr1 + k * instr2;
  const int out_idx = i * outshp1 * outshp2 + j * outshp2 + k;
  out[out_idx] = in[in_idx];
}

// 4 and 5d
template <typename T>
__global__ void copy_kernel_4d(const stride_t instr0, const stride_t instr1,
                               const stride_t instr2, const stride_t instr3,
                               const uint64_t inshp0, const uint64_t inshp1,
                               const uint64_t inshp2, const uint64_t inshp3,
                               const uint64_t outshp0, const uint64_t outshp1,
                               const uint64_t outshp2, const uint64_t outshp3,
                               const T *in, T *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= inshp0 * inshp1 * inshp2 * inshp3)
    return;
  const int i = idx / (inshp1 * inshp2 * inshp3);
  const int j = (idx / (inshp2 * inshp3)) % inshp1;
  const int k = (idx / inshp3) % inshp2;
  const int l = idx % inshp3;
  const int in_idx = i * instr0 + j * instr1 + k * instr2 + l * instr3;
  const int out_idx =
      i * outshp1 * outshp2 * outshp3 + j * outshp2 * outshp3 + k * outshp3 + l;
  out[out_idx] = in[in_idx];
}

template <typename T>
__global__ void copy_kernel_5d(const stride_t instr0, const stride_t instr1,
                               const stride_t instr2, const stride_t instr3,
                               const stride_t instr4, const uint64_t inshp0,
                               const uint64_t inshp1, const uint64_t inshp2,
                               const uint64_t inshp3, const uint64_t inshp4,
                               const uint64_t outshp0, const uint64_t outshp1,
                               const uint64_t outshp2, const uint64_t outshp3,
                               const uint64_t outshp4, const T *in, T *out) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= inshp0 * inshp1 * inshp2 * inshp3 * inshp4)
    return;
  const int i = idx / (inshp1 * inshp2 * inshp3 * inshp4);
  const int j = (idx / (inshp2 * inshp3 * inshp4)) % inshp1;
  const int k = (idx / (inshp3 * inshp4)) % inshp2;
  const int l = (idx / inshp4) % inshp3;
  const int m = idx % inshp4;
  const int in_idx =
      i * instr0 + j * instr1 + k * instr2 + l * instr3 + m * instr4;
  const int out_idx = i * outshp1 * outshp2 * outshp3 * outshp4 +
                      j * outshp2 * outshp3 * outshp4 + k * outshp3 * outshp4 +
                      l * outshp4 + m;
  out[out_idx] = in[in_idx];
}

namespace view {
View as_contiguous(const View &view) {
  if (view.is_contiguous()) {
    return view;
  }
  View contiguous_view = View(view.shape(), view.dtype(), view.device());

  if (view.ndim() == 3) {
    PG_DISPATCH_ALL_TYPES(view.dtype(), "as_contiguous_3d", [&]() {
      cuda::copy_kernel_3d<scalar_t>
          <<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
             dim3(DEFAULT_BLOCK_SIZE)>>>(
              view.strides()[0], view.strides()[1], view.strides()[2],
              view.shape()[0], view.shape()[1], view.shape()[2],
              contiguous_view.shape()[0], contiguous_view.shape()[1],
              contiguous_view.shape()[2], view.get_casted_base_ptr<scalar_t>(),
              contiguous_view.get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
    return contiguous_view;
  }

  if (view.ndim() == 4) {
    PG_DISPATCH_ALL_TYPES(view.dtype(), "as_contiguous_4d", [&]() {
      cuda::copy_kernel_4d<scalar_t>
          <<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
             dim3(DEFAULT_BLOCK_SIZE)>>>(
              view.strides()[0], view.strides()[1], view.strides()[2],
              view.strides()[3], view.shape()[0], view.shape()[1],
              view.shape()[2], view.shape()[3], contiguous_view.shape()[0],
              contiguous_view.shape()[1], contiguous_view.shape()[2],
              contiguous_view.shape()[3], view.get_casted_base_ptr<scalar_t>(),
              contiguous_view.get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
    return contiguous_view;
  }

  if (view.ndim() == 5) {
    PG_DISPATCH_ALL_TYPES(view.dtype(), "as_contiguous_5d", [&]() {
      cuda::copy_kernel_5d<scalar_t>
          <<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
             dim3(DEFAULT_BLOCK_SIZE)>>>(
              view.strides()[0], view.strides()[1], view.strides()[2],
              view.strides()[3], view.strides()[4], view.shape()[0],
              view.shape()[1], view.shape()[2], view.shape()[3],
              view.shape()[4], contiguous_view.shape()[0],
              contiguous_view.shape()[1], contiguous_view.shape()[2],
              contiguous_view.shape()[3], contiguous_view.shape()[4],
              view.get_casted_base_ptr<scalar_t>(),
              contiguous_view.get_casted_base_ptr<scalar_t>());
    });
    PG_CUDA_KERNEL_END;
    return contiguous_view;
  }

  auto d_shape =
      cuda_unique_ptr_from_host(view.shape().size(), view.shape().data());
  auto d_strides =
      cuda_unique_ptr_from_host(view.strides().size(), view.strides().data());

  PG_DISPATCH_ALL_TYPES(view.dtype(), "as_contiguous", [&]() {
    size_t smem = sizeof(size_t) * view.ndim() + sizeof(stride_t) * view.ndim();
    cuda::copy_kernel_fast<<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) /
                                  DEFAULT_BLOCK_SIZE),
                             dim3(DEFAULT_BLOCK_SIZE), smem>>>(
        d_strides.get(), d_shape.get(), view.shape().size(),
        view.get_casted_base_ptr<scalar_t>(),
        contiguous_view.get_casted_base_ptr<scalar_t>());
  });
  PG_CUDA_KERNEL_END;
  return contiguous_view;
}
void copy(const View &src, const View &dst) {
  auto d_src_shape =
      cuda_unique_ptr_from_host(src.shape().size(), src.shape().data());
  auto d_src_strides =
      cuda_unique_ptr_from_host(src.strides().size(), src.strides().data());
  auto d_dst_shape =
      cuda_unique_ptr_from_host(dst.shape().size(), dst.shape().data());
  auto d_dst_strides =
      cuda_unique_ptr_from_host(dst.strides().size(), dst.strides().data());

  PG_DISPATCH_ALL_TYPES(src.dtype(), "copy_with_out_strides_kernel", [&]() {
    cuda::copy_with_out_strides_kernel<scalar_t>
        <<<dim3((src.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
           dim3(DEFAULT_BLOCK_SIZE)>>>(d_src_strides.get(), d_src_shape.get(),
                                       d_dst_strides.get(), d_dst_shape.get(),
                                       src.shape().size(), dst.shape().size(),
                                       src.get_casted_base_ptr<scalar_t>(),
                                       dst.get_casted_base_ptr<scalar_t>());
  });
  PG_CUDA_KERNEL_END;
}

View astype(const View &view, const DType &dtype) {
  if (view.dtype() == dtype) {
    return view;
  }
  View new_view = View(view.shape(), dtype, view.device());
  auto d_shape =
      cuda_unique_ptr_from_host(view.shape().size(), view.shape().data());
  auto d_strides =
      cuda_unique_ptr_from_host(view.strides().size(), view.strides().data());
  PG_DISPATCH_ALL_TYPES_TWO_TYPES(view.dtype(), dtype, "astype", [&]() {
    cuda::astype_kernel<scalar_t1, scalar_t2>
        <<<dim3((view.numel() + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE),
           dim3(DEFAULT_BLOCK_SIZE)>>>(
            d_strides.get(), d_shape.get(), view.shape().size(),
            view.get_casted_base_ptr<scalar_t1>(),
            new_view.get_casted_base_ptr<scalar_t2>());
  });
  PG_CUDA_KERNEL_END;
  return new_view;
}
} // namespace view
} // namespace cuda
} // namespace pg