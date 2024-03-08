#pragma once

#include "cuda_array.cuh"
#include "dtype.hpp"
#include "kernels/all.cuh"
#include "utils.cuh"
#include <cmath>
#include <iostream>
#include <string>

#define MAX_THREADS_PER_BLOCK 512

CudaArray CudaArray::im2col(shape_t kernel_shape, int stride_y, int stride_x,
                            int dilation_y, int dilation_x) const {
  if (!is_contiguous()) {
    return as_contiguous().im2col(kernel_shape, stride_y, stride_x, dilation_y,
                                  dilation_x);
  }
  PG_CHECK_ARG(ndim() == 4, "ndim has to be 4 in im2col, got shape ",
               vec_to_string(shape));
  PG_CHECK_ARG(kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               vec_to_string(kernel_shape));
  size_t k_h = kernel_shape[0];
  size_t k_w = kernel_shape[1];

  size_t batch_size = shape[0];
  size_t in_channels = shape[1];
  size_t x_h = shape[2];
  size_t x_w = shape[3];

  size_t out_h = (x_h - dilation_y * (k_h - 1) - 1) / stride_y + 1;
  size_t out_w = (x_w - dilation_x * (k_w - 1) - 1) / stride_x + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, in_channels * k_h * k_w, out_h * out_w};
  size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                    std::multiplies<size_t>());

  CudaArray out(out_size, out_shape, dtype);

  int total_iters = batch_size * out_h * out_w * in_channels * k_h *
                    k_w; // check kernel code for more details
  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_iters / (float)block_size);

  launch_im2col_kernel(dtype, grid_size, block_size, get_base_ptr(), out.get_base_ptr(),
                       k_h, k_w, x_h, x_w, stride_x, stride_y, batch_size,
                       in_channels, dilation_x, dilation_y);
  PG_CUDA_KERNEL_END;
  return out;
}

CudaArray CudaArray::col2im(shape_t kernel_shape, shape_t out_shape,
                            int stride_y, int stride_x, int dilation_y,
                            int dilation_x) const {
  if (!is_contiguous()) {
    return as_contiguous().col2im(kernel_shape, out_shape, stride_y, stride_x,
                                  dilation_y, dilation_x);
  }

  PG_CHECK_ARG(ndim() == 3, "ndim has to be 3 in col2im, got shape ",
               vec_to_string(shape));
  PG_CHECK_ARG(kernel_shape.size() == 2, "kernel shape size must be 2, got ",
               vec_to_string(kernel_shape));
  PG_CHECK_ARG(out_shape.size() == 2, "out shape size must be 2, got ",
               vec_to_string(out_shape));

  size_t k_h = kernel_shape[0];
  size_t k_w = kernel_shape[1];
  size_t out_h = out_shape[0];
  size_t out_w = out_shape[1];
  size_t in_h = shape[1];
  size_t in_w = shape[2];

  // out_shape is just (out_h, out_w)
  size_t out_channels = shape[1] / (k_h * k_w);

  size_t out_batch_size = shape[0];
  shape_t _out_shape = {out_batch_size, out_channels, out_h, out_w};
  size_t out_size = std::accumulate(_out_shape.begin(), _out_shape.end(), 1,
                                    std::multiplies<size_t>());
  CudaArray out(out_size, _out_shape, dtype);
  CHECK_CUDA(cudaMemset(out.get_base_ptr(), 0, out_size * dtype_to_size(dtype)));

  dim3 block_size(DEFAULT_BLOCK_SIZE);
  // batch size and out_channels are parallelized
  dim3 grid_size(
      ceil(out_batch_size * out_channels / (float)DEFAULT_BLOCK_SIZE));
  launch_col2im_kernel(dtype, grid_size, block_size, get_base_ptr(), out.get_base_ptr(),
                       out_channels, k_h, k_w, in_h, in_w, out_batch_size,
                       out_h, out_w, stride_x, stride_y, dilation_x,
                       dilation_y);
  PG_CUDA_KERNEL_END;
  return out;
}

bool CudaArray::is_contiguous() const {
  if (strides.size() != shape.size()) {
    return false;
  }
  if (strides.size() == 0) { // scalar
    return true;
  }
  shape_t expected_strides(shape.size());
  expected_strides[shape.size() - 1] = dtype_to_size(dtype);
  for (int i = shape.size() - 2; i >= 0; --i) {
    expected_strides[i] = expected_strides[i + 1] * shape[i + 1];
  }
  if (expected_strides != strides) {
    return false;
  }
  return true;
}

CudaArray::CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
                     const std::shared_ptr<void> &ptr, DType dtype)
    : size(size), shape(shape), strides(strides), ptr(ptr), dtype(dtype), offset(0) {}

CudaArray::CudaArray(size_t size, shape_t shape, shape_t strides, DType dtype)
    : size(size), shape(shape), strides(strides), dtype(dtype), offset(0) {
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * dtype_to_size(dtype)));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

CudaArray::CudaArray(size_t size, shape_t shape, DType dtype)
    : size(size), shape(shape), dtype(dtype), offset(0) {
  strides.resize(shape.size());
  // Only calculate strides if we don't have a scalar
  if (shape.size() > 0) {
    strides[shape.size() - 1] = dtype_to_size(dtype);
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  void *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * dtype_to_size(dtype)));
  ptr =
      std::shared_ptr<void>(raw_ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

CudaArray CudaArray::broadcast_to(const shape_t _shape) const {
  const shape_t shape_from = this->shape;
  const shape_t shape_to = _shape;
  // determine if we can broadcast
  const int from_ndim = (const int)shape_from.size();
  const int to_ndim = (const int)shape_to.size();
  // cannot broadcast if the number of dimensions of the from array is greater
  // than the number of dimensions of the to array
  PG_CHECK_ARG(from_ndim <= to_ndim,
               "from_ndim must be <= to_ndim, trying to broadcast from ",
               vec_to_string(shape_from), " to ", vec_to_string(shape_to));

  int new_size = 1;
  shape_t new_strides(to_ndim, 0);
  // reverse test if the dim is 1 or they are equal
  for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
    py::ssize_t dim_to = shape_to[i];
    py::ssize_t dim_from =
        (j >= 0) ? shape_from[j]
                 : -1; // -1 means we 'ran' out of dimensions for j

    PG_CHECK_ARG(dim_to == dim_from || dim_from == 1 || dim_from == -1,
                 "got incompatible shapes: ", vec_to_string(shape_from),
                 " cannot be broadcasted to ", vec_to_string(shape_to),
                 ". In dimension ", i, " got dim_to=", dim_to,
                 " and dim_from=", dim_from);

    if (dim_from != 1 && dim_from != -1) {
      new_strides[i] = strides[j];
    }
    new_size *= dim_to;
  }
  CudaArray out(new_size, shape_to, new_strides, dtype);
  CHECK_CUDA(cudaMemcpy(out.get_base_ptr(), get_base_ptr(), size * dtype_to_size(dtype),
                        cudaMemcpyDeviceToDevice));
  return out;
}

CudaArray CudaArray::astype(DType new_type) const {
  if (dtype == new_type) {
    return *this;
  }
  CudaArray out(size, shape, new_type);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  auto &in_strides = cuda_unique_ptr_from_host(shape.size(), strides.data());
  auto &in_shape = cuda_unique_ptr_from_host(shape.size(), this->shape.data());
  launch_astype_kernel(dtype, new_type, grid_size, block_size, in_strides.get(),
                       in_shape.get(), ndim(), get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;

  return out;
}

CudaArray CudaArray::binop_same_dtype(const CudaArray &other,
                                      BinaryKernelType kt) const {
  if (shape != other.shape) {
    // try to broadcast, from smaller to larger
    if (shape.size() < other.shape.size()) {
      return broadcast_to(other.shape).binop_same_dtype(other, kt);
    } else if (shape.size() > other.shape.size()) {
      return binop_same_dtype(other.broadcast_to(shape), kt);
    } else {
      // we need to check the one with less product of shape, and try to
      // broadcast
      int64_t prod_shape = 1;
      int64_t prod_other_shape = 1;
      for (int i = 0; i < shape.size(); i++) {
        prod_shape *= shape[i];
        prod_other_shape *= other.shape[i];
      }
      if (prod_shape < prod_other_shape) {
        return broadcast_to(other.shape).binop_same_dtype(other, kt);
      } else {
        return binop_same_dtype(other.broadcast_to(shape), kt);
      }
    }
  }
  assert(shape == other.shape);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  // Default stride calculation
  CudaArray out(size, shape, dtype);
  size_t n_dims = shape.size();

  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_other_strides =
      cuda_unique_ptr_from_host(n_dims, other.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());

  launch_binary_kernel(kt, dtype, grid_size, block_size, d_strides.get(),
                       d_other_strides.get(), d_shape.get(), n_dims, get_base_ptr(),
                       other.get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return out;
}

CudaArray CudaArray::binop(const CudaArray &other, BinaryKernelType kt) const {
  // we need to decide on the casting, always biggest type
  return binop_same_dtype(other.astype(this->dtype), kt);
}

CudaArray CudaArray::ternaryop(const CudaArray &second, const CudaArray &third,
                               TernaryKernelType ker) const {
  if (second.shape != third.shape || shape != second.shape ||
      shape != third.shape) {
    size_t biggest_size =
        std::max({shape.size(), second.shape.size(), third.shape.size()});
    shape_t target_shape(biggest_size, 1);
    for (size_t i = 0; i < biggest_size; i++) {
      if (i < shape.size()) {
        target_shape[i] = shape[i];
      }
      if (i < second.shape.size()) {
        target_shape[i] = std::max(target_shape[i], second.shape[i]);
      }
      if (i < third.shape.size()) {
        target_shape[i] = std::max(target_shape[i], third.shape[i]);
      }
    }

    return broadcast_to(target_shape)
        .ternaryop(second.broadcast_to(target_shape),
                   third.broadcast_to(target_shape), ker);
  }
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));

  // Default stride calculation
  CudaArray out(size, shape, dtype);
  size_t n_dims = shape.size();
  cuda_unique_ptr<size_t> d_first_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_second_strides =
      cuda_unique_ptr_from_host(n_dims, second.strides.data());
  cuda_unique_ptr<size_t> d_third_strides =
      cuda_unique_ptr_from_host(n_dims, third.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  launch_ternary_kernel(ker, dtype, grid_size, block_size,
                        d_first_strides.get(), d_second_strides.get(),
                        d_third_strides.get(), d_shape.get(), n_dims, get_base_ptr(),
                        second.get_base_ptr(), third.get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return out;
}

int CudaArray::ndim() const { return shape.size(); }

CudaArray CudaArray::mat_mul(const CudaArray &other) const {
  // if a is a vector and the other is a matrix, add a dimension to a
  bool added_a_dim = false;
  bool added_b_dim = false;
  CudaArray a = (this->ndim() == 1 && other.ndim() != 1)
                    ? this->unsqueeze(0)
                    : this->as_contiguous();
  if (this->ndim() == 1 && other.ndim() != 1) {
    added_a_dim = true;
  }
  CudaArray b = (other.ndim() == 1 && this->ndim() != 1)
                    ? other.unsqueeze(1)
                    : other.as_contiguous();
  if (other.ndim() == 1 && this->ndim() != 1) {
    added_b_dim = true;
  }
  dim3 block_size = dim3(DEFAULT_BLOCK_SIZE);
  shape_t new_shape;
  size_t size1, midsize, size2;
  if (a.ndim() == 1 && b.ndim() == 1) {
    PG_CHECK_ARG(a.shape == b.shape,
                 "shapes must be equal in vector dot prod, got ",
                 vec_to_string(a.shape), " and ", vec_to_string(b.shape));
    // vector_dot_product_accum accumulates vector_a * vector_b, but if the size
    // is too large, it will not accumulate all of that into a single value, but
    // rather into a vector of size (size / MAX_THREADS_PER_BLOCK) + 1 check its
    // implementation for more details
    int new_size = (a.shape.at(0) / MAX_THREADS_PER_BLOCK) + 1;
    CudaArray out(new_size, {(size_t)new_size}, dtype);

    launch_vector_dot_product_accum_kernel(
        dim3(new_size), dim3(MAX_THREADS_PER_BLOCK),
        MAX_THREADS_PER_BLOCK * dtype_to_size(dtype), dtype, a.get_base_ptr(),
        b.get_base_ptr(), out.get_base_ptr(), a.shape.at(0));
    PG_CUDA_KERNEL_END;
    if (new_size > 1) {
      // if size > 1, we need to reduce the vector to a single value
      return out.sum(false);
    }
    return out.squeeze();
  } else {
    int a_prod = std::accumulate(a.shape.begin(), a.shape.end() - 2, 1,
                                 std::multiplies<int>());
    int b_prod = std::accumulate(b.shape.begin(), b.shape.end() - 2, 1,
                                 std::multiplies<int>());
    if (a.ndim() > b.ndim()) { // we will try to broadcast, but keep
                               // last to dims
      shape_t b_new = shape_t(a.shape);
      b_new[b_new.size() - 1] = b.shape[b.shape.size() - 1];
      b_new[b_new.size() - 2] = b.shape[b.shape.size() - 2];
      b = b.broadcast_to(b_new).as_contiguous();
    } else if (a.ndim() < b.ndim()) {
      shape_t a_new = shape_t(b.shape);
      a_new[a_new.size() - 1] = a.shape[a.shape.size() - 1];
      a_new[a_new.size() - 2] = a.shape[a.shape.size() - 2];
      a = a.broadcast_to(a_new).as_contiguous();
      // if ndim are equal, we will broadcast the one with the smallest product
    } else if (a_prod >= b_prod) {
      shape_t b_new = shape_t(a.shape);
      b_new[b_new.size() - 1] = b.shape[b.shape.size() - 1];
      b_new[b_new.size() - 2] = b.shape[b.shape.size() - 2];
      b = b.broadcast_to(b_new).as_contiguous();
    } else if (a_prod < b_prod) {
      shape_t a_new = shape_t(b.shape);
      a_new[a_new.size() - 1] = a.shape[a.shape.size() - 1];
      a_new[a_new.size() - 2] = a.shape[a.shape.size() - 2];
      a = a.broadcast_to(a_new).as_contiguous();
    }
    size1 = a.shape.at(a.ndim() - 2);
    midsize = a.shape.at(a.ndim() - 1);
    size2 = b.shape.at(b.ndim() - 1);
    new_shape = a.shape;
    new_shape[new_shape.size() - 1] = size2;
    if (added_a_dim) {
      new_shape.erase(new_shape.begin());
    }
    if (added_b_dim) {
      new_shape.erase(new_shape.end() - 1);
    }
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>());

    dim3 gridSize(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
    CudaArray out(new_size, new_shape, dtype);
    cuda_unique_ptr<size_t> lhs_shape =
        cuda_unique_ptr_from_host(a.ndim(), a.shape.data());
    cuda_unique_ptr<size_t> rhs_shape =
        cuda_unique_ptr_from_host(b.ndim(), b.shape.data());
    launch_batched_matmul_kernel(gridSize, block_size, dtype, a.get_base_ptr(),
                                 b.get_base_ptr(), out.get_base_ptr(), lhs_shape.get(),
                                 rhs_shape.get(), a.ndim());
    PG_CUDA_KERNEL_END;
    return out;
  }
}

CudaArray CudaArray::outer_product(const CudaArray &other) const {
  PG_CHECK_ARG(ndim() == 1 && other.ndim() == 1,
               "got non vectors in outer product, shapes: ",
               vec_to_string(shape), " and ", vec_to_string(other.shape));
  int total_idxs = size * other.size;
  dim3 grid_size(ceil(total_idxs / (float)DEFAULT_BLOCK_SIZE));
  shape_t new_shape = {size, other.size};
  CudaArray out(total_idxs, new_shape, dtype);
  launch_vector_outer_product_kernel(grid_size, DEFAULT_BLOCK_SIZE, dtype,
                                     get_base_ptr(), other.get_base_ptr(), out.get_base_ptr(),
                                     size, other.size);
  PG_CUDA_KERNEL_END;
  return out;
}

CudaArray CudaArray::reduce(ReduceKernelType ker, axis_t axis,
                            bool keepdims) const {
  if (!is_contiguous()) {
    return as_contiguous().reduce(ker, axis, keepdims);
  }
  // if axis is negative, we need to convert it to a positive axis
  if (axis < 0) {
    axis = shape.size() + axis;
  }
  PG_CHECK_ARG(axis < shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  shape_t new_shape = shape;
  new_shape[axis] = 1;
  size_t new_size = size / shape[axis];
  size_t n_dims = shape.size();
  CudaArray out(new_size, new_shape, dtype);
  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
  launch_reduce_kernel(ker, dtype, grid_size, block_size, get_base_ptr(),
                       out.get_base_ptr(), d_strides.get(), d_shape.get(), n_dims,
                       axis);
  PG_CUDA_KERNEL_END;
  if (keepdims) {
    return out;
  }
  return out.squeeze(axis);
}

CudaArray CudaArray::reduce(ReduceKernelType ker, axes_t axes,
                            bool keepdims) const {
  CudaArray out = *this;
  for (size_t axis : axes) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze(axes);
}

CudaArray CudaArray::reduce(ReduceKernelType ker, bool keepdims) const {
  CudaArray out = *this;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze();
}

CudaArray CudaArray::max(bool keepdims) const {
  return reduce(ReduceKernelType::MAX, keepdims);
}

CudaArray CudaArray::max(axes_t axes, bool keepdims) const {
  return reduce(ReduceKernelType::MAX, axes, keepdims);
}

CudaArray CudaArray::max(axis_t axis, bool keepdims) const {
  return reduce(ReduceKernelType::MAX, axis, keepdims);
}

CudaArray CudaArray::sum(bool keepdims) const {
  return reduce(ReduceKernelType::SUM, keepdims);
}

CudaArray CudaArray::sum(axes_t axes, bool keepdims) const {
  return reduce(ReduceKernelType::SUM, axes, keepdims);
}

CudaArray CudaArray::sum(axis_t axis, bool keepdims) const {
  return reduce(ReduceKernelType::SUM, axis, keepdims);
}

CudaArray CudaArray::squeeze(axis_t axis) const {
  if (axis < 0) {
    axis = shape.size() + axis;
  }
  PG_CHECK_ARG(axis < shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  PG_CHECK_ARG(shape[axis] == 1,
               "cannot squeeze on a dimension that is not 1, got ", shape[axis],
               " in axis number ", axis, " for shape ", vec_to_string(shape));

  CudaArray out(*this);
  out.shape.erase(out.shape.begin() + axis);
  out.strides.erase(out.strides.begin() + axis);

  return out;
}

CudaArray CudaArray::squeeze(axes_t _axes) const {
  CudaArray out(*this);
  // since axes may not be sorted, we need to sort them first, substituting
  // negatives first and then sorting
  axes_t axes = _axes;
  for (int i = 0; i < axes.size(); i++) {
    if (axes[i] < 0) {
      axes[i] = shape.size() + axes[i];
    }
  }
  // squeeze in reverse order
  std::sort(axes.begin(), axes.end(), std::greater<int>());
  for (size_t axis : axes) {
    out = out.squeeze(axis);
  }
  return out;
}

CudaArray CudaArray::squeeze() const {
  CudaArray out(*this);
  // squeezes all dims that are 1
  shape_t indices_to_squeeze;

  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == 1) {
      indices_to_squeeze.push_back(i);
    }
  }

  shape_t new_shape(shape.size() - indices_to_squeeze.size());
  shape_t new_strides(strides.size() - indices_to_squeeze.size());

  for (int i = 0, j = 0; i < shape.size(); i++) {
    if (std::find(indices_to_squeeze.begin(), indices_to_squeeze.end(), i) ==
        indices_to_squeeze.end()) {
      new_shape[j] = shape[i];
      new_strides[j] = strides[i];
      j++;
    }
  }
  out.shape = new_shape;
  out.strides = new_strides;
  return out;
}

CudaArray CudaArray::unsqueeze(axes_t axes) const {
  CudaArray out(*this);
  for (size_t axis : axes) {
    out = out.unsqueeze(axis);
  }
  return out;
}

CudaArray CudaArray::unsqueeze(axis_t axis) const {
  if (axis < 0) {
    axis = shape.size() + axis + 1;
  }
  PG_CHECK_ARG(axis <= shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  CudaArray out(*this);
  out.shape.insert(out.shape.begin() + axis, 1);
  size_t new_stride = (axis < strides.size())
                          ? strides[std::max(0, (int)axis - 1)]
                          : dtype_to_size(dtype);
  out.strides.insert(out.strides.begin() + axis, new_stride);
  return out;
}

CudaArray CudaArray::reshape(std::vector<int> &_new_shape) const {
  shape_t new_shape(_new_shape.size());
  size_t total_new = 1;

  int neg_pos = -1;
  for (size_t i = 0; i < _new_shape.size(); i++) {
    if (_new_shape[i] < 0) {
      PG_CHECK_ARG(
          neg_pos == -1,
          "Can only specify one unknown dimension (-1) for reshape, got ",
          neg_pos, " and ", i, " for shape ", vec_to_string(_new_shape));
      neg_pos = i;
    }
    new_shape[i] = _new_shape[i];
    total_new *= new_shape[i] == -1 ? 1 : new_shape[i];
  }

  size_t total_old =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (neg_pos != -1) {
    new_shape[neg_pos] = total_old / total_new;
    PG_CHECK_ARG(
        total_old % total_new == 0,
        "New shape is not compatible with old shape: ", vec_to_string(shape),
        " not compatible with ", vec_to_string(_new_shape));
  }
  total_new = total_old;
  // if first array is contiguous, return a 'view' of the array
  if (is_contiguous()) {
    shape_t new_strides(new_shape.size());
    for (int i = new_shape.size() - 1; i >= 0; --i) {
      new_strides[i] = (i == new_shape.size() - 1)
                           ? dtype_to_size(dtype)
                           : new_strides[i + 1] * new_shape[i + 1];
    }
    return CudaArray(total_new, new_shape, new_strides, ptr, dtype);
  }
  CudaArray out(total_new, new_shape, dtype);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(total_new / (float)DEFAULT_BLOCK_SIZE));
  auto &in_strides = cuda_unique_ptr_from_host(shape.size(), strides.data());
  auto &in_shape = cuda_unique_ptr_from_host(shape.size(), shape.data());
  auto &out_shape =
      cuda_unique_ptr_from_host(new_shape.size(), new_shape.data());
  auto &out_strides =
      cuda_unique_ptr_from_host(new_shape.size(), out.strides.data());

  launch_copy_with_out_strides_kernel(
      dtype, grid_size, block_size, in_strides.get(), in_shape.get(),
      out_strides.get(), out_shape.get(), ndim(), out.ndim(), get_base_ptr(),
      out.get_base_ptr());

  PG_CUDA_KERNEL_END;
  return out;
}

std::string CudaArray::to_string() const {
  /*void *host = malloc(size * dtype_to_size(dtype));
  CHECK_CUDA(
      cudaMemcpy(host, get_base_ptr(), size * sizeof(T), cudaMemcpyDeviceToHost));
  */
  std::stringstream ss;
  ss << "CudaArray<" << dtype_to_string(dtype) << ">(" << size
     << ") with shape " << vec_to_string(shape) << " and strides "
     << vec_to_string(strides);
  return ss.str();
}

CudaArray::~CudaArray() {}

CudaArray::CudaArray(const CudaArray &other)
    : size(other.size), shape(other.shape), strides(other.strides),
      ptr(other.ptr), dtype(other.dtype), offset(other.offset) {}

CudaArray &CudaArray::operator=(const CudaArray &other) {
  if (this != &other) {
    size = other.size;
    shape = other.shape;
    strides = other.strides;
    ptr = other.ptr;
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}

CudaArray::CudaArray(CudaArray &&other)
    : size(other.size), shape(std::move(other.shape)),
      strides(std::move(other.strides)), ptr(std::move(other.ptr)),
      dtype(other.dtype), offset(other.offset) {}

CudaArray &CudaArray::operator=(CudaArray &&other) {
  if (this != &other) {
    size = other.size;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    ptr = std::move(other.ptr);
    dtype = other.dtype;
    offset = other.offset;
  }
  return *this;
}

CudaArray CudaArray::clone() const {
  CudaArray out(size, shape, strides, dtype);
  CHECK_CUDA(cudaMemcpy(out.get_base_ptr(), get_base_ptr(), size * dtype_to_size(dtype),
                        cudaMemcpyDeviceToDevice));
  return out;
}

CudaArray CudaArray::elwiseop(UnaryKernelType ker) const {
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  size_t n_dims = shape.size();
  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  CudaArray out(size, shape, dtype);

  launch_unary_kernel(ker, dtype, grid_size, block_size, d_strides.get(),
                      d_shape.get(), n_dims, get_base_ptr(), out.get_base_ptr());
  PG_CUDA_KERNEL_END;
  return out;
}

CudaArray CudaArray::as_contiguous() const {
  return is_contiguous() ? *this : elwiseop(UnaryKernelType::COPY);
}

CudaArray CudaArray::permute(shape_t axes) const {
  PG_CHECK_ARG(axes.size() == shape.size(),
               "axes must have same size as shape, got ", axes.size(), " and ",
               shape.size());
  shape_t new_shape(shape.size());
  shape_t new_strides(strides.size());

  for (size_t i = 0; i < axes.size(); ++i) {
    new_shape[i] = shape[axes[i]];
    new_strides[i] = strides[axes[i]];
  }

  CudaArray out(size, new_shape, new_strides, ptr, dtype);
  return out;
}