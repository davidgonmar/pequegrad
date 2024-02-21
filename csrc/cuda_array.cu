#include "binary_ops_kernels.cuh"
#include "cuda_array.cuh"
#include "folding_kernels.cuh"
#include "init_kernels.cuh"
#include "matmul_kernels.cuh"
#include "reduce_ops_kernels.cuh"
#include "ternary_ops_kernels.cuh"
#include "unary_ops_kernels.cuh"
#include "utils.cuh"
#include <cmath>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 512

template <typename T> std::string vec_to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

// Read numpy storage docs
CudaArray CudaArray::im2col(shape_t kernel_shape, int stride) const {
  if (!is_contiguous()) {
    return as_contiguous().im2col(kernel_shape, stride);
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

  size_t out_h = (x_h - k_h) / stride + 1;
  size_t out_w = (x_w - k_w) / stride + 1;

  PG_CHECK_RUNTIME(out_h > 0 && out_w > 0,
                   "output height and width should be > 0, got out_h=", out_h,
                   " and out_w=", out_w);

  shape_t out_shape = {batch_size, in_channels * k_h * k_w, out_h * out_w};
  size_t out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                    std::multiplies<size_t>());

  CudaArray out(out_size, out_shape);

  int total_iters = batch_size * out_h * out_w * in_channels * k_h *
                    k_w; // check kernel code for more details
  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_iters / (float)block_size);
  im2col_kernel<<<grid_size, block_size>>>(ptr.get(), out.ptr.get(), k_h, k_w,
                                           x_h, x_w, stride, batch_size,
                                           in_channels);

  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}

CudaArray CudaArray::col2im(shape_t kernel_shape, shape_t out_shape,
                            int stride) const {
  if (!is_contiguous()) {
    return as_contiguous().col2im(kernel_shape, out_shape, stride);
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
  CudaArray out(out_size, _out_shape);
  CHECK_CUDA(cudaMemset(out.ptr.get(), 0, out_size * ELEM_SIZE));

  dim3 block_size(DEFAULT_BLOCK_SIZE);
  // batch size and out_channels are parallelized
  dim3 grid_size(
      ceil(out_batch_size * out_channels / (float)DEFAULT_BLOCK_SIZE));
  col2im_kernel<<<grid_size, block_size>>>(
      ptr.get(), out.ptr.get(), out_channels, k_h, k_w, in_h, in_w,
      out_batch_size, out_h, out_w, stride);

  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
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
  expected_strides[shape.size() - 1] = ELEM_SIZE;
  for (int i = shape.size() - 2; i >= 0; --i) {
    expected_strides[i] = expected_strides[i + 1] * shape[i + 1];
  }
  if (expected_strides != strides) {
    return false;
  }
  return true;
}

CudaArray::CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
                     const std::shared_ptr<float> &ptr)
    : size(size), shape(shape), strides(strides), ptr(ptr) {}

CudaArray::CudaArray(size_t size, shape_t shape, shape_t strides)
    : size(size), shape(shape), strides(strides) {
  float *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * ELEM_SIZE));
  ptr = std::shared_ptr<float>(raw_ptr, [](float *p) { cudaFree(p); });
}

CudaArray::CudaArray(size_t size, shape_t shape) : size(size), shape(shape) {
  strides.resize(shape.size());
  // Only calculate strides if we don't have a scalar
  if (shape.size() > 0) {
    strides[shape.size() - 1] = ELEM_SIZE;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  float *raw_ptr;
  CHECK_CUDA(cudaMalloc(&raw_ptr, size * ELEM_SIZE));
  ptr = std::shared_ptr<float>(raw_ptr, [](float *p) { cudaFree(p); });
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
  CudaArray out(new_size, shape_to, new_strides);
  CHECK_CUDA(cudaMemcpy(out.ptr.get(), ptr.get(), size * ELEM_SIZE,
                        cudaMemcpyDeviceToDevice));
  return out;
}

CudaArray CudaArray::binop(const py::array_t<float> &np_array,
                           binary_op_kernel ker) const {
  CudaArray other = CudaArray::from_numpy(np_array);
  return binop(other, ker);
}
CudaArray CudaArray::binop(const CudaArray &other, binary_op_kernel ker) const {
  if (shape != other.shape) {
    // try to broadcast, from smaller to larger
    if (shape.size() < other.shape.size()) {
      return broadcast_to(other.shape).binop(other, ker);
    } else if (shape.size() > other.shape.size()) {
      return binop(other.broadcast_to(shape), ker);
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
        return broadcast_to(other.shape).binop(other, ker);
      } else {
        return binop(other.broadcast_to(shape), ker);
      }
    }
  }
  assert(shape == other.shape);
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  // Default stride calculation
  CudaArray out(size, shape);
  size_t n_dims = shape.size();

  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_other_strides =
      cuda_unique_ptr_from_host(n_dims, other.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());

  ker<<<grid_size, block_size>>>(d_strides.get(), d_other_strides.get(),
                                 d_shape.get(), n_dims, ptr.get(),
                                 other.ptr.get(), out.ptr.get());
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}
CudaArray CudaArray::ternaryop(const py::array_t<float> &second,
                               const py::array_t<float> &third,
                               ternary_op_kernel ker) const {
  CudaArray second_arr = CudaArray::from_numpy(second);
  CudaArray third_arr = CudaArray::from_numpy(third);
  return ternaryop(second_arr, third_arr, ker);
}
CudaArray CudaArray::ternaryop(const CudaArray &second,
                               const py::array_t<float> &third,
                               ternary_op_kernel ker) const {
  CudaArray third_arr = CudaArray::from_numpy(third);
  return ternaryop(second, third_arr, ker);
}
CudaArray CudaArray::ternaryop(const py::array_t<float> &second,
                               const CudaArray &third,
                               ternary_op_kernel ker) const {
  CudaArray second_arr = CudaArray::from_numpy(second);
  return ternaryop(second_arr, third, ker);
}
CudaArray CudaArray::ternaryop(const CudaArray &second, const CudaArray &third,
                               ternary_op_kernel ker) const {
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
  CudaArray out(size, shape);
  size_t n_dims = shape.size();
  cuda_unique_ptr<size_t> d_first_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_second_strides =
      cuda_unique_ptr_from_host(n_dims, second.strides.data());
  cuda_unique_ptr<size_t> d_third_strides =
      cuda_unique_ptr_from_host(n_dims, third.strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());

  ker<<<grid_size, block_size>>>(d_first_strides.get(), d_second_strides.get(),
                                 d_third_strides.get(), d_shape.get(),
                                 shape.size(), ptr.get(), second.ptr.get(),
                                 third.ptr.get(), out.ptr.get());
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}

float CudaArray::getitem(shape_t index) const {
  PG_CHECK_ARG(index.size() == shape.size(),
               "index size must be equal to shape size, got ", index.size(),
               " and ", shape.size());
  // Calculate the offset for the multi-dimensional index
  size_t offset = 0;
  for (size_t i = 0; i < index.size(); i++) {
    PG_CHECK_ARG(index[i] < shape[i] && index[i] >= 0,
                 "index out of bounds, got ", index[i], " for shape ",
                 vec_to_string(shape));
    offset += index[i] * strides[i] / ELEM_SIZE; // since strides are in bytes,
    // we need to divide by ELEM_SIZE to get the correct offset
  }
  // Copy the requested element from device to host
  float value;
  CHECK_CUDA(cudaMemcpy(&value, ptr.get() + offset, ELEM_SIZE,
                        cudaMemcpyDeviceToHost));
  return value;
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
    CudaArray out(new_size, {(size_t)new_size});
    vector_dot_product_accum<<<new_size, MAX_THREADS_PER_BLOCK,
                               MAX_THREADS_PER_BLOCK * ELEM_SIZE>>>(
        a.ptr.get(), b.ptr.get(), out.ptr.get(), a.shape.at(0));
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
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
    CudaArray out(new_size, new_shape);
    cuda_unique_ptr<size_t> lhs_shape =
        cuda_unique_ptr_from_host(a.ndim(), a.shape.data());
    cuda_unique_ptr<size_t> rhs_shape =
        cuda_unique_ptr_from_host(b.ndim(), b.shape.data());
    batched_matmul_kernel<<<gridSize, block_size>>>(
        a.ptr.get(), b.ptr.get(), out.ptr.get(), lhs_shape.get(),
        rhs_shape.get(), a.ndim());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
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
  CudaArray out(total_idxs, new_shape);
  vector_outer_product_kernel<<<grid_size, DEFAULT_BLOCK_SIZE>>>(
      ptr.get(), other.ptr.get(), out.ptr.get(), size, other.size);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}

CudaArray CudaArray::reduce(reduction_kernel ker, axis_t axis,
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
  CudaArray out(new_size, new_shape);
  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
  ker<<<grid_size, block_size>>>(ptr.get(), out.ptr.get(), d_strides.get(),
                                 d_shape.get(), n_dims, axis);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  if (keepdims) {
    return out;
  }
  return out.squeeze(axis);
}

CudaArray CudaArray::reduce(reduction_kernel ker, axes_t axes,
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

CudaArray CudaArray::reduce(reduction_kernel ker, bool keepdims) const {
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
  return reduce(max_kernel, keepdims);
}

CudaArray CudaArray::max(axes_t axes, bool keepdims) const {
  return reduce(max_kernel, axes, keepdims);
}

CudaArray CudaArray::max(axis_t axis, bool keepdims) const {
  return reduce(max_kernel, axis, keepdims);
}

CudaArray CudaArray::sum(bool keepdims) const {
  return reduce(sum_kernel, keepdims);
}

CudaArray CudaArray::sum(axes_t axes, bool keepdims) const {
  return reduce(sum_kernel, axes, keepdims);
}

CudaArray CudaArray::sum(axis_t axis, bool keepdims) const {
  return reduce(sum_kernel, axis, keepdims);
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
  size_t new_stride =
      (axis < strides.size()) ? strides[std::max(0, (int)axis - 1)] : ELEM_SIZE;
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

  CudaArray out(total_new, new_shape);

  CHECK_CUDA(cudaMemcpy(out.ptr.get(), ptr.get(), total_new * ELEM_SIZE,
                        cudaMemcpyDeviceToDevice));

  return out;
}

CudaArray CudaArray::from_numpy(py::array_t<float> np_array) {
  py::buffer_info buffer_info = np_array.request();
  std::vector<py::ssize_t> py_strides = buffer_info.strides;
  shape_t strides(py_strides.begin(), py_strides.end());
  auto size = buffer_info.size;
  auto *ptr = static_cast<float *>(buffer_info.ptr);
  std::vector<py::ssize_t> py_shape = buffer_info.shape;
  shape_t shape(py_shape.begin(), py_shape.end());
  CudaArray arr(size, shape, strides);
  CHECK_CUDA(
      cudaMemcpy(arr.ptr.get(), ptr, size * ELEM_SIZE, cudaMemcpyHostToDevice));
  return arr;
}

py::array_t<float> CudaArray::to_numpy() const {
  py::array_t<float> result(shape, strides);
  CHECK_CUDA(cudaMemcpy(result.mutable_data(), ptr.get(), size * ELEM_SIZE,
                        cudaMemcpyDeviceToHost));
  return result;
}

std::string CudaArray::to_string() const {
  float *host = new float[size];
  CHECK_CUDA(
      cudaMemcpy(host, ptr.get(), size * ELEM_SIZE, cudaMemcpyDeviceToHost));

  std::stringstream ss;
  ss << "CudaArray(" << size << ") [";
  for (size_t i = 0; i < size; i++) {
    ss << host[i] << " ";
  }
  ss << "]";

  delete[] host;
  return ss.str();
}

CudaArray CudaArray::fill(shape_t shape, float value) {
  CudaArray out(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()),
      shape);
  fill_kernel<<<ceil(out.size / (float)DEFAULT_BLOCK_SIZE),
                DEFAULT_BLOCK_SIZE>>>(out.ptr.get(), out.size, value);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
};
CudaArray::~CudaArray() {}

CudaArray::CudaArray(const CudaArray &other)
    : size(other.size), shape(other.shape), strides(other.strides),
      ptr(other.ptr) {}

CudaArray &CudaArray::operator=(const CudaArray &other) {
  if (this != &other) {
    size = other.size;
    shape = other.shape;
    strides = other.strides;
    ptr = other.ptr;
  }
  return *this;
}

CudaArray::CudaArray(CudaArray &&other)
    : size(other.size), shape(std::move(other.shape)),
      strides(std::move(other.strides)), ptr(std::move(other.ptr)) {}

CudaArray &CudaArray::operator=(CudaArray &&other) {
  if (this != &other) {
    size = other.size;
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    ptr = std::move(other.ptr);
  }
  return *this;
}

CudaArray CudaArray::clone() const {
  CudaArray out(size, shape, strides);
  CHECK_CUDA(cudaMemcpy(out.ptr.get(), ptr.get(), size * ELEM_SIZE,
                        cudaMemcpyDeviceToDevice));
  return out;
}

CudaArray CudaArray::elwiseop(element_wise_op_kernel ker) const {
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(size / (float)DEFAULT_BLOCK_SIZE));
  size_t n_dims = shape.size();
  cuda_unique_ptr<size_t> d_strides =
      cuda_unique_ptr_from_host(n_dims, strides.data());
  cuda_unique_ptr<size_t> d_shape =
      cuda_unique_ptr_from_host(n_dims, shape.data());
  CudaArray out(size, shape);
  ker<<<grid_size, block_size>>>(d_strides.get(), d_shape.get(), n_dims,
                                 this->ptr.get(), out.ptr.get());
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}

CudaArray CudaArray::as_contiguous() const { return elwiseop(copy_kernel); }

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

  CudaArray out(size, new_shape, new_strides, ptr);
  return out;
}