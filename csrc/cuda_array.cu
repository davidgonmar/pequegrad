#include "binary_ops_kernels.cuh"
#include "cuda_array.cuh"
#include "matmul_kernels.cuh"
#include "reduce_ops_kernels.cuh"
#include "ternary_ops_kernels.cuh"
#include "unary_ops_kernels.cuh"
#include "utils.cuh"
#include <cmath>

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
  if (from_ndim > to_ndim) {
    throw std::runtime_error("got incompatible shapes, to_ndim < from_ndim: " +
                             std::to_string(to_ndim) + " < " +
                             std::to_string(from_ndim));
  }

  int new_size = 1;
  shape_t new_strides(to_ndim, 0);
  // reverse test if the dim is 1 or they are equal
  for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
    py::ssize_t dim_to = shape_to[i];
    py::ssize_t dim_from =
        (j >= 0) ? shape_from[j]
                 : -1; // -1 means we 'ran' out of dimensions for j
    if (dim_to != dim_from && dim_from != 1 && dim_from != -1) {
      // we can only 'broadcast' a dimension if dim_from == 1 or we ran out of
      // dimensions.
      throw std::runtime_error("got incompatible shapes, dim_to != dim_from: " +
                               std::to_string(dim_to) +
                               " != " + std::to_string(dim_from));
    }
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
  size_t *d_strides, *d_other_strides, *d_shape;
  CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_other_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMemcpy(d_strides, strides.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_other_strides, other.strides.data(),
                        n_dims * sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  ker<<<grid_size, block_size>>>(d_strides, d_other_strides, d_shape, n_dims,
                                 ptr.get(), other.ptr.get(), out.ptr.get());
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
  size_t *d_first_strides, *d_second_strides, *d_third_strides, *d_shape;
  CHECK_CUDA(cudaMalloc(&d_first_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_second_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_third_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(size_t)));

  CHECK_CUDA(cudaMemcpy(d_first_strides, strides.data(),
                        n_dims * sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_second_strides, second.strides.data(),
                        n_dims * sizeof(size_t), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_third_strides, third.strides.data(),
                        n_dims * sizeof(size_t), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  ker<<<grid_size, block_size>>>(
      d_first_strides, d_second_strides, d_third_strides, d_shape, shape.size(),
      ptr.get(), second.ptr.get(), third.ptr.get(), out.ptr.get());
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  return out;
}

float CudaArray::getitem(shape_t index) const {
  if (index.size() != shape.size()) {
    throw std::runtime_error("Index dimension mismatch");
  }
  // Calculate the offset for the multi-dimensional index
  size_t offset = 0;
  for (size_t i = 0; i < index.size(); i++) {
    if (index[i] < 0 || index[i] >= shape[i]) {
      throw std::runtime_error("Index out of bounds");
    }
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
  CudaArray a = this->is_contiguous() ? *this : this->as_contiguous();
  CudaArray b = other.is_contiguous() ? other : other.as_contiguous();
  dim3 block_size = dim3(DEFAULT_BLOCK_SIZE);
  shape_t new_shape;
  size_t size1, midsize, size2;
  if (a.ndim() == 2 && b.ndim() == 2) {
    size1 = a.shape.at(0);
    midsize = a.shape.at(1);
    size2 = b.shape.at(1);
    new_shape = {size1, size2};
  } else if (a.ndim() == 1 && b.ndim() == 1) {
    if (a.shape != b.shape)
      throw std::invalid_argument("shapes must be equal in vector dot prod");
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
      return out.sum(false);
    }
    return out.squeeze();

  } else if (a.ndim() == 2 && b.ndim() == 1) {
    size1 = a.shape.at(0);
    midsize = a.shape.at(1);
    size2 = 1;
    new_shape = {size1};
  } else if (a.ndim() == 1 && b.ndim() == 2) {
    size1 = 1;
    midsize = b.shape.at(0);
    size2 = b.shape.at(1);
    new_shape = {size2};
  } else if (a.ndim() == 3 && b.ndim() == 3) {
    // assert that a batchsize = b batchsize
    size1 = a.shape.at(1);
    midsize = a.shape.at(2);
    size2 = b.shape.at(2);
    new_shape = {a.shape.at(0), size1, size2};
  } else {
    std::string error_message =
        "Invalid shapes for matmul, only 1D/2D combinations, 2Dx2D and 1Dx1D "
        "tensors supported";
    throw std::runtime_error(error_message);
  }

  int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                 std::multiplies<int>());
  dim3 gridSize(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
  CudaArray out(new_size, new_shape);
  size_t *lhs_shape, *rhs_shape;
  CHECK_CUDA(cudaMalloc(&lhs_shape, a.ndim() * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&rhs_shape, b.ndim() * sizeof(size_t)));

  CHECK_CUDA(cudaMemcpy(lhs_shape, a.shape.data(), a.ndim() * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(rhs_shape, b.shape.data(), b.ndim() * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  matmul_kernel<<<gridSize, block_size>>>(a.ptr.get(), b.ptr.get(),
                                          out.ptr.get(), lhs_shape, rhs_shape,
                                          a.ndim(), b.ndim());

  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());

  return out;
}

CudaArray CudaArray::sum(bool keepdims) const {
  // check if the array is already reduced
  if (std::all_of(shape.begin(), shape.end(),
                  [](size_t i) { return i == 1; })) {
    return *this;
  }
  // simply sum along all axes
  CudaArray result = *this;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    result = result.sum(axis, true);
  }
  if (keepdims) {
    return result;
  }
  return result.squeeze();
}

CudaArray CudaArray::sum(shape_t axes, bool keepdims) const {
  // simply sum along all axes requested
  CudaArray result = *this;
  for (size_t axis : axes) {
    result = result.sum(axis, true);
  }
  if (keepdims) {
    return result;
  }
  return result.squeeze();
}

CudaArray CudaArray::sum(size_t axis, bool keepdims) const {
  if (!is_contiguous()) {
    return as_contiguous().sum(axis, keepdims);
  }
  shape_t new_shape = shape;
  new_shape[axis] = 1;
  size_t new_size = size / shape[axis];
  size_t n_dims = shape.size();
  CudaArray out(new_size, new_shape);
  size_t *d_strides, *d_shape;
  CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMemcpy(d_strides, strides.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
  sum_kernel<<<grid_size, block_size>>>(ptr.get(), out.ptr.get(), d_strides,
                                        d_shape, n_dims, axis);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());

  if (keepdims) {
    return out;
  }
  return out.squeeze();
}

CudaArray CudaArray::max(bool keepdims) const {
  // check if the array is already reduced
  if (std::all_of(shape.begin(), shape.end(),
                  [](size_t i) { return i == 1; })) {
    return *this;
  }
  CudaArray result = *this;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    result = result.max(axis, true);
  }
  if (keepdims) {
    return result;
  }
  return result.squeeze();
}

CudaArray CudaArray::max(shape_t axes, bool keepdims) const {
  CudaArray result = *this;
  for (size_t axis : axes) {
    result = result.max(axis, true);
  }
  if (keepdims) {
    return result;
  }
  return result.squeeze();
}

CudaArray CudaArray::max(size_t axis, bool keepdims) const {
  if (!is_contiguous()) {
    return as_contiguous().max(axis, keepdims);
  }
  shape_t new_shape = shape;
  new_shape[axis] = 1;
  size_t new_size = size / shape[axis];
  size_t n_dims = shape.size();
  CudaArray out(new_size, new_shape);
  size_t *d_strides, *d_shape;
  CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMemcpy(d_strides, strides.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  dim3 block_size(DEFAULT_BLOCK_SIZE);
  dim3 grid_size(ceil(new_size / (float)DEFAULT_BLOCK_SIZE));
  max_kernel<<<grid_size, block_size>>>(ptr.get(), out.ptr.get(), d_strides,
                                        d_shape, n_dims, axis);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  if (keepdims) {
    return out;
  }
  return out.squeeze();
}

CudaArray CudaArray::squeeze(size_t axis) const {
  if (shape.size() <= axis)
    throw std::invalid_argument("requested axis is out of bounds");
  if (shape[axis] != 1)
    throw std::invalid_argument(
        "cannot squeeze on a dimension that is not 1, got " +
        std::to_string(shape[axis]));

  CudaArray out(*this);

  out.shape.erase(out.shape.begin() + axis);
  out.strides.erase(out.strides.begin() + axis);

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

CudaArray CudaArray::unsqueeze(shape_t axes) const {
  CudaArray out(*this);
  for (size_t axis : axes) {
    out = out.unsqueeze(axis);
  }
  return out;
}

CudaArray CudaArray::unsqueeze(size_t axis) const {
  if (axis > shape.size()) {
    throw std::invalid_argument("Axis out of bounds for unsqueeze operation");
  }
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
      if (neg_pos != -1) {
        throw std::runtime_error("Can only specify one unknown dimension");
      }
      neg_pos = i;
    }
    new_shape[i] = _new_shape[i];
    total_new *= new_shape[i] == -1 ? 1 : new_shape[i];
  }

  size_t total_old =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (neg_pos != -1) {
    new_shape[neg_pos] = total_old / total_new;
    if (total_old % total_new != 0) {
      throw std::runtime_error("New shape is not compatible with old shape: " +
                               vec_to_string(shape) + " not compatible with " +
                               vec_to_string(_new_shape));
    }
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
  size_t *d_strides, *d_shape;

  CHECK_CUDA(cudaMalloc(&d_strides, n_dims * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_shape, n_dims * sizeof(size_t)));

  CHECK_CUDA(cudaMemcpy(d_strides, strides.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  CudaArray out(size, shape);
  ker<<<grid_size, block_size>>>(d_strides, d_shape, n_dims, this->ptr.get(),
                                 out.ptr.get());
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());

  return out;
}

CudaArray CudaArray::as_contiguous() const { return elwiseop(copy_kernel); }

CudaArray CudaArray::permute(shape_t axes) const {
  // TODO - check that axes is from 0 to shape.size - 1, in any order
  if (axes.size() != shape.size()) {
    throw std::runtime_error("axes must have same size as shape");
  }
  shape_t new_shape(shape.size());
  shape_t new_strides(strides.size());

  for (size_t i = 0; i < axes.size(); ++i) {
    new_shape[i] = shape[axes[i]];
    new_strides[i] = strides[axes[i]];
  }

  CudaArray out(size, new_shape, new_strides, ptr);
  return out;
}