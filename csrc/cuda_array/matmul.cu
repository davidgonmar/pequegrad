#include "cuda_array.cuh"


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
