#include "cuda_tensor.cuh"

CudaTensor CudaTensor::reduce(ReduceKernelType ker, axis_t axis,
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
  CudaTensor out(new_size, new_shape, dtype);
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



CudaTensor CudaTensor::reduce(ReduceKernelType ker, axes_t axes,
                            bool keepdims) const {
  CudaTensor out = *this;
  for (size_t axis : axes) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze(axes);
}

CudaTensor CudaTensor::reduce(ReduceKernelType ker, bool keepdims) const {
  CudaTensor out = *this;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    out = out.reduce(ker, axis, true);
  }
  if (keepdims) {
    return out;
  }
  return out.squeeze();
}

CudaTensor CudaTensor::mean(bool keepdims) const {
  int total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  return sum(keepdims).binop(total_elements, BinaryKernelType::DIV);
}

CudaTensor CudaTensor::mean(axes_t axes, bool keepdims) const {
  int total_elements = 1;
  for (axis_t axis : axes) {
    axis = axis < 0 ? shape.size() + axis : axis;
    total_elements *= shape[axis];
  }

  return sum(axes, keepdims).binop(total_elements, BinaryKernelType::DIV);
}

CudaTensor CudaTensor::mean(axis_t axis, bool keepdims) const {
  axis = axis < 0 ? shape.size() + axis : axis;
  int axis_size = shape[axis];
  return sum(axis, keepdims).binop(axis_size, BinaryKernelType::DIV);
}

CudaTensor CudaTensor::max(bool keepdims) const {
  return reduce(ReduceKernelType::MAX, keepdims);
}

CudaTensor CudaTensor::max(axes_t axes, bool keepdims) const {
  return reduce(ReduceKernelType::MAX, axes, keepdims);
}

CudaTensor CudaTensor::max(axis_t axis, bool keepdims) const {
  return reduce(ReduceKernelType::MAX, axis, keepdims);
}

CudaTensor CudaTensor::sum(bool keepdims) const {
  return reduce(ReduceKernelType::SUM, keepdims);
}

CudaTensor CudaTensor::sum(axes_t axes, bool keepdims) const {
  return reduce(ReduceKernelType::SUM, axes, keepdims);
}

CudaTensor CudaTensor::sum(axis_t axis, bool keepdims) const {
  return reduce(ReduceKernelType::SUM, axis, keepdims);
}
