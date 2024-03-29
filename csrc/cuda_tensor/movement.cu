#include "cuda_tensor.cuh"
#include "shape.hpp"

CudaTensor CudaTensor::reshape(const std::vector<int> &_new_shape) const {
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
    return CudaTensor(nbytes, new_shape, new_strides, ptr, dtype);
  }
  CudaTensor out(new_shape, dtype);
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

CudaTensor CudaTensor::squeeze(axis_t axis) const {
  if (axis < 0) {
    axis = shape.size() + axis;
  }
  PG_CHECK_ARG(axis < shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  PG_CHECK_ARG(shape[axis] == 1,
               "cannot squeeze on a dimension that is not 1, got ", shape[axis],
               " in axis number ", axis, " for shape ", vec_to_string(shape));

  CudaTensor out(*this);
  out.shape.erase(out.shape.begin() + axis);
  out.strides.erase(out.strides.begin() + axis);

  return out;
}

CudaTensor CudaTensor::squeeze(axes_t _axes) const {
  CudaTensor out(*this);
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

CudaTensor CudaTensor::squeeze() const {
  CudaTensor out(*this);
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

CudaTensor CudaTensor::unsqueeze(axes_t axes) const {
  CudaTensor out(*this);
  for (size_t axis : axes) {
    out = out.unsqueeze(axis);
  }
  return out;
}

CudaTensor CudaTensor::unsqueeze(axis_t axis) const {
  if (axis < 0) {
    axis = shape.size() + axis + 1;
  }
  PG_CHECK_ARG(axis <= shape.size(), "axis out of bounds, got ", axis,
               " for shape ", vec_to_string(shape));
  CudaTensor out(*this);
  out.shape.insert(out.shape.begin() + axis, 1);
  size_t new_stride = (axis < strides.size())
                          ? strides[std::max(0, (int)axis - 1)]
                          : dtype_to_size(dtype);
  out.strides.insert(out.strides.begin() + axis, new_stride);
  return out;
}

CudaTensor CudaTensor::broadcast_to(const shape_t shape_to) const {
  shape_t new_strides =
      get_strides_for_broadcasting(this->shape, this->strides, shape_to);
  CudaTensor out(nbytes, shape_to, new_strides, ptr, dtype);
  return out;
}

CudaTensor CudaTensor::permute(shape_t axes) const {
  PG_CHECK_ARG(axes.size() == shape.size(),
               "axes must have same size as shape, got ", axes.size(), " and ",
               shape.size());
  shape_t new_shape(shape.size());
  shape_t new_strides(strides.size());

  for (size_t i = 0; i < axes.size(); ++i) {
    new_shape[i] = shape[axes[i]];
    new_strides[i] = strides[axes[i]];
  }

  CudaTensor out(nbytes, new_shape, new_strides, ptr, dtype);
  return out;
}