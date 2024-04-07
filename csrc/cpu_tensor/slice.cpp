#include "copy.hpp"
#include "cpu_tensor.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include <memory>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
#include <vector>

int get_max_idx(const shape_t &shape, int ndim) {
  int max_idx = 1;
  for (int i = 0; i < ndim; i++) {
    max_idx *= shape[i];
  }
  return max_idx;
}

template <typename T>
void _slice_and_assign_with_array_kernel(T *non_sliced, T *sliced,
                                         const std::vector<size_t> out_shape,
                                         const std::vector<size_t> src_shape,
                                         const std::vector<size_t> src_strides,
                                         const std::vector<size_t> out_strides,
                                         const slice_t slices, bool is_assign) {
  int slices_size = slices.size();
  int src_shape_len = src_shape.size();
  int max_idx_src = get_max_idx(src_shape, src_shape_len);
  int max_idx_out = get_max_idx(out_shape, slices_size);

  for (int out_idx = 0; out_idx < max_idx_out; out_idx++) {
    int leftover = out_idx;
    int src_idx = 0;
    for (int i = slices_size - 1; i >= 0; i--) {
      slice_item_t _slice = slices[i];
      int curr_out_dim = leftover % out_shape[i];
      leftover /= out_shape[i];
      if (std::holds_alternative<SliceFromSSS>(_slice)) {
        SliceFromSSS slice = std::get<SliceFromSSS>(_slice);
        // start, stop, step
        int start = slice.start;
        int stop = slice.stop;
        int step = slice.step;
        int src_dim = start + curr_out_dim * step;
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
        src_idx += src_advancement;

      } else if (std::holds_alternative<SliceFromSingleIdx>(_slice)) {
        SliceFromSingleIdx slice = std::get<SliceFromSingleIdx>(_slice);
        int src_dim = slice.index;
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
        src_idx += src_advancement;
      } else if (std::holds_alternative<SliceFromIdxArray>(_slice)) {
        SliceFromIdxArray slice = std::get<SliceFromIdxArray>(_slice);
        int src_dim = slice.indices[curr_out_dim];
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int stride_offset = (src_strides[i] / sizeof(T) * src_dim);
        src_idx += stride_offset;
      } else if (std::holds_alternative<SliceKeepDim>(_slice)) {
        // this means dimension is kept entirely, so:
        src_idx += src_strides[i] / sizeof(T) * curr_out_dim;
      }
    }
    int max_idx_src = get_max_idx(src_shape, src_shape_len);
    int max_idx_out = get_max_idx(out_shape, slices_size);
    if (out_idx >= max_idx_out || src_idx >= max_idx_src) {
      continue;
    }

    if (!is_assign) {
      sliced[out_idx] = non_sliced[src_idx];
    } else {
      non_sliced[src_idx] = sliced[out_idx];
    }
  }
}
CpuTensor _slice_with_array(const CpuTensor &ten, const slice_t &_slices) {
  slice_t slices = _slices;

  shape_t new_shape;
  for (int i = 0; i < slices.size(); i++) {
    slice_item_t item = slices[i];
    if (std::holds_alternative<SliceFromSSS>(item)) {
      auto _item = std::get<SliceFromSSS>(item);
      int start = _item.start;
      int stop = _item.stop;
      int step = _item.step;
      new_shape.push_back((stop - start + step - 1) / step);
    } else if (std::holds_alternative<SliceFromSingleIdx>(item)) {
      new_shape.push_back(1);
    } else if (std::holds_alternative<SliceFromIdxArray>(item)) {
      auto _item = std::get<SliceFromIdxArray>(item);
      new_shape.push_back(_item.indices.size());
    }
  }

  // also pad device_slices KeepDim
  // 'pad' the shape with same if the slices are less than the original shape
  if (slices.size() < ten.ndim()) {
    for (int i = slices.size(); i < ten.ndim(); i++) {
      new_shape.push_back(ten.shape[i]);
      slices.push_back(SliceKeepDim());
    }
  }

  int total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>());

  CpuTensor out(new_shape, ten.dtype);
  switch (ten.dtype) {
  case DType::Float32:
    _slice_and_assign_with_array_kernel<float>(
        (float *)ten.get_base_ptr(), (float *)out.get_base_ptr(), ten.shape,
        new_shape, ten.strides, out.strides, slices, false);
  }

  // now we need to squeeze the array to remove the dimensions that are 1, where
  // a single index was used (like [:, 1])
  axes_t squeeze_dims;
  for (int i = 0; i < new_shape.size(); i++) {
    if (std::holds_alternative<SliceFromSingleIdx>(slices[i])) {
      squeeze_dims.push_back(i);
    }
  }
  return out.squeeze(squeeze_dims);
}

CpuTensor CpuTensor::slice(const slice_t &slices) const {
  shape_t new_shape;
  shape_t new_strides;
  int _offset = 0;
  bool slice_with_array = false;

  for (int i = 0; i < slices.size(); i++) {
    slice_item_t item = slices[i];
    if (std::holds_alternative<SliceFromSSS>(item)) {
      // at the moment, only positive slices are supported
      auto _item = std::get<SliceFromSSS>(item);
      int start = _item.start;
      int stop = _item.stop;
      int step = _item.step;
      PG_CHECK_ARG(start < shape[i] && stop <= shape[i],
                   "Slice out of bounds, start: " + std::to_string(start) +
                       ", end: " + std::to_string(stop) +
                       ", shape: " + std::to_string(shape[i]));
      _offset += start * strides[i];
      new_shape.push_back((stop - start + step - 1) / step);
      new_strides.push_back(strides[i] * step);
    } else if (std::holds_alternative<SliceFromSingleIdx>(item)) {
      int _item = std::get<SliceFromSingleIdx>(item).index;
      PG_CHECK_ARG(_item >= 0, "Only positive slices are supported, got: " +
                                   std::to_string(_item));
      PG_CHECK_ARG(_item < shape[i], "Slice out of bounds, index: ",
                   std::to_string(_item) +
                       ", shape: " + std::to_string(shape[i]));
      _offset += _item * strides[i];
      // but here, since we are doing something like [:, 1], we dont add
      // anything to the shape we also dont add anything to the strides
    } else if (std::holds_alternative<SliceFromIdxArray>(item)) {
      // this is something like [:, [1, 2, 3]], where we are indexing over the i
      // dimension with an array we cant work with memory views here, so we just
      // run through a kernel to copy the values into a new array
      slice_with_array = true;
      break;
    } else if (std::holds_alternative<SliceKeepDim>(item)) {
      new_shape.push_back(shape[i]);
      new_strides.push_back(strides[i]);
    }
  }
  if (slice_with_array) {
    return _slice_with_array(*this, slices);
  }

  // handle the case where we dont index over ALL dimensions
  if (slices.size() < shape.size()) {
    for (int i = slices.size(); i < shape.size(); i++) {
      new_shape.push_back(shape[i]);
      new_strides.push_back(strides[i]);
    }
  }
  // nbytes is the original one, not computed
  CpuTensor out(nbytes, new_shape, new_strides, ptr, dtype, _offset);

  return out;
}

CpuTensor _assign_with_array(const CpuTensor &ten, const slice_t &_slices,
                             const CpuTensor &_vals) {
  if (_vals.dtype != ten.dtype) {
    throw std::runtime_error(
        "Dtype mismatch on assign, got: " + dtype_to_string(_vals.dtype) +
        ", expected: " + dtype_to_string(ten.dtype));
  }
  slice_t slices = _slices;
  // 'pad' the shape with same if the slices are less than the original shape
  if (slices.size() < ten.ndim()) {
    for (int i = slices.size(); i < ten.ndim(); i++) {
      slices.push_back(SliceKeepDim());
    }
  }

  int total_size = std::accumulate(ten.shape.begin(), ten.shape.end(), 1,
                                   std::multiplies<int>());
  switch (ten.dtype) {
  case DType::Float32:
    _slice_and_assign_with_array_kernel<float>(
        (float *)ten.get_base_ptr(), (float *)_vals.get_base_ptr(), ten.shape,
        _vals.shape, ten.strides, _vals.strides, slices, true);
    break;
  case DType::Int32:
    _slice_and_assign_with_array_kernel<int>(
        (int *)ten.get_base_ptr(), (int *)_vals.get_base_ptr(), ten.shape,
        _vals.shape, ten.strides, _vals.strides, slices, true);
    break;
  case DType::Float64:
    _slice_and_assign_with_array_kernel<double>(
        (double *)ten.get_base_ptr(), (double *)_vals.get_base_ptr(), ten.shape,
        _vals.shape, ten.strides, _vals.strides, slices, true);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }

  return ten;
}

CpuTensor CpuTensor::assign(const slice_t &slices, const CpuTensor &vals) {
  if (vals.dtype != dtype) {
    throw std::runtime_error(
        "Dtype mismatch on assign, got: " + dtype_to_string(vals.dtype) +
        ", expected: " + dtype_to_string(dtype));
  }
  // We just create a sliced view of the original memory, and then copy the vals
  // into it ez pz
  for (int i = 0; i < slices.size(); i++) {
    slice_item_t item = slices[i];
    if (std::holds_alternative<SliceFromIdxArray>(item)) {
      // we cant work with memory views here, so we just run through a kernel to
      // copy the values into a new array
      return _assign_with_array(*this, slices, vals);
    }
  }
  const CpuTensor _sliced = this->slice(slices);

  // broadcast the vals to the shape of the sliced array. We must first remove
  // the dimensions that are 1 on the left for example, we would be trying to bc
  // from [1, 3, 1] to [3, 1] if we didnt remove the 1s
  axes_t squeeze_dims;
  for (int i = 0; i < vals.shape.size(); i++) {
    if (vals.shape[i] != 1) {
      break;
    } else {
      squeeze_dims.push_back(i);
    }
  }
  const CpuTensor _vals =
      vals.squeeze(squeeze_dims).broadcast_to(_sliced.shape);

  copy::dispatch_copy(_sliced.shape, _vals.strides, _sliced.strides,
                      _vals.get_base_ptr(), _sliced.get_base_ptr(), dtype);

  return *this;
}