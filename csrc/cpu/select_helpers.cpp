#include "select_helpers.hpp"
#include "ad_primitives.hpp"

namespace pg {
void _select_with_tensor(const Tensor &inp, Tensor &outp, select_t items,
                         std::vector<Tensor> &idxs) {
  shape_t new_shape;
  int curr_tensor_idx = 0;
  for (int i = 0; i < items.size(); i++) {
    select_item_t item = items[i];
    if (std::holds_alternative<SelectWithSlice>(item)) {
      auto _item = std::get<SelectWithSlice>(item);
      int start = _item.start;
      int stop = _item.stop;
      int step = _item.step;
      new_shape.push_back((stop - start + step - 1) / step);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      new_shape.push_back(1);
    } else if (std::holds_alternative<SelectWithTensor>(item)) {
      auto _item = std::get<SelectWithTensor>(item);
      new_shape.push_back(idxs[curr_tensor_idx].shape()[0]);
      curr_tensor_idx++;
    } else if (std::holds_alternative<SelectKeepDim>(item)) {
      new_shape.push_back(inp.shape()[i]);
    }
  }

  if (items.size() < inp.ndim()) {
    for (int i = items.size(); i < inp.ndim(); i++) {
      new_shape.push_back(inp.shape()[i]);
      items.push_back(SelectKeepDim());
    }
  }

  int total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>());

  outp.init_view(std::make_shared<View>(new_shape, inp.dtype(), device::CPU));

  std::vector<int *> tensor_indices;
  for (int i = 0; i < idxs.size(); i++) {
    PG_CHECK_ARG(idxs[i].dtype() == DType::Int32,
                 "Index must be of type int32");
    PG_CHECK_ARG(idxs[i].ndim() == 1, "Index must be 1D");
    PG_CHECK_ARG(idxs[i].is_contiguous(), "Index must be contiguous");
    tensor_indices.push_back(idxs[i].get_casted_base_ptr<int>());
  }
  switch (inp.dtype()) {
  case DType::Float32:
    _slice_and_assign_with_array_kernel<float>(
        inp.get_casted_base_ptr<float>(), outp.get_casted_base_ptr<float>(),
        tensor_indices, inp.shape(), new_shape, inp.strides(), outp.strides(),
        items);
    break;
  case DType::Int32:
    _slice_and_assign_with_array_kernel<int>(
        inp.get_casted_base_ptr<int>(), outp.get_casted_base_ptr<int>(),
        tensor_indices, inp.shape(), new_shape, inp.strides(), outp.strides(),
        items);
    break;
  case DType::Float64:
    _slice_and_assign_with_array_kernel<double>(
        inp.get_casted_base_ptr<double>(), outp.get_casted_base_ptr<double>(),
        tensor_indices, inp.shape(), new_shape, inp.strides(), outp.strides(),
        items);
    break;
  default:
    throw std::runtime_error("Unsupported dtype for select with tensor: " +
                             dtype_to_string(inp.dtype()));
  }

  // now we need to squeeze the array to remove the dimensions that are 1, where
  // a single index was used (like [:, 1])
  axes_t squeeze_dims;
  for (int i = 0; i < new_shape.size(); i++) {
    if (std::holds_alternative<SelectWithSingleIdx>(items[i])) {
      squeeze_dims.push_back(i);
    }
  }
  // return out.squeeze(squeeze_dims);
}

void Select::dispatch_cpu(const std::vector<Tensor> &inputs,
                          std::vector<Tensor> &outputs) {
  shape_t new_shape;
  strides_t new_strides;
  int _offset = 0;
  bool select_with_tensor = false;
  Tensor inp = inputs[0];
  PG_CHECK_ARG(inp.ndim() == _items.size(),
               "Number of slices must match number of dimensions");
  std::vector<Tensor> idxs =
      std::vector<Tensor>(inputs.begin() + 1, inputs.end());
  for (int i = 0; i < _items.size(); i++) {
    select_item_t item = _items[i];
    if (std::holds_alternative<SelectWithSlice>(item)) {
      // at the moment, only positive slices are supported
      auto _item = std::get<SelectWithSlice>(item);
      int start = _item.start;
      int stop = _item.stop;
      int step = _item.step;
      PG_CHECK_ARG(start < inp.shape()[i] && stop <= inp.shape()[i],
                   "Slice out of bounds, start: " + std::to_string(start) +
                       ", end: " + std::to_string(stop) +
                       ", shape: " + std::to_string(inp.shape()[i]));
      _offset += start * inp.strides()[i];
      new_shape.push_back((stop - start + step - 1) / step);
      new_strides.push_back(inp.strides()[i] * step);
    } else if (std::holds_alternative<SelectWithSingleIdx>(item)) {
      long _item = std::get<SelectWithSingleIdx>(item).index;
      PG_CHECK_ARG(_item >= 0, "Only positive slices are supported, got: " +
                                   std::to_string(_item));
      PG_CHECK_ARG(_item < inp.shape()[i], "Slice out of bounds, index: ",
                   std::to_string(_item) +
                       ", shape: " + std::to_string(inp.shape()[i]));
      _offset += _item * inp.strides()[i];
      // but here, since we are doing something like [:, 1], we dont add
      // anything to the shape we also dont add anything to the strides
    } else if (std::holds_alternative<SelectWithTensor>(item)) {
      // this is something like [:, [1, 2, 3]], where we are indexing over the i
      // dimension with an array we cant work with memory views here, so we just
      // run through a kernel to copy the values into a new array
      select_with_tensor = true;
      break;
    } else if (std::holds_alternative<SelectKeepDim>(item)) {
      new_shape.push_back(inp.shape()[i]);
      new_strides.push_back(inp.strides()[i]);
    }
  }
  if (select_with_tensor) {
    _select_with_tensor(inp, outputs[0], _items, idxs);
    return;
  }

  outputs[0].init_view(std::make_shared<View>(
      inp.view().shared_ptr(), inp.nbytes(), new_shape, new_strides,
      (size_t)_offset, inp.dtype(), inp.device()));
}
} // namespace pg