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
}

void select(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs,
            select_t _items) {
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
    throw std::runtime_error("Assigning with tensor not supported");
    _select_with_tensor(inp, outputs[0], _items, idxs);
    return;
  }
  outputs[0].init_view(std::make_shared<View>(
      inp.view().shared_ptr(), inp.nbytes(), new_shape, new_strides,
      (size_t)_offset, inp.dtype(), inp.device()));
}

void _assign_with_array(const Tensor &dst, Tensor &src, select_t items,
                        std::vector<Tensor> &idxs) {
  if (src.dtype() != dst.dtype()) {
    throw std::runtime_error(
        "Dtype mismatch on assign, got: " + dtype_to_string(src.dtype()) +
        ", expected: " + dtype_to_string(dst.dtype()));
  }

  int total_size = std::accumulate(dst.shape().begin(), dst.shape().end(), 1,
                                   std::multiplies<int>());
  std::vector<int *> _idxs(idxs.size());
  for (int i = 0; i < idxs.size(); i++) {
    PG_CHECK_ARG(idxs[i].dtype() == DType::Int32,
                 "Index must be of type int32");
    PG_CHECK_ARG(idxs[i].ndim() == 1, "Index must be 1D");
    PG_CHECK_ARG(idxs[i].is_contiguous(), "Index must be contiguous");
    _idxs[i] = idxs[i].get_casted_base_ptr<int>();
  }
  switch (dst.dtype()) {
  case DType::Float32:
    _slice_and_assign_with_array_kernel<float>(
        (float *)dst.get_base_ptr(), (float *)src.get_base_ptr(), _idxs,
        dst.shape(), src.shape(), dst.strides(), src.strides(), items, true);
    break;
  case DType::Int32:
    _slice_and_assign_with_array_kernel<int>(
        (int *)dst.get_base_ptr(), (int *)src.get_base_ptr(), _idxs,
        dst.shape(), src.shape(), dst.strides(), src.strides(), items, true);
    break;
  case DType::Float64:
    _slice_and_assign_with_array_kernel<double>(
        (double *)dst.get_base_ptr(), (double *)src.get_base_ptr(), _idxs,
        dst.shape(), src.shape(), dst.strides(), src.strides(), items, true);
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void AssignAt::dispatch_cpu(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {

  Tensor dst = inputs[0];
  Tensor src = inputs[1];
  std::vector<Tensor> idxs =
      std::vector<Tensor>(inputs.begin() + 2, inputs.end());
  Tensor out = outputs[0];
  if (src.dtype() != dst.dtype()) {
    throw std::runtime_error(
        "Dtype mismatch on assign, got: " + dtype_to_string(src.dtype()) +
        ", expected: " + dtype_to_string(dst.dtype()));
  }
  // We just create a sliced view of the original memory, and then copy the vals
  // into it ez pz
  for (int i = 0; i < _items.size(); i++) {
    select_item_t item = _items[i];
    if (std::holds_alternative<SelectWithTensor>(item)) {
      // we cant work with memory views here, so we just run through a kernel to
      // copy the values into a new array
      // instead of sharing the memory, we just copy the values
      outputs[0].init_view(
          std::make_shared<View>(dst.shape(), dst.dtype(), dst.device()));
      // copy dst into outputs[0]
      copy::dispatch_copy(dst.shape(), dst.strides(), dst.strides(),
                          dst.get_base_ptr(), outputs[0].get_base_ptr(),
                          dst.dtype());
      return _assign_with_array(outputs[0], src, _items, idxs);
    }
  }

  // First, out = dst
  outputs[0].init_view(
      std::make_shared<View>(dst.shape(), dst.dtype(), dst.device()));
  // copy dst into outputs[0]
  copy::dispatch_copy(dst.shape(), dst.strides(), dst.strides(),
                      dst.get_base_ptr(), outputs[0].get_base_ptr(),
                      dst.dtype());
  // Now select from out into tmp
  Tensor tmp = Tensor();
  std::vector<Tensor> _inputs = {outputs[0]};

  _inputs.insert(_inputs.end(), idxs.begin(), idxs.end());
  select(_inputs, std::vector<Tensor>{tmp}, _items);

  // Now copy the values from src to the sliced output
  copy::dispatch_copy(tmp.shape(), src.strides(), tmp.strides(),
                      src.get_base_ptr(), tmp.get_base_ptr(), src.dtype());
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