#include "ad_primitives.hpp"
#include "cuda_tensor/cuda_utils.cuh"
#include "select_helpers.cuh"
#include "shape.hpp"
#include "utils.hpp"
#include <memory>
#include <numeric>
#include <variant>
namespace pg {
CudaSelect convert_to_slice(const select_item_t &_item,
                            std::shared_ptr<Tensor> tensor = nullptr) {
  CudaSelect item;
  if (std::holds_alternative<SelectWithSlice>(_item)) {
    item.type = CudaSelectKind::SelectWithSlice;
    const auto &sss = std::get<SelectWithSlice>(_item);
    item.start = sss.start;
    item.stop = sss.stop;
    item.step = sss.step;
  } else if (std::holds_alternative<SelectWithTensor>(_item)) {
    item.type = CudaSelectKind::SelectWithTensor;
    const auto &idxArray = std::get<SelectWithTensor>(_item);
    PG_CHECK_ARG(tensor != nullptr, "Tensor is null");
    PG_CHECK_ARG(tensor->dtype() == DType::Int32,
                 "Index tensor must be of type int32");
    PG_CHECK_ARG(tensor->ndim() == 1, "Index tensor must be 1D");
    PG_CHECK_ARG(tensor->is_contiguous(), "Index tensor must be contiguous");
    auto indices = tensor->get_casted_base_ptr<int>();
    item.indices = indices;
    item.indexSize = tensor->numel();
  } else if (std::holds_alternative<SelectWithSingleIdx>(_item)) {
    item.type = CudaSelectKind::SelectWithSingleIndex;
    const auto &singleIdx = std::get<SelectWithSingleIdx>(_item);
    item.index = singleIdx.index;
  } else if (std::holds_alternative<SelectKeepDim>(_item)) {
    item.type = CudaSelectKind::SelectKeepDim;
  }
  return item;
}

namespace cuda {
void _select_with_tensor(const Tensor &inp, Tensor &outp, select_t items,
                         std::vector<Tensor> &idxs) {

  shape_t new_shape;
  int visited_tensors = 0;
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
      new_shape.push_back(idxs[visited_tensors].numel());
      visited_tensors++;
    }
  }

  CudaSelect *cuda_select_items = new CudaSelect[items.size()];
  visited_tensors = 0;
  for (int i = 0; i < items.size(); i++) {
    std::shared_ptr<Tensor> tensor;
    if (std::holds_alternative<SelectWithTensor>(items[i])) {
      tensor = std::make_shared<Tensor>(idxs[visited_tensors]);
      visited_tensors++;
    }
    cuda_select_items[i] = convert_to_slice(items[i], tensor);
  }
  // now copy to GPU
  auto d_items = cuda_unique_ptr_from_host(items.size(), cuda_select_items);

  int total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                   std::multiplies<int>());

  auto d_shape = cuda_unique_ptr_from_host(inp.ndim(), inp.shape().data());
  outp.init_view(std::make_shared<View>(new_shape, inp.dtype(), device::CUDA));
  auto out_d_shape =
      cuda_unique_ptr_from_host(outp.ndim(), outp.shape().data());
  auto src_strides =
      cuda_unique_ptr_from_host(inp.ndim(), inp.strides().data());
  int block_size = DEFAULT_BLOCK_SIZE;
  int grid_size = ceil(total_size / (float)block_size);
  switch (inp.dtype()) {
  case DType::Float32:
    _slice_and_assign_with_array_kernel<float><<<grid_size, block_size>>>(
        (float *)inp.get_base_ptr(), (float *)outp.get_base_ptr(),
        d_shape.get(), out_d_shape.get(), src_strides.get(), inp.ndim(),
        d_items.get(), items.size(), false);
    break;
  case DType::Int32:
    _slice_and_assign_with_array_kernel<int><<<grid_size, block_size>>>(
        (int *)inp.get_base_ptr(), (int *)outp.get_base_ptr(), d_shape.get(),
        out_d_shape.get(), src_strides.get(), inp.ndim(), d_items.get(),
        items.size(), false);
    break;
  case DType::Float64:
    _slice_and_assign_with_array_kernel<double><<<grid_size, block_size>>>(
        (double *)inp.get_base_ptr(), (double *)outp.get_base_ptr(),
        d_shape.get(), out_d_shape.get(), src_strides.get(), inp.ndim(),
        d_items.get(), items.size(), false);
    break;
  }

  PG_CUDA_KERNEL_END;

  /*// now we need to squeeze the array to remove the dimensions that are 1,
  where
  // a single index was used (like [:, 1])
  axes_t squeeze_dims;
  for (int i = 0; i < new_shape.size(); i++) {
    if (std::holds_alternative<SliceFromSingleIdx>(slices[i])) {
      squeeze_dims.push_back(i);
    }
  }

  // cleanup the device slices
  for (int i = 0; i < slices.size(); i++) {
    if (std::holds_alternative<SliceFromIdxArray>(slices[i])) {
      CHECK_CUDA(cudaFree(device_slices[i].indices));
    }
  }

  delete[] device_slices;

  return out.squeeze(squeeze_dims);*/
}
} // namespace cuda

void Select::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  shape_t new_shape;
  strides_t new_strides;
  int _offset = 0;
  bool slice_with_array = false;
  Tensor inp = inputs[0];
  PG_CHECK_ARG(inp.ndim() == _items.size(),
               "Number of slices must match number of dimensions");
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
      int _item = std::get<SelectWithSingleIdx>(item).index;
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
      slice_with_array = true;
      break;
    } else if (std::holds_alternative<SelectKeepDim>(item)) {
      new_shape.push_back(inp.shape()[i]);
      new_strides.push_back(inp.strides()[i]);
    }
  }
  if (slice_with_array) {
    std::cout << "Using slice with array" << std::endl;
    std::vector<Tensor> idxs =
        std::vector<Tensor>(inputs.begin() + 1, inputs.end());
    cuda::_select_with_tensor(inp, outputs[0], _items, idxs);
    return;
  }

  outputs[0].init_view(std::make_shared<View>(
      inp.view().shared_ptr(), inp.nbytes(), new_shape, new_strides,
      (size_t)_offset, inp.dtype(), inp.device()));

  /*// handle the case where we dont index over ALL dimensions
  if (slices.size() < shape.size()) {
    for (int i = slices.size(); i < shape.size(); i++) {
      new_shape.push_back(shape[i]);
      new_strides.push_back(strides[i]);
    }
  }
  // nbytes is the original one, not computed
  CudaTensor out(nbytes, new_shape, new_strides, ptr, dtype, _offset);

  return out;*/
}

} // namespace pg
