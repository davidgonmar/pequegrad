#include "cuda_tensor.cuh"
#include "utils.cuh"


CudaTensor CudaTensor::slice(const slice_t &slices) const {
    shape_t new_shape;
    shape_t new_strides;
    int _offset = 0;
    for (int i = 0; i < slices.size(); i++) {
        slice_item_t item = slices[i];
        if (std::holds_alternative<std::pair<int, int>>(item)) {
            // at the moment, only positive slices are supported
            auto pair = std::get<std::pair<int, int>>(item);
            int start = pair.first;
            int end = pair.second;
            PG_CHECK_ARG(start >= 0 && end >= 0, "Only positive slices are supported, got: " + std::to_string(start) + ", " + std::to_string(end));
            PG_CHECK_ARG(start < shape[i] && end <= shape[i], "Slice out of bounds, start: " + std::to_string(start) + ", end: " + std::to_string(end) + ", shape: " + std::to_string(shape[i]));
            _offset += start * strides[i];

            new_shape.push_back(end - start);
            new_strides.push_back(strides[i]);
        } else if (std::holds_alternative<int>(item)) {
            PG_CHECK_ARG(std::get<int>(item) >= 0, "Only positive slices are supported, got: " + std::to_string(std::get<int>(item)));
            PG_CHECK_ARG(std::get<int>(item) < shape[i], "Slice out of bounds, index: " + std::to_string(std::get<int>(item)) + ", shape: " + std::to_string(shape[i]));
            _offset += std::get<int>(item) * strides[i];
            // but here, since we are doing something like [:, 1], we dont add anything to the shape
            // we also dont add anything to the strides
        }
    }

    // handle the case where we dont index over ALL dimensions
    if (slices.size() < shape.size()) {
        for (int i = slices.size(); i < shape.size(); i++) {
            new_shape.push_back(shape[i]);
        }
    }

    size_t size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    CudaTensor out(size, new_shape, new_strides, ptr, dtype, _offset);

    return out;
}


CudaTensor CudaTensor::assign(const slice_t &slices, const CudaTensor &vals) {
    // We just create a sliced view of the original memory, and then copy the vals into it
    // ez pz
    const CudaTensor _sliced = this->slice(slices);

    // broadcast the vals to the shape of the sliced array. We must first remove the dimensions that are 1 on the left
    // for example, we would be trying to bc from [1, 3, 1] to [3, 1] if we didnt remove the 1s
    axes_t squeeze_dims;
    for (int i = 0; i < vals.shape.size(); i++) {
        if (vals.shape[i] != 1) {
            break;
        } else  {
            squeeze_dims.push_back(i);
        }
    }
    const CudaTensor _vals = vals.squeeze(squeeze_dims).broadcast_to(_sliced.shape).astype(_sliced.dtype);
    dim3 block_size(DEFAULT_BLOCK_SIZE);
    dim3 grid_size(ceil(_vals.size / (float)DEFAULT_BLOCK_SIZE));
    auto &sliced_strides = cuda_unique_ptr_from_host(_sliced.shape.size(), _sliced.strides.data());
    auto &sliced_shape = cuda_unique_ptr_from_host(_sliced.shape.size(), _sliced.shape.data());
    auto &vals_shape =
      cuda_unique_ptr_from_host(_vals.shape.size(), _vals.shape.data());
    auto &vals_strides =
      cuda_unique_ptr_from_host(_vals.strides.size(), _vals.strides.data());

    // copy vals into _sliced (which is a memory view of original array)
    launch_copy_with_out_strides_kernel(
      dtype, grid_size, block_size, vals_strides.get(), vals_shape.get(),
      sliced_strides.get(), sliced_shape.get(), _vals.ndim(), _sliced.ndim(), _vals.get_base_ptr(),
      _sliced.get_base_ptr());


    PG_CUDA_KERNEL_END;


    return *this;

}