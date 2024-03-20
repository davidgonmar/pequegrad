#include "cuda_tensor.cuh"
#include "kernels/slice.cuh"
#include "utils.cuh"

Device_Slice convert_to_slice(const slice_item_t& item) {
    Device_Slice slice;
    if (std::holds_alternative<SliceFromSSS>(item)) {
        slice.type = Device_SliceType::SliceFromSSS;
        const auto& sss = std::get<SliceFromSSS>(item);
        slice.start = sss.start;
        slice.stop = sss.stop;
        slice.step = sss.step;
    } else if (std::holds_alternative<SliceFromIdxArray>(item)) {
        slice.type = Device_SliceType::SliceFromIdxArray;
        const auto& idxArray = std::get<SliceFromIdxArray>(item);
        slice.indexSize = idxArray.indices.size();
        cudaMalloc(&slice.indices, slice.indexSize * sizeof(int));
        cudaMemcpy(slice.indices, idxArray.indices.data(), slice.indexSize * sizeof(int), cudaMemcpyHostToDevice);
    } else if (std::holds_alternative<SliceFromSingleIdx>(item)) {
        slice.type = Device_SliceType::SliceFromSingleIdx;
        const auto& singleIdx = std::get<SliceFromSingleIdx>(item);
        slice.index = singleIdx.index;
    } else if (std::holds_alternative<SliceKeepDim>(item)) {
        slice.type = Device_SliceType::SliceKeepDim;
    }
    return slice;
}


CudaTensor _slice_with_array(const CudaTensor &ten, const slice_t &_slices) {
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

     Device_Slice* device_slices = new Device_Slice[slices.size()];

    for (int i = 0; i < slices.size(); i++) {
        device_slices[i] = convert_to_slice(slices[i]);
    }
    // now copy to GPU
    auto d_slices = cuda_unique_ptr_from_host(slices.size(), device_slices);

    int total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());

    auto d_shape = cuda_unique_ptr_from_host(ten.ndim(), ten.shape.data());
    CudaTensor out(total_size, new_shape, ten.dtype);
    auto out_d_shape = cuda_unique_ptr_from_host(out.ndim(), out.shape.data());
    auto src_strides = cuda_unique_ptr_from_host(ten.ndim(), ten.strides.data());
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = ceil(total_size / (float)block_size);
    switch (ten.dtype) {
        case DType::Float32:
            _slice_and_assign_with_array_kernel<float><<<grid_size, block_size>>>((float*)ten.get_base_ptr(), (float*)out.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), false);
            break;
        case DType::Int32:
            _slice_and_assign_with_array_kernel<int><<<grid_size, block_size>>>((int*)ten.get_base_ptr(), (int*)out.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), false);
            break;
        case DType::Float64:
            _slice_and_assign_with_array_kernel<double><<<grid_size, block_size>>>((double*)ten.get_base_ptr(), (double*)out.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), false);
            break;
    }

    PG_CUDA_KERNEL_END;

    // now we need to squeeze the array to remove the dimensions that are 1, where a single index was used (like [:, 1])
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

    return out.squeeze(squeeze_dims);
}

CudaTensor CudaTensor::slice(const slice_t &slices) const {
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
            PG_CHECK_ARG(start < shape[i] && stop <= shape[i], "Slice out of bounds, start: " + std::to_string(start) + ", end: " + std::to_string(stop) + ", shape: " + std::to_string(shape[i]));
            _offset += start * strides[i];
            new_shape.push_back((stop - start + step - 1) / step);
            new_strides.push_back(strides[i] * step);
        } else if (std::holds_alternative<SliceFromSingleIdx>(item)) {
            int _item = std::get<SliceFromSingleIdx>(item).index;
            PG_CHECK_ARG(_item >= 0, "Only positive slices are supported, got: " + std::to_string(_item));
            PG_CHECK_ARG(_item < shape[i], "Slice out of bounds, index: ", std::to_string(_item) + ", shape: " + std::to_string(shape[i]));
            _offset += _item * strides[i];
            // but here, since we are doing something like [:, 1], we dont add anything to the shape
            // we also dont add anything to the strides
        } else if (std::holds_alternative<SliceFromIdxArray>(item)) {
            // this is something like [:, [1, 2, 3]], where we are indexing over the i dimension with an array
            // we cant work with memory views here, so we just run through a kernel to copy the values into a new array
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

    size_t size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    CudaTensor out(size, new_shape, new_strides, ptr, dtype, _offset);

    return out;
}



CudaTensor _assign_with_array(const CudaTensor &ten, const slice_t &_slices, const CudaTensor &_vals) {
    CudaTensor vals = _vals.astype(ten.dtype);
    slice_t slices = _slices;
    // 'pad' the shape with same if the slices are less than the original shape
    if (slices.size() < ten.ndim()) {
        for (int i = slices.size(); i < ten.ndim(); i++) {
            slices.push_back(SliceKeepDim());
        }
    }
    Device_Slice* device_slices = new Device_Slice[slices.size()];

    for (int i = 0; i < slices.size(); i++) {
        device_slices[i] = convert_to_slice(slices[i]);
    }
    // now copy to GPU
    auto d_slices = cuda_unique_ptr_from_host(slices.size(), device_slices);

    int total_size = std::accumulate(ten.shape.begin(), ten.shape.end(), 1, std::multiplies<int>());

    auto d_shape = cuda_unique_ptr_from_host(ten.ndim(), ten.shape.data());
    auto out_d_shape = cuda_unique_ptr_from_host(vals.ndim(), vals.shape.data());
    auto src_strides = cuda_unique_ptr_from_host(ten.ndim(), ten.strides.data());
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = ceil(total_size / (float)block_size);
    switch (ten.dtype) {
        case DType::Float32:
            _slice_and_assign_with_array_kernel<float><<<grid_size, block_size>>>((float*)ten.get_base_ptr(), (float*)vals.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), true);
            break;
        case DType::Int32:
            _slice_and_assign_with_array_kernel<int><<<grid_size, block_size>>>((int*)ten.get_base_ptr(), (int*)vals.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), true);
            break;
        case DType::Float64:
            _slice_and_assign_with_array_kernel<double><<<grid_size, block_size>>>((double*)ten.get_base_ptr(), (double*)vals.get_base_ptr(), d_shape.get(), out_d_shape.get(), src_strides.get(), ten.ndim(), d_slices.get(), slices.size(), true);
            break;
    }

    PG_CUDA_KERNEL_END;

    // cleanup the device slices
    for (int i = 0; i < slices.size(); i++) {
        if (std::holds_alternative<SliceFromIdxArray>(slices[i])) {
            CHECK_CUDA(cudaFree(device_slices[i].indices));
        }
    }

    delete[] device_slices;

    return ten;
}

      
CudaTensor CudaTensor::assign(const slice_t &slices, const CudaTensor &vals) {
    // We just create a sliced view of the original memory, and then copy the vals into it
    // ez pz
    for (int i = 0; i < slices.size(); i++) {
        slice_item_t item = slices[i];
        if (std::holds_alternative<SliceFromIdxArray>(item)) {
            // we cant work with memory views here, so we just run through a kernel to copy the values into a new array
            return _assign_with_array(*this, slices, vals);
        }
    }
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