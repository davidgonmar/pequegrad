#include "cuda_array.cuh"


CudaArray CudaArray::slice(const slice_t &slices) const {

    shape_t new_shape;
    shape_t new_strides = strides;
    int _offset =0;
    for (int i = 0; i < slices.size(); i++) {
        slice_item_t item = slices[i];
        if (std::holds_alternative<std::pair<int, int>>(item)) {
            // at the moment, only positive slices are supported
            auto pair = std::get<std::pair<int, int>>(item);
            int start = pair.first;
            int end = pair.second;
            PG_CHECK_ARG(start >= 0 && end >= 0, "Only positive slices are supported");
            PG_CHECK_ARG(start < shape[i] && end <= shape[i], "Slice out of bounds");
            _offset += start * strides[i];

            new_shape.push_back(end - start);
        } else if (std::holds_alternative<int>(item)) {
            PG_CHECK_ARG(std::get<int>(item) >= 0, "Only positive slices are supported");
            PG_CHECK_ARG(std::get<int>(item) < shape[i], "Slice out of bounds");
            _offset += std::get<int>(item) * strides[i];
            // but here, since we are doing something like [:, 1], we dont add anything to the shape
            // however, we need to delete the stride for this dimension
            new_strides.erase(new_strides.begin() + i);
        }
    }

    // handle the case where we dont index over ALL dimensions
    if (slices.size() < shape.size()) {
        for (int i = slices.size(); i < shape.size(); i++) {
            new_shape.push_back(shape[i]);
        }
    }

    size_t size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    CudaArray out(size, new_shape, new_strides, ptr, dtype, _offset);

    return out;
}