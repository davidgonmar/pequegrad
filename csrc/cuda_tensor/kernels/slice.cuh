#include "cuda_tensor.cuh"

template <typename T>
__global__ void _slice_and_assign_with_array_kernel(T* non_sliced,  T *sliced, const size_t* src_shape, const size_t *out_shape, const size_t *src_strides, const int src_shape_len, const Device_Slice *slices, unsigned long slices_size, bool is_assign) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    int leftover = out_idx;
    int src_idx = 0;
    for (int i = slices_size - 1; i >= 0; i--) {
        Device_Slice slice = slices[i];
        int curr_out_dim = leftover % out_shape[i];
        leftover /= out_shape[i];
        if (slice.type == Device_SliceType::SliceFromSSS) {
            // start, stop, step
            int start = slice.start;
            int stop = slice.stop;
            int step = slice.step;
            int src_dim = start + curr_out_dim * step;
            // now calculate 'advancement' in the src array given we want to access its src_dim dimension
            int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
            src_idx += src_advancement;

        } else if (slice.type == Device_SliceType::SliceFromSingleIdx) {
            int src_dim = slice.index;
            // now calculate 'advancement' in the src array given we want to access its src_dim dimension
            int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
            src_idx += src_advancement;
        } else if (slice.type == Device_SliceType::SliceFromIdxArray) {
            int src_dim = slice.indices[curr_out_dim];
            // now calculate 'advancement' in the src array given we want to access its src_dim dimension
            int stride_offset = (src_strides[i] / sizeof(T) * src_dim);
            src_idx += stride_offset;
        } else if (slice.type == Device_SliceType::SliceKeepDim) {
            // this means dimension is kept entirely, so:
            src_idx += src_strides[i] / sizeof(T) * curr_out_dim;
        }
    }
    int max_idx_src = get_max_idx(src_shape, src_shape_len);
    int max_idx_out = get_max_idx(out_shape, slices_size);
    if (out_idx >= max_idx_out || src_idx >= max_idx_src) {
        return;
    }

    if (!is_assign) {
        sliced[out_idx] = non_sliced[src_idx];
    } else {
        non_sliced[src_idx] = sliced[out_idx];
    }

}