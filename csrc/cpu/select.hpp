#pragma once
#include "ad_primitives.hpp"
#include "shape.hpp"
#include "utils.hpp"
#include <memory>
#include <numeric>

#include <variant>
#include <vector>

namespace pg {
int get_max_idx(const shape_t &shape, int ndim) {
  int max_idx = 1;
  for (int i = 0; i < ndim; i++) {
    max_idx *= shape[i];
  }
  return max_idx;
}

template <typename T>
void _slice_and_assign_with_array_kernel(
    T *non_sliced, T *sliced, std::vector<int *> tensor_indices,
    shape_t src_shape, shape_t out_shape, strides_t src_strides,
    strides_t out_strides, const select_t indices, bool is_assign = false) {
  int slices_size = indices.size();
  int src_shape_len = src_shape.size();
  int max_idx_src = get_max_idx(src_shape, src_shape_len);
  int max_idx_out = get_max_idx(out_shape, slices_size);

  for (int out_idx = 0; out_idx < max_idx_out; out_idx++) {
    int visited_tensors = 0;
    int leftover = out_idx;
    int src_idx = 0;
    for (int i = slices_size - 1; i >= 0; i--) {
      select_item_t _slice = indices[i];
      int curr_out_dim = leftover % out_shape[i];
      leftover /= out_shape[i];
      if (std::holds_alternative<SelectWithSlice>(_slice)) {
        SelectWithSlice slice = std::get<SelectWithSlice>(_slice);
        // start, stop, step
        int start = slice.start;
        int stop = slice.stop;
        int step = slice.step;
        int src_dim = start + curr_out_dim * step;
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
        src_idx += src_advancement;

      } else if (std::holds_alternative<SelectWithSingleIdx>(_slice)) {
        SelectWithSingleIdx slice = std::get<SelectWithSingleIdx>(_slice);
        int src_dim = slice.index;
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int src_advancement = (src_strides[i] / sizeof(T)) * src_dim;
        src_idx += src_advancement;
      } else if (std::holds_alternative<SelectWithTensor>(_slice)) {
        SelectWithTensor slice = std::get<SelectWithTensor>(_slice);
        int src_dim = tensor_indices[visited_tensors][curr_out_dim];
        visited_tensors++;
        // now calculate 'advancement' in the src array given we want to access
        // its src_dim dimension
        int stride_offset = (src_strides[i] / sizeof(T) * src_dim);
        src_idx += stride_offset;
      } else if (std::holds_alternative<SelectKeepDim>(_slice)) {
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
void _select_with_tensor(const Tensor &inp, Tensor &outp, select_t items,
                         std::vector<Tensor> &idxs);
} // namespace pg