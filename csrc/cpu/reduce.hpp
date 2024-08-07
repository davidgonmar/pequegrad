#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include <limits>
#include <vector>

namespace pg {
namespace cpu {

template <typename Op, typename T>
void reduce_base_fn(const T *in, T *out, const strides_t &in_strides,
                    const shape_t &in_shape, const size_t red_axis) {
  Op op;
  size_t red_axis_stride = in_strides[red_axis];
  size_t n_dims = in_shape.size();

  size_t red_axis_size = in_shape[red_axis];
  size_t red_axis_stride_size = in_strides[red_axis] / sizeof(T);

  size_t total_out_elements = 1;
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (i != red_axis) {
      total_out_elements *= in_shape[i];
    }
  }
#pragma omp parallel for collapse(1)
  for (int i = 0; i < total_out_elements; i++) {
    int idx = i;

    int red_elements = in_shape[red_axis];

    T accum = op.initial_value();

    for (int i = 0; i < red_elements; i++) {
      int reduced_idx = idx;
      int in_idx = 0;
      for (int j = n_dims - 1; j >= 0; j--) {
        if (j == red_axis) {
          in_idx +=
              i * in_strides[j] / sizeof(T); // simply advance by 'i * stride'
        } else { // do the general algorithm to go from idx -> actual
                 // displacement
          int current_dim_idx = reduced_idx % in_shape[j];
          in_idx += current_dim_idx * in_strides[j] / sizeof(T);
          reduced_idx /= in_shape[j];
        }
      }
      T el = in[in_idx];
      accum = op.apply(accum, el);
    }

    out[idx] = op.after_reduce(accum, red_elements);
  }

  return;
}

template <typename T> struct SumOp {
  T apply(T a, T b) { return a + b; }
  T initial_value() { return (T)0; }
  T after_reduce(T a, int elems) { return a; }
};

template <typename T> struct MaxOp {
  T apply(T a, T b) { return std::max(a, b); }
  // depending on the type, we might want to use the smallest possible value
  T initial_value() {
    if (std::is_same<T, float>::value) {
      return std::numeric_limits<float>::lowest();
    } else if (std::is_same<T, double>::value) {
      return std::numeric_limits<double>::lowest();
    } else if (std::is_same<T, int>::value) {
      return std::numeric_limits<int>::lowest();
    } else {
      return 0;
    }
  }
  T after_reduce(T a, int elems) { return a; }
};

template <typename T> struct MeanOp {
  T apply(T a, T b) { return (a + b); }
  T initial_value() { return (T)0; }
  T after_reduce(T a, int elems) { return a / elems; }
};

enum class ReduceOp { Sum, Max, Mean };

} // namespace cpu
} // namespace pg