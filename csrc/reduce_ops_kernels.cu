#include "utils.cuh"
#include <stdio.h>

__global__ void sum_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t red_axis) {

  int idx =
      blockDim.x * blockIdx.x + threadIdx.x; // one element of the output array

  // the general explanation is:
  // each idx represents one output value, so each thread will reduce to said
  // output value therefore, we'll loop accross the reduced dimension, and
  // accumulate those values in order to calculate where we take the input from,
  // we can just do 2 things:
  // 1. calculate actual index as normally (impl in utils.cuh)
  // 2. if the dim we are iterating over is the one we are reducing over,
  // we must use the 'i' value (current iteration over the dimension we are
  // reducing)

  // example: in_shape = (4, 3, 2), contiguous (strides = (6, 2, 1)), wanna
  // reduce axis=1, so output shape will be (4, 1, 2), with strides not being
  // relevant

  // therefore, idx will be valid if it is < 4 * 1 * 2 (remember, idx represents
  // 1 output value) so, we will calculate the actual index to take the input
  // from based on those strides, but since we are 'moving' accross the
  // shape[dim=1] in the loop, we will not use the idx accross that dimension,
  // but the iterator value (i).

  int total_out_elements = 1;
  for (int i = 0; i < n_dims; i++) {
    total_out_elements *= in_shape[i];
  }
  total_out_elements /= in_shape[red_axis];

  if (idx >= total_out_elements) {
    return;
  }

  int red_elements = in_shape[red_axis];

  float accum = 0;

  for (int i = 0; i < red_elements; i++) {
    int reduced_idx = idx;
    int in_idx = 0;
    for (int j = n_dims - 1; j >= 0; j--) {
      if (j == red_axis) {
        in_idx +=
            i * in_strides[j] / sizeof(float); // simply advance by 'i * stride'
      } else { // do the general algorithm to go from idx -> actual displacement
        int current_dim_idx = reduced_idx % in_shape[j];
        in_idx += current_dim_idx * in_strides[j] / sizeof(float);
        reduced_idx /= in_shape[j];
      }
    }
    float el = in[in_idx];
    accum += el;
  }

  out[idx] = accum;
}

// same as with sum
__global__ void max_kernel(const float *in, float *out,
                           const size_t *in_strides, const size_t *in_shape,
                           const size_t n_dims, const size_t red_axis) {

  int idx =
      blockDim.x * blockIdx.x + threadIdx.x; // one element of the output array
  int total_out_elements = 1;
  for (int i = 0; i < n_dims; i++) {
    total_out_elements *= in_shape[i];
  }
  total_out_elements /= in_shape[red_axis];

  if (idx >= total_out_elements) {
    return;
  }

  int red_elements = in_shape[red_axis];

  float accum = -INFINITY;

  for (int i = 0; i < red_elements; i++) {
    int reduced_idx = idx;
    int in_idx = 0;
    for (int j = n_dims - 1; j >= 0; j--) {
      if (j == red_axis) {
        in_idx +=
            i * in_strides[j] / sizeof(float); // simply advance by 'i * stride'
      } else { // do the general algorithm to go from idx -> actual displacement
        int current_dim_idx = reduced_idx % in_shape[j];
        in_idx += current_dim_idx * in_strides[j] / sizeof(float);
        reduced_idx /= in_shape[j];
      }
    }
    float el = in[in_idx];
    accum = max(accum, el);
  }

  out[idx] = accum;
}