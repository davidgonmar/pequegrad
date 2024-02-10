#include "utils.cuh"

__device__ int get_idx_from_strides(const size_t *shape, const size_t *strides,
                                    const size_t num_dims, const int abs_idx) {
  int tmp_idx = abs_idx;
  int idx = 0;
  for (int d = num_dims - 1; d >= 0; d--) {
    int curr_dim = tmp_idx % shape[d]; // 'how much of dimension d'
    idx += strides[d] * curr_dim;
    tmp_idx /= shape[d];
  }
  return idx / sizeof(float); // strides are in bytes
}

__device__ int get_max_idx(const size_t *shape, const size_t num_dims) {
  int accum = 1;
  for (int d = 0; d < num_dims; d++) {
    accum *= shape[d];
  }
  return accum;
}