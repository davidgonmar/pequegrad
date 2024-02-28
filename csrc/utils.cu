#include "utils.cuh"

__device__ int get_max_idx(const size_t *shape, const size_t num_dims) {
  int accum = 1;
  for (int d = 0; d < num_dims; d++) {
    accum *= shape[d];
  }
  return accum;
}